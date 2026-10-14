// Qwen3.6-35B-A3B resident Strata engine.
//
// The Strata Python service owns tokenisation, HTTP and the dashboard.  This
// executable is deliberately narrow: it validates one exact GGUF family,
// keeps the model resident, and translates the established stdin/stdout
// protocol to llama.cpp's target + MTP speculative contexts.

#include "strata/core/qwen35moe_layout.hpp"
#include "strata/core/conversation_memory.hpp"
#include "qwen35moe_limits.hpp"
#include "qwen35moe_prompt_cache.hpp"

#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr const char* kEngineName = "qwen35moe-native-adapter";

struct Options {
    std::string model;
    int32_t max_context = 262144;
    int32_t max_output_tokens = 32768;
    int32_t gpu_layers = -1;
    int32_t threads = 16;
    int32_t threads_batch = 16;
    int32_t batch_size = 8192;
    int32_t ubatch_size = 512;
    bool flash_attn = true;
    std::string kv_cache = "q8_0";
    bool mtp = true;
    int32_t spec_n_max = 3;
    int32_t spec_n_min = 1;
    bool fit = true;
    int32_t fit_target_mib = 1024;
    // Host-RAM snapshots of completed prompt prefixes.  The target and MTP
    // draft K/V are restored together, so this remains a prefill cache rather
    // than a lossy text-response cache.
    int32_t conversation_cache_mib = 0;
    int32_t conversation_cache_slots = 4;
    int32_t conversation_cache_min_free_mib = 16384;
    int32_t conversation_cache_min_tokens = 1024;
};

struct Sampling {
    float temperature = 0.0f;
    float top_p = 1.0f;
    int32_t top_k = 0;
    float min_p = 0.0f;
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    int32_t penalty_last_n = 64;
    uint32_t seed = LLAMA_DEFAULT_SEED;
};

struct Request {
    int32_t max_new = 0;
    Sampling sampling;
    std::vector<llama_token> tokens;
};

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string protocol_text(std::string value) {
    for (char& c : value) {
        if (c == '\r' || c == '\n' || c == '\t') c = ' ';
    }
    return trim(std::move(value));
}

std::vector<std::string> split_ws(const std::string& value) {
    std::istringstream in(value);
    std::vector<std::string> parts;
    std::string part;
    while (in >> part) parts.push_back(std::move(part));
    return parts;
}

bool parse_i32(std::string_view text, int32_t& out) {
    if (text.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(std::string(text).c_str(), &end, 10);
    if (errno || end == nullptr || *end != '\0' || value < std::numeric_limits<int32_t>::min() ||
        value > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    out = static_cast<int32_t>(value);
    return true;
}

bool parse_u32(std::string_view text, uint32_t& out) {
    if (text.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(std::string(text).c_str(), &end, 10);
    if (errno || end == nullptr || *end != '\0' || value > std::numeric_limits<uint32_t>::max()) return false;
    out = static_cast<uint32_t>(value);
    return true;
}

bool parse_float(std::string_view text, float& out) {
    if (text.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const float value = std::strtof(std::string(text).c_str(), &end);
    if (errno || end == nullptr || *end != '\0' || !std::isfinite(value)) return false;
    out = value;
    return true;
}

bool parse_switch(std::string_view text, bool& out) {
    if (text == "on" || text == "true" || text == "1") {
        out = true;
        return true;
    }
    if (text == "off" || text == "false" || text == "0") {
        out = false;
        return true;
    }
    return false;
}

void usage(std::ostream& out) {
    out << "usage: strata-qwen35moe-engine --serve --model MODEL.gguf [options]\n"
        << "  --max-context N             Context window (default 262144)\n"
        << "  --max-output-tokens N       Per-request output hard cap (default 32768)\n"
        << "  --gpu-layers N              llama.cpp GPU layer policy (default -1)\n"
        << "  --threads N --threads-batch N\n"
        << "  --batch-size N --ubatch-size N\n"
        << "  --flash-attn on|off         Enable CUDA FlashAttention (default on)\n"
        << "  --kv-cache q8_0|f16|bf16    K/V cache type (default q8_0)\n"
        << "  --mtp on|off                Use verified MTP speculative decoding (default on)\n"
        << "  --spec-n-max N --spec-n-min N\n"
        << "  --fit on|off --fit-target-mib N\n"
        << "  --conversation-cache-mib N          Host-RAM budget for reusable prompt prefixes (0 = off)\n"
        << "  --conversation-cache-slots N        Maximum cached prefixes; required >0 when cache is on\n"
        << "  --conversation-cache-min-free-mib N Physical RAM floor before capturing a prefix\n"
        << "  --conversation-cache-min-tokens N   Required >0 when cache is on; skip shorter prefixes\n";
}

std::optional<Options> parse_options(int argc, char** argv, std::string& error, bool& show_help) {
    Options options;
    bool serve = false;
    show_help = false;

    auto need_value = [&](int& i, const char* flag) -> const char* {
        if (i + 1 >= argc) {
            error = std::string("missing value for ") + flag;
            return nullptr;
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            show_help = true;
            return options;
        }
        if (arg == "--serve") {
            serve = true;
            continue;
        }
        const char* value = nullptr;
        if (arg == "--model") {
            if (!(value = need_value(i, "--model"))) return std::nullopt;
            options.model = value;
        } else if (arg == "--max-context") {
            if (!(value = need_value(i, "--max-context")) || !parse_i32(value, options.max_context)) goto bad_number;
        } else if (arg == "--max-output-tokens") {
            if (!(value = need_value(i, "--max-output-tokens")) || !parse_i32(value, options.max_output_tokens)) goto bad_number;
        } else if (arg == "--gpu-layers") {
            if (!(value = need_value(i, "--gpu-layers")) || !parse_i32(value, options.gpu_layers)) goto bad_number;
        } else if (arg == "--threads") {
            if (!(value = need_value(i, "--threads")) || !parse_i32(value, options.threads)) goto bad_number;
        } else if (arg == "--threads-batch") {
            if (!(value = need_value(i, "--threads-batch")) || !parse_i32(value, options.threads_batch)) goto bad_number;
        } else if (arg == "--batch-size") {
            if (!(value = need_value(i, "--batch-size")) || !parse_i32(value, options.batch_size)) goto bad_number;
        } else if (arg == "--ubatch-size") {
            if (!(value = need_value(i, "--ubatch-size")) || !parse_i32(value, options.ubatch_size)) goto bad_number;
        } else if (arg == "--flash-attn") {
            if (!(value = need_value(i, "--flash-attn")) || !parse_switch(value, options.flash_attn)) goto bad_switch;
        } else if (arg == "--kv-cache") {
            if (!(value = need_value(i, "--kv-cache"))) return std::nullopt;
            options.kv_cache = value;
        } else if (arg == "--mtp") {
            if (!(value = need_value(i, "--mtp")) || !parse_switch(value, options.mtp)) goto bad_switch;
        } else if (arg == "--spec-n-max") {
            if (!(value = need_value(i, "--spec-n-max")) || !parse_i32(value, options.spec_n_max)) goto bad_number;
        } else if (arg == "--spec-n-min") {
            if (!(value = need_value(i, "--spec-n-min")) || !parse_i32(value, options.spec_n_min)) goto bad_number;
        } else if (arg == "--fit") {
            if (!(value = need_value(i, "--fit")) || !parse_switch(value, options.fit)) goto bad_switch;
        } else if (arg == "--fit-target-mib") {
            if (!(value = need_value(i, "--fit-target-mib")) || !parse_i32(value, options.fit_target_mib)) goto bad_number;
        } else if (arg == "--conversation-cache-mib") {
            if (!(value = need_value(i, "--conversation-cache-mib")) || !parse_i32(value, options.conversation_cache_mib)) goto bad_number;
        } else if (arg == "--conversation-cache-slots") {
            if (!(value = need_value(i, "--conversation-cache-slots")) || !parse_i32(value, options.conversation_cache_slots)) goto bad_number;
        } else if (arg == "--conversation-cache-min-free-mib") {
            if (!(value = need_value(i, "--conversation-cache-min-free-mib")) || !parse_i32(value, options.conversation_cache_min_free_mib)) goto bad_number;
        } else if (arg == "--conversation-cache-min-tokens") {
            if (!(value = need_value(i, "--conversation-cache-min-tokens")) || !parse_i32(value, options.conversation_cache_min_tokens)) goto bad_number;
        } else {
            error = "unknown option " + std::string(arg);
            return std::nullopt;
        }
        continue;

    bad_number:
        error = "invalid integer value for " + std::string(arg);
        return std::nullopt;
    bad_switch:
        error = "expected on or off for " + std::string(arg);
        return std::nullopt;
    }

    if (!serve) {
        error = "--serve is required for the Strata engine protocol";
        return std::nullopt;
    }
    if (options.model.empty()) {
        error = "--model is required";
        return std::nullopt;
    }
    if (options.max_context < 2 || options.max_output_tokens < 1 || options.threads < 1 || options.threads_batch < 1 ||
        options.batch_size < 1 || options.ubatch_size < 1 || options.fit_target_mib < 0 || options.spec_n_max < 1 ||
        options.spec_n_min < 0 || options.spec_n_min > options.spec_n_max || options.conversation_cache_mib < 0 ||
        options.conversation_cache_slots < 0 || options.conversation_cache_min_free_mib < 0 ||
        options.conversation_cache_min_tokens < 0) {
        error = "invalid non-positive or inconsistent runtime limit";
        return std::nullopt;
    }
    if (options.conversation_cache_mib > 0 &&
        (options.conversation_cache_slots == 0 || options.conversation_cache_min_tokens == 0)) {
        error = "a positive --conversation-cache-mib requires positive --conversation-cache-slots and --conversation-cache-min-tokens";
        return std::nullopt;
    }
    if (options.kv_cache != "q8_0" && options.kv_cache != "f16" && options.kv_cache != "bf16") {
        error = "--kv-cache must be q8_0, f16, or bf16";
        return std::nullopt;
    }
    return options;
}

ggml_type kv_type_from_name(const std::string& name) {
    if (name == "q8_0") return GGML_TYPE_Q8_0;
    if (name == "f16") return GGML_TYPE_F16;
    if (name == "bf16") return GGML_TYPE_BF16;
    throw std::runtime_error("unsupported KV cache type " + name);
}

void emit_info(const std::string& value) {
    std::cout << "INFO " << protocol_text(value) << std::endl;
}

void emit_error(const std::string& value) {
    std::cout << "ERR " << protocol_text(value) << std::endl;
}

void emit_done(int generated, int prompt_tokens, double prompt_ms, double decode_ms, const char* finish,
               int drafts_accepted, int drafts_offered, int reused) {
    std::cout << "DONE " << generated << ' ' << prompt_tokens << ' ' << std::fixed << std::setprecision(3) << prompt_ms
              << ' ' << decode_ms << ' ' << finish << ' ' << drafts_accepted << ' ' << drafts_offered << ' ' << reused << std::endl;
}

class CommandChannel {
public:
    CommandChannel() = default;
    CommandChannel(const CommandChannel&) = delete;
    CommandChannel& operator=(const CommandChannel&) = delete;

    void start() {
        reader_ = std::thread([this] {
            std::string line;
            while (std::getline(std::cin, line)) {
                line = trim(std::move(line));
                if (line.empty()) continue;
                std::lock_guard lock(mu_);
                if (line == "STOP") {
                    // A STOP after a queued GEN but before the worker acquires it
                    // belongs to that request.  A STOP between requests is ignored.
                    if (active_ || !commands_.empty()) stop_requested_ = true;
                } else if (line == "QUIT") {
                    quitting_ = true;
                    stop_requested_ = true;
                    cv_.notify_all();
                    return;
                } else {
                    commands_.push_back(std::move(line));
                }
                cv_.notify_all();
            }
            {
                std::lock_guard lock(mu_);
                input_closed_ = true;
            }
            cv_.notify_all();
        });
    }

    bool next(std::string& command) {
        std::unique_lock lock(mu_);
        cv_.wait(lock, [this] { return quitting_ || !commands_.empty() || input_closed_; });
        if (quitting_ || commands_.empty()) return false;
        command = std::move(commands_.front());
        commands_.pop_front();
        active_ = true;
        return true;
    }

    bool cancelled() const {
        return stop_requested_.load(std::memory_order_relaxed) || quitting_.load(std::memory_order_relaxed);
    }

    void complete() {
        std::lock_guard lock(mu_);
        active_ = false;
        stop_requested_ = false;
    }

    bool quitting() const { return quitting_.load(std::memory_order_relaxed); }

    void join() {
        if (reader_.joinable()) reader_.join();
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::string> commands_;
    std::thread reader_;
    std::atomic<bool> active_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> quitting_{false};
    bool input_closed_ = false;
};

bool parse_token_ids(const std::string& encoded, std::vector<llama_token>& tokens, std::string& error) {
    if (encoded.empty()) {
        error = "GEN has no token ids";
        return false;
    }
    std::size_t from = 0;
    while (from <= encoded.size()) {
        const std::size_t to = encoded.find(',', from);
        const std::string_view part(encoded.data() + from, (to == std::string::npos ? encoded.size() : to) - from);
        int32_t id = 0;
        if (!parse_i32(part, id) || id < 0) {
            error = "invalid token id in GEN";
            return false;
        }
        tokens.push_back(static_cast<llama_token>(id));
        if (to == std::string::npos) break;
        from = to + 1;
        if (from == encoded.size()) {
            error = "empty token id in GEN";
            return false;
        }
    }
    return !tokens.empty();
}

bool set_sampling_key(Sampling& sampling, const std::string& key, const std::string& value, std::string& error) {
    if (key == "temperature") {
        if (!parse_float(value, sampling.temperature) || sampling.temperature < 0.0f) goto bad;
    } else if (key == "top_p") {
        if (!parse_float(value, sampling.top_p) || sampling.top_p <= 0.0f || sampling.top_p > 1.0f) goto bad;
    } else if (key == "top_k") {
        if (!parse_i32(value, sampling.top_k) || sampling.top_k < 0) goto bad;
    } else if (key == "min_p") {
        if (!parse_float(value, sampling.min_p) || sampling.min_p < 0.0f || sampling.min_p > 1.0f) goto bad;
    } else if (key == "penalty_repeat") {
        if (!parse_float(value, sampling.penalty_repeat) || sampling.penalty_repeat <= 0.0f) goto bad;
    } else if (key == "penalty_freq") {
        if (!parse_float(value, sampling.penalty_freq)) goto bad;
    } else if (key == "penalty_present") {
        if (!parse_float(value, sampling.penalty_present)) goto bad;
    } else if (key == "penalty_last_n") {
        if (!parse_i32(value, sampling.penalty_last_n) || sampling.penalty_last_n < 0) goto bad;
    } else if (key == "seed") {
        if (!parse_u32(value, sampling.seed)) goto bad;
    } else {
        // Strata also forwards engine-specific calibration keys.  This adapter
        // deliberately ignores unknown keys rather than changing target
        // sampling semantics or rejecting a compatible service request.
    }
    return true;
bad:
    error = "invalid sampling value for " + key;
    return false;
}

bool parse_gen(const std::string& line, Request& request, std::string& error) {
    const auto fields = split_ws(line);
    if (fields.size() < 3 || fields[0] != "GEN") {
        error = "expected GEN <max_new> [key=value ...] <id,id,...>";
        return false;
    }
    if (!parse_i32(fields[1], request.max_new) || request.max_new < 0) {
        error = "GEN has invalid max_new";
        return false;
    }
    for (std::size_t i = 2; i + 1 < fields.size(); ++i) {
        const std::size_t eq = fields[i].find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 == fields[i].size()) {
            error = "GEN has malformed sampling key";
            return false;
        }
        if (!set_sampling_key(request.sampling, fields[i].substr(0, eq), fields[i].substr(eq + 1), error)) return false;
    }
    return parse_token_ids(fields.back(), request.tokens, error);
}

common_params_sampling to_common_sampling(const Sampling& sampling) {
    common_params_sampling result;
    result.seed = sampling.seed;
    result.temp = sampling.temperature;
    result.top_p = sampling.top_p;
    result.top_k = sampling.top_k;
    result.min_p = sampling.min_p;
    result.penalty_repeat = sampling.penalty_repeat;
    result.penalty_freq = sampling.penalty_freq;
    result.penalty_present = sampling.penalty_present;
    result.penalty_last_n = sampling.penalty_last_n;
    result.backend_sampling = false;
    result.no_perf = true;
    return result;
}

struct PromptCacheEntry {
    std::vector<llama_token> tokens;
    std::vector<uint8_t> target_state;
    std::vector<uint8_t> draft_state;
    std::vector<uint8_t> speculative_state;
    uint64_t last_used = 0;

    template <typename T>
    static bool storage_bytes(const std::vector<T>& values, size_t& result) {
        if (values.capacity() > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
        result = values.capacity() * sizeof(T);
        return true;
    }

    // Account for the allocated host backing, including the token identity
    // metadata that makes a restore safe.  Capacity, rather than size, keeps
    // the reported and admitted total honest on implementations that grow a
    // vector beyond its requested element count.
    bool bytes(size_t& result) const {
        size_t total = 0;
        size_t part = 0;
        // The vectors have distinct value types, so keep the checked additions
        // explicit rather than hiding a lossy cast behind a generic range.
        if (!storage_bytes(tokens, part) || part > std::numeric_limits<size_t>::max() - total) return false;
        total += part;
        if (!storage_bytes(target_state, part) || part > std::numeric_limits<size_t>::max() - total) return false;
        total += part;
        if (!storage_bytes(draft_state, part) || part > std::numeric_limits<size_t>::max() - total) return false;
        total += part;
        if (!storage_bytes(speculative_state, part) || part > std::numeric_limits<size_t>::max() - total) return false;
        result = total + part;
        return true;
    }
};

static_assert(std::is_nothrow_move_constructible_v<PromptCacheEntry>,
              "Prompt cache commit must not throw after eviction");

class Engine {
public:
    explicit Engine(Options options) : options_(std::move(options)) {
        validate_layout();
        initialize_target();
    }

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    uint32_t max_context() const { return llama_n_ctx(ctx_); }

    void announce() const {
        emit_info(std::string("engine=") + kEngineName + " runtime=" + protocol_text(llama_version()));
        emit_info(std::string("architecture=qwen35moe mtp=") + (options_.mtp ? "on" : "off") +
                  " spec_n_max=" + std::to_string(options_.spec_n_max) +
                  " spec_n_min=" + std::to_string(options_.spec_n_min));
        emit_info("context_requested=" + std::to_string(options_.max_context) +
                  " context_actual=" + std::to_string(max_context()) +
                  " max_output=" + std::to_string(options_.max_output_tokens) +
                  " kv=" + options_.kv_cache +
                  " flash_attn=" + (options_.flash_attn ? "on" : "off") +
                  " fit=" + (options_.fit ? "on" : "off"));
        emit_info("conversation_cache_mib=" + std::to_string(options_.conversation_cache_mib) +
                  " conversation_cache_slots=" + std::to_string(options_.conversation_cache_slots) +
                  " conversation_cache_min_free_mib=" + std::to_string(options_.conversation_cache_min_free_mib) +
                  " conversation_cache_min_tokens=" + std::to_string(options_.conversation_cache_min_tokens));
    }

    void generate(const Request& request, CommandChannel& channel) {
        if (request.tokens.empty()) {
            emit_error("GEN has no token ids");
            return;
        }
        const uint32_t context = max_context();
        if (request.tokens.size() > context) {
            emit_error("prompt tokens exceed configured context");
            return;
        }
        const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_));
        for (llama_token token : request.tokens) {
            if (token < 0 || token >= vocab) {
                emit_error("GEN contains a token outside this model vocabulary");
                return;
            }
        }

        // The engine itself is the final authority on the live context.  A
        // client can request more than either the profile's output ceiling or
        // the room remaining after its prompt; fit it here so a large-codebase
        // request completes with the usable prefix instead of receiving a 400.
        Request fitted = request;
        fitted.max_new = strata::qwen35moe::fit_max_new(
            request.tokens.size(), request.max_new, context, options_.max_output_tokens);

        try {
            if (options_.mtp) {
                generate_mtp(fitted, channel);
            } else {
                generate_direct(fitted, channel);
            }
        } catch (const std::exception& e) {
            emit_error(std::string("qwen35moe request failed: ") + e.what());
        }
    }

private:
    bool prompt_cache_enabled() const {
        return options_.conversation_cache_mib > 0 && options_.conversation_cache_slots > 0 &&
               options_.conversation_cache_min_tokens > 0;
    }

    uint64_t prompt_cache_budget_bytes() const {
        return static_cast<uint64_t>(options_.conversation_cache_mib) * 1024u * 1024u;
    }

    uint64_t prompt_cache_floor_bytes() const {
        return static_cast<uint64_t>(options_.conversation_cache_min_free_mib) * 1024u * 1024u;
    }

    static bool add_size_checked(size_t left, size_t right, size_t& result) {
        if (right > std::numeric_limits<size_t>::max() - left) return false;
        result = left + right;
        return true;
    }

    std::optional<size_t> best_prompt_cache(const std::vector<llama_token>& tokens, size_t max_reused) const {
        std::optional<size_t> best;
        size_t best_tokens = 0;
        uint64_t best_used = 0;
        for (size_t i = 0; i < prompt_cache_.size(); ++i) {
            const auto& entry = prompt_cache_[i];
            const size_t reusable = strata::qwen35moe::prompt_cache_reusable_prefix(
                entry.tokens, tokens, max_reused);
            if (reusable < static_cast<size_t>(options_.conversation_cache_min_tokens)) continue;
            if (!best || reusable > best_tokens || (reusable == best_tokens && entry.last_used > best_used)) {
                best = i;
                best_tokens = reusable;
                best_used = entry.last_used;
            }
        }
        return best;
    }

    void erase_prompt_cache(size_t index, bool evicted) {
        if (index >= prompt_cache_.size()) return;
        size_t entry_bytes = 0;
        if (!prompt_cache_[index].bytes(entry_bytes)) {
            // An overflowed accounting value must never make a later admission
            // look safe.  Clear the accounting total while discarding the
            // invalid entry and allow future captures to rebuild it exactly.
            prompt_cache_bytes_ = 0;
        } else {
            prompt_cache_bytes_ = entry_bytes > prompt_cache_bytes_ ? 0 : prompt_cache_bytes_ - entry_bytes;
        }
        prompt_cache_.erase(prompt_cache_.begin() + static_cast<std::ptrdiff_t>(index));
        if (evicted) ++prompt_cache_evictions_;
    }

    size_t oldest_prompt_cache() const {
        size_t oldest = 0;
        for (size_t i = 1; i < prompt_cache_.size(); ++i) {
            if (prompt_cache_[i].last_used < prompt_cache_[oldest].last_used) oldest = i;
        }
        return oldest;
    }

    void cache_log(const char* action, size_t tokens, double elapsed, const char* detail = nullptr) const {
        std::cerr << "strata serve: conversation cache: " << action << ' ' << tokens << " tokens";
        if (detail && *detail) std::cerr << " (" << detail << ')';
        std::cerr << " in " << std::fixed << std::setprecision(1) << elapsed << " ms; parked=" << prompt_cache_.size()
                  << " bytes=" << prompt_cache_bytes_ << " evictions=" << prompt_cache_evictions_ << std::endl;
    }

    // Clear first even on a cache hit.  A failed restore must never leave a
    // partially restored target or draft context eligible for inference.
    int32_t restore_prompt_cache(const Request& request, size_t max_reused, common_speculative* spec,
                                 std::chrono::steady_clock::time_point started) {
        reset_request_memory();
        if (!prompt_cache_enabled()) return 0;
        const auto candidate = best_prompt_cache(request.tokens, max_reused);
        if (!candidate) return 0;

        const size_t index = *candidate;
        const size_t cached_tokens = prompt_cache_[index].tokens.size();
        try {
            auto& entry = prompt_cache_[index];
            const auto verify_sequence_state = [](llama_context* context, const std::vector<uint8_t>& expected,
                                                  const char* label) {
                if (!context || expected.empty()) {
                    throw std::runtime_error(std::string(label) + " restored state is unavailable");
                }
                const size_t restored_size = llama_state_seq_get_size_ext(context, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
                if (restored_size != expected.size()) {
                    throw std::runtime_error(std::string(label) + " restored state size mismatch");
                }
                std::vector<uint8_t> restored(expected.size());
                if (llama_state_seq_get_data_ext(context, restored.data(), restored.size(), 0,
                                                 LLAMA_STATE_SEQ_FLAGS_NONE) != restored.size() ||
                    restored != expected) {
                    throw std::runtime_error(std::string(label) + " restored state byte round-trip mismatch");
                }
            };
            if (entry.target_state.empty() || (spec && (entry.draft_state.empty() || entry.speculative_state.empty()))) {
                throw std::runtime_error("incomplete target, draft, or MTP snapshot");
            }
            if (llama_state_seq_set_data_ext(ctx_, entry.target_state.data(), entry.target_state.size(), 0,
                                              LLAMA_STATE_SEQ_FLAGS_NONE) != entry.target_state.size()) {
                throw std::runtime_error("target K/V restore returned a short write");
            }
            if (spec) {
                if (!ctx_dft_ || llama_state_seq_set_data_ext(ctx_dft_, entry.draft_state.data(), entry.draft_state.size(), 0,
                                                                LLAMA_STATE_SEQ_FLAGS_NONE) != entry.draft_state.size()) {
                    throw std::runtime_error("MTP draft K/V restore returned a short write");
                }
                verify_sequence_state(ctx_dft_, entry.draft_state, "MTP draft K/V");
                common_speculative_set_state(spec, 0, entry.speculative_state);
                // common_speculative_set_state has a void public API.  Ask
                // the implementation to serialize again and compare every
                // byte so an unsupported or malformed MTP payload cannot
                // masquerade as a valid K/V cache hit.
                std::vector<uint8_t> restored_speculative_state;
                if (!common_speculative_get_state(spec, 0, restored_speculative_state) ||
                    restored_speculative_state != entry.speculative_state) {
                    throw std::runtime_error("MTP carry-over state round-trip validation failed");
                }
            }
            verify_sequence_state(ctx_, entry.target_state, "target K/V");
            const llama_pos expected_last = static_cast<llama_pos>(cached_tokens - 1);
            const auto target_memory = llama_get_memory(ctx_);
            const auto draft_memory = spec ? llama_get_memory(ctx_dft_) : nullptr;
            const auto bounds_match = [expected_last](llama_memory_t memory) {
                if (!memory) return false;
                const llama_pos first = llama_memory_seq_pos_min(memory, 0);
                const llama_pos last = llama_memory_seq_pos_max(memory, 0);
                // Qwen35MoE's hybrid Gated-DeltaNet memory deliberately
                // reports the non-rollbackable recurrent tail as
                // pos_min == pos_max. Its full sequence-state payload carries
                // both attention and recurrent state, so pos_min need only
                // name a valid retained position. Requiring zero would reject
                // every valid long hybrid snapshot. The exact cache entry
                // tokens, K/V byte round-trip above, MTP-state round-trip,
                // and exact terminal position jointly guard reuse.
                return first >= 0 && first <= expected_last && last == expected_last;
            };
            const bool target_exact = bounds_match(target_memory);
            const bool draft_exact = !spec || bounds_match(draft_memory);
            if (!target_exact || !draft_exact) {
                const llama_pos target_first = target_memory ? llama_memory_seq_pos_min(target_memory, 0) : -1;
                const llama_pos target_last = target_memory ? llama_memory_seq_pos_max(target_memory, 0) : -1;
                const llama_pos draft_first = draft_memory ? llama_memory_seq_pos_min(draft_memory, 0) : -1;
                const llama_pos draft_last = draft_memory ? llama_memory_seq_pos_max(draft_memory, 0) : -1;
                std::ostringstream error;
                error << "restored cache bounds mismatch: target=" << target_first << ".." << target_last
                      << " draft=" << draft_first << ".." << draft_last << " expected-last=" << expected_last;
                throw std::runtime_error(error.str());
            }
            entry.last_used = ++prompt_cache_clock_;
            std::cout << "RESUME " << cached_tokens << std::endl;
            cache_log("restored", cached_tokens, elapsed_ms(started), options_.mtp ? "MTP" : "direct");
            return static_cast<int32_t>(cached_tokens);
        } catch (const std::exception& e) {
            erase_prompt_cache(index, false);
            cache_log("skipped", cached_tokens, elapsed_ms(started), e.what());
            reset_request_memory();
            return 0;
        }
    }

    void capture_prompt_cache(const std::vector<llama_token>& tokens, common_speculative* spec) {
        if (!prompt_cache_enabled() || tokens.size() < static_cast<size_t>(options_.conversation_cache_min_tokens)) return;
        if (options_.mtp && !spec) return;
        for (auto& entry : prompt_cache_) {
            if (entry.tokens == tokens) {
                entry.last_used = ++prompt_cache_clock_;
                return;
            }
        }

        const auto started = std::chrono::steady_clock::now();
        try {
            // get_size_ext does not itself synchronize asynchronous backends.
            // The state size and bytes must describe the same quiescent K/V
            // boundary, after the caller has checked MTP prompt retention.
            llama_synchronize(ctx_);
            if (spec && ctx_dft_) llama_synchronize(ctx_dft_);
            const size_t target_bytes = llama_state_seq_get_size_ext(ctx_, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
            const size_t draft_bytes = spec && ctx_dft_
                ? llama_state_seq_get_size_ext(ctx_dft_, 0, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
            if (spec && (!ctx_dft_ || draft_bytes == 0)) {
                cache_log("skipped", tokens.size(), elapsed_ms(started), "MTP draft snapshot unavailable");
                return;
            }
            std::vector<uint8_t> speculative_state;
            if (spec && !common_speculative_get_state(spec, 0, speculative_state)) {
                cache_log("skipped", tokens.size(), elapsed_ms(started), "MTP carry-over snapshot unavailable");
                return;
            }

            size_t planned_bytes = 0;
            if (tokens.size() > std::numeric_limits<size_t>::max() / sizeof(llama_token)) {
                cache_log("skipped", tokens.size(), elapsed_ms(started), "token metadata size overflow");
                return;
            }
            const size_t token_bytes = tokens.size() * sizeof(llama_token);
            if (!add_size_checked(target_bytes, draft_bytes, planned_bytes) ||
                !add_size_checked(planned_bytes, speculative_state.size(), planned_bytes) ||
                !add_size_checked(planned_bytes, token_bytes, planned_bytes) || target_bytes == 0) {
                cache_log("skipped", tokens.size(), elapsed_ms(started), "invalid snapshot size");
                return;
            }
            const uint64_t budget = prompt_cache_budget_bytes();
            if (planned_bytes > budget) {
                cache_log("skipped", tokens.size(), elapsed_ms(started), "snapshot exceeds configured RAM budget");
                return;
            }
            if (!strata::core::conversation_memory_admit(strata::core::conversation_available_memory(), planned_bytes,
                                                         prompt_cache_floor_bytes())) {
                cache_log("skipped", tokens.size(), elapsed_ms(started), "physical RAM admission");
                return;
            }

            // Reserve metadata before any eviction.  PromptCacheEntry has a
            // no-throw move constructor, so the later commit cannot drop an
            // old useful entry merely because the vector grows.
            prompt_cache_.reserve(prompt_cache_.size() + 1);

            // Do not evict a useful prior entry until the new one is fully
            // captured and validated.  A cache failure remains invisible to
            // model inference and leaves the existing cache usable.
            PromptCacheEntry entry;
            entry.last_used = ++prompt_cache_clock_;
            entry.tokens = tokens;
            entry.speculative_state = std::move(speculative_state);
            entry.target_state.resize(target_bytes);
            entry.draft_state.resize(draft_bytes);
            if (llama_state_seq_get_data_ext(ctx_, entry.target_state.data(), entry.target_state.size(), 0,
                                             LLAMA_STATE_SEQ_FLAGS_NONE) != entry.target_state.size() ||
                (spec && llama_state_seq_get_data_ext(ctx_dft_, entry.draft_state.data(), entry.draft_state.size(), 0,
                                                       LLAMA_STATE_SEQ_FLAGS_NONE) != entry.draft_state.size())) {
                cache_log("skipped", tokens.size(), elapsed_ms(started), "K/V snapshot returned a short read");
                return;
            }
            size_t actual_bytes = 0;
            if (!entry.bytes(actual_bytes)) {
                cache_log("skipped", tokens.size(), elapsed_ms(started), "snapshot accounting overflow");
                return;
            }
            if (actual_bytes > budget) {
                cache_log("skipped", tokens.size(), elapsed_ms(started), "allocated snapshot exceeds configured RAM budget");
                return;
            }
            // The allocation now already exists.  Require the desired free
            // floor itself, rather than reserving the same snapshot twice.
            if (!strata::core::conversation_memory_admit(strata::core::conversation_available_memory(), 0,
                                                         prompt_cache_floor_bytes())) {
                cache_log("skipped", tokens.size(), elapsed_ms(started), "physical RAM floor after capture");
                return;
            }
            while (!prompt_cache_.empty() &&
                   (prompt_cache_.size() >= static_cast<size_t>(options_.conversation_cache_slots) ||
                    prompt_cache_bytes_ > budget || actual_bytes > budget - prompt_cache_bytes_)) {
                erase_prompt_cache(oldest_prompt_cache(), true);
            }
            if (prompt_cache_bytes_ > budget || actual_bytes > budget - prompt_cache_bytes_) {
                cache_log("skipped", tokens.size(), elapsed_ms(started), "cache budget unavailable after eviction");
                return;
            }
            prompt_cache_bytes_ += actual_bytes;
            prompt_cache_.push_back(std::move(entry));
            cache_log("parked", tokens.size(), elapsed_ms(started), options_.mtp ? "MTP" : "direct");
        } catch (const std::bad_alloc&) {
            cache_log("skipped", tokens.size(), elapsed_ms(started), "host allocation failed");
        } catch (const std::exception& e) {
            cache_log("skipped", tokens.size(), elapsed_ms(started), e.what());
        } catch (...) {
            cache_log("skipped", tokens.size(), elapsed_ms(started), "unknown cache capture failure");
        }
    }

    // A full request is usually followed by an edited task, not an identical
    // prompt.  Keep geometric completed prefill boundaries so a changed tail
    // can resume from the largest common source/context prefix.  This avoids
    // taking a host-RAM K/V snapshot after every --batch-size chunk on a
    // 262K-token request.
    void capture_prefill_checkpoint(const Request& request, int32_t processed, int32_t total,
                                    common_speculative* spec) {
        if (!prompt_cache_enabled() || (options_.mtp && !spec) ||
            processed <= 0 || total <= 0 || processed > total ||
            static_cast<size_t>(processed) > request.tokens.size() ||
            !strata::qwen35moe::prompt_cache_checkpoint_due(
                static_cast<size_t>(processed), static_cast<size_t>(total),
                static_cast<size_t>(options_.batch_size))) {
            return;
        }

        // A target-only snapshot cannot faithfully resume MTP.  This is an
        // inference invariant rather than a cache admission condition, so an
        // invalid MTP prefill remains a hard request failure.
        if (spec) assert_mtp_prefill_retained(processed, 0);

        // Copying token metadata is optional cache work.  A host allocation
        // failure must not turn an otherwise valid model response into ERR.
        try {
            capture_prompt_cache(
                std::vector<llama_token>(request.tokens.begin(),
                                         request.tokens.begin() + static_cast<std::ptrdiff_t>(processed)), spec);
        } catch (const std::bad_alloc&) {
            cache_log("skipped", static_cast<size_t>(processed), 0.0, "checkpoint token copy allocation failed");
        } catch (const std::exception& e) {
            cache_log("skipped", static_cast<size_t>(processed), 0.0, e.what());
        } catch (...) {
            cache_log("skipped", static_cast<size_t>(processed), 0.0, "unknown checkpoint token-copy failure");
        }
    }

    void validate_layout() const {
        if (!std::filesystem::is_regular_file(options_.model)) {
            throw std::runtime_error("model is not a regular file: " + options_.model);
        }
        strata::GgufFile gguf(options_.model);
        std::string error;
        if (!strata::core::qwen35moe_validate_layout(gguf, error)) {
            throw std::runtime_error("Qwen35MoE layout admission failed: " + error);
        }
    }

    void initialize_target() {
        params_.model.path = options_.model;
        params_.n_ctx = options_.max_context;
        params_.n_batch = options_.batch_size;
        params_.n_ubatch = options_.ubatch_size;
        params_.n_parallel = 1;
        params_.n_sequences = 1;
        params_.n_gpu_layers = options_.gpu_layers;
        params_.cpuparams.n_threads = options_.threads;
        params_.cpuparams_batch.n_threads = options_.threads_batch;
        params_.flash_attn_type = options_.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params_.cache_type_k = kv_type_from_name(options_.kv_cache);
        params_.cache_type_v = params_.cache_type_k;
        params_.fit_params = options_.fit;
        for (auto& target : params_.fit_params_target) {
            target = static_cast<size_t>(options_.fit_target_mib) * 1024u * 1024u;
        }
        params_.sampling.backend_sampling = false;
        params_.sampling.no_perf = true;
        params_.no_perf = true;

        // common_params_parse() normally performs this padding after it has
        // parsed the llama.cpp command line.  This resident adapter builds
        // common_params directly, so it must supply the writable override
        // array itself before common_init_from_params() enters --fit.  Without
        // it, the fitter rejects the request and the default auto-GPU policy
        // loads every layer into CUDA virtual memory instead of placing the
        // excess MoE weights on the CPU.  On the 15 GiB A5500 that means a
        // roughly 21 GiB model allocation and severe paging.
        const size_t max_tensor_overrides = llama_max_tensor_buft_overrides();
        while (params_.tensor_buft_overrides.size() < max_tensor_overrides) {
            params_.tensor_buft_overrides.push_back({nullptr, nullptr});
        }

        if (options_.mtp) {
            params_.load_mtp = true;
            params_.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
            params_.speculative.draft.n_max = options_.spec_n_max;
            params_.speculative.draft.n_min = options_.spec_n_min;
            params_.speculative.draft.n_gpu_layers = options_.gpu_layers;
            params_.speculative.draft.cache_type_k = params_.cache_type_k;
            params_.speculative.draft.cache_type_v = params_.cache_type_v;
            params_.speculative.draft.backend_sampling = true;
            const auto limits = common_speculative_get_output_limits(
                params_.n_batch, params_.n_parallel, common_speculative_n_max(&params_.speculative));
            params_.n_outputs_max = limits.total;
            params_.n_outputs_max_per_seq = limits.per_seq;
        } else {
            params_.speculative.types = {COMMON_SPECULATIVE_TYPE_NONE};
        }

        target_ = common_init_from_params(params_);
        model_ = target_ ? target_->model() : nullptr;
        ctx_ = target_ ? target_->context() : nullptr;
        if (!model_ || !ctx_) {
            throw std::runtime_error("llama.cpp failed to load the Qwen35MoE target model/context");
        }

        // This probe performs a tiny decode and deliberately clears the
        // context afterwards.  It is safe only while the resident target
        // context is empty, never after a request's prompt prefill.
        target_seq_rm_type_ = common_context_can_seq_rm(ctx_);

        if (options_.mtp) {
            common_params draft_params = common_base_params_to_speculative(params_);
            mtp_init_ = common_speculative_init_from_params(draft_params, model_, ctx_);
            ctx_dft_ = mtp_init_ ? mtp_init_->context() : nullptr;
            if (!ctx_dft_) {
                throw std::runtime_error("llama.cpp failed to create the Qwen35MoE MTP draft context");
            }
            params_.speculative.draft.ctx_tgt = ctx_;
            params_.speculative.draft.ctx_dft = ctx_dft_;

            // Same destructive capability probe for the MTP context, before
            // it receives any target hidden states.
            draft_seq_rm_type_ = common_context_can_seq_rm(ctx_dft_);
        }
    }

    void reset_request_memory() const {
        if (const auto memory = llama_get_memory(ctx_)) {
            llama_memory_clear(memory, true);
        }
        if (ctx_dft_) {
            if (const auto memory = llama_get_memory(ctx_dft_)) {
                llama_memory_clear(memory, true);
            }
        }
    }

    static double elapsed_ms(std::chrono::steady_clock::time_point since) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
    }

    static void emit_progress(int read, int total, int chunk_tokens,
                              std::chrono::steady_clock::time_point chunk_started) {
        const double chunk_ms = std::max(0.001, elapsed_ms(chunk_started));
        const double rate = static_cast<double>(std::max(1, chunk_tokens)) * 1000.0 / chunk_ms;
        std::cout << "PP " << read << ' ' << total << ' ' << std::fixed << std::setprecision(3) << chunk_ms << ' '
                  << rate << std::endl;
    }

    bool prefill(const Request& request, common_speculative* spec, CommandChannel& channel, int32_t start_at,
                 int32_t token_count, std::chrono::steady_clock::time_point start, double& prompt_ms) {
        common_batch batch(ctx_);
        int32_t read = start_at;
        while (read < token_count) {
            if (channel.cancelled()) {
                prompt_ms = elapsed_ms(start);
                return false;
            }
            int32_t take = std::min(options_.batch_size, token_count - read);
            if (prompt_cache_enabled() && (!options_.mtp || spec)) {
                const size_t before = static_cast<size_t>(read);
                const size_t normal_end = before + static_cast<size_t>(take);
                const size_t checkpoint = strata::qwen35moe::prompt_cache_next_checkpoint(
                    before, static_cast<size_t>(token_count), static_cast<size_t>(options_.batch_size));
                if (checkpoint > before && checkpoint < normal_end) {
                    take = static_cast<int32_t>(checkpoint - before);
                }
            }
            const auto chunk_started = std::chrono::steady_clock::now();
            batch.clear();
            for (int32_t i = 0; i < take; ++i) {
                batch.add(request.tokens[static_cast<std::size_t>(read + i)], read + i, 0, false);
            }
            const int rc = llama_process(ctx_, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (rc != 0) throw std::runtime_error("target prompt decode returned " + std::to_string(rc));
            if (spec && !common_speculative_process(spec, batch)) {
                throw std::runtime_error("MTP prompt synchronization failed");
            }
            read += take;
            emit_progress(read, static_cast<int>(request.tokens.size()), take, chunk_started);
            if (!channel.cancelled()) {
                capture_prefill_checkpoint(request, read, token_count, spec);
            }
        }
        prompt_ms = elapsed_ms(start);
        return true;
    }

    void assert_mtp_prefill_retained(int32_t prefetched_tokens, llama_seq_id seq_id) const {
        if (prefetched_tokens <= 0) return;
        const llama_pos expected_last = prefetched_tokens - 1;
        const auto target_last = llama_memory_seq_pos_max(llama_get_memory(ctx_), seq_id);
        const auto draft_last = llama_memory_seq_pos_max(llama_get_memory(ctx_dft_), seq_id);
        if (target_last < expected_last || draft_last < expected_last) {
            throw std::runtime_error(
                "MTP prompt prefill was not retained; refusing to generate from a cleared context");
        }
    }

    void generate_direct(const Request& request, CommandChannel& channel) {
        const auto start = std::chrono::steady_clock::now();
        double prompt_ms = 0.0;
        const int total = static_cast<int>(request.tokens.size());

        if (request.max_new == 0) {
            const int reused = restore_prompt_cache(request, total, nullptr, start);
            const bool ok = prefill(request, nullptr, channel, reused, total, start, prompt_ms);
            emit_done(0, total, prompt_ms, 0.0, ok ? "length" : "cancel", 0, 0, reused);
            return;
        }

        // Keep one token for the first target evaluation, which produces the
        // logits from which the first completion token is sampled.
        const int reused = restore_prompt_cache(request, total - 1, nullptr, start);
        if (!prefill(request, nullptr, channel, reused, total - 1, start, prompt_ms)) {
            emit_done(0, total, prompt_ms, 0.0, "cancel", 0, 0, reused);
            return;
        }

        common_params_sampling sampling = to_common_sampling(request.sampling);
        common_sampler_ptr sampler(common_sampler_init(model_, sampling));
        for (llama_token token : request.tokens) common_sampler_accept(sampler.get(), token, false);

        common_batch batch(ctx_);
        llama_token current = request.tokens.back();
        int32_t n_past = total - 1;
        int generated = 0;
        const auto decode_start = std::chrono::steady_clock::now();
        const llama_vocab* vocab = llama_model_get_vocab(model_);

        while (generated < request.max_new && !channel.cancelled()) {
            batch.clear();
            batch.add(current, n_past++, 0, true);
            const int rc = llama_process(ctx_, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (rc != 0) throw std::runtime_error("target decode returned " + std::to_string(rc));
            if (generated == 0) {
                prompt_ms = elapsed_ms(start);
                emit_progress(total, total, 1, decode_start);
            }
            current = common_sampler_sample(sampler.get(), ctx_, -1);
            common_sampler_accept(sampler.get(), current, true);
            if (llama_vocab_is_eog(vocab, current)) {
                // The resident Strata service owns the chat template and must
                // observe the terminal token to translate this turn into the
                // API's finish_reason=stop.  Suppressing it here leaves the
                // service with only its default "length", even though the
                // model ended normally.
                std::cout << "T " << current << std::endl;
                ++generated;
                emit_done(generated, total, prompt_ms, elapsed_ms(decode_start), "stop", 0, 0, reused);
                return;
            }
            if (channel.cancelled()) break;
            std::cout << "T " << current << std::endl;
            ++generated;
        }
        emit_done(generated, total, prompt_ms, elapsed_ms(decode_start),
                  channel.cancelled() ? "cancel" : "length", 0, 0, reused);
    }

    void generate_mtp(const Request& request, CommandChannel& channel) {
        const auto start = std::chrono::steady_clock::now();
        const int total = static_cast<int>(request.tokens.size());
        double prompt_ms = 0.0;

        if (request.max_new == 0) {
            // No-output requests have no MTP decoder to seed.  They are
            // deliberately kept as a cold prefill rather than parking a
            // target-only snapshot that would later look MTP-compatible.
            reset_request_memory();
            const bool ok = prefill(request, nullptr, channel, 0, total, start, prompt_ms);
            emit_done(0, total, prompt_ms, 0.0, ok ? "length" : "cancel", 0, 0, 0);
            return;
        }

        common_speculative_ptr spec(common_speculative_init(params_.speculative, 1));
        if (!spec) throw std::runtime_error("MTP speculative decoder failed to initialize");

        // The MTP process hook consumes target hidden states for every input
        // row.  Processing prompt chunks individually preserves the trained
        // cross-chunk carry-over instead of reducing a long prompt to its last
        // batch's state.
        const int reused = restore_prompt_cache(request, total - 1, spec.get(), start);
        if (!prefill(request, spec.get(), channel, reused, total - 1, start, prompt_ms)) {
            emit_done(0, total, prompt_ms, 0.0, "cancel", 0, 0, reused);
            return;
        }
        // Regression guard for a tempting but destructive mistake: the
        // capability probe above clears its context, so it must remain before
        // prefill.  Both target and MTP KV state must contain the prompt
        // before a snapshot is eligible to enter the reusable cache.
        assert_mtp_prefill_retained(total - 1, 0);

        common_params_sampling sampling = to_common_sampling(request.sampling);
        common_sampler_ptr sampler(common_sampler_init(model_, sampling));
        for (llama_token token : request.tokens) common_sampler_accept(sampler.get(), token, false);

        const llama_seq_id seq_id = 0;
        llama_tokens prompt(request.tokens.begin(), request.tokens.end() - 1);
        prompt.reserve(max_context());
        llama_token last = request.tokens.back();
        int32_t n_past = total - 1;
        common_speculative_begin(spec.get(), seq_id, prompt);

        const bool checkpoint_target = target_seq_rm_type_ == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        const bool checkpoint_draft = draft_seq_rm_type_ == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        common_prompt_checkpoint checkpoint;
        common_batch target_batch(ctx_);
        llama_tokens draft;
        int generated = 0;
        int drafts_offered = 0;
        int drafts_accepted = 0;
        bool prompt_complete = false;
        const auto decode_start = std::chrono::steady_clock::now();
        const llama_vocab* vocab = llama_model_get_vocab(model_);

        while (!channel.cancelled() && generated < request.max_new) {
            if (draft.empty()) {
                checkpoint.update_pos(prompt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_), seq_id));
                if (checkpoint_draft) {
                    checkpoint.update_dft(ctx_dft_, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                int32_t draft_room = static_cast<int32_t>(max_context()) - n_past - 2;
                draft_room = std::min(draft_room, request.max_new - generated - 1);
                draft_room = std::max(0, draft_room);

                // llama.cpp's MTP implementation intentionally uses its
                // profile n_max, rather than the per-call dp.n_max.  Only
                // call it when the entire configured draft plus the target
                // sample can fit.  At the output/context tail we verify one
                // target token at a time, which preserves correctness instead
                // of overflowing the context with a three-token MTP batch.
                if (strata::qwen35moe::mtp_full_draft_fits(draft_room, options_.spec_n_max)) {
                    auto& draft_params = common_speculative_get_draft_params(spec.get(), seq_id);
                    draft_params.drafting = true;
                    draft_params.n_max = draft_room;
                    draft_params.pos0 = n_past;
                    draft_params.id_last = last;
                    draft_params.prompt = &prompt;
                    draft_params.result = &draft;
                    draft_params.result_q = nullptr;
                    draft_params.temp = request.sampling.temperature;
                    draft_params.seed = request.sampling.seed;
                    common_speculative_draft(spec.get());
                }

                if (!draft.empty() && checkpoint_target) {
                    checkpoint.update_tgt(ctx_, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
                if (!draft.empty() && checkpoint_draft) {
                    if (!checkpoint.load_dft(ctx_dft_, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                        throw std::runtime_error("could not restore MTP draft checkpoint");
                    }
                }
                if (ctx_dft_) {
                    llama_memory_seq_rm(llama_get_memory(ctx_dft_), seq_id, checkpoint.pos_max + 1, -1);
                }
            }

            target_batch.clear();
            target_batch.add(last, n_past++, seq_id, true);
            for (std::size_t i = 0; i < draft.size(); ++i) {
                target_batch.add(draft[i], n_past + static_cast<int32_t>(i), seq_id, true);
            }
            const int rc = llama_process(ctx_, LLAMA_PROCESS_TYPE_DECODE, target_batch.get());
            if (rc != 0) throw std::runtime_error("target MTP verification decode returned " + std::to_string(rc));
            if (!common_speculative_process(spec.get(), target_batch)) {
                throw std::runtime_error("MTP verification synchronization failed");
            }
            if (!prompt_complete) {
                prompt_complete = true;
                prompt_ms = elapsed_ms(start);
                emit_progress(total, total, 1, decode_start);
            }

            common_sampler_ptr sampler_saved;
            if (checkpoint_target) sampler_saved.reset(common_sampler_clone(sampler.get()));
            const std::size_t offered_this_round = draft.size();
            auto accepted = common_sampler_sample_and_accept_n(sampler.get(), ctx_, draft);
            if (accepted.empty()) throw std::runtime_error("target sampler returned no verification token");

            if (checkpoint_target && accepted.size() - 1 < offered_this_round) {
                draft = std::move(accepted);
                if (!checkpoint.load_tgt(ctx_, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                    throw std::runtime_error("could not restore target checkpoint after partial MTP acceptance");
                }
                llama_memory_seq_rm(llama_get_memory(ctx_), seq_id, checkpoint.pos_max + 1, -1);
                if (checkpoint_draft) {
                    if (!checkpoint.load_dft(ctx_dft_, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                        throw std::runtime_error("could not restore MTP draft checkpoint after partial acceptance");
                    }
                    llama_memory_seq_rm(llama_get_memory(ctx_dft_), seq_id, checkpoint.pos_max + 1, -1);
                }
                prompt.resize(checkpoint.n_tokens);
                sampler = std::move(sampler_saved);
                n_past = static_cast<int32_t>(prompt.size());
                continue;
            }

            const int accepted_drafts = static_cast<int>(accepted.size() - 1);
            common_speculative_accept(spec.get(), seq_id, static_cast<uint16_t>(accepted_drafts));
            n_past += accepted_drafts;
            drafts_offered += static_cast<int>(offered_this_round);
            drafts_accepted += accepted_drafts;

            bool stop = false;
            for (llama_token token : accepted) {
                prompt.push_back(last);
                last = token;
                if (llama_vocab_is_eog(vocab, last)) {
                    // See the direct-decode path: emit the final marker so
                    // Strata can recognize a clean model stop at its API
                    // boundary.  It is never exposed as assistant text.
                    std::cout << "T " << last << std::endl;
                    ++generated;
                    stop = true;
                    break;
                }
                if (channel.cancelled() || generated >= request.max_new) break;
                std::cout << "T " << last << std::endl;
                ++generated;
            }
            draft.clear();
            llama_memory_seq_rm(llama_get_memory(ctx_), seq_id, n_past, -1);
            llama_memory_seq_rm(llama_get_memory(ctx_dft_), seq_id, n_past, -1);

            if (stop) {
                emit_done(generated, total, prompt_ms, elapsed_ms(decode_start), "stop", drafts_accepted, drafts_offered, reused);
                return;
            }
        }

        emit_done(generated, total, prompt_ms, elapsed_ms(decode_start),
                  channel.cancelled() ? "cancel" : "length", drafts_accepted, drafts_offered, reused);
    }

    Options options_;
    common_params params_;
    common_init_result_ptr target_;
    common_speculative_init_result_ptr mtp_init_;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    llama_context* ctx_dft_ = nullptr;
    common_context_seq_rm_type target_seq_rm_type_ = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type draft_seq_rm_type_ = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    std::vector<PromptCacheEntry> prompt_cache_;
    size_t prompt_cache_bytes_ = 0;
    size_t prompt_cache_evictions_ = 0;
    uint64_t prompt_cache_clock_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    std::string error;
    bool show_help = false;
    const auto options = parse_options(argc, argv, error, show_help);
    if (show_help) {
        usage(std::cout);
        return 0;
    }
    if (!options) {
        std::cerr << kEngineName << ": " << error << '\n';
        usage(std::cerr);
        return 2;
    }

    // The service protocol owns stdout, while llama.cpp diagnostics are
    // retained on stderr.  Match llama.cpp's normal front ends by installing
    // the common logger before backend initialization and keep the ordinary
    // INFO threshold.  Leaving the raw backend callback active emitted graph
    // construction messages for every decode step, creating avoidable file
    // I/O and synchronization pressure on this resident engine.
    common_init();
    common_log_set_verbosity_thold(LOG_LEVEL_INFO);
    llama_backend_init();
    int result = 0;
    try {
        Engine engine(*options);
        engine.announce();
        std::cout << "READY " << engine.max_context() << " stop" << std::endl;

        CommandChannel channel;
        channel.start();
        std::string command;
        while (channel.next(command)) {
            if (command.rfind("GENI", 0) == 0) {
                emit_error("vision requests are not supported by qwen35moe-native-adapter");
            } else if (command.rfind("GEN", 0) == 0) {
                Request request;
                std::string request_error;
                if (parse_gen(command, request, request_error)) {
                    engine.generate(request, channel);
                } else {
                    emit_error(request_error);
                }
            } else if (command == "VRAM" || command.rfind("VRAM ", 0) == 0) {
                emit_error("VRAM elastic control is not implemented by qwen35moe-native-adapter");
            } else {
                emit_error("unsupported engine command");
            }
            channel.complete();
            if (channel.quitting()) break;
        }
        channel.join();
    } catch (const std::exception& e) {
        std::cerr << kEngineName << ": " << protocol_text(e.what()) << '\n';
        result = 1;
    }
    llama_backend_free();
    common_log_flush(common_log_main());
    return result;
}
