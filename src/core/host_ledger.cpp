#include "strata/core/host_ledger.hpp"

#include "strata/core/conversation_memory.hpp"
#include "strata/core/host_reserve.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace strata::core {

namespace {
// One mutex over the tickets and the registry: both live with the admissions (startup sequence,
// the serve loop), never on a hot path, and the registry must not reorder against a concurrent
// reclaim even if a future site runs off the loop thread.
std::mutex g_mu;
std::unordered_map<uint64_t, uint64_t> g_tickets;
std::atomic<uint64_t> g_ticket_total{0};
std::atomic<uint64_t> g_next{1};

struct Reclaimer {
    const char* what;
    host_ledger_reclaim_fn fn;
    void* ctx;
};
std::vector<Reclaimer> g_reclaimers;

uint64_t margin_bytes() {
    static const uint64_t mib = [] {
        const char* v = std::getenv("STRATA_HOST_LEDGER_MARGIN_MIB");
        if (v == nullptr || v[0] == '\0') return 512ull;
        const unsigned long long parsed = std::strtoull(v, nullptr, 10);
        return (uint64_t) parsed;
    }();
    return mib << 20;
}
} // namespace

bool host_ledger_env() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HOST_LEDGER");
        return v && v[0] != '0';
    }();
    return on;
}

uint64_t host_ledger_ticket_open(uint64_t bytes) {
    if (!host_ledger_env()) return 0;
    const uint64_t t = g_next.fetch_add(1, std::memory_order_relaxed);
    if (bytes != 0) {
        std::lock_guard<std::mutex> lk(g_mu);
        g_tickets[t] = bytes;
        g_ticket_total.fetch_add(bytes, std::memory_order_relaxed);
    }
    return t;
}

void host_ledger_ticket_resize(uint64_t t, uint64_t bytes) {
    if (t == 0 || bytes == 0) return;
    std::lock_guard<std::mutex> lk(g_mu);
    const auto it = g_tickets.find(t);
    if (it == g_tickets.end() || bytes <= it->second) return;   // monotone: the estimate only firms up
    g_ticket_total.fetch_add(bytes - it->second, std::memory_order_relaxed);
    it->second = bytes;
}

void host_ledger_ticket_close(uint64_t t) {
    if (t == 0) return;
    std::lock_guard<std::mutex> lk(g_mu);
    const auto it = g_tickets.find(t);
    if (it == g_tickets.end()) return;
    g_ticket_total.fetch_sub(it->second, std::memory_order_relaxed);
    g_tickets.erase(it);
}

uint64_t host_ledger_ticket_bytes() {
    return g_ticket_total.load(std::memory_order_relaxed);
}

std::optional<uint64_t> host_ledger_admission_available(std::optional<uint64_t> sample) {
    if (!host_ledger_env()) return host_reserve_admission_available(std::move(sample));
    if (!sample) return sample;
    // the page reserve's holds, the tickets, then the margin — each saturating
    uint64_t room = host_reserve_admission_available(std::move(sample)).value_or(0);
    const uint64_t held = g_ticket_total.load(std::memory_order_relaxed);
    room = room > held ? room - held : 0;
    const uint64_t margin = margin_bytes();
    return room > margin ? room - margin : uint64_t(0);
}

void host_ledger_register(const char* what, host_ledger_reclaim_fn fn, void* ctx) {
    if (!host_ledger_env() || fn == nullptr) return;
    std::lock_guard<std::mutex> lk(g_mu);
    g_reclaimers.push_back(Reclaimer{what, fn, ctx});
}

uint64_t host_ledger_reclaim(uint64_t need) {
    if (!host_ledger_env() || need == 0) return 0;
    uint64_t freed = 0;
    std::lock_guard<std::mutex> lk(g_mu);
    for (const Reclaimer& r : g_reclaimers) {
        if (freed >= need) break;
        freed += r.fn(r.ctx, need - freed);   // each reclaimer sees what is still missing
    }
    return freed;
}

bool host_ledger_admit(uint64_t need, uint64_t floor, const char* what) {
    auto sample_of = [] { const auto m = sample_host_memory(); return m ? std::optional<uint64_t>(m->available) : std::nullopt; };
    std::optional<uint64_t> avail = host_ledger_admission_available(sample_of());
    if (conversation_memory_admit(avail, need, floor)) return true;
    if (!host_ledger_env()) return false;
    const uint64_t freed = host_ledger_reclaim(need);
    if (freed != 0) {
        // credit the freed bytes at once: the fresh sample may lag the kernel, and the freed
        // figure is the one thing the ledger knows exactly
        const std::optional<uint64_t> fresh = host_ledger_admission_available(sample_of());
        const uint64_t credited = fresh.value_or(0) + freed;
        if (conversation_memory_admit(std::optional<uint64_t>(credited), need, floor)) {
            std::fprintf(stderr, "strata: host ledger: %s admitted after the reclaimers freed %llu MiB "
                                 "(needed %llu MiB beside a %llu MiB floor)\n",
                         what, (unsigned long long) (freed >> 20), (unsigned long long) (need >> 20),
                         (unsigned long long) (floor >> 20));
            return true;
        }
    }
    std::fprintf(stderr, "strata: host ledger: %s short: %llu MiB beside a %llu MiB floor, room %s MiB%s\n",
                 what, (unsigned long long) (need >> 20), (unsigned long long) (floor >> 20),
                 avail ? std::to_string(*avail >> 20).c_str() : "unknown",
                 freed != 0 ? "; the reclaimers had nothing more to give" : "");
    return false;
}

} // namespace strata::core
