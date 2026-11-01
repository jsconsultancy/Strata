#include "strata/core/host_reserve.hpp"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace strata::core {

namespace {
// The tickets are opened once per request-read and closed at its first decode, so this map
// lives with the requests, not with the streaming: a mutex is far cheaper than the hot paths
// it would guard, and the hot paths never touch it — they only grow a ticket at a read's end.
std::mutex g_mu;
std::unordered_map<uint64_t, uint64_t> g_tickets;
std::atomic<uint64_t> g_total{0};
std::atomic<uint64_t> g_next{1};

// the observed file-tier bytes per read token.  Seeded at 2 MiB: the prompt path's measured
// 24 GB over a fresh 17K-token prompt is 1.44 MB/token, rounded up; the EMA (alpha 0.3) walks
// it to the pack's and the cache's real figure after a few requests.
double g_price_per_token = 2.0 * (1ull << 20);
}

bool page_reserve_env() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PAGE_RESERVE");
        return v && v[0] != '0';
    }();
    return on;
}

uint64_t host_reserve_open() {
    if (!page_reserve_env()) return 0;
    return g_next.fetch_add(1, std::memory_order_relaxed);
}

void host_reserve_add(uint64_t t, uint64_t bytes) {
    if (t == 0 || bytes == 0) return;
    std::lock_guard<std::mutex> lk(g_mu);
    g_tickets[t] += bytes;
    g_total.fetch_add(bytes, std::memory_order_relaxed);
}

uint64_t host_reserve_grow(uint64_t t, uint64_t bytes, uint64_t cap) {
    if (t == 0 || bytes == 0) return 0;
    std::lock_guard<std::mutex> lk(g_mu);
    const uint64_t cur = g_total.load(std::memory_order_relaxed);
    const uint64_t room = cur < cap ? cap - cur : 0;
    const uint64_t add = bytes < room ? bytes : room;
    if (add == 0) return 0;
    g_tickets[t] += add;
    g_total.fetch_add(add, std::memory_order_relaxed);
    return add;
}

void host_reserve_close(uint64_t t) {
    if (t == 0) return;
    std::lock_guard<std::mutex> lk(g_mu);
    const auto it = g_tickets.find(t);
    if (it == g_tickets.end()) return;
    g_total.fetch_sub(it->second, std::memory_order_relaxed);
    g_tickets.erase(it);
}

uint64_t host_reserve_bytes() {
    return g_total.load(std::memory_order_relaxed);
}

std::optional<uint64_t> host_reserve_admission_available(std::optional<uint64_t> available) {
    if (!available) return available;
    const uint64_t held = host_reserve_bytes();
    return *available > held ? *available - held : uint64_t(0);
}

uint64_t page_reserve_price(int64_t tokens) {
    if (!page_reserve_env() || tokens <= 0) return 0;
    std::lock_guard<std::mutex> lk(g_mu);
    return (uint64_t) ((double) tokens * g_price_per_token);
}

void page_reserve_observe(uint64_t bytes, int64_t tokens) {
    if (!page_reserve_env() || tokens <= 0) return;
    const double per = (double) bytes / (double) tokens;
    if (bytes == 0) return;   // a fully cached read streamed nothing: it says nothing about a cold read
    std::lock_guard<std::mutex> lk(g_mu);
    g_price_per_token += 0.3 * (per - g_price_per_token);
}

} // namespace strata::core
