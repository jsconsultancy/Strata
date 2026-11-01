// The host ledger (STRATA_HOST_LEDGER): one room figure for every host admission — the sample
// minus the page reserve's holds, the open tickets and the margin — with tickets for large
// allocations still in flight and a reclaimer registry the arbiter runs on a shortage.  Host-only:
// the ticket lifecycle, the seam's arithmetic and its off-delegation, the registry order, the
// arbiter's reclaim-then-credit recheck, the cache-shed rule.  The flag is read once, so one
// executable runs both modes: plain = off (the gate and the exact delegation to
// host_reserve_admission_available), `--on` = on with STRATA_HOST_LEDGER_MARGIN_MIB=4 so every
// room figure below is exact.  The page reserve stays off in both modes (its holds are seeded by
// hand where the seam's delegation needs one).
#include "strata/core/host_ledger.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/core/host_reserve.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
constexpr uint64_t MiB = 1ull << 20;

// one-shot reclaimers: the first call frees min(target, cap), then they have nothing left
struct FakeReclaim { int calls = 0; uint64_t cap; };
uint64_t fake_reclaim(void* ctx, uint64_t target) {
    auto& f = *static_cast<FakeReclaim*>(ctx);
    ++f.calls;
    if (f.calls > 1) return 0;
    return target < f.cap ? target : f.cap;
}
} // namespace

int main(int argc, char** argv) {
    const bool on = argc > 1 && std::strcmp(argv[1], "--on") == 0;
#ifdef _WIN32
    _putenv_s("STRATA_PAGE_RESERVE", "");
    if (on) { _putenv_s("STRATA_HOST_LEDGER", "1"); _putenv_s("STRATA_HOST_LEDGER_MARGIN_MIB", "4"); }
    else    { _putenv_s("STRATA_HOST_LEDGER", "");   _putenv_s("STRATA_HOST_LEDGER_MARGIN_MIB", ""); }
#else
    unsetenv("STRATA_PAGE_RESERVE");
    if (on) { setenv("STRATA_HOST_LEDGER", "1", 1); setenv("STRATA_HOST_LEDGER_MARGIN_MIB", "4", 1); }
    else    { unsetenv("STRATA_HOST_LEDGER");       unsetenv("STRATA_HOST_LEDGER_MARGIN_MIB"); }
#endif

    // the pure cache-shed rule (mode-independent, so once here)
    check(host_ledger_effective_cache_budget(0, 100 * MiB, 30 * MiB) == 0, "a zero cache stays zero");
    check(host_ledger_effective_cache_budget(100 * MiB, 150 * MiB, 30 * MiB) == 100 * MiB,
          "the cache is never grown past its configured budget");
    check(host_ledger_effective_cache_budget(100 * MiB, 120 * MiB, 30 * MiB) == 90 * MiB,
          "the cache shrinks to the room beside the planned asks");
    check(host_ledger_effective_cache_budget(100 * MiB, 30 * MiB, 30 * MiB) == 0,
          "a room that cannot cover the plans sheds the cache entirely");

    check(host_ledger_env() == on, "STRATA_HOST_LEDGER is read once and matches the mode");

    if (!on) {
        // ---- off: every entry point is inert, and the seam is exactly the page reserve's ----
        check(host_ledger_ticket_open(10 * MiB) == 0, "off: no ticket is ever opened");
        check(host_ledger_ticket_bytes() == 0, "off: nothing is held");
        host_ledger_ticket_resize(0, 5 * MiB);
        host_ledger_ticket_close(0);
        check(host_ledger_ticket_bytes() == 0, "off (ticket 0): resize and close are no-ops");
        host_ledger_register("nothing", &fake_reclaim, nullptr);
        check(host_ledger_reclaim(8 * MiB) == 0, "off: the registry stays empty and reclaims nothing");
        check(host_ledger_admit(0, 0, "the off check"), "off: the plain sample check still runs (need 0, floor 0)");
        check(!host_ledger_admit(0, UINT64_MAX, "the off check"), "off: an impossible floor is refused without reclaiming");

        // the byte-identical property: the seam delegates, margin and tickets never appear
        host_reserve_add(7, 5 * MiB);   // a hand-seeded page-reserve hold (the flag off, the arithmetic live)
        check(host_ledger_admission_available(std::optional<uint64_t>(100 * MiB)) ==
                  std::optional<uint64_t>(95 * MiB),
              "off: the seam is the page reserve's subtraction and nothing else");
        check(host_ledger_admission_available(std::optional<uint64_t>(4 * MiB)) == std::optional<uint64_t>(0),
              "off: the delegated subtraction saturates at zero like the reserve's own");
        check(!host_ledger_admission_available(std::nullopt).has_value(), "off: an unknown sample stays unknown");
        host_reserve_close(7);

        std::printf("host_ledger_test: %d checks passed (off)\n", checks);
        return 0;
    }

    // ---- on: margin 4 MiB, page reserve off (its term is 0), real tickets ----
    const uint64_t a = host_ledger_ticket_open(10 * MiB);
    check(a != 0, "on: a ticket opens with a real id");
    check(host_ledger_ticket_bytes() == 10 * MiB, "on: the promise is held");
    host_ledger_ticket_resize(a, 6 * MiB);
    check(host_ledger_ticket_bytes() == 10 * MiB, "on: resize only firms the estimate up");
    const uint64_t b = host_ledger_ticket_open(3 * MiB);
    check(b != 0 && b != a, "on: a second ticket gets its own id");
    check(host_ledger_ticket_bytes() == 13 * MiB, "on: two tickets add up");
    host_ledger_ticket_resize(a, 12 * MiB);
    check(host_ledger_ticket_bytes() == 15 * MiB, "on: the estimate firms up and the hold follows");

    check(host_ledger_admission_available(std::optional<uint64_t>(100 * MiB)) == std::optional<uint64_t>(81 * MiB),
          "on: the seam is the sample minus the holds minus the margin (100 - 15 - 4)");
    check(host_ledger_admission_available(std::optional<uint64_t>(5 * MiB)) == std::optional<uint64_t>(0),
          "on: a sample under the holds saturates at zero");
    check(!host_ledger_admission_available(std::nullopt).has_value(), "on: an unknown sample stays unknown");

    host_ledger_ticket_close(a);
    check(host_ledger_ticket_bytes() == 3 * MiB, "on: closing one ticket keeps the other's promise");
    host_ledger_ticket_close(a);
    check(host_ledger_ticket_bytes() == 3 * MiB, "on: closing twice is a no-op");
    host_ledger_ticket_close(b);
    check(host_ledger_admission_available(std::optional<uint64_t>(100 * MiB)) == std::optional<uint64_t>(96 * MiB),
          "on: with the tickets closed only the margin is held (100 - 4)");
    const uint64_t c = host_ledger_ticket_open(0);
    check(c != 0 && host_ledger_ticket_bytes() == 0, "on: a zero-byte ticket holds nothing");

    // the reclaimers: registration order, until-need, exhaustion
    FakeReclaim ra{0, 5 * MiB}, rb{0, 4 * MiB};
    host_ledger_register("first", &fake_reclaim, &ra);
    host_ledger_register("second", &fake_reclaim, &rb);
    check(host_ledger_reclaim(2 * MiB) == 2 * MiB, "a short ask stops at the first reclaimer that covers it");
    check(ra.calls == 1 && rb.calls == 0, "the later reclaimer was not run");
    check(host_ledger_reclaim(100 * MiB) == 4 * MiB, "the run continues down the order until the registry is spent");
    check(ra.calls == 2 && rb.calls == 1, "each reclaimer was asked in turn");
    check(host_ledger_reclaim(10 * MiB) == 0, "an exhausted registry frees nothing");
    check(host_ledger_reclaim(0) == 0, "a zero need asks nothing");

    // the arbiter: reclaim, then credit the freed bytes at once (a fresh sample would lag)
    FakeReclaim rc{0, UINT64_MAX / 2};
    host_ledger_register("last", &fake_reclaim, &rc);
    check(host_ledger_admit(UINT64_MAX / 4, 1 * MiB, "the arbiter check"),
          "a shortage runs the reclaimers and the freed bytes are credited immediately");
    check(rc.calls == 1, "the arbiter stopped at the reclaimer that covered the need");

    std::printf("host_ledger_test: %d checks passed (on)\n", checks);
    return 0;
}
