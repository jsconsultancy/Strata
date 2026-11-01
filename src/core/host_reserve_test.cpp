// The page reserve (#3350's Windows counterpart of the Linux file-page credit): each running
// request's read prices its file-tier streaming into a ticket, every host admission subtracts the
// sum, and the first decode token closes the ticket.  Host-only: tickets, the subtraction every
// admission runs through, the growth cap, closing one ticket beside an open one, the price model.
// The flag's effect is at the call sites (pricing/tickets ask page_reserve_env() first), so this
// test clears STRATA_PAGE_RESERVE and checks the default-off gate plus the arithmetic that is
// flag-independent.
#include "strata/core/host_reserve.hpp"
#include "strata/core/conversation_memory.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
constexpr uint64_t MiB = 1ull << 20;
}

int main() {
#ifdef _WIN32
    _putenv_s("STRATA_PAGE_RESERVE", "");
#else
    (void) unsetenv("STRATA_PAGE_RESERVE");
#endif
    check(!page_reserve_env(), "STRATA_PAGE_RESERVE is read once and defaults off");
    check(host_reserve_open() == 0, "off: a request opens no ticket");
    host_reserve_add(0, 9 * MiB);
    check(host_reserve_bytes() == 0, "off (ticket 0): nothing is ever held");
    check(host_reserve_grow(0, 4 * MiB, 16 * MiB) == 0, "off (ticket 0): nothing grows");
    host_reserve_close(0);
    check(host_reserve_bytes() == 0, "off (ticket 0): closing changes nothing");

    // the empty hold is a no-op on every admission
    check(host_reserve_admission_available(std::optional<uint64_t>(100 * MiB)) == std::optional<uint64_t>(100 * MiB),
          "an empty hold leaves the sample as it was");
    check(!host_reserve_admission_available(std::nullopt).has_value(), "unknown stays unknown");

    // a priced request (the ticket calls work on ids whatever the gate handed out at the sites)
    const uint64_t a = 7, b = 8;
    host_reserve_add(a, 5 * MiB);
    check(host_reserve_bytes() == 5 * MiB, "the price is held");
    check(host_reserve_admission_available(std::optional<uint64_t>(100 * MiB)) == std::optional<uint64_t>(95 * MiB),
          "an admission sees the sample minus the hold");
    host_reserve_add(b, 3 * MiB);   // a second reading request (a BYIELD continuation opens its own)
    check(host_reserve_bytes() == 8 * MiB, "two tickets add up");
    check(host_reserve_admission_available(std::optional<uint64_t>(4 * MiB)) == std::optional<uint64_t>(0),
          "a sample smaller than the hold saturates at zero");
    check(!conversation_memory_admit(host_reserve_admission_available(std::optional<uint64_t>(9 * MiB)),
                                     2 * MiB, 6 * MiB),
          "the admission fails on the reduced room (9 MiB less 8 held cannot cover 2 plus a 6 MiB floor)");
    check(conversation_memory_admit(host_reserve_admission_available(std::optional<uint64_t>(16 * MiB)),
                                    2 * MiB, 6 * MiB),
          "the same admission passes with room beside the hold");

    // the refault growth is capped by the global figure
    check(host_reserve_grow(a, 2 * MiB, 10 * MiB) == 2 * MiB, "a stall grows the hold within the cap");
    check(host_reserve_grow(b, 1 * MiB, 10 * MiB) == 0, "growth at the cap adds nothing");
    check(host_reserve_bytes() == 10 * MiB, "the hold sits at the cap");

    // one request's decode closes its ticket; the other's price stands (a holds 5 + the 2 grown)
    host_reserve_close(a);
    check(host_reserve_bytes() == 3 * MiB, "closing one ticket keeps the other's price");
    host_reserve_close(a);
    check(host_reserve_bytes() == 3 * MiB, "closing twice is a no-op");
    host_reserve_close(b);
    check(host_reserve_bytes() == 0, "closing every ticket drops the hold");
    check(host_reserve_admission_available(std::optional<uint64_t>(100 * MiB)) == std::optional<uint64_t>(100 * MiB),
          "after the closes the admissions see the sample again");

    // saturation and width: a huge hold never underflows the subtraction
    host_reserve_add(a, UINT64_MAX / 2);
    check(host_reserve_admission_available(std::optional<uint64_t>(100 * MiB)) == std::optional<uint64_t>(0),
          "a sample under a huge hold saturates at zero");
    host_reserve_close(a);
    check(host_reserve_bytes() == 0, "and it closes clean");

    // the price model: seeded, observed, flag-gated
    check(page_reserve_env() ? true : page_reserve_price(100) == 0, "off: no price is asked");
    page_reserve_observe(300 * MiB, 100);   // off: remembers nothing
    check(page_reserve_price(100) == 0, "off: a read of any size prices 0");

    std::printf("host_reserve_test: %d checks passed\n", checks);
    return 0;
}
