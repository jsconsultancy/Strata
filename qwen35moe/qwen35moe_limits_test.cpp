#include "qwen35moe_limits.hpp"

#include <cstdint>
#include <iostream>

int main() {
    using strata::qwen35moe::fit_max_new;
    using strata::qwen35moe::mtp_full_draft_fits;
    struct Case {
        std::size_t prompt;
        int32_t requested;
        uint32_t context;
        int32_t cap;
        int32_t expected;
    };
    constexpr Case cases[] = {
        {200, 1024, 262144, 32768, 1024},       // normal request
        {200, 65536, 262144, 32768, 32768},     // profile output ceiling
        {250000, 32768, 262144, 32768, 12144},  // context room wins
        {262144, 1, 262144, 32768, 0},          // no room remains
        {10, 0, 262144, 32768, 0},              // valid no-output prefill
    };
    for (const auto& test : cases) {
        const int32_t actual = fit_max_new(test.prompt, test.requested, test.context, test.cap);
        if (actual != test.expected) {
            std::cerr << "fit_max_new failed: expected " << test.expected << ", got " << actual << '\n';
            return 1;
        }
    }
    if (!mtp_full_draft_fits(3, 3) || mtp_full_draft_fits(2, 3) || mtp_full_draft_fits(0, 3)) {
        std::cerr << "MTP tail-fit decision failed\n";
        return 1;
    }
    return 0;
}
