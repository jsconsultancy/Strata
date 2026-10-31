// STRATA_PROFILE_SEED: the adaptive tier's counts start from the profile - the rank-graded prior for exactly the
// experts the fill placed, the seed never lowering a count already there, and the environment parse.  CPU only: no
// device is touched.
#include "strata/core/expert_cache.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}
void check_near(float got, float want, const char* what) {
    if (std::fabs(got - want) > 1e-5f) {
        std::fprintf(stderr, "FAIL: %s (got %.6f, want %.6f)\n", what, (double) got, (double) want);
        ++fails;
    }
}
}  // namespace

int main() {
    using Pair = std::pair<int32_t, int32_t>;
    const int64_t L = 2, E = 4;   // 8 entries, index = layer * 4 + expert
    // hottest first: (0,3), (1,0), (0,1), (1,2)
    const std::vector<Pair> ranked{{0, 3}, {1, 0}, {0, 1}, {1, 2}};
    // the fill placed (0,3), (0,1), (1,2); (1,0) never got a slot; (1,3) is resident but not ranked
    std::vector<int32_t> resident(L * E, strata::core::kNotResident);
    resident[0 * 4 + 3] = 0;
    resident[0 * 4 + 1] = 1;
    resident[1 * 4 + 2] = 2;
    resident[1 * 4 + 3] = 3;

    {
        std::vector<float> usage((size_t) (L * E), 0.0f);
        const int64_t n = strata::core::seed_profile_usage(usage, L, E, ranked, resident, 4.0f);
        check(n == 3, "three placed, profiled experts are seeded");
        check_near(usage[0 * 4 + 3], 4.0f, "rank 0 gets the full top");
        check_near(usage[0 * 4 + 1], 2.0f, "rank 2 of four gets the middle of the ramp");   // 4, 3, 2, 1
        check_near(usage[1 * 4 + 2], 1.0f, "the last rank gets a quarter of the top");
        check_near(usage[1 * 4 + 0], 0.0f, "a profiled expert the fill did not place is not seeded");
        check_near(usage[1 * 4 + 3], 0.0f, "an unprofiled resident is not seeded");
    }
    {
        // the seed is a floor: a count the engine already raised is never lowered
        std::vector<float> usage((size_t) (L * E), 0.0f);
        usage[0 * 4 + 3] = 9.0f;
        strata::core::seed_profile_usage(usage, L, E, ranked, resident, 4.0f);
        check_near(usage[0 * 4 + 3], 9.0f, "an existing higher count is kept");
    }
    {
        // an empty residency table means "no filter": every ranked pair is seeded (the all-resident shape)
        std::vector<float> usage((size_t) (L * E), 0.0f);
        const int64_t n = strata::core::seed_profile_usage(usage, L, E, ranked, {}, 4.0f);
        check(n == 4, "an empty table seeds every ranked pair");
        check_near(usage[1 * 4 + 0], 3.0f, "rank 1 of four gets 3.0");
    }
    {
        // the off and malformed shapes are no-ops, not writes
        std::vector<float> usage((size_t) (L * E), 0.0f);
        std::vector<float> empty;
        check(strata::core::seed_profile_usage(usage, L, E, ranked, resident, 0.0f) == 0, "top 0 is off");
        check(strata::core::seed_profile_usage(empty, L, E, ranked, resident, 4.0f) == 0, "a short usage vector is refused");
        check(strata::core::seed_profile_usage(usage, L, E, {}, resident, 4.0f) == 0, "an empty profile is a no-op");
        check_near(usage[0 * 4 + 3], 0.0f, "a refused call writes nothing");
    }
    // the environment parse, before profile_seed_top's one-time read: a number in range, then an invalid one (this
    // process only; the second call must keep the first value, which is the point of the memo)
    {
#if defined(_WIN32)
        _putenv_s("STRATA_PROFILE_SEED", "2.5");
#else
        setenv("STRATA_PROFILE_SEED", "2.5", 1);
#endif
        check_near(strata::core::profile_seed_top(), 2.5f, "STRATA_PROFILE_SEED parses");
#if defined(_WIN32)
        _putenv_s("STRATA_PROFILE_SEED", "not-a-number");
#else
        setenv("STRATA_PROFILE_SEED", "not-a-number", 1);
#endif
        check_near(strata::core::profile_seed_top(), 2.5f, "the value is read once");
    }
    if (fails == 0) std::printf("PASS: expert_profile_seed_test\n");
    return fails == 0 ? 0 : 1;
}
