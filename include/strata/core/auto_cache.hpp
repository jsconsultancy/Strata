// include/strata/core/auto_cache.hpp - the slot count of the automatic expert cache (--expert-cache auto), as a rule
// with a test.  The auto cache is sized from the VRAM free after the weights, the session and the drafter, minus
// --vram-reserve-mib.  What was never priced is what the load allocates AFTER the cache whatever the settings: the
// CUDA libraries the prompt path and the verifier warm up (cuBLAS handles and workspaces, lazily loaded modules) and
// the verifier's small buffers.  Measured 244-247 MiB on RTX A4000 / A6000; the cache now leaves kLateAllocMib too,
// so the card ends with about the reserve it was asked to keep (#4474).  Upstream cfa197e2.
#pragma once

#include <cstdint>

namespace strata::core {

/// MiB the load allocates after the cache that no setting prices (measured 244-247, a little margin).
constexpr int64_t kLateAllocMib = 256;

struct AutoCacheSlots {
    int64_t slots = 0;        ///< the cache's slot count (of the largest blob), >= 0
    bool late_priced = false; ///< kLateAllocMib was left out of the cache
};

/// `free_b`: VRAM free now; `reserve_b`: the reserve and everything else that is priced (bytes); `blob`: the largest
/// slot; `min_slots`: the least a working cache needs (#496).  `late_mib` 0 turns the late allocation off.  The late
/// allocation is only left out when the cache still has `min_slots` after it: a small card keeps the sizing it had.
inline AutoCacheSlots auto_cache_slots(int64_t free_b, int64_t reserve_b, int64_t blob, int64_t min_slots,
                                       int64_t late_mib = kLateAllocMib) {
    AutoCacheSlots r;
    if (blob <= 0) return r;
    const int64_t plain = (free_b - reserve_b) / blob;
    const int64_t late = late_mib > 0 ? (free_b - reserve_b - (late_mib << 20)) / blob : plain;
    if (late_mib > 0 && late >= min_slots && late > 0) {
        r.slots = late;
        r.late_priced = true;
    } else {
        r.slots = plain > 0 ? plain : 0;
    }
    return r;
}

} // namespace strata::core
