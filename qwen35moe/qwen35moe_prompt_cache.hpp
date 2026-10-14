#pragma once

#include <cstddef>
#include <vector>

namespace strata::qwen35moe {

// A snapshot may only be restored when its token sequence is an exact prefix
// of the next request.  `max_reused` leaves at least the final input token for
// a fresh forward pass, so restored K/V state is never paired with stale logits.
template<class Token>
inline std::size_t prompt_cache_reusable_prefix(const std::vector<Token>& cached,
                                                 const std::vector<Token>& request,
                                                 std::size_t max_reused) {
    if (cached.empty() || cached.size() > max_reused || cached.size() > request.size()) return 0;
    for (std::size_t i = 0; i < cached.size(); ++i) {
        if (cached[i] != request[i]) return 0;
    }
    return cached.size();
}

// Capture a bounded set of completed prefill boundaries.  Powers of two give
// a changed request tail a large reusable checkpoint without making a
// full-context request copy every batch.  The final completed prefix is always
// useful for an exact follow-up, even when it is not a power-of-two boundary.
inline bool prompt_cache_checkpoint_due(std::size_t processed, std::size_t total,
                                        std::size_t batch_size) {
    if (processed == 0 || processed > total || batch_size == 0) return false;
    if (processed == total) return true;
    if (processed < batch_size || processed % batch_size != 0) return false;
    const std::size_t multiple = processed / batch_size;
    return (multiple & (multiple - 1)) == 0;
}

// Pick the next global checkpoint after an already-restored prefix.  A resume
// may begin at an arbitrary final-prefix boundary, so callers can trim one
// otherwise normal batch to realign later work with 32K/64K/... checkpoints.
// The returned value is always in (processed, total] for valid unfinished
// input; the final prefix is the safe fallback when no geometric boundary
// remains before it.
inline std::size_t prompt_cache_next_checkpoint(std::size_t processed, std::size_t total,
                                                 std::size_t batch_size) {
    if (batch_size == 0 || processed >= total) return total;
    std::size_t boundary = batch_size;
    while (boundary <= processed) {
        // Doubling would lie beyond total, so the final prefix is next.  This
        // comparison also avoids an overflow for a generic size_t caller.
        if (boundary > total / 2) return total;
        boundary *= 2;
    }
    return boundary <= total ? boundary : total;
}

} // namespace strata::qwen35moe
