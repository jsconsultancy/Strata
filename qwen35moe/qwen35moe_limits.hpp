#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::qwen35moe {

// The model context is the hard boundary.  Return the largest safe completion
// for a syntactically valid GEN request; callers reject a prompt that itself
// is over context before calling this helper.
constexpr int32_t fit_max_new(std::size_t prompt_tokens, int32_t requested, uint32_t context,
                              int32_t output_cap) noexcept {
    if (requested <= 0 || output_cap <= 0 || prompt_tokens >= context) return 0;
    const uint64_t room = static_cast<uint64_t>(context) - prompt_tokens;
    const uint64_t fitted = std::min<uint64_t>(
        {static_cast<uint64_t>(requested), static_cast<uint64_t>(output_cap), room});
    return static_cast<int32_t>(fitted);
}

// The upstream MTP draft implementation uses the configured profile n_max.
// Per-call room can only decide whether that full draft is safe to offer.
constexpr bool mtp_full_draft_fits(int32_t draft_room, int32_t spec_n_max) noexcept {
    return spec_n_max > 0 && draft_room >= spec_n_max;
}

}  // namespace strata::qwen35moe
