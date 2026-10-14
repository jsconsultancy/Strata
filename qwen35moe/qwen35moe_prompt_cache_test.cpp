#include "qwen35moe_prompt_cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void check(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}
}

int main() {
    using strata::qwen35moe::prompt_cache_reusable_prefix;
    using strata::qwen35moe::prompt_cache_checkpoint_due;
    using strata::qwen35moe::prompt_cache_next_checkpoint;
    const std::vector<int> root{1, 2, 3};
    check(prompt_cache_reusable_prefix(root, std::vector<int>{1, 2, 3, 4}, 3) == 3,
          "an exact cached prefix may be reused");
    check(prompt_cache_reusable_prefix(root, std::vector<int>{1, 2, 9, 4}, 3) == 0,
          "a mismatch never reuses beyond the first differing token");
    check(prompt_cache_reusable_prefix(root, std::vector<int>{1, 2, 3}, 2) == 0,
          "the final request token remains uncached for logits");
    check(prompt_cache_reusable_prefix(root, std::vector<int>{1, 2}, 2) == 0,
          "a cache entry cannot exceed the request prefix");
    check(prompt_cache_reusable_prefix(std::vector<int>{}, std::vector<int>{1}, 1) == 0,
          "an empty cache entry is not a cache hit");
    check(prompt_cache_checkpoint_due(8192, 23151, 8192),
          "the first completed batch is checkpointed");
    check(prompt_cache_checkpoint_due(16384, 23151, 8192),
          "the second completed batch is checkpointed");
    check(!prompt_cache_checkpoint_due(24576, 32768, 8192),
          "non-geometric completed batches are not checkpointed");
    check(prompt_cache_checkpoint_due(32768, 65536, 8192),
          "power-of-two batch multiples are checkpointed");
    check(prompt_cache_checkpoint_due(23151, 23151, 8192),
          "the final completed prefix is checkpointed");
    check(!prompt_cache_checkpoint_due(0, 23151, 8192),
          "an empty prefix is never checkpointed");
    check(!prompt_cache_checkpoint_due(8192, 23151, 0),
          "a zero batch size cannot schedule a checkpoint");
    check(!prompt_cache_checkpoint_due(23152, 23151, 8192),
          "a prefix beyond the request cannot be checkpointed");
    check(prompt_cache_next_checkpoint(23151, 50000, 8192) == 32768,
          "a resumed final prefix realigns with the next geometric checkpoint");
    check(prompt_cache_next_checkpoint(32768, 50000, 8192) == 50000,
          "the request final prefix is used after the last geometric boundary");
    check(prompt_cache_next_checkpoint(0, 5000, 8192) == 5000,
          "a short request has only its final prefix checkpoint");
    check(prompt_cache_next_checkpoint(0, 262143, 8192) == 8192,
          "a cold long request starts at its first batch checkpoint");
    std::puts("qwen35moe_prompt_cache_test: 17 checks passed");
    return 0;
}
