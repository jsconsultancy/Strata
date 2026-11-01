#pragma once

#include <cstdint>
#include <optional>

namespace strata::core {

// The page reserve (#3350's Windows counterpart of the Linux reclaimable-file credit;
// STRATA_PAGE_RESERVE=1): the host bytes a running prompt's file-tier streaming is priced to
// need, held from every host admission — parking, session save, session restore — from the
// request's read until its first decode token.
//
// The reason: the host figure on Windows is min(ullAvailPhys, ullAvailPageFile), a sample that
// is *not a reservation* and carries no file-page credit (the header of conversation_memory.hpp
// says so; Linux credits the cgroup's clean inactive_file).  A cold prompt streaming its experts
// through the host is exactly the window where the concurrent consumers' samples go stale: a park
// admitted on a fresh sample takes the room the read is crossing, the read refaults, and the
// measured 19K cold-cache prompt pays it back in synchronous stalls.  The price is the observed
// file-tier bytes per read token (an EMA; seeded at 2 MiB/token, the 24 GB / 17K-token streaming
// measured for the prompt path, rounded up), times the tokens this read covers; a read stall
// grows the request's ticket, the first decode token closes it.  Several requests run at once
// (batch), so the hold is per ticket: a closing decode never drops another request's price.
// Off (the flag): no ticket opens, the counter stays 0, every admission subtracts 0 —
// byte-identical to before.

/// STRATA_PAGE_RESERVE=1 gates pricing, growth and tickets (read once; off by default).
bool page_reserve_env();

/// Opens a request's ticket (0 = the flag is off: no hold, every call below is inert at 0).
uint64_t host_reserve_open();

/// The bytes priced to ticket `t` (added to; BYIELD continuations open their own ticket).
void host_reserve_add(uint64_t t, uint64_t bytes);

/// Grows ticket `t` after a read stall (the refault signal), never past the reserve's global
/// cap; returns the bytes actually added.  Says nothing: the caller logs with its own words.
uint64_t host_reserve_grow(uint64_t t, uint64_t bytes, uint64_t cap);

/// The first decode token (or the request's end — a cancelled read releases like a finished
/// one).  Idempotent.
void host_reserve_close(uint64_t t);

/// The bytes held now, over all open tickets (0 when nothing is reading, and always 0 off).
uint64_t host_reserve_bytes();

/// The available RAM a host admission may use: the sample minus what running prompts are
/// holding.  nullopt (unknown) stays unknown; 0 subtracted when nothing is held.
std::optional<uint64_t> host_reserve_admission_available(std::optional<uint64_t> available);

/// Prices a read of `tokens` from the observed streaming: 0 when the flag is off.
uint64_t page_reserve_price(int64_t tokens);

/// Tells the price model what this request's read streamed (file-tier bytes over read tokens).
/// EMA, 0.3 toward the new figure.  Off: remembers nothing.
void page_reserve_observe(uint64_t bytes, int64_t tokens);

} // namespace strata::core
