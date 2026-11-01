#pragma once

#include <cstdint>
#include <optional>

namespace strata::core {

// The host ledger (the InferSSD memory-governance idea; STRATA_HOST_LEDGER=1, opt-in): one room
// figure every host admission shares, instead of each site sampling the OS and applying its own
// scattered margin.  room = sample − page-reserve holds − ledger tickets − margin, where the
// sample is what sample_host_memory() reads (already min(available physical, available commit),
// lowered by the cgroup / --memory-limit-mib), the page reserve is the running prompts' priced
// file streaming (host_reserve.hpp), the tickets are large allocations asked for but not yet
// visible to a fresh sample, and the margin (STRATA_HOST_LEDGER_MARGIN_MIB, default 512 MiB) is
// headroom for the consumers nobody prices (the OS, other processes' growth).
//
// The reason: the sample is a photograph, and every large ask photographs badly.  A resident PLE
// table admits itself counting an expert arena that is not there yet; the commit verdict warns on
// a planned figure no admission reads; the conversation cache is sized from its flag alone and a
// start under commit pressure (the measured 3.2 GiB free) dies at cublasCreate with room it
// never knew it had to shed.  The ledger prices the asks while they are in flight (a ticket:
// open when the bytes are promised, close when a fresh sample can see them — or when the ask was
// refused), keeps the page reserve's holds in the same subtraction, and — when an admission comes
// up short — runs registered reclaimers (caches that can give RAM back on demand) before the
// site falls back to its refusal.  A recheck after reclaiming credits the freed bytes at once: a
// fresh sample lags the kernel, and the freed figure is the one thing the ledger knows exactly.
//
// Off (the flag): ticket_open hands out no ids, the registry stays empty, and the admission seam
// is exactly host_reserve_admission_available() — which subtracts 0 when the page reserve is off
// too — so every site is byte-identical to before.

/// STRATA_HOST_LEDGER=1 gates ids, the registry and the margin (read once; off by default).
bool host_ledger_env();

/// Opens a ticket promising `bytes` (0 = the flag is off: nothing is held, the calls below are
/// inert at id 0).  Close it when the pages are visible to a fresh sample — or when the ask was
/// refused; the ticket covers only the window the sample photographs badly.
uint64_t host_ledger_ticket_open(uint64_t bytes);

/// Firms the estimate up, never down (monotone max) — a parse that learns the real peak mid-read.
void host_ledger_ticket_resize(uint64_t t, uint64_t bytes);

/// The ask landed (its pages are visible now) or was refused.  Idempotent.
void host_ledger_ticket_close(uint64_t t);

/// The bytes promised over all open tickets (0 when nothing is in flight, and always 0 off).
uint64_t host_ledger_ticket_bytes();

/// The admission seam — the room a host admission may use.  Off: exactly
/// host_reserve_admission_available(sample) (the page reserve's subtraction, 0 held when that
/// flag is off too).  On: the sample minus the page reserve's holds, the open tickets and the
/// margin, saturating at 0; nullopt (an unreadable sample) stays unknown.
std::optional<uint64_t> host_ledger_admission_available(std::optional<uint64_t> sample);

/// A cache that can give RAM back on demand: returns the bytes it actually freed for `target`
/// (0 = it has nothing left).  Called from the engine's startup / request-loop thread only —
/// the admissions are loop-thread, and the caches are not thread-safe beyond that.
using host_ledger_reclaim_fn = uint64_t (*)(void* ctx, uint64_t target);

/// Registers a reclaimer (no-op when the flag is off).  Registration order is the run order:
/// the cheapest gives-first order the sites were written in.
void host_ledger_register(const char* what, host_ledger_reclaim_fn fn, void* ctx);

/// Runs the reclaimers in order until `need` bytes are freed or the registry is exhausted;
/// returns the bytes freed (0 off).  Says nothing: the caller logs with its own words.
uint64_t host_ledger_reclaim(uint64_t need);

/// The arbiter: room for `need` beside `floor`; on a shortage, reclaim and recheck (the freed
/// bytes are credited at once).  Logs the one line that says what reclaimed or why it could not.
/// Off: the plain sample check, no reclaim, no line.
bool host_ledger_admit(uint64_t need, uint64_t floor, const char* what);

/// The cache-shed rule (pure): the conversation cache gets min(configured, room − planned) —
/// never more than configured, 0 when the room cannot cover the planned asks.  This is the
/// "frees cache slots" half of the low-commit start: the cache is the one big ask sized after
/// the room is known.
inline uint64_t host_ledger_effective_cache_budget(uint64_t configured, uint64_t room, uint64_t planned) {
    if (configured == 0) return 0;
    const uint64_t afford = room > planned ? room - planned : 0;
    return configured < afford ? configured : afford;
}

} // namespace strata::core
