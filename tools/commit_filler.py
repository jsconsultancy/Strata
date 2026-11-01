"""tools/commit_filler.py - squeeze Windows' commit charge down to a target, hold it, release clean.

The Phase 7.1 acceptance is a reproduction: a start under ~3.2 GiB of available commit (RAM + page file -
`ullAvailPageFile`) that dies at `cublasCreate` / pinned staging today, and reaches READY with
`STRATA_HOST_LEDGER=1` (the ledger sheds the conversation cache, the 16384 chunk ceiling and PLE residency).
This tool IS the pressure: it allocates and *touches* memory until the commit that is left sits at the
target, holds it while the engine under test starts, and gives every byte back on exit or Ctrl+C.

    python tools/commit_filler.py --target-mib 3200 [--hold-seconds 120] [--sample-ms 2000]
    python tools/commit_filler.py --squeeze-mib 256 --hold-seconds 3        # the smoke test

  --target-mib N    allocate until ullAvailPageFile sits at (or just under) N MiB - the acceptance knob.
  --squeeze-mib N   allocate exactly N MiB - for smoke tests and stepwise squeezes; --target-mib wins if both.
  --min-free-mib N  refuse to squeeze below this (default 1024): a typo with Swift still serving must not
                    thrash the box.  The acceptance runs in the maintenance window, but the tool itself
                    stays honest at any hour.
  --hold-seconds N  hold the squeeze for N seconds (default 120); --hold-seconds 0 holds until Enter.

Touching pages matters: VirtualAlloc reserves commit at allocation, but an untouched filler measures nothing
and squeezes only the pagefile counter - the touched pages are what pushes the other processes' working sets
out.  Windows-only by design (the acceptance is the Windows commit limit; Linux has cgroups and needs no
counterpart).  The squeeze runs in whole 64 MiB chunks, so the final commit sits within one chunk of the
target; the hold loop re-reads the counter so you can see what the engine under test is really given.
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time

CHUNK_MIB = 64
PAGE = 4096

if sys.platform == "win32":
    import ctypes.wintypes as wintypes

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]


def read_status() -> dict | None:
    """The figure the engine's guard reads: GlobalMemoryStatusEx, in MiB.  None when it cannot be read."""
    if sys.platform != "win32":
        return None
    st = _MemoryStatusEx()
    st.dwLength = ctypes.sizeof(st)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    mib = 1 << 20
    return {
        "avail_phys_mib": st.ullAvailPhys // mib,
        "avail_commit_mib": st.ullAvailPageFile // mib,
        "total_commit_mib": st.ullTotalPageFile // mib,
        "load_pct": st.dwMemoryLoad,
    }


def plan(target_mib: int, squeeze_mib: int, status: dict, min_free_mib: int) -> tuple[int, str]:
    """How many MiB to allocate and why (the guards the operator sees).  0 with a reason = nothing to do."""
    if target_mib > 0:
        if target_mib < min_free_mib:
            return 0, f"target {target_mib} MiB is under the --min-free-mib floor of {min_free_mib} MiB"
        want = status["avail_commit_mib"] - target_mib
        if want <= 0:
            return 0, (f"commit is already at {status['avail_commit_mib']} MiB, at or under the "
                       f"{target_mib} MiB target - nothing to squeeze")
        return want, (f"squeezing {want} MiB: commit {status['avail_commit_mib']} -> {target_mib} MiB "
                      f"(of a {status['total_commit_mib']} MiB limit)")
    return squeeze_mib, f"squeezing a flat {squeeze_mib} MiB"


def allocate(want_mib: int, floor_mib: int = 0, chunk: int = CHUNK_MIB * (1 << 20)) -> tuple[list[bytearray], int]:
    """Allocate+touch `want_mib` in `chunk`-sized buffers; stop early when the commit counter reads
    <= floor_mib (the limit moved under us) or on a MemoryError.  Returns (held buffers, MiB allocated
    in whole chunks)."""
    held: list[bytearray] = []
    done_bytes = 0
    want_bytes = want_mib << 20
    while done_bytes + chunk <= want_bytes:
        try:
            buf = bytearray(chunk)
        except MemoryError:
            break
        buf[::PAGE] = b"\x01" * (chunk // PAGE)   # touched pages are the real squeeze
        held.append(buf)
        done_bytes += chunk
        if floor_mib > 0:
            s = read_status()
            if s and s["avail_commit_mib"] <= floor_mib:
                break
    return held, done_bytes >> 20


def main() -> int:
    ap = argparse.ArgumentParser(description="squeeze Windows commit charge to a target and hold it")
    ap.add_argument("--target-mib", type=int, default=0, help="allocate until this much commit is left (the 7.1 acceptance knob)")
    ap.add_argument("--squeeze-mib", type=int, default=0, help="allocate exactly this many MiB")
    ap.add_argument("--min-free-mib", type=int, default=1024, help="refuse to squeeze below this floor")
    ap.add_argument("--hold-seconds", type=int, default=120, help="hold for N seconds; 0 = until Enter")
    ap.add_argument("--sample-ms", type=int, default=2000, help="hold-loop sample period")
    args = ap.parse_args()
    if args.target_mib <= 0 and args.squeeze_mib <= 0:
        ap.error("one of --target-mib / --squeeze-mib is required")

    before = read_status()
    if before is None:
        print("commit_filler: GlobalMemoryStatusEx would not read (or not Windows) - nothing to do", file=sys.stderr)
        return 2
    want, why = plan(args.target_mib, args.squeeze_mib, before, args.min_free_mib)
    if want <= 0:
        print(f"commit_filler: {why}", file=sys.stderr)
        return 2
    print(f"commit_filler: {why}")
    t0 = time.monotonic()
    held, got = allocate(want, floor_mib=args.target_mib if args.target_mib > 0 else 0)
    after = read_status() or before
    print(f"commit_filler: holding {got} MiB in {len(held)} chunks; commit is now "
          f"{after['avail_commit_mib']} MiB (target {args.target_mib or '-'}, floor {args.min_free_mib}), physical "
          f"{after['avail_phys_mib']} MiB, load {after['load_pct']}% ({time.monotonic() - t0:.1f} s to squeeze)")
    if got == 0:
        print("commit_filler: the first chunk was refused - the commit limit is already spent", file=sys.stderr)
        return 2
    try:
        if args.hold_seconds > 0:
            deadline = time.monotonic() + args.hold_seconds
            while time.monotonic() < deadline:
                time.sleep(args.sample_ms / 1000)
                s = read_status()
                if s:
                    print(f"commit_filler: ... commit {s['avail_commit_mib']} MiB, physical {s['avail_phys_mib']} MiB")
        else:
            input("commit_filler: press Enter to release")
    except KeyboardInterrupt:
        print("commit_filler: interrupted - releasing")
    held.clear()
    end = read_status() or before
    print(f"commit_filler: released; commit is back to {end['avail_commit_mib']} MiB, physical {end['avail_phys_mib']} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
