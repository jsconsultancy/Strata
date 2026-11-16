"""Regression tests for the 2026-11-03 audit pass, round four: the
startup-critical orphan scan is now time-bounded.

Root cause of the wholesale tests/terminal/test_runtime.py failures on this
machine (27 failed / 26 errors): every engine start runs
`end_leftover_engines` -> `procs.end_orphans` -> `psutil.process_iter` as a
SYNCHRONOUS, unbounded full-system scan before the engine can serve. While
per-handle process queries from this interpreter were slow (~90 ms each,
measured 47-57 s for ~500 processes -- the same table PowerShell lists in
0.1 s), the scan alone consumed every startup budget: the server child
never reached READY, the host answered state=starting until
startup_wait_expired, and the leaked host then held its log open so the
tests' temp-dir teardown cascaded (WinError 32). A best-effort cleanup must
never be the reason a start fails: the scan is now bounded by
STRATA_ORPHAN_SCAN_BUDGET_S (default 5 s; 0/negative = the old unbounded
behavior), and a tripped scan says so, returning PARTIAL results.
"""

import os
import time

import pytest


def test_orphan_scan_budget_env_parsing(monkeypatch):
    from strata_control import procs
    monkeypatch.delenv("STRATA_ORPHAN_SCAN_BUDGET_S", raising=False)
    assert procs.orphan_scan_budget() == 5.0          # the default ceiling
    monkeypatch.setenv("STRATA_ORPHAN_SCAN_BUDGET_S", "45")
    assert procs.orphan_scan_budget() == 45.0
    monkeypatch.setenv("STRATA_ORPHAN_SCAN_BUDGET_S", "0")
    assert procs.orphan_scan_budget() == float("inf")  # the old behavior
    monkeypatch.setenv("STRATA_ORPHAN_SCAN_BUDGET_S", "-3")
    assert procs.orphan_scan_budget() == float("inf")
    monkeypatch.setenv("STRATA_ORPHAN_SCAN_BUDGET_S", "garbage")
    assert procs.orphan_scan_budget() == 5.0           # a bad value = default


def test_orphans_of_returns_partial_when_the_budget_trips(monkeypatch):
    """A trip must come back PROMPTLY with partial results -- the start that
    needs the GPU must not wait out a slow enumeration."""
    from strata_control import procs
    monkeypatch.delenv("STRATA_ORPHAN_SCAN_BUDGET_S", raising=False)
    t0 = time.monotonic()
    found, tripped = procs.orphans_of(r"D:\nonexistent-engine.exe", budget_s=0.05)
    elapsed = time.monotonic() - t0
    assert isinstance(found, list) and found == []      # nothing matched anyway
    assert tripped is True, "a 50 ms budget cannot finish a real scan"
    assert elapsed < 5.0, f"the trip took {elapsed:.1f} s to notice"


def test_orphans_of_completes_without_tripping_when_fast(monkeypatch):
    """A fast (empty) enumeration completes: budget_tripped stays False and
    the return is the old contract's content, ([], False)."""
    from strata_control import procs

    class _FakeIter:
        @staticmethod
        def process_iter(attrs):
            return iter(())

    monkeypatch.setattr(procs.psutil, "process_iter", _FakeIter.process_iter)
    monkeypatch.setattr(procs.psutil, "Process",
                        lambda *a, **k: (_ for _ in ()).throw(procs.psutil.Error))
    found, tripped = procs.orphans_of(r"D:\nonexistent-engine.exe")
    assert found == [] and tripped is False
    # explicit budget=None means "use the env": honored either way
    found2, tripped2 = procs.orphans_of(r"D:\nonexistent-engine.exe", budget_s=None)
    assert found2 == [] and tripped2 is False


def test_end_orphans_says_when_the_scan_was_cut_short(monkeypatch):
    from strata_control import procs
    monkeypatch.setattr(procs, "orphan_scan_budget", lambda: 0.001)
    said = []
    ended = procs.end_orphans(r"D:\nonexistent-engine.exe", grace=0, settle=0,
                              say=said.append)
    assert ended == []
    assert any("budget" in s for s in said), said
    assert any("STRATA_ORPHAN_SCAN_BUDGET_S" in s for s in said), said


def test_end_orphans_without_a_say_still_returns(monkeypatch):
    from strata_control import procs
    monkeypatch.setattr(procs, "orphan_scan_budget", lambda: 0.001)
    assert procs.end_orphans(r"D:\nonexistent-engine.exe", grace=0, settle=0) == []


def test_end_leftover_engines_still_signatures(monkeypatch):
    """The server-child entry keeps its contract (None return; a failed scan
    must not stop a start) with the budgeted scan underneath."""
    import io
    import contextlib
    from strata_control import procs
    from serve import server
    monkeypatch.setattr(procs, "orphan_scan_budget", lambda: 0.001)
    monkeypatch.setenv("STRATA_ORPHAN_SCAN_BUDGET_S", "0.001")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        server.end_leftover_engines(r"D:\nonexistent-engine.exe")
    assert "budget" in buf.getvalue()


def test_end_leftover_engines_skipped_for_host_spawned_children(monkeypatch):
    """A server the runtime HOST spawned (STRATA_RUNTIME_PROFILE set) skips
    the sweep: the host cleaned this profile's recorded leftovers right
    before spawning, and the scan must not sit inside its startup/rollback
    wait windows (the rolled_back=False failures on a slow-enumeration
    machine were exactly that)."""
    import io
    import contextlib
    from strata_control import procs
    from serve import server
    monkeypatch.setattr(procs, "orphan_scan_budget", lambda: 0.001)
    monkeypatch.delenv("STRATA_KEEP_LEFTOVERS", raising=False)
    monkeypatch.setenv("STRATA_RUNTIME_PROFILE", "local")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        server.end_leftover_engines(r"D:\nonexistent-engine.exe")
    assert buf.getvalue() == "", "a host-spawned child must not sweep"
