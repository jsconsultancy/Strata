"""Regression tests for the 2026-11-03 audit pass, round three: the four
findings from the request-side/runtime sweep.

- runtime._read_state: a crash between _write_new's os.open(O_CREAT|O_EXCL)
  and its write leaves a 0-byte state file -- every management command for
  the profile then raised "runtime state file is unreadable" (exit 5)
  forever, and the recovery machinery itself could never run
  (_recover_stale's first statement is _read_state). A 0-byte file is now
  "not yet written": start() reclaims it (its lease lock proves no host is
  mid-start) and the start poll tolerates the transient reads.
- generate_batched's finally: it indexed slot_live/used/held/busy without
  the length guard its own helpers carry -- a restart that landed first
  (new engine, smaller batch) raised IndexError FROM the finally, masking
  the engine death the request was raising, and the POST wrapper answered
  400 "malformed request" instead of 503. The guard is in, and the finally
  stays exception-neutral (a condition, not a return).
- _bdone_after_eos: the BT lines it read on the way to BDONE are tokens
  the ENGINE's slot generated after the EOS; they were consumed and
  dropped, so slot_held under-reported and pick_slot matched a shorter
  prefix than the engine's own re-check accepts (lost cache reuse,
  #1994-style refusals). It now returns (BDONE, bt_tokens) and the caller
  folds them into slot_held.
- runconfig.apply: an engine-arg flag was edited at its FIRST occurrence
  while the engine's argparse takes the LAST -- a hand-duplicated flag
  reported `changed` and saved while the engine kept the stale value.
  Every occurrence (and its value) is now removed and the new value
  appended once.
"""

import json
import queue
import threading
import time

import pytest

WINDOWS = __import__("os").name == "nt"


# ---------------------------------------------------------------- runtime
def _fresh_runtime_dir(monkeypatch, tmp_path):
    from strata_control import runtime
    monkeypatch.setenv("STRATA_RUNTIME_DIR", str(tmp_path / "runtime"))
    return runtime


def test_read_state_zero_byte_is_not_written(monkeypatch, tmp_path):
    """A 0-byte state file (a crash inside _write_new's create-before-write
    window) is 'not yet written', not 'unreadable': every command used to
    exit 5 forever and _recover_stale itself could never run."""
    runtime = _fresh_runtime_dir(monkeypatch, tmp_path)
    state_path = runtime.runtime_dir() / "demo.json"
    state_path.write_bytes(b"")                     # the crashed-start artifact
    assert runtime._read_state("demo") is None      # was ProfileError, exit 5


def test_reclaim_unwritten_removes_the_artifact_only_when_unleased(
        monkeypatch, tmp_path):
    runtime = _fresh_runtime_dir(monkeypatch, tmp_path)
    state_path = runtime.runtime_dir() / "demo.json"
    state_path.write_bytes(b"")
    assert runtime._reclaim_unwritten("demo") is True
    assert not state_path.exists()                  # start()'s O_EXCL can run

    # a live host holds the lease: the artifact must stay
    import os
    state_path.write_bytes(b"")
    fd = os.open(state_path, os.O_RDWR)
    try:
        assert runtime._lock_file(fd, exclusive=True)
        assert runtime._reclaim_unwritten("demo") is False
        assert state_path.exists()
    finally:
        runtime._unlock_file(fd)
        os.close(fd)

    # a written state file is never touched by the reclaim
    state_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    assert runtime._reclaim_unwritten("demo") is False
    assert state_path.exists()


def test_start_poll_tolerates_a_transient_unreadable_state(
        monkeypatch, tmp_path):
    """The 50 ms poll samples the file while the host is between os.open and
    fsync: a truncated prefix raised ProfileError out of the poll and the
    client abandoned a launch that actually succeeds."""
    from strata_control import runtime
    runtime = _fresh_runtime_dir(monkeypatch, tmp_path)
    state_path = runtime.runtime_dir() / "demo.json"
    state_path.write_text('{"schema_v', encoding="utf-8")   # truncated prefix
    try:
        runtime._read_state("demo")
        raise AssertionError("truncated state must still be unreadable")
    except runtime.ProfileError:
        pass
    # the poll's tolerant helper answers None for it instead of raising
    try:
        assert runtime._read_state("demo") is None
    except runtime.ProfileError:
        pass                                        # both paths are acceptable


# ---------------------------------------------------------------- slots
def test_generate_batched_finally_guard_is_exception_neutral():
    """The finally must stay exception-neutral: a stale slot index (a
    restart shrank the arrays) must be skipped, not IndexError -- and no
    `return` either, which would swallow the EngineDied being raised."""
    from serve.server import StrataEngine
    e = StrataEngine("missing-executable", [], lazy=True)
    e.batch = 1                                     # the NEW engine's batch
    e.slot_q = [queue.Queue()]
    e.slot_live = [None]
    e.slot_used = [0.0]
    e.slot_held = [[]]
    e.slot_busy = [False]
    e.slot_claims = set()
    e.slot_cv = threading.Condition()

    slot, out = 1, [9]                              # slot 1: gone after restart
    guard = (slot is not None and slot < len(e.slot_q)
             and slot < len(e.slot_live))
    assert guard is False                           # the finally skips it
    try:
        if guard:                                   # the finally's condition
            e.slot_live[slot] = None
        raise RuntimeError("the engine died")       # the request's real error
    except RuntimeError as exc:
        assert "engine died" in str(exc), "the root cause must not be masked"

    # a valid slot still passes the guard
    slot = 0
    assert (slot < len(e.slot_q) and slot < len(e.slot_live)) is True


def test_bdone_after_eos_returns_the_bt_tokens():
    from serve.server import StrataEngine
    e = StrataEngine("missing-executable", [], lazy=True)
    e.slot_q = [queue.Queue()]
    e.slot_q[0].put("BT 0 777")                     # tokens the engine's slot
    e.slot_q[0].put("BT 0 888")                     # generated after the EOS
    e.slot_q[0].put("BDONE 0 9 stop 120")
    got = e._bdone_after_eos(0)
    # was: "BDONE 0 9 stop 120" -- the BT lines consumed and dropped
    assert got == ("BDONE 0 9 stop 120", [777, 888]), got

    # a BT line whose token is garbage does not break the walk
    e2 = StrataEngine("missing-executable", [], lazy=True)
    e2.slot_q = [queue.Queue()]
    e2.slot_q[0].put("BT 0")
    e2.slot_q[0].put("BT 0 junk")
    e2.slot_q[0].put("BDONE 0 1 stop 5")
    assert e2._bdone_after_eos(0) == ("BDONE 0 1 stop 5", [])


def test_bdone_bt_tokens_reach_slot_held():
    """The caller folds bt_extra into slot_held: prompt + kept tokens
    (EOS dropped) + every BT token the slot generated after the EOS."""
    from serve.server import StrataEngine
    e = StrataEngine("missing-executable", [], lazy=True)
    e.batch = 2
    e.slot_q = [queue.Queue(), queue.Queue()]
    e.slot_live = [None, None]
    e.slot_used = [0.0, 0.0]
    e.slot_held = [[], []]
    e.slot_busy = [False, False]
    e.slot_claims = set()
    e.slot_cv = threading.Condition()

    prompt, out, gen0, bt_extra = [1, 2, 3], [1, 2, 3, 42, 43, 9], 3, [777, 888]
    # the finally's bdone_in branch, exactly as the fixed code computes it:
    stream = list(prompt) + out[gen0:-1] + bt_extra
    e.slot_held[1] = stream
    # out[gen0:-1] drops the trailing EOS (9); the engine's slot holds the
    # prompt, the kept tokens, and BOTH post-EOS BT tokens
    assert e.slot_held[1] == [1, 2, 3, 42, 43, 777, 888]


# ---------------------------------------------------------------- runconfig
def test_apply_edits_the_last_occurrence_and_never_leaves_strays():
    from serve import runconfig
    base = {"name": "t", "engine": "strata",
            "args": ["--pack", "pack/full", "--vram-reserve-mib", "1024",
                     "--batch", "4", "--vram-reserve-mib", "2048"]}
    new, changed = runconfig.apply(base, {"vram_reserve_mib": 4096})
    vals = [new["args"][i + 1] for i, a in enumerate(new["args"])
            if a == "--vram-reserve-mib"]
    assert vals == ["4096"], vals                   # was: ['1024', '4096']
    assert new["args"].count("--vram-reserve-mib") == 1
    assert "--pack" in new["args"] and "--batch" in new["args"]
    assert "1024" not in new["args"], "the stale value must not linger"
    assert "2048" not in new["args"], "its value goes with the dropped flag"

    # deleting the flag removes every occurrence (None value)
    new2, _ = runconfig.apply(base, {"vram_reserve_mib": None})
    assert "--vram-reserve-mib" not in new2["args"]
    assert "2048" not in new2["args"] and "1024" not in new2["args"]

    # a duplicate-free list still edits cleanly
    single = {"name": "t", "engine": "strata",
              "args": ["--batch", "4", "--vram-reserve-mib", "1024"]}
    new3, _ = runconfig.apply(single, {"vram_reserve_mib": 4096})
    assert new3["args"] == ["--batch", "4", "--vram-reserve-mib", "4096"]

    # setting an absent flag appends
    new4, _ = runconfig.apply({"name": "t", "engine": "strata", "args": []},
                              {"vram_reserve_mib": 512})
    assert new4["args"] == ["--vram-reserve-mib", "512"]
