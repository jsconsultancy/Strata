"""Regression tests for the 2026-11-03 audit pass, round two: the nine
findings the first round traced and reported. Each test pins a fix:

- procs.end_one: the Windows settle predicate waited on is_same, whose
  start_token reads None the instant TerminateProcess sets the exit code --
  the #4051 wait never waited. It now waits for the process OBJECT (its
  teardown) via winproc.process_object_matches; a refused signal (another
  account's leftover) answers 'stuck' at once instead of stalling every
  start by ~128 s.
- lifeline.popen: a stop signal landing between the child's creation and its
  bookkeeping used to leave it with no job/watcher membership, no
  end_children entry and no procs record; registration is now exception-safe
  (the anchor's completed child is recovered too), and every downgrade
  (contain/watcher/prctl refusal) says so instead of silently going off.
- replicas: heal()'s claim was a check-then-set outside any lock and
  restart() never claimed at all -- two concurrent StrataEngine.restart calls
  on one replica raced its proc=None swap and orphaned a loaded engine.
  One atomic claim now guards both.
- mcp: a call-triggered restart holds the server lock for up to
  start_timeout_s; close() waits at most 5 s per server and skips a wedged
  one (its process is ended by the stop path either way).
- engine_component: a bad STRATA_ENGINE_BACKEND raised ComponentProblem (a
  bare ValueError) through paths whose contract is the typed EngineMissing
  family (exit 5); status() also probed the GPU stacks twice per call.
"""

import sys
import threading
import time

import pytest

WINDOWS = sys.platform.startswith("win")


# ---------------------------------------------------------------- procs
def test_winproc_process_object_matches_identity_and_absence():
    if not WINDOWS:
        pytest.skip("Windows only")
    import subprocess
    from strata_control import procs, winproc
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        token = procs.start_token(p.pid)
        assert token and token.startswith("w:")
        ticks = int(token[2:])
        assert winproc.process_object_matches(p.pid, ticks) is True
        assert winproc.process_object_matches(p.pid, ticks + 1) is False   # identity
        assert winproc.process_object_matches(999999, 1) is False          # no such object
    finally:
        p.kill()
        p.wait()


def test_end_one_eperm_answers_stuck_at_once(monkeypatch, tmp_path):
    from strata_control import procs
    monkeypatch.setattr(procs, "WINDOWS", False)     # the POSIX branch is what this pins
    record = {"pid": 424242, "token": "l:12345", "cmd": ["/bin/engine", "--serve"],
              "role": "engine", "profile": None, "written_at": 0.0,
              "parent": {"pid": 1, "token": "l:1"}}
    monkeypatch.setattr(procs, "is_same", lambda pid, token, cmd: True)
    monkeypatch.setattr(procs, "command_line", lambda pid: ["/bin/engine", "--serve"])
    monkeypatch.setattr(procs, "_signal_verified", lambda rec, sig: False)   # PermissionError inside
    monkeypatch.setattr(procs, "procs_dir", lambda create=True: tmp_path)
    said = []
    t0 = time.monotonic()
    out = procs.end_one(record, grace=8.0, settle=120.0, say=said.append)
    dt = time.monotonic() - t0
    assert out == "stuck"
    assert dt < 2.0                       # was ~128 s of grace + settle, every start
    assert any("another account" in s for s in said)
    # and the record is KEPT (the process truthfully still runs)


def test_end_one_signal_race_gone(monkeypatch):
    from strata_control import procs
    monkeypatch.setattr(procs, "WINDOWS", False)     # the POSIX branch is what this pins
    record = {"pid": 424242, "token": "l:12345", "cmd": ["/bin/engine"],
              "role": "engine", "profile": None, "written_at": 0.0,
              "parent": {"pid": 1, "token": "l:1"}}
    monkeypatch.setattr(procs, "is_same", lambda pid, token, cmd: True)
    monkeypatch.setattr(procs, "command_line", lambda pid: None)             # it vanished
    monkeypatch.setattr(procs, "_signal_verified", lambda rec, sig: False)
    dropped = []
    monkeypatch.setattr(procs, "drop_record", lambda pid: dropped.append(pid))
    assert procs.end_one(record, grace=1.0, settle=1.0) == "gone"
    assert dropped == [424242]


def test_end_one_windows_settle_waits_on_the_process_object(monkeypatch):
    """The #4051 predicate on Windows: not is_same (which reads None the moment
    TerminateProcess sets the exit code) but the process OBJECT's existence."""
    from types import SimpleNamespace
    from strata_control import procs, winproc
    monkeypatch.setattr(procs, "WINDOWS", True)
    record = {"pid": 424242, "token": "w:12345", "cmd": ["C:\\eng\\engine.exe"],
              "role": "engine", "profile": None, "written_at": 0.0,
              "parent": {"pid": 1, "token": "w:1"}}
    monkeypatch.setattr(procs, "is_same", lambda pid, token, cmd: True)   # would keep 'stuck' forever if consulted
    runs = []
    monkeypatch.setattr(procs.subprocess, "run",
                        lambda cmd, **kw: runs.append(cmd) or SimpleNamespace(returncode=0, stdout=b"", stderr=b""))
    probed = []

    def fake_matches(pid, created):
        probed.append((pid, created))
        return False                       # the object's teardown finished
    monkeypatch.setattr(winproc, "process_object_matches", fake_matches)
    dropped = []
    monkeypatch.setattr(procs, "drop_record", lambda pid: dropped.append(pid))
    said = []
    out = procs.end_one(record, grace=1.0, settle=2.0, say=said.append)
    assert out == "ended"                  # not 'stuck': the object, not is_same, decided
    assert runs and runs[0][:3] == ["taskkill", "/F", "/PID"]
    assert probed and probed[0] == (424242, 12345)
    assert dropped == [424242]


# ---------------------------------------------------------------- lifeline
def test_popen_registers_the_child_when_the_start_is_interrupted(monkeypatch):
    from serve import lifeline

    class FakeProc:
        pid = 31337

        def poll(self):
            return None

    made = FakeProc()
    real_popen = lifeline.subprocess.Popen

    def boom_after_child(args, **kw):
        # the child EXISTS; then the caller's stop signal lands before registration
        monkeypatch.setattr(lifeline.subprocess, "Popen", real_popen)
        raise KeyboardInterrupt
    monkeypatch.setattr(lifeline.subprocess, "Popen", boom_after_child)
    registered = []
    monkeypatch.setattr(lifeline, "mechanism", lambda: "job")
    monkeypatch.setattr(lifeline.winjob, "contain", lambda child: registered.append(child) or True)
    with pytest.raises(KeyboardInterrupt):
        lifeline.popen(["x"], role="engine")
    assert registered == []           # the child was never returned: nothing to register
    # the direct path's guarantee is _register's per-net best effort; simulate a
    # contain failure mid-registration -- the record+reaper still happen
    monkeypatch.setattr(lifeline.subprocess, "Popen", lambda args, **kw: made)
    monkeypatch.setattr(lifeline.winjob, "contain", lambda child: (_ for _ in ()).throw(OSError("job gone")))
    written = []
    monkeypatch.setattr(lifeline.procs, "write_record",
                        lambda pid, role, profile=None: written.append((pid, role)) or None)
    monkeypatch.setattr(lifeline.threading, "Thread", lambda **kw: type("T", (), {"start": lambda self: None})())
    child = lifeline.popen(["x"], role="engine")
    assert child is made
    assert written == [(31337, "engine")]     # the record (the last net) still landed


def test_popen_recovers_the_anchor_child_after_an_interrupt(monkeypatch):
    from serve import lifeline
    import concurrent.futures

    class FakeProc:
        pid = 31338

    made = FakeProc()

    class RacyFuture(concurrent.futures.Future):
        """The first unbounded wait is the one a stop signal interrupts: it raises
        WITHOUT resolving the future -- the anchor thread is still mid-Popen and
        will set the result later.  The recovery wait (timeout=5) sees it."""
        def __init__(self):
            super().__init__()
            self._first_wait = True

        def result(self, timeout=None):
            if timeout is None and self._first_wait:
                self._first_wait = False
                raise KeyboardInterrupt
            return super().result(timeout)

    class FakeJobs:
        def put(self, item):                   # the anchor protocol: run fn, resolve
            fn, fut = item

            def run():
                try:
                    fut.set_result(fn())
                except BaseException as exc:  # noqa: BLE001
                    fut.set_exception(exc)
            threading.Timer(0.15, run).start()  # lands AFTER the caller's wait was interrupted

    class FakeAnchor:
        _jobs = FakeJobs()

    monkeypatch.setattr(lifeline, "_the_anchor", lambda: FakeAnchor())
    monkeypatch.setattr(lifeline, "mechanism", lambda: "pdeathsig")
    monkeypatch.setattr(lifeline.concurrent.futures, "Future", RacyFuture)
    monkeypatch.setattr(lifeline.subprocess, "Popen", lambda args, **kw: made)
    written = []
    monkeypatch.setattr(lifeline.procs, "write_record",
                        lambda pid, role, profile=None: written.append((pid, role)) or None)

    class _ThreadStub:
        def __init__(self, **kw):
            pass

        def start(self):
            pass

    class _ThreadingShim:                     # patch lifeline's view only: the real module's
        Thread = _ThreadStub                  # Timer needs its real Thread (shim, not a global patch)

        def __getattr__(self, name):
            return getattr(threading, name)

    monkeypatch.setattr(lifeline, "threading", _ThreadingShim())
    with pytest.raises(KeyboardInterrupt):
        lifeline.popen(["x"], role="engine")
    assert written == [(31338, "engine")]     # the recovered child still got its record


def test_popen_anchor_failure_propagates_unregistered(monkeypatch):
    from serve import lifeline

    class FakeJobs:
        def put(self, item):
            _, fut = item
            threading.Timer(0.05, lambda: fut.set_exception(OSError("Popen failed on the anchor"))).start()

    class FakeAnchor:
        _jobs = FakeJobs()

    monkeypatch.setattr(lifeline, "_the_anchor", lambda: FakeAnchor())
    monkeypatch.setattr(lifeline, "mechanism", lambda: "pdeathsig")
    monkeypatch.setattr(lifeline.subprocess, "Popen",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Popen must run on the anchor")))
    written = []
    monkeypatch.setattr(lifeline.procs, "write_record",
                        lambda pid, role, profile=None: written.append(pid) or None)
    with pytest.raises(OSError):              # the anchor's failure, not a hang, not a silent None
        lifeline.popen(["x"], role="engine")
    assert written == []


def test_watch_and_contain_downgrades_are_said(capsys, monkeypatch):
    from serve import lifeline

    class FakeProc:
        pid = 55

    monkeypatch.setattr(lifeline.subprocess, "Popen",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("EMFILE")))
    lifeline._watch(FakeProc())                       # was a silent return
    assert "lifeline watcher" in capsys.readouterr().out

    from serve import lifeline as ll
    monkeypatch.setattr(ll, "mechanism", lambda: "job")
    monkeypatch.setattr(ll.winjob, "contain", lambda child: False)   # the job refused
    written = []
    monkeypatch.setattr(ll.procs, "write_record", lambda pid, role, profile=None: written.append(pid) or None)
    monkeypatch.setattr(ll.threading, "Thread", lambda **kw: type("T", (), {"start": lambda self: None})())
    monkeypatch.setattr(ll.subprocess, "Popen", lambda *a, **kw: FakeProc())
    ll.popen(["x"], role="engine")
    assert "job object" in capsys.readouterr().out    # the downgrade is observable


# ---------------------------------------------------------------- replicas
def test_replica_claim_is_atomic_and_restart_honours_it():
    from serve import replicas as rp

    class FakeEngine:
        def __init__(self, alive):
            self._alive = alive
            self.restarts = 0
            self.batch = 0

        def alive(self):
            return self._alive

        def restart(self, tries=3):
            self.restarts += 1
            self._alive = True

    a, b = FakeEngine(False), FakeEngine(False)
    re_ = rp.ReplicaEngine.__new__(rp.ReplicaEngine)
    rp.ReplicaEngine.__init__(re_, [a, b])

    # heal() claims; a concurrent restart() must NOT claim the same replica
    assert re_._claim(0, cooldown=True) is True
    assert re_._claim(0, cooldown=False) is False     # already healing
    re_._release(0)

    # two explicit restarts of the same dead replica: exactly one wins
    assert re_._claim(1, cooldown=False) is True
    assert re_._claim(1, cooldown=False) is False
    re_._release(1)

    # restart() claims before spawning threads: one engine, one restart
    re_.restart()
    assert (a.restarts, b.restarts) == (1, 1)
    assert re_.healing == {0: False, 1: False}        # the finally released every claim


def test_replica_restart_does_not_reload_a_healing_unloaded_replica():
    from serve import replicas as rp

    class FakeEngine:
        def __init__(self):
            self.unloaded = True
            self.batch = 0
            self.restarts = 0

        def alive(self):
            return False

        def restart(self, tries=3):
            self.restarts += 1

    e, f = FakeEngine(), FakeEngine()     # ReplicaEngine takes two or more
    re_ = rp.ReplicaEngine.__new__(rp.ReplicaEngine)
    rp.ReplicaEngine.__init__(re_, [e, f])
    re_.heal()
    time.sleep(0.05)
    assert e.restarts == 0 and f.restarts == 0   # unloaded on purpose: heal leaves it alone
    assert re_.healing == {}                     # nothing was claimed, so nothing lingers


# ---------------------------------------------------------------- mcp
def test_mcp_close_skips_a_server_wedged_in_a_restart(monkeypatch):
    from serve import mcp

    class FakeTransport:
        closed = False

        def close(self):
            self.closed = True

    s = mcp.McpServer.__new__(mcp.McpServer)
    s.name = "wedged"
    s.lock = threading.Lock()
    s.transport = None
    s.status = "starting"

    acquired = threading.Event()

    def hold():
        with s.lock:
            acquired.set()
            time.sleep(30)                # a restart in flight
    t = threading.Thread(target=hold, daemon=True)
    t.start()
    acquired.wait(2)
    t0 = time.monotonic()
    s.close()                             # was: blocked for the full 30 s
    dt = time.monotonic() - t0
    assert dt < 10, f"close waited {dt:.1f} s on a wedged server"


# ---------------------------------------------------------------- engine_component
def test_bad_backend_env_raises_the_typed_family(monkeypatch, tmp_path):
    from strata_control import engine_component as ec
    monkeypatch.setenv(ec.ENV_BACKEND, "nvidia")           # a plausible-but-wrong guess
    monkeypatch.setattr(ec, "components_dir", lambda: tmp_path)
    try:
        ec.resolve(tmp_path)
        raise AssertionError("no error")
    except ec.EngineMissing:
        pass                                               # was: raw ComponentProblem (ValueError)
    except ec.ComponentProblem as e:
        raise AssertionError(f"untyped escape: {e!r}")


def test_status_answers_and_probes_once(monkeypatch, tmp_path):
    from strata_control import engine_component as ec
    calls = []
    real = ec.detect_backend

    def counted(*a, **kw):
        calls.append(1)
        return real(*a, **kw)
    monkeypatch.setattr(ec, "detect_backend", counted)
    monkeypatch.setattr(ec, "components_dir", lambda: tmp_path)
    monkeypatch.setattr(ec, "cached_notice", lambda base: None)
    r = ec.status(tmp_path)
    assert r["usable"] is False and isinstance(calls, list)
    assert len(calls) == 1, "status ran the GPU probes twice"


# ------------------------------------------------- server.py, the fresh pass (S1-S5)
def _bare_service():
    """A Service with only what the auto-sessions code touches: no tokenizer, no engine process."""
    import queue
    from serve import server as srv
    svc = srv.Service.__new__(srv.Service)
    eng = type("Engine", (), {})()
    eng.ctl = threading.Lock()                    # the control lines: one admission or save at a time
    eng.gen = 1
    eng.alive = lambda: True
    eng.unloaded = False
    eng.slot_busy = [False]
    eng.slot_held = [[1, 2, 3]]
    eng.info = {"session_slots": 1}
    eng.batch = 1
    svc.engine = eng
    return svc, eng


def test_auto_par_worker_releases_the_control_lines_when_a_save_fails_anyway():
    """The worker held the engine's ctl lock across the save with no finally: any exception past its catch
    list leaked the lock -- every later admission waited on the lines forever, and the dead worker meant
    every later auto save silently never ran."""
    import queue
    svc, eng = _bare_service()
    svc._auto_par_q = queue.Queue()
    svc.auto_par_save_min_s = 0

    failed = threading.Event()

    def boom(slot, key):
        failed.set()
        raise RuntimeError("the disk caught fire")
    svc._auto_par_save_locked = boom

    t = threading.Thread(target=svc._auto_par_worker_loop, daemon=True)
    t.start()
    svc._auto_par_q.put((1, 0, "k", [1, 2, 3]))
    assert failed.wait(5)
    got = False
    for _ in range(50):                           # the finally must give the lines back, promptly
        if eng.ctl.acquire(timeout=0.1):
            got = True
            break
    assert got, "the worker leaked the engine's control lines when a save raised past its catches"
    eng.ctl.release()
    assert t.is_alive(), "the worker died: every later auto save would silently never run"

    saved = threading.Event()
    svc._auto_par_save_locked = lambda slot, key: saved.set()
    svc._auto_par_q.put((1, 0, "k2", [1, 2, 3]))
    assert saved.wait(5), "the worker stopped processing saves after the failure"


def test_shutdown_drain_releases_the_control_lines(tmp_path):
    """The stop path's best-effort drain of queued slot saves held ctl across the save: a failing save broke
    out with the lock still held, into a shutdown that then tears the engine down holding it."""
    import queue
    svc, eng = _bare_service()
    svc.auto_sessions = True
    svc.slot_save_path = str(tmp_path)
    svc.auto_conv = None
    svc._auto_par_q = queue.Queue()
    svc._auto_par_worker = object()               # a worker exists: the drain runs
    svc._auto_par_capable = lambda: True

    def boom(slot, key):
        raise RuntimeError("no")
    svc._auto_par_save_locked = boom
    svc._auto_par_q.put((1, 0, "k", [1, 2, 3]))

    svc.auto_session_shutdown_save()              # was: returned with eng.ctl held on the break path
    assert svc._auto_par_q.empty(), "the drain did not take the queued save"
    assert eng.ctl.acquire(timeout=2), "the stop path leaked the engine's control lines"
    eng.ctl.release()


def test_auto_par_restore_refused_file_does_not_brick_the_conversation(tmp_path):
    """A refused RESTORE used to be raised on every turn: one bad file bricked the conversation until it
    was deleted by hand.  The first refusal is still raised (the client sees the engine's reason, and a
    transient one -- a prefetch still holding the slot -- passes once done); when the same file refuses
    again a minute apart it never will restore: it is removed and the request reads cold."""
    from serve.server import SessionRefused
    svc, eng = _bare_service()
    svc.slot_save_path = str(tmp_path)
    svc.auto_tls = threading.local()
    svc.auto_tls.key = "deadbeef"
    svc._auto_par_capable = lambda: True
    svc.auto_par_manifest = {}
    path = tmp_path / "auto-deadbeef.bin"
    calls = []

    def refused(cmd, p, slot=None, **kw):
        calls.append((cmd, slot))
        raise SessionRefused("io", "the file is corrupt")
    eng.session_file = refused

    path.write_bytes(b"x")
    eng.slot_held = [[9, 9, 9]]
    with pytest.raises(SessionRefused):           # the first refusal stays visible to the client
        svc._auto_par_restore(0, [1, 2, 3])
    assert calls == [("restore", 0)]
    assert path.exists(), "one refusal must not delete the file (the refusal may be transient)"
    assert eng.slot_held[0] == [], "the slot's stale held list is dropped after a refused restore"
    assert svc.auto_par_manifest["deadbeef"]["refusals"] == 1

    # a refusal a minute later: the file is never going to restore -- remove it, proceed cold
    svc.auto_par_manifest["deadbeef"]["refused_at"] = time.time() - 61
    eng.slot_held = [[9, 9, 9]]
    svc._auto_par_restore(0, [1, 2, 3])           # no raise: the request continues
    assert calls == [("restore", 0), ("restore", 0)]
    assert not path.exists(), "a file that refuses twice a minute apart is removed"
    assert eng.slot_held[0] == []
    assert "refusals" not in svc.auto_par_manifest["deadbeef"]

    # and a restore that goes through clears the refusal bookkeeping
    path.write_bytes(b"x")
    svc.auto_par_manifest["deadbeef"]["refusals"] = 1
    svc.auto_par_manifest["deadbeef"]["refused_at"] = time.time()
    eng.session_file = lambda cmd, p, slot=None, **kw: {"tokens": 5, "bytes": 10, "ms": 1}
    svc._auto_par_restore(0, [1, 2, 3])
    assert "refusals" not in svc.auto_par_manifest["deadbeef"]


def test_auto_par_restore_refusals_seconds_apart_never_delete(tmp_path):
    """A prefetch holds its slot for a while: refusals seconds apart (the fake's kind for it is
    "invalid") must never delete a good file, and the mapped raise the SlotPrefetch tests pin stays."""
    from serve.server import SessionRefused
    svc, eng = _bare_service()
    svc.slot_save_path = str(tmp_path)
    svc.auto_tls = threading.local()
    svc.auto_tls.key = "loading"
    svc._auto_par_capable = lambda: True
    svc.auto_par_manifest = {}
    eng.session_file = lambda cmd, p, slot=None, **kw: (_ for _ in ()).throw(
        SessionRefused("invalid", "the slot's session is still loading (a prefetch restore holds it)"))
    path = tmp_path / "auto-loading.bin"
    path.write_bytes(b"x")
    eng.slot_held = [[]]
    for _ in range(3):                            # a burst of retries inside the prefetch window
        with pytest.raises(SessionRefused):
            svc._auto_par_restore(0, [1, 2, 3])
    assert path.exists()
    assert svc.auto_par_manifest["loading"]["refusals"] == 1


def test_auto_par_restore_engine_death_still_propagates(tmp_path):
    from serve.server import EngineDied, SessionRefused
    svc, eng = _bare_service()
    svc.slot_save_path = str(tmp_path)
    svc.auto_tls = threading.local()
    svc.auto_tls.key = "deadbeef"
    svc._auto_par_capable = lambda: True

    def died(cmd, p, slot=None, **kw):
        raise EngineDied("the engine is gone")
    eng.session_file = died
    eng.slot_held = [[]]                          # not the prompt's prefix: no early "keep it" return
    (tmp_path / "auto-deadbeef.bin").write_bytes(b"x")
    with pytest.raises(EngineDied):               # the request path's restart handles this one
        svc._auto_par_restore(0, [1, 2, 3])
    assert issubclass(SessionRefused, ValueError)  # the refusal family stays catchable as a ValueError


def test_slot_action_restore_resets_the_stale_held_list(tmp_path):
    """POST /slots/1?action=restore loads conversation B into a slot whose cache lists conversation A: the
    stale slot_held made routing match A's next turn against a slot that now holds B -- the shape that
    manufactured #1994-style BGEN refusals."""
    from serve import server as srv
    svc, eng = _bare_service()
    eng.batch = 2
    eng.slot_busy = [False, False]
    eng.slot_held = [[7, 7, 7], [8, 8, 8]]
    eng.slot_used = [0.0, 0.0]
    done = []

    def session_file(action, p, slot=None, **kw):
        done.append((action, slot))
        return {"tokens": 10, "bytes": 100, "ms": 1}
    eng.session_file = session_file
    svc.slot_save_path = str(tmp_path)
    svc.status_lock = threading.Lock()
    svc.status = {"queued": 0}
    svc.last_request_at = 0.0
    svc.loaded = lambda: True
    (tmp_path / "conv.bin").write_bytes(b"x")

    code, body = srv.Service.slot_action(svc, "1", "restore", "conv.bin")
    assert code == 200, body
    assert done == [("restore", 1)]
    assert eng.slot_held[1] == [], "the restored slot's cache list is the old conversation's"
    assert eng.slot_used[1] > 0.0
    assert eng.slot_held[0] == [7, 7, 7], "the other slot is untouched"


def test_slots_view_survives_a_restart_that_shrinks_the_slots():
    """restart() replaces slot_live/slot_held with lists sized for the new batch while /metrics iterates
    range(batch): the loop could index past the new, smaller lists and 500 the scrape."""
    from serve import server as srv
    e = srv.StrataEngine.__new__(srv.StrataEngine)
    e.batch = 2
    e.slot_live = [None, {"state": "decoding", "prompt_tokens": 5, "generated": 3, "started": time.time()}]
    e.slot_held = [[1], []]
    view = srv.StrataEngine.slots_view(e)
    assert [s["slot"] for s in view] == [0, 1]
    assert view[0]["held_tokens"] == 1

    # the racing observation: batch still says 8, the lists were already swapped for 2
    e.batch = 8
    e.slot_live = [None, None]
    e.slot_held = [[], []]
    view = srv.StrataEngine.slots_view(e)          # was: IndexError -> a 500 on every scrape mid-restart
    assert [s["slot"] for s in view] == [0, 1]
