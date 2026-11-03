"""serve/test_slot_sessions.py - /slots/{n}?action=save|restore for PARALLEL (batch) slots: the engine's
SAVE SLOT / RESTORE SLOT forms, issued on the control lines; capability, busy-slot and refusal mapping.
Mock engine: no GPU, no pack.

    python -m unittest serve.test_slot_sessions -v
"""
from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (ByteTokenizer, Service, SessionRefused, StrataEngine,  # noqa: E402
                          configure_slot_overlap_env)

ROOT = Path(__file__).resolve().parents[1]


class SlotProc:
    """Stands in for the engine process: answers the SAVE / RESTORE lines, in slot form and plain, exactly
    as `strata --serve` does (answers on the stdout line queue; refused ops answer SERR)."""

    killed = False

    def __init__(self, engine):
        self.engine = engine
        self.stdin = self
        self.sent: list[str] = []

    def write(self, s):
        self.sent.append(s)
        parts = s.strip().split(" ")
        cmd = parts[0]
        if cmd == "DIE":
            self.engine.lines.put(None)
            return
        if cmd not in ("SAVE", "RESTORE"):
            self.engine.lines.put("ERR expected SAVE/RESTORE\n")
            return
        slot = -1
        path = " ".join(parts[1:])
        if parts[1:2] == ["SLOT"]:
            slot = int(parts[2])
            path = " ".join(parts[3:])
        if slot >= 0:
            if slot in self.engine.slot_fail:
                self.engine.lines.put(f"SERR invalid 0 {self.engine.slot_fail[slot]}\n")
                return
        elif self.engine.fail:
            self.engine.lines.put("SERR invalid 0 no complete session to save\n")
            return
        if cmd == "SAVE":
            Path(path).write_bytes(b"x" * 1234)
            self.engine.lines.put("SAVED 1000 1234 40.1\n")
        else:
            async_form = path.endswith(" ASYNC")
            if async_form:
                path = path[:-6]
            if not Path(path).exists():
                self.engine.lines.put("SERR invalid 0 session file: not a session file\n")
                return
            if async_form:
                self.engine.lines.put("RSTART 3 1234\n")   # Phase 6.3: the prefetch form's whole answer
                return
            self.engine.lines.put("RESTORED 1000 1234 55.2\n")

    def flush(self):
        pass

    def poll(self):
        return -9 if self.killed else None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return -9


class SlotEngine(StrataEngine):
    def __init__(self, batch=3, session_slots=True):
        super().__init__("strata", ["--batch", str(batch)] if batch else [], lazy=True)
        self.max_context = 65536
        self.unloaded, self.ended = False, False
        self.lines = queue.Queue()
        self.proc = SlotProc(self)
        self.silence_s = 300.0
        self.batch = batch
        self.info = {"batch_slots": str(batch), "session_slots": "1"} if (batch and session_slots) else {}
        self.ctl = threading.Lock()
        self.slot_busy = [False] * max(batch, 1)
        self.slot_held = [[] for _ in range(max(batch, 1))]
        self.slot_fail: dict[int, str] = {}
        self.fail = False
        self.gen = 1

    def fresh(self):
        self.ended = False
        self.slot_fail.clear()
        self.fail = False
        self.proc = SlotProc(self)
        while not self.lines.empty():
            self.lines.get_nowait()


class SlotSessions(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.engine = SlotEngine()
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.slot_save_path = self.dir.name

    def act(self, slot, action, filename="a.bin"):
        self.engine.fresh()
        return self.svc.slot_action(slot, action, filename)

    def test_batch_save_sends_the_slot_form_and_answers(self):
        code, body = self.act("1", "save")
        self.assertEqual(code, 200, body)
        self.assertEqual(body["id_slot"], 1)
        self.assertEqual(body["n_saved"], 1000)
        self.assertEqual(self.engine.proc.sent, [f"SAVE SLOT 1 {self.dir.name}{os.sep}a.bin\n"])
        self.assertEqual(Path(self.dir.name, "a.bin").read_bytes(), b"x" * 1234)

    def test_batch_restore_round_trip_uses_the_slot_form(self):
        self.assertEqual(self.act("0", "save")[0], 200)
        code, body = self.act("0", "restore")
        self.assertEqual(code, 200, body)
        self.assertEqual(body["n_restored"], 1000)
        self.assertEqual(self.engine.proc.sent[0], f"RESTORE SLOT 0 {self.dir.name}{os.sep}a.bin\n")

    def test_no_capability_is_501_and_sends_nothing(self):
        self.engine.info = {"batch_slots": "3"}          # an engine without the slot forms
        code, body = self.act("0", "save")
        self.assertEqual(code, 501, body)
        self.assertIn("session_slots", body["error"]["message"])
        self.assertEqual(self.engine.proc.sent, [])

    def test_slot_outside_the_batch_is_400(self):
        for slot in ("3", "9", "x", ""):
            code, body = self.act(slot, "save")
            self.assertEqual(code, 400, (slot, body))
        self.assertEqual(self.engine.proc.sent, [])

    def test_a_decoding_slot_is_503_before_any_command(self):
        self.engine.slot_busy[2] = True
        code, body = self.act("2", "save")
        self.assertEqual(code, 503, body)
        self.assertEqual(self.engine.proc.sent, [])

    def test_engine_refusal_maps_the_kind(self):
        self.engine.fresh()
        self.engine.slot_fail[1] = "the slot is decoding: it holds no idle conversation"   # set after fresh()
        code, body = self.svc.slot_action("1", "save", "a.bin")
        self.assertEqual(code, 400, body)
        self.assertEqual(body["error"]["kind"], "invalid")
        self.assertIn("decoding", body["error"]["message"])

    def test_restore_of_a_missing_file_is_404(self):
        code, body = self.act("2", "restore")
        self.assertEqual(code, 404, body)

    def test_bad_file_name_is_400(self):
        for name in ("sub/x.bin", "..", "NUL", ".hidden"):
            code, body = self.act("0", "save", name)
            self.assertEqual(code, 400, (name, body))

    def test_solo_engine_keeps_the_main_form_and_slot_zero(self):
        self.engine.batch, self.engine.info = 0, {}
        self.engine.slot_busy = [False]
        code, body = self.act("0", "save")
        self.assertEqual(code, 200, body)
        self.assertEqual(self.engine.proc.sent, [f"SAVE {self.dir.name}{os.sep}a.bin\n"])
        self.assertEqual(body["id_slot"], 0)
        self.engine.fresh()
        code, body = self.svc.slot_action("1", "save", "b.bin")
        self.assertEqual(code, 400, body)

    def test_action_checked(self):
        code, body = self.act("0", "erase")
        self.assertEqual(code, 400, body)


class ParallelAutoSessions(unittest.TestCase):
    """The parallel auto-sessions flow: BDONE hook -> worker save (on ctl), admission hook -> restore,
    manifest, and the inert paths (no capability, live prefix, no file)."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.engine = SlotEngine()
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.slot_save_path = self.dir.name
        self.svc.auto_sessions = True
        self.svc.auto_sessions_min_tokens = 4
        self.svc.auto_par_save_min_s = 0.0
        self.engine.fresh()

    def file(self, key):
        return Path(self.dir.name, f"auto-{key}.bin")

    def test_finished_turn_is_saved_on_the_worker_with_the_slot_form(self):
        self.svc.auto_tls.key = "conv1"
        self.engine.slot_held[1] = [10, 11, 12, 13, 14]
        self.svc._auto_par_note(1)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not self.file("conv1").exists():
            time.sleep(0.05)
        self.assertTrue(self.file("conv1").exists(), self.engine.proc.sent)
        self.assertEqual(self.engine.proc.sent, [f"SAVE SLOT 1 {self.dir.name}{os.sep}auto-conv1.bin\n"])
        self.assertEqual(self.svc.auto_par_manifest["conv1"]["tokens"], 1000)   # the engine's own saved count
        with open(Path(self.dir.name, "manifest.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["conv1"]["tokens"], 1000)

    def test_a_re_admitted_slot_is_not_saved(self):
        self.svc.auto_tls.key = "conv2"
        self.svc.auto_par_save_min_s = 0.3
        self.engine.slot_held[2] = [20, 21, 22, 23, 24]
        self.svc._auto_par_note(2)
        self.engine.slot_busy[2] = True              # the next turn took the slot while the save waited
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(self.file("conv2").exists())
        self.assertEqual(self.engine.proc.sent, [])

    def test_admission_restores_the_file_into_the_picked_slot(self):
        self.svc.auto_tls.key = "conv3"
        self.file("conv3").write_bytes(b"x" * 10)
        self.engine.slot_held[0] = []
        self.svc._auto_par_restore(0, [1, 2, 3])
        self.assertEqual(self.engine.proc.sent, [f"RESTORE SLOT 0 {self.dir.name}{os.sep}auto-conv3.bin\n"])

    def test_a_live_matching_prefix_beats_the_file(self):
        self.svc.auto_tls.key = "conv4"
        self.file("conv4").write_bytes(b"x" * 10)
        self.engine.slot_held[1] = [1, 2]
        self.svc._auto_par_restore(1, [1, 2, 3])
        self.assertEqual(self.engine.proc.sent, [])

    def test_no_file_no_command(self):
        self.svc.auto_tls.key = "conv5"
        self.engine.slot_held[1] = []
        self.svc._auto_par_restore(1, [1, 2, 3])
        self.assertEqual(self.engine.proc.sent, [])

    def test_hooks_are_inert_without_the_capability(self):
        self.engine.info = {"batch_slots": "3"}      # no session_slots
        self.svc.auto_tls.key = "conv6"
        self.engine.slot_held[1] = [1, 2, 3, 4, 5]
        self.svc._auto_par_note(1)
        self.svc._auto_par_restore(1, [1, 2, 3])
        self.assertEqual(self.engine.proc.sent, [])
        self.assertFalse(self.file("conv6").exists())

    def test_hooks_are_inert_below_the_min_tokens(self):
        self.svc.auto_tls.key = "conv7"
        self.engine.slot_held[1] = [1, 2, 3]         # 3 < min_tokens 4
        self.svc._auto_par_note(1)
        time.sleep(0.3)
        self.assertEqual(self.engine.proc.sent, [])


class SlotOverlapEnv(unittest.TestCase):
    """PHASE 6.3: the engine's STRATA_SESSION_OVERLAP is set at the engine start exactly for the config that
    uses it - auto sessions on batch slots ("auto_sessions" + "parallel": N>=2); everything else is left
    without it, including a stale one the caller inherited."""

    def test_parallel_auto_sessions_enable_it(self):
        env = {}
        configure_slot_overlap_env(env, {"auto_sessions": True, "parallel": 3})
        self.assertEqual(env.get("STRATA_SESSION_OVERLAP"), "1")

    def test_solo_auto_sessions_do_not(self):
        for cfg in ({"auto_sessions": True}, {"auto_sessions": True, "parallel": 1},
                    {"auto_sessions": True, "parallel": 0}):
            env = {}
            configure_slot_overlap_env(env, cfg)
            self.assertNotIn("STRATA_SESSION_OVERLAP", env)

    def test_parallel_without_auto_sessions_does_not(self):
        env = {"STRATA_SESSION_OVERLAP": "1"}        # inherited: the server owns the switch
        configure_slot_overlap_env(env, {"parallel": 3})
        self.assertNotIn("STRATA_SESSION_OVERLAP", env)

    def test_a_non_number_parallel_is_inert(self):
        for n in ("3", True, None, 2.0):
            env = {}
            configure_slot_overlap_env(env, {"auto_sessions": True, "parallel": n})
            self.assertNotIn("STRATA_SESSION_OVERLAP", env)

    def test_parallel_auto_sessions_arm_the_prefetch_too(self):
        env = {}
        self.assertTrue(configure_slot_overlap_env(env, {"auto_sessions": True, "parallel": 3}))
        self.assertEqual(env.get("STRATA_SESSION_PREFETCH"), "1")

    def test_the_prefetch_is_cleared_with_the_overlap(self):
        env = {"STRATA_SESSION_PREFETCH": "1"}       # inherited: the server owns the switch
        self.assertFalse(configure_slot_overlap_env(env, {"auto_sessions": True, "parallel": 1}))
        self.assertNotIn("STRATA_SESSION_PREFETCH", env)


class SlotPrefetch(unittest.TestCase):
    """Phase 6.3 commit 3: the server's prefetch form - when the engine was armed (STRATA_SESSION_PREFETCH=1
    at the start), the admission's restore sends `RESTORE SLOT <n> <path> ASYNC`, takes RSTART as the whole
    answer and moves on (the request's BGEN); an unarmed engine keeps the synchronous form."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.engine = SlotEngine()
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.slot_save_path = self.dir.name
        self.svc.auto_sessions = True
        self.engine.fresh()

    def file(self, key):
        return Path(self.dir.name, f"auto-{key}.bin")

    def test_armed_engine_sends_the_async_form_and_moves_on(self):
        self.svc.slot_prefetch = True
        self.svc.auto_tls.key = "convp"
        self.file("convp").write_bytes(b"x" * 10)
        self.engine.slot_held[0] = []
        self.svc._auto_par_restore(0, [1, 2, 3])
        self.assertEqual(self.engine.proc.sent,
                         [f"RESTORE SLOT 0 {self.dir.name}{os.sep}auto-convp.bin ASYNC\n"])
        self.assertEqual(self.engine.slot_held[0], [])

    def test_unarmed_engine_keeps_the_synchronous_form(self):
        self.svc.slot_prefetch = False
        self.svc.auto_tls.key = "convs"
        self.file("convs").write_bytes(b"x" * 10)
        self.engine.slot_held[0] = []
        self.svc._auto_par_restore(0, [1, 2, 3])
        self.assertEqual(self.engine.proc.sent,
                         [f"RESTORE SLOT 0 {self.dir.name}{os.sep}auto-convs.bin\n"])

    def test_async_refusal_maps_like_a_sync_one(self):
        self.svc.slot_prefetch = True
        self.svc.auto_tls.key = "convx"
        self.file("convx").write_bytes(b"x" * 10)
        self.engine.slot_held[0] = []
        self.engine.slot_fail[0] = "the slot's session is still loading (a prefetch restore holds it)"
        self.svc._auto_par_restore(0, [1, 2, 3])          # returns; no SessionRefused: this request
        # continues on the same load (its admission waits at the loader's per-layer gate)
        self.assertEqual(self.engine.slot_held[0], [])
        self.assertIsNone(self.svc.auto_par_manifest.get("convx", {}).get("refusals"))

    def test_a_real_async_refusal_still_raises_and_tracks(self):
        self.svc.slot_prefetch = True
        self.svc.auto_tls.key = "convy"
        self.file("convy").write_bytes(b"x" * 10)
        self.engine.slot_held[0] = []
        self.engine.slot_fail[0] = "session file: geometry mismatch"
        with self.assertRaises(SessionRefused):
            self.svc._auto_par_restore(0, [1, 2, 3])
        self.assertIn("ASYNC", self.engine.proc.sent[0])
        self.assertEqual(self.svc.auto_par_manifest["convy"]["refusals"], 1)


if __name__ == "__main__":
    unittest.main()
