"""serve/test_slot_sessions.py - /slots/{n}?action=save|restore for PARALLEL (batch) slots: the engine's
SAVE SLOT / RESTORE SLOT forms, issued on the control lines; capability, busy-slot and refusal mapping.
Mock engine: no GPU, no pack.

    python -m unittest serve.test_slot_sessions -v
"""
from __future__ import annotations

import os
import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, StrataEngine  # noqa: E402

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
            if not Path(path).exists():
                self.engine.lines.put("SERR invalid 0 session file: not a session file\n")
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
        self.slot_fail: dict[int, str] = {}
        self.fail = False

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
        self.assertEqual(body["id_slot"], "1")
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
        self.assertEqual(body["id_slot"], "0")
        self.engine.fresh()
        code, body = self.svc.slot_action("1", "save", "b.bin")
        self.assertEqual(code, 400, body)

    def test_action_checked(self):
        code, body = self.act("0", "erase")
        self.assertEqual(code, 400, body)


if __name__ == "__main__":
    unittest.main()
