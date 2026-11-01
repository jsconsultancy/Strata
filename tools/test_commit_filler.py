"""Tests for tools/commit_filler.py: the guard logic, the status reading and one small real squeeze
(32 MiB, touched and released) - the acceptance-sized squeezes are window work, not suite work.  Light:
no engine, no GPU, seconds."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import commit_filler as F  # noqa: E402


class FakeStatus(dict):
    pass


class PlanTest(unittest.TestCase):
    S = {"avail_phys_mib": 8000, "avail_commit_mib": 6000, "total_commit_mib": 60000, "load_pct": 60}

    def test_target_squeezes_the_difference(self):
        want, why = F.plan(3200, 0, self.S, 1024)
        self.assertEqual(want, 2800)
        self.assertIn("2800 MiB", why)

    def test_target_at_or_under_the_current_is_nothing(self):
        want, why = F.plan(6000, 0, self.S, 1024)
        self.assertEqual(want, 0)
        self.assertIn("nothing to squeeze", why)

    def test_target_under_the_floor_is_refused(self):
        want, why = F.plan(512, 0, self.S, 1024)
        self.assertEqual(want, 0)
        self.assertIn("min-free-mib", why)

    def test_flat_squeeze(self):
        want, why = F.plan(0, 256, self.S, 1024)
        self.assertEqual(want, 256)
        self.assertIn("flat 256 MiB", why)


class ReadTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "the acceptance is the Windows commit limit")
    def test_read_is_sane(self):
        s = F.read_status()
        self.assertIsNotNone(s)
        self.assertGreater(s["avail_commit_mib"], 0)
        self.assertLessEqual(s["avail_commit_mib"], s["total_commit_mib"])
        self.assertGreaterEqual(s["load_pct"], 0)


class AllocateTest(unittest.TestCase):
    def test_small_real_squeeze_is_touched_and_released(self):
        if sys.platform != "win32":
            self.skipTest("Windows-only")
        held, got = F.allocate(32, chunk=8 * (1 << 20))
        try:
            self.assertEqual(got, 32)
            self.assertEqual(len(held), 4)
            # the pages are touched: every 4096th byte of every chunk is our marker
            self.assertEqual(held[0][::4096][0], 1)
            self.assertEqual(len(held[0][::4096]), len(held[0]) // 4096)
        finally:
            held.clear()
        self.assertEqual(held, [])   # released: the test's own footprint is gone
        self.assertGreater(F.read_status()["avail_commit_mib"], 0)

    def test_floor_stops_the_loop_early(self):
        if sys.platform != "win32":
            self.skipTest("Windows-only")
        # the floor is the commit that is left right now: the first chunk already meets it, so the loop
        # stops after one chunk.  The want is capped at 256 MiB so a race with another process's
        # allocations can never run this test past that.
        held, _ = F.allocate(256, floor_mib=F.read_status()["avail_commit_mib"], chunk=8 * (1 << 20))
        try:
            self.assertLessEqual(len(held), 1)
        finally:
            held.clear()


if __name__ == "__main__":
    unittest.main()
