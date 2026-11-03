"""tools/ holds standalone harness scripts (AGENTS.md: `python tools/test_<name>.py`),
not pytest modules: a bare `pytest tools/` imports every test_*.py / *_test.py file,
and the harness scripts that parse sys.argv or spawn the engine at module level
(stop_resend_test, cancel_park_test, early_close_test, session_pressure_test,
batch_groups_cache_test) crash collection and abort the whole run.

Keep pytest to the real test modules here — the ones that import pytest — and
ignore everything else, so a new harness script never re-breaks `pytest tools/`.
Run the ignored files the way AGENTS.md says: `python tools/test_<name>.py`."""
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
collect_ignore = [
    p.name for p in sorted(set(_HERE.glob("test_*.py")) | set(_HERE.glob("*_test.py")))
    if not re.search(r"^\s*(import pytest|from pytest)",
                     p.read_text(encoding="utf-8", errors="replace"), re.M)
]
