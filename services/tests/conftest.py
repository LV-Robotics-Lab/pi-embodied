"""Suite-wide guards."""

from __future__ import annotations

import time

import pytest

#: The clock functions tests stub. A replacement left on the global ``time`` module outlives its
#: test: a helper that set ``time.sleep = lambda s: None`` once made every later test in the
#: process sleep for zero seconds, and the stop-timing tests failed in full-suite runs only.
_REAL_TIME = {name: getattr(time, name) for name in ("sleep", "monotonic", "time")}


@pytest.fixture(autouse=True)
def _no_leaked_time_patches():
    """Fail the test that leaves a ``time`` function replaced, and restore it for the rest."""
    yield
    leaked = [n for n, f in _REAL_TIME.items() if getattr(time, n) is not f]
    for name in leaked:
        setattr(time, name, _REAL_TIME[name])
    assert not leaked, f"this test left time.{', time.'.join(leaked)} replaced"
