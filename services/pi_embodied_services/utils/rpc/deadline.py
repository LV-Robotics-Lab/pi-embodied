# Copyright 2026 The pi-embodied Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A per-thread deadline for the call that is running (a ``run_code`` primitive).

The code runner sets it around every primitive it executes; :class:`HttpRpcClient` clamps
each outbound request's timeout to what is left, so a primitive that waits on another
service (SAM3's ``segment``, the grasp planners) cannot outlive the run's wall clock.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator

_local = threading.local()


@contextlib.contextmanager
def call_deadline(deadline: float | None) -> Iterator[None]:
    """Bound this thread's outbound calls by ``deadline`` (``time.monotonic()``) while inside."""
    prev = getattr(_local, "deadline", None)
    _local.deadline = deadline
    try:
        yield
    finally:
        _local.deadline = prev


def remaining(timeout_s: float) -> float:
    """``timeout_s`` clamped to the current deadline (unchanged without one); 0 once it passed."""
    deadline = getattr(_local, "deadline", None)
    if deadline is None:
        return timeout_s
    return max(0.0, min(timeout_s, deadline - time.monotonic()))


__all__ = ["call_deadline", "remaining"]
