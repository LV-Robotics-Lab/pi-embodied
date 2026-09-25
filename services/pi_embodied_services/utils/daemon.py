# Copyright 2026 The RPent Authors.
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
#
# Modified by pi-embodied: import paths rewritten; the parent-side ProcessDaemon
# spawner is omitted (pi-embodied spawns servers from TypeScript).

"""Subprocess RPC servers: death-watch (child side)."""

from __future__ import annotations

import sys
import threading
from typing import Callable

# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------


def watch_parent_death(on_death: Callable[[], None]) -> None:
    """Call ``on_death()`` once, from a background thread, when stdin hits EOF.

    Under :class:`ProcessDaemon` this fires exactly when the parent dies. If
    invoked from a terminal, ``read()`` blocks on user input and never fires;
    if stdin is redirected from ``/dev/null`` or already closed, it fires
    immediately.
    """

    def _watch() -> None:
        try:
            sys.stdin.buffer.read()
        except Exception:
            pass
        on_death()

    threading.Thread(target=_watch, daemon=True).start()
