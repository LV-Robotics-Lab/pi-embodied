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

"""A main-thread server's error reply carries the call's own message, not its traceback."""

from __future__ import annotations

import threading
import time

from pi_embodied_services.utils.rpc.main_thread_serve import MainThreadServeMixin
from pi_embodied_services.utils.rpc.rpc_facade import RpcFacade, make_error_response


class _Server(MainThreadServeMixin, RpcFacade):
    def __init__(self) -> None:
        super().__init__()
        self._rpc["env.move_delta"] = self.move_delta

    def move_delta(self, delta):
        raise ValueError(
            "delta moves 0.300 m; the limit is 0.2 m per call. Split the motion."
        )


def test_a_refusal_keeps_its_type_and_message():
    f = _Server()
    got: dict = {}

    def client():
        while getattr(f, "_main_thread_queue", None) is None:
            time.sleep(0.01)
        try:
            f._dispatch_main_thread("env.move_delta", ([0.3, 0, 0],), {})
        except Exception as exc:
            got["exc"] = exc
        finally:
            f._dispatch_main_thread("shutdown", (), {})

    worker = threading.Thread(target=client, daemon=True)
    worker.start()
    f.serve(transport="http", host="127.0.0.1", port=0)
    worker.join(10)
    exc = got["exc"]
    assert isinstance(exc, ValueError)
    reply = make_error_response(exc)
    assert reply["error"] == (
        "delta moves 0.300 m; the limit is 0.2 m per call. Split the motion."
    )
    assert "Traceback" not in reply["error"] and "Traceback" in reply["traceback"]
