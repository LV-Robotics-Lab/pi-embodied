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

"""One-call-at-a-time dispatch, stop and healthz on the RPC facade (no simulator)."""

from __future__ import annotations

import threading
import time

import pytest

from pi_embodied_services import __version__
from pi_embodied_services.utils.rpc import RpcError, RpcFacade
from pi_embodied_services.utils.rpc.http_rpc import HttpRpcClient
from pi_embodied_services.utils.rpc.main_thread_serve import MainThreadServeMixin


class Dummy(RpcFacade):
    SERVICE_NAME = "dummy"

    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0
        self.ran: list[str] = []
        self.started = threading.Event()
        self._count_lock = threading.Lock()
        self._rpc["slow"] = self.slow
        self._rpc["read"] = self.read
        self._rpc["loop"] = self.loop
        # Formerly allowed to run concurrently; must not any more.
        self._readonly_methods.add("read")

    def _enter(self, name: str) -> None:
        with self._count_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.ran.append(name)
        self.started.set()

    def _leave(self) -> None:
        with self._count_lock:
            self.active -= 1

    def slow(self, seconds: float = 0.2) -> str:
        self._enter("slow")
        try:
            time.sleep(seconds)
            return "slow"
        finally:
            self._leave()

    def read(self) -> str:
        self._enter("read")
        try:
            time.sleep(0.05)
            return "read"
        finally:
            self._leave()

    def loop(self, steps: int = 200) -> dict:
        self._enter("loop")
        try:
            done = 0
            for _ in range(steps):
                if self.stop_requested():
                    return {"steps": done, "cancelled": True}
                time.sleep(0.01)
                done += 1
            return {"steps": done}
        finally:
            self._leave()


class MainThreadDummy(MainThreadServeMixin, Dummy):
    pass


def _serve(facade: RpcFacade):
    """Run ``facade.serve`` on a thread; return (client, stop_fn)."""
    bound: dict = {}
    original = facade._bind_and_announce

    def capture(*args, **kwargs):
        server = original(*args, **kwargs)
        bound["port"] = server.server_address[1]
        return server

    facade._bind_and_announce = capture  # type: ignore[method-assign]
    thread = threading.Thread(
        target=facade.serve,
        kwargs={"transport": "http", "host": "127.0.0.1", "port": 0},
        daemon=True,
    )
    thread.start()
    deadline = time.time() + 5
    while "port" not in bound:
        assert time.time() < deadline, "server did not bind"
        time.sleep(0.01)
    client = HttpRpcClient(f"http://127.0.0.1:{bound['port']}")

    def stop() -> None:
        client.call("shutdown", timeout_s=10)
        thread.join(timeout=10)

    return client, stop


def _call_async(client: HttpRpcClient, method: str, *args, **kwargs):
    box: dict = {}

    def run() -> None:
        try:
            box["result"] = client.call(method, args, kwargs, timeout_s=30)
        except Exception as exc:  # collected for assertions
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


@pytest.fixture(params=[Dummy, MainThreadDummy], ids=["threaded", "main-thread"])
def served(request):
    facade = request.param()
    client, stop = _serve(facade)
    yield facade, client
    stop()


def test_calls_never_overlap(served):
    facade, client = served
    threads = [_call_async(client, "slow", 0.1) for _ in range(3)]
    threads += [_call_async(client, "read") for _ in range(3)]
    for thread, box in threads:
        thread.join(timeout=10)
        assert "error" not in box, box
    assert facade.max_active == 1
    assert len(facade.ran) == 6


def test_healthz_reports_version_and_bypasses_lock(served):
    facade, client = served
    thread, _ = _call_async(client, "slow", 0.5)
    assert facade.started.wait(5)
    t0 = time.monotonic()
    health = client.call("healthz", timeout_s=5)
    assert time.monotonic() - t0 < 0.4
    assert health == {"status": "ok", "version": __version__, "service": "dummy"}
    thread.join(timeout=10)


def test_stop_interrupts_running_loop_and_drops_queued_calls(served):
    facade, client = served
    running, running_box = _call_async(client, "loop", 500)
    assert facade.started.wait(5)
    queued, queued_box = _call_async(client, "slow", 0.01)
    time.sleep(0.1)  # let the queued call reach the server and wait for the lock
    t0 = time.monotonic()
    reply = client.call("stop", timeout_s=5)
    assert time.monotonic() - t0 < 0.5
    assert reply["ok"] is True and reply["call_in_progress"] is True
    running.join(timeout=10)
    queued.join(timeout=10)
    assert running_box["result"]["cancelled"] is True
    assert running_box["result"]["steps"] < 500
    assert isinstance(queued_box.get("error"), RpcError)
    assert "cancelled by stop" in str(queued_box["error"])
    assert "slow" not in facade.ran
    # Calls received after the stop run normally.
    assert client.call("loop", (3,), timeout_s=5) == {"steps": 3}
    assert client.call("cancel", timeout_s=5)["call_in_progress"] is False


def test_socket_transport_is_gone():
    with pytest.raises(ValueError, match="only 'http'"):
        Dummy()._bind_and_announce("socket", "127.0.0.1", 0, lambda *a, **k: None)
