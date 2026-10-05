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

import os
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
        #: The peer address the transport reported for the running call (code mode's "remote caller").
        self._rpc["peer"] = lambda: self.active_peer
        #: Set once a ``slow`` call has arrived and captured its stop generation.
        self.slow_arrived = threading.Event()
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

    def _run_call(self, method, *args, **kwargs):
        # Threaded serving: the arrival generation is already captured when this runs, and
        # the call has not taken the lock yet.
        if method == "slow":
            self.slow_arrived.set()
        return super()._run_call(method, *args, **kwargs)


class MainThreadDummy(MainThreadServeMixin, Dummy):
    def _dispatch_main_thread(self, method, *args, **kwargs):
        # Main-thread serving: the call's generation is captured and it is queued before
        # this returns; signal once it is in the queue.
        if method == "slow":
            threading.Thread(target=self._signal_queued, daemon=True).start()
        return super()._dispatch_main_thread(method, *args, **kwargs)

    def _signal_queued(self) -> None:
        deadline = time.monotonic() + 300
        while self._main_thread_queue.qsize() < 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.slow_arrived.set()


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
    deadline = time.time() + 60
    while "port" not in bound:
        assert time.time() < deadline, "server did not bind"
        time.sleep(0.01)
    client = HttpRpcClient(f"http://127.0.0.1:{bound['port']}")

    def stop() -> None:
        client.call("shutdown", timeout_s=10)
        thread.join(timeout=10)

    return client, stop


#: A client wait no loaded machine exhausts: a timed-out client thread once passed for a finished
#: server call (is_alive() on the client thread measured the HTTP timeout, not the lock).
CLIENT_TIMEOUT_S = 300.0


def _call_async(client: HttpRpcClient, method: str, *args, **kwargs):
    box: dict = {}

    def run() -> None:
        try:
            box["result"] = client.call(
                method, args, kwargs, timeout_s=CLIENT_TIMEOUT_S
            )
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
    thread, box = _call_async(client, "loop", 10**9)  # ends only through the stop below
    assert facade.started.wait(CLIENT_TIMEOUT_S)
    health = client.call("healthz", timeout_s=CLIENT_TIMEOUT_S)
    # Server-side, not the client thread: the loop still holds the lock (only a stop ends it).
    assert facade.active == 1, (
        "healthz was answered while the call held the lock",
        box,
    )
    client.call("stop", timeout_s=CLIENT_TIMEOUT_S)
    # The pid names the answering process (a client that started it checks it is its own).
    assert health == {
        "status": "ok",
        "version": __version__,
        "service": "dummy",
        "pid": os.getpid(),
    }
    thread.join(timeout=CLIENT_TIMEOUT_S)


def test_stop_interrupts_running_loop_and_drops_queued_calls(served):
    facade, client = served
    running, running_box = _call_async(
        client, "loop", 10**9
    )  # ends only through the stop
    assert facade.started.wait(CLIENT_TIMEOUT_S)
    queued, queued_box = _call_async(client, "slow", 0.01)
    # The queued call has reached the server and waits for the lock (not a sleep: under load
    # a sleep let the stop overtake it, and it then ran as a call received after the stop).
    assert facade.slow_arrived.wait(CLIENT_TIMEOUT_S)
    reply = client.call("stop", timeout_s=CLIENT_TIMEOUT_S)
    # The lock bypass, judged server-side: the server answered the stop while the loop (which
    # only a stop ends) held the lock. No wall-clock bound and no client-thread liveness, both of
    # which a loaded machine breaks.
    assert reply["ok"] is True and reply["call_in_progress"] is True
    running.join(timeout=CLIENT_TIMEOUT_S)
    queued.join(timeout=CLIENT_TIMEOUT_S)
    assert running_box["result"]["cancelled"] is True, running_box
    assert isinstance(queued_box.get("error"), RpcError)
    assert "cancelled by stop" in str(queued_box["error"])
    assert "slow" not in facade.ran
    # Calls received after the stop run normally.
    assert client.call("loop", (3,), timeout_s=CLIENT_TIMEOUT_S) == {"steps": 3}
    assert (
        client.call("cancel", timeout_s=CLIENT_TIMEOUT_S)["call_in_progress"] is False
    )


def test_socket_transport_is_gone():
    with pytest.raises(ValueError, match="only 'http'"):
        Dummy()._bind_and_announce("socket", "127.0.0.1", 0, lambda *a, **k: None)


# ---- token and run_code exclusivity ---------------------------------------------------------


class Guarded(Dummy):
    """A server like LIBERO's: token required; ``busy`` stands for a running code.run."""

    REQUIRE_TOKEN = True

    def __init__(self) -> None:
        super().__init__()
        self.busy = False

    def _exclusive_call_active(self) -> bool:
        return self.busy


@pytest.mark.parametrize("cls", ["plain", "main_thread"])
def test_a_token_server_refuses_calls_without_its_token_but_answers_healthz_and_stop(
    cls, capsys
):
    facade = (
        Guarded()
        if cls == "plain"
        else type("M", (MainThreadServeMixin, Guarded), {})()
    )
    client, _ = _serve(facade)
    line = [
        ln
        for ln in capsys.readouterr().out.splitlines()
        if "RPC server listening" in ln
    ][-1]
    token = facade._rpc_token
    assert token and line.endswith(f"(token {token})")
    assert client.call("healthz")["status"] == "ok"
    assert client.call("stop")["ok"] is True
    for method in ("read", "shutdown"):
        with pytest.raises(RpcError, match="requires its RPC token"):
            client.call(method)
    wrong = HttpRpcClient(client._base_url, token="0" * 32)
    with pytest.raises(RpcError, match="requires its RPC token"):
        wrong.call("read")
    good = HttpRpcClient(client._base_url, token=token)
    assert good.call("read") == "read"
    # While a program runs, even the token holder's business calls are refused at once
    # (they would otherwise run after it, outside its budgets); stop still goes through.
    facade.busy = True
    t0 = time.monotonic()
    with pytest.raises(RpcError, match="run_code program is running"):
        good.call("read")
    assert time.monotonic() - t0 < 2
    assert good.call("stop")["ok"] is True
    facade.busy = False
    good.call("shutdown", timeout_s=10)


def test_servers_without_the_opt_in_need_no_token(capsys):
    client, stop = _serve(Dummy())
    assert "token" not in capsys.readouterr().out
    assert client.call("read") == "read"
    stop()


def test_the_running_call_sees_its_peer_address_and_nothing_outlives_the_call():
    """Audit 92245e3 CM-3: code mode decides "remote caller" from the transport's peer address,
    per call; it reaches the handler through the facade and is gone once the call returns."""
    facade = Dummy()
    client, stop = _serve(facade)
    try:
        assert client.call("peer", timeout_s=CLIENT_TIMEOUT_S) == "127.0.0.1"
        assert facade.active_peer is None
    finally:
        stop()
    # A call that came without one (no transport): unknown, not loopback.
    assert facade._serve_dispatch("peer", (), {}) is None
