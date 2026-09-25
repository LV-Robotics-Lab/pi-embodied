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
# Modified by pi-embodied: import paths rewritten; business calls now run strictly
# one at a time behind a process-wide lock (RPent let read-only calls overlap);
# added the lock-free ``stop``/``cancel`` method and the stop generation that
# long operations poll; ``healthz`` reports version and service name; the
# pickle socket transport is removed (HTTP only).

"""Base class for subprocess RPC servers.

``RpcFacade`` owns the shutdown event, the ``healthz`` / ``stop`` /
``shutdown`` RPC methods, transport binding, parent-watch, and clean
teardown. Subclasses register business methods in ``_register_rpc`` (called
from ``__init__``).

One call at a time: every business call (and ``session.*``) runs under one
process-wide lock, so two calls never overlap even when a client drops a
connection and retries while the first call is still running. Env worker
pipes and simulators are not safe for concurrent use (a concurrent
``render_camera`` broke LIBERO's worker pipe). ``self._readonly_methods`` is
kept for compatibility but no longer grants concurrency.

Stop: ``stop`` (alias ``cancel``) bypasses the lock. It bumps a stop
generation; a call that was received before the stop and is still waiting
for the lock is rejected with :class:`CallCancelled` without running, and the
call that is executing sees :meth:`RpcFacade.stop_requested` turn true, which
long operations poll between steps. Subclasses propagate the stop to
backends with their own loops by overriding :meth:`RpcFacade._on_stop`.

Client-side counterparts live in :mod:`pi_embodied_services.utils.rpc.rpc_client`.

Usage::

    class MyFacade(RpcFacade):
        def __init__(self):
            super().__init__()
            self._rpc["hello"] = self.say_hello

        def say_hello(self):
            return "world"


    MyFacade().serve(transport="http", host="127.0.0.1", port=0)
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable, Literal

from pi_embodied_services import __version__
from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.rpc.rpc_client import RpcError

logger = get_logger("rpc")

DEFAULT_SESSION_TIMEOUT_S = 3600.0

#: Methods answered on the transport thread without taking the call lock.
LOCK_FREE_METHODS = frozenset({"healthz", "stop", "cancel", "shutdown"})


class CallCancelled(RuntimeError):
    """A call received before a ``stop`` that had not started running yet."""


def make_error_response(exc: Exception) -> dict:
    """Build the error envelope for a caught exception."""

    return {
        "ok": False,
        "error": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


class RpcFacade:
    """Base class for subprocess RPC servers.

    Subclasses register methods in ``self._rpc`` (typically in ``__init__``
    or a ``_register_rpc`` hook). Business calls run one at a time under
    ``self._call_lock``; see the module docstring for ``stop``.

    The base owns the shutdown event, the ``shutdown`` / ``healthz`` /
    ``stop`` RPC methods, transport binding, parent-watch, and clean teardown.
    """

    #: Service name reported by ``healthz``; defaults to the class name.
    SERVICE_NAME: str | None = None

    def __init__(
        self,
        *,
        enable_sessions: bool = False,
        session_timeout_s: float = DEFAULT_SESSION_TIMEOUT_S,
    ) -> None:
        self._enable_sessions = enable_sessions
        self._session_timeout_s = float(session_timeout_s)
        self._shutdown_event = threading.Event()
        self._session_lock = threading.Lock()
        self._sessions: dict[str, float] = {}
        # Re-entrant so a handler that re-enters dispatch on the same thread
        # (``session.close`` running the drop hook) does not deadlock.
        self._call_lock = threading.RLock()
        self._stop_lock = threading.Lock()
        self._stop_generation = 0
        self._active_generation: int | None = None
        self._rpc: dict[str, Callable] = {}
        self._readonly_methods: set[str] = set()

    @property
    def service_name(self) -> str:
        return self.SERVICE_NAME or type(self).__name__

    def close(self) -> None:
        """Clean up resources. Override in subclasses that hold resources."""
        pass

    def _builtin_dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        """Handle framework methods (healthz, stop/cancel, shutdown).

        Returns ``None`` for business methods so that callers fall through to
        the locked business dispatch.
        """
        if method == "healthz":
            return {
                "status": "ok",
                "version": __version__,
                "service": self.service_name,
            }
        if method in ("stop", "cancel"):
            return self.request_stop()
        if method == "shutdown":
            with self._call_lock:
                self._shutdown_event.set()
            return {"ok": True}
        return None

    # ---- stop ------------------------------------------------------------

    def request_stop(self) -> dict:
        """Stop the running call at its next step boundary and drop queued calls.

        Never waits for the call lock. Returns immediately; the running call
        returns (or raises) on its own once it observes the stop.
        """
        with self._stop_lock:
            self._stop_generation += 1
            generation = self._stop_generation
        running = self._active_generation is not None
        try:
            self._on_stop(generation)
        except Exception:
            logger.warning("stop hook failed", exc_info=True)
        return {"ok": True, "stop_generation": generation, "call_in_progress": running}

    def _on_stop(self, generation: int) -> None:
        """Hook: forward a stop to a backend that runs its own step loop."""

    def stop_requested(self) -> bool:
        """Whether ``stop`` arrived after the currently running call was received.

        Long operations poll this between steps and return early when true.
        Always false outside a call.
        """
        active = self._active_generation
        return active is not None and self._stop_generation != active

    @property
    def active_stop_generation(self) -> int | None:
        """Stop generation the running call was received under (None when idle)."""
        return self._active_generation

    # ---- dispatch ----------------------------------------------------------

    def _serve_dispatch(
        self, method: str, args: tuple, kwargs: dict, *, session_id: str | None = None
    ) -> Any:
        """Transport entry point: lock-free framework methods, locked business calls."""
        if method in LOCK_FREE_METHODS:
            return self._builtin_dispatch(method, args, kwargs)
        return self._run_call(
            method,
            args,
            kwargs,
            session_id=session_id,
            arrival_generation=self._stop_generation,
        )

    def _run_call(
        self,
        method: str,
        args: tuple,
        kwargs: dict,
        *,
        session_id: str | None,
        arrival_generation: int,
    ) -> Any:
        """Run one business call under the call lock (one call at a time)."""
        with self._call_lock:
            if self._stop_generation != arrival_generation and not method.startswith(
                "session."
            ):
                raise CallCancelled(f"{method}: cancelled by stop before it started")
            self._active_generation = arrival_generation
            try:
                if session_id is None:
                    return self._dispatch(method, args, kwargs)
                return self._dispatch(method, args, kwargs, session_id=session_id)
            finally:
                self._active_generation = None

    def _dispatch(
        self, method: str, args: tuple, kwargs: dict, *, session_id: str | None = None
    ) -> Any:
        """Business RPC dispatch using a registration dict.

        Subclasses register handlers in ``self._rpc`` (typically in
        ``_register_rpc``). Transports reach this through
        :meth:`_serve_dispatch`, which holds the call lock.

        When sessions are enabled, ``session_id`` is the caller's bound
        session; the base handles ``session.register`` and ``session.close``
        RPC methods, session validation, and idle-timeout tracking.
        """
        result = self._builtin_dispatch(method, args, kwargs)
        if result is not None:
            return result
        if self._enable_sessions:
            if session_id is None:
                raise RpcError(
                    method,
                    "this server requires a bound session: call session.register first",
                )
            if method == "session.register":
                return self.register_session(session_id)
            if method == "session.close":
                return self.drop_session(session_id)
            self._touch_session(session_id)
            return self._dispatch_session(method, args, kwargs, session_id=session_id)
        return self._dispatch_nosession(method, args, kwargs)

    def _dispatch_nosession(self, method: str, args: tuple, kwargs: dict) -> Any:
        handler = self._rpc.get(method)
        if handler is None:
            raise ValueError(f"unknown RPC method: {method!r}")
        return handler(*args, **kwargs)

    def _dispatch_session(
        self, method: str, args: tuple, kwargs: dict, *, session_id: str | None = None
    ) -> Any:
        handler = self._rpc.get(method)
        if handler is None:
            raise ValueError(f"unknown RPC method: {method!r}")
        return handler(*args, **kwargs, session_id=session_id)

    # ---- session lifecycle (RPC methods + hooks) -------------------------

    def register_session(self, session_id: str) -> dict:
        """Register a session.

        The idle timeout is the server's own ``session_timeout_s``
        (set at construction); the client does not carry one. Registering an
        already-live session refreshes its ``last_active``.

        Called automatically by :func:`wait_for_ready` on connect for
        session-aware clients; business code does not call this.
        """
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(
                f"session_id must be a non-empty string, got {session_id!r}"
            )
        with self._session_lock:
            self._sessions[session_id] = time.monotonic()
        return {"ok": True, "session_id": session_id}

    def drop_session(self, session_id: str) -> dict:
        """Drop a session and fire its cleanup hook (no expiry check).

        Called by the ``session.close`` RPC (client-side atexit); unlike
        :meth:`_expire_session`, this removes the session unconditionally.
        The session lock is released before the call lock is taken, so the
        lock order is never session -> call while a call holds call -> session.
        """
        with self._session_lock:
            existed = self._sessions.pop(session_id, None) is not None
        if existed:
            with self._call_lock:
                try:
                    self._on_session_drop(session_id)
                except Exception:
                    logger.warning(
                        "session %s close: cleanup hook failed",
                        session_id,
                        exc_info=True,
                    )
        return {"ok": True, "session_id": session_id}

    def _touch_session(self, session_id: str) -> None:
        """Refresh last_active for an active session; raise if unknown."""
        with self._session_lock:
            last = self._sessions.get(session_id)
            if last is None:
                raise RpcError("session", f"session not found: {session_id}")
            self._sessions[session_id] = time.monotonic()

    def _on_session_drop(self, session_id: str) -> None:
        """Hook: policy-state cleanup when a session is dropped.

        Fired on both drop paths: the ``session.close`` RPC (client-side
        atexit) and idle-expiry by the sweep thread. See :meth:`drop_session`
        and :meth:`_expire_session`.
        """

    def _expire_session(self, session_id: str) -> None:
        """Drop a session whose idle timeout has elapsed and run its cleanup hook."""
        with self._session_lock:
            last = self._sessions.get(session_id)
            if last is None:
                return
            if (time.monotonic() - last) <= self._session_timeout_s:
                return
            self._sessions.pop(session_id, None)
        with self._call_lock:
            self._on_session_drop(session_id)

    def _sweep_sessions(self, interval_s: float) -> None:
        """Background loop: drop idle-expired sessions every ``interval_s`` s."""
        while not self._shutdown_event.wait(interval_s):
            now = time.monotonic()
            with self._session_lock:
                expired = [
                    sid
                    for sid, last in self._sessions.items()
                    if (now - last) > self._session_timeout_s
                ]
            for sid in expired:
                try:
                    self._expire_session(sid)
                except Exception:
                    logger.warning("session sweep: drop %s failed", sid, exc_info=True)

    def _bind_and_announce(
        self,
        transport: Literal["http"],
        host: str,
        port: int,
        dispatch: Callable[..., Any],
    ):
        """Bind the HTTP server and print its listening URL.

        Returns the bound server, whose ``server_address`` reflects the
        actually-bound ``(host, port)`` (useful when ``port == 0``). The
        pickle socket transport RPent offered is removed: unpickling request
        frames is remote code execution for anyone who can reach the port.
        """
        from pi_embodied_services.utils.rpc.http_rpc import HttpRpcServer

        if transport != "http":
            raise ValueError(
                f"unsupported transport {transport!r}: only 'http' is available"
            )
        server = HttpRpcServer((host, port), dispatch)
        bound_host, bound_port = server.server_address
        client_host = "127.0.0.1" if bound_host == "0.0.0.0" else bound_host
        url = f"{transport}://{client_host}:{bound_port}"
        print(f"RPC server listening on {url}", flush=True)
        logger.info("RPC server listening on %s", url)
        return server

    def serve(
        self,
        *,
        transport: Literal["http"],
        host: str,
        port: int,
        parent_watch: bool = False,
        session_sweep_s: float | None = None,
    ) -> None:
        """Bind, announce, watch-parent, serve-forever, shut down cleanly.

        Session support (per-session state + idle-timeout validation) is
        governed by ``enable_sessions`` passed to :meth:`__init__` — servers
        that don't isolate per-client policy state leave it off.

        When *parent_watch* is True, a background thread reads stdin (a pipe
        from the spawning process) and triggers shutdown when the pipe
        closes — i.e., when the parent process dies.

        When sessions are enabled (``enable_sessions=True`` at construction),
        *session_sweep_s* MUST be a positive number — the idle timeout is
        only enforced by the sweep thread, so a non-positive value raises
        ``ValueError``. The sweep thread drops idle-expired sessions every
        that many seconds and fires :meth:`_on_session_drop`.
        """
        from pi_embodied_services.utils.daemon import watch_parent_death

        server = self._bind_and_announce(transport, host, port, self._serve_dispatch)

        if self._enable_sessions and (session_sweep_s is None or session_sweep_s <= 0):
            raise ValueError(
                "session_sweep_s is required (and > 0) when sessions "
                "are enabled; idle timeout is only enforced by the "
                f"sweep thread, got {session_sweep_s!r}"
            )
        if parent_watch:
            watch_parent_death(self._shutdown_event.set)
        if self._enable_sessions:
            threading.Thread(
                target=self._sweep_sessions,
                args=(session_sweep_s,),
                daemon=True,
                name="rpc-session-sweep",
            ).start()
        try:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self._shutdown_event.wait()
        finally:
            server.shutdown()
            server.server_close()
            self.close()


__all__ = [
    "LOCK_FREE_METHODS",
    "CallCancelled",
    "RpcFacade",
    "make_error_response",
]
