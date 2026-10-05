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

"""RPC bridge from pi-embodied to an XPolicyLab policy server.

XPolicyLab (github.com/XPolicyLab/XPolicyLab, pinned ``d6332bf``) serves one policy per
process over websocket + msgpack (``client_server/ws``). pi-embodied is only the
environment client: ``packages/embodied/src/primitives/xpolicy.ts`` builds the observation and
executes the action chunk, and calls this bridge over the usual ``POST /call``
(PROTOCOL.md). The bridge holds XPolicyLab's own client,
``client_server.ws.WsModelClient``, imported from an XPolicyLab checkout
(``--xpolicylab-root`` / ``XPOLICYLAB_ROOT``), so every protocol detail stays the
reference implementation's: the HELLO handshake and its ``server_instance_id``, request
ids reused across a reconnect (the server answers a duplicate from its cache instead of
running ``update_obs`` / ``get_action`` twice), ``ServerRestartedError`` when a reconnect
lands on a new server process, the cold-start retry budget, keepalive, and msgpack with
the msgpack-numpy extension.

A ``timeout`` (the server may still be running the call) and ``ServerRestartedError``
(the fresh server lost the model state) end the trial, as XPolicyLab prescribes: every
later call fails until the next ``xpolicy.connect``.

Methods (``xpolicy.*``):

- ``connect(url, trial_id, evaluation_id, action_case_id, encode_images, request_timeout_s,
  max_connect_seconds)``: a new client for one trial (an episode); closes the previous one.
- ``action_dims(env_cfg_type)``: ``{"robot", "arm_dim", "ee_dim"}`` from ``xpolicy_env_cfg``,
  looked up like XPolicyLab's ``utils.process_data.get_robot_action_dim_info``.
- ``prepare_case(case_meta)``, ``reset()``, ``update_obs(obs)``, ``trial_end(result)``: the
  model calls, returning ``{"result", "ms"}``.
- ``get_action()``: ``{"actions": [...], "ms"}``, the chunk as a list whose items are action
  dicts (numeric values as flat float lists) or flat float vectors.
- ``close()``.

Run it in any venv with ``services[xpolicy]`` (websockets, msgpack, msgpack-numpy,
pydantic, opencv for ``encode_images``); the policy runs in its own XPolicyLab env.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.rpc import RpcFacade

logger = get_logger("xpolicy_bridge")

#: The XPolicyLab commit this bridge was written against (reported, not enforced).
XPOLICYLAB_PIN = "d6332bf"
#: ``<env_cfg_type>.yml`` names the robot; ``robot/_robot_info.json`` is XPolicyLab's own robot table (@d6332bf).
ENV_CFG = Path(__file__).with_name("xpolicy_env_cfg")


def action_dims(env_cfg_type: str, root: Path = ENV_CFG) -> dict[str, Any]:
    """XPolicyLab's ``get_robot_action_dim_info`` against pi-embodied's env_cfg directory."""
    path = root / f"{env_cfg_type}.yml"
    if not path.is_file():
        known = sorted(p.stem for p in root.glob("*.yml"))
        raise ValueError(f"unknown env_cfg_type {env_cfg_type!r}; known: {known}")
    robot = yaml.safe_load(path.read_text())["config"]["robot"]
    info = json.loads((root / "robot" / "_robot_info.json").read_text())[robot]
    return {
        "robot": robot,
        "arm_dim": list(info["arm_dim"]),
        "ee_dim": list(info["ee_dim"]),
    }


def _plain(value: Any) -> Any:
    """Numeric arrays and scalars as flat float lists / numbers; containers recursively."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "biuf":
            return value.astype(np.float64).reshape(-1).tolist()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_plain(v) for v in value]
        if items and all(isinstance(v, list) for v in items):
            flat = [x for v in items for x in v]
            if all(isinstance(x, (int, float)) for x in flat):
                return flat
        return items
    return value


def normalize_chunk(result: Any) -> list[Any]:
    """``get_action``'s result as a list of actions (RoboTwin's ``normalize_action_chunk``)."""
    if result is None:
        return []
    if isinstance(result, Mapping) and "actions" in result:
        return normalize_chunk(result["actions"])
    if isinstance(result, Mapping):
        return [_plain(result)]
    if isinstance(result, np.ndarray):
        if result.ndim == 0:
            raise ValueError("a scalar action chunk is not supported")
        rows = [result] if result.ndim == 1 else list(result)
        return [_plain(r) for r in rows]
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        if result and all(isinstance(v, (int, float, np.generic)) for v in result):
            return [_plain(np.asarray(result))]
        return [_plain(v) for v in result]
    raise TypeError(f"unsupported action chunk type {type(result).__name__}")


def _git_rev(root: str | None) -> str | None:
    if not root:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _describe(exc: BaseException) -> str:
    """``ServerRestartedError: ...`` / ``WsError (timeout): ...``: the class name leads."""
    code = getattr(getattr(exc, "code", None), "value", None)
    return f"{type(exc).__name__}{f' ({code})' if code else ''}: {exc}"


def _ends_trial(exc: BaseException) -> bool:
    """XPolicyLab's two fatal outcomes: a timed-out call and a restarted server."""
    code = getattr(getattr(exc, "code", None), "value", None)
    return type(exc).__name__ == "ServerRestartedError" or code == "timeout"


class XPolicyBridge(RpcFacade):
    """``xpolicy.*`` over XPolicyLab's ``WsModelClient``; ``client_factory`` stands in for it in tests."""

    SERVICE_NAME = "xpolicy-bridge"

    def __init__(
        self,
        *,
        xpolicylab_root: str | None = None,
        client_factory: Callable[..., Any] | None = None,
        env_cfg: Path = ENV_CFG,
    ) -> None:
        super().__init__()
        self._root = xpolicylab_root
        self._factory = client_factory or self._ws_client
        self._env_cfg = env_cfg
        self._client: Any = None
        self._fatal: str | None = None
        self._encode = False
        for name in (
            "connect",
            "action_dims",
            "prepare_case",
            "reset",
            "update_obs",
            "get_action",
            "trial_end",
        ):
            self._rpc[f"xpolicy.{name}"] = getattr(self, name)
        self._rpc["xpolicy.close"] = self.close

    def _import_path(self) -> None:
        if self._root and self._root not in sys.path:
            sys.path.insert(0, self._root)

    def _ws_client(self, **kwargs: Any) -> Any:
        self._import_path()
        from client_server.ws.model_client import WsModelClient

        return WsModelClient(**kwargs)

    def close(self) -> dict[str, Any]:
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.warning("closing the XPolicyLab client failed", exc_info=True)
        return {"ok": True}

    def connect(
        self,
        url: str,
        trial_id: str,
        evaluation_id: str = "pi-embodied",
        action_case_id: str | None = None,
        encode_images: bool = False,
        request_timeout_s: float | None = None,
        max_connect_seconds: float | None = None,
    ) -> dict[str, Any]:
        self.close()
        self._fatal = None
        self._encode = bool(encode_images)
        t0 = time.monotonic()
        self._client = self._factory(
            url=url,
            evaluation_id=evaluation_id,
            trial_id=trial_id,
            action_case_id=action_case_id,
            request_timeout_s=request_timeout_s,
            max_connect_seconds=max_connect_seconds,
        )
        # PolicyEvalClient pins the HELLO_ACK's server_instance_id; report it.
        inner = getattr(self._client, "_client", None)
        return {
            "url": url,
            "server_instance_id": getattr(inner, "_server_instance_id", None),
            "xpolicylab_root": self._root,
            "xpolicylab_rev": _git_rev(self._root),
            "xpolicylab_pin": XPOLICYLAB_PIN,
            "encode_images": self._encode,
            # XPolicyLab's HELLO_ACK carries no dtype: the server's weight precision is what the
            # operator declares to pi (--xpolicy-precision), not something this bridge can know.
            "ms": round((time.monotonic() - t0) * 1000.0, 1),
        }

    def action_dims(self, env_cfg_type: str) -> dict[str, Any]:
        return action_dims(env_cfg_type, self._env_cfg)

    def _call(self, func_name: str, obs: Any = None) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError(
                "not connected to an XPolicyLab server: xpolicy.connect first"
            )
        if self._fatal is not None:
            raise RuntimeError(
                f"the XPolicyLab trial is over ({self._fatal}); reconnect for a new one"
            )
        t0 = time.monotonic()
        try:
            result = self._client.call(func_name=func_name, obs=obs)
        except Exception as exc:
            if _ends_trial(exc):
                self._fatal = _describe(exc)
            raise RuntimeError(f"{func_name}: {_describe(exc)}") from exc
        return {"result": result, "ms": round((time.monotonic() - t0) * 1000.0, 1)}

    def prepare_case(self, case_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        out = self._call("prepare_case", case_meta)
        return {**out, "result": _plain(out["result"])}

    def reset(self) -> dict[str, Any]:
        out = self._call("reset")
        return {**out, "result": _plain(out["result"])}

    def _encode_colors(self, obs: dict[str, Any]) -> None:
        """JPEG every camera color through XPolicyLab's ``encode_image_bit`` (the server decodes)."""
        self._import_path()
        from XPolicyLab.utils.process_data import encode_image_bit

        for camera in (obs.get("vision") or {}).values():
            if isinstance(camera, dict) and isinstance(camera.get("color"), np.ndarray):
                camera["color"] = encode_image_bit(camera["color"])

    def update_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obs, dict):
            raise TypeError(f"obs must be a dict, got {type(obs).__name__}")
        if self._encode:
            self._encode_colors(obs)
        out = self._call("update_obs", obs)
        return {**out, "result": _plain(out["result"])}

    def get_action(self) -> dict[str, Any]:
        out = self._call("get_action")
        return {"actions": normalize_chunk(out["result"]), "ms": out["ms"]}

    def trial_end(self, result: dict[str, Any] | None = None) -> dict[str, Any]:
        out = self._call("trial_end", result)
        return {**out, "result": _plain(out["result"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--transport", choices=["http"], default="http")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--parent-watch", action="store_true")
    p.add_argument(
        "--xpolicylab-root",
        default=os.environ.get("XPOLICYLAB_ROOT"),
        help="XPolicyLab checkout whose client_server/ws client is used (default XPOLICYLAB_ROOT); "
        "unset: XPolicyLab must be pip-installed in this venv",
    )
    args = p.parse_args()
    if (
        args.xpolicylab_root
        and not Path(args.xpolicylab_root, "client_server", "ws").is_dir()
    ):
        raise SystemExit(
            f"--xpolicylab-root {args.xpolicylab_root} has no client_server/ws"
        )
    XPolicyBridge(xpolicylab_root=args.xpolicylab_root).serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
