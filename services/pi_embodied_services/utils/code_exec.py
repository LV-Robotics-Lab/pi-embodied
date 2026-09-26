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

"""Code mode (CaP-X's "write code, execute it"): ``code.api`` and ``code.run`` for an env server.

The model's program runs in a **spawned subprocess that holds no simulator and no env
object**, only stub functions, one per primitive of the requested tier. A stub sends
``(name, args, kwargs)`` to the parent over a pipe and waits for the answer; the parent (the
env server, inside its one-call-at-a-time ``code.run``) runs the primitive through the
facade method the robot's tools call, so the same limits and the same stop generation
apply. CaP-X's executor instead ``exec``'d the code in-process with ``env`` in its globals
(``capx/envs/tasks/base.py``), which let a program bypass every check.

The pipe carries JSON, never pickle (arrays as ``{"__ndarray__": ...}`` of a numeric dtype,
see :func:`encode` / :func:`decode`): a message from the child cannot make the server
unpickle an object, and a message over :data:`MAX_MESSAGE` ends the run. The child starts
with the secret-looking environment variables removed (:func:`scrub_env`: names containing
KEY, TOKEN, SECRET, PASSWORD and the like, the cloud and git prefixes), in its own process
group (killed as a group, so the program's own subprocesses die with it), and under
``RLIMIT_NPROC`` (no new processes or threads), ``RLIMIT_FSIZE``, ``RLIMIT_AS`` and
``RLIMIT_CPU``. It can still open sockets: a server that must keep the program off the
network runs inside a container, as pi's own isolation does.

Per run: a wall-clock timeout (a stop is issued to the robot the moment it passes, also
inside a running primitive, and the child is killed), a primitive-call budget, an
accumulated translation cap (the units' ``maxMoveM`` idea), a temporary working directory,
and stdout, stderr, the traceback and ``RESULT`` each capped at 8 KB. A ``stop`` that
arrives while a run executes kills the child (:meth:`CodeRunner.abort`, from the facade's
``_on_stop``).

An env server that declares its primitive registry (``components/code_api.py``, ``code.api``)
builds the runner with :func:`registry_primitives`: every call then goes through
``CodeApi.resolve`` (the declared name, parameters and tier) to the registered RPC method, and
``code.api`` stays the registry's. See ``robots/libero/env_server.py``. A server without a
registry may still hand :class:`CodeRunner` bound facade methods as :class:`Primitive` objects and
serve ``runner.api`` itself.
"""

from __future__ import annotations

import base64
import contextlib
import inspect
import io
import json
import math
import multiprocessing
import os
import re
import shutil
import signal
import tempfile
import threading
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from pi_embodied_services.components import code_api as registry

#: ``code.api`` / ``code.run`` tiers: CaP-X's S2 (perception + high-level primitives), S3
#: (low-level moves only) and S4 (S3 with the docstrings' examples stripped). S1 is any tier
#: with ``privileged=True``, which adds the simulator's ground-truth primitive.
TIERS = ("high", "low", "low-noexamples")
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_CALLS = 50
#: stdout, stderr, the traceback and the encoded RESULT of a run are each cut here.
OUTPUT_CAP = 8 * 1024
#: A pipe message (either direction) above this ends the run.
MAX_MESSAGE = 64 << 20
#: Files the program writes are cut here (RLIMIT_FSIZE).
RLIMIT_FSIZE_BYTES = 64 << 20
#: Environment variable names the child never sees (case-insensitive substrings / prefixes).
SECRET_ENV_PATTERN = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|COOKIE|AUTH)"
    r"|^(AWS_|AZURE_|GOOGLE_|GCP_|GITHUB_|GH_|SSH_|ANTHROPIC_|OPENAI_)",
    re.IGNORECASE,
)
#: The child's address-space headroom beyond what the interpreter has mapped when the program
#: starts (numpy and the stubs need well under 1 GiB). Relative, because a spawned child
#: re-imports the server's main module, and torch/CUDA libraries map tens of GB of virtual
#: address space that an absolute cap would already exceed.
RLIMIT_AS_BYTES = 4 << 30
#: Elements above which an array in RESULT or the call log is replaced by its shape.
ARRAY_CAP = 4096
#: Characters above which a string in the call log is cut.
LOG_STR_CAP = 200
#: Array dtypes the pipe carries (no object arrays: nothing on the pipe may hold code).
WIRE_DTYPES = frozenset(
    "bool int8 int16 int32 int64 uint8 uint16 uint32 uint64 float16 float32 float64".split()
)


class CodeLimitError(RuntimeError):
    """A primitive call refused by the run's budget (raised inside the child)."""


@dataclass(frozen=True)
class Primitive:
    """One function the program may call: a bound facade method, the tiers that offer it,
    how far it may move the arm (for the run's translation cap), and whether it needs
    ``--privileged`` (simulator ground truth)."""

    name: str
    fn: Callable[..., Any]
    tiers: tuple[str, ...] = ("high", "low")
    move_m: Callable[[tuple, dict], float] | None = None
    privileged: bool = False
    #: Shown instead of ``fn``'s own signature and docstring (registry-declared primitives).
    signature: str | None = None
    doc: str | None = None

    def describe(self, *, examples: bool = True) -> dict:
        doc = self.doc if self.doc is not None else (inspect.getdoc(self.fn) or "")
        return {
            "name": self.name,
            "signature": self.signature or signature_of(self.fn),
            "doc": doc if examples else strip_examples(doc),
            "kind": "primitive",
        }


def signature_of(fn: Callable[..., Any]) -> str:
    """``(a, b=1) -> dict`` with the annotations evaluated (a module's ``from __future__
    import annotations`` would otherwise quote them)."""
    try:
        return str(inspect.signature(fn, eval_str=True))
    except Exception:
        return str(inspect.signature(fn))


def base_tier(tier: str) -> tuple[str, bool]:
    """``(the primitive tier, whether docstring examples are kept)`` of a ``code.api`` tier.
    The registry's ``privileged`` tier is the high tier (its ground truth is added by
    :meth:`CodeRunner.primitives`)."""
    if tier == "privileged":
        return "high", True
    if tier not in TIERS:
        raise ValueError(f"unknown code tier {tier!r}; one of {list(TIERS)}")
    return ("low" if tier.startswith("low") else "high"), tier != "low-noexamples"


def registry_primitives(
    api: registry.CodeApi,
    rpc: Mapping[str, Callable[..., Any]],
    *,
    move_m: Callable[[str, dict], float] | None = None,
    after: Callable[[registry.Primitive], None] | None = None,
) -> list[Primitive]:
    """The runner's primitives for a server's declared registry: one stub per declared primitive
    whose calls go through ``api.resolve`` (declared name, parameters, tier) to the registered RPC
    method. Positional arguments fill the declared parameters in order. ``move_m(method, kwargs)``
    estimates a call's translation for the run's cap; ``after(primitive)`` runs after every
    mutating call (frames for the episode video)."""
    out: list[Primitive] = []
    for p in api.primitives("privileged") + [
        q for q in api.primitives("low") if "high" not in q.tiers
    ]:
        privileged = "privileged" in p.tiers
        tier_of = "privileged" if privileged else None
        tiers = (
            ("high", "low")
            if privileged
            else tuple(t for t in p.tiers if t in ("high", "low"))
        )
        out.append(
            Primitive(
                p.name,
                _registry_call(
                    api, rpc, p, "privileged" if privileged else None, after
                ),
                tiers,
                move_m=(
                    lambda a, k, _p=p, _t=tier_of: move_m(
                        *api.resolve(_p.name, _bind(_p, a, k), _t)
                    )
                )
                if move_m
                else None,
                privileged=privileged,
                signature=_registry_signature(p),
                doc=_registry_doc(p),
            )
        )
    return out


def _bind(p: registry.Primitive, args: tuple, kwargs: dict) -> dict:
    names = list(p.params)
    if len(args) > len(names):
        raise TypeError(
            f"{p.name}() takes at most {len(names)} positional arguments ({len(args)} given)"
        )
    bound = dict(zip(names, args))
    dup = [k for k in kwargs if k in bound]
    if dup:
        raise TypeError(f"{p.name}() got multiple values for {', '.join(dup)}")
    bound.update(kwargs)
    return bound


def _registry_call(api, rpc, p: registry.Primitive, tier: str | None, after):
    def call(*args, **kwargs):
        method, kw = api.resolve(p.name, _bind(p, args, kwargs), tier)
        out = rpc[method](**kw)
        if after is not None and p.mutating:
            after(p)
        return out

    call.__name__ = call.__qualname__ = p.name
    return call


def _registry_signature(p: registry.Primitive) -> str:
    parts = [
        f"{k}: {v.type}" if v.required else f"{k}: {v.type} = None"
        for k, v in p.params.items()
    ]
    return f"({', '.join(parts)})"


def _registry_doc(p: registry.Primitive) -> str:
    lines = [p.doc.strip()]
    if p.params:
        lines += ["", "Args:"]
        for k, v in p.params.items():
            kind = v.type if v.required else f"{v.type}, optional"
            lines.append(f"    {k} ({kind}): {v.description}".rstrip(": "))
    if p.mutating:
        lines += ["", "Moves the robot."]
    return "\n".join(lines)


def strip_examples(doc: str) -> str:
    """Drop the ``Example:`` / ``Examples:`` sections of a Google-style docstring (CaP-X's
    exampleless tier, ``control_reduced_exampleless.py``): from the header to the next
    section header at the same indentation, or the end."""
    out: list[str] = []
    skipping: int | None = None
    for line in doc.splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if skipping is not None:
            if stripped and indent <= skipping and stripped.endswith(":"):
                skipping = None
            else:
                continue
        if stripped.lower() in ("example:", "examples:"):
            skipping = indent
            continue
        out.append(line)
    return "\n".join(out).rstrip()


def jsonable(value: Any, cap: int = ARRAY_CAP, str_cap: int | None = None) -> Any:
    """A JSON-able copy: arrays become lists (or a shape stub past ``cap`` elements),
    numpy scalars plain numbers, strings cut at ``str_cap``, anything else its ``repr``."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if str_cap is not None and len(value) > str_cap:
            return f"{value[:str_cap]}...[{len(value) - str_cap} more chars]"
        return value
    if isinstance(value, np.ndarray):
        if value.size > cap:
            return {"ndarray": str(value.dtype), "shape": list(value.shape)}
        return jsonable(value.tolist(), cap, str_cap)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v, cap, str_cap) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v, cap, str_cap) for v in value]
    return repr(value)[:LOG_STR_CAP]


def _cap(text: str, tail: bool = False) -> str:
    data = text.encode("utf-8", "replace")
    if len(data) <= OUTPUT_CAP:
        return text
    if tail:  # a traceback: its last line is the error
        kept = data[-OUTPUT_CAP:].decode("utf-8", "ignore")
        return f"[truncated {len(data) - OUTPUT_CAP} bytes]\n{kept}"
    head = data[:OUTPUT_CAP].decode("utf-8", "ignore")
    return f"{head}\n[truncated {len(data) - OUTPUT_CAP} bytes]"


# ---------------------------------------------------------------------------
# The wire: JSON both ways. Arrays travel as {"__ndarray__": dtype, "shape": [...], "data":
# base64}; a dtype outside WIRE_DTYPES is refused. Anything else non-JSON is its repr.


def encode(value: Any, strict: bool = False) -> Any:
    """``value`` as JSON-able data with arrays kept intact (see the module docstring).
    Anything else is its ``repr``, or a TypeError when ``strict`` (the program's arguments:
    a primitive must not silently get a string for an object)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        if str(value.dtype) not in WIRE_DTYPES:
            raise TypeError(f"an array of dtype {value.dtype} cannot cross the pipe")
        data = np.ascontiguousarray(value)
        return {
            "__ndarray__": str(data.dtype),
            "shape": list(data.shape),
            "data": base64.b64encode(data.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): encode(v, strict) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v, strict) for v in value]
    if strict:
        raise TypeError(
            f"an argument of type {type(value).__name__} cannot cross the pipe; pass "
            "numbers, strings, lists, dicts or numeric arrays"
        )
    return repr(value)[:LOG_STR_CAP]


def decode(value: Any) -> Any:
    """The inverse of :func:`encode`; a malformed array stub is a ValueError."""
    if isinstance(value, dict):
        if "__ndarray__" in value:
            dtype = value.get("__ndarray__")
            shape = value.get("shape")
            data = value.get("data")
            if (
                dtype not in WIRE_DTYPES
                or not isinstance(shape, list)
                or not all(isinstance(n, int) and n >= 0 for n in shape)
                or not isinstance(data, str)
            ):
                raise ValueError("malformed array on the pipe")
            raw = base64.b64decode(data, validate=True)
            arr = np.frombuffer(raw, dtype=np.dtype(dtype))
            if arr.size != math.prod(shape):
                raise ValueError("array bytes do not match its shape")
            return arr.reshape(shape).copy()
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def _send(conn, message: Any, strict: bool = False) -> None:
    conn.send_bytes(json.dumps(encode(message, strict)).encode("utf-8"))


def _recv(conn) -> Any:
    """The next message; ``OSError("bad message length")`` when it is over MAX_MESSAGE."""
    return decode(json.loads(conn.recv_bytes(MAX_MESSAGE).decode("utf-8")))


def scrub_env(env: Mapping[str, str]) -> dict[str, str]:
    """``env`` without the variables :data:`SECRET_ENV_PATTERN` matches."""
    return {k: v for k, v in env.items() if not SECRET_ENV_PATTERN.search(k)}


# ---------------------------------------------------------------------------
# Helpers (--code-helpers): CaP-X's skill library, control_reduced_skill_library.py:44-56.
# Pure numpy; none of them moves the robot. They run inside the child.


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z] (Sheppard's method)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w, x, y, z = (
            0.25 * S,
            (R[2, 1] - R[1, 2]) / S,
            (R[0, 2] - R[2, 0]) / S,
            (R[1, 0] - R[0, 1]) / S,
        )
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (
            (R[2, 1] - R[1, 2]) / S,
            0.25 * S,
            (R[0, 1] + R[1, 0]) / S,
            (R[0, 2] + R[2, 0]) / S,
        )
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (
            (R[0, 2] - R[2, 0]) / S,
            (R[0, 1] + R[1, 0]) / S,
            0.25 * S,
            (R[1, 2] + R[2, 1]) / S,
        )
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (
            (R[1, 0] - R[0, 1]) / S,
            (R[0, 2] + R[2, 0]) / S,
            (R[1, 2] + R[2, 1]) / S,
            0.25 * S,
        )
    return np.array([w, x, y, z])


def decompose_transform(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 homogeneous transform into (position (3,), quaternion [w, x, y, z])."""
    T = np.asarray(T, dtype=np.float64)
    return T[:3, 3], rotation_matrix_to_quaternion(T[:3, :3])


def depth_to_point_cloud(depth_img: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Depth map (H, W) in metres + 3x3 K -> (H, W, 3) points in the camera frame."""
    depth_img = np.asarray(depth_img, dtype=np.float64)
    if depth_img.ndim == 3:
        depth_img = depth_img[:, :, 0]
    K = np.asarray(intrinsics, dtype=np.float64)
    h, w = depth_img.shape
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    z = depth_img
    x = (x_grid - K[0, 2]) * z / K[0, 0]
    y = (y_grid - K[1, 2]) * z / K[1, 1]
    return np.dstack((x, y, z))


def mask_to_world_points(
    mask: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray
) -> np.ndarray:
    """Pixels of a binary mask (H, W) -> (N, 3) world points (extrinsics: 4x4 cam-to-world);
    pixels without depth (<= 0) are dropped."""
    mask = np.asarray(mask)
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return np.empty((0, 3))
    z = depth[ys, xs]
    valid = z > 0
    ys, xs, z = ys[valid], xs[valid], z[valid]
    K = np.asarray(intrinsics, dtype=np.float64)
    x_cam = (xs - K[0, 2]) * z / K[0, 0]
    y_cam = (ys - K[1, 2]) * z / K[1, 1]
    points_cam = np.stack([x_cam, y_cam, z], axis=-1)
    hom = np.hstack([points_cam, np.ones((len(points_cam), 1))])
    return (np.asarray(extrinsics, dtype=np.float64) @ hom.T).T[:, :3]


def pixel_to_world_point(
    u: int, v: int, z: float, intrinsics: np.ndarray, extrinsics: np.ndarray
) -> np.ndarray:
    """One pixel (u = column, v = row) at depth z (m) -> [x, y, z] in the world frame."""
    K = np.asarray(intrinsics, dtype=np.float64)
    p_cam = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, 1.0])
    return (np.asarray(extrinsics, dtype=np.float64) @ p_cam)[:3]


def transform_points(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N, 3) or (H, W, 3) points; same shape back."""
    points = np.asarray(points, dtype=np.float64)
    flat = points.reshape(-1, 3)
    hom = np.hstack((flat, np.ones((flat.shape[0], 1))))
    out = (np.asarray(transform_matrix, dtype=np.float64) @ hom.T).T
    return out[:, :3].reshape(points.shape)


def interpolate_segment(
    p1: np.ndarray, p2: np.ndarray, step: float = 0.03
) -> list[np.ndarray]:
    """Waypoints from p1 to p2 (both included) at most ``step`` metres apart."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    dist = np.linalg.norm(p2 - p1)
    if dist < 1e-6:
        return [p1]
    n = int(np.ceil(dist / step))
    return [p1 + (p2 - p1) * t for t in np.linspace(0, 1, n + 1)]


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """v / |v| (v itself when |v| < 1e-6)."""
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    return v if norm < 1e-6 else v / norm


def select_top_down_grasp(
    grasps: np.ndarray,
    scores: np.ndarray,
    cam_to_world: np.ndarray,
    vertical_threshold: float = 0.8,
) -> tuple:
    """Best-scoring grasp (N, 4, 4, camera frame) whose approach axis (gripper +z) points
    down in the world at least ``vertical_threshold``: (4x4 world grasp, score), or (None, -inf)."""
    best, best_score = None, -np.inf
    world_z = np.array([0.0, 0.0, 1.0])
    T = np.asarray(cam_to_world, dtype=np.float64)
    for g_camera, score in zip(np.asarray(grasps, dtype=np.float64), scores):
        g_world = T @ g_camera
        if -np.dot(g_world[:3, 2], world_z) > vertical_threshold and score > best_score:
            best, best_score = g_world, float(score)
    return best, best_score


HELPERS: dict[str, Callable[..., Any]] = {
    f.__name__: f
    for f in (
        rotation_matrix_to_quaternion,
        decompose_transform,
        depth_to_point_cloud,
        mask_to_world_points,
        pixel_to_world_point,
        transform_points,
        interpolate_segment,
        normalize_vector,
        select_top_down_grasp,
    )
}


def describe_helpers() -> list[dict]:
    return [
        {
            "name": name,
            "signature": signature_of(fn),
            "doc": inspect.getdoc(fn) or "",
            "kind": "helper",
        }
        for name, fn in HELPERS.items()
    ]


# ---------------------------------------------------------------------------
# The child


def _mapped_bytes() -> int:
    """The process's virtual size (Linux: VmSize; 0 elsewhere)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _set_limit(which: int, soft: int, hard: int) -> None:
    try:
        import resource

        resource.setrlimit(which, (soft, hard))
    except (ImportError, ValueError, OSError):
        # Not every platform accepts every limit (macOS refuses RLIMIT_AS); best effort.
        pass


def _stub(conn, name: str) -> Callable[..., Any]:
    def call(*args, **kwargs):
        try:
            _send(conn, ["call", name, list(args), dict(kwargs)], strict=True)
        except TypeError as exc:
            raise TypeError(f"{name}(): {exc}") from None
        kind, payload = _recv(conn)
        if kind == "ok":
            return payload
        if kind == "limit":
            raise CodeLimitError(payload)
        raise RuntimeError(payload)

    call.__name__ = call.__qualname__ = name
    return call


def _child_main(
    conn, code: str, names: list[str], helpers: list[str], limits: dict, cwd: str
):
    """The spawned process: rlimits, a temp cwd, stubs for the tier's primitives, then the
    program; nothing else of the server is in its globals."""
    import resource

    # Its own process group: the parent kills the group, so the program's subprocesses go too.
    with contextlib.suppress(OSError):
        os.setpgrp()
    cap = _mapped_bytes() + limits["as_bytes"]
    _set_limit(resource.RLIMIT_AS, cap, cap)
    # Relative too: importing the server's main module (torch) already cost CPU seconds.
    used = resource.getrusage(resource.RUSAGE_SELF)
    cpu = int(math.ceil(used.ru_utime + used.ru_stime)) + limits["cpu_s"]
    _set_limit(resource.RLIMIT_CPU, cpu, cpu + 5)
    _set_limit(resource.RLIMIT_FSIZE, limits["fsize_bytes"], limits["fsize_bytes"])
    # No new processes or threads (every clone counts against NPROC; the user already has
    # more than one process, so any further one is refused; root is exempt).
    _set_limit(resource.RLIMIT_NPROC, 1, 1)
    os.chdir(cwd)
    g: dict[str, Any] = {"__name__": "__main__", "RESULT": None, "np": np, "math": math}
    for name in names:
        g[name] = _stub(conn, name)
    for name in helpers:
        g[name] = HELPERS[name]
    out, err = io.StringIO(), io.StringIO()
    tb: str | None = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            exec(compile(code, "<run_code>", "exec"), g, g)
        except BaseException:  # the program's own failure, whatever it raised
            tb = traceback.format_exc()
    result = jsonable(g.get("RESULT"))
    if len(json.dumps(result).encode("utf-8")) > OUTPUT_CAP:
        result = {"truncated": f"RESULT is over {OUTPUT_CAP} bytes; return less"}
    _send(
        conn,
        [
            "done",
            {
                "stdout": _cap(out.getvalue()),
                "stderr": _cap(err.getvalue()),
                "traceback": None if tb is None else _cap(tb, tail=True),
                "result": result,
            },
        ],
    )
    conn.close()


def _kill(proc: multiprocessing.process.BaseProcess) -> None:
    """SIGKILL the child's process group (its subprocesses with it, whether or not the child
    itself still runs), then the child if it is somehow still alive. Called before the child
    is joined: until then its pid is not reused, so the group is its own."""
    pid = proc.pid
    if pid is None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pid, signal.SIGKILL)
    if proc.is_alive():
        proc.kill()


# ---------------------------------------------------------------------------
# The parent


class CodeRunner:
    """``code.api`` / ``code.run`` over a facade's primitives.

    ``stop_requested`` is the facade's; ``on_timeout`` is called once when a run hits its
    wall-clock timeout (the facade issues a stop to the robot); ``begin`` / ``finish`` bracket
    a run for the facade's bookkeeping (``finish`` returns fields merged into the result:
    steps taken, the latest observation, frames).
    """

    def __init__(
        self,
        primitives: list[Primitive],
        *,
        stop_requested: Callable[[], bool] = lambda: False,
        on_timeout: Callable[[], None] | None = None,
        begin: Callable[[], None] | None = None,
        finish: Callable[[], dict] | None = None,
    ) -> None:
        self._primitives = {p.name: p for p in primitives}
        self._stop_requested = stop_requested
        self._on_timeout = on_timeout
        self._begin = begin
        self._finish = finish
        self._lock = threading.Lock()
        self._proc: multiprocessing.process.BaseProcess | None = None
        self._abort = threading.Event()

    def primitives(self, tier: str, privileged: bool = False) -> list[Primitive]:
        """The tier's primitives; the registry's ``privileged`` tier (or ``privileged=True``) adds
        the privileged ones."""
        base, _ = base_tier(tier)
        privileged = privileged or tier == "privileged"
        return [
            p
            for p in self._primitives.values()
            if base in p.tiers and (privileged or not p.privileged)
        ]

    def api(
        self, tier: str = "high", helpers: bool = False, privileged: bool = False
    ) -> list[dict]:
        """``[{name, signature, doc, kind}]`` of what a program may call in ``tier``."""
        _, examples = base_tier(tier)
        out = [p.describe(examples=examples) for p in self.primitives(tier, privileged)]
        if helpers:
            out.extend(describe_helpers())
        return out

    def abort(self) -> None:
        """Kill the running child (a ``stop`` while a run executes); no-op when idle."""
        self._abort.set()
        with self._lock:
            proc = self._proc
        if proc is not None:
            _kill(proc)

    def run(
        self,
        code: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        tier: str = "high",
        max_calls: int = DEFAULT_MAX_CALLS,
        max_move_m: float | None = None,
        helpers: bool = False,
        privileged: bool = False,
    ) -> dict:
        """Run ``code`` in a fresh spawned process; see the module docstring for the limits.

        Result: ``status`` (``ran``, ``error``: the program raised or was stopped, ``timeout``),
        ``stdout``, ``stderr`` (8 KB each), ``traceback``, ``error`` (its last line), ``result``
        (the program's ``RESULT`` variable, JSON-able), ``calls`` (the primitive log), ``n_calls``,
        ``move_m``, ``limit`` (the budget that refused a call, if any), ``cancelled`` (a stop
        arrived), ``stop_issued`` (a timeout stopped the robot), ``ms``, plus ``finish``'s fields.
        """
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string")
        timeout_s = float(timeout_s)
        if not (timeout_s > 0):
            raise ValueError("timeout_s must be positive")
        allowed = self.primitives(tier, privileged)
        names = [p.name for p in allowed]
        helper_names = list(HELPERS) if helpers else []
        started = time.monotonic()
        state = _Run(
            max_calls=int(max_calls),
            max_move_m=max_move_m,
            deadline=started + timeout_s,
        )
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        cwd = tempfile.mkdtemp(prefix="run_code-")
        limits = {
            "as_bytes": RLIMIT_AS_BYTES,
            "cpu_s": int(math.ceil(timeout_s)) + 2,
            "fsize_bytes": RLIMIT_FSIZE_BYTES,
        }
        proc = ctx.Process(
            target=_child_main,
            args=(child_conn, code, names, helper_names, limits, cwd),
            daemon=True,
        )
        if self._begin:
            self._begin()
        with self._lock:
            self._abort.clear()
            self._proc = proc
        try:
            self._start_scrubbed(proc)
            child_conn.close()
            outcome = self._serve(parent_conn, proc, allowed, state)
        finally:
            _kill(proc)
            proc.join(5)
            with self._lock:
                self._proc = None
            parent_conn.close()
            shutil.rmtree(cwd, ignore_errors=True)
        done = state.done or {}
        result: dict[str, Any] = {
            "status": "ran",
            "stdout": _cap(done.get("stdout", "")),
            "stderr": _cap(done.get("stderr", "")),
            "traceback": done.get("traceback"),
            "error": None,
            "result": done.get("result"),
            "calls": state.log,
            "n_calls": state.calls,
            "move_m": round(state.moved, 5),
            "timeout_s": timeout_s,
            "ms": int((time.monotonic() - started) * 1000),
        }
        if state.limit:
            result["limit"] = state.limit
        if outcome == "timeout":
            result["status"] = "timeout"
            result["error"] = (
                f"the program ran past its {timeout_s:g} s timeout and was killed"
            )
            result["stop_issued"] = True
            self._issue_stop(state)
        elif outcome == "oversized":
            result["status"] = "error"
            result["error"] = (
                f"the program sent a message over {MAX_MESSAGE >> 20} MB and was killed"
            )
        elif outcome == "cancelled":
            result["status"] = "error"
            result["cancelled"] = True
            result["error"] = "stopped: the run was aborted"
        elif outcome == "died":
            result["status"] = "error"
            result["error"] = (
                f"the code process exited unexpectedly (exit code {proc.exitcode})"
            )
        elif done.get("traceback"):
            result["status"] = "error"
            result["error"] = done["traceback"].rstrip().splitlines()[-1][:2000]
        if self._finish:
            result.update(self._finish())
        return result

    @staticmethod
    def _start_scrubbed(proc: multiprocessing.process.BaseProcess) -> None:
        """Start the child with the secret-looking variables out of its environment: spawn
        hands the child the parent's environment as it is at start, so they are removed for
        the moment of the start and put back (/proc/<child>/environ never held them)."""
        kept = scrub_env(os.environ)
        removed = {k: v for k, v in os.environ.items() if k not in kept}
        for k in removed:
            del os.environ[k]
        try:
            proc.start()
        finally:
            os.environ.update(removed)

    def _issue_stop(self, state: _Run) -> None:
        """Stop the robot once per run (the timeout: from the timer inside a primitive, or
        after the child was killed)."""
        with self._lock:
            if state.stop_issued:
                return
            state.stop_issued = True
        if self._on_timeout:
            self._on_timeout()

    def _serve(self, conn, proc, allowed: list[Primitive], state: _Run) -> str:
        """Answer the child's primitive calls until it is done, times out, is aborted or dies."""
        by_name = {p.name: p for p in allowed}
        deadline = state.deadline
        while True:
            now = time.monotonic()
            if state.stop_issued or now >= deadline:
                return "timeout"
            if self._abort.is_set():
                return "cancelled"
            try:
                ready = conn.poll(min(0.05, deadline - now))
            except (EOFError, OSError):
                return "cancelled" if self._abort.is_set() else "died"
            if not ready:
                if not proc.is_alive():
                    return "cancelled" if self._abort.is_set() else "died"
                continue
            try:
                msg = _recv(conn)
            except (EOFError, OSError) as exc:
                if "bad message length" in str(exc):
                    return "oversized"
                return "cancelled" if self._abort.is_set() else "died"
            except (ValueError, UnicodeDecodeError):
                return "died"  # not our protocol: the program tampered with the pipe
            if not isinstance(msg, list) or len(msg) < 2:
                return "died"
            if msg[0] == "done":
                state.done = msg[1] if isinstance(msg[1], dict) else {}
                return "done"
            if len(msg) != 4 or not isinstance(msg[3], dict):
                return "died"
            _, name, args, kwargs = msg
            reply = self._call(by_name, str(name), tuple(args), kwargs, state)
            # The timeout's stop may have aborted this run too: it is still a timeout.
            if state.stop_issued or time.monotonic() >= deadline:
                return "timeout"
            if self._abort.is_set():
                return "cancelled"
            try:
                _send(conn, reply)
            except (BrokenPipeError, OSError):
                return "cancelled" if self._abort.is_set() else "died"

    def _call(
        self, by_name: dict[str, Primitive], name: str, args, kwargs, state: _Run
    ):
        entry: dict[str, Any] = {
            "name": name,
            "args": jsonable(list(args), 64, LOG_STR_CAP),
            "kwargs": jsonable(dict(kwargs), 64, LOG_STR_CAP),
        }
        state.log.append(entry)
        prim = by_name.get(name)
        if prim is None:
            entry["error"] = f"{name} is not a primitive of this tier"
            return ("error", entry["error"])
        if self._stop_requested():
            entry["error"] = "stopped: a stop was requested"
            self._abort.set()
            return ("error", entry["error"])
        if state.calls >= state.max_calls:
            state.limit = "max_calls"
            entry["error"] = entry["refused"] = (
                f"call budget exhausted: at most {state.max_calls} primitive calls per run_code"
            )
            return ("limit", entry["error"])
        move = 0.0
        if prim.move_m is not None:
            try:
                move = float(prim.move_m(args, kwargs))
            except Exception as exc:  # a malformed argument: the primitive reports it
                entry["error"] = f"{type(exc).__name__}: {exc}"
                return ("error", entry["error"])
            if (
                state.max_move_m is not None
                and state.moved + move > state.max_move_m + 1e-9
            ):
                state.limit = "max_move_m"
                entry["error"] = entry["refused"] = (
                    f"move budget exhausted: this call moves {move:.3f} m, {state.moved:.3f} m "
                    f"already moved, at most {state.max_move_m:g} m per run_code"
                )
                return ("limit", entry["error"])
        state.calls += 1
        t0 = time.monotonic()
        # The wall clock holds inside the primitive too: when the deadline passes while it
        # runs, the robot gets its stop now (the primitive returns through it), not after.
        timer = threading.Timer(
            max(0.0, state.deadline - t0), self._issue_stop, args=(state,)
        )
        timer.daemon = True
        timer.start()
        try:
            out = prim.fn(*args, **kwargs)
        except Exception as exc:
            entry["ms"] = int((time.monotonic() - t0) * 1000)
            entry["error"] = f"{type(exc).__name__}: {exc}"[:2000]
            return ("error", entry["error"])
        finally:
            timer.cancel()
        entry["ms"] = int((time.monotonic() - t0) * 1000)
        if move:
            state.moved += move
            entry["move_m"] = round(move, 5)
        if isinstance(out, dict) and out.get("cancelled"):
            entry["cancelled"] = True
        return ("ok", out)


class _Run:
    """One run's budget and log."""

    def __init__(
        self, *, max_calls: int, max_move_m: float | None, deadline: float
    ) -> None:
        self.max_calls = max_calls
        self.max_move_m = None if max_move_m is None else float(max_move_m)
        self.deadline = deadline
        self.calls = 0
        self.moved = 0.0
        self.limit: str | None = None
        self.log: list[dict] = []
        self.done: dict | None = None
        self.stop_issued = False


__all__ = [
    "registry_primitives",
    "DEFAULT_MAX_CALLS",
    "DEFAULT_TIMEOUT_S",
    "HELPERS",
    "MAX_MESSAGE",
    "OUTPUT_CAP",
    "SECRET_ENV_PATTERN",
    "TIERS",
    "CodeLimitError",
    "CodeRunner",
    "Primitive",
    "base_tier",
    "decode",
    "describe_helpers",
    "encode",
    "jsonable",
    "scrub_env",
    "strip_examples",
]
