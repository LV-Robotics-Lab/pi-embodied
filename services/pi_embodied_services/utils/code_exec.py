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
unpickle an object, a message over :data:`MAX_MESSAGE` ends the run, and any other message
the protocol does not allow (wrong kinds or field types, nesting too deep) ends it as the
child's failure, with the run's accounting kept.

Isolation. The child is a fresh ``python -c`` process (it does not re-import the server's
main module) started with an environment built for it (:func:`child_env`: the server's
minus the secret-looking variables, :func:`scrub_env`; the server's own ``os.environ`` is
never touched). It runs in its own session, with stdin and stdout on /dev/null. The server
is non-dumpable (``prctl(PR_SET_DUMPABLE, 0)``, set when a runner is built and before every
run), so no process of its uid can read its /proc/<pid>/environ or memory or ptrace it. A
root server also runs the program under an unprivileged uid of its own
(:func:`sandbox_uid`, ``PI_EMBODIED_CODE_UID`` overrides): the child imports what it needs
(:data:`PRELOAD`), sets its rlimits, then drops to that uid and gid for good before the
program runs, in a temporary directory it owns. ``RLIMIT_NPROC`` (then binding: the uid
already has this process) refuses new processes and threads, and ``RLIMIT_FSIZE``,
``RLIMIT_AS`` and ``RLIMIT_CPU`` apply. At the end of every run :func:`kill_tree` kills the
child's session and group, its descendants and every process of the sandbox uid (cgroups
are read-only in the containers the servers run in). It can still open sockets: the env
server refuses business calls while a run executes, and a server that must keep the
program off the network runs inside a container, as pi's own isolation does.

Per run: a wall-clock timeout (a stop is issued to the robot the moment it passes, also
inside a running primitive, whose outbound RPCs are bounded by what is left of it, and the
child is killed), a primitive-call budget, arguments with NaN or infinity refused, an
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
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from pi_embodied_services.components import code_api as registry
from pi_embodied_services.utils.rpc.deadline import call_deadline

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
    #: Raises to refuse a call before it runs (e.g. a size the run's wall clock cannot bound).
    check: Callable[[tuple, dict], None] | None = None
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
    check: Callable[[str, dict], None] | None = None,
) -> list[Primitive]:
    """The runner's primitives for a server's declared registry: one stub per declared primitive
    whose calls go through ``api.resolve`` (declared name, parameters, tier) to the registered RPC
    method. Positional arguments fill the declared parameters in order. ``move_m(method, kwargs)``
    estimates a call's translation for the run's cap; ``check(method, kwargs)`` raises to refuse a
    call before it runs; ``after(primitive)`` runs after every mutating call (frames for the
    episode video)."""
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
                check=(
                    lambda a, k, _p=p, _t=tier_of: check(
                        *api.resolve(_p.name, _bind(_p, a, k), _t)
                    )
                )
                if check
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
# Process isolation (Linux; best effort elsewhere)

#: The unprivileged uids a root server runs its programs as: one per server, held by an
#: flock on ``<lock dir>/pi-embodied-sandbox-<uid>.lock`` for the server's lifetime, so a
#: sweep of "every process of the sandbox uid" only ever hits this server's program.
SANDBOX_UID_BASE = 61000
SANDBOX_UID_COUNT = 256
#: ``PI_EMBODIED_CODE_UID``: a fixed uid for the programs, or ``none`` to keep the server's
#: uid (then only PR_SET_DUMPABLE protects the server).
SANDBOX_UID_ENV = "PI_EMBODIED_CODE_UID"
#: Modules imported before the child drops its uid: afterwards a root server's interpreter
#: (e.g. under /root, mode 0700) may be unreadable, so later imports can fail.
PRELOAD = tuple(
    "array bisect cmath collections copy dataclasses datetime decimal enum fractions "
    "functools heapq itertools json math numbers operator pprint random re statistics "
    "string struct textwrap time typing numpy.linalg numpy.random numpy.fft".split()
)
PRELOAD_OPTIONAL = ("scipy.spatial.transform",)
PR_SET_DUMPABLE = 4

_sandbox_lock: Any = None
_sandbox_uid: int | None = None


def _prctl_dumpable_off() -> bool:
    """``prctl(PR_SET_DUMPABLE, 0)``: other processes of this uid can no longer read this
    process's /proc/<pid>/environ, /proc/<pid>/mem or ptrace it. Linux only."""
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) == 0
    except (OSError, AttributeError):
        return False


def sandbox_uid() -> int | None:
    """The uid this server's programs run as, or None (not root, not Linux, or disabled)."""
    global _sandbox_lock, _sandbox_uid
    if _sandbox_uid is not None:
        return _sandbox_uid
    fixed = os.environ.get(SANDBOX_UID_ENV, "").strip()
    if fixed.lower() == "none":
        return None
    if not hasattr(os, "getuid") or os.getuid() != 0 or not os.path.isdir("/proc"):
        return None
    if fixed:
        _sandbox_uid = int(fixed)
        return _sandbox_uid
    import fcntl

    lock_dir = "/run/lock" if os.path.isdir("/run/lock") else tempfile.gettempdir()
    for uid in range(SANDBOX_UID_BASE, SANDBOX_UID_BASE + SANDBOX_UID_COUNT):
        path = os.path.join(lock_dir, f"pi-embodied-sandbox-{uid}.lock")
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue
        if _uid_processes(uid):
            # A leftover of a server that died without its sweep: take the next uid.
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            continue
        _sandbox_lock, _sandbox_uid = fd, uid
        return uid
    return None


def _proc_status(pid: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                out[k] = v.strip()
    except OSError:
        pass
    return out


def _pids() -> list[int]:
    try:
        return [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return []


def _uid_processes(uid: int) -> list[int]:
    """Live processes with ``uid`` as any of their real / effective / saved uids."""
    out = []
    for pid in _pids():
        st = _proc_status(pid)
        uids = st.get("Uid", "").split()
        if str(uid) in uids[:3] and not st.get("State", "").startswith("Z"):
            out.append(pid)
    return out


def _tree(root: int) -> set[int]:
    """``root``'s descendants (by parent pid) and every process in its session or group."""
    table = {}
    for pid in _pids():
        try:
            with open(f"/proc/{pid}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if fields[0] == "Z":
            continue
        table[pid] = (int(fields[1]), int(fields[2]), int(fields[3]))  # ppid, pgrp, sid
    found = {p for p, (_, pgrp, sid) in table.items() if pgrp == root or sid == root}
    frontier = {root} | found
    while frontier:
        nxt = {p for p, (ppid, _, _) in table.items() if ppid in frontier} - found
        found |= nxt
        frontier = nxt
    found.discard(os.getpid())
    return found


def kill_tree(pid: int | None, uid: int | None) -> None:
    """SIGKILL the child's session and group, its descendants and, when it ran under the
    sandbox uid, every process of that uid (which catches double-forked orphans), until
    none is left. Called before the child is reaped, so its pid is not reused meanwhile."""
    if pid is None:
        return
    with contextlib.suppress(OSError):
        os.killpg(pid, signal.SIGKILL)
    if not os.path.isdir("/proc"):
        return
    for _ in range(50):
        victims = _tree(pid) | (set(_uid_processes(uid)) if uid is not None else set())
        if not victims:
            return
        for v in victims:
            with contextlib.suppress(OSError):
                os.kill(v, signal.SIGKILL)
        time.sleep(0.02)


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


def _drop_privileges(uid: int) -> None:
    """Become ``uid`` (and gid ``uid``, no supplementary groups) for good; refuse to go on
    if it did not take."""
    os.setgroups([])
    os.setresgid(uid, uid, uid)
    os.setresuid(uid, uid, uid)
    if os.getresuid() != (uid, uid, uid) or os.getresgid() != (uid, uid, uid):
        raise SystemExit("run_code: could not drop privileges")
    try:
        os.setuid(0)
    except PermissionError:
        return
    raise SystemExit("run_code: privileges came back")


def _child_entry(fd: int) -> None:
    """``python -c`` entry of the child: the setup arrives as the first pipe message."""
    from multiprocessing.connection import Connection

    conn = Connection(fd)
    setup = json.loads(conn.recv_bytes(MAX_MESSAGE).decode("utf-8"))
    _child_main(
        conn,
        setup["code"],
        setup["names"],
        setup["helpers"],
        setup["limits"],
        setup["cwd"],
        setup.get("uid"),
    )


def _child_main(
    conn,
    code: str,
    names: list[str],
    helpers: list[str],
    limits: dict,
    cwd: str,
    uid: int | None = None,
):
    """The child: stdin off, preloaded modules, rlimits, the sandbox uid, a temp cwd, stubs
    for the tier's primitives, then the program; nothing else of the server is in its globals."""
    import importlib
    import resource

    with contextlib.suppress(OSError):
        null = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null, 0)
        os.close(null)
    if uid is not None:  # only a dropped uid may be unable to import later
        for name in (*PRELOAD, *PRELOAD_OPTIONAL):
            with contextlib.suppress(ImportError):
                importlib.import_module(name)
    cap = _mapped_bytes() + limits["as_bytes"]
    _set_limit(resource.RLIMIT_AS, cap, cap)
    used = resource.getrusage(resource.RUSAGE_SELF)
    cpu = int(math.ceil(used.ru_utime + used.ru_stime)) + limits["cpu_s"]
    _set_limit(resource.RLIMIT_CPU, cpu, cpu + 5)
    _set_limit(resource.RLIMIT_FSIZE, limits["fsize_bytes"], limits["fsize_bytes"])
    if uid is not None:
        _drop_privileges(uid)
    # No new processes or threads: every clone counts against NPROC and the uid already has
    # this process (root is exempt, hence the uid drop above).
    _set_limit(resource.RLIMIT_NPROC, 1, 1)
    os.chdir(cwd)
    os.environ["HOME"] = cwd
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


def child_env(env: Mapping[str, str], cwd: str) -> dict[str, str]:
    """The child's whole environment, built explicitly (the server's own ``os.environ`` is
    never modified): ``env`` scrubbed, the services root on PYTHONPATH, HOME the temp dir."""
    out = scrub_env(env)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out["PYTHONPATH"] = os.pathsep.join(
        p for p in (root, out.get("PYTHONPATH", "")) if p
    )
    out["HOME"] = cwd
    out["PYTHONDONTWRITEBYTECODE"] = "1"
    return out


def _malformed(reason: str) -> _Outcome:
    return _Outcome("died", f"it sent a malformed message ({reason})")


@dataclass(frozen=True)
class _Outcome:
    kind: str
    reason: str = ""


# ---------------------------------------------------------------------------
# The parent


def _nonfinite(value: Any, depth: int = 0) -> bool:
    """Whether ``value`` holds a NaN or an infinity (numbers, numeric arrays, lists, dicts)."""
    if depth > 64:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, np.integer)):
        return False
    if isinstance(value, (float, np.floating)):
        return not math.isfinite(float(value))
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "fc" and not bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return any(_nonfinite(v, depth + 1) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_nonfinite(v, depth + 1) for v in value)
    return False


def _parse(msg: Any) -> tuple[str, Any]:
    """A child message as ``("done", fields)`` or ``("call", (name, args, kwargs))``; a
    ValueError names what is wrong with anything else."""
    if not isinstance(msg, list) or not msg or not isinstance(msg[0], str):
        raise ValueError("not a [kind, ...] list")
    if msg[0] == "done":
        if len(msg) != 2 or not isinstance(msg[1], dict):
            raise ValueError("done without its fields")
        done = msg[1]
        for k in ("stdout", "stderr"):
            if not isinstance(done.get(k, ""), str):
                raise ValueError(f"done.{k} is not a string")
        if not isinstance(done.get("traceback"), (str, type(None))):
            raise ValueError("done.traceback is not a string")
        result = jsonable(done.get("result"))
        if len(json.dumps(result).encode("utf-8")) > OUTPUT_CAP:
            result = {"truncated": f"RESULT is over {OUTPUT_CAP} bytes; return less"}
        return "done", {
            "stdout": done.get("stdout", ""),
            "stderr": done.get("stderr", ""),
            "traceback": done.get("traceback"),
            "result": result,
        }
    if msg[0] == "call":
        if len(msg) != 4:
            raise ValueError("call is not [call, name, args, kwargs]")
        _, name, args, kwargs = msg
        if not isinstance(name, str) or not isinstance(args, list):
            raise ValueError("call name / args of the wrong type")
        if not isinstance(kwargs, dict):
            raise ValueError("call kwargs is not an object")
        return "call", (name, tuple(args), kwargs)
    raise ValueError(f"unknown message kind {msg[0][:40]!r}")


class CodeRunner:
    """``code.api`` / ``code.run`` over a facade's primitives.

    ``stop_requested`` is the facade's; ``on_timeout`` is called once when a run hits its
    wall-clock timeout (the facade issues a stop to the robot); ``begin`` / ``finish`` bracket
    a run for the facade's bookkeeping (``finish`` returns fields merged into the result:
    steps taken, the latest observation, frames). :attr:`active` is true while a run
    executes (the facade refuses other business calls meanwhile).
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
        self._proc: subprocess.Popen | None = None
        self._uid: int | None = None
        self._abort = threading.Event()
        self._active = False
        # The floor of the isolation: no process of this uid may read this one's memory or
        # /proc/<pid>/environ (the server's API keys), whether or not the uid drop happens.
        _prctl_dumpable_off()

    @property
    def active(self) -> bool:
        """Whether a run is executing (its program may be calling the server)."""
        return self._active

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
            proc, uid = self._proc, self._uid
        if proc is not None:
            kill_tree(proc.pid, uid)

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
        """Run ``code`` in a fresh subprocess; see the module docstring for the limits.

        Result: ``status`` (``ran``, ``error``: the program raised, was stopped or broke the
        pipe protocol, ``timeout``), ``stdout``, ``stderr`` (8 KB each), ``traceback``,
        ``error`` (its last line), ``result`` (the program's ``RESULT`` variable, JSON-able),
        ``calls`` (the primitive log), ``n_calls``, ``move_m``, ``limit`` (the budget that
        refused a call, if any), ``cancelled`` (a stop arrived), ``stop_issued`` (a timeout
        stopped the robot), ``ms``, plus ``finish``'s fields (always, whatever the child did).
        """
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string")
        timeout_s = float(timeout_s)
        if not (timeout_s > 0) or not math.isfinite(timeout_s):
            raise ValueError("timeout_s must be positive")
        if max_move_m is not None and not (float(max_move_m) >= 0):
            raise ValueError("max_move_m must be a non-negative number")
        allowed = self.primitives(tier, privileged)
        names = [p.name for p in allowed]
        helper_names = list(HELPERS) if helpers else []
        started = time.monotonic()
        state = _Run(
            max_calls=int(max_calls),
            max_move_m=max_move_m,
            deadline=started + timeout_s,
        )
        uid = sandbox_uid()
        _prctl_dumpable_off()
        cwd = tempfile.mkdtemp(prefix="run_code-")
        if uid is not None:
            os.chown(cwd, uid, uid)
        parent_conn, child_conn = multiprocessing.Pipe()
        setup = {
            "code": code,
            "names": names,
            "helpers": helper_names,
            "limits": {
                "as_bytes": RLIMIT_AS_BYTES,
                "cpu_s": int(math.ceil(timeout_s)) + 2,
                "fsize_bytes": RLIMIT_FSIZE_BYTES,
            },
            "cwd": cwd,
            "uid": uid,
        }
        if self._begin:
            self._begin()
        with self._lock:
            self._abort.clear()
            self._active = True
            self._uid = uid
        proc: subprocess.Popen | None = None
        outcome = _Outcome("died", "it did not start")
        try:
            fd = child_conn.fileno()
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from pi_embodied_services.utils.code_exec import _child_entry; "
                    f"_child_entry({fd})",
                ],
                pass_fds=(fd,),
                env=child_env(os.environ, cwd),
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                start_new_session=True,
            )
            with self._lock:
                self._proc = proc
            child_conn.close()
            if not self._abort.is_set():
                parent_conn.send_bytes(json.dumps(setup).encode("utf-8"))
            outcome = self._serve(parent_conn, proc, allowed, state)
        except Exception as exc:  # never lose the run's accounting (finish below)
            outcome = _Outcome("died", f"{type(exc).__name__}: {exc}"[:500])
        finally:
            kill_tree(proc.pid if proc is not None else None, uid)
            if proc is not None:
                with contextlib.suppress(Exception):
                    proc.kill()
                with contextlib.suppress(Exception):
                    proc.wait(5)
            with self._lock:
                self._proc = None
                self._active = False
            parent_conn.close()
            child_conn.close()
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
        if outcome.kind == "timeout":
            result["status"] = "timeout"
            result["error"] = (
                f"the program ran past its {timeout_s:g} s timeout and was killed"
            )
            result["stop_issued"] = True
            self._issue_stop(state)
        elif outcome.kind == "oversized":
            result["status"] = "error"
            result["error"] = (
                f"the program sent a message over {MAX_MESSAGE >> 20} MB and was killed"
            )
        elif outcome.kind == "cancelled":
            result["status"] = "error"
            result["cancelled"] = True
            result["error"] = "stopped: the run was aborted"
        elif outcome.kind == "died":
            result["status"] = "error"
            rc = proc.returncode if proc is not None else None
            result["error"] = (
                f"the code process was killed: {outcome.reason}"
                if outcome.reason
                else f"the code process exited unexpectedly (exit code {rc})"
            )
        elif done.get("traceback"):
            result["status"] = "error"
            lines = done["traceback"].rstrip().splitlines()
            result["error"] = (lines[-1] if lines else "the program failed")[:2000]
        if self._finish:
            result.update(self._finish())
        return result

    def _issue_stop(self, state: _Run) -> None:
        """Stop the robot once per run (the timeout: from the timer inside a primitive, or
        after the child was killed)."""
        with self._lock:
            if state.stop_issued:
                return
            state.stop_issued = True
        if self._on_timeout:
            self._on_timeout()

    def _gone(self, proc: subprocess.Popen) -> _Outcome:
        return _Outcome("cancelled" if self._abort.is_set() else "died")

    def _serve(
        self, conn, proc: subprocess.Popen, allowed: list[Primitive], state: _Run
    ) -> _Outcome:
        """Answer the child's primitive calls until it is done, times out, is aborted, dies or
        breaks the protocol (the child's failure: the run's accounting is kept)."""
        by_name = {p.name: p for p in allowed}
        deadline = state.deadline
        while True:
            now = time.monotonic()
            if state.stop_issued or now >= deadline:
                return _Outcome("timeout")
            if self._abort.is_set():
                return _Outcome("cancelled")
            try:
                ready = conn.poll(min(0.05, deadline - now))
            except (EOFError, OSError):
                return self._gone(proc)
            if not ready:
                if proc.poll() is not None:
                    return self._gone(proc)
                continue
            try:
                raw = conn.recv_bytes(MAX_MESSAGE)
            except (EOFError, OSError) as exc:
                if "bad message length" in str(exc):
                    return _Outcome("oversized")
                return self._gone(proc)
            try:
                kind, payload = _parse(decode(json.loads(raw.decode("utf-8"))))
            except (ValueError, TypeError, RecursionError, MemoryError) as exc:
                return _malformed(f"{type(exc).__name__}: {exc}"[:200])
            if kind == "done":
                state.done = payload
                return _Outcome("done")
            name, args, kwargs = payload
            try:
                reply = self._call(by_name, name, args, kwargs, state)
            except RecursionError:
                return _malformed("arguments nested too deeply")
            # The timeout's stop may have aborted this run too: it is still a timeout.
            if state.stop_issued or time.monotonic() >= deadline:
                return _Outcome("timeout")
            if self._abort.is_set():
                return _Outcome("cancelled")
            try:
                data = json.dumps(encode(reply)).encode("utf-8")
            except (TypeError, ValueError, RecursionError) as exc:
                why = f"{name} returned a value that cannot cross the pipe: {exc}"[:500]
                state.log[-1]["error"] = why
                data = json.dumps(["error", why]).encode("utf-8")
            try:
                conn.send_bytes(data)
            except (BrokenPipeError, OSError):
                return self._gone(proc)

    def _call(
        self, by_name: dict[str, Primitive], name: str, args, kwargs, state: _Run
    ):
        entry: dict[str, Any] = {
            "name": name[:LOG_STR_CAP],
            "args": jsonable(list(args), 64, LOG_STR_CAP),
            "kwargs": jsonable(dict(kwargs), 64, LOG_STR_CAP),
        }
        state.log.append(entry)
        prim = by_name.get(name)
        if prim is None:
            entry["error"] = f"{name[:LOG_STR_CAP]} is not a primitive of this tier"
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
        # Before any accounting: a NaN would pass every budget comparison (and poison the
        # accumulated translation, so that every later move passed the cap too).
        if _nonfinite(args) or _nonfinite(kwargs):
            entry["error"] = (
                f"ValueError: {name}() got a non-finite number (NaN or inf)"
            )
            return ("error", entry["error"])
        if prim.check is not None:
            try:
                prim.check(args, kwargs)
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"[:2000]
                return ("error", entry["error"])
        move = 0.0
        if prim.move_m is not None:
            try:
                move = float(prim.move_m(args, kwargs))
            except Exception as exc:  # a malformed argument: the primitive reports it
                entry["error"] = f"{type(exc).__name__}: {exc}"[:2000]
                return ("error", entry["error"])
            if not math.isfinite(move) or move < 0:
                entry["error"] = (
                    f"ValueError: {name}() commands a non-finite translation"
                )
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
        # runs, the robot gets its stop now (the primitive returns through it), not after;
        # and every outbound RPC it makes is bounded by what is left (utils/rpc/deadline.py).
        timer = threading.Timer(
            max(0.0, state.deadline - t0), self._issue_stop, args=(state,)
        )
        timer.daemon = True
        timer.start()
        try:
            with call_deadline(state.deadline):
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
    "child_env",
    "kill_tree",
    "sandbox_uid",
    "scrub_env",
    "strip_examples",
]
