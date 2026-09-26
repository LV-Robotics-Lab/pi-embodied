"""Single-machine hardware lock: one env server per physical arm.

OpenETA's real-robot MCP (real/mcp/observation_core.py ``RealEnvManager._acquire_lock``) holds a
non-blocking ``fcntl.flock`` on one lock file while its env exists, so two agents on one machine
cannot drive the bench at once. Here the lock is per arm: ``<lock dir>/<arm id>.lock``, where the
arm id names the physical arm (``franka:172.16.0.2``, ``ur5e:192.168.1.10``, ``piper:<CAN serial>``),
so two servers for different arms run side by side while a second server for the same arm (a
single-arm and a dual-arm server sharing an arm included) is refused at startup, before it touches
the hardware. The file records the holder (pid, server, time) for the refusal message; the kernel
drops the lock when the holder exits, crashed or not.

The lock directory is ``--lock-dir``, else ``$PI_EMBODIED_LOCK_DIR``, else
``/tmp/pi-embodied-locks``: every server on the machine must use the same one.
"""

from __future__ import annotations

import errno
import fcntl
import os
import re
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

DEFAULT_LOCK_DIR = "/tmp/pi-embodied-locks"
LOCK_DIR_ENV = "PI_EMBODIED_LOCK_DIR"


class RobotBusyError(RuntimeError):
    """Another process holds the lock of an arm this server would drive."""

    def __init__(self, arm_id: str, holder: str) -> None:
        self.arm_id = arm_id
        self.holder = holder
        super().__init__(
            f"robot busy: arm {arm_id} is driven by another env server ({holder or 'unknown holder'}); "
            "stop it first"
        )


def lock_dir(value: str | None = None) -> Path:
    return Path(value or os.environ.get(LOCK_DIR_ENV) or DEFAULT_LOCK_DIR)


def _file_name(arm_id: str) -> str:
    return re.sub(r"[^\w.:-]+", "_", arm_id).replace(":", "_") + ".lock"


class HardwareLock:
    """The held locks of one server; released by ``release()`` or by process exit."""

    def __init__(self, fds: dict[str, int]) -> None:
        self._fds = fds

    @property
    def arm_ids(self) -> list[str]:
        return list(self._fds)

    def release(self) -> None:
        for fd in self._fds.values():
            try:
                os.ftruncate(fd, 0)
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            finally:
                os.close(fd)
        self._fds = {}

    def __enter__(self) -> HardwareLock:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def acquire(
    arm_ids: Iterable[str], *, directory: str | None = None, holder: str = ""
) -> HardwareLock:
    """Lock every arm in ``arm_ids`` (non-blocking), or none: raises ``RobotBusyError``
    naming the first arm another process holds."""
    ids = list(dict.fromkeys(a for a in arm_ids if a))
    if not ids:
        raise ValueError("no arm id to lock")
    root = lock_dir(directory)
    root.mkdir(parents=True, exist_ok=True)
    held: dict[str, int] = {}
    try:
        for arm in ids:
            fd = os.open(root / _file_name(arm), os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                prev = ""
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    try:
                        prev = os.read(fd, 4096).decode("utf-8", "replace").strip()
                    except OSError:
                        pass
                os.close(fd)
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise RobotBusyError(arm, prev) from None
                raise
            os.ftruncate(fd, 0)
            os.write(
                fd,
                f"pid={os.getpid()} arm={arm} {holder} t={time.time():.0f}".encode(),
            )
            os.fsync(fd)
            held[arm] = fd
    except BaseException:
        HardwareLock(held).release()
        raise
    return HardwareLock(held)


ADDRESS_KEYS = r"(\w+_)?robot_ip|ip|nuc_ip"


def config_arm_ids(kind: str, tree: Any, keys: str = ADDRESS_KEYS) -> list[str]:
    """``<kind>:<address>`` for every arm address in a (nested) robot config: the values of keys
    matching ``keys`` (default ``ip``, ``robot_ip``, ``*_robot_ip``, ``nuc_ip``), in order, deduplicated."""
    found: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, Mapping):
            for k, x in v.items():
                if isinstance(k, str) and re.fullmatch(keys, k):
                    if isinstance(x, (str, int)) and str(x).strip():
                        found.append(f"{kind}:{str(x).strip()}")
                else:
                    walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(tree)
    return list(dict.fromkeys(found))


def add_lock_arguments(parser: Any) -> None:
    parser.add_argument(
        "--arm-id",
        action="append",
        default=[],
        help="Physical arm id to lock (repeatable; default: derived from the robot config)",
    )
    parser.add_argument(
        "--lock-dir",
        default="",
        help=f"Hardware lock directory (default ${LOCK_DIR_ENV}, else {DEFAULT_LOCK_DIR})",
    )


def lock_from_args(args: Any, derived: Iterable[str], server: str) -> HardwareLock:
    """Lock ``--arm-id`` (when given) or the ids derived from the config; exits with the
    refusal on stderr when an arm is busy."""
    ids = list(getattr(args, "arm_id", None) or []) or list(derived)
    try:
        return acquire(
            ids, directory=getattr(args, "lock_dir", "") or None, holder=server
        )
    except RobotBusyError as exc:
        print(f"[{server}] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(3) from None
