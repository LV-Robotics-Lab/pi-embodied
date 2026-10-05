"""Single-machine hardware lock: one env server per physical arm.

OpenETA's real-robot MCP (real/mcp/observation_core.py ``RealEnvManager._acquire_lock``) holds a
non-blocking ``fcntl.flock`` on one lock file while its env exists, so two agents on one machine
cannot drive the bench at once. Here each device has its own lock, ``<lock dir>/<id>.lock``: the
arm by its address (``franka:172.16.0.2``, ``ur5e:192.168.1.10``, ``piper:<CAN serial>``) and,
when the config names one, by its serial as well (``robot.serial`` / ``calibration.arm_id``, the
same id whichever backend drives it: an RLinf and a Polymetis server of one Franka collide on it
even when they address the arm differently), and every camera device the config opens
(``camera:<serial or /dev path>``). The address lock is never dropped for the serial one: a server
whose config names no serial still collides with one whose config does. Two servers for different
devices run side by side; a second server for a held one is refused at startup, before it
touches the hardware, naming the holder (pid, server, time). The kernel drops a lock when its
holder exits, crashed or not.

The lock directory is ``--lock-dir``, else ``$PI_EMBODIED_LOCK_DIR``, else
``/tmp/pi-embodied-locks``: every server on the machine must use the same one. It must be a plain
directory owned by this user (or root) and not group/world-writable; lock files are opened with
O_NOFOLLOW; a lock file another user owns reads as the device being in use.
"""

from __future__ import annotations

import errno
import fcntl
import os
import re
import stat
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


def check_dir(root: Path) -> None:
    """Refuse a lock directory another user could swap files in: it must be a real directory
    (not a symlink) owned by this user or root, and not group- or world-writable."""
    st = os.lstat(root)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise RuntimeError(f"hardware lock directory {root} is not a plain directory")
    if st.st_uid not in (os.geteuid(), 0):
        raise RuntimeError(
            f"hardware lock directory {root} belongs to uid {st.st_uid}; set --lock-dir or ${LOCK_DIR_ENV}"
        )
    if st.st_mode & 0o022:
        raise RuntimeError(
            f"hardware lock directory {root} is group/world-writable (mode {oct(st.st_mode & 0o777)}); chmod go-w it"
        )


def acquire(
    arm_ids: Iterable[str], *, directory: str | None = None, holder: str = ""
) -> HardwareLock:
    """Lock every id in ``arm_ids`` (arms and cameras; non-blocking), or none: raises
    ``RobotBusyError`` naming the first one another process holds. Lock files are opened with
    O_NOFOLLOW (a planted symlink is refused); a lock file this user may not open is another
    user's server holding the device, reported as in use."""
    ids = list(dict.fromkeys(a for a in arm_ids if a))
    if not ids:
        raise ValueError("no arm id to lock")
    root = lock_dir(directory)
    root.mkdir(mode=0o755, parents=True, exist_ok=True)
    check_dir(root)
    held: dict[str, int] = {}
    try:
        for arm in ids:
            try:
                fd = os.open(
                    root / _file_name(arm),
                    os.O_RDWR
                    | os.O_CREAT
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    0o644,
                )
            except PermissionError:
                raise RobotBusyError(
                    arm, "its lock file belongs to another user"
                ) from None
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise RuntimeError(
                        f"lock file for {arm} is a symlink; refusing it"
                    ) from None
                raise
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
                if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EPERM):
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


SERIAL_KEYS = r"serial|serial_number|arm_id"


def arm_serials(kind: str, tree: Any) -> list[str]:
    """``<kind>:<serial>`` of every arm in a robot config that names its serial (``serial``,
    ``serial_number`` or ``arm_id`` in ``robot`` / ``robot.arms.*`` / ``calibration``); the
    serial is the same whichever backend drives the arm, so RLinf and Polymetis servers of one
    Franka lock the same id."""
    found: list[str] = []
    if not isinstance(tree, Mapping):
        return found
    robot = tree.get("robot") if isinstance(tree.get("robot"), Mapping) else {}
    arms = robot.get("arms") if isinstance(robot.get("arms"), Mapping) else {}
    for part in (robot, tree.get("calibration"), *arms.values()):
        if not isinstance(part, Mapping):
            continue
        for k, v in part.items():
            if (
                re.fullmatch(SERIAL_KEYS, str(k))
                and isinstance(v, (str, int))
                and str(v).strip()
            ):
                found.append(f"{kind}:{str(v).strip()}")
    return list(dict.fromkeys(found))


def camera_ids(tree: Any) -> list[str]:
    """``camera:<serial or device>`` of every camera device in a robot config (the ``cameras``
    subtree: realsense serials, webcam devices, ``/dev`` paths), so two servers never open one
    camera. Empty serials (the first device) and URLs are skipped."""
    found: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, Mapping):
            for k, x in v.items():
                if (
                    k in ("serial", "device")
                    and isinstance(x, (str, int))
                    and str(x).strip()
                ):
                    value = str(x).strip()
                    if "://" not in value:
                        found.append(f"camera:{value}")
                else:
                    walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    if isinstance(tree, Mapping):
        walk(tree.get("cameras"))
    return list(dict.fromkeys(found))


def read_config(path: Any) -> dict[str, Any]:
    """The robot YAML at ``path`` as a dict, or {} when there is none to read."""
    if not path:
        return {}
    try:
        import yaml

        data = yaml.safe_load(Path(str(path)).expanduser().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def hardware_ids(kind: str, tree: Any, addresses: Iterable[str] = ()) -> list[str]:
    """What a server of ``kind`` locks: its arms by serial when the config names one AND by
    ``addresses`` (or the config's addresses), plus its camera devices. The serial ids come first;
    the address ids are kept alongside, never replaced, so a server configured by serial and one
    configured by address alone collide on the address, and a dual-arm config naming one arm's
    serial still locks the other arm's address."""
    robot = tree.get("robot") if isinstance(tree, Mapping) else None
    arms = [
        *arm_serials(kind, tree),
        *(list(addresses) or config_arm_ids(kind, robot or {})),
    ]
    return list(dict.fromkeys([*arms, *camera_ids(tree)]))


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
        "--lock-id",
        action="append",
        default=[],
        help="Physical arm to lock, e.g. franka:172.16.0.2 (repeatable; default: the config's arm addresses)",
    )
    parser.add_argument(
        "--lock-dir",
        default="",
        help=f"Hardware lock directory (default ${LOCK_DIR_ENV}, else {DEFAULT_LOCK_DIR})",
    )


def lock_from_args(args: Any, derived: Iterable[str], server: str) -> HardwareLock:
    """Lock ``--lock-id`` (when given) or the ids derived from the config; exits with the
    refusal on stderr when an arm is busy."""
    ids = list(getattr(args, "lock_id", None) or []) or list(derived)
    try:
        return acquire(
            ids, directory=getattr(args, "lock_dir", "") or None, holder=server
        )
    except RobotBusyError as exc:
        print(f"[{server}] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(3) from None
