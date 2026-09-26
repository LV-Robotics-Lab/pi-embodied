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

"""Code mode's runner (``code.api`` / ``code.run``) over toy primitives, no simulator: the
child holds no env, budgets refuse, a timeout kills and stops, an abort kills."""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from pi_embodied_services.utils import code_exec
from pi_embodied_services.utils.code_exec import (
    OUTPUT_CAP,
    CodeRunner,
    Primitive,
    decode,
    encode,
    jsonable,
    scrub_env,
    strip_examples,
)


class Toy:
    """A toy arm: `move` translates, `look` returns an image, `slow` sleeps in the parent."""

    def __init__(self):
        self.pos = np.zeros(3)
        self.stopped = False
        self.calls: list[str] = []
        self.stop_flag = False

    def move(self, dxyz) -> dict:
        """Move by dxyz.

        Args:
            dxyz: [dx, dy, dz] in metres.

        Example:
            >>> move([0.1, 0, 0])
        """
        self.calls.append("move")
        self.pos = self.pos + np.asarray(dxyz, dtype=np.float64)
        return {"pos": self.pos.tolist()}

    def look(self) -> dict:
        """The camera image and the position."""
        self.calls.append("look")
        return {"rgb": np.zeros((4, 4, 3), dtype=np.uint8), "pos": self.pos}

    def slow(self, seconds: float) -> dict:
        """Sleep in the parent, honouring a stop between ticks."""
        self.calls.append("slow")
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.stop_flag:
                return {"cancelled": True}
            time.sleep(0.01)
        return {"slept": seconds}

    def boom(self) -> None:
        """Always fails."""
        raise ValueError("bad thing")


def runner(toy: Toy, **kw) -> CodeRunner:
    def move_m(args, kwargs):
        d = kwargs.get("dxyz", args[0] if args else None)
        return float(np.linalg.norm(np.asarray(d, dtype=np.float64)))

    return CodeRunner(
        [
            Primitive("move", toy.move, ("high", "low"), move_m=move_m),
            Primitive("look", toy.look, ("high",)),
            Primitive("slow", toy.slow, ("high", "low")),
            Primitive("boom", toy.boom, ("high",)),
            Primitive("cheat", toy.look, ("high", "low"), privileged=True),
        ],
        stop_requested=lambda: toy.stop_flag,
        on_timeout=lambda: setattr(toy, "stopped", True),
        **kw,
    )


def test_api_lists_the_tier_with_signatures_docs_and_helpers():
    r = runner(Toy())
    high = r.api("high")
    assert [p["name"] for p in high] == ["move", "look", "slow", "boom"]
    move = high[0]
    assert move["signature"] == "(dxyz) -> dict"
    assert "Example:" in move["doc"]
    assert [p["name"] for p in r.api("low")] == ["move", "slow"]
    assert "Example:" not in r.api("low-noexamples")[0]["doc"]
    assert "Args:" in r.api("low-noexamples")[0]["doc"]
    assert [p["name"] for p in r.api("low", privileged=True)] == [
        "move",
        "slow",
        "cheat",
    ]
    helpers = [p["name"] for p in r.api("low", helpers=True) if p["kind"] == "helper"]
    assert "normalize_vector" in helpers and len(helpers) == 9
    with pytest.raises(ValueError, match="unknown code tier"):
        r.api("medium")


def test_strip_examples_keeps_the_other_sections():
    doc = "Move.\n\nArgs:\n    x: a\n\nExample:\n    >>> move(1)\n    >>> move(2)\n\nReturns:\n    None"
    assert strip_examples(doc) == "Move.\n\nArgs:\n    x: a\n\nReturns:\n    None"


def test_the_program_runs_in_another_process_without_an_env_and_returns_result():
    toy = Toy()
    out = runner(toy).run(
        "import os\n"
        "print('hello')\n"
        "r = move([0.1, 0, 0])\n"
        "img = look()['rgb']\n"
        "RESULT = {'pid': os.getpid(), 'env': 'env' in globals() or '_env' in globals(),"
        " 'self': 'self' in globals(), 'pos': r['pos'], 'shape': list(img.shape),"
        " 'np': np.array([1, 2]).sum()}\n",
        timeout_s=20,
    )
    assert out["status"] == "ran", out
    assert out["result"]["pid"] != os.getpid(), "the program ran in a subprocess"
    assert out["result"]["env"] is False and out["result"]["self"] is False
    assert out["result"]["pos"] == [0.1, 0, 0]
    assert out["result"]["shape"] == [4, 4, 3]
    assert out["result"]["np"] == 3
    assert out["stdout"] == "hello\n"
    assert toy.calls == ["move", "look"]
    assert [c["name"] for c in out["calls"]] == ["move", "look"]
    assert out["calls"][0]["move_m"] == pytest.approx(0.1)
    assert out["n_calls"] == 2 and out["move_m"] == pytest.approx(0.1)
    assert out["error"] is None and out["traceback"] is None


def test_only_the_tiers_primitives_exist_in_the_child():
    toy = Toy()
    out = runner(toy).run("look()", tier="low", timeout_s=20)
    assert out["status"] == "error"
    assert "NameError" in out["error"]
    assert toy.calls == []
    out = runner(toy).run("RESULT = cheat()['pos'].tolist()", timeout_s=20)
    assert "NameError" in out["error"]
    out = runner(toy).run(
        "RESULT = cheat()['pos'].tolist()", privileged=True, timeout_s=20
    )
    assert out["result"] == [0, 0, 0]


def test_an_exception_is_an_error_with_its_traceback_and_a_primitive_error_reaches_the_program():
    out = runner(Toy()).run("move([0, 0, 0])\nraise KeyError('x')", timeout_s=20)
    assert out["status"] == "error"
    assert out["error"] == "KeyError: 'x'"
    assert '<run_code>", line 2' in out["traceback"]
    out = runner(Toy()).run(
        "try:\n    boom()\nexcept RuntimeError as e:\n    RESULT = str(e)\n",
        timeout_s=20,
    )
    assert out["status"] == "ran"
    assert out["result"] == "ValueError: bad thing"
    assert out["calls"][0]["error"] == "ValueError: bad thing"


def test_an_infinite_loop_is_killed_at_the_timeout_and_the_robot_is_stopped():
    toy = Toy()
    out = runner(toy).run("while True:\n    pass\n", timeout_s=0.5)
    assert out["status"] == "timeout"
    assert out["stop_issued"] is True and toy.stopped is True
    assert "0.5 s timeout" in out["error"]
    assert out["ms"] < 5000


def test_an_abort_kills_the_child_and_ends_the_running_primitive():
    toy = Toy()
    r = runner(toy)
    box: dict = {}

    def go():
        box["out"] = r.run("slow(30)\nRESULT = 'never'", timeout_s=60)

    t = threading.Thread(target=go)
    t.start()
    for _ in range(200):
        if toy.calls:
            break
        time.sleep(0.05)
    assert toy.calls == ["slow"], "the primitive started"
    # pi's abort: `stop` on the server sets the stop generation (the primitive returns) and
    # the facade's _on_stop kills the child.
    toy.stop_flag = True
    r.abort()
    t.join(10)
    assert not t.is_alive()
    out = box["out"]
    assert out["status"] == "error" and out["cancelled"] is True
    assert out["result"] is None
    assert out["calls"][0].get("cancelled") is True


def test_a_stop_before_a_call_refuses_it_and_ends_the_run():
    toy = Toy()
    toy.stop_flag = True
    out = runner(toy).run("move([0.1, 0, 0])\nRESULT = 1", timeout_s=20)
    assert out["status"] == "error" and out["cancelled"] is True
    assert toy.calls == []
    assert "stopped" in out["calls"][0]["error"]


def test_the_call_budget_and_the_move_cap_refuse():
    toy = Toy()
    code = (
        "log = []\n"
        "for i in range(4):\n"
        "    try:\n"
        "        move([0.03, 0, 0])\n"
        "        log.append('ok')\n"
        "    except Exception as e:\n"
        "        log.append(type(e).__name__ + ': ' + str(e))\n"
        "RESULT = log\n"
    )
    out = runner(toy).run(code, timeout_s=20, max_calls=2)
    assert out["status"] == "ran" and out["limit"] == "max_calls"
    assert out["result"][:2] == ["ok", "ok"]
    assert out["result"][2].startswith("CodeLimitError: call budget exhausted")
    assert toy.calls == ["move", "move"]
    assert out["calls"][2]["refused"]
    toy = Toy()
    out = runner(toy).run(code, timeout_s=20, max_move_m=0.07)
    assert out["limit"] == "max_move_m"
    assert out["result"][:2] == ["ok", "ok"]
    assert "move budget exhausted" in out["result"][2]
    assert toy.pos[0] == pytest.approx(0.06)
    assert out["move_m"] == pytest.approx(0.06)


def test_stdout_and_stderr_are_capped_and_the_result_is_json_able():
    out = runner(Toy()).run(
        "import sys\nprint('x' * 20000)\nprint('e' * 20000, file=sys.stderr)\n"
        "RESULT = {'a': np.arange(3), 'b': np.float32(1.5), 'c': (1, 2), 'd': object()}\n",
        timeout_s=20,
    )
    assert out["status"] == "ran"
    assert len(out["stdout"].encode()) < OUTPUT_CAP + 100
    assert out["stdout"].endswith("bytes]")
    assert out["stderr"].endswith("bytes]")
    assert out["result"]["a"] == [0, 1, 2]
    assert out["result"]["b"] == 1.5
    assert out["result"]["c"] == [1, 2]
    assert out["result"]["d"].startswith("<object object")


def test_helpers_are_injected_only_when_asked():
    out = runner(Toy()).run(
        "RESULT = normalize_vector(np.array([3.0, 4.0])).tolist()",
        helpers=True,
        timeout_s=20,
    )
    assert out["result"] == [0.6, 0.8]
    out = runner(Toy()).run("RESULT = normalize_vector([1, 0])", timeout_s=20)
    assert "NameError" in out["error"]


def test_begin_and_finish_bracket_a_run_and_extend_the_result():
    seen: list[str] = []
    r = runner(Toy(), begin=lambda: seen.append("begin"), finish=lambda: {"steps": 7})
    out = r.run("RESULT = 1", timeout_s=20)
    assert seen == ["begin"] and out["steps"] == 7


def test_run_refuses_bad_arguments():
    with pytest.raises(ValueError):
        runner(Toy()).run("", timeout_s=1)
    with pytest.raises(ValueError):
        runner(Toy()).run("RESULT = 1", timeout_s=0)
    with pytest.raises(ValueError):
        runner(Toy()).run("RESULT = 1", tier="nope")


def test_jsonable_caps_large_arrays():
    assert jsonable(np.zeros((100, 100)))["shape"] == [100, 100]
    assert jsonable({"k": np.int64(3)}) == {"k": 3}


# ---- the sandbox --------------------------------------------------------------------------


def test_the_pipe_is_json_arrays_round_trip_and_object_arguments_are_refused():
    toy = Toy()
    # A primitive's array reaches the program as an ndarray; the program's array reaches the
    # primitive as one (dtype kept); a tuple is a list on the other side.
    out = runner(toy).run(
        "img = look()['rgb']\n"
        "assert type(img).__name__ == 'ndarray' and img.dtype == np.uint8, img\n"
        "move(np.array([0.5, 0, 0], dtype=np.float32))\n"
        "RESULT = [img.shape, look()['pos'].tolist()]\n"
    )
    assert out["status"] == "ran", out
    assert out["result"] == [[4, 4, 3], [0.5, 0.0, 0.0]]
    assert np.allclose(toy.pos, [0.5, 0, 0])
    # An object is not JSON and never becomes a string behind the program's back.
    out = runner(toy).run("class X: pass\nmove(X())\n")
    assert out["status"] == "error" and "cannot cross the pipe" in out["traceback"]
    assert toy.calls.count("move") == 1


def test_encode_and_decode_refuse_object_arrays_and_malformed_stubs():
    a = np.arange(6, dtype=np.int16).reshape(2, 3)
    wire = encode({"a": a, "t": (1, 2), "s": np.float32(1.5)})
    assert wire["t"] == [1, 2] and wire["s"] == 1.5
    back = decode(wire)
    assert np.array_equal(back["a"], a) and back["a"].dtype == np.int16
    with pytest.raises(TypeError, match="dtype object"):
        encode(np.array([object()]))
    for bad in (
        {"__ndarray__": "object", "shape": [1], "data": ""},
        {"__ndarray__": "int8", "shape": [5], "data": "AAA="},
        {"__ndarray__": "int8", "shape": "x", "data": "AAA="},
    ):
        with pytest.raises(ValueError):
            decode(bad)
    # No pickle anywhere: a message is bytes of JSON.
    src = open(code_exec.__file__).read()
    assert (
        "conn.recv()" not in src
        and "conn.send(" not in src
        and "import pickle" not in src
    )


def test_the_child_sees_no_secret_environment_variables(monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "sk-1")
    monkeypatch.setenv("SOME_TOKEN", "t")
    monkeypatch.setenv("AWS_REGION", "eu")
    monkeypatch.setenv("PLAIN_SETTING", "keep")
    assert set(scrub_env({"A_KEY": "", "Path": "", "auth_x": ""})) == {"Path"}
    out = runner(Toy()).run(
        "import os\n"
        "RESULT = [k for k in ('FAKE_API_KEY', 'SOME_TOKEN', 'AWS_REGION', 'PLAIN_SETTING', 'PATH')"
        " if k in os.environ]\n"
        # /proc exists on Linux only; elsewhere the scrubbed os.environ is the whole check.
        # Under a dropped uid the child is non-dumpable: even its own environ is unreadable.
        "p = '/proc/self/environ'\n"
        "try:\n"
        "    RESULT.append(open(p, 'rb').read().count(b'sk-1') if os.path.exists(p) else 0)\n"
        "except PermissionError:\n"
        "    RESULT.append(0)\n"
    )
    assert out["status"] == "ran", out
    assert out["result"] == ["PLAIN_SETTING", "PATH", 0]
    # The parent keeps them.
    assert os.environ["FAKE_API_KEY"] == "sk-1"


def test_the_programs_subprocesses_die_with_the_run():
    out = runner(Toy()).run(
        "import subprocess\n"
        "try:\n"
        "    p = subprocess.Popen(['sleep', '30'])\n"
        "    RESULT = p.pid\n"
        "except OSError:\n"
        "    RESULT = None  # RLIMIT_NPROC refused it (a non-root user)\n"
    )
    assert out["status"] == "ran", out
    pid = out["result"]
    if pid is not None:
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            os.kill(pid, 9)
            raise AssertionError("the program's subprocess survived the run")


def test_result_and_traceback_are_capped_and_oversized_messages_end_the_run(
    monkeypatch,
):
    out = runner(Toy()).run("RESULT = 'x' * 20000\n")
    assert out["status"] == "ran" and "over" in out["result"]["truncated"]
    # 300 distinct frames (Python collapses repeated ones): a traceback well over the cap.
    chain = "".join(f"def f{i}():\n    return f{i + 1}()\n" for i in range(300))
    out = runner(Toy()).run(f"{chain}def f300():\n    return 1 / 0\nf0()\n")
    assert out["status"] == "error"
    assert len(out["traceback"].encode()) < OUTPUT_CAP + 200
    assert out["traceback"].startswith("[truncated"), (
        "the tail is kept: it has the error"
    )
    assert out["error"] == "ZeroDivisionError: division by zero"
    monkeypatch.setattr(code_exec, "MAX_MESSAGE", 1 << 20)
    out = runner(Toy()).run("move(np.zeros(400000))\n")
    assert out["status"] == "error" and "over 1 MB" in out["error"]


def test_the_timeout_stops_the_robot_inside_a_running_primitive():
    toy = Toy()

    def stop():
        toy.stopped = True
        toy.stop_flag = True  # what the facade's stop does: the primitive returns

    r = CodeRunner(
        [Primitive("slow", toy.slow, ("high",))],
        stop_requested=lambda: toy.stop_flag,
        on_timeout=stop,
    )
    t0 = time.monotonic()
    out = r.run("slow(30)\nRESULT = 'never'", timeout_s=2)
    assert time.monotonic() - t0 < 5, (
        "the primitive returned through the stop, not after 30 s"
    )
    assert out["status"] == "timeout" and out["stop_issued"] is True
    assert out["calls"][0]["name"] == "slow" and out["result"] is None


# ---- the audit's findings (sandbox) ----------------------------------------------------------

LINUX = os.path.isdir("/proc/self") and hasattr(os, "getresuid")


def _pipe_of(name: str) -> str:
    """Program text that digs the stub's pipe out of its closure (a tampering program)."""
    return (
        f"c = [x.cell_contents for x in {name}.__closure__"
        " if hasattr(x.cell_contents, 'send_bytes')][0]\n"
    )


def test_a_nan_argument_is_refused_before_any_accounting_and_keeps_the_move_cap():
    toy = Toy()
    out = runner(toy).run(
        "errs = []\n"
        "for d in ([float('nan'), 0, 0], [float('inf'), 0, 0], np.array([0, np.nan, 0])):\n"
        "    try:\n"
        "        move(d)\n"
        "    except RuntimeError as e:\n"
        "        errs.append(str(e))\n"
        "for _ in range(5):\n"
        "    move([0.5, 0, 0])\n",
        max_move_m=1.0,
    )
    assert out["status"] == "error" and out["limit"] == "max_move_m", out
    assert toy.calls == ["move", "move"], "only the two finite moves within the cap ran"
    assert out["move_m"] == 1.0
    assert all("non-finite" in c["error"] for c in out["calls"][:3])


@pytest.mark.parametrize(
    "tamper, reason",
    [
        ('c.send_bytes(b\'["call", "move", 5, {}]\')\nc.recv_bytes()\n', "args"),
        (
            "c.send_bytes(b'[' * 200000 + b']' * 200000)\nc.recv_bytes()\n",
            "RecursionError",
        ),
        (
            'c.send_bytes(b\'["done", {"stdout": 5, "traceback": ""}]\')\n'
            "import time; time.sleep(2)\n",
            "stdout",
        ),
        ("c.send_bytes(b'{\"call\": 1}')\nc.recv_bytes()\n", "list"),
    ],
)
def test_a_malformed_message_is_the_childs_failure_and_the_accounting_is_kept(
    tamper, reason
):
    finished = []
    r = runner(Toy(), finish=lambda: finished.append(1) or {"steps": 3})
    out = r.run("move([0.01, 0, 0])\n" + _pipe_of("move") + tamper)
    assert finished == [1]
    assert out["status"] == "error" and out["steps"] == 3, out
    assert "malformed message" in out["error"] and reason in out["error"], out["error"]
    assert out["n_calls"] == 1 and out["move_m"] == 0.01


def test_a_result_that_cannot_cross_the_pipe_is_that_calls_error():
    r = CodeRunner([Primitive("names", lambda: np.array(["a", "b"]), ("high",))])
    out = r.run("try:\n    names()\nexcept RuntimeError as e:\n    RESULT = str(e)\n")
    assert out["status"] == "ran", out
    assert "cannot cross the pipe" in out["result"]
    assert "cannot cross the pipe" in out["calls"][0]["error"]


def test_the_runs_deadline_bounds_every_outbound_rpc_of_a_primitive():
    import http.server

    class Hang(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            time.sleep(10)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hang)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    from pi_embodied_services.utils.rpc.http_rpc import HttpRpcClient

    client = HttpRpcClient(f"http://127.0.0.1:{srv.server_address[1]}")

    def segment():
        # like env.segment: one SAM3 call with its own 120 s timeout, no stop polling
        return client.call("sam3.segment", timeout_s=120.0)

    r = CodeRunner([Primitive("segment", segment, ("high",))], on_timeout=lambda: None)
    t0 = time.monotonic()
    out = r.run("segment()\n", timeout_s=1.5)
    took = time.monotonic() - t0
    srv.shutdown()
    assert out["status"] == "timeout", out
    assert took < 4, f"the primitive outlived the run's timeout by {took - 1.5:.1f} s"


def test_the_servers_environment_is_never_modified_during_a_spawn(monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "sk-1")
    seen: list[str | None] = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            seen.append(os.environ.get("FAKE_API_KEY"))

    th = threading.Thread(target=watch)
    th.start()
    try:
        out = runner(Toy()).run("x = 1\n" + "#" * (2 << 20) + "\n")
    finally:
        stop.set()
        th.join()
    assert out["status"] == "ran", out
    assert seen and all(v == "sk-1" for v in seen)
    env = code_exec.child_env({"FAKE_API_KEY": "x", "PATH": "/bin"}, "/tmp/w")
    assert "FAKE_API_KEY" not in env and env["HOME"] == "/tmp/w"
    assert env["PYTHONPATH"].split(os.pathsep)[0].endswith("services")


@pytest.mark.skipif(not LINUX, reason="/proc and prctl are Linux-only")
def test_the_child_cannot_read_the_servers_environ_or_memory():
    # The secret is in the server's initial environment block (as pi's API keys are), which
    # is what /proc/<pid>/environ shows: start the server as a subprocess to get one.
    import subprocess
    import sys

    prog = (
        "import os, json\n"
        "from pi_embodied_services.utils.code_exec import CodeRunner, Primitive\n"
        "r = CodeRunner([Primitive('noop', lambda: {}, ('high',))])\n"
        'out = r.run("import os\\n'
        "pid = os.getppid()\\n"
        "res = {'uid': os.getuid()}\\n"
        "for what in ('environ', 'mem'):\\n"
        "    try:\\n"
        "        with open(f'/proc/{pid}/{what}', 'rb') as f:\\n"
        "            res[what] = f.read(1 << 20).count(b'sk-audit-1')\\n"
        "    except OSError as e:\\n"
        "        res[what] = type(e).__name__\\n"
        'RESULT = res\\n")\n'
        "print(json.dumps([os.getuid(), out['status'], out['result']]))\n"
    )
    env = {**os.environ, "FAKE_API_KEY": "sk-audit-1"}
    done = subprocess.run(
        [sys.executable, "-c", prog],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    import json

    server_uid, status, res = json.loads(done.stdout.strip().splitlines()[-1])
    assert status == "ran", res
    assert res["environ"] == "PermissionError", res
    assert res["mem"] == "PermissionError", res
    if server_uid == 0:
        assert res["uid"] >= code_exec.SANDBOX_UID_BASE, "a root server drops the uid"


@pytest.mark.skipif(not LINUX, reason="/proc is Linux-only")
def test_a_setsid_double_forked_grandchild_dies_with_the_run():
    marker = f"/tmp/run_code_marker_{os.getpid()}"
    out = runner(Toy()).run(
        "import os, time\n"
        "try:\n"
        "    pid = os.fork()\n"
        "except OSError:\n"
        "    RESULT = None  # RLIMIT_NPROC refused it\n"
        "else:\n"
        "    if pid == 0:\n"
        "        os.setsid()\n"
        "        if os.fork() == 0:\n"
        f"            open({marker!r}, 'w').write(str(os.getpid()))\n"
        "            time.sleep(60)\n"
        "        os._exit(0)\n"
        "    os.waitpid(pid, 0)\n"
        "    time.sleep(0.5)\n"
        f"    RESULT = int(open({marker!r}).read())\n"
    )
    with __import__("contextlib").suppress(OSError):
        os.unlink(marker)
    assert out["status"] == "ran", out
    gpid = out["result"]
    if gpid is None:
        return
    time.sleep(0.2)
    try:
        with open(f"/proc/{gpid}/stat") as f:
            state = f.read().rsplit(")", 1)[1].split()[0]
    except OSError:
        return
    if state != "Z":
        os.kill(gpid, 9)
        raise AssertionError("the grandchild outlived the run")
