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

"""Code mode on the RoboTwin server (``code.run`` over its registry), without the simulator: the
facade methods behind the registry are the tools' own, replaced here by recording fakes. The
server module's simulator imports (torch, cuRobo, RoboTwin, RLinf) are stubbed when absent and
removed again after the import, so other tests still see what is installed."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types

import numpy as np
import pytest

from pi_embodied_services.utils.rpc import RpcFacade

_STUBS = {
    "torch": {},
    "omegaconf": {"OmegaConf": object},
    "curobo": {},
    "curobo.opt": {},
    "curobo.opt.newton": {},
    "curobo.opt.newton.lbfgs": {"LBFGSOpt": type("LBFGSOpt", (), {})},
    "robotwin": {},
    "robotwin.assets": {"validate_root": lambda root: {"root": root}},
    "robotwin.config": {"load_task_config": lambda name: {}},
    "rlinf": {},
    "rlinf.envs": {},
    "rlinf.envs.robotwin": {},
    "rlinf.envs.robotwin.robotwin_env": {"RoboTwinEnv": type("RoboTwinEnv", (), {})},
}


def _import_env_server():
    missing = {
        top
        for top in {name.split(".")[0] for name in _STUBS}
        if top not in sys.modules and importlib.util.find_spec(top) is None
    }
    added = []
    for name, attrs in _STUBS.items():
        if name.split(".")[0] not in missing:
            continue
        module = types.ModuleType(name)
        module.__path__ = []  # a package, so its stubbed submodules import
        for k, v in attrs.items():
            setattr(module, k, v)
        sys.modules[name] = module
        added.append(name)
    try:
        return importlib.import_module(
            "pi_embodied_services.robots.robotwin.env_server"
        )
    finally:
        if added:
            for name in added:
                sys.modules.pop(name, None)
            for name in ("env_server", "rlinf_env"):
                sys.modules.pop(f"pi_embodied_services.robots.robotwin.{name}", None)


env_server = _import_env_server()
RoboTwinEnvFacade = env_server.RoboTwinEnvFacade

QPOS0 = [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0]


def robot_state(q):
    return {
        "left_eef_pose": np.array([-0.2, 0.0, 0.9, 1.0, 0, 0, 0]),
        "right_eef_pose": np.array([0.2, 0.0, 0.9, 1.0, 0, 0, 0]),
        "left_gripper": q[6],
        "right_gripper": q[13],
        "qpos_target14": np.asarray(q, dtype=np.float64),
    }


class FakeEnv:
    """The RLinf env: the policy frame (the joint targets and eef16 state a run starts from), the
    robot state, a planner and a chunk runner that commands qpos14 rows."""

    def __init__(self):
        self.frames = 0
        self.q = np.asarray(QPOS0, dtype=np.float64)
        self.count, self.limit = 0, 50
        self.solved_at: int | None = None
        self.chunks: list[np.ndarray] = []
        self.plans: list[tuple[str, list]] = []
        self.plan_ok = True

    def status(self):
        return {
            "eval_success": self.solved_at is not None and self.count >= self.solved_at,
            "take_action_cnt": self.count,
            "step_lim": self.limit,
            "actual_seed": 100000,
        }

    def robot_state(self, env_id=0):
        return {
            "robot_state": robot_state(list(self.q)),
            "episode_status": self.status(),
        }

    def plan_arm_path(self, env_id, arm, target_pose):
        self.plans.append((arm, list(target_pose)))
        if not self.plan_ok:
            return {"status": "Failure", "position": None, "velocity": None}
        off = 0 if arm == "left" else 7
        start = self.q[off : off + 6]
        end = np.full(6, 0.5)
        path = np.stack([start + (end - start) * (k / 10) for k in range(1, 11)])
        return {"status": "Success", "position": path, "velocity": None}

    def chunk_step(
        self,
        actions,
        action_type="qpos",
        env_id=0,
        return_all_frames=False,
        return_policy_frames=False,
        should_stop=None,
    ):
        actions = np.asarray(actions, dtype=np.float64)
        self.chunks.append(actions)
        frames, policy = [], []
        for a in actions:
            s = self.status()
            if s["eval_success"] or s["take_action_cnt"] >= s["step_lim"]:
                break
            self.q = a.copy()
            self.count += 1
            frames.append(np.full((2, 2, 3), self.count, np.uint8))
            if return_policy_frames:
                policy.append(self.policy_frame(0))
        n = len(frames)
        obs = {"main_images": np.zeros((1, 2, 2, 3), np.uint8)}
        if return_all_frames:
            obs = {"frames": frames, "final": obs}
        if return_policy_frames:
            obs = {**obs, "policy_frames": policy}
        info = {
            "action_type": action_type,
            "requested_actions": len(actions),
            "executed_actions": n,
            "robot_state": robot_state(list(self.q)),
            "episode_status": self.status(),
            "per_step": [{}] * n,
        }
        return (
            [obs],
            np.zeros((1, n)),
            np.zeros((1, n), bool),
            np.zeros((1, n), bool),
            [info],
        )

    def policy_frame(self, env_id):
        self.frames += 1
        return {
            "head": np.zeros((2, 2, 3), np.uint8),
            "left_wrist": np.zeros((2, 2, 3), np.uint8),
            "right_wrist": np.zeros((2, 2, 3), np.uint8),
            "qpos": np.asarray(QPOS0),
            "qpos_target": np.asarray(QPOS0),
            "state": np.r_[
                [-0.2, 0.0, 0.9, 1.0, 0, 0, 0], 1.0, [0.2, 0.0, 0.9, 1.0, 0, 0, 0], 1.0
            ],
        }


def facade():
    """A registered RoboTwin facade whose step methods are fakes that count native actions."""
    f = object.__new__(RoboTwinEnvFacade)
    RpcFacade.__init__(f)
    f._env, f._metadata = FakeEnv(), {"seed": 100000}
    f._run_steps, f._run_info, f._run_ref, f._run_frames = 0, None, None, []
    f.count, f.limit, f.moves = 0, 10, []
    RoboTwinEnvFacade._register_rpc(f)

    def status():
        return {
            "eval_success": f.count >= 3,
            "take_action_cnt": f.count,
            "step_lim": f.limit,
            "actual_seed": 100000,
        }

    def run(rows):
        for q in rows:
            f.count += 1
            f.moves.append(list(q))
        info = {
            "action_type": "qpos",
            "requested_actions": len(rows),
            "executed_actions": len(rows),
            "robot_state": robot_state(list(rows[-1])),
            "episode_status": status(),
            "per_step": [{"episode_status": status()}] * len(rows),
        }
        return info

    def step(action, action_type="qpos"):
        info = run([action])
        obs = {"main_images": np.full((2, 2, 3), 5, np.uint8), "states": np.zeros(14)}
        return obs, 0.0, info["episode_status"]["eval_success"], False, info

    def chunk_step(actions, action_type="qpos", return_all_frames=False):
        info = run(actions)
        obs = {"main_images": np.zeros((2, 2, 3), np.uint8)}
        if return_all_frames:
            obs = {
                "frames": [np.zeros((2, 2, 3), np.uint8)] * len(actions),
                "final": obs,
            }
        return (
            obs,
            np.zeros(len(actions)),
            np.zeros(len(actions), bool),
            np.zeros(len(actions), bool),
            info,
        )

    f._rpc["env.step"] = step
    f._rpc["env.chunk_step"] = chunk_step
    return f


def test_the_server_serves_code_run_with_a_token_and_exclusivity():
    f = facade()
    assert RoboTwinEnvFacade.REQUIRE_TOKEN and f._rpc_token
    assert "code.run" in f._rpc and "code.helpers" in f._rpc
    with pytest.raises(PermissionError):
        f._serve_dispatch("env.get_task_language", (), {})


def test_a_run_reports_native_actions_success_the_new_info_and_the_frames():
    f = facade()
    out = f._rpc["code.run"](
        "q = list(raw_obs()['qpos_target'])\nq[13] = 0.0\nstep(q)\n"
        "r = chunk_step([q, q], return_all_frames=True)\nRESULT = sorted(r)\n",
        timeout_s=30,
        tier="raw",
    )
    assert out["status"] == "ran", out
    assert out["result"] == ["info", "reward", "terminated", "truncated"]
    assert out["steps"] == 3 and out["success"] is True
    assert out["budget_exhausted"] is False
    assert out["info"]["episode_status"]["take_action_cnt"] == 3
    assert list(out["info"]["robot_state"]["qpos_target14"])[13] == 0.0
    # One head frame of the step, one per action of the chunk.
    assert len(out["frames"]) == 3
    assert [m[13] for m in f.moves] == [0.0, 0.0, 0.0]


def test_a_program_receives_no_camera_observation():
    f = facade()
    out = f._rpc["code.run"](
        "s = raw_obs()\nr = step(list(s['qpos_target']))\n"
        "RESULT = [sorted(s), sorted(r['info'])]\n",
        timeout_s=30,
        tier="raw",
    )
    assert out["status"] == "ran", out
    assert out["result"][0] == ["qpos", "qpos_target", "state"]
    assert "per_step" not in out["result"][1] and "robot_state" in out["result"][1]


def test_the_move_cap_counts_joint_changes_from_the_commanded_state_and_refuses():
    f = facade()
    q = list(QPOS0)
    q[0] = 0.3  # 0.3 rad of the left shoulder: at most 0.21 m
    out = f._rpc["code.run"](
        f"step({q})\nstep({q})\nq = {q}\nq[7] = 0.2\nstep(q)\n",
        timeout_s=30,
        tier="raw",
        max_move_m=0.3,
    )
    assert out["status"] == "error" and out["limit"] == "max_move_m", out
    moves = [c.get("move_m", 0.0) for c in out["calls"]]
    # The second step starts where the first one commanded; the third is refused.
    assert moves[:2] == [pytest.approx(0.3 * env_server.JOINT_REACH_M), 0.0]
    assert "refused" in out["calls"][2]
    assert len(f.moves) == 2 and f._env.frames == 1
    # An ee action is estimated by how far each eef position moves.
    f = facade()
    ee = [-0.2, 0.1, 0.9, 1, 0, 0, 0, 1, 0.2, 0.0, 0.8, 1, 0, 0, 0, 1]
    out = f._rpc["code.run"](
        f"step({ee}, action_type='ee')\n", timeout_s=30, tier="raw", max_move_m=0.15
    )
    assert out["status"] == "error" and out["limit"] == "max_move_m", out
    assert f.moves == []


def test_oversized_chunks_are_refused_before_they_run():
    f = facade()
    n = env_server.CODE_MAX_CHUNK + 1
    out = f._rpc["code.run"](f"chunk_step([{QPOS0}] * {n})\n", timeout_s=30, tier="raw")
    assert (
        out["status"] == "error"
        and f"at most {env_server.CODE_MAX_CHUNK}" in out["error"]
    ), out
    assert f.moves == []


def test_the_run_video_is_bounded():
    f = facade()
    f._keep_frames([np.zeros((1, 1, 3), np.uint8)] * (env_server.CODE_MAX_FRAMES + 5))
    assert len(f._run_frames) <= env_server.CODE_MAX_FRAMES


def test_the_low_tier_shows_examples_and_the_s4_tier_drops_them():
    f = facade()
    low = f._rpc["code.api"]("low")
    s4 = f._rpc["code.api"]("low-noexamples")
    assert s4["available"] == low["available"] and s4["digest"] != low["digest"]
    docs = {p["name"]: p["doc"] for p in f._code.api("low")}
    assert "Example:" in docs["move_to"] and "Example:" in docs["solve_ik"]
    assert all("Example:" not in p["doc"] for p in f._code.api("low-noexamples"))


def test_the_tiers_follow_the_manifest():
    f = facade()
    tiers = {
        t: set(f._rpc["code.api"](t)["available"]) for t in ("low", "raw", "privileged")
    }
    assert {"move_to", "rotate_wrist", "set_gripper", "release"} <= tiers["low"]
    assert {
        "solve_ik",
        "move_to_joints",
        "traj_plan",
        "move_along_trajectory",
    } <= tiers["low"]
    assert tiers["raw"] == {"raw_obs", "step", "chunk_step"}
    assert tiers["privileged"] == {"ground_truth_poses"}, "no high tier on RoboTwin"
    assert "detect" not in tiers["low"], "no perception without --sam3"


def test_the_server_checks_itself_against_its_manifest():
    """A facade method the manifest does not declare (nor lists as internal) stops the server,
    and so does a declared one that is missing."""
    from pi_embodied_services.components.manifest import ManifestError

    f = facade()
    f._rpc["env.secret_teleport"] = lambda: None
    with pytest.raises(ManifestError, match="env.secret_teleport"):
        f._manifest_ready()
    g = facade()
    del g._rpc["env.traj_plan"]
    with pytest.raises(ManifestError, match="traj_plan"):
        g._manifest_ready()
    facade()._manifest_ready()


def real_facade():
    """A facade whose motion methods run on the fake RLinf env (no step fakes)."""
    f = object.__new__(RoboTwinEnvFacade)
    RpcFacade.__init__(f)
    f._env, f._metadata = FakeEnv(), {"seed": 100000}
    f._run_steps, f._run_info, f._run_ref, f._run_frames = 0, None, None, []
    RoboTwinEnvFacade._register_rpc(f)
    return f


def test_move_to_plans_on_the_server_and_runs_the_subsampled_path_as_one_chunk():
    f = real_facade()
    r = f._rpc["env.move_to"]("left", [-0.1, 0.0, 0.95], substeps=4, gripper=0.5)
    assert f._env.plans == [("left", [-0.1, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0])], (
        "orientation kept"
    )
    (chunk,) = f._env.chunks
    assert chunk.shape == (4, 14)
    assert np.allclose(chunk[-1][:6], 0.5) and np.allclose(chunk[:, 6], 0.5)
    assert np.allclose(chunk[:, 7:], QPOS0[7:]), "the other arm holds"
    assert r["executed_steps"] == 4 and r["stop_reason"] == "completed"
    assert r["waypoints"] == 4 and r["plan_status"] == "Success"
    assert len(r["frames"]) == 4 and "policy_frames" not in r
    assert r["info"]["executed_actions"] == 4 and "per_step" not in r["info"]
    f._env.plan_ok = False
    r = f._rpc["env.move_to"]("right", [0.1, 0.0, 0.95])
    assert r["stop_reason"] == "plan_failed" and r["executed_steps"] == 0
    assert len(f._env.chunks) == 1


def test_set_gripper_ramps_and_the_flywheel_gets_policy_frames():
    f = real_facade()
    r = f._rpc["env.set_gripper"]("right", 0.0, steps=4, return_policy_frames=True)
    (chunk,) = f._env.chunks
    assert np.allclose(chunk[:, 13], [0.75, 0.5, 0.25, 0.0])
    assert np.allclose(chunk[:, :13], QPOS0[:13])
    assert r["gripper_val"] == 0.0 and len(r["policy_frames"]) == 4
    assert r["per_step"]["reward"] == [0.0] * 4
    f._env.solved_at = f._env.count
    r = f._rpc["env.release"]("right")
    assert r["executed_steps"] == 0 and r["stop_reason"] == "native_success"


def test_joint_space_primitives():
    f = real_facade()
    q = f._rpc["env.solve_ik"]([-0.1, 0.0, 0.95], [1, 0, 0, 0], "left")
    assert np.allclose(q, 0.5)
    start = [1, 0, 0, 0, -0.2, 0.0, 0.9]
    traj = f._rpc["env.traj_plan"](start, [1, 0, 0, 0, -0.1, 0.0, 0.95], "left")
    assert np.asarray(traj).shape == (10, 6)
    with pytest.raises(ValueError, match="current pose"):
        f._rpc["env.traj_plan"]([1, 0, 0, 0, 0.0, 0.0, 0.9], start, "left")
    r = f._rpc["env.move_along_trajectory"](traj[::3], "left")
    assert r["executed_steps"] == 4 and np.allclose(r["final_joints"], traj[-1])
    r = f._rpc["env.move_to_joints"]([0.1] * 6, "right", gripper=0.0)
    assert np.allclose(f._env.chunks[-1][0][7:], [0.1] * 6 + [0.0])
    with pytest.raises(ValueError):
        f._rpc["env.move_to_joints"]([0.1] * 5, "right")
    f._env.plan_ok = False
    with pytest.raises(ValueError, match="no IK solution"):
        f._rpc["env.solve_ik"]([-0.1, 0.0, 0.95], [1, 0, 0, 0], "left")


def test_a_program_moves_through_the_motion_methods_under_the_move_cap():
    f = real_facade()
    out = f._rpc["code.run"](
        "r = move_to('left', [-0.2, 0.0, 1.0], substeps=2)\n"
        "RESULT = [r['stop_reason'], 'frames' in r, sorted(r['info'])]\n",
        timeout_s=30,
        tier="low",
    )
    assert out["status"] == "ran", out
    assert out["result"][0] == "completed" and out["result"][1] is False
    assert "robot_state" in out["result"][2]
    assert out["steps"] == 2 and len(out["frames"]) == 2
    assert out["calls"][0]["move_m"] == pytest.approx(0.1)
    out = f._rpc["code.run"](
        "move_to_joints([1.5] * 6, 'right')\n", timeout_s=30, tier="low", max_move_m=1.0
    )
    assert out["status"] == "error" and out["limit"] == "max_move_m", out
