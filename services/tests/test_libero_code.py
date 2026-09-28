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

"""The LIBERO facade's code-mode primitives (`code.api` / `code.run`) over a mock
LiberoEnv, no simulator: tiers, the servo primitives, back-projection, stop handling and
the run's bookkeeping for pi."""

from __future__ import annotations

import numpy as np
import pytest

from pi_embodied_services.robots.libero.env_server import (
    CODE_RES,
    LiberoEnvFacade,
)


class ArmSim:
    """A LiberoEnv stand-in: an OSC-like arm (action[:3] * 0.05 m per step, action[5] * 0.1 rad
    yaw, action[6] drives the fingers), a flat table 0.5 m below a camera looking straight down
    with a wrist camera, and success once the arm rises above z = 0.5 with the gripper closed."""

    def __init__(self):
        self.pos = np.array([0.0, 0.0, 0.2])
        self.yaw = 0.0
        self.width = 0.08
        self.steps = 0
        self.succeeded = False
        self.workers = [self]

    def _quat(self):
        return np.array([0.0, 0.0, np.sin(self.yaw / 2), np.cos(self.yaw / 2)])

    @property
    def current_raw_obs(self):
        half = self.width / 2
        return [
            {
                "robot0_eef_pos": self.pos.copy(),
                "robot0_eef_quat": self._quat(),
                "robot0_gripper_qpos": np.array([half, -half]),
            }
        ]

    def _obs(self):
        return {
            "main_images": np.full((1, 4, 4, 3), self.steps % 256, dtype=np.uint8),
            "wrist_images": np.zeros((1, 4, 4, 3), dtype=np.uint8),
            "states": np.zeros((1, 8), dtype=np.float32),
        }

    def _info(self):
        return {"episode": {"success_once": np.array([self.succeeded])}}

    def reset(self):
        self.__init__()
        return self._obs(), self._info()

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(7)
        self.steps += 1
        self.pos = self.pos + a[:3] * 0.05
        self.pos[2] = max(self.pos[2], 0.0)  # the table
        self.yaw += a[5] * 0.1
        if a[6] > 0:
            self.width = max(0.02, self.width - 0.02)  # closes on a 2 cm object
        elif a[6] < 0:
            self.width = min(0.08, self.width + 0.02)
        if self.pos[2] > 0.5 and self.width < 0.05:
            self.succeeded = True
        zeros = np.zeros(1, dtype=bool)
        return self._obs(), np.zeros(1), zeros, zeros, self._info()

    def chunk_step(self, actions):
        obs, rews, terms, truncs, infos = [], [], [], [], []
        for a in np.asarray(actions, dtype=np.float64).reshape(-1, 7):
            o, r, t, tr, i = self.step(a)
            obs.append(o)
            rews.append(r)
            terms.append(t)
            truncs.append(tr)
            infos.append(i)
        return obs, np.stack(rews), np.stack(terms), np.stack(truncs), infos

    def render_camera(self, camera_name, height, width, depth):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[0, 0] = (
            255  # a marker at the raw image's first row: upright it is the last row
        )
        d = np.full((height, width), 0.5, dtype=np.float32)
        return (rgb, d) if depth else rgb

    def get_camera_meta(self, camera_name, height, width):
        f = height / 2
        return {
            "intrinsic_K": [[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]],
            # Camera 0.7 m above the table looking down: cam +z is world -z, cam +x is world +x.
            "extrinsic_cam2world": [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0.7],
                [0, 0, 0, 1],
            ],
        }

    def env_call(self, name, target):
        return {"cube": {"pos": [0.1, 0.0, 0.02], "quat_xyzw": [0, 0, 0, 1]}}

    task_descriptions = ["lift the cube"]

    @property
    def env(self):
        return self


def facade(sam3=None) -> LiberoEnvFacade:
    """A facade as main() builds it: with --sam3 its perception primitives are installed too."""
    f = LiberoEnvFacade(ArmSim(), meta={}, sam3=sam3)
    if sam3:
        import argparse

        from pi_embodied_services.utils.perception import install_perception

        install_perception(
            f,
            argparse.Namespace(sam3=sam3, unidepth=""),
            cameras=["agentview", "wrist"],
            view=f._view,
        )
    f.reset()
    return f


def names(api):
    """Primitive names of a `code.api` reply, a registry tier list or a runner list."""
    if isinstance(api, dict):
        api = api["primitives"]
    return [p["name"] if isinstance(p, dict) else p.name for p in api]


def available(f, tier):
    return f._rpc["code.api"](tier)["available"]


HIGH = ["goto_pose", "home_pose", "open_gripper", "close_gripper"]
LOW = [
    "get_task_language",
    "get_state",
    "get_observation",
    "back_project",
    "move_to",
    "move_pose",
    "rotate_wrist",
    "rotate_pitch",
    "move_delta",
    "rotate_delta",
    "set_gripper",
    "release",
    "get_camera_meta",
]
RAW = ["render_camera", "raw_obs", "step", "chunk_step"]


def test_the_manifest_tiers_list_the_code_primitives_and_privileged_adds_ground_truth():
    f = facade()
    assert available(f, "high") == HIGH, (
        "CaP-X's semantic functions only (no SAM3 here)"
    )
    assert available(f, "low") == LOW
    assert available(f, "raw") == RAW
    priv = available(f, "privileged")
    assert set(priv) == set(HIGH) | {
        "ground_truth_poses",
        "get_object_pose",
        "sample_grasp_pose",
    }, "the privileged pose functions need no SAM3"
    low_priv = available(f, "low+privileged")
    assert set(low_priv) == set(LOW) | {
        "ground_truth_poses",
        "get_object_pose",
        "sample_grasp_pose",
    }
    high = f._rpc["code.api"]("high")
    assert high["tier"] == "high" and isinstance(high["digest"], str)
    # The runner renders the manifest's declaration as a signature and a doc for the program.
    runner = {p.name: p for p in f._code.primitives("low")}
    assert set(runner) == set(LOW)
    sig = runner["move_to"].describe()["signature"]
    assert sig.startswith("(xyz: vec3, gripper: number = None, tol: number = None")
    doc = runner["move_to"].describe()["doc"]
    assert "Args:" in doc and "Moves the robot." in doc and "Example:" in doc


def test_every_low_primitive_has_an_example_the_s4_tier_drops():
    f = facade()
    f._manifest_ready()
    low = {p.name: p.describe()["doc"] for p in f._code.primitives("low")}
    assert all("Example:" in d for d in low.values()), [
        n for n, d in low.items() if "Example:" not in d
    ]
    s4 = f._code.api("low-noexamples")
    assert [p["name"] for p in s4] == list(low)
    assert all("Example:" not in p["doc"] for p in s4)


def test_segment_and_the_perception_pose_functions_need_a_sam3_server():
    f = facade()
    assert "segment" not in available(f, "low")
    assert "get_object_pose" not in available(f, "high")
    with_sam3 = facade(sam3="http://127.0.0.1:1")
    assert "segment" in available(with_sam3, "low")
    assert {"get_object_pose", "sample_grasp_pose"} <= set(available(with_sam3, "high"))
    assert "segment" not in available(with_sam3, "high")


def test_move_to_servos_with_the_tools_step_rule_and_opens_unless_told():
    f = facade()
    out = f.move_to([0.1, 0.05, 0.2])
    assert out["final_dist_m"] < 0.012
    assert out["eef_pos"] == pytest.approx([0.1, 0.05, 0.2], abs=0.012)
    # 2.5 cm per step at most: 0.1 m takes at least 4 steps.
    assert 4 <= out["steps_used"] <= 6
    assert out["gripper_width"] == pytest.approx(0.08)
    f.set_gripper(1)
    assert f.get_state()["gripper_cmd"] == 1
    f.move_to([0.1, 0.05, 0.3], gripper=None)
    assert f.get_state()["gripper_cmd"] == 1, "None keeps the last gripper command"
    f.move_to([0.1, 0.05, 0.3], gripper=1)
    assert f.get_state()["gripper_cmd"] == 1
    opened = f.move_to([0.1, 0.05, 0.2])
    assert opened["gripper_width"] == pytest.approx(0.08), "the tool's default opens"
    assert f.get_state()["gripper_cmd"] == -1
    with pytest.raises(ValueError):
        f.move_to([0, 0, 0.2], gripper=0.5)


def test_move_delta_and_rotate_delta_are_bounded_and_report_travel():
    f = facade()
    out = f.move_delta([0, 0, -0.05])
    assert out["moved_m"] == pytest.approx(0.05, abs=0.005)
    with pytest.raises(ValueError, match="0.10 m"):
        f.move_delta([0.2, 0, 0])
    f.move_to([0, 0, 0.05])
    blocked = f.move_delta([0, 0, -0.1])  # the table is at z = 0
    assert blocked["moved_m"] == pytest.approx(0.05, abs=0.005)
    turn = f.rotate_delta(0.5)
    assert turn["yaw"] == pytest.approx(0.5, abs=0.02)
    with pytest.raises(ValueError):
        f.rotate_delta(2.0)
    assert f.rotate_wrist(target_yaw=0.0)["yaw"] == pytest.approx(0.0, abs=0.02)
    with pytest.raises(ValueError):
        f.rotate_wrist()


def test_set_gripper_drives_its_steps_and_the_executors_closing_stops_with_the_fingers():
    f = facade()
    out = f.set_gripper(1, steps=8)
    assert out["gripper_width"] == pytest.approx(0.02)
    assert out["steps_used"] == 8, "the tool's rule: every step it was asked for"
    assert f.set_gripper()["steps_used"] == 5, "default: open for 5 steps"
    assert f._actuate(1.0)["steps_used"] < 8, "the executors stop once the fingers rest"
    assert f.open_gripper()["steps_used"] == 40
    assert f.close_gripper()["steps_used"] == 60


def test_observation_is_upright_metric_and_back_projects_with_its_calibration():
    f = facade()
    obs = f.get_observation()
    assert set(obs) >= {"agentview", "wrist", "eef_pos", "gripper_width", "terminated"}
    view = obs["agentview"]
    assert (
        view["rgb"].shape == (CODE_RES, CODE_RES, 3) and view["rgb"].dtype == np.uint8
    )
    assert (
        view["depth"].shape == (CODE_RES, CODE_RES)
        and view["depth"].dtype == np.float32
    )
    assert view["rgb"][-1, 0, 0] == 255, "rows are flipped: LIBERO renders upside down"
    # No depth_near/far in the meta: the depth is already metric (0.5 m to the table).
    assert float(view["depth"][10, 10]) == pytest.approx(0.5)
    # The principal pixel looks straight down from 0.7 m onto the table at z = 0.2.
    centre = f.back_project(CODE_RES // 2, CODE_RES // 2)["world_xyz"]
    assert centre == pytest.approx([0.0, 0.0, 0.2], abs=1e-3)
    # One pixel right of centre is +x in the camera, so +x in the world (f = 256 px, z = 0.5).
    right = f.back_project(CODE_RES // 2, CODE_RES // 2 + 128)["world_xyz"]
    assert right[0] == pytest.approx(0.25, abs=1e-3)
    with pytest.raises(ValueError, match="out of bounds"):
        f.back_project(CODE_RES, 0)
    with pytest.raises(ValueError, match="camera"):
        f.back_project(0, 0, camera="overhead")


def test_a_run_reports_steps_success_the_latest_obs_and_frames():
    f = facade()
    out = f._rpc["code.run"](
        "g = set_gripper(True)\n"
        "r = move_to([0, 0, 0.6], gripper=1)\n"
        "RESULT = [get_state()['terminated'], r['steps_used'] + g['steps_used']]\n",
        timeout_s=30,
        tier="low",
    )
    assert out["status"] == "ran", out
    assert out["result"][0] is True
    assert out["terminated"] is True and out["truncated"] is False
    assert out["success_step"] is not None and out["success_step"] <= out["steps"]
    assert out["steps"] == out["result"][1], "the gripper's steps and the move's"
    assert out["obs"]["main_images"].shape == (4, 4, 3)
    assert len(out["frames"]) == 2, "one agentview frame per motion primitive"
    assert [c["name"] for c in out["calls"]] == ["set_gripper", "move_to", "get_state"]
    assert out["calls"][1]["move_m"] == pytest.approx(0.4)


def test_the_raw_tiers_steps_count_and_its_raw_obs_has_no_object_poses():
    f = facade()
    raw = f.raw_obs
    f.raw_obs = lambda: {
        **raw(),
        "akita_black_bowl_1_pos": np.zeros(3),
        "agentview_image": np.zeros((4, 4, 3), np.uint8),
    }
    out = f._rpc["code.run"](
        "obs = raw_obs()\n"
        "for _ in range(3):\n"
        "    step([0, 0, 1, 0, 0, 0, -1])\n"
        "chunk_step([[0, 0, 1, 0, 0, 0, -1]] * 2)\n"
        "RESULT = sorted(obs)\n",
        timeout_s=30,
        tier="raw",
    )
    assert out["status"] == "ran", out
    assert out["result"] == [
        "agentview_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ], "the object pose is privileged"
    assert out["steps"] == 5, "raw steps count like the primitives' steps"
    assert out["terminated"] == (out["success_step"] is not None)


def test_a_finished_episode_stops_every_motion_primitive():
    f = facade()
    f.set_gripper(True)
    f.move_to([0, 0, 0.6], gripper=1)
    assert f.get_state()["terminated"]
    assert f.move_to([0.3, 0, 0.6])["steps_used"] == 0
    assert f.set_gripper(False)["steps_used"] == 0
    f.reset()
    assert not f.get_state()["terminated"] and f.get_state()["gripper_cmd"] == -1


def test_a_stop_ends_a_servo_loop_between_steps():
    f = facade()
    f._active_generation = f._stop_generation  # as inside a running call
    f.request_stop()
    out = f.move_to([0.25, 0, 0.2])
    assert out["steps_used"] == 0 and out["cancelled"] is True


def test_move_budget_is_estimated_from_the_target_and_refuses():
    f = facade()
    out = f._rpc["code.run"](
        "log = []\n"
        "for t in ([0.05, 0, 0.2], [0.10, 0, 0.2], [0.30, 0, 0.2]):\n"
        "    try:\n"
        "        move_to(t); log.append('ok')\n"
        "    except Exception as e:\n"
        "        log.append(type(e).__name__)\n"
        "RESULT = log\n",
        timeout_s=30,
        tier="low",
        max_move_m=0.12,
    )
    assert out["result"] == ["ok", "ok", "CodeLimitError"]
    assert out["limit"] == "max_move_m"
    assert out["steps"] > 0


def test_ground_truth_primitive_only_with_privileged():
    f = facade()
    out = f._rpc["code.run"]("RESULT = ground_truth_poses()", timeout_s=30, tier="low")
    assert "NameError" in out["error"]
    out = f._rpc["code.run"](
        "RESULT = ground_truth_poses(['cube'])", timeout_s=30, tier="privileged"
    )
    assert out["result"]["poses"]["cube"]["pos"] == [0.1, 0.0, 0.02]


def test_calls_go_through_the_registry_resolve():
    f = facade()
    out = f._rpc["code.run"](
        "log = []\n"
        "calls = (lambda: move_to([0, 0, 0.3], sideways=1), lambda: move_to(),"
        " lambda: move_to([0, 0, 0.3], 1, tol=0.01, max_steps=5, extra=2))\n"
        "for call in calls:\n"
        "    try:\n"
        "        call(); log.append('ok')\n"
        "    except Exception as e:\n"
        "        log.append(str(e))\n"
        "RESULT = log\n",
        timeout_s=30,
        tier="low",
    )
    assert "unknown parameter(s) sideways" in out["result"][0]
    assert "missing parameter(s) xyz" in out["result"][1]
    assert "unknown parameter(s) extra" in out["result"][2]
    assert out["steps"] == 0, "a refused call never reaches the env"
    # Positional arguments fill the declared parameters in order; a raw step is capped too.
    out = f._rpc["code.run"](
        "r = move_delta([0, 0, 0.05], -1)\nRESULT = r['moved_m']",
        timeout_s=30,
        tier="low",
    )
    assert out["result"] == pytest.approx(0.05, abs=0.005)
    out = f._rpc["code.run"](
        "step([0.2, 0, 0, 0, 0, 0, -1])\n", timeout_s=30, tier="raw"
    )
    assert out["calls"][0]["move_m"] == pytest.approx(0.01)
    assert len(out["frames"]) == 1, "raw steps are mutating too"


def test_non_finite_targets_are_refused_by_the_facade_methods_pi_calls_too():
    f = facade()
    nan, inf = float("nan"), float("inf")
    for call in (
        lambda: f.move_to([nan, 0, 0.2]),
        lambda: f.move_to([0, 0, 0.2], tol=nan),
        lambda: f.move_delta([0, inf, 0]),
        lambda: f.rotate_wrist(target_yaw=nan),
        lambda: f.rotate_wrist(delta_yaw=inf),
        lambda: f.rotate_delta(nan),
    ):
        with pytest.raises(ValueError, match="finite"):
            call()
    assert f._env.steps == 0, "nothing reached the simulator"


def test_a_programs_nan_step_is_refused_and_the_move_cap_holds():
    f = facade()
    out = f._rpc["code.run"](
        "try:\n"
        "    step([float('nan'), 0, 0, 0, 0, 0, -1])\n"
        "except RuntimeError as e:\n"
        "    RESULT = str(e)\n"
        "for _ in range(30):\n"
        "    step([1, 0, 0, 0, 0, 0, -1])\n",
        timeout_s=30,
        tier="raw",
        max_move_m=0.5,
    )
    assert "non-finite" in out["calls"][0]["error"]
    assert out["limit"] == "max_move_m" and out["steps"] == 10, out


def test_a_long_raw_chunk_is_refused_before_it_runs():
    f = facade()
    out = f._rpc["code.run"](
        "chunk_step([[0, 0, 0, 0, 0, 0, -1]] * 1000)\n", timeout_s=30, tier="raw"
    )
    assert out["status"] == "error" and "at most 64 actions" in out["error"], out
    assert out["steps"] == 0


def test_the_server_requires_a_token_and_is_exclusive_while_a_program_runs():
    f = facade()
    assert LiberoEnvFacade.REQUIRE_TOKEN and f._rpc_token
    token = f._rpc_token
    seen = {}

    def probe():
        # What a program calling the server's port itself would get, mid-run.
        try:
            f._serve_dispatch("env.step", ([0] * 7,), {}, token=token)
        except RuntimeError as exc:
            seen["refused"] = str(exc)
        return {}

    f._manifest_ready()
    f._rpc["env.get_state"] = probe  # the runner looks the method up at call time
    out = f._serve_dispatch(
        "code.run", ("get_state()\n",), {"timeout_s": 30, "tier": "low"}, token=token
    )
    assert out["status"] == "ran", out
    assert "run_code program is running" in seen["refused"]
    assert out["steps"] == 0
    with pytest.raises(PermissionError):
        f._serve_dispatch("env.get_state", (), {})


# ---- planned grasps: plan_grasp -> execute_grasp -> plan_place -> execute_place ----


class DownArm(ArmSim):
    """ArmSim with LIBERO's hand orientation: pointing down (180 deg about x), then the yaw."""

    def _quat(self):
        return np.array([np.cos(self.yaw / 2), np.sin(self.yaw / 2), 0.0, 0.0])


def grasp_facade():
    """A LIBERO facade with a fake grasp server, SAM3 and AnyPlace: one candidate 5 cm along
    +x on the 0.2 m high table plane, and a place 10 cm further along +x."""
    from test_grasp import FakeAnyPlace, FakeSam3, FakeServer

    from pi_embodied_services.utils import grasp as G

    mask = np.zeros((CODE_RES, CODE_RES), bool)
    mask[200:300, 200:300] = True
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.0, 0.0]  # camera +x is world +x
    # Approach along camera +z (straight down), 0.5 m from the camera: world z = 0.2.
    server = FakeServer(
        [
            G.make_candidate(
                score=0.9,
                rotation=G.ZX_NATIVE_TO_GRASPNET,
                center=[0.05, 0.0, 0.5],
                width=0.04,
                depth=0.0,
                source_model="fake",
            )
        ]
    )
    f = LiberoEnvFacade(
        DownArm(),
        meta={},
        grasp={"contact_graspnet": server, "anyplace": FakeAnyPlace([T])},
    )
    f._grasp._sam3 = FakeSam3(mask)
    f.reset()
    return f, G


def test_a_planned_grasp_and_place_run_end_to_end_from_one_resolution_each():
    f, G = grasp_facade()
    rpc = f._rpc
    plan = rpc["env.plan_grasp"](object="bowl")
    gid = plan["active"]
    assert plan["candidates"][0]["eef_position"] == pytest.approx(
        [0.05, 0, 0.2], abs=1e-6
    )
    # Looking at the robot does not expire the plan (the sim did not step).
    rpc["env.get_observation"]()
    rpc["env.get_state"]()
    grasp = rpc["env.execute_grasp"](grasp_id=gid, standoff=0.1, lift=0.1)
    assert "error" not in grasp, grasp
    assert [leg.get("to") for leg in grasp["legs"]] == [
        "pre_grasp",
        "grasp",
        None,
        "lift",
    ]
    assert all(leg["final_dist_m"] < 0.012 for leg in grasp["legs"] if "to" in leg)
    assert grasp["legs"][2]["gripper_width"] == pytest.approx(0.02), "closed on it"
    assert grasp["eef_pos"] == pytest.approx([0.05, 0, 0.3], abs=0.012)
    assert f._yaw() == pytest.approx(np.pi / 2, abs=0.05), (
        "turned to the candidate's yaw"
    )
    # The grasp's own motion spent the id; plan_place takes it as the held grasp.
    with pytest.raises(G.GraspError, match="stale"):
        rpc["env.resolve_grasp"](gid)
    region = rpc["env.segment_mask"]("plate")["id"]
    place = rpc["env.plan_place"](region, gid)
    assert place["held"] is True
    pid = place["active"]
    assert place["candidates"][0]["eef_position"] == pytest.approx(
        [0.15, 0, 0.21], abs=0.013
    ), "10 cm along +x from where the gripper holds it, set down on the table's top"
    out = rpc["env.execute_place"](place_id=pid)
    assert "error" not in out, out
    assert [leg.get("to") for leg in out["legs"]] == [
        "pre_place",
        "place",
        None,
        "retreat",
    ]
    assert out["gripper_width"] == pytest.approx(0.08), "released"
    assert f._grasp.held() is None
    with pytest.raises(G.GraspError, match="stale"):
        rpc["env.execute_place"](place_id=pid)


def test_the_old_two_move_flow_is_what_the_executor_replaces():
    """Finding: move_to(grasp_id, standoff 0.10) then move_to(grasp_id, standoff 0) could never
    work: the first move's steps expire the id before the second resolves it."""
    f, G = grasp_facade()
    gid = f._rpc["env.plan_grasp"](object="bowl")["active"]
    pre = f._rpc["env.resolve_grasp"](gid, standoff=0.1)["eef_position"]
    f._rpc["env.move_to"](pre)
    with pytest.raises(G.GraspError, match="stale"):
        f._rpc["env.resolve_grasp"](gid, standoff=0.0)


def test_a_wrist_turn_expires_the_ids_and_an_empty_grasp_is_not_placed():
    f, G = grasp_facade()
    gid = f._rpc["env.plan_grasp"](object="bowl")["active"]
    f._rpc["env.rotate_wrist"](delta_yaw=0.3)
    with pytest.raises(G.GraspError, match="stale"):
        f._rpc["env.execute_grasp"](grasp_id=gid)
    # A refused call leaves the sim untouched: the ids survive it.
    gid = f._rpc["env.plan_grasp"](object="bowl")["active"]
    with pytest.raises(ValueError, match="0.10"):
        f._rpc["env.move_delta"]([0.0, 0.0, 0.5])
    assert f._rpc["env.resolve_grasp"](gid)["id"] == gid
    f._rpc["env.execute_grasp"](grasp_id=gid)
    f._rpc["env.set_gripper"](False)  # dropped it
    region = f._rpc["env.segment_mask"]("plate")["id"]
    with pytest.raises(G.GraspError, match="holds nothing"):
        f._rpc["env.plan_place"](region, gid)


def test_the_executors_are_code_primitives_only_with_a_grasp_server():
    plain = set(available(facade(), "low"))
    assert {"execute_grasp", "execute_place"} & plain == set()
    f, _ = grasp_facade()
    low = available(f, "low")
    assert {"execute_grasp", "execute_place", "claim_waypoints", "plan_grasp"} <= set(
        low
    )
    assert "execute_grasp" not in available(f, "high")
    f._manifest_ready()
    ex = next(p for p in f._code.api("low") if p["name"] == "execute_grasp")
    assert ex["signature"].startswith("(grasp_id: string, standoff: number = None")
    assert "Moves the robot." in ex["doc"]
    # The run's translation cap counts the whole path.
    gid = f._rpc["env.plan_grasp"](object="bowl")["active"]
    moved = f._code_move_m("env.execute_grasp", {"grasp_id": gid})
    # From (0, 0, 0.2) to the pre-grasp (0.05, 0, 0.3), 0.10 down, 0.10 up.
    assert moved == pytest.approx(np.hypot(0.05, 0.1) + 0.2, abs=1e-3)
    travel = []
    act = f._act

    def tracked(action):
        before = f._eef()
        act(action)
        travel.append(float(np.linalg.norm(f._eef() - before)))

    f._act = tracked
    assert "error" not in f._rpc["env.execute_grasp"](grasp_id=gid)
    assert sum(travel) <= moved + 0.02, "the cap covers the path the arm travels"
    f._act = act
    gid = f._rpc["env.plan_grasp"](object="bowl")["active"]
    with pytest.raises(ValueError, match="finite"):
        f._rpc["env.execute_grasp"](grasp_id=gid, standoff=float("nan"))
    assert f._rpc["env.resolve_grasp"](gid)["id"] == gid, "refused before moving"


def test_a_reset_expires_the_plan_and_a_replan_renders_afresh():
    """LIBERO's scene only changes through the sim, whose exact state is the digest; a reset
    always expires the ids (the arm may come back to the same pose)."""
    f, G = grasp_facade()
    gid = f._rpc["env.plan_grasp"](object="bowl")["active"]
    f._rpc["env.reset"]()
    with pytest.raises(G.GraspError, match="stale"):
        f._rpc["env.execute_grasp"](grasp_id=gid)
    gid = f._rpc["env.plan_grasp"](object="bowl")["active"]
    assert f._rpc["env.resolve_grasp"](gid)["id"] == gid


def test_a_place_that_stalls_before_opening_keeps_the_held_grasp():
    """Audit: execute_place dropped the held record at its claim, so a place stuck above the
    target left an object in hand that plan_place refused. It ends only once the hand opened."""
    f, G = grasp_facade()
    gid = f._rpc["env.plan_grasp"](object="bowl")["active"]
    assert "error" not in f._rpc["env.execute_grasp"](grasp_id=gid)
    region = f._rpc["env.segment_mask"]("plate")["id"]
    pid = f._rpc["env.plan_place"](region, gid)["active"]
    servo = f._servo_pose

    def blocked(
        target, *args, **kwargs
    ):  # something under the pre-place stops the descent
        if target[2] < 0.25:
            return 1, False
        return servo(target, *args, **kwargs)

    f._servo_pose = blocked
    out = f._rpc["env.execute_place"](place_id=pid)
    assert out.get("stalled") == "place", out
    assert f._grasp.held()["grasp_id"] == gid
    f._servo_pose = servo
    f._rpc["env.move_to"]([0.05, 0.0, 0.3], gripper=1)  # back down, still holding
    region = f._rpc["env.segment_mask"]("plate")["id"]
    place = f._rpc["env.plan_place"](region, gid)
    assert place["held"] is True, "the object is still in hand: plan again"
    done = f._rpc["env.execute_place"](place_id=place["active"])
    assert "error" not in done, done
    assert f._grasp.held() is None


class FullArm(DownArm):
    """DownArm whose OSC turns the hand by the full world-frame rotation vector a[3:6] * 0.1."""

    def __init__(self):
        super().__init__()
        self.R = np.diag([1.0, -1.0, -1.0])  # pointing down

    def _quat(self):
        from pi_embodied_services.utils.grasp import quat_xyzw

        return np.array(quat_xyzw(self.R))

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(7)
        v = a[3:6] * 0.1
        angle = float(np.linalg.norm(v))
        if angle > 0:
            k = v / angle
            Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            turn = np.eye(3) + np.sin(angle) * Kx + (1 - np.cos(angle)) * Kx @ Kx
            self.R = turn @ self.R
        a[3:6] = 0.0
        return super().step(a)


def test_the_executor_servos_the_full_orientation_of_a_tilted_rolled_grasp():
    """Audit: execute_grasp servoed only pitch about world x and yaw, so a grasp tilted toward
    +x kept the hand vertical and a roll was never servoed."""
    from pi_embodied_services.utils import grasp as G

    f = LiberoEnvFacade(FullArm(), meta={})
    f.reset()
    tilt, roll = np.deg2rad(30), np.deg2rad(40)
    ry = np.array(
        [[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(tilt)]]
    )
    rz = np.array(
        [[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]]
    )
    target = (
        ry @ np.diag([1.0, -1.0, -1.0]) @ rz
    )  # tilted toward +x, rolled about approach
    steps, _ = f._servo_pose(np.array([0.05, 0.0, 0.2]), G.quat_xyzw(target), -1.0, 150)
    now = G.quat_xyzw_matrix(f._quat_xyzw())
    assert np.linalg.norm(G.orientation_error(now, target)) < 0.05
    assert now[:, 2] == pytest.approx(target[:, 2], abs=0.05), "the approach tilted too"
    assert steps < 150


def test_an_episode_that_ends_mid_leg_is_not_reported_as_a_stall():
    f, _ = grasp_facade()
    gid = f._rpc["env.plan_grasp"](object="bowl")["active"]

    def ends(*args, **kwargs):
        f._terminated = True  # LIBERO judged the task done on this step
        return 1, False

    f._servo_pose = ends
    out = f._rpc["env.execute_grasp"](grasp_id=gid)
    assert "stalled" not in out and "error" not in out and out["terminated"] is True


def test_a_move_longer_than_0_30_m_in_xy_is_refused_unmoved():
    f = facade()
    with pytest.raises(ValueError, match="0.3 m"):
        f.move_to([0.25, 0.25, 0.2])
    assert f._eef() == pytest.approx([0.0, 0.0, 0.2])
    assert f.move_to([0.2, 0.2, 0.3])["final_dist_m"] < 0.012, "0.28 m in xy is allowed"
    g, G = grasp_facade()
    gid = g._rpc["env.plan_grasp"](object="bowl")["active"]
    g.move_to([-0.25, 0.1, 0.3])  # now 0.32 m in xy from the pre-grasp
    with pytest.raises(ValueError, match="pre-grasp leg"):
        g.execute_grasp(gid)


class JointArmSim(ArmSim):
    """ArmSim whose raw obs also carries the joints ``env.preview_reach`` reads."""

    @property
    def current_raw_obs(self):
        obs = super().current_raw_obs
        obs[0]["robot0_joint_pos"] = np.zeros(7)
        return obs


class ShortArm:
    """A ReachPreview stand-in (``--ik``): world targets above z = 0.4 m are out of reach."""

    def __init__(self):
        self.asked = []

    def preview(self, q, pos, quat_xyzw, base_pose=None):
        self.asked.append([float(v) for v in pos])
        ok = float(pos[2]) <= 0.4
        return {
            "status": "reachable" if ok else "unreachable",
            "reachable": ok,
            "message": "ok" if ok else "misses by 1 m",
            "target": {"frame": "world", "pos": list(pos)},
        }


def test_with_ik_move_to_and_move_delta_refuse_an_unreachable_target_unmoved():
    arm = ShortArm()
    f = LiberoEnvFacade(JointArmSim(), meta={}, ik_reach=arm)
    f.reset()
    assert f.move_to([0.1, 0.05, 0.2])["final_dist_m"] < 0.012
    with pytest.raises(ValueError, match="env.move_to refused"):
        f.move_to([0.1, 0.05, 0.6])
    assert arm.asked[1] == pytest.approx([0.1, 0.05, 0.6])
    assert f.get_state()["eef_pos"] == pytest.approx([0.1, 0.05, 0.2], abs=0.012)
    f.move_to([0.1, 0.05, 0.35])
    with pytest.raises(ValueError, match="env.move_delta refused"):
        f.move_delta([0.0, 0.0, 0.1])
    assert f.get_state()["eef_pos"] == pytest.approx([0.1, 0.05, 0.35], abs=0.012)


def test_without_ik_move_to_is_unchecked():
    f = facade()
    assert f._reach is None
    assert f.move_to([0.0, 0.0, 0.6])["final_dist_m"] < 0.012


def test_business_calls_are_refused_while_a_primitive_is_abandoned():
    import threading

    f = facade()
    f._manifest_ready()
    release = threading.Event()
    stuck = threading.Thread(target=release.wait, name="run_code:move_to", daemon=True)
    stuck.start()
    f._code._abandoned.append(stuck)
    assert f._code.wedged == "move_to"
    with pytest.raises(RuntimeError, match="left running"):
        f._serve_dispatch("env.get_state", (), {}, token=f._rpc_token)
    release.set()
    stuck.join(2)
    assert f._serve_dispatch("env.get_state", (), {}, token=f._rpc_token)["eef_pos"]


# ---- the manifest (packages/embodied/src/primitives/manifests/libero.json) -----------------


def full_facade():
    """Every optional part on: SAM3 with perception, a grasp server and AnyPlace, --geometry."""
    f, _ = grasp_facade()
    g = LiberoEnvFacade(DownArm(), meta={}, sam3="http://127.0.0.1:1", geometry=True)
    import argparse

    from pi_embodied_services.utils.perception import install_perception

    install_perception(
        g,
        argparse.Namespace(sam3="http://127.0.0.1:1", unidepth=""),
        cameras=["agentview", "wrist"],
        view=g._view,
    )
    g.reset()
    return f, g


def test_the_server_checks_itself_against_its_manifest():
    """Every declared primitive is served and every served business method is declared or
    internal, with and without the optional parts."""
    from pi_embodied_services.components.manifest import ManifestError

    for f in (facade(), *full_facade()):
        f._manifest_ready()
    g = facade()
    g._rpc["env.secret_teleport"] = lambda: None
    with pytest.raises(ManifestError, match="env.secret_teleport"):
        g._manifest_ready()
    h = facade()
    del h._rpc["env.move_pose"]
    with pytest.raises(ManifestError, match="move_pose"):
        h._manifest_ready()


def test_a_tool_call_returns_every_env_step_and_a_refusal_as_its_result():
    f = facade()
    out = f._rpc["env.move_to"](xyz=[0.1, 0.0, 0.2], tool_call=True)
    steps = out["transitions"]
    assert len(steps) == out["steps_used"] > 0
    assert steps[0]["action"].shape == (7,) and steps[0]["action"][6] == -1
    assert set(steps[0]) == {"action", "obs", "reward", "terminated", "truncated"}
    assert "main_images" in steps[0]["obs"]
    far = f._rpc["env.move_to"](xyz=[0.5, 0.3, 0.2], tool_call=True)
    assert (
        "0.3 m" in far["refused"]
        and far["transitions"] == []
        and far["steps_used"] == 0
    )
    with pytest.raises(ValueError, match="0.3 m"):
        f._rpc["env.move_to"](xyz=[0.5, 0.3, 0.2])  # a program gets the error
    assert f._record is None
    # A program cannot ask for the tool's extras: the manifest does not declare tool_call.
    out = f._rpc["code.run"](
        "move_to([0.1, 0, 0.2], tool_call=True)\n", timeout_s=30, tier="low"
    )
    assert "unknown parameter(s) tool_call" in out["error"]


def test_move_to_with_a_yaw_target_turns_while_it_servos():
    f = facade()
    out = f.move_to([0.05, 0.0, 0.2], target_yaw=0.3, max_steps=10)
    assert out["steps_used"] <= 10
    assert f._yaw() > 0.05, "turned toward the target yaw"


def test_move_pose_rotate_pitch_and_release_are_the_tools_loops():
    f = LiberoEnvFacade(FullArm(), meta={})
    f.reset()
    r = f.move_pose([0.05, 0.0, 0.25], target_pitch=0.3, gripper=1)
    assert r["final_dist_m"] < 0.012 and r["final_pitch"] == pytest.approx(
        0.3, abs=0.05
    )
    assert f.get_state()["gripper_cmd"] == 1
    p = f.rotate_pitch(delta_pitch=-0.2)
    assert p["final_err"] == pytest.approx(0.0, abs=0.02)
    assert p["target_pitch"] == pytest.approx(p["start_pitch"] - 0.2, abs=1e-3)
    w = f.rotate_wrist(target_yaw=0.4, gripper=-1)
    assert w["final_yaw"] == pytest.approx(0.4, abs=0.02) and w["yaw"] == w["final_yaw"]
    with pytest.raises(ValueError, match="target_pitch or delta_pitch"):
        f.rotate_pitch()
    out = f.release(max_steps=7)
    assert (
        out["steps_used"] == 7
        and out["final_gripper_opening"] >= out["start_gripper_opening"]
    )
    assert f.get_state()["gripper_cmd"] == -1


def test_back_project_has_a_region_mode():
    f = facade()
    r = f.back_project(row_range=[200, 300], col_range=[250, 260])
    assert r["mode"] == "region" and r["center_xyz"][2] == pytest.approx(0.2, abs=1e-3)
    with pytest.raises(ValueError, match="both row_range and col_range"):
        f.back_project(row_range=[0, 10])
    with pytest.raises(ValueError, match="current 512x512"):
        f.back_project(10, 10, step=0)


# ---- CaP-X's high tier ------------------------------------------------------------------------


def sam3_facade(arm=None):
    from test_grasp import FakeSam3

    mask = np.zeros((CODE_RES, CODE_RES), bool)
    mask[240:270, 300:320] = True  # 30 x 20 px on the table plane (z = 0.2)
    f = LiberoEnvFacade(arm or DownArm(), meta={}, sam3="http://127.0.0.1:1")
    f._sam3 = FakeSam3(mask)
    f.reset()
    return f


def test_get_object_pose_and_the_fallback_grasp_come_from_sam3_and_depth():
    f = sam3_facade()
    pos, quat = f.get_object_pose("black bowl", use_multiview=False)
    # Pixel (255, 310) of the 512 image, 0.5 m below the camera: world x = (310 - 256) / 256 * 0.5.
    assert pos == pytest.approx([0.1055, -0.0, 0.2], abs=0.01)
    assert np.linalg.norm(quat) == pytest.approx(1.0)
    both, _ = f.get_object_pose("black bowl")  # both views see the same points
    assert both == pytest.approx(pos, abs=1e-3)
    gpos, gquat = f.sample_grasp_pose("black bowl", use_multiview=False)
    assert gpos[2] == pytest.approx(0.2 - 0.03, abs=1e-3), "3 cm below the top"
    # Pointing down: the hand's approach axis is world -z.
    from pi_embodied_services.utils import object_pose as op

    assert op.matrix(gquat)[:, 2] == pytest.approx([0, 0, -1], abs=1e-6)


def test_sample_grasp_pose_asks_the_grasp_server_when_there_is_one():
    f, _ = grasp_facade()
    pos, quat = f.sample_grasp_pose("bowl")
    assert pos == pytest.approx([0.05, 0.0, 0.2], abs=1e-6)
    from pi_embodied_services.utils import object_pose as op

    best = f._rpc["env.plan_grasp"](object="bowl")["candidates"][0]
    assert op.site_yaw(quat) == pytest.approx(best["eef_yaw"], abs=1e-6)


def test_goto_pose_turns_to_the_nearer_yaw_then_servos_in_legs_from_the_approach():
    from pi_embodied_services.utils import object_pose as op

    f = LiberoEnvFacade(DownArm(), meta={})
    f.reset()
    moves = []
    move_to = f._rpc["env.move_to"]
    f._rpc["env.move_to"] = lambda xyz, **kw: (
        moves.append(list(xyz)),
        move_to(xyz, **kw),
    )[1]
    q = op.hand_of_yaw(0.4)
    out = f.goto_pose([0.3, 0.1, 0.2], q, z_approach=0.1)
    assert out["final_dist_m"] < 0.012 and out["tilt_rad"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert f._yaw() == pytest.approx(0.4, abs=0.05)
    assert moves[-1] == pytest.approx([0.3, 0.1, 0.2])
    assert any(m == pytest.approx([0.3, 0.1, 0.3]) for m in moves), "0.1 m above first"
    legs = np.diff(np.array([[0.0, 0.0, 0.2]] + moves), axis=0)
    assert np.linalg.norm(legs, axis=1).max() <= 0.25 + 1e-6
    # A yaw half a turn away is the same grasp: the nearer one is taken.
    f.goto_pose([0.3, 0.1, 0.2], op.hand_of_yaw(0.4 + np.pi))
    assert f._yaw() == pytest.approx(0.4, abs=0.05)
    # The gripper command is kept (the tool's move_to default would open it).
    f.close_gripper()
    f.goto_pose([0.25, 0.1, 0.25], q)
    assert f.get_state()["gripper_cmd"] == 1
    home = f.home_pose()
    assert home["final_dist_m"] < 0.012 and f._yaw() == pytest.approx(0.0, abs=0.05)


def test_the_privileged_pose_functions_match_names_as_capx_does():
    f = facade()
    poses = {
        "milk_1": {"pos": [0.1, 0.2, 0.9], "quat_xyzw": [0, 0, 0, 1]},
        "basket_1": {"pos": [-0.1, 0.0, 0.88], "quat_xyzw": [0, 0, 0.6, 0.8]},
        "plate_1": {"pos": [0, 0, 0.9], "quat_xyzw": [0, 0, 0, 1]},
        "plate_2": {"pos": [1, 0, 0.9], "quat_xyzw": [0, 0, 0, 1]},
    }
    f.ground_truth_poses = lambda names=None: {"frame": "world", "poses": poses}
    assert f.get_object_pose_privileged("milk") == [
        [0.1, 0.2, 0.9],
        [1.0, 0.0, 0.0, 0.0],
    ]
    assert f.get_object_pose_privileged("woven basket")[1] == [0.8, 0.0, 0.0, 0.6]
    assert f.get_object_pose_privileged("plate")[0] == [0, 0, 0.9], "plate_1"
    with pytest.raises(KeyError, match="Available objects"):
        f.get_object_pose_privileged("stove")
    assert f.sample_grasp_pose_privileged("milk") == [
        [0.1, 0.2, 0.9],
        [0.0, 1.0, 0.0, 0.0],
    ]


def test_a_high_tier_program_runs_capx_calls_on_the_server():
    f = sam3_facade()
    out = f._rpc["code.run"](
        "open_gripper()\n"
        "pos, quat = sample_grasp_pose('black bowl', use_multiview=False)\n"
        "goto_pose(pos, quat, z_approach=0.1)\n"
        "close_gripper()\n"
        "RESULT = [round(v, 3) for v in get_object_pose('black bowl', use_multiview=False)[0]]\n",
        timeout_s=60,
        tier="high",
    )
    assert out["status"] == "ran", out
    assert [c["name"] for c in out["calls"]] == [
        "open_gripper",
        "sample_grasp_pose",
        "goto_pose",
        "close_gripper",
        "get_object_pose",
    ]
    assert out["steps"] > 100
    out = f._rpc["code.run"]("move_to([0, 0, 0.3])\n", timeout_s=30, tier="high")
    assert "NameError" in out["error"], "the high tier has only CaP-X's functions"
