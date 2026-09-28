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

"""The geometric toolset (utils/geometry.py) on the LIBERO facade over a mock LiberoEnv: the
point-cloud views and their pixel mapping, two-view and camera marks, grip-site targets from
approach/jaw directions, the frozen close preview, the server-side move with its residual and
contacts, and the MuJoCo reader over a fake sim."""

from __future__ import annotations

import base64
import io
import math

import numpy as np
import pytest
from PIL import Image

from pi_embodied_services.robots.libero.env_server import LiberoEnvFacade
from pi_embodied_services.utils import geometry as geo
from pi_embodied_services.utils.geometry import (
    GeometryError,
    matrix_to_quat,
    mujoco_grip_state,
    quat_to_matrix,
    resolve_rotation,
    rotation_angle,
)


def rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rotx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def expmap(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    t = np.linalg.norm(v)
    if t < 1e-12:
        return np.eye(3)
    k = v / t
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


#: The grip site in the hand frame (LIBERO's eef quaternion is the hand's), and the pads, which
#: lie along the site's Y: the grip frame is the site turned a quarter about Z.
SITE_IN_HAND = rotx(0.3) @ rotz(-0.4)


class ArmSim:
    """A LiberoEnv stand-in with a full OSC pose: action[:3] * 0.05 m and a world-frame rotation
    vector action[3:6] * 0.1 rad per step; a table at z = 0 seen by a camera 0.7 m above looking
    down, with a 10 cm tall box under pixels 20..40 (of 64); the fingers touch the table below
    z = 0.01 (a contact)."""

    def __init__(self):
        self.pos = np.array([0.0, 0.0, 0.3])
        # Pointing down: the site's +Z is world -Z.
        self.hand = rotx(math.pi) @ SITE_IN_HAND.T
        self.width = 0.08
        self.steps = 0
        self.workers = [self]

    @property
    def site(self) -> np.ndarray:
        return self.hand @ SITE_IN_HAND

    @property
    def current_raw_obs(self):
        half = self.width / 2
        return [
            {
                "robot0_eef_pos": self.pos.copy(),
                "robot0_eef_quat": np.array(matrix_to_quat(self.hand)),
                "robot0_gripper_qpos": np.array([half, -half]),
            }
        ]

    def _obs(self):
        return {
            "main_images": np.zeros((1, 4, 4, 3), dtype=np.uint8),
            "wrist_images": np.zeros((1, 4, 4, 3), dtype=np.uint8),
            "states": np.zeros((1, 8), dtype=np.float32),
        }

    def _info(self):
        return {"episode": {"success_once": np.array([False])}}

    def reset(self):
        self.__init__()
        return self._obs(), self._info()

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(7)
        self.steps += 1
        self.pos = self.pos + a[:3] * 0.05
        self.pos[2] = max(self.pos[2], 0.005)
        self.hand = expmap(a[3:6] * 0.1) @ self.hand
        if a[6] > 0:
            self.width = max(0.02, self.width - 0.02)
        elif a[6] < 0:
            self.width = min(0.08, self.width + 0.02)
        zeros = np.zeros(1, dtype=bool)
        return self._obs(), np.zeros(1), zeros, zeros, self._info()

    def render_camera(self, camera_name, height, width, depth):
        rgb = np.full((height, width, 3), 90, dtype=np.uint8)
        d = np.full((height, width), 0.7, dtype=np.float32)
        s = slice(height * 20 // 64, height * 40 // 64)
        # Raw LIBERO frames are upside down; the facade flips rows.
        d[::-1][s, s] = 0.6
        rgb[::-1][s, s] = (200, 30, 30)
        return (rgb, d) if depth else rgb

    def get_camera_meta(self, camera_name, height, width):
        f = height / 2
        return {
            "intrinsic_K": [[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]],
            "extrinsic_cam2world": [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0.7],
                [0, 0, 0, 1],
            ],
        }

    def env_call(self, name, target):
        if name == "robot_base_pose":
            return {"pos": [-0.5, 0.0, 0.0], "quat_xyzw": [0, 0, 0, 1]}
        if name == "grip_geometry":
            contacts = (
                [
                    {
                        "xyz_m": [*self.pos[:2].tolist(), 0.0],
                        "robot_geom": "gripper0_finger1",
                        "other_geom": "table",
                    }
                ]
                if self.pos[2] < 0.01
                else []
            )
            return {
                "site_xyz": self.pos.tolist(),
                "site_xmat": self.site.tolist(),
                "pads_local": [
                    [0.0, -self.width / 2, 0.01],
                    [0.0, self.width / 2, 0.01],
                ],
                "contacts": contacts,
            }
        return {"error": f"unknown {name}"}

    task_descriptions = ["lift the box"]

    @property
    def env(self):
        return self


def facade(geometry=True) -> LiberoEnvFacade:
    f = LiberoEnvFacade(ArmSim(), meta={}, geometry=geometry)
    f._rpc["env.reset"]()
    return f


def call(f, method, **kw):
    return f._rpc[method](**kw)


def decode(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))


GEOMETRY_RPC = (
    "env.point_views",
    "env.mark_point",
    "env.grip_target",
    "env.grip_state",
    "env.move_grip",
)


def test_off_by_default_nothing_is_served():
    f = facade(geometry=False)
    assert not any(m in f._rpc for m in GEOMETRY_RPC)
    names = f._rpc["code.api"]("low")["available"]
    assert "mark_point" not in names and "move_grip" not in names


def test_the_manifest_lists_the_toolset_in_the_low_tier():
    """The geometry parts are motion and perception parts: the low tier (manifests/common/geometry.json)."""
    f = facade()
    assert all(m in f._rpc for m in GEOMETRY_RPC)
    low = f._rpc["code.api"]("low")["available"]
    for name in ("view_points", "mark_point", "grip_target", "grip_state", "move_grip"):
        assert name in low
    f._manifest_ready()
    docs = {p["name"]: p["doc"] for p in f._code.api("low")}
    assert (
        "Moves the robot." in docs["move_grip"]
        and "Moves the robot." not in docs["mark_point"]
    )
    assert "move_grip" not in f._rpc["code.api"]("high")["available"]


def test_grip_frame_is_measured_from_the_site_and_its_pads():
    f = facade()
    p, R = f._geometry.grip()
    sim = f._env
    np.testing.assert_allclose(p, sim.pos)
    # Approach = the site's +Z (down), jaw = the pads' axis (the site's Y).
    np.testing.assert_allclose(R[:, 2], sim.site[:, 2], atol=1e-6)
    np.testing.assert_allclose(R[:, 0], sim.site[:, 1], atol=1e-6)
    np.testing.assert_allclose(R[:, 2], [0, 0, -1], atol=1e-6)


def test_point_views_map_pixels_to_world_axes():
    f = facade()
    out = call(f, "env.point_views")
    assert [v["view"] for v in out["views"]] == list(geo.SCENE_VIEWS)
    top = out["views"][0]
    assert top["right"] == "+x" and top["up"] == "+y"
    img = decode(out["images"][0])
    assert img.shape == (top["height"], top["width"], 3)
    spec = f._geometry._specs["pointcloud_top"]
    for p in ([0.05, -0.02, 0.1], [0.12, 0.08, 0.0]):
        x, y = spec.to_pixel(np.array(p))
        back = spec.to_world(x, y)
        assert back[0] == pytest.approx(p[0]) and back[1] == pytest.approx(p[1])
    # The box top (z = 0.1) is drawn above the table in the front view: red pixels exist.
    front = decode(out["images"][1])
    assert (front[..., 0] > 200).any()
    cams = call(f, "env.point_views", views=["agentview"])
    assert cams["views"][0]["camera"] is True
    with pytest.raises(GeometryError, match="unknown view"):
        call(f, "env.point_views", views=["pointcloud_back"])


def test_two_orthographic_clicks_solve_a_free_space_point():
    f = facade()
    call(f, "env.point_views")
    specs = f._geometry._specs
    goal = np.array([0.03, -0.04, 0.2])  # above the box: free space
    tx, ty = specs["pointcloud_top"].to_pixel(goal)
    first = call(
        f,
        "env.mark_point",
        point_id="P1",
        view="pointcloud_top",
        x=round(tx),
        y=round(ty),
    )
    assert first["status"] == "pending"
    assert set(first["needs"]) == {"pointcloud_front", "pointcloud_side"}
    assert len(first["images"]) == 2
    with pytest.raises(GeometryError, match="still pending"):
        f._geometry.mark("P1")
    # A front click that disagrees on x by 5 cm is refused and the pending half kept.
    fx, fy = specs["pointcloud_front"].to_pixel(goal + [0.05, 0, 0])
    bad = call(
        f,
        "env.mark_point",
        point_id="P1",
        view="pointcloud_front",
        x=round(fx),
        y=round(fy),
    )
    assert bad["status"] == "inconsistent" and bad["shared_axis"] == "x"
    fx, fy = specs["pointcloud_front"].to_pixel(goal)
    solved = call(
        f,
        "env.mark_point",
        point_id="P1",
        view="pointcloud_front",
        x=round(fx),
        y=round(fy),
    )
    assert solved["status"] == "solved"
    np.testing.assert_allclose(solved["xyz_m"], goal, atol=0.003)
    assert solved["source"] == ["pointcloud_top", "pointcloud_front"]
    assert call(f, "env.point_views")["marks"]["P1"] == solved["xyz_m"]


def test_a_camera_click_is_the_visible_surface():
    f = facade()
    call(f, "env.point_views", views=["agentview"])
    # The box occupies rows/cols 160..320 of the 512 image: its top is at z = 0.1.
    out = call(f, "env.mark_point", point_id="B", view="agentview", x=240, y=240)
    assert out["status"] == "solved"
    assert out["xyz_m"][2] == pytest.approx(0.1, abs=1e-3)
    table = call(f, "env.mark_point", point_id="T", view="agentview", x=20, y=20)
    assert table["xyz_m"][2] == pytest.approx(0.0, abs=1e-3)


def test_views_and_pending_clicks_expire_when_the_robot_moves():
    f = facade()
    call(f, "env.point_views")
    call(f, "env.mark_point", point_id="P", view="pointcloud_top", x=100, y=100)
    call(f, "env.move_delta", dxyz=[0, 0, 0.02])
    with pytest.raises(GeometryError, match="not rendered for the current state"):
        call(f, "env.mark_point", point_id="P", view="pointcloud_front", x=100, y=100)
    call(f, "env.point_views")
    again = call(
        f, "env.mark_point", point_id="P", view="pointcloud_front", x=100, y=100
    )
    assert again["status"] == "pending", "the old half was dropped"


def test_directions_resolve_to_the_nearest_grip_frame():
    cur = rotx(math.pi)  # pointing down, jaw +x
    R, how = resolve_rotation(cur, approach=[1, 0, 0])
    assert how == "approach"
    np.testing.assert_allclose(R[:, 2], [1, 0, 0], atol=1e-9)
    assert abs(np.linalg.det(R) - 1) < 1e-9
    # A jaw along -x is the same gripper as +x: the nearer sign is kept (no half turn).
    R2, _ = resolve_rotation(cur, jaw=[-1, 0, 0])
    assert rotation_angle(cur, R2) < 1e-9
    R3, how3 = resolve_rotation(cur, approach=[0, 0, -1], jaw=[0, 1, 0])
    assert how3 == "approach_and_jaw"
    assert rotation_angle(cur, R3) == pytest.approx(math.pi / 2)
    with pytest.raises(GeometryError, match="parallel"):
        resolve_rotation(cur, approach=[0, 0, -1], jaw=[0, 0, 1])


def test_grip_target_resolves_positions_and_refuses_ambiguity():
    f = facade()
    plan = call(f, "env.grip_target", delta_mm=[10, 0, 0], delta_frame="grip_site")
    assert plan["status"] == "execute" and plan["motion"]
    jaw = np.asarray(plan["current"]["jaw_world"])
    np.testing.assert_allclose(np.asarray(plan["delta_mm"]), jaw * 10, atol=0.2)
    plan = call(f, "env.grip_target", xyz=[0.1, 0.0, 0.2], approach=[0, 0, -1])
    assert plan["target"]["grip_xyz_m"] == [0.1, 0.0, 0.2]
    # tool_quat_xyzw is the hand's orientation that puts the grip frame at the target.
    R = quat_to_matrix(plan["target"]["tool_quat_xyzw"]) @ f._geometry._grip_frame()
    np.testing.assert_allclose(R[:, 2], [0, 0, -1], atol=1e-4)
    with pytest.raises(GeometryError, match="at most one"):
        call(f, "env.grip_target", xyz=[0, 0, 0.2], delta_mm=[0, 0, 1])
    with pytest.raises(GeometryError, match="no solved point"):
        call(f, "env.grip_target", point_id="nope")
    with pytest.raises(GeometryError, match="give a position"):
        call(f, "env.grip_target")
    preview = call(f, "env.grip_target", xyz=[0.1, 0.0, 0.2], preview=True)
    assert preview["status"] == "preview" and "preview_id" not in preview
    assert preview["views"] == ["agentview", *geo.PREVIEW_VIEWS]
    assert len(preview["images"]) == 4


def test_a_close_is_previewed_frozen_and_executed_only_unchanged():
    f = facade()
    plan = call(f, "env.grip_target", delta_mm=[0, 0, -50], gripper="close")
    assert plan["status"] == "preview" and plan["preview_id"].startswith("pv")
    pid = plan["preview_id"]
    with pytest.raises(GeometryError, match="takes no other target field"):
        call(f, "env.grip_target", execute_preview_id=pid, gripper="close")
    with pytest.raises(GeometryError, match="is not pending"):
        call(f, "env.grip_target", execute_preview_id="pv00000000")
    # The preview views are markable: a point in the zoomed top view.
    mark = call(f, "env.mark_point", point_id="Z", view="preview_top", x=10, y=10)
    assert mark["status"] == "pending"
    commit = call(f, "env.grip_target", execute_preview_id=pid)
    assert commit["status"] == "execute" and commit["gripper"] == "close"
    assert commit["target"]["grip_xyz_m"][2] == pytest.approx(0.25)
    with pytest.raises(GeometryError, match="is not pending"):
        call(f, "env.grip_target", execute_preview_id=pid)
    # A preview whose gripper moved since is stale.
    stale = call(f, "env.grip_target", delta_mm=[0, 0, -50], gripper="close")
    call(f, "env.move_delta", dxyz=[0.01, 0, 0])
    with pytest.raises(GeometryError, match="is stale"):
        call(f, "env.grip_target", execute_preview_id=stale["preview_id"])


def test_move_grip_reaches_a_pose_then_closes():
    f = facade()
    out = call(f, "env.move_grip", xyz=[0.05, 0.02, 0.2], approach=[0.3, 0, -1])
    assert out["motion_status"] == "reached", out
    assert out["remaining_distance_mm"] <= 5
    app = np.asarray([0.3, 0, -1]) / np.linalg.norm([0.3, 0, -1])
    _, R = f._geometry.grip()
    assert float(np.dot(R[:, 2], app)) > math.cos(0.05)
    assert out["contacts"] == []
    close = call(f, "env.move_grip", gripper="close")
    assert close["motion_status"] == "previewed" and "images" not in close
    done = call(f, "env.move_grip", execute_preview_id=close["preview_id"])
    assert done["gripper"] == "close" and done["motion_status"] == "not_requested"
    assert done["gripper_width"] < 0.08


def test_move_grip_not_reached_skips_the_gripper_and_reports_residual_and_contacts():
    f = facade()
    # Below the table: the arm stops at z = 0.005 and the finger touches it.
    out = call(f, "env.move_grip", xyz=[0.0, 0.0, -0.05], max_steps=40)
    assert out["motion_status"] == "not_reached"
    assert out["remaining_delta_mm"][2] == pytest.approx(-55, abs=1)
    assert out["contacts"] and out["contacts"][0]["other_geom"] == "table"
    state = call(f, "env.grip_state", target_xyz=[0.0, 0.0, -0.05])
    assert state["motion_status"] == "not_reached"
    assert state["views"] == ["agentview_contacts"] and len(state["images"]) == 1
    plan = call(f, "env.move_grip", xyz=[0.0, 0.0, -0.05], gripper="open", max_steps=5)
    assert plan["gripper_skipped"] and "gripper" not in plan


def test_code_mode_counts_move_grip_translation():
    f = facade()
    assert f._code_move_m("env.move_grip", {"xyz": [0.0, 0.0, 0.1]}) == pytest.approx(
        0.2
    )
    assert f._code_move_m("env.move_grip", {"gripper": "open"}) == 0.0
    assert f._code_move_m("env.move_grip", {"point_id": "missing"}) == 0.0


def test_reset_forgets_the_marks():
    f = facade()
    call(f, "env.point_views", views=["agentview"])
    call(f, "env.mark_point", point_id="B", view="agentview", x=240, y=240)
    f._rpc["env.reset"]()
    with pytest.raises(GeometryError, match="no solved point"):
        f._geometry.mark("B")


class FakeContact:
    def __init__(self, pos, g1, g2):
        self.pos, self.geom1, self.geom2 = np.asarray(pos), g1, g2


class FakeSim:
    """A MuJoCo sim's surface as robosuite's MjSim exposes it."""

    class model:
        names = {
            0: "gripper0_finger1_pad_collision",
            1: "gripper0_finger2_pad_collision",
            2: "table",
            3: "robot0_link7",
        }
        geom_size = np.array([[0.01, 0.005, 0.02]] * 4)

        @staticmethod
        def site_name2id(name):
            assert name == "gripper0_grip_site"
            return 0

        @classmethod
        def geom_name2id(cls, name):
            return {v: k for k, v in cls.names.items()}[name]

        @classmethod
        def geom_id2name(cls, i):
            return cls.names[i]

    class data:
        site_xpos = np.array([[0.0, 0.0, 0.1]])
        site_xmat = np.eye(3).reshape(1, 9)
        geom_xpos = np.array([[-0.03, 0, 0.1], [0.03, 0, 0.1], [0, 0, 0], [0, 0, 0.3]])
        geom_xmat = np.tile(np.eye(3).reshape(9), (4, 1))
        ncon = 3
        contact = [
            FakeContact([0.0, 0.0, 0.08], 0, 2),  # a pad on the table
            FakeContact([0.0, 0.001, 0.08], 1, 2),  # the same patch, 1 mm away
            FakeContact([0.0, 0.0, 0.3], 3, 0),  # the robot touching itself
        ]


def test_mujoco_grip_state_reads_the_site_pads_and_world_contacts():
    out = mujoco_grip_state(FakeSim())
    assert out["site_xyz"] == [0.0, 0.0, 0.1]
    # The pads' inner faces, 1 cm (their half size) toward the site along x.
    np.testing.assert_allclose(
        out["pads_local"], [[-0.02, 0, 0], [0.02, 0, 0]], atol=1e-9
    )
    assert out["contacts"] == [
        {
            "xyz_m": [0.0, 0.0, 0.08],
            "robot_geom": "gripper0_finger1_pad_collision",
            "other_geom": "table",
        }
    ]
    np.testing.assert_allclose(geo.jaw_frame(out["pads_local"]), np.eye(3))
    assert "error" in mujoco_grip_state(object())


# -- the single Franka (both backends share grasp_views.franka_geometry) ---------------------


def polymetis(monkeypatch, geometry=True):
    """The Polymetis facade over its test doubles, the TCP pointing down at (0.5, 0, 0.3), the
    third-person camera 0.9 m above the table looking down."""
    import test_franka_polymetis as tfp

    from pi_embodied_services.robots.franka import perception as franka_perception
    from pi_embodied_services.robots.franka_polymetis.env_server import (
        FrankaPolymetisFacade,
    )
    from pi_embodied_services.robots.franka_polymetis.mock import (
        MOCK_ENV,
        MockPolymetisRobot,
        MockRGBD,
    )

    monkeypatch.setenv(MOCK_ENV, "1")
    down = np.array([[1, 0, 0, 0.5], [0, -1, 0, 0.0], [0, 0, -1, 0.9], [0, 0, 0, 1.0]])
    monkeypatch.setattr(
        franka_perception,
        "load_calibration_bundle",
        lambda: {
            "external": {"matrix": down.tolist()},
            "wrist": {"matrix": np.eye(4).tolist()},
        },
    )
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *tfp.DOWN))
    cams = {"wrist": MockRGBD("1"), "third_person": MockRGBD("2", depth_m=0.9)}
    return FrankaPolymetisFacade(
        tfp.cfg(), robot, cams, sleep=lambda s: None, geometry=geometry
    )


def test_franka_plans_only_and_its_jaw_is_the_tcp_y(monkeypatch):
    off = polymetis(monkeypatch, geometry=False)
    assert not any(m in off._rpc for m in GEOMETRY_RPC)
    f = polymetis(monkeypatch)
    for m in GEOMETRY_RPC[:4]:
        assert m in f._rpc
    assert "env.move_grip" not in f._rpc, "a real arm's client executes the plans"
    names = f._rpc["code.api"]("low")["available"]
    assert "grip_target" in names and "move_grip" not in names
    kit = f._rpc["env.grip_target"].__self__
    p, R = kit.grip()
    tcp = np.asarray(f.get_robot_state()["raw_base_state"]["tcp_pose"])
    np.testing.assert_allclose(p, tcp[:3])
    tcp_R = quat_to_matrix(tcp[3:])
    np.testing.assert_allclose(R[:, 0], tcp_R[:, 1], atol=1e-9)
    np.testing.assert_allclose(R[:, 2], tcp_R[:, 2], atol=1e-9)
    views = f._rpc["env.point_views"](views=["pointcloud_top", "third_person"])
    assert views["views"][0]["right"] == "+x" and views["views"][1]["camera"] is True
    plan = f._rpc["env.grip_target"](delta_mm=[0, 0, -20], jaw=[1, 0, 0])
    assert plan["status"] == "execute"
    # The TCP quaternion that puts the jaw along world x: its Y axis is +-x.
    q = quat_to_matrix(plan["target"]["tool_quat_xyzw"])
    assert abs(q[0, 1]) == pytest.approx(1, abs=1e-4)
    state = f._rpc["env.grip_state"](target_xyz=plan["target"]["grip_xyz_m"])
    assert state["motion_status"] == "not_reached" and "contacts" not in state
    assert state["gripper_width"] >= 0


def test_franka_rlinf_serves_the_same_toolset():
    import test_grasp

    from pi_embodied_services.robots.franka.env_server import FrankaEnvFacade

    f = FrankaEnvFacade(test_grasp._FrankaBackend(), geometry=True)
    assert "env.grip_target" in f._rpc and "env.move_grip" not in f._rpc
    assert "grip_target" in f._rpc["code.api"]("low")["available"]
    assert "env.grip_target" not in FrankaEnvFacade(test_grasp._FrankaBackend())._rpc


class Robosuite15Sim:
    """robosuite 1.5's naming (``gripper0_right_*``): the pads lie along the site's Y."""

    def __init__(self, site_R: np.ndarray):
        names = {
            0: "gripper0_right_finger1_pad_collision",
            1: "gripper0_right_finger2_pad_collision",
            2: "table_collision",
        }
        site = np.array([0.1, 0.0, 0.9])
        pads = [
            site + site_R @ [0, -0.03, 0],
            site + site_R @ [0, 0.03, 0],
            [0, 0, 0.8],
        ]

        class model:
            nsite, ngeom = 2, 3
            geom_size = np.array([[0.005, 0.01, 0.02]] * 3)

            @staticmethod
            def site_name2id(name):
                raise ValueError(f"no site {name}")

            @staticmethod
            def site_id2name(i):
                return [
                    "gripper0_right_grip_site_cylinder",
                    "gripper0_right_grip_site",
                ][i]

            @staticmethod
            def geom_name2id(name):
                raise ValueError(f"no geom {name}")

            @staticmethod
            def geom_id2name(i):
                return names[i]

        class data:
            site_xpos = np.array([[0, 0, 0], site])
            site_xmat = np.array([np.eye(3).reshape(9), site_R.reshape(9)])
            geom_xpos = np.asarray(pads, dtype=np.float64)
            geom_xmat = np.array([site_R.reshape(9)] * 2 + [np.eye(3).reshape(9)])
            ncon = 1
            contact = [FakeContact(pads[0] - [0, 0, 0.02], 0, 2)]

        self.model, self.data = model, data


def test_robosuite_kit_finds_the_15_names_and_measures_its_grip_frame():
    from pi_embodied_services.robots.robosuite import tasks
    from pi_embodied_services.robots.robosuite.env_server import RobosuiteEnvFacade

    site_R = rotx(math.pi) @ rotz(0.3)
    f = object.__new__(RobosuiteEnvFacade)
    f._task_name, f._task, f._steps = "Lift", tasks.TASKS["Lift"], 0
    f._env = type("E", (), {"sim": Robosuite15Sim(site_R)})()
    hand = site_R @ rotx(0.2).T  # the eef quaternion's frame differs from the site's
    f._eef = lambda i: (np.array([0.1, 0.0, 0.9]), np.array(matrix_to_quat(hand)))
    f._robot_state = lambda: {"robot0_gripper_width": 0.06}
    f._view = lambda camera: {
        "rgb": np.zeros((32, 32, 3), dtype=np.uint8),
        "depth": np.full((32, 32), 0.5, dtype=np.float32),
        "intrinsic_K": np.array([[16.0, 0, 16], [0, 16.0, 16], [0, 0, 1]]),
        "extrinsic_cam2world": np.array(
            [[1.0, 0, 0, 0.1], [0, -1, 0, 0], [0, 0, -1, 1.3], [0, 0, 0, 1]]
        ),
    }
    kit = f._geometry_kit()
    p, R = kit.grip()
    np.testing.assert_allclose(R[:, 2], site_R[:, 2], atol=1e-9)
    np.testing.assert_allclose(R[:, 0], site_R[:, 1], atol=1e-9)
    state = kit.grip_state()
    assert state["gripper_width"] == 0.06 and state["closed_on_nothing"] is False
    assert [c["other_geom"] for c in state["contacts"]] == ["table_collision"]
    assert state["views"] == ["agentview_contacts"] and len(state["images"]) == 1
