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

"""Grasp and placement primitives (utils/grasp.py) with mock model servers: the frame
conversions against known poses, the world transform and grasp-to-EEF calibration, short ids
that expire with the observation, and the same-snapshot rule of plan_place."""

from __future__ import annotations

import numpy as np
import pytest

from pi_embodied_services.components.grasp_server_base import GraspServer
from pi_embodied_services.utils import grasp as G
from pi_embodied_services.utils.detections import DetectionBook

# --- frames ---------------------------------------------------------------------------------


def _rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array(
        [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]]
    )


def test_contact_graspnet_native_frame_becomes_graspnet_frame():
    # Native: Z is the approach, X the closing direction, origin at the gripper base. A native
    # identity pose approaches along camera +z; the GraspNet X must then be camera +z, Y camera +x.
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.2, 0.5]
    [c] = G.contact_graspnet_candidates([T], [0.9], [[0.1, 0.2, 0.6034]], [0.04])
    R = np.asarray(c["rotation_matrix"])
    assert np.allclose(R[:, 0], [0, 0, 1]), "approach"
    assert np.allclose(R[:, 1], [1, 0, 0]), "closing"
    assert G.is_rotation(R)
    # The grasp center is the base moved by the Panda gripper depth along the approach.
    assert np.allclose(
        c["translation_xyz"], [0.1, 0.2, 0.5 + G.CONTACT_GRASPNET_GRIPPER_DEPTH]
    )
    # The two contacts straddle the center along the closing direction, width apart.
    a, b = np.asarray(c["contact_points_xyz"])
    assert np.isclose(np.linalg.norm(a - b), 0.04)
    assert np.allclose((a + b) / 2, c["translation_xyz"])
    assert c["gripper_base_xyz"] == [0.1, 0.2, 0.5]
    # A rotated native pose rotates the same way.
    T2 = np.eye(4)
    T2[:3, :3] = _rot_z(90)
    [c2] = G.contact_graspnet_candidates([T2], [0.5], [[0, 0, 0]], [0.02])
    assert np.allclose(np.asarray(c2["rotation_matrix"])[:, 1], [0, 1, 0]), (
        "closing turned to +y"
    )
    assert np.allclose(np.asarray(c2["rotation_matrix"])[:, 0], [0, 0, 1]), (
        "approach unchanged"
    )


def test_graspgenx_native_frame_uses_the_gripper_fingertip_as_center():
    T = np.eye(4)
    T[:3, :3] = _rot_z(-90)
    T[:3, 3] = [0, 0, 1]
    [c] = G.graspgenx_candidates(
        [T], [0.7], fingertip_xyz=[0, 0, 0.1], width=0.08, tags=["diff"]
    )
    R = np.asarray(c["rotation_matrix"])
    assert np.allclose(R[:, 0], [0, 0, 1])
    assert np.allclose(R[:, 1], [0, -1, 0])
    assert np.allclose(c["translation_xyz"], [0, 0, 1.1])
    assert c["candidate_source"] == "diffusion" and c["width"] == 0.08


def test_anygrasp_is_already_the_graspnet_frame():
    class Gr:
        score, width, depth, height = 0.8, 0.05, 0.02, 0.03
        rotation_matrix = _rot_z(30)
        translation = np.array([0.3, 0.0, 0.7])

    [c] = G.anygrasp_candidates([Gr()])
    assert np.allclose(c["rotation_matrix"], _rot_z(30))
    assert c["translation_xyz"] == [0.3, 0.0, 0.7] and c["depth"] == 0.02


def test_placement_composition_moves_the_pick_grasp_with_the_object():
    T = np.eye(4)
    T[:3, :3] = _rot_z(90)
    T[:3, 3] = [1, 0, 0]
    R, t = G.compose_placement(T, np.eye(3), np.array([0.5, 0, 0]))
    assert np.allclose(R, _rot_z(90))
    assert np.allclose(t, [1, 0.5, 0])
    with pytest.raises(G.GraspError):
        G.compose_placement(np.diag([2.0, 1, 1, 1]), np.eye(3), np.zeros(3))


def test_grasp_to_eef_default_is_the_panda_hand():
    cal = G.GraspToEef()
    # A grasp approaching along world -z with the fingers closing along world y: the EEF points
    # down (+z of the EEF = approach), its yaw is zero, its pitch zero (LIBERO's pointing-down).
    R = np.array(
        [[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]]
    )  # columns approach, closing, normal
    pose = cal.eef_pose(R, np.array([0.1, 0.2, 0.3]))
    assert pose["eef_position"] == [0.1, 0.2, 0.3]
    assert abs(pose["eef_pitch"]) < 1e-6
    T = G.rigid(R, [0.1, 0.2, 0.3]) @ cal.matrix()
    assert np.allclose(T[:3, 2], [0, 0, -1]), "EEF +z is the approach"
    assert np.allclose(T[:3, 1], [0, 1, 0]), "EEF +y is the closing direction"
    # A translation offset moves the EEF origin along the grasp frame.
    off = G.GraspToEef(translation=(-0.02, 0, 0))
    assert np.allclose(off.eef_pose(R, np.zeros(3))["eef_position"], [0, 0, 0.02])
    with pytest.raises(G.GraspError):
        G.GraspToEef.from_config({"rotation": np.eye(3) * 2})


# --- planner ------------------------------------------------------------------------------


class FakeServer:
    """A grasp server whose `plan` returns fixed camera-frame candidates."""

    def __init__(self, candidates, grasp_frame="graspnet"):
        self.candidates = candidates
        self.grasp_frame = grasp_frame
        self.calls = []

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, kwargs))
        if method.endswith(".plan"):
            return {
                "grasp_frame": self.grasp_frame,
                "candidates": self.candidates,
                "latency_s": 0.01,
            }
        raise AssertionError(method)


class FakeAnyPlace:
    def __init__(self, transforms):
        self.transforms = transforms

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        assert method == "anyplace.plan"
        return {
            "placements": [
                {"rank": i, "score": None, "transform_matrix": T.tolist()}
                for i, T in enumerate(self.transforms)
            ]
        }


class FakeSam3:
    def __init__(self, mask):
        self.mask = mask

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        import base64
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray((self.mask * 255).astype(np.uint8), mode="L").save(
            buf, format="PNG"
        )
        return {
            "found": True,
            "score": 0.9,
            "mask_png_base64": base64.b64encode(buf.getvalue()).decode(),
        }


H = W = 16
K = np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1]])
#: A camera 1 m above the table looking straight down, its x along world y (OpenCV: z forward).
CAM2WORLD = np.array([[0, -1, 0, 0.0], [-1, 0, 0, 0.0], [0, 0, -1, 1.0], [0, 0, 0, 1]])


def _view(camera: str) -> dict:
    depth = np.full((H, W), 0.9, dtype=np.float32)  # the table
    depth[4:12, 4:12] = 0.8  # a 10 cm high block in the middle
    return {
        "rgb": np.zeros((H, W, 3), np.uint8),
        "depth": depth,
        "intrinsic_K": K,
        "extrinsic_cam2world": CAM2WORLD,
    }


def _block_mask():
    m = np.zeros((H, W), bool)
    m[4:12, 4:12] = True
    return m


def _camera_candidate(score: float, x: float):
    # Approach along camera +z (down onto the table), closing along camera +x, center on the block top.
    return G.make_candidate(
        score=score,
        rotation=G.ZX_NATIVE_TO_GRASPNET,
        center=[x, 0.0, 0.8],
        width=0.04,
        depth=0.0,
        source_model="fake",
    )


def _planner(**kw):
    server = FakeServer([_camera_candidate(0.2, 0.01), _camera_candidate(0.9, 0.0)])
    planner = G.GraspPlanner(
        _view,
        cameras=["agentview", "wrist"],
        backends={"contact_graspnet": server},
        wrist_camera="wrist",
        **kw,
    )
    return planner, server


def test_plan_grasp_transforms_to_world_ranks_and_hands_out_ids():
    planner, server = _planner(sam3=FakeSam3(_block_mask()))
    out = planner.plan_grasp(object="block")
    assert out["backend"] == "contact_graspnet" and out["candidate_count"] == 2
    assert out["mask_id"].startswith("d") and [c["id"] for c in out["candidates"]] == [
        "g2",
        "g3",
    ]
    best = out["candidates"][0]
    assert best["score"] == 0.9 and best["rank"] == 0 and out["active"] == "g2"
    # Camera +z is world -z (the camera looks down): the approach points down, the grasp center is
    # 0.2 m above the table plane (camera at z = 1, block top at 0.8 m depth).
    assert np.allclose(best["approach"], [0, 0, -1])
    assert np.allclose(best["position"], [0, 0, 0.2], atol=1e-6)
    assert np.allclose(best["eef_position"], best["position"])
    assert np.allclose(best["closing"], [0, -1, 0]), "camera +x is world -y"
    # The server saw the mask, metric depth and the world up in the camera frame (-z).
    method, kwargs = server.calls[0]
    assert method == "contact_graspnet.plan"
    assert kwargs["mask"].dtype == np.uint8 and kwargs["mask"].sum() == 64
    assert np.allclose(kwargs["up_direction_camera"], [0, 0, -1])
    # resolve: the standoff backs off against the approach (up).
    pose = planner.resolve_grasp("g2", standoff=0.1)
    assert np.allclose(pose["eef_position"], [0, 0, 0.3], atol=1e-6)


def test_ids_expire_when_the_robot_moves():
    planner, _ = _planner(sam3=FakeSam3(_block_mask()))
    out = planner.plan_grasp(object="block")
    gid, mid = out["active"], out["mask_id"]
    planner.invalidate()
    with pytest.raises(G.GraspError, match="stale"):
        planner.resolve_grasp(gid)
    with pytest.raises(G.GraspError, match="stale"):
        planner.plan_grasp(mask_id=mid)
    # The next plan reports what was dropped, and never reuses an id.
    again = planner.plan_grasp(object="block")
    assert set(again["expired_ids"]) >= {gid, mid}
    assert again["active"] not in (gid, mid)
    with pytest.raises(G.GraspError, match="unknown"):
        planner.resolve_grasp("g999")


def test_install_wraps_mutating_rpcs_so_they_invalidate():
    planner, _ = _planner(sam3=FakeSam3(_block_mask()))

    class Facade:
        _rpc = {
            "env.step": lambda action: "stepped",
            "env.render_camera": lambda: "frame",
        }
        _readonly_methods = set()

    f = Facade()
    planner.install(f)
    gid = planner.plan_grasp(object="block")["active"]
    assert f._rpc["env.render_camera"]() == "frame"
    planner.resolve_grasp(gid)
    assert f._rpc["env.step"]([0] * 7) == "stepped"
    with pytest.raises(G.GraspError, match="stale"):
        planner.resolve_grasp(gid)
    assert set(f._rpc) >= {
        "env.plan_grasp",
        "env.plan_place",
        "env.next_grasp",
        "env.resolve_grasp",
        "env.attachment_frames",
    }


def test_greedy_policy_advances_only_through_rejection():
    planner, _ = _planner(sam3=FakeSam3(_block_mask()))
    out = planner.plan_grasp(object="block")
    first, second = [c["id"] for c in out["candidates"]]
    nxt = planner.next_grasp(first, "unreachable")
    assert (
        nxt["active"] == second
        and nxt["candidate"]["id"] == second
        and nxt["remaining"] == 1
    )
    assert planner.next_grasp(second, "collision")["active"] is None
    # A rejected candidate still resolves (the caller may inspect it) but is marked.
    assert planner.resolve_grasp(first)["id"] == first


def test_external_mask_ids_are_accepted_from_the_segment_book():
    book = DetectionBook()
    book.bind(1)
    mid = book.add({"mask": _block_mask(), "camera": "agentview"})
    planner, _ = _planner(masks=book)
    out = planner.plan_grasp(mask_id=mid)
    assert out["mask_id"] == mid and out["candidate_count"] == 2
    with pytest.raises(G.GraspError, match="camera"):
        planner.plan_grasp(mask_id=mid, camera="wrist")
    book.bind(2)  # the segment book moved on: its id is stale there too
    with pytest.raises(G.GraspError, match="stale"):
        planner.plan_grasp(mask_id=mid)


def test_plan_place_composes_and_refuses_mixed_snapshots():
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.05, 0.0]  # AnyPlace: move the object 5 cm along camera +y
    planner, _ = _planner(sam3=FakeSam3(_block_mask()), anyplace=FakeAnyPlace([T]))
    obj = planner.segment_mask("block")["id"]
    region = planner.segment_mask("plate")["id"]
    gid = planner.plan_grasp(mask_id=obj)["active"]
    place = planner.plan_place(region, gid)
    explicit = planner.plan_place(region, gid, object_mask_id=obj)
    assert (
        explicit["candidates"][0]["eef_position"]
        == place["candidates"][0]["eef_position"]
    )
    assert place["candidate_count"] == 1 and place["active"].startswith("p")
    p = place["candidates"][0]
    # Camera +y is world -x: the place pose is the grasp moved 5 cm along world -x.
    assert np.allclose(p["eef_position"], [-0.05, 0, 0.2], atol=1e-6)
    assert planner.resolve_grasp(place["active"])["kind"] == "placement"
    # A grasp planned on another mask is refused.
    with pytest.raises(G.GraspError, match="planned on mask"):
        planner.plan_place(obj, gid, object_mask_id=region)
    # Ids from an earlier observation are refused (the robot moved between them).
    planner.invalidate()
    obj2 = planner.segment_mask("block")["id"]
    region2 = planner.segment_mask("plate")["id"]
    with pytest.raises(G.GraspError, match="stale"):
        planner.plan_place(region2, gid)
    gid2 = planner.plan_grasp(mask_id=obj2)["active"]
    planner.invalidate()
    obj3 = planner.segment_mask("block")["id"]
    with pytest.raises(G.GraspError, match="stale"):
        planner.plan_place(region2, gid2)
    assert obj3 != obj2


def test_the_planner_shares_ids_and_the_observation_clock_with_the_segment_book():
    """One id names one mask: the segment book and the planner draw from one counter, and a
    motion (or a new observation) expires both books at once."""
    book = DetectionBook()
    book.bind(book.epoch.observation)
    d1 = book.add({"mask": _block_mask(), "camera": "agentview"})
    planner, _ = _planner(masks=book, sam3=FakeSam3(_block_mask()))
    d2 = planner.segment_mask("block")["id"]
    assert d1 == "d1" and d2 == "d2", "no collision between the two books"
    gid = planner.plan_grasp(mask_id=d1)["active"]
    assert gid == "g3"
    # The robot moved (a wrapped motion method ran): the segment book's id is stale too.
    planner.invalidate()
    with pytest.raises(G.GraspError, match="stale"):
        planner.plan_grasp(mask_id=d1)
    with pytest.raises(G.GraspError, match="stale"):
        planner.plan_grasp(mask_id=d2)
    assert book.observation == planner.observation
    # A new observation in the segment book expires the planner's ids as well.
    book.epoch.tick()
    with pytest.raises(G.GraspError, match="stale"):
        planner.resolve_grasp(gid)
    assert book.observation == planner.observation


def test_motion_methods_tick_once_even_when_the_segment_book_installed_them():
    from pi_embodied_services.utils.detections import Epoch

    class Facade:
        def __init__(self):
            self._rpc = {"env.step": lambda: "stepped", "env.plan_grasp": None}
            self._readonly_methods = set()

    facade = Facade()
    epoch = Epoch()
    epoch.install(facade)
    epoch.install(facade)  # a second install (the planner's) wraps nothing twice
    epoch.install(facade, ("env.move_to",))  # absent methods are skipped
    assert facade._rpc["env.step"]() == "stepped" and epoch.observation == 1


def test_a_digest_keeps_ids_across_calls_that_did_not_move_the_robot():
    """Finding: every observation and every refused motion expired the ids. With a state
    digest the clock ticks only when the robot state changed."""
    from pi_embodied_services.utils.detections import Epoch, state_digest

    robot = {"raw_base_state": {"tcp_pose": [0.5, 0.0, 0.3, 1, 0, 0, 0]}}

    def move(dz):
        if abs(dz) > 0.1:
            raise ValueError("refused: more than 0.10 m per call")  # before moving
        robot["raw_base_state"]["tcp_pose"][2] += dz
        return "moved"

    class Facade:
        def __init__(self):
            self._rpc = {"env.move_delta": move}
            self._readonly_methods = set()

    planner, _ = _planner(
        sam3=FakeSam3(_block_mask()), state_digest=lambda: state_digest(robot)
    )
    f = Facade()
    planner.install(f)
    gid = planner.plan_grasp(object="block")["active"]
    with pytest.raises(ValueError, match="refused"):
        f._rpc["env.move_delta"](0.5)
    assert f._rpc["env.move_delta"](0.0) == "moved"
    # Sub-millimetre noise on an unmoved arm is not motion either.
    robot["raw_base_state"]["tcp_pose"][0] += 0.0001
    f._rpc["env.move_delta"](0.0)
    assert planner.resolve_grasp(gid)["id"] == gid, "unmoved: the id is current"
    f._rpc["env.move_delta"](0.05)
    with pytest.raises(G.GraspError, match="stale"):
        planner.resolve_grasp(gid)
    # Without a digest every wrapped call expires them, as before.
    epoch = Epoch()
    assert epoch.refresh() and epoch.refresh() and epoch.observation == 2
    # An unreadable state counts as moved.
    broken = Epoch(digest=lambda: 1 / 0)
    broken.next_id()
    assert broken.refresh()


def test_state_digest_reads_the_pose_keys_at_any_depth_rounded():
    from pi_embodied_services.utils.detections import state_digest

    a = {
        "left_arm": {"tcp_pose": [0.1, 0.2, 0.3], "force": [9.0]},
        "right_arm": {"tcp_pose": [0.4, 0.5, 0.6], "gripper_open": True},
        "stamp": 12.5,
    }
    b = {
        **a,
        "left_arm": {"tcp_pose": [0.1002, 0.2, 0.3], "force": [3.0]},
        "stamp": 99.0,
    }
    assert state_digest(a) == state_digest(b), "noise, forces and stamps are ignored"
    c = {**a, "right_arm": {"tcp_pose": [0.4, 0.5, 0.6], "gripper_open": False}}
    assert state_digest(a) != state_digest(c), "a gripper change is a change"
    assert state_digest({"nothing": 1}) == ()


def test_claim_resolves_the_whole_grasp_path_once_and_remembers_the_held_grasp():
    planner, _ = _planner(sam3=FakeSam3(_block_mask()))
    out = planner.plan_grasp(object="block")
    gid, other = out["active"], out["candidates"][1]["id"]
    claim = planner.claim_waypoints(gid, standoff=0.1, lift=0.05)
    assert claim["kind"] == "grasp" and claim["id"] == gid
    # The block's grasp is 0.2 m up with a downward approach: the pre-grasp sits 0.1 m above.
    assert np.allclose(claim["waypoints"]["pre_grasp"], [0, 0, 0.3], atol=1e-6)
    assert np.allclose(claim["waypoints"]["grasp"], [0, 0, 0.2], atol=1e-6)
    assert np.allclose(claim["waypoints"]["lift"], [0, 0, 0.25], atol=1e-6)
    assert claim["steps"] == [
        {"to": "pre_grasp", "gripper": -1},
        {"to": "grasp", "gripper": -1},
        {"gripper": 1},
        {"to": "lift", "gripper": 1},
    ]
    held = planner.held()
    assert held["grasp_id"] == gid and held["prompt"] == "block"
    # A rejected candidate is not executed; offsets are bounded; a stale id is refused.
    planner.next_grasp(other, "unreachable")
    with pytest.raises(G.GraspError, match="rejected"):
        planner.claim_waypoints(other)
    with pytest.raises(G.GraspError, match="standoff"):
        planner.claim_waypoints(gid, standoff=1.0)
    planner.invalidate()
    with pytest.raises(G.GraspError, match="stale"):
        planner.claim_waypoints(gid)
    assert planner.held()["grasp_id"] == gid, "the held grasp outlives its id"


def test_plan_place_after_the_grasp_uses_the_held_pose_and_refuses_an_empty_gripper():
    """plan_grasp -> execute (claim, motions) -> plan_place(executed id) -> claim the place."""
    T = np.eye(4)
    T[:3, 3] = [
        0.0,
        0.05,
        0.0,
    ]  # AnyPlace: move the object 5 cm along camera +y (world -x)
    eef = {"xyz": np.array([0.0, 0.0, 0.2]), "quat": np.array([1.0, 0, 0, 0])}
    holding = {"now": True}
    planner, _ = _planner(
        sam3=FakeSam3(_block_mask()),
        anyplace=FakeAnyPlace([T]),
        eef_pose=lambda arm: (eef["xyz"], eef["quat"]),
        holding=lambda arm: holding["now"],
    )
    gid = planner.plan_grasp(object="block")["active"]
    planner.claim_waypoints(gid, lift=0.1)
    planner.invalidate()  # the grasp's motions
    eef["xyz"] = np.array([0.0, 0.0, 0.3])  # lifted 10 cm, holding the block
    region = planner.segment_mask("plate")["id"]
    place = planner.plan_place(region, gid)
    assert place["held"] is True and place["grasp_id"] == gid
    assert place["object_mask_id"] != region, "the held block was segmented again"
    p = place["candidates"][0]
    # The composition starts from the gripper's actual pose (0.3 up), not the planned 0.2.
    assert np.allclose(p["eef_position"], [-0.05, 0, 0.3], atol=1e-6)
    assert np.allclose(p["eef_quat_xyzw"], [1, 0, 0, 0], atol=1e-6)
    claim = planner.claim_waypoints(place["active"])
    assert claim["kind"] == "placement" and [s.get("to") for s in claim["steps"]] == [
        "pre_place",
        "place",
        None,
        "retreat",
    ]
    assert planner.held() is None, "claiming the place releases the held grasp"
    with pytest.raises(G.GraspError, match="stale"):
        planner.plan_place(region, gid)
    # An empty gripper is refused (and forgotten).
    planner.invalidate()
    gid = planner.plan_grasp(object="block")["active"]
    planner.claim_waypoints(gid)
    planner.invalidate()
    holding["now"] = False
    region = planner.segment_mask("plate")["id"]
    with pytest.raises(G.GraspError, match="holds nothing"):
        planner.plan_place(region, gid)
    assert planner.held() is None


def test_a_reset_forgets_the_held_grasp():
    planner, _ = _planner(sam3=FakeSam3(_block_mask()))

    class Facade:
        _rpc = {"env.reset": lambda: "reset"}
        _readonly_methods = set()

    f = Facade()
    planner.install(f)
    assert "env.claim_waypoints" in f._rpc
    planner.claim_waypoints(planner.plan_grasp(object="block")["active"])
    assert planner.held() is not None
    assert f._rpc["env.reset"]() == "reset" and planner.held() is None


def test_planner_refuses_unknown_backends_and_frames():
    with pytest.raises(ValueError, match="unknown grasp backends"):
        G.GraspPlanner(_view, cameras=["agentview"], backends={"nope": FakeServer([])})
    bad = G.GraspPlanner(
        _view,
        cameras=["agentview"],
        backends={"graspgenx": FakeServer([], grasp_frame="graspgenx")},
        sam3=FakeSam3(_block_mask()),
    )
    with pytest.raises(G.GraspError, match="frame"):
        bad.plan_grasp(object="block")
    assert G.GraspPlanner.from_args(_view, cameras=["agentview"]) is None
    none = G.GraspPlanner(_view, cameras=["agentview"])
    with pytest.raises(G.GraspError, match="no grasp backend"):
        none.plan_grasp(object="block")
    with pytest.raises(G.GraspError, match="exactly one"):
        _planner()[0].plan_grasp(object="a", mask_id="d1")


def test_attachment_frames_crop_the_front_view_around_the_eef():
    planner, _ = _planner(
        eef_pose=lambda arm: (np.array([0.0, 0.0, 0.2]), np.array([0, 0, 0, 1.0]))
    )
    out = planner.attachment_frames()
    cams = {f["camera"]: f for f in out["frames"]}
    assert cams["wrist"]["crop_rc"] is None and cams["agentview"]["crop_rc"] is not None
    r0, c0, r1, c1 = cams["agentview"]["crop_rc"]
    assert 0 <= r0 < r1 <= H and 0 <= c0 < c1 <= W and cams["agentview"]["png_base64"]


# --- servers --------------------------------------------------------------------------------


class Echo(GraspServer):
    SERVICE_NAME = "echo"

    def predict(self, *, depth, K, mask, rgb, up_direction_camera, max_candidates):
        self.seen = dict(depth=depth, mask=mask, rgb=rgb, up=up_direction_camera)
        return [
            _camera_candidate(0.1, 0.0),
            _camera_candidate(0.5, 0.01),
            _camera_candidate(0.3, 0.02),
        ], {"n": 3}


def test_grasp_server_validates_and_ranks():
    s = Echo()
    assert "echo.plan" in s._rpc and "echo.info" in s._rpc
    depth = np.ones((H, W), np.float32)
    out = s.plan(
        depth,
        K,
        _block_mask().astype(np.uint8),
        max_candidates=2,
        up_direction_camera=[0, 0, -2],
    )
    assert [c["score"] for c in out["candidates"]] == [0.5, 0.3] and out[
        "model_candidate_count"
    ] == 3
    assert out["grasp_frame"] == "graspnet" and np.allclose(s.seen["up"], [0, 0, -1])
    with pytest.raises(ValueError, match="mask"):
        s.plan(depth, K, np.zeros((H, W), np.uint8))
    with pytest.raises(ValueError, match="match"):
        s.plan(depth, K, np.ones((H + 1, W), np.uint8))
    with pytest.raises(ValueError, match="intrinsic_K"):
        s.plan(depth, np.zeros((3, 3)), np.ones((H, W), np.uint8))


class _FrankaBackend:
    """Any env.* worker method answers {}; the robot state carries the TCP pose."""

    def get_robot_state(self):
        return {
            "raw_base_state": {"tcp_pose": [0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]},
            "left_arm": {"tcp_pose": [0.4, 0.3, 0.3, 1.0, 0.0, 0.0, 0.0]},
            "right_arm": {"tcp_pose": [0.4, -0.3, 0.3, 1.0, 0.0, 0.0, 0.0]},
        }

    def get_camera_meta(self):
        return {}

    def __getattr__(self, name):
        return lambda *a, **k: {}


def test_the_franka_planners_get_the_perception_sam3_and_its_camera_names(monkeypatch):
    """Finding: the Franka and dual-Franka planners had no SAM3, so plan_grasp(object=<text>)
    and plan_place(region=<text>) always failed; the dual arm's env.segment cameras were the
    single arm's names, not its projection views."""
    from pi_embodied_services.robots.dual_franka import env_server as dual_server
    from pi_embodied_services.robots.dual_franka import perception as dual_perception
    from pi_embodied_services.robots.franka.env_server import FrankaEnvFacade
    from pi_embodied_services.utils.perception import Perception

    sam3 = FakeSam3(_block_mask())
    single_cams, _ = FrankaEnvFacade.perception_layout(_FrankaBackend())
    single = FrankaEnvFacade(
        _FrankaBackend(),
        Perception(sam3=sam3, cameras=single_cams),
        {"graspnet": FakeServer([])},
    )
    planner = single._rpc["env.plan_grasp"].__self__
    assert planner._sam3 is sam3 and planner.capabilities()["segment"] is True
    assert planner.capabilities()["cameras"] == list(single_cams)

    views = {
        "d455": {"raw_key": "d455_rgb", "calibration_key": "d455_camera"},
        "base": {"raw_key": "base_0_rgb", "calibration_key": "base_camera"},
    }
    monkeypatch.setattr(dual_perception, "_projection_cameras", lambda: views)
    monkeypatch.setattr(dual_perception, "load_calibration_bundle", lambda: {})
    backend = _FrankaBackend()
    dual_cams, intrinsics = dual_server.DualFrankaEnvFacade.perception_layout(backend)
    dual = dual_server.DualFrankaEnvFacade(
        backend,
        Perception(sam3=sam3, cameras=dual_cams, intrinsics=intrinsics),
        {"graspnet": FakeServer([])},
    )
    planner = dual._rpc["env.plan_grasp"].__self__
    assert planner._sam3 is sam3
    assert (
        sorted(planner.capabilities()["cameras"])
        == sorted(dual_cams)
        == ["base", "d455"]
    )
    # The digest reads both arms' poses.
    assert len(dual._state_digest()) == 3
