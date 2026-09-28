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
    # The composition starts from the gripper's actual pose (x, y and orientation), and the
    # block is set down on the region: the grasp was at the block's top (0.2, its support
    # read from the mask's lowest points is 0.2 too), so the EEF goes to the region's top
    # (0.2) plus the 1 cm clearance, not the 0.3 it is held at.
    assert np.allclose(p["eef_position"], [-0.05, 0, 0.21], atol=1e-6)
    assert p["settled_m"] == pytest.approx(-0.09)
    assert np.allclose(p["eef_quat_xyzw"], [1, 0, 0, 0], atol=1e-6)
    claim = planner.claim_waypoints(place["active"])
    assert claim["kind"] == "placement" and [s.get("to") for s in claim["steps"]] == [
        "pre_place",
        "place",
        None,
        "retreat",
    ]
    assert planner.held()["grasp_id"] == gid, (
        "a claimed place still holds until it opens"
    )
    holding["now"] = True
    assert planner.release_held() == {"released": False, "held": gid}
    assert planner.release_held(opened=True)["released"] is True
    assert planner.held() is None
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
        {"contact_graspnet": FakeServer([])},
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
        {"contact_graspnet": FakeServer([])},
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


class _Scene:
    """A view whose table the test can change under an unmoved robot."""

    def __init__(self):
        self.depth = np.full((H, W), 0.9, dtype=np.float32)
        self.depth[4:12, 4:12] = 0.8
        self.rgb = np.full((H, W, 3), 120, np.uint8)
        self.captures = 0

    def __call__(self, camera):
        self.captures += 1
        return {
            "rgb": self.rgb.copy(),
            "depth": self.depth.copy(),
            "intrinsic_K": K,
            "extrinsic_cam2world": CAM2WORLD,
        }


def _scene_planner(**kw):
    scene = _Scene()
    server = FakeServer([_camera_candidate(0.9, 0.0)])
    planner = G.GraspPlanner(
        scene,
        cameras=["agentview"],
        backends={"contact_graspnet": server},
        sam3=FakeSam3(_block_mask()),
        **kw,
    )
    return planner, scene


def test_an_object_moved_under_a_still_arm_expires_the_plan():
    """Audit: with ids expiring only on robot motion, the planner kept the first frame while
    the arm stood still, so a plan survived a person moving the object. Every call captures
    anew; a changed scene expires the ids, sensor noise does not."""
    robot = {"tcp_pose": [0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]}
    from pi_embodied_services.utils.detections import state_digest

    planner, scene = _scene_planner(state_digest=lambda: state_digest(robot))
    gid = planner.plan_grasp(object="block")["active"]
    # Noise (a few colour levels, a millimetre of depth) is the same scene: the id holds.
    rng = np.random.default_rng(0)
    scene.rgb = (scene.rgb + rng.integers(-4, 5, scene.rgb.shape)).astype(np.uint8)
    scene.depth = scene.depth + rng.normal(0, 0.001, scene.depth.shape).astype(
        np.float32
    )
    captures = scene.captures
    claim = planner.claim_waypoints(gid)
    assert claim["id"] == gid and scene.captures == captures + 1, "a fresh capture"
    # Someone slides the block 4 px along the table; the arm never moved.
    gid = planner.plan_grasp(object="block")["active"]
    scene.depth[:] = 0.9
    scene.depth[4:12, 8:16] = 0.8
    with pytest.raises(G.GraspError, match="stale"):
        planner.claim_waypoints(gid)
    again = planner.plan_grasp(object="block")
    assert gid in again["expired_ids"] and again["active"] != gid
    assert planner.claim_waypoints(again["active"])["id"] == again["active"]


def test_a_reset_expires_every_id_even_when_the_arm_returns_to_the_same_pose():
    from pi_embodied_services.utils.detections import state_digest

    robot = {"tcp_pose": [0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]}

    class Facade:
        _rpc = {"env.reset": lambda: "reset"}
        _readonly_methods = set()

    planner, scene = _scene_planner(state_digest=lambda: state_digest(robot))
    f = Facade()
    planner.install(f)
    gid = planner.plan_grasp(object="block")["active"]
    assert f._rpc["env.reset"]() == "reset"
    with pytest.raises(G.GraspError, match="stale"):
        planner.resolve_grasp(gid)


def test_scene_signatures_ignore_noise_and_catch_a_moved_object():
    from pi_embodied_services.utils.detections import frame_signature, scene_changed

    rgb = np.full((96, 128, 3), 100, np.uint8)
    depth = np.full((96, 128), 0.9, np.float32)
    depth[40:56, 40:56] = 0.8
    a = frame_signature(rgb, depth)
    rng = np.random.default_rng(1)
    noisy = frame_signature(
        (rgb + rng.integers(-6, 7, rgb.shape)).astype(np.uint8),
        depth + rng.normal(0, 0.002, depth.shape).astype(np.float32),
    )
    assert not scene_changed(a, noisy)
    moved = depth.copy()
    moved[40:56, 40:56] = 0.9
    moved[40:56, 70:86] = 0.8
    assert scene_changed(a, frame_signature(rgb, moved))
    recoloured = rgb.copy()
    recoloured[40:56, 40:56] = 200
    assert scene_changed(a, frame_signature(recoloured, depth))
    holes = depth.copy()
    holes[:, :64] = 0.0
    assert scene_changed(a, frame_signature(rgb, holes))
    assert scene_changed(None, a) and scene_changed(
        a, frame_signature(rgb[:64], depth[:64])
    )


def test_orientation_error_is_the_short_way_round_the_symmetric_fingers():
    down = np.diag([1.0, -1.0, -1.0])
    assert np.allclose(G.orientation_error(down, down), 0)
    # A half turn about the approach is the same grasp: no error.
    assert np.allclose(G.orientation_error(down, down @ G.FLIP_ABOUT_APPROACH), 0)
    tilt = np.deg2rad(20)
    ry = np.array(
        [[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(tilt)]]
    )
    assert np.allclose(G.orientation_error(down, ry @ down), [0, tilt, 0], atol=1e-9)
    half = np.diag([-1.0, 1.0, -1.0])  # a half turn about world y
    assert np.linalg.norm(G.rotvec_of(half)) == pytest.approx(np.pi)


class FakeSam3All:
    """SAM3 answering all=True with every mask (two same-named objects)."""

    def __init__(self, masks):
        self.masks = masks
        self.calls = []

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        import base64
        import io

        from PIL import Image

        self.calls.append(kwargs)

        def png(m):
            buf = io.BytesIO()
            Image.fromarray((m * 255).astype(np.uint8), mode="L").save(
                buf, format="PNG"
            )
            return base64.b64encode(buf.getvalue()).decode()

        dets = [
            {"score": 0.9 - 0.1 * i, "mask_png_base64": png(m)}
            for i, m in enumerate(self.masks)
        ]
        if kwargs.get("all"):
            return {"found": True, "detections": dets}
        return {"found": True, **dets[0]}


def test_plan_place_after_the_grasp_segments_the_held_object_not_its_twin():
    """Audit: the held object was re-segmented by text alone, and LIBERO scenes often have two
    same-named objects: the mask nearest the gripper is the held one."""
    far = np.zeros((H, W), bool)
    far[0:3, 0:3] = True  # the twin, in a corner (SAM3's best score)
    near = _block_mask()
    T = np.eye(4)
    eef = {"xyz": np.array([0.0, 0.0, 0.2]), "quat": np.array([1.0, 0, 0, 0])}
    planner, _ = _planner(
        sam3=FakeSam3All([near]),
        anyplace=FakeAnyPlace([T]),
        eef_pose=lambda arm: (eef["xyz"], eef["quat"]),
    )
    gid = planner.plan_grasp(object="bowl")["active"]
    planner.claim_waypoints(gid)
    planner.invalidate()  # the grasp's motions
    planner._sam3 = FakeSam3(np.ones((H, W), bool))  # the plate: the whole table
    region = planner.segment_mask("plate")["id"]
    planner._sam3 = FakeSam3All([far, near])
    place = planner.plan_place(region, gid)
    held = planner._book.get(place["object_mask_id"])
    assert held["mask"][6, 6] and not held["mask"][1, 1], "the mask at the gripper"
    assert planner._sam3.calls[-1]["all"] is True


def test_anyplace_sees_a_gravity_aligned_cloud_and_answers_in_the_camera_frame():
    """Box run (LIBERO spatial t0): fed OpenCV camera-frame clouds, AnyPlace 'placed' the held
    bowl along the camera's axes, so the composed place approached almost horizontally,
    about (0.95, -0.1, -0.3), and the arm drove up to z 1.5 m. The server now hands the model
    world-frame clouds and converts its transforms back to the camera frame."""
    import threading

    from pi_embodied_services.components import anyplace_server as A

    seen = {}

    class Stub(A.AnyPlaceFacade):
        def __init__(self):  # no model: record what it would see
            self._depth_max = 2.0
            self._lock = threading.Lock()

        def _predict(self, obj, region):
            seen["obj"], seen["region"] = obj, region
            lift = np.eye(4)
            lift[:3, 3] = [0.1, 0.0, 0.05]  # world: 10 cm along +x, 5 cm up
            return np.stack([lift])

        def info(self):
            return {}

    depth = np.full((H, W), 0.9, dtype=np.float32)
    depth[4:12, 4:12] = 0.8
    obj = _block_mask()
    region = np.zeros((H, W), bool)
    region[12:16, :] = True
    A.MIN_POINTS = 1  # a 16x16 test frame
    try:
        out = Stub().plan(
            np.zeros((H, W, 3), np.uint8), depth, K, obj, region, 1, CAM2WORLD
        )
    finally:
        A.MIN_POINTS = 1024
    assert out["model_frame"] == "world"
    # The camera is 1 m up looking down: the model sees the block top at world z 0.2, the
    # table at 0.1, not at camera depths 0.8 / 0.9.
    assert np.allclose(seen["obj"][:, 2], 0.2, atol=1e-5)
    assert np.allclose(seen["region"][:, 2], 0.1, atol=1e-5)
    T_cam = np.asarray(out["placements"][0]["transform_matrix"])
    p_cam = np.array([0.0, 0.0, 0.8, 1.0])
    moved_world = CAM2WORLD @ (T_cam @ p_cam)
    assert np.allclose(moved_world[:3], (CAM2WORLD @ p_cam)[:3] + [0.1, 0.0, 0.05])


class _RecordingAnyPlace(FakeAnyPlace):
    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.kwargs = kwargs
        return super().call(method, args, kwargs, timeout_s=timeout_s)


def test_a_place_is_kept_upright_and_refused_off_the_region():
    """The box run's AnyPlace transforms tipped the held bowl 50-60 deg (the composed place
    then approached from the side). A place keeps only the model's turn about the vertical
    and its landing point, and is refused when the object would land off or high above the
    region; every candidate refused is an error that says why."""
    region = np.ones((H, W), bool)  # the whole table as the region
    a = np.deg2rad(70)
    tip = np.eye(4)
    tip[:3, :3] = [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    toward_camera = tip.copy()
    toward_camera[:3, 3] = [0.0, 0.3, -0.5]  # tipped and lifted half a metre
    beside = np.eye(4)
    beside[:3, 3] = [0.6, 0.0, 0.0]  # 60 cm off the table's edge
    # Tipped 70 deg about the block's own centre, landing 5 cm over: kept, upright.
    centre = np.array([-0.02, -0.02, 0.8])  # the block mask's centroid (pixels 4-11)
    in_place = tip.copy()
    in_place[:3, 3] = centre - tip[:3, :3] @ centre + [0.0, 0.05, 0.0]
    anyplace = _RecordingAnyPlace([toward_camera, beside, in_place])
    planner, _ = _planner(sam3=FakeSam3(_block_mask()), anyplace=anyplace)
    obj = planner.segment_mask("block")["id"]
    planner._sam3 = FakeSam3(region)
    reg = planner.segment_mask("table")["id"]
    gid = planner.plan_grasp(mask_id=obj)["active"]
    place = planner.plan_place(reg, gid)
    assert np.allclose(anyplace.kwargs["extrinsic_cam2world"], CAM2WORLD)
    assert [r["rank"] for r in place["refused"]] == [0, 1]
    assert all(
        "off the region" in r["reason"] or "not on the region" in r["reason"]
        for r in place["refused"]
    )
    kept = place["candidates"][0]
    assert kept["rank"] == 2 and kept["model_tilt_deg"] == pytest.approx(70, abs=0.5)
    assert kept["approach"] == pytest.approx([0, 0, -1], abs=1e-6), (
        "upright: from above"
    )
    # Camera +y is world -x: the block's grasp lands 5 cm along -x, at its own height.
    assert kept["eef_position"] == pytest.approx([-0.05, 0.0, 0.2], abs=1e-3)
    anyplace.transforms = [toward_camera]
    with pytest.raises(G.GraspError, match="can be executed from above"):
        planner.plan_place(reg, gid)


def test_a_place_from_a_tilted_grasp_is_refused():
    """The approach check stays for the grasp itself: held 60 deg off vertical, a place would
    come in from the side."""
    b = np.deg2rad(60)
    tilt = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    side = G.make_candidate(
        score=0.9,
        rotation=tilt @ G.ZX_NATIVE_TO_GRASPNET,
        center=[0.0, 0.0, 0.8],
        width=0.04,
        depth=0.0,
        source_model="fake",
    )
    planner = G.GraspPlanner(
        _view,
        cameras=["agentview"],
        backends={"contact_graspnet": FakeServer([side])},
        sam3=FakeSam3(np.ones((H, W), bool)),
        anyplace=FakeAnyPlace([np.eye(4)]),
    )
    gid = planner.plan_grasp(object="block")["active"]
    reg = planner.segment_mask("table")["id"]
    with pytest.raises(G.GraspError, match="from straight down"):
        planner.plan_place(reg, gid)


def test_a_perception_installed_after_the_planner_shares_its_ids():
    """Audit a2c880c #1: Robosuite and LIBERO built the planner first and the perception
    later on a second Epoch, so both minted d1 and plan_grasp(mask_id="d1") could grasp the
    other object. The perception now runs on the planner's epoch and its book is accepted."""
    from pi_embodied_services.utils.perception import install_perception

    planner, _ = _planner(sam3=FakeSam3(_block_mask()))
    mine = planner.segment_mask("block")["id"]

    class Facade:
        def __init__(self):
            self._rpc = {
                "env.get_observation": lambda: {},
                "env.get_env_meta": lambda: {},
            }
            self._readonly_methods = set()

    class Args:
        sam3 = "http://127.0.0.1:1"
        unidepth = None

    perception = install_perception(
        Facade(), Args(), cameras=["agentview"], view=_view, grasp=planner
    )
    assert perception.epoch is planner.epoch
    theirs = perception.book.add({"mask": _block_mask(), "camera": "agentview"})
    assert theirs != mine, "one id counter"
    assert planner.plan_grasp(mask_id=theirs)["mask_id"] == theirs
    planner.invalidate()
    with pytest.raises(G.GraspError, match="stale"):
        planner.plan_grasp(mask_id=theirs)
