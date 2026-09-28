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

"""Detection ids bound to the observation (utils/detections.py) and the env-server
perception primitives over fake SAM3 / UniDepth clients (utils/perception.py)."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

from pi_embodied_services.utils.detections import (
    DetectionBook,
    DetectionStale,
    decode_mask_png,
    describe_mask,
    overlay_masks,
)
from pi_embodied_services.utils.perception import (
    FRANKA_CAMERAS,
    SIM_NAMES,
    Perception,
    add_perception_arguments,
    franka_intrinsics,
    install_perception,
    render_view,
)
from pi_embodied_services.utils.rpc import RpcFacade

Image = pytest.importorskip("PIL.Image")

H, W = 24, 32
K = np.array([[40.0, 0.0, 16.0], [0.0, 40.0, 12.0], [0.0, 0.0, 1.0]])


def _mask_png(mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _box(r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


# ---- DetectionBook -----------------------------------------------------------


def test_ids_are_unique_and_die_with_the_observation() -> None:
    book = DetectionBook()
    with pytest.raises(RuntimeError):
        book.add({"mask": None})
    assert book.bind(1) == []
    a = book.add({"mask": "m1"})
    b = book.add({"mask": "m2"})
    assert (a, b) == ("d1", "d2")
    assert book.get(a)["observation"] == 1
    assert book.bind(1) == [], "re-binding the same observation keeps the ids"
    assert book.bind(2) == [a, b]
    assert book.ids == []
    with pytest.raises(DetectionStale, match="belongs to observation 1"):
        book.get(a)
    with pytest.raises(DetectionStale, match="unknown detection id"):
        book.get("d9")
    c = book.add({"mask": "m3"})
    assert c == "d3", "ids are never reused"
    assert book.drain_invalidated() == [a, b]
    assert book.drain_invalidated() == []


def test_select_and_reject_bookkeeping() -> None:
    book = DetectionBook()
    book.bind(1)
    a, b = book.add({"mask": 1}), book.add({"mask": 2})
    book.select(a)
    book.reject(b)
    assert book.summary() == {
        "observation": 1,
        "ids": [a, b],
        "selected": a,
        "rejected": [b],
    }
    book.reject(a)  # rejecting the selected one clears the selection
    assert book.selected is None and book.rejected == [b, a]
    book.select(a)  # selecting a rejected one un-rejects it
    assert book.selected == a and book.rejected == [b]
    with pytest.raises(DetectionStale):
        book.select("d7")
    book.bind(2)
    assert book.summary() == {
        "observation": 2,
        "ids": [],
        "selected": None,
        "rejected": [],
    }


# ---- mask geometry -----------------------------------------------------------


def test_describe_mask_projects_the_centroid_through_depth_and_K() -> None:
    mask = _box(4, 12, 10, 20)
    depth = np.full((H, W), 0.5, dtype=np.float32)
    depth[5, 11] = 0.0  # a hole inside the mask is ignored by the median
    d = describe_mask(mask, depth, K)
    assert d["area_px"] == 80
    assert d["centroid_rc"] == [7, 14]  # medians of rows 4..11, cols 10..19 (int)
    assert d["depth_m"] == pytest.approx(0.5)
    assert d["depth_valid_px"] == 79
    # (u - cx) / fx * z, (v - cy) / fy * z, z
    assert d["point_camera"] == pytest.approx(
        [(14 - 16) / 40 * 0.5, (7 - 12) / 40 * 0.5, 0.5]
    )
    assert describe_mask(mask, None, K)["point_camera"] is None
    assert describe_mask(mask, depth, None)["depth_m"] == pytest.approx(0.5)
    empty = describe_mask(np.zeros((H, W), bool), depth, K)
    assert empty == {
        "area_px": 0,
        "centroid_rc": None,
        "depth_m": None,
        "point_camera": None,
    }


def test_mask_png_roundtrip_and_overlay() -> None:
    mask = _box(0, 2, 0, 3)
    assert np.array_equal(decode_mask_png(_mask_png(mask)), mask)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    out = overlay_masks(rgb, [mask, _box(10, 12, 0, 3)])
    assert out.dtype == np.uint8 and out.shape == rgb.shape
    assert tuple(out[0, 0]) == (int(0.45 * 255), int(0.45 * 64), int(0.45 * 64)), (
        "rank 0 is red"
    )
    assert out[10, 0, 2] > out[10, 0, 0], "rank 1 is blue"
    assert not out[20, 20].any(), "pixels outside every mask are untouched"


def test_franka_intrinsics_lookup() -> None:
    meta = {
        "observation_camera_map": {"main": "wrist_cam", "extra_0": "ext"},
        "cameras": {
            "wrist_cam": {"intrinsic_K": K.tolist()},
            "ext": {"intrinsic_K": None},
        },
    }
    assert np.array_equal(franka_intrinsics(meta, "main"), K)
    assert franka_intrinsics(meta, "extra_0") is None
    assert franka_intrinsics({"error": "no cameras"}, "main") is None
    assert franka_intrinsics(None, "main") is None


# ---- Perception over fake services -----------------------------------------------


class FakeSam3:
    """Answers sam3.segment: two masks for all=True, the best for all=False."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.masks = [_box(4, 12, 10, 20), _box(14, 20, 2, 8)]
        self.scores = [0.9, 0.4]

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        assert method == "sam3.segment"
        kwargs = dict(kwargs or {})
        self.calls.append(kwargs)
        assert isinstance(kwargs["image_base64"], str)
        keep = [i for i, s in enumerate(self.scores) if s >= kwargs["min_score"]]
        if kwargs.get("all"):
            return {
                "found": bool(keep),
                "count": len(keep),
                "detections": [
                    {
                        "index": n,
                        "score": self.scores[i],
                        "box": [0, 0, 1, 1],
                        "area_px": int(self.masks[i].sum()),
                        "mask_png_base64": _mask_png(self.masks[i]),
                        "mask_shape": [H, W],
                    }
                    for n, i in enumerate(keep)
                ],
            }
        if not keep:
            return {"found": False, "reason": "below min_score"}
        return {
            "found": True,
            "score": self.scores[keep[0]],
            "box": [0, 0, 1, 1],
            "mask_png_base64": _mask_png(self.masks[keep[0]]),
            "mask_shape": [H, W],
        }


class FakeUniDepth:
    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        assert method == "depth.estimate"
        rgb = kwargs["rgb"]
        # Half the metric scale: the fusion must recover x2 from the overlap.
        return {"depth": np.full(rgb.shape[:2], 0.25, np.float32), "model": "fake"}


class Server(RpcFacade):
    """A franka-shaped env server: get_observation / get_env_meta / get_camera_meta."""

    SERVICE_NAME = "fake-franka-env"

    def __init__(self) -> None:
        super().__init__()
        self.observations = 0
        self.depth = np.full((H, W), 0.5, dtype=np.float32)
        self.depth[14:20, 2:8] = 0.0  # the second object has no sensor depth
        self._rpc["env.get_observation"] = self.get_observation
        self._rpc["env.get_env_meta"] = lambda: {
            "ok": True,
            "capabilities": {"has_vla": True},
        }
        self._rpc["env.get_camera_meta"] = self.get_camera_meta

    def get_observation(self) -> dict:
        self.observations += 1
        return {
            "main_images": np.full((H, W, 3), 90, dtype=np.uint8),
            "main_depths": self.depth.copy(),
            "extra_view_images": np.zeros((1, H, W, 3), dtype=np.uint8),
            "extra_view_depths": np.zeros((1, H, W), dtype=np.float32),
        }

    def get_camera_meta(self) -> dict:
        return {
            "observation_camera_map": {"main": "wrist", "extra_0": "ext"},
            "cameras": {
                "wrist": {"intrinsic_K": K.tolist()},
                "ext": {"intrinsic_K": K.tolist()},
            },
        }


def _server(sam3=None, unidepth=None) -> tuple[Server, FakeSam3 | None]:
    server = Server()
    perception = Perception(
        sam3=sam3,
        unidepth=unidepth,
        cameras=FRANKA_CAMERAS,
        intrinsics=lambda key: franka_intrinsics(server.get_camera_meta(), key),
    )
    perception.install(server)
    return server, sam3


def test_nothing_is_registered_without_services() -> None:
    server = Server()
    before = set(server._rpc)
    assert Perception.from_urls(sam3="", unidepth="", cameras=FRANKA_CAMERAS) is None
    assert set(server._rpc) == before
    meta = server._dispatch("env.get_env_meta", (), {})
    assert "perception" not in meta["capabilities"]


def test_segment_all_gives_ids_geometry_and_overlay() -> None:
    server, sam3 = _server(sam3=FakeSam3())
    assert set(server._rpc) >= {
        "env.segment",
        "env.select_detection",
        "env.reject_detection",
    }
    assert "env.enhance_depth" not in server._rpc
    caps = server._dispatch("env.get_env_meta", (), {})["capabilities"]
    assert caps == {
        "has_vla": True,
        "perception": {"segment": True, "enhance_depth": False},
    }

    # segment captures a fresh observation itself (never a frame older than the call).
    server._dispatch("env.get_observation", (), {})
    before = server.observations
    out = server._dispatch(
        "env.segment", (), {"camera": "wrist", "prompt": " bowl ", "all": True}
    )

    assert server.observations == before + 1, "a fresh capture per segment"
    assert sam3.calls[0]["text_prompt"] == "bowl" and sam3.calls[0]["all"] is True
    assert out["found"] and out["count"] == 2 and out["ids"] == ["d1", "d2"]
    assert out["observation"] == 0 and out["invalidated"] == []
    first, second = out["detections"]
    assert first["id"] == "d1" and first["score"] == 0.9 and first["rank"] == 0
    assert first["centroid_rc"] == [7, 14] and first["depth_m"] == pytest.approx(0.5)
    assert first["point_camera"] == pytest.approx([-0.025, -0.0625, 0.5])
    assert "mask" not in first and first["mask_png_base64"]
    assert second["depth_m"] is None, "no sensor depth under the second object"
    assert out["overlay"].shape == (H, W, 3) and out["overlay"].dtype == np.uint8
    assert tuple(out["overlay"][7, 14]) != (90, 90, 90)
    assert tuple(out["overlay"][0, 0]) == (90, 90, 90)

    # all=False: the best mask only, one id.
    out = server._dispatch("env.segment", (), {"point": [7, 14]})
    assert sam3.calls[1]["point"] == [7, 14] and "all" not in sam3.calls[1]
    assert out["ids"] == ["d3"] and out["count"] == 1
    out = server._dispatch("env.segment", (), {"prompt": "x", "min_score": 0.95})
    assert out == {
        "found": False,
        "observation": 0,
        "camera": "wrist",
        "count": 0,
        "detections": [],
        "ids": [],
        "invalidated": [],
        "reason": "below min_score",
    }
    with pytest.raises(ValueError, match="text prompt or a point"):
        server._dispatch("env.segment", (), {})
    with pytest.raises(ValueError, match="unknown camera"):
        server._dispatch("env.segment", (), {"camera": "head", "prompt": "x"})


def test_ids_are_invalidated_by_a_new_observation_and_reported_once() -> None:
    server, _ = _server(sam3=FakeSam3())
    server._dispatch("env.get_observation", (), {})
    ids = server._dispatch("env.segment", (), {"prompt": "bowl", "all": True})["ids"]
    ok = server._dispatch("env.select_detection", (), {"id": ids[0]})
    assert ok["ok"] and ok["selected"] == ids[0] and ok["detection"]["id"] == ids[0]
    assert "mask" not in ok["detection"]
    rej = server._dispatch("env.reject_detection", (), {"id": ids[1]})
    assert rej["ok"] and rej["rejected"] == [ids[1]] and rej["selected"] == ids[0]

    server.depth[2:10, 10:18] = 0.35  # someone moved the object; the arm did not move
    server._dispatch("env.get_observation", (), {})
    stale = server._dispatch("env.select_detection", (), {"id": ids[0]})
    assert stale["ok"] is False
    assert "belongs to observation 0" in stale["error"]
    assert stale["invalidated"] == ids, "reported on the first perception call after"
    assert (
        stale["ids"] == [] and stale["selected"] is None and stale["observation"] == 1
    )
    again = server._dispatch("env.reject_detection", (), {"id": ids[0]})
    assert again["ok"] is False and again["invalidated"] == [], "reported only once"
    unknown = server._dispatch("env.select_detection", (), {"id": "d99"})
    assert "unknown detection id" in unknown["error"]

    # Fresh ids after re-segmenting; the old ones were never reused.
    fresh = server._dispatch("env.segment", (), {"prompt": "bowl", "all": True})
    assert fresh["ids"] == ["d3", "d4"] and fresh["observation"] == 1


def test_enhance_depth_replaces_the_camera_depth_for_later_segments() -> None:
    server, _ = _server(sam3=FakeSam3(), unidepth=FakeUniDepth())
    assert "env.enhance_depth" in server._rpc
    server._dispatch("env.get_observation", (), {})
    before = server._dispatch("env.segment", (), {"prompt": "bowl", "all": True})
    assert before["detections"][1]["depth_m"] is None

    out = server._dispatch("env.enhance_depth", (), {"camera": "wrist"})
    assert out["ok"] and out["observation"] == 0
    assert out["report"]["mode"] == "filled"
    assert out["report"]["scale"] == pytest.approx(2.0)
    assert out["report"]["filled_pixels"] == 36
    assert out["depth"].shape == (H, W) and out["depth"][15, 3] == pytest.approx(0.5)
    assert out["depth"][0, 0] == pytest.approx(0.5), "sensor pixels untouched"
    assert out["estimate"] == {"model": "fake"}

    after = server._dispatch("env.segment", (), {"prompt": "bowl", "all": True})
    assert after["detections"][1]["depth_m"] == pytest.approx(0.5)
    assert after["ids"] == ["d3", "d4"], (
        "enhancing depth keeps the observation (no invalidation)"
    )
    assert after["invalidated"] == []

    # A camera without any depth takes the estimate as-is.
    server.get_observation = lambda: {"main_images": np.zeros((H, W, 3), np.uint8)}
    server._rpc["env.get_observation"] = server.get_observation
    Perception(
        unidepth=FakeUniDepth(),
        cameras=FRANKA_CAMERAS,
    ).install(server)
    server._dispatch("env.get_observation", (), {})
    out = server._dispatch("env.enhance_depth", (), {})
    assert out["report"]["mode"] == "mono_only" and out["depth"][0, 0] == pytest.approx(
        0.25
    )


def test_enhance_depth_only_server_has_no_segment() -> None:
    server, _ = _server(unidepth=FakeUniDepth())
    assert "env.segment" not in server._rpc and "env.enhance_depth" in server._rpc
    caps = server._dispatch("env.get_env_meta", (), {})["capabilities"]["perception"]
    assert caps == {"segment": False, "enhance_depth": True}


def test_an_observation_of_an_unmoved_robot_keeps_the_ids() -> None:
    """With the facade's state digest, env.get_observation expires the ids only when the robot
    moved since they were cut; a refused motion (nothing moved) keeps them too."""
    from pi_embodied_services.utils.detections import state_digest

    server = Server()
    pose = {"raw_base_state": {"tcp_pose": [0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]}}

    def move_delta(dz: float) -> dict:
        if abs(dz) > 0.08:
            raise ValueError("limit is 0.08 m per call")
        pose["raw_base_state"]["tcp_pose"][2] += dz
        return {"ok": True}

    server._rpc["env.move_delta"] = move_delta
    perception = Perception(sam3=FakeSam3(), cameras=FRANKA_CAMERAS)
    perception.epoch.set_digest(lambda: state_digest(pose))
    perception.install(server)
    server._dispatch("env.get_observation", (), {})
    ids = server._dispatch("env.segment", (), {"prompt": "bowl"})["ids"]
    server._dispatch("env.get_observation", (), {})
    with pytest.raises(ValueError, match="0.08"):
        server._dispatch("env.move_delta", (0.5,), {})
    server._dispatch("env.get_observation", (), {})
    kept = server._dispatch("env.select_detection", (), {"id": ids[0]})
    assert kept["ok"] and kept["invalidated"] == [], "nothing moved: the id is current"
    server._dispatch("env.move_delta", (0.05,), {})
    server._dispatch("env.get_observation", (), {})
    stale = server._dispatch("env.select_detection", (), {"id": ids[0]})
    assert stale["ok"] is False and stale["invalidated"] == ids


def test_perception_reads_per_camera_dict_frames_for_the_dual_arm_views() -> None:
    """The dual Franka's env.segment works on its projection views (the planner's names)."""
    from pi_embodied_services.robots.franka.grasp_views import dual_perception_layout

    views = {"d455": {"raw_key": "d455_rgb", "calibration_key": "d455_camera"}}
    meta = {"d455_rgb": {"color_intrinsics": {"fx": 20, "fy": 20, "ppx": 8, "ppy": 8}}}
    cameras, intrinsics = dual_perception_layout(views, lambda: meta)
    assert cameras == {"d455": ("raw_camera_frames", "raw_camera_depths", "d455_rgb")}
    assert intrinsics("d455_rgb")[0, 0] == 20
    server = Server()
    server._rpc["env.get_observation"] = lambda: {
        "raw_camera_frames": {"d455_rgb": np.full((H, W, 4), 90, dtype=np.uint8)},
        "raw_camera_depths": {"d455_rgb": np.full((H, W), 0.5, dtype=np.float32)},
    }
    Perception(sam3=FakeSam3(), cameras=cameras, intrinsics=intrinsics).install(server)
    server._dispatch("env.get_observation", (), {})
    out = server._dispatch("env.segment", (), {"camera": "d455", "prompt": "bowl"})
    assert out["found"] and out["camera"] == "d455"
    assert out["detections"][0]["point_camera"] is not None, (
        "K came from the view's meta"
    )


def test_a_moved_object_or_a_reset_expires_ids_under_a_still_arm() -> None:
    """Audit: the digest kept ids while the arm stood still, even after a person moved an
    object or the scene was reset. The frames now expire them; a reset always does."""
    from pi_embodied_services.utils.detections import state_digest

    server = Server()
    pose = {"raw_base_state": {"tcp_pose": [0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]}}
    server._rpc["env.reset"] = lambda: {"ok": True}
    perception = Perception(sam3=FakeSam3(), cameras=FRANKA_CAMERAS)
    perception.epoch.set_digest(lambda: state_digest(pose))
    perception.install(server)
    ids = server._dispatch("env.segment", (), {"prompt": "bowl"})["ids"]
    server._dispatch("env.get_observation", (), {})
    assert server._dispatch("env.select_detection", (), {"id": ids[0]})["ok"]
    server.depth[2:10, 10:18] = 0.35  # moved by hand
    fresh = server._dispatch("env.segment", (), {"prompt": "bowl"})
    assert fresh["invalidated"] == ids and fresh["ids"] != ids
    kept = server._dispatch("env.select_detection", (), {"id": fresh["ids"][0]})
    assert kept["ok"], "the new frame's id is current"
    server._dispatch("env.reset", (), {})  # the arm reads the same pose after the reset
    after = server._dispatch("env.select_detection", (), {"id": fresh["ids"][0]})
    assert after["ok"] is False and "stale" in after["error"]


# ---- Perception over a simulator's rendered views ----------------------------------


class SimServer(RpcFacade):
    """A simulator-shaped env server: env.render_camera / env.get_camera_meta, its own
    env.segment code primitive, env.move_delta; no env.get_observation."""

    SERVICE_NAME = "fake-sim-env"

    def __init__(self, depth: bool = True) -> None:
        super().__init__()
        self.renders = 0
        self.z = 0.5
        self._depth = depth
        self._rpc["env.get_env_meta"] = lambda: {"task": "t", "capabilities": {}}
        self._rpc["env.render_camera"] = self.render_camera
        self._rpc["env.get_camera_meta"] = lambda camera_name, **_: {
            "intrinsic_K": K.tolist(),
            "extrinsic_cam2world": np.eye(4),
        }
        self._rpc["env.segment"] = lambda prompt, camera="agentview": {"own": True}
        self._rpc["env.move_delta"] = self.move_delta

    def render_camera(self, camera_name, depth=False, **_):
        self.renders += 1
        rgb = np.full((H, W, 3), 90, dtype=np.uint8)
        if not (depth and self._depth):
            return rgb
        return [rgb, np.full((H, W), self.z, dtype=np.float32)]

    def move_delta(self, dz: float) -> dict:
        self.z += dz
        return {"ok": True}


def _sim(sam3=None, unidepth=None, depth=True):
    server = SimServer(depth)
    perception = Perception(
        sam3=sam3,
        unidepth=unidepth,
        cameras=["agentview", "wrist"],
        view=render_view(server),
        names=SIM_NAMES,
    )
    perception.install(server)
    return server, perception


def test_a_rendered_view_serves_detect_beside_the_servers_own_segment() -> None:
    server, _ = _sim(sam3=FakeSam3())
    assert server._dispatch("env.segment", ("bowl",), {}) == {"own": True}
    meta = server._dispatch("env.get_env_meta", (), {})["capabilities"]["perception"]
    assert meta == {"segment": True, "enhance_depth": False}
    assert "env.enhance_depth" not in server._rpc
    out = server._dispatch(
        "env.detect", (), {"camera": "agentview", "prompt": "bowl", "all": True}
    )
    assert out["found"] and len(out["ids"]) == 2
    assert out["detections"][0]["depth_m"] == pytest.approx(0.5)
    assert out["detections"][0]["point_camera"] is not None, "K from get_camera_meta"
    renders = server.renders
    # The same observation: the frame is rendered once.
    server._dispatch("env.detect", (), {"camera": "agentview", "prompt": "cup"})
    assert server.renders == renders
    assert server._dispatch("env.select_detection", (), {"id": out["ids"][1]})["ok"]
    # A motion starts a new observation: the ids expire and the next call renders again.
    server._dispatch("env.move_delta", (0.1,), {})
    stale = server._dispatch("env.select_detection", (), {"id": out["ids"][0]})
    assert stale["ok"] is False and set(stale["invalidated"]) >= set(out["ids"])
    again = server._dispatch(
        "env.detect", (), {"camera": "agentview", "prompt": "bowl"}
    )
    assert server.renders == renders + 1
    assert again["detections"][0]["depth_m"] == pytest.approx(0.6)
    with pytest.raises(ValueError, match="unknown camera"):
        server._dispatch("env.detect", (), {"camera": "top", "prompt": "bowl"})


def test_enhance_depth_supplies_depth_to_a_view_without_it() -> None:
    server, _ = _sim(sam3=FakeSam3(), unidepth=FakeUniDepth(), depth=False)
    assert "env.select_detection" in server._rpc and "env.enhance_depth" in server._rpc
    bare = server._dispatch("env.detect", (), {"camera": "wrist", "prompt": "bowl"})
    assert bare["detections"][0]["depth_m"] is None
    out = server._dispatch("env.enhance_depth", (), {"camera": "wrist"})
    assert out["ok"] and out["depth"].shape == (H, W)
    after = server._dispatch("env.detect", (), {"camera": "wrist", "prompt": "bowl"})
    assert after["detections"][0]["depth_m"] == pytest.approx(0.25)
    # The estimate belongs to this observation only.
    server._dispatch("env.move_delta", (0.1,), {})
    moved = server._dispatch("env.detect", (), {"camera": "wrist", "prompt": "bowl"})
    assert moved["detections"][0]["depth_m"] is None


def test_without_a_view_the_cameras_must_name_observation_keys() -> None:
    with pytest.raises(ValueError, match="observation keys"):
        Perception(sam3=FakeSam3(), cameras=["wrist"])


def test_install_perception_from_args_on_a_real_robots_observation_layout() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    add_perception_arguments(parser, sam3=True)
    off = parser.parse_args([])
    server = Server()
    before = set(server._rpc)
    assert install_perception(server, off, cameras={"cam": ("a", "b", "c")}) is None
    assert set(server._rpc) == before, "nothing without --sam3 / --unidepth"

    server = Server()
    server._rpc["env.get_observation"] = lambda: {
        "images": {"front": np.full((H, W, 3), 90, dtype=np.uint8)},
        "depths": {},
    }
    args = parser.parse_args(["--sam3", "http://sam3", "--unidepth", "http://unidepth"])
    perception = install_perception(
        server,
        args,
        cameras={"front": ("images", "depths", "front")},
        intrinsics=lambda key: K if key == "front" else None,
    )
    assert perception is not None and perception.methods() == {
        "segment": "env.detect",
        "select_detection": "env.select_detection",
        "reject_detection": "env.reject_detection",
        "enhance_depth": "env.enhance_depth",
    }
    assert "env.segment" not in server._rpc, "the id primitive is env.detect here"


def test_every_env_server_but_the_frankas_installs_perception() -> None:
    """Every robot's env server takes --sam3 / --unidepth and installs the primitives (the
    Franka servers install theirs under the Franka names; HumanCLAW runs the paper's skills-only
    tier, no perception primitive)."""
    from pathlib import Path

    robots = Path(__file__).resolve().parents[1] / "pi_embodied_services" / "robots"
    servers = sorted(robots.glob("*/env_server.py"))
    assert len(servers) >= 13
    for path in servers:
        text = path.read_text()
        if path.parent.name in (
            "franka",
            "franka_polymetis",
            "dual_franka",
            "humanclaw",
        ):
            continue
        assert "install_perception(" in text, path.parent.name
        assert "add_perception_arguments(" in text, path.parent.name


def test_install_grasp_planner_shares_the_perception_ids() -> None:
    """The simulators' planner (utils/grasp.install_grasp_planner): nothing without a grasp or
    place URL; with one its methods are installed (code.api comes from the robot's manifest), its views are render_view's
    (with the camera pose) and env.detect ids live on its epoch."""
    import argparse

    from pi_embodied_services.utils.grasp import (
        add_grasp_arguments,
        install_grasp_planner,
    )

    parser = argparse.ArgumentParser()
    add_perception_arguments(parser, sam3=True)
    add_grasp_arguments(parser)
    server, perception = _sim(sam3=FakeSam3())
    view = render_view(server)
    assert view("agentview")["extrinsic_cam2world"].shape == (4, 4)
    off = parser.parse_args([])
    assert (
        install_grasp_planner(
            server, off, view=view, cameras=["agentview"], eef_pose=lambda a: None
        )
        is None
    )
    assert "env.plan_grasp" not in server._rpc
    on = parser.parse_args(["--anyplace", "http://anyplace"])
    planner = install_grasp_planner(
        server,
        on,
        view=view,
        cameras=["agentview", "wrist"],
        eef_pose=lambda a: (np.zeros(3), np.array([1.0, 0, 0, 0])),
        perception=perception,
    )
    assert planner is not None
    for m in (
        "env.plan_grasp",
        "env.claim_waypoints",
        "env.resolve_grasp",
        "env.plan_place",
    ):
        assert m in server._rpc, m
    ids = server._dispatch("env.detect", (), {"camera": "agentview", "prompt": "bowl"})[
        "ids"
    ]
    assert perception.book.epoch is planner._epoch, "one observation clock"
    server._dispatch("env.move_delta", (0.1,), {})
    assert server._dispatch("env.select_detection", (), {"id": ids[0]})["ok"] is False
