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
from PIL import Image

from pi_embodied_services.utils.detections import (
    DetectionBook,
    DetectionStale,
    decode_mask_png,
    describe_mask,
    overlay_masks,
)
from pi_embodied_services.utils.perception import (
    FRANKA_CAMERAS,
    Perception,
    franka_intrinsics,
)
from pi_embodied_services.utils.rpc import RpcFacade

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

    with pytest.raises(ValueError, match="take an observation first"):
        server._dispatch("env.segment", (), {"text_prompt": "bowl"})
    server._dispatch("env.get_observation", (), {})
    out = server._dispatch(
        "env.segment", (), {"camera": "wrist", "text_prompt": " bowl ", "all": True}
    )

    assert sam3.calls[0]["text_prompt"] == "bowl" and sam3.calls[0]["all"] is True
    assert out["found"] and out["count"] == 2 and out["ids"] == ["d1", "d2"]
    assert out["observation"] == 1 and out["invalidated"] == []
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
    out = server._dispatch("env.segment", (), {"text_prompt": "x", "min_score": 0.95})
    assert out == {
        "found": False,
        "observation": 1,
        "camera": "wrist",
        "count": 0,
        "detections": [],
        "ids": [],
        "invalidated": [],
        "reason": "below min_score",
    }
    with pytest.raises(ValueError, match="text_prompt or a point"):
        server._dispatch("env.segment", (), {})
    with pytest.raises(ValueError, match="unknown camera"):
        server._dispatch("env.segment", (), {"camera": "head", "text_prompt": "x"})


def test_ids_are_invalidated_by_a_new_observation_and_reported_once() -> None:
    server, _ = _server(sam3=FakeSam3())
    server._dispatch("env.get_observation", (), {})
    ids = server._dispatch("env.segment", (), {"text_prompt": "bowl", "all": True})[
        "ids"
    ]
    ok = server._dispatch("env.select_detection", (), {"id": ids[0]})
    assert ok["ok"] and ok["selected"] == ids[0] and ok["detection"]["id"] == ids[0]
    assert "mask" not in ok["detection"]
    rej = server._dispatch("env.reject_detection", (), {"id": ids[1]})
    assert rej["ok"] and rej["rejected"] == [ids[1]] and rej["selected"] == ids[0]

    server._dispatch("env.get_observation", (), {})  # the robot moved
    stale = server._dispatch("env.select_detection", (), {"id": ids[0]})
    assert stale["ok"] is False
    assert "belongs to observation 1" in stale["error"]
    assert stale["invalidated"] == ids, "reported on the first perception call after"
    assert (
        stale["ids"] == [] and stale["selected"] is None and stale["observation"] == 2
    )
    again = server._dispatch("env.reject_detection", (), {"id": ids[0]})
    assert again["ok"] is False and again["invalidated"] == [], "reported only once"
    unknown = server._dispatch("env.select_detection", (), {"id": "d99"})
    assert "unknown detection id" in unknown["error"]

    # Fresh ids after re-segmenting; the old ones were never reused.
    fresh = server._dispatch("env.segment", (), {"text_prompt": "bowl", "all": True})
    assert fresh["ids"] == ["d3", "d4"] and fresh["observation"] == 2


def test_enhance_depth_replaces_the_camera_depth_for_later_segments() -> None:
    server, _ = _server(sam3=FakeSam3(), unidepth=FakeUniDepth())
    assert "env.enhance_depth" in server._rpc
    server._dispatch("env.get_observation", (), {})
    before = server._dispatch("env.segment", (), {"text_prompt": "bowl", "all": True})
    assert before["detections"][1]["depth_m"] is None

    out = server._dispatch("env.enhance_depth", (), {"camera": "wrist"})
    assert out["ok"] and out["observation"] == 1
    assert out["report"]["mode"] == "filled"
    assert out["report"]["scale"] == pytest.approx(2.0)
    assert out["report"]["filled_pixels"] == 36
    assert out["depth"].shape == (H, W) and out["depth"][15, 3] == pytest.approx(0.5)
    assert out["depth"][0, 0] == pytest.approx(0.5), "sensor pixels untouched"
    assert out["estimate"] == {"model": "fake"}

    after = server._dispatch("env.segment", (), {"text_prompt": "bowl", "all": True})
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
