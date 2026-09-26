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

"""sam3.segment's ``all`` option against a fake SAM3 processor (no model)."""

from __future__ import annotations

import base64
import io
import threading

import numpy as np
import pytest

from pi_embodied_services.components.sam3_server import Sam3Facade
from pi_embodied_services.utils.detections import decode_mask_png

Image = pytest.importorskip("PIL.Image")

H, W = 16, 20


def _png_base64() -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _mask(r0: int, r1: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.float32)
    mask[r0:r1] = 1.0
    return mask


class FakeProcessor:
    """SAM3's text path: three candidates, unsorted, one empty."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def set_image(self, image):
        return {"original_height": H, "original_width": W}

    def set_text_prompt(self, *, prompt, state):
        self.prompts.append(prompt)
        return {
            "masks": np.stack([_mask(0, 4), _mask(0, 0), _mask(8, 16), _mask(4, 6)])[
                :, None
            ],
            "scores": np.array([0.5, 0.95, 0.8, 0.1], dtype=np.float32),
            "boxes": np.array(
                [[0, 0, W, 4], [0, 0, 0, 0], [0, 8, W, 16], [0, 4, W, 6]], np.float32
            ),
        }


def _facade() -> Sam3Facade:
    facade = Sam3Facade.__new__(Sam3Facade)
    Sam3Facade.__mro__[1].__init__(facade)  # RpcFacade.__init__
    facade._torch = None
    facade._device = "cpu"
    facade._model = object()
    facade._processor = FakeProcessor()
    facade._lock = threading.Lock()
    facade._image_digest = None
    facade._image_state = None
    facade._register_rpc()
    return facade


def test_all_returns_every_valid_mask_best_first() -> None:
    facade = _facade()
    out = facade._dispatch(
        "sam3.segment",
        (),
        {"image_base64": _png_base64(), "text_prompt": "bowl", "all": True},
    )
    assert out["found"] is True and out["count"] == 2
    # 0.95 has an empty mask (dropped), 0.1 is below min_score 0.2: 0.8 then 0.5 remain.
    assert [d["score"] for d in out["detections"]] == [
        pytest.approx(0.8),
        pytest.approx(0.5),
    ]
    assert [d["index"] for d in out["detections"]] == [0, 1]
    assert out["detections"][0]["box"] == [0.0, 8.0, W, 16.0]
    assert out["detections"][0]["area_px"] == 8 * W
    assert out["detections"][0]["mask_shape"] == [H, W]
    assert np.array_equal(
        decode_mask_png(out["detections"][0]["mask_png_base64"]), _mask(8, 16) > 0
    )
    assert "score" not in out and "mask_png_base64" not in out, (
        "the all= shape is a list"
    )


def test_all_below_min_score_reports_reason() -> None:
    out = _facade()._dispatch(
        "sam3.segment",
        (),
        {
            "image_base64": _png_base64(),
            "text_prompt": "bowl",
            "all": True,
            "min_score": 0.99,
        },
    )
    assert out == {
        "found": False,
        "count": 0,
        "detections": [],
        "reason": "no candidate at or above min_score 0.990 (top score 0.950)",
    }


def test_single_mask_output_is_unchanged() -> None:
    """The default (all=False) keeps the original shape: the top score wins even when
    its mask is empty, which is then reported as before."""
    out = _facade()._dispatch(
        "sam3.segment", (), {"image_base64": _png_base64(), "text_prompt": "bowl"}
    )
    assert out == {
        "found": False,
        "score": pytest.approx(0.95),
        "box": [0.0, 0.0, 0.0, 0.0],
        "reason": "SAM3 returned an empty mask",
    }
    facade = _facade()
    facade._processor.set_text_prompt = lambda *, prompt, state: {
        "masks": _mask(0, 4)[None, None],
        "scores": np.array([0.7], np.float32),
        "boxes": np.array([[0, 0, W, 4]], np.float32),
    }
    out = facade._dispatch(
        "sam3.segment", (), {"image_base64": _png_base64(), "text_prompt": "bowl"}
    )
    assert set(out) == {"found", "score", "box", "mask_png_base64", "mask_shape"}
    assert out["found"] is True and out["mask_shape"] == [H, W]
    assert np.array_equal(decode_mask_png(out["mask_png_base64"]), _mask(0, 4) > 0)
