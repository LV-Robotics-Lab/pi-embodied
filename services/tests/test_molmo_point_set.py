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

"""molmo_server backends: the MolmoPoint ``molmo.ground_set`` request/response and the Molmo2
``molmo.ground`` output staying as it was, against fake models (no weights)."""

from __future__ import annotations

import base64
import contextlib
import io
import threading

import numpy as np
import pytest
from PIL import Image

from pi_embodied_services.components.molmo_server import BACKENDS, MolmoFacade


def _png_base64(w: int, h: int) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def to(self, device):
        return self

    def __getitem__(self, item):
        return self

    def size(self, dim):
        return self.shape[dim]


class FakeTorch:
    def inference_mode(self):
        return contextlib.nullcontext()

    class cuda:
        @staticmethod
        def empty_cache():
            pass


class FakePointProcessor:
    """MolmoPoint's remote-code processor surface used by molmo.ground_set."""

    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs["return_pointing_metadata"] is True and kwargs["padding"] is True
        return {
            "input_ids": FakeTensor((1, 7)),
            "metadata": {
                "token_pooling": "tp",
                "subpatch_mapping": "sm",
                "image_sizes": "is",
            },
        }

    def post_process_image_text_to_text(self, tokens, **kwargs):
        return ["<points>the bowl</points>"]


class FakePointModel:
    device = "cuda:0"

    def __init__(self, raw_points):
        self.raw_points = raw_points
        self.generate_kwargs = None

    def build_logit_processor_from_inputs(self, inputs):
        return ["lp"]

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return FakeTensor((1, 12))

    def extract_image_points(self, text, token_pooling, subpatch_mapping, image_sizes):
        assert (token_pooling, subpatch_mapping, image_sizes) == ("tp", "sm", "is")
        return np.asarray(self.raw_points, dtype=np.float32)


def _facade(backend: str, model, processor) -> MolmoFacade:
    facade = MolmoFacade.__new__(MolmoFacade)
    MolmoFacade.__mro__[1].__init__(facade)  # RpcFacade.__init__
    facade.backend = backend
    facade._model = model
    facade._processor = processor
    facade._torch = FakeTorch()
    facade._lock = threading.Lock()
    facade._register_rpc()
    return facade


def test_backend_aliases() -> None:
    assert BACKENDS["allenai/molmopoint-8b"] == "molmopoint"
    assert BACKENDS["allenai/molmo2-8b"] == "molmo2"


def test_point_set_tags_points_with_their_image_index() -> None:
    # (object id, image index, x, y): two points in image 0, one in image 1, one outside.
    model = FakePointModel(
        [[0, 0, 10.0, 20.0], [0, 1, 5.5, 6.5], [1, 0, 30.0, 40.0], [1, 1, 99.0, 1.0]]
    )
    processor = FakePointProcessor()
    facade = _facade("molmopoint", model, processor)
    assert set(facade._rpc) >= {"molmo.ground", "molmo.ground_set"}

    out = facade._dispatch(
        "molmo.ground_set",
        (),
        {
            "images_base64": [_png_base64(64, 48), _png_base64(32, 16)],
            "query": " Point to the bowl in Image 2. ",
        },
    )
    assert out["points"] == [
        {"id": "point_000", "image_index": 0, "pixel_x": 10.0, "pixel_y": 20.0},
        {"id": "point_001", "image_index": 1, "pixel_x": 5.5, "pixel_y": 6.5},
        {"id": "point_002", "image_index": 0, "pixel_x": 30.0, "pixel_y": 40.0},
    ], "the point outside image 1 (x=99 of 32) is dropped"
    assert out["point_count"] == 3
    assert out["image_sizes"] == [[64, 48], [32, 16]]
    assert out["answer"] == "<points>the bowl</points>"
    assert out["coordinate_convention"]["origin"] == "top_left"
    # The prompt is preserved as authored (stripped), images follow it in order.
    content = processor.messages[0]["content"]
    assert content[0] == {"type": "text", "text": "Point to the bowl in Image 2."}
    assert [c["type"] for c in content[1:]] == ["image", "image"]
    assert content[1]["image"].size == (64, 48) and content[2]["image"].size == (32, 16)
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == 200
    assert model.generate_kwargs["logits_processor"] == ["lp"]
    assert "metadata" not in model.generate_kwargs


def test_point_set_validates_its_inputs() -> None:
    facade = _facade("molmopoint", FakePointModel([]), FakePointProcessor())
    with pytest.raises(ValueError, match="1 to 4"):
        facade.ground_set([], "x")
    with pytest.raises(ValueError, match="1 to 4"):
        facade.ground_set([_png_base64(4, 4)] * 5, "x")
    with pytest.raises(ValueError, match="non-empty query"):
        facade.ground_set([_png_base64(4, 4)], "  ")
    with pytest.raises(ValueError, match=r"images_base64\[0\] must be a base64 string"):
        facade.ground_set([1], "x")
    out = facade.ground_set([_png_base64(4, 4)], "x")
    assert out["points"] == [] and out["point_count"] == 0


def test_point_set_refuses_an_image_index_outside_the_set() -> None:
    facade = _facade(
        "molmopoint", FakePointModel([[0, 3, 1.0, 1.0]]), FakePointProcessor()
    )
    with pytest.raises(RuntimeError, match="image index 3"):
        facade.ground_set([_png_base64(4, 4)], "x")


def test_molmopoint_ground_is_the_one_image_case() -> None:
    facade = _facade(
        "molmopoint", FakePointModel([[0, 0, 3.0, 4.0]]), FakePointProcessor()
    )
    out = facade._dispatch(
        "molmo.ground", (), {"image_base64": _png_base64(8, 6), "query": "the bowl"}
    )
    assert out == {
        "point_xy": [3.0, 4.0],
        "answer": "<points>the bowl</points>",
        "image_size": [8, 6],
    }
    facade = _facade("molmopoint", FakePointModel([]), FakePointProcessor())
    out = facade._dispatch(
        "molmo.ground", (), {"image_base64": _png_base64(8, 6), "query": "the bowl"}
    )
    assert out == {"answer": "<points>the bowl</points>", "image_size": [8, 6]}


class FakeMolmo2Processor:
    class tokenizer:
        @staticmethod
        def decode(tokens, skip_special_tokens=False):
            return '<points coords="1 1 500 250">the bowl</points>'

    def apply_chat_template(self, messages, **kwargs):
        assert "return_pointing_metadata" not in kwargs
        return {"input_ids": FakeTensor((1, 5))}


class FakeMolmo2Model:
    device = "cuda:0"

    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 48 and "logits_processor" not in kwargs
        return FakeTensor((1, 9))


def test_molmo2_ground_output_is_unchanged_and_has_no_point_set() -> None:
    facade = _facade("molmo2", FakeMolmo2Model(), FakeMolmo2Processor())
    assert "molmo.ground_set" not in facade._rpc
    out = facade._dispatch(
        "molmo.ground", (), {"image_base64": _png_base64(200, 100), "query": "the bowl"}
    )
    assert out == {
        "point_xy": [100.0, 25.0],
        "answer": '<points coords="1 1 500 250">the bowl</points>',
        "image_size": [200, 100],
    }
    with pytest.raises(ValueError, match="--model molmopoint"):
        facade.ground_set([_png_base64(4, 4)], "x")
