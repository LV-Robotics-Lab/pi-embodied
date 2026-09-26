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

"""The OpenVLA / OpenVLA-OFT / GR00T adapters with fake policies (no model, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from pi_embodied_services.components.gr00t_server import ACTION_KEYS, Gr00tFacade
from pi_embodied_services.components.openvla_oft_server import OpenVLAOFTFacade
from pi_embodied_services.components.openvla_server import OpenVLAFacade, prompt_for
from pi_embodied_services.components.vla_adapter_base import (
    center_crop_resize,
    frame_of,
    libero_gripper,
    resize_jpeg_lanczos,
)


def obs(wrist=True, state_dim=8):
    rng = np.random.default_rng(0)
    main = rng.integers(0, 255, (1, 256, 256, 3), dtype=np.uint8)
    return {
        "main_images": main,
        "wrist_images": rng.integers(0, 255, (1, 256, 256, 3), dtype=np.uint8)
        if wrist
        else None,
        "extra_view_images": None,
        "states": np.arange(state_dim, dtype=np.float32)[None] / 10,
        "task_descriptions": ["Pick up the black bowl"],
    }


def call(facade, method="vla.predict", *args):
    return facade._serve_dispatch(method, args, {})


class Sampler:
    """A fake policy whose actions come from numpy's global RNG (what `seeded` must pin)."""

    def __init__(self, horizon, gr00t=False):
        self.horizon = horizon
        self.gr00t = gr00t
        self.seen = []
        self.seen_out = []

    def __call__(self, *args):
        self.seen.append(args)
        raw = np.random.uniform(-1, 1, (self.horizon, 7)).astype(np.float32)
        raw[:, 6] = np.random.uniform(0, 1, self.horizon)
        self.seen_out.append(raw)
        if self.gr00t:  # Gr00tPolicy.get_action: {key: [B, horizon, d]}
            return {k: raw[None, :, i : i + 1] for i, k in enumerate(ACTION_KEYS)}
        return raw[0] if self.horizon == 1 else raw


def gr00t_out(h, value):
    return {k: np.full((1, h, 1), value, np.float32) for k in ACTION_KEYS}


# ---- shared adapter behaviour -------------------------------------------------------------

ADAPTERS = [
    ("openvla", lambda pol: OpenVLAFacade(policy=pol, model="m", revision="r"), 1),
    (
        "openvla-oft",
        lambda pol: OpenVLAOFTFacade(policy=pol, model="m", revision="r"),
        8,
    ),
    (
        "gr00t",
        lambda pol: Gr00tFacade(policy=pol, model="m", revision="r", horizon=16),
        16,
    ),
]


@pytest.mark.parametrize("service,make,horizon", ADAPTERS)
def test_predict_shape_seed_and_framework_methods(service, make, horizon):
    vla = make(Sampler(horizon, gr00t=service == "gr00t"))
    assert call(vla, "healthz")["service"] == service
    info = call(vla, "vla.info")
    assert (info["model"], info["revision"], info["horizon"], info["action_dim"]) == (
        "m",
        "r",
        horizon,
        7,
    )
    assert call(vla, "vla.reset") == {"ok": True}

    a = call(vla, "vla.predict", obs(), {"mode": "eval", "seed": 7})
    assert a.dtype == np.float32 and a.shape == (1, horizon, 7)
    assert set(np.unique(a[..., 6])) <= {-1.0, 1.0}, (
        "gripper is LIBERO's -1 open / +1 close"
    )
    call(vla, "vla.predict", obs(), None)  # an unseeded call in between changes nothing
    assert np.array_equal(call(vla, "vla.predict", obs(), {"seed": 7}), a)
    assert not np.array_equal(call(vla, "vla.predict", obs(), {"seed": 8}), a)


@pytest.mark.parametrize("service,make,horizon", ADAPTERS)
def test_wrong_policy_output_is_an_error(service, make, horizon):
    gr00t = service == "gr00t"
    long = (
        gr00t_out(horizon + 1, 0.0) if gr00t else np.zeros((horizon + 1, 7), np.float32)
    )
    vla = make(lambda *a: long)
    with pytest.raises((RuntimeError, ValueError)):
        vla.predict(obs(), None)
    nans = (
        gr00t_out(horizon, np.nan)
        if gr00t
        else np.full((horizon, 7), np.nan, np.float32)
    )
    vla = make(lambda *a: nans)
    with pytest.raises(RuntimeError, match="expected finite actions"):
        vla.predict(obs(), None)


def test_frame_of_checks_the_wire_dict():
    f = frame_of(obs())
    assert f.main.shape == (256, 256, 3) and f.wrist.shape == (256, 256, 3)
    assert f.state.shape == (8,) and f.instruction == "Pick up the black bowl"
    assert frame_of(obs(wrist=False)).wrist is None
    bad = obs()
    bad["main_images"] = bad["main_images"].astype(np.float32)
    with pytest.raises(ValueError, match="main_images"):
        frame_of(bad)
    bad = obs()
    bad["task_descriptions"] = []
    with pytest.raises(ValueError, match="task_descriptions"):
        frame_of(bad)
    bad = obs()
    bad["main_images"] = np.zeros((2, 256, 256, 3), np.uint8)
    with pytest.raises(ValueError, match="batch"):
        frame_of(bad)


# ---- the preprocessing / postprocessing the adapters copy from the model repos -----------


def test_libero_gripper_binarises_and_inverts():
    raw = np.zeros((3, 7), np.float32)
    raw[:, 6] = [0.9, 0.1, 0.5]
    out = libero_gripper(raw)
    assert out[:, 6].tolist() == [
        -1.0,
        1.0,
        0.0,
    ]  # sign(2g-1) flipped: dataset open -> LIBERO -1
    assert raw[0, 6] == 0.9, "the input is not modified"


def test_resizes():
    big = np.random.default_rng(1).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    small = resize_jpeg_lanczos(big, 224)
    assert small.shape == (224, 224, 3) and small.dtype == np.uint8
    flat = np.full((224, 224, 3), 200, np.uint8)
    flat[100:124, 100:124] = 20
    cropped = center_crop_resize(flat, 0.9)
    assert cropped.shape == flat.shape and cropped.dtype == np.uint8
    # The crop zooms in: the dark square grows (24 px -> ~25 px) and stays centered.
    dark = np.argwhere(cropped[..., 0] < 110)
    assert 24 < dark[:, 0].max() - dark[:, 0].min() + 1 <= 27
    assert abs(dark.mean() - 111.5) < 1


# ---- per-model wiring -----------------------------------------------------------------------


def test_openvla_sees_a_224_frame_and_the_prompt_is_openvlas():
    pol = Sampler(1)
    OpenVLAFacade(policy=pol, model="m", revision=None).predict(obs(), None)
    image, instruction = pol.seen[-1]
    assert image.shape == (224, 224, 3) and image.dtype == np.uint8
    assert instruction == "Pick up the black bowl"
    assert (
        prompt_for(instruction)
        == "In: What action should the robot take to pick up the black bowl?\nOut:"
    )


def test_openvla_oft_passes_the_env_frames_state_and_needs_the_wrist():
    pol = Sampler(8)
    o = obs()
    OpenVLAOFTFacade(policy=pol, model="m", revision=None).predict(o, None)
    main, wrist, state, instruction = pol.seen[-1]
    # RLinf's LiberoEnv already rotates the frames 180 degrees; the adapter must not do it again.
    assert np.array_equal(main, o["main_images"][0])
    assert np.array_equal(wrist, o["wrist_images"][0])
    assert state.shape == (8,) and instruction == "Pick up the black bowl"
    with pytest.raises(ValueError, match="wrist"):
        OpenVLAOFTFacade(policy=pol, model="m", revision=None).predict(
            obs(wrist=False), None
        )


def test_gr00t_builds_the_libero_panda_observation():
    pol = Sampler(16, gr00t=True)
    o = obs()
    Gr00tFacade(policy=pol, model="m", revision=None, horizon=16).predict(o, None)
    (gr00t_obs,) = pol.seen[-1]
    # The libero_panda embodiment as Gr00tPolicy takes it: videos [B=1, T=1, H, W, 3] uint8, states
    # [1, 1, d] per key (x..yaw 1-D, gripper 2-D), the instruction as [[str]] under the language key.
    assert gr00t_obs["video"]["image"].shape == (1, 1, 256, 256, 3)
    assert np.array_equal(gr00t_obs["video"]["wrist_image"][0, 0], o["wrist_images"][0])
    assert (
        gr00t_obs["state"]["x"].shape == (1, 1, 1)
        and gr00t_obs["state"]["x"][0, 0, 0] == 0
    )
    assert gr00t_obs["state"]["gripper"].shape == (1, 1, 2)
    assert gr00t_obs["state"]["gripper"][0, 0].tolist() == pytest.approx([0.6, 0.7])
    assert gr00t_obs["language"] == {
        "annotation.human.action.task_description": [["Pick up the black bowl"]]
    }
    a = Gr00tFacade(policy=pol, model="m", revision=None, horizon=16).predict(
        o, {"seed": 1}
    )
    assert a.shape == (1, 16, 7) and set(np.unique(a[..., 6])) <= {-1.0, 1.0}
    # The fine-tune's gripper is in [0, 1] with 1 = open: like OpenVLA, RLinf thresholds at 0.5 and
    # inverts for LIBERO (-1 open), so 0.9 -> -1 and 0.1 -> +1.
    raw = np.asarray(pol.seen_out[-1], np.float32)
    assert np.array_equal(a[0, :, 6], np.where(raw[:, 6] > 0.5, -1.0, 1.0))
