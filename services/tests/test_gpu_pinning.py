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

"""utils/gpu.py: one GPU for CUDA and the renderer, from the flag, the deployment or CUDA_VISIBLE_DEVICES."""

from pathlib import Path

import pytest

from pi_embodied_services.utils import gpu


@pytest.fixture
def env(monkeypatch):
    for k in (
        "CUDA_VISIBLE_DEVICES",
        "MUJOCO_EGL_DEVICE_ID",
        "EGL_DEVICE_ID",
        gpu.ENV_DEVICE,
        "CUDA_DEVICE_ORDER",
    ):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_the_gpu_is_the_flag_else_the_deployment_else_the_first_visible(env):
    assert gpu.resolve_gpu(None) is None
    env.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert gpu.resolve_gpu(None) == 1
    env.setenv(gpu.ENV_DEVICE, "0")
    assert gpu.resolve_gpu(None) == 0
    assert gpu.resolve_gpu(1) == 1


def test_egl_servers_pin_cuda_and_the_egl_device_together(env):
    env.setattr(gpu, "_egl_index", lambda g: g)
    env.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert gpu.pin_egl(None) is None, "the GPU is the only visible device"
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "1" == os.environ["EGL_DEVICE_ID"]
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_a_different_egl_order_clears_cuda_visible_devices_for_robosuites_assert(env):
    import os

    env.setattr(gpu, "_egl_index", lambda g: 0)
    assert gpu.pin_egl(1) == 1, "the caller selects CUDA device 1 itself"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "0"


def test_isaac_and_sapien_see_only_their_gpu(env):
    import os

    assert gpu.pin_isaac(None) is None and "CUDA_VISIBLE_DEVICES" not in os.environ
    assert gpu.pin_isaac(1) == 1 and os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    env.setenv(gpu.ENV_DEVICE, "1")
    env.delenv("CUDA_VISIBLE_DEVICES")
    assert gpu.pin_cuda(None) == 1 and os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_every_simulator_env_server_pins_its_gpu():
    robots = Path(gpu.__file__).resolve().parents[1] / "robots"
    pins = {
        "libero": "pin_egl",
        "robosuite": "pin_egl",
        "robocasa": "pin_egl",
        "metaworld": "pin_egl",
        "genesis": "pin_egl",
        "maniskill": "pin_cuda",
        "robotwin": "pin_cuda",
        "robolab": "pin_isaac",
        "robodojo": "pin_isaac",
        "behavior": "pin_isaac",
    }
    for robot, pin in pins.items():
        text = (robots / robot / "env_server.py").read_text()
        assert f"{pin}(" in text, robot
        assert 'os.environ.pop("CUDA_VISIBLE_DEVICES"' not in text, robot
    # RoboLab's real2sim generator starts the same Isaac app.
    real2sim = (robots / "robolab" / "real2sim.py").read_text()
    assert "pin_isaac(" in real2sim and '"CUDA_VISIBLE_DEVICES", None' not in real2sim
