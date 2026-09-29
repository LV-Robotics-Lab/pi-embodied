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

"""Pin an env server's CUDA work AND its rendering to one physical GPU.

CUDA honours ``CUDA_VISIBLE_DEVICES``; the renderers do not: EGL (MuJoCo, robosuite, LIBERO,
RoboCasa, Genesis's rasterizer) enumerates every GPU and picks device 0 unless
``MUJOCO_EGL_DEVICE_ID`` / ``EGL_DEVICE_ID`` names one, and Vulkan (Isaac's Kit, SAPIEN) likewise.
A server that sets only one side leaks a context onto GPU 0, which on the shared box belongs to
the planner's vLLM.

The GPU is ``--cuda-device`` when given, else ``PI_EMBODIED_CUDA_DEVICE`` (the deployment's
``cuda_device``, which pi exports), else the first entry of ``CUDA_VISIBLE_DEVICES``; with none
of them nothing is pinned (the process's default, device 0).

- :func:`pin_egl` (MuJoCo / EGL servers): ``CUDA_VISIBLE_DEVICES`` = the GPU (torch then sees it as
  ``cuda:0``) and ``MUJOCO_EGL_DEVICE_ID`` / ``EGL_DEVICE_ID`` = its EGL index. robosuite asserts
  that ``MUJOCO_EGL_DEVICE_ID`` appears in ``CUDA_VISIBLE_DEVICES``; where the EGL order differs
  from the CUDA order that cannot hold, so there ``CUDA_VISIBLE_DEVICES`` is cleared and the
  caller selects the CUDA device itself (the returned ordinal).
- :func:`pin_isaac` (Kit / Isaac Lab, OmniGibson): ``CUDA_VISIBLE_DEVICES`` = the GPU, so CUDA,
  PhysX and Kit's renderer see one device, index 0.
- :func:`pin_cuda` (SAPIEN, Taichi): ``CUDA_VISIBLE_DEVICES`` = the GPU. SAPIEN's Vulkan still
  enumerates every GPU and leaves a ~6 MiB context on each; it renders on the visible one.
"""

from __future__ import annotations

import os

from pi_embodied_services.utils.logging import get_logger

logger = get_logger("gpu")

#: The deployment's GPU (pi exports it from the deployment's ``cuda_device``).
ENV_DEVICE = "PI_EMBODIED_CUDA_DEVICE"


def resolve_gpu(cuda_device: int | None) -> int | None:
    """The physical GPU: the flag, else the deployment's, else the first visible one."""
    if cuda_device is not None:
        return int(cuda_device)
    env = os.environ.get(ENV_DEVICE, "").strip()
    if env.isdigit():
        return int(env)
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(first) if first.isdigit() else None


def pin_cuda(cuda_device: int | None) -> int | None:
    """``CUDA_VISIBLE_DEVICES`` = the resolved GPU; returns it (None: nothing pinned)."""
    gpu = resolve_gpu(cuda_device)
    if gpu is None:
        return None
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    return gpu


def pin_isaac(cuda_device: int | None) -> int | None:
    """Kit / Isaac Lab / OmniGibson: only the GPU visible, so everything runs on its index 0."""
    return pin_cuda(cuda_device)


def _egl_index(gpu: int) -> int:
    """The EGL device index of CUDA ordinal ``gpu`` (the same index when the map is unavailable)."""
    from pi_embodied_services.utils.egl import cuda_to_egl_map

    saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        return cuda_to_egl_map().get(gpu, gpu)
    except (
        Exception
    ) as exc:  # no EGL here (osmesa, a CPU box): the ordinal is the best guess
        logger.warning("EGL device probe failed (%s); using EGL device %d", exc, gpu)
        return gpu
    finally:
        if saved is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved


def pin_egl(cuda_device: int | None, *, probe: bool = True) -> int | None:
    """MuJoCo / EGL servers: CUDA and EGL on the resolved GPU.

    Returns the CUDA ordinal the caller must select (``torch.cuda.set_device``) when the EGL
    order differs from the CUDA order (then ``CUDA_VISIBLE_DEVICES`` is cleared, see the module
    doc), else None (the GPU is the only visible device). An existing ``MUJOCO_EGL_DEVICE_ID``
    is kept."""
    gpu = resolve_gpu(cuda_device)
    if gpu is None:
        return None
    egl = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    if egl is None:
        egl = str(_egl_index(gpu) if probe else gpu)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = egl
    os.environ.setdefault("EGL_DEVICE_ID", egl)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if egl == str(gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return None
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    logger.info(
        "EGL device %s != CUDA device %d: CUDA_VISIBLE_DEVICES cleared", egl, gpu
    )
    return gpu


def add_cuda_argument(parser, help_: str | None = None) -> None:
    """``--cuda-device N`` (physical ordinal); absent: the deployment's, else CUDA_VISIBLE_DEVICES."""
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help=help_
        or "physical GPU for CUDA and rendering (default: PI_EMBODIED_CUDA_DEVICE, else the first "
        "CUDA_VISIBLE_DEVICES entry)",
    )


__all__ = [
    "ENV_DEVICE",
    "add_cuda_argument",
    "pin_cuda",
    "pin_egl",
    "pin_isaac",
    "resolve_gpu",
]
