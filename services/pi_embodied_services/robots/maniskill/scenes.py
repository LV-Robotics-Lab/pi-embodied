# Copyright 2026 The Show-Harness Authors.
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
#
# Modified by pi-embodied: core/sim/maniskill_scenes.py (the RLinf real2sim rows and the
# generic registrar, verbatim in behaviour) and the layout sampler of
# scripts/trajectory/real2sim/maniskill/tasks.py of github.com/showlab/Show-Harness @137d571,
# with the per-scene camera contract of configs/robot_maniskill.yaml and the reset sequence of
# their real2sim generators (record_demos.py / follow_tokenize.py) folded into one table.

"""The RLinf real2sim rigs Show-Harness's sim adapter was trained on.

``BlockPAP-v1`` / ``BlockStack-v1`` are RLinf's 1:1 replicas of the real Franka rig (table,
pedestal, ground, lighting, block + coaster, and the calibrated front RealSense as
``external_cam``). Their env modules live in ``real_franka/real2sim_env/`` of an RLinf
checkout (``RLINF_ROOT``; ``fetch_real2sim.sh`` gets the public fork that carries them):
importing one registers the gym id and its robot agent. That agent has no wrist camera, so
:func:`register_scene` derives ``<Base>WristCam`` = the rig's agent + Show-Harness's centred
``hand_camera`` -- the same mount, resolution and FOV on every scene (a training contract).

Deployment contract per scene (configs/robot_maniskill.yaml, the generators' reset):

* agentview = ``external_cam`` raw (640x480, the RealSense K), wrist = ``hand_camera`` with
  ``flip: both`` (a 180 deg turn: fingertips at the TOP, wrist left == agentview left); the
  model-side letterbox to 256 (and the wrist's 4:3 crop) is the client's job.
* reset = seed the GLOBAL ``np.random`` (the rig samples its own layout from it), reset with
  the seed, hold 8 steps open, then (``layout: wide``) re-sample both objects over the
  reachable box with ``default_rng(seed)`` anchored at the settled gripper XY and hold 6 --
  exactly how the ms_0717 training episodes (Show-Harness-Data ``sim/`` rollout_130..229,
  seeds 20000..20099) were produced, so a seed here reproduces their first frame.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import numpy as np

RLINF_ROOT = os.environ.get("RLINF_ROOT", "/root/autodl-tmp/assets/RLinf-real2sim")

# Wrist camera placement, relative to the gripper: the D415 rig's ORIENTATION
# (realsense_joint rpy = 0, -1.5707, 3.1415 in panda_v3.urdf), mounted on ``panda_hand``
# without the stock rig's 2 cm lateral ``camera_link`` hop, which throws the finger pair
# 115 px off centre (Show-Harness measured -1.0 px centred). Only correct with flip=both.
_Q_D415 = [0.0, 0.7071068, 0.0, 0.7071068]  # wxyz, = rpy(0, -1.5707, 3.1415)
WRIST_MOUNTS: dict[str, tuple[str, list[float], list[float]]] = {
    "camera_link": ("camera_link", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
    "centered": ("panda_hand", [0.035, 0.0, 0.036], _Q_D415),
}

COMMON_DEFAULTS: dict[str, Any] = {
    "wrist_resolution": 256,
    "wrist_mount": "centered",
    "layout": "wide",
}

#: Reachable sampling box on the RLinf rig, robot-base world frame (tasks.WIDE_BOX): the
#: training layout distribution. The table spans X 0.20-0.80.
WIDE_BOX = {"x": (0.40, 0.62), "y": (-0.22, 0.22), "min_sep": 0.12}
_SAMPLE_ATTEMPTS = 400
#: Coaster orientation in BlockPAP (flat-lying cylinder, wxyz).
_COASTER_Q = [float(np.cos(np.pi / 4)), 0.0, float(np.sin(np.pi / 4)), 0.0]


@dataclass(frozen=True)
class SceneSpec:
    """One RLinf rig: how to register it, what it means, and how to lay it out.

    ``scene_globals`` maps a MODULE-LEVEL global of ``module`` (read inside ``_load_scene`` /
    ``_initialize_episode``, so set before construction) to the option key that sets it,
    optionally as ``(option_key, coerce)``; ``make_kwargs`` maps a ``gym.make`` keyword to an
    option key. ``urdf_path`` (relative to ManiSkill's ``PACKAGE_ASSET_DIR``) overrides the
    base agent's URDF on the derived wrist-cam agent. ``carried`` / ``target`` are the two
    actors (the layout sampler moves both; the agentview visibility check wants both);
    ``snap`` quantises the layout onto the 2 cm lattice (tasks.TASKS ``snap_layout``).
    """

    env_id: str
    instruction: str
    robot_uids: str
    module: str
    agent_base: str
    carried: str
    target: str
    target_z: Callable[[Any], float]
    target_q: Optional[list[float]] = None
    snap: bool = False
    urdf_path: Optional[str] = None
    scene_globals: Mapping[str, Any] = field(default_factory=dict)
    make_kwargs: Mapping[str, str] = field(default_factory=dict)
    defaults: Mapping[str, Any] = field(default_factory=dict)


def _block_z(env) -> float:
    return float(env.TABLE_Z + env.BLOCK_HALF_SIZE[2])


def _coaster_z(env) -> float:
    return float(env.TABLE_Z + env.COASTER_THICKNESS)


SCENES: dict[str, SceneSpec] = {
    "BlockPAP-v1": SceneSpec(
        env_id="BlockPAP-v1",
        # The block renders orange -- "red" would describe something the model does not see.
        instruction="pick up the orange block and place it on the coaster",
        robot_uids="panda_high_friction_wristcam",
        module="real_franka.real2sim_env.pick_and_place",
        # panda_high_friction inherits Panda's panda_v2.urdf (no camera_link); panda_v3 has
        # one, and the agent's own customisation (pad friction, ee_pose_at_robot_base)
        # survives subclassing.
        agent_base="PandaHighFriction",
        urdf_path="robots/panda/panda_v3.urdf",
        carried="cube",
        target="target",
        target_z=_coaster_z,
        target_q=_COASTER_Q,
        scene_globals={
            "TABLE_TEX_ID": "table_tex",
            "TRAJ_ID": ("traj_id", str),
            "CAM_JITTER_RAD": "cam_jitter",
        },
        make_kwargs={"cam_t": "cam_t"},
        # table_tex 006 (wood) and traj_id random are what the training data used
        # (generator defaults + the blockpap_follow_wood shard); Show-Harness's eval yaml
        # says white / "0" but its README prefers the wood for wrist-dependent policies.
        defaults={
            "table_tex": "006",
            "traj_id": "random",
            "cam_jitter": None,
            "cam_t": "og",
        },
    ),
    "BlockStack-v1": SceneSpec(
        env_id="BlockStack-v1",
        instruction="pick up the white block and stack it on the gray block",
        robot_uids="panda_extended_gripper_wristcam",
        module="real_franka.real2sim_env.block_stack",
        # panda_v2_extended.urdf (2 cm longer fingers) must stay: the centred mount hangs off
        # panda_hand, which every panda variant has. The URDF ships only with RLinf's own
        # ManiSkill build; write_extended_urdf() rebuilds it from its spec.
        agent_base="PandaExtendedGripper",
        carried="white_block",
        target="gray_block",
        target_z=_block_z,
        snap=True,  # 1 cm success tolerance vs a 2 cm step
        scene_globals={"TRAJ_ID": ("traj_id", str)},
        defaults={"traj_id": "random"},
    ),
}

_REGISTERED_AGENTS: set[str] = set()


def scene_options(env_id: str, **overrides: Any) -> dict[str, Any]:
    """COMMON_DEFAULTS <- the row's defaults <- caller overrides (None = keep)."""
    merged = dict(COMMON_DEFAULTS)
    merged.update(SCENES[env_id].defaults)
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def register_scene(env_id: str, **overrides: Any) -> dict[str, Any]:
    """Import the rig (registers the gym id), set its scene globals, derive the wrist-cam
    agent, and return the extra ``gym.make`` kwargs (``robot_uids`` included). Idempotent."""
    spec = SCENES[env_id]
    opts = scene_options(env_id, **overrides)
    if not os.path.isdir(os.path.join(RLINF_ROOT, "real_franka", "real2sim_env")):
        raise RuntimeError(
            f"{env_id} needs real_franka/real2sim_env under RLINF_ROOT={RLINF_ROOT!r}; run "
            "services/pi_embodied_services/robots/maniskill/fetch_real2sim.sh"
        )
    if RLINF_ROOT not in sys.path:
        sys.path.insert(0, RLINF_ROOT)
    module = importlib.import_module(spec.module)
    for global_name, source in spec.scene_globals.items():
        key, coerce = source if isinstance(source, tuple) else (source, lambda v: v)
        if key in opts:
            setattr(
                module, global_name, None if opts[key] is None else coerce(opts[key])
            )
    _register_wristcam_agent(spec, module, opts)
    kwargs = {k: opts[o] for k, o in spec.make_kwargs.items() if o in opts}
    return {"robot_uids": spec.robot_uids, **kwargs}


def _register_wristcam_agent(
    spec: SceneSpec, module: Any, opts: dict[str, Any]
) -> None:
    """``<agent_base>`` + one ``hand_camera``, registered under ``spec.robot_uids``.

    Subclassing keeps the rig's customisations (friction pads, extended fingers, the
    ``ee_pose_at_robot_base`` the RLinf envs' obs plumbing calls)."""
    if spec.robot_uids in _REGISTERED_AGENTS:
        return
    import sapien
    from mani_skill import PACKAGE_ASSET_DIR
    from mani_skill.agents.registration import register_agent
    from mani_skill.sensors.camera import CameraConfig

    mount_link, mount_p, mount_q = WRIST_MOUNTS[str(opts["wrist_mount"])]
    resolution = int(opts["wrist_resolution"])
    namespace: dict[str, Any] = {
        "uid": spec.robot_uids,
        "_sensor_configs": property(
            lambda self: [
                CameraConfig(
                    uid="hand_camera",
                    pose=sapien.Pose(p=mount_p, q=mount_q),
                    width=resolution,
                    height=resolution,
                    fov=np.pi / 2,
                    near=0.01,
                    far=100,
                    mount=self.robot.links_map[mount_link],
                )
            ]
        ),
    }
    if spec.urdf_path:
        namespace["urdf_path"] = f"{PACKAGE_ASSET_DIR}/{spec.urdf_path}"
    base = getattr(module, spec.agent_base)
    urdf = namespace.get("urdf_path", base.urdf_path)
    if not os.path.isfile(urdf):
        raise RuntimeError(
            f"{spec.env_id}: robot URDF {urdf} is not installed; run fetch_real2sim.sh "
            "(it writes panda_v2_extended.urdf with write_extended_urdf)"
        )
    register_agent()(type(f"{spec.agent_base}WristCam", (base,), namespace))
    _REGISTERED_AGENTS.add(spec.robot_uids)


# -- layout (scripts/trajectory/real2sim/maniskill/tasks.py) ------------------


def sample_layout(
    env, spec: SceneSpec, rng: np.random.Generator, anchor_xy, step_m: float = 0.02
) -> dict[str, dict]:
    """Absolute poses for the two actors, uniform in WIDE_BOX, >= min_sep apart; snapped
    onto the ``step_m`` lattice anchored at ``anchor_xy`` when the scene needs it."""

    def q(x: float, y: float) -> tuple[float, float]:
        if not spec.snap:
            return float(x), float(y)
        return (
            float(anchor_xy[0] + round((x - anchor_xy[0]) / step_m) * step_m),
            float(anchor_xy[1] + round((y - anchor_xy[1]) / step_m) * step_m),
        )

    ax = ay = bx = by = 0.0
    for _ in range(_SAMPLE_ATTEMPTS):
        ax, ay = q(rng.uniform(*WIDE_BOX["x"]), rng.uniform(*WIDE_BOX["y"]))
        bx, by = q(rng.uniform(*WIDE_BOX["x"]), rng.uniform(*WIDE_BOX["y"]))
        if float(np.hypot(ax - bx, ay - by)) >= WIDE_BOX["min_sep"]:
            break
    target = {"p": [bx, by, spec.target_z(env)]}
    if spec.target_q is not None:
        target["q"] = list(spec.target_q)
    return {spec.carried: {"p": [ax, ay, _block_z(env)]}, spec.target: target}


def apply_layout(env, layout: dict[str, dict]) -> None:
    import sapien

    for name, pose in layout.items():
        p = [float(v) for v in pose["p"]]
        q = pose.get("q")
        getattr(env, name).set_pose(
            sapien.Pose(p=p, q=[float(v) for v in q]) if q else sapien.Pose(p=p)
        )


# -- the deployment contract (configs/robot_maniskill.yaml, the generators' reset) ------

#: The two views the policy sees, by sensor uid.
CAMERAS = {"agentview": "external_cam", "wrist": "hand_camera"}
#: core/record/images.prepare_view arguments per view (square_size is the caller's):
#: agentview raw; wrist turned 180 deg (fingertips top, left == agentview left), then
#: cropped to 4:3 so it carries the real rigs' 32 black rows after the 256 letterbox.
VIEWS = {
    "agentview": {"rotation": 0, "flip": "none", "crop": None},
    "wrist": {"rotation": 0, "flip": "both", "crop": 1.3333},
}
#: Open-gripper hold steps after the reset (num_steps_wait) and after the layout.
SETTLE_STEPS = 8
LAYOUT_SETTLE_STEPS = 6


def make_env(env_id: str, options: dict[str, Any], **gym_kwargs: Any):
    """Register the rig with ``options`` and ``gym.make`` it (its own calibrated cameras:
    no sensor overrides, which would corrupt the external_cam intrinsics)."""
    import gymnasium as gym

    return gym.make(
        env_id, num_envs=1, **register_scene(env_id, **options), **gym_kwargs
    )


def reset(env, env_id: str, options: dict[str, Any], seed: int, step) -> dict:
    """The training episodes' reset: global ``np.random`` seeded (the rig's own sampler draws
    from it), ``env.reset(seed)``, SETTLE_STEPS open holds, then for ``layout: wide`` both
    objects re-sampled with ``default_rng(seed)`` anchored at the settled TCP XY and
    LAYOUT_SETTLE_STEPS more holds. ``step()`` performs one open hold and returns the obs.
    Returns the layout (empty when the rig kept its own)."""
    np.random.seed(int(seed))
    env.reset(seed=int(seed))
    for _ in range(SETTLE_STEPS):
        step()
    if options.get("layout") != "wide":
        return {}
    u = env.unwrapped
    tcp = u.agent.tcp.pose.p.reshape(-1, 3)[0].cpu().numpy()
    layout = sample_layout(u, SCENES[env_id], np.random.default_rng(int(seed)), tcp[:2])
    apply_layout(u, layout)
    for _ in range(LAYOUT_SETTLE_STEPS):
        step()
    return layout


def prepare_view(
    image: np.ndarray, rotation: int, flip: str, crop: Optional[float], square: int
) -> np.ndarray:
    """core/record/images.prepare_view: rotate/flip -> centre-crop to ``crop`` (w/h) ->
    letterbox into ``square`` with the real rigs' resize_with_pad (cv2 INTER_LINEAR), byte
    for byte (checked against Show-Harness-Data's stored frames)."""
    import cv2

    k = (int(rotation) % 360) // 90
    img = np.rot90(image, k=k) if k else image
    if flip in ("vertical", "both"):
        img = img[::-1]
    if flip in ("horizontal", "both"):
        img = img[:, ::-1]
    h, w = img.shape[:2]
    if crop and abs(w / h - crop) >= 1e-6:
        if w / h > crop:
            nw = int(round(h * crop))
            img = img[:, (w - nw) // 2 : (w - nw) // 2 + nw]
        else:
            nh = int(round(w / crop))
            img = img[(h - nh) // 2 : (h - nh) // 2 + nh]
    img = np.ascontiguousarray(img)
    if not square:
        return img
    h, w = img.shape[:2]
    scale = min(square / w, square / h)
    nw, nh = int(w * scale), int(h * scale)
    out = np.zeros((square, square, 3), dtype=np.uint8)
    y0, x0 = (square - nh) // 2, (square - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_LINEAR
    )
    return out


# -- BlockStack's gripper -------------------------------------------------------------

#: The extra fingertip box of panda_v2_extended.urdf, per block_stack.PandaExtendedGripper's
#: spec: "panda_hand_tcp_joint xyz = 0 0 0.1234 (+20 mm vs. stock 0.1034); each finger has an
#: extra 20 mm collision box appended beyond the rubber tip (center at z = 64.5 mm from finger
#: joint origin)". The box keeps the rubber tip's cross-section and lateral offset (its
#: 18.5 mm tip spans z 36-54.5 mm, the extension 54.5-74.5 mm); collision only, as specified.
_TIP_BOX = '<box size="17.5e-3 15.2e-3 18.5e-3"/>'
_EXT = (
    "    <!-- 20 mm fingertip extension (panda_v2_extended) -->\n"
    '    <collision>\n      <origin rpy="0 0 0" xyz="0 {y} 64.5e-3"/>\n'
    '      <geometry>\n        <box size="17.5e-3 15.2e-3 20e-3"/>\n      </geometry>\n'
    "    </collision>\n"
)


def write_extended_urdf() -> str:
    """Write ``robots/panda/panda_v2_extended.urdf`` next to ManiSkill's panda_v2.urdf (where
    block_stack.py loads it from), derived from panda_v2.urdf by the spec above. Idempotent;
    returns the path."""
    import re

    from mani_skill import PACKAGE_ASSET_DIR

    d = os.path.join(str(PACKAGE_ASSET_DIR), "robots", "panda")
    src = open(os.path.join(d, "panda_v2.urdf")).read()
    out = src.replace(
        '<origin rpy="0 0 0" xyz="0 0 0.1034"/>',
        '<origin rpy="0 0 0" xyz="0 0 0.1234"/>',
    )
    for link, y in (("panda_leftfinger", "7.58e-3"), ("panda_rightfinger", "-7.58e-3")):
        # After the link's rubber-tip collision, before its <inertial>.
        pat = re.compile(
            rf'(<link name="{link}">.*?{re.escape(_TIP_BOX)}\s*</geometry>\s*</collision>\n)',
            re.S,
        )
        out, n = pat.subn(lambda m, y=y: m.group(1) + _EXT.format(y=y), out, count=1)
        assert n == 1, f"panda_v2.urdf: no rubber tip on {link}"
    assert out.count("0.1234") == 1, "panda_v2.urdf: tcp joint not found"
    path = os.path.join(d, "panda_v2_extended.urdf")
    if not os.path.isfile(path) or open(path).read() != out:
        with open(path, "w") as f:
            f.write(out)
    return path


if __name__ == "__main__":
    print(write_extended_urdf())
