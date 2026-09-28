"""env.align_wrist (utils/wrist_alignment.py): the correction, its clamp, execute, the registry entry."""

from __future__ import annotations

import numpy as np
import pytest

from pi_embodied_services.components.manifest import (
    _signature_mismatch,
    code_primitives,
    load_manifest,
)
from pi_embodied_services.utils.wrist_alignment import WristAligner, alignment

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
# Looking straight down from 0.2 m above the gripper centre: camera x = world x, y = -world y.
DOWN = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0.5], [0, 0, 0, 1]])


def view(depth_m: float = 0.4) -> dict:
    return {
        "rgb": np.zeros((480, 640, 3), np.uint8),
        "depth": np.full((480, 640), depth_m, np.float32),
        "intrinsic_K": K,
        "extrinsic_cam2world": DOWN,
    }


def test_the_correction_puts_the_target_on_the_gripper_pixel_and_is_clamped():
    a = alignment(K, DOWN, np.array([0.01, 0.02, 0.1]), np.array([0, 0, 0.3]), 0.03)
    assert a["desired_pixel"] == [240, 320]
    assert a["target_pixel"] == [215, 333]
    assert np.allclose(a["delta_world"], [0.01, 0.02, 0], atol=1e-5)
    assert not a["clamped"]
    far = alignment(K, DOWN, np.array([0.3, 0.4, 0.1]), np.array([0, 0, 0.3]), 0.03)
    assert far["clamped"] and np.isclose(
        np.linalg.norm(far["delta_world"]), 0.03, atol=1e-4
    )
    with pytest.raises(ValueError, match="gripper centre"):
        alignment(K, DOWN, np.array([0, 0, 0.1]), np.array([0, 0, 0.6]), 0.03)


def test_align_wrist_reports_by_default_and_moves_only_with_execute():
    moves: list = []
    aligner = WristAligner(
        view,
        lambda: [0, 0, 0.3],
        lambda xyz, delta: moves.append((xyz.tolist(), delta.tolist())) or {"ok": True},
        move_with="env.move_to to aligned_xyz",
    )
    out = aligner.align_wrist(215, 333)
    assert out["executed"] is False and "moved" not in out and moves == []
    assert np.allclose(out["target_world"], [0.01, 0.02, 0.1], atol=1e-3)
    assert np.allclose(out["aligned_xyz"], [0.01, 0.02, 0.3], atol=1e-3)
    out = aligner.align_wrist(215, 333, execute=True, max_correction_m=0.5)
    assert out["moved"] == {"ok": True} and out["max_correction_m"] == 0.05
    assert np.allclose(moves[0][0], [0.01, 0.02, 0.3], atol=1e-3)
    with pytest.raises(ValueError, match="no depth"):
        WristAligner(
            lambda: view(0.0), lambda: [0, 0, 0.3], lambda *_: {}, move_with="x"
        ).align_wrist(1, 1)
    with pytest.raises(ValueError, match="out of bounds"):
        aligner.align_wrist(480, 0)


@pytest.mark.parametrize("robot", ["libero", "franka"])
def test_the_manifest_declares_align_wrist_and_the_method_matches_it(robot):
    aligner = WristAligner(view, lambda: [0, 0, 0.3], lambda *_: {}, move_with="x")

    class Facade:
        def __init__(self) -> None:
            self._rpc: dict = {}

    f = Facade()
    aligner.install(f)
    manifest = load_manifest(robot)
    (entry,) = [e for e in manifest["primitives"] if e["name"] == "align_wrist"]
    assert entry["method"] == "env.align_wrist" and entry["requires"] == ["align_wrist"]
    # Without --align-wrist the program never sees it; with it, its code signature is the method's.
    assert "align_wrist" not in [
        p.name for p in code_primitives(manifest, lambda c: False)
    ]
    (prim,) = [
        p
        for p in code_primitives(manifest, lambda c: c == "align_wrist")
        if p.name == "align_wrist"
    ]
    assert (
        set(prim.params) == {"row", "col", "max_correction_m", "execute"}
        and prim.mutating
    )
    assert (
        _signature_mismatch(
            f._rpc["env.align_wrist"],
            {
                k: p
                for k, p in entry["params"].items()
                if "code" in p.get("modes", ["tool", "code"])
            },
        )
        == ""
    )
