"""RoboLab glue that runs without Isaac Sim: quaternion order across Isaac Lab 2.x/3.x, the
orientation hold, tilt, and the subtask read-out."""

from types import SimpleNamespace

import numpy as np
import pytest

from pi_embodied_services.robots.robolab import sim


@pytest.mark.parametrize("xyzw", [False, True])
def test_lab_quat_and_ee_quat_follow_the_installed_isaaclab_order(monkeypatch, xyzw):
    monkeypatch.setattr(sim, "isaaclab_xyzw", lambda: xyzw)
    wxyz = (0.9, 0.1, 0.2, 0.3)
    lab = sim.lab_quat(wxyz)
    assert lab == ((0.1, 0.2, 0.3, 0.9) if xyzw else wxyz)
    # rl_ee_quat reads the hand body's quaternion in Isaac Lab's order and always returns wxyz.
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_names=["panda_link0", "panda_hand"],
            body_quat_w=np.array([[[1.0, 0, 0, 0], list(lab)]]),
        )
    )
    env = SimpleNamespace(scene={"robot": robot})
    assert np.allclose(sim.rl_ee_quat(env), wxyz)


def test_to_np_unwraps_a_proxy_array():
    class Proxy:  # Isaac Lab 3 ProxyArray: explicit .torch
        torch = np.arange(3.0)

    assert np.array_equal(sim.to_np(Proxy()), np.arange(3.0))


def test_tilt_and_orientation_hold():
    down = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x: +Z points down
    assert sim.ee_tilt_deg(down) == pytest.approx(0.0, abs=1e-3)
    assert np.allclose(sim.hold_orientation_rotvec(down, down), 0.0)
    a = 0.1  # 0.1 rad off about world x
    off = np.array([np.cos((np.pi + a) / 2), np.sin((np.pi + a) / 2), 0.0, 0.0])
    rv = sim.hold_orientation_rotvec(down, off)
    assert np.allclose(rv, [-a, 0.0, 0.0], atol=1e-9)
    big = np.array([np.cos((np.pi + 1.0) / 2), np.sin((np.pi + 1.0) / 2), 0.0, 0.0])
    assert np.linalg.norm(sim.hold_orientation_rotvec(down, big)) == pytest.approx(
        sim.ORIENT_HOLD_MAX_RAD
    )
