"""Genesis env server helpers that need no simulator: the motion limits, the success and
grasp predicates, the waypoint split, the camera conventions and the primitive registry."""

import numpy as np
import pytest

from pi_embodied_services.robots.genesis import env_server as g


def test_check_target_refuses_before_anything_moves():
    start = [0.5, 0.0, 0.3]
    assert np.allclose(g.check_target(start, [0.1, -0.1, -0.1]), [0.6, -0.1, 0.2])
    with pytest.raises(ValueError, match="limit is 0.2 m per call"):
        g.check_target(start, [0.25, 0, 0])
    with pytest.raises(ValueError, match="below the floor"):
        g.check_target([0.5, 0.0, 0.1], [0, 0, -0.1])
    with pytest.raises(ValueError, match="leaves the workspace box"):
        g.check_target([0.8, 0.0, 0.3], [0.1, 0, 0])
    with pytest.raises(ValueError, match="leaves the workspace box"):
        g.check_target([0.5, 0.35, 0.3], [0, 0.1, 0])
    # The floor is the tighter bound below, the box everywhere else.
    assert g.Z_FLOOR_M == g.WORKSPACE["min"][2]


def test_waypoints_are_two_centimetre_decisions():
    w = g.waypoints([0, 0, 0.2], [0.05, 0, 0.17])
    assert len(w) == 3  # 5.8 cm -> 3 decisions
    assert np.allclose(w[-1], [0.05, 0, 0.17])
    assert np.allclose(w[0], [0.05 / 3, 0, 0.19])
    assert len(g.waypoints([0, 0, 0.2], [0, 0, 0.2])) == 1


def test_success_is_a_lift_and_a_grasp_needs_both_fingers():
    half = g.CUBE_SIZE_M / 2
    assert not g.lifted(half)  # resting on the table
    assert not g.lifted(half + g.LIFT_M - 0.001)
    assert g.lifted(half + g.LIFT_M)
    fingers = (11, 12)
    both = {"link_a": np.array([11, 3]), "link_b": np.array([20, 12])}
    assert g.grasped(both, fingers, width=0.04)
    assert not g.grasped(both, fingers, width=2 * g.FINGER_OPEN_M), "fully open"
    one = {"link_a": np.array([11]), "link_b": np.array([20])}
    assert not g.grasped(one, fingers, width=0.04)
    assert not g.grasped({}, fingers, width=0.0)


def test_camera_conventions_round_trip():
    # A Genesis (OpenGL) camera at (1, 0, 1) looking along -x: its cam2world in OpenCV terms
    # keeps the position and flips the y/z axes; a pixel on the optical axis back-projects
    # to the point `depth` ahead of the camera.
    forward, up = np.array([-1.0, 0, 0]), np.array([0, 0, 1.0])
    right = np.cross(forward, up)
    t_gl = np.eye(4)
    t_gl[:3, 0], t_gl[:3, 1], t_gl[:3, 2], t_gl[:3, 3] = right, up, -forward, [1, 0, 1]
    c2w = g.cam2world_cv(t_gl)
    assert np.allclose(c2w[:3, 2], forward) and np.allclose(c2w[:3, 1], -up)
    K = np.array([[128.0, 0, 128.0], [0, 128.0, 128.0], [0, 0, 1]])
    depth = np.full((256, 256), 0.5, dtype=np.float32)
    depth[0, 0] = 0.0  # background
    pts = g.back_project(depth, K, c2w, [[128, 128], [0, 0], [300, 5]])
    assert np.allclose(pts[0], [0.5, 0, 1])
    assert pts[1] is None and pts[2] is None
    # letterbox: a square image passes through untouched; a 4:3 one gets bars.
    img = np.full((48, 64, 3), 200, dtype=np.uint8)
    out = g.letterbox(img, 64)
    assert out.shape == (64, 64, 3) and out[0].max() == 0 and out[32].min() == 200
    assert g.letterbox(out, 64) is not None and (g.letterbox(out, 64) == out).all()


def test_primitive_registry_names_only_registered_methods():
    """The Genesis registry (primitives.py) resolves against the facade's RPC table: every
    primitive's method is registered, the motion primitives are mutating, and ground truth is
    the privileged tier alone."""
    from pi_embodied_services.components.code_api import CodeApi
    from pi_embodied_services.robots.genesis.primitives import GENESIS_PRIMITIVES

    methods = {
        "env.get_task_language", "env.state", "env.move_delta", "env.set_gripper",
        "env.back_project", "env.render_camera", "env.get_camera_meta", "env.step",
        "env.chunk_step", "env.ground_truth_poses",
    }  # fmt: skip
    api = CodeApi(GENESIS_PRIMITIVES, dict.fromkeys(methods))
    high = [p.name for p in api.primitives("high")]
    assert high == [
        "get_task_language",
        "state",
        "move_delta",
        "set_gripper",
        "back_project",
    ]
    assert [p.name for p in api.primitives("privileged")] == [
        *high,
        "ground_truth_poses",
    ]
    assert "step" in [p.name for p in api.primitives("low")]
    assert {p.name for p in GENESIS_PRIMITIVES if p.mutating} == {
        "move_delta", "set_gripper", "step", "chunk_step",
    }  # fmt: skip
    method, kwargs = api.resolve(
        "move_delta", {"delta_xyz": [0, 0, 0.05], "gripper": "open"}
    )
    assert method == "env.move_delta"
    with pytest.raises(ValueError, match="unknown parameter"):
        api.resolve("move_delta", {"delta_xyz": [0, 0, 0], "yaw": 1})
    # A registry whose method the facade lacks is refused at construction.
    with pytest.raises(ValueError, match="not registered"):
        CodeApi(GENESIS_PRIMITIVES, {"env.state": None})


def test_task_table_and_limits_are_what_the_robot_describes():
    assert list(g.TASKS) == ["cube_pick"]
    assert g.STEP_M == 0.02 and g.MAX_MOVE_M == 0.2 and g.EMPTY_WIDTH_M == 0.005
    assert g.CUBE_X[0] >= g.WORKSPACE["min"][0] and g.CUBE_X[1] <= g.WORKSPACE["max"][0]
    assert g.CUBE_Y[0] >= g.WORKSPACE["min"][1] and g.CUBE_Y[1] <= g.WORKSPACE["max"][1]
