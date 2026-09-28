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


def test_the_manifest_declares_the_motions_mutating_and_ground_truth_privileged():
    """manifests/genesis.json: every motion is mutating, ground truth is the privileged tier
    alone, and the joint-space primitives are code-only (low)."""
    from pi_embodied_services.components.manifest import load_manifest

    m = {e["name"]: e for e in load_manifest("genesis")["primitives"]}
    mutating = {n for n, e in m.items() if e.get("mutating") and e["side"] != "ts"}
    assert mutating == {
        "move_delta", "set_gripper", "execute_grasp", "execute_place", "move_to_joints",
        "move_along_trajectory", "step", "chunk_step",
    }  # fmt: skip
    assert [n for n, e in m.items() if e["tier"] == "privileged"] == [
        "ground_truth_poses"
    ]
    assert {n for n, e in m.items() if e["tier"] == "raw"} == {
        "step",
        "chunk_step",
        "act",
        "plan",
    }
    for n in ("solve_ik", "move_to_joints", "traj_plan", "move_along_trajectory"):
        assert m[n]["side"] == "code" and m[n]["tier"] == "low", n
    assert m["set_gripper"]["params"]["close"]["required"]


def test_task_table_and_limits_are_what_the_robot_describes():
    assert list(g.TASKS) == ["cube_pick"]
    assert g.STEP_M == 0.02 and g.MAX_MOVE_M == 0.2 and g.EMPTY_WIDTH_M == 0.005
    assert g.CUBE_X[0] >= g.WORKSPACE["min"][0] and g.CUBE_X[1] <= g.WORKSPACE["max"][0]
    assert g.CUBE_Y[0] >= g.WORKSPACE["min"][1] and g.CUBE_Y[1] <= g.WORKSPACE["max"][1]


def test_visible_pixels_counts_the_cubes_segmentation_index_not_its_entity_idx():
    """Genesis's segmentation image holds the renderer's index of each entity (0 background,
    then 1, 2, ... in registration order: plane, robot, cube), not entity.idx; the visibility
    check must look the cube up through scene.segmentation_idx_dict."""
    from types import SimpleNamespace

    seg_dict = {
        0: -1,
        1: 0,
        2: 1,
        3: 2,
    }  # plane idx 0 -> 1, robot idx 1 -> 2, cube idx 2 -> 3
    assert g.segmentation_index(seg_dict, 2) == 3
    assert g.segmentation_index(seg_dict, 0) == 1
    assert g.segmentation_index(seg_dict, 7) is None
    assert g.segmentation_index({0: -1, 1: (2, 0, 0), 2: (2, 0, 1)}, 2) == 1, (
        "geom level"
    )
    seg = np.zeros((8, 8), dtype=np.int32)
    seg[0:4, :] = 2  # the robot, 32 px
    seg[6:8, 6:8] = 3  # the cube, 4 px
    f = object.__new__(g.GenesisEnvFacade)
    f._cams = {
        "agentview": SimpleNamespace(render=lambda **kw: (None, None, seg, None))
    }
    f._cube = SimpleNamespace(idx=2)
    f._scene = SimpleNamespace(segmentation_idx_dict=seg_dict)
    f._meta = {}
    assert f.visible_pixels() == {"cube": 4}
    with pytest.raises(RuntimeError, match="not visible"):
        f.check_visible()
    assert f._meta["visible_px"] == {"cube": 4}
    f._cube = SimpleNamespace(idx=9)
    with pytest.raises(RuntimeError, match="no segmentation index"):
        f.visible_pixels()


def _moving_facade():
    """A facade whose TCP moves half its gap to the commanded point each control step (no
    Genesis): the IK, the scene and the renders are stand-ins."""
    from types import SimpleNamespace

    f = object.__new__(g.GenesisEnvFacade)
    g.BaseEnvFacade.__init__(f)
    tcp = np.array([0.5, 0.0, 0.3])
    f._torch = SimpleNamespace(
        as_tensor=lambda x, dtype=None: x, tensor=lambda x, dtype=None: x, float32=None
    )
    f._robot = SimpleNamespace(
        inverse_kinematics=lambda **kw: np.zeros(9),
        control_dofs_position=lambda q, dofs: None,
        get_dofs_position=lambda: np.full(9, 0.02),
    )
    f._cube = SimpleNamespace(get_pos=lambda: np.array([0.6, 0.0, 0.02]))

    def step():
        if f._cmd_tcp is not None:
            tcp[:] = tcp + 0.5 * (f._cmd_tcp - tcp)

    f._scene = SimpleNamespace(step=step)
    f._tcp = lambda: tcp.copy()
    f._obs = lambda: {"tcp_pos": tcp.astype(np.float32), "success": f._success}
    f._hand, f._hold_quat = None, np.array([0.0, 1.0, 0.0, 0.0])
    f._cmd_tcp, f._record, f._offset = None, None, np.zeros(3)
    f._gripper_open, f._success, f._hold, f._steps = True, False, 0, 0
    f._rule = "lift"
    return f


def test_a_recorded_motion_returns_every_control_step_as_env_step_takes_it():
    f = _moving_facade()
    start = f._tcp()
    out = f.move_delta([0.02, 0, 0], record=True)
    steps = out["steps"]
    assert len(steps) == out["control_steps"] > 0
    first = steps[0]["action"]
    assert first.dtype == np.float32 and first.shape == (4,)
    # The first servo step commands the waypoint plus half its error (the offset integrator).
    assert first.tolist() == pytest.approx([0.03, 0, 0, 1.0])
    assert steps[0]["tcp_pos"][0] == pytest.approx(start[0] + 0.015)
    assert f._record is None, "recording ends with the call"
    # A gripper change holds the commanded point: each step's action is its remaining gap.
    out = f.set_gripper(True, record=True)
    actions = np.stack([s["action"] for s in out["steps"]])
    assert len(actions) == g.GRIPPER_STEPS and (actions[:, 3] == -1).all()
    gaps = np.linalg.norm(actions[:, :3], axis=1)
    assert (np.diff(gaps) <= 1e-9).all()
    # Without record nothing is kept.
    assert "steps" not in f.move_delta([0, 0.02, 0])


def test_move_to_joints_servos_the_arm_and_stops_outside_the_workspace():
    """The joint servo steps the scene until the joints are within tol; refused outside the
    Panda's limits; stopped (and holding) when the TCP leaves the workspace box."""
    f = _moving_facade()
    q = np.full(9, 0.02)

    def control(target, dofs):
        if list(dofs) == g.MOTOR_DOFS:
            q[:7] = q[:7] + 0.5 * (np.asarray(target) - q[:7])

    f._robot.control_dofs_position = control
    f._robot.get_dofs_position = lambda: q.copy()
    f._render = lambda name, depth=False: np.zeros((4, 4, 3), np.uint8)
    target = [0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8]
    out = f.move_to_joints(target, tol_rad=0.01)
    assert (
        out["joint_error_rad"] < 0.01 and 0 < out["control_steps"] < g.JOINT_MAX_STEPS
    )
    assert "error" not in out and f._offset.tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="limits"):
        f.move_to_joints([0.0, -0.4, 0.0, 0.5, 0.0, 2.0, 0.8])
    with pytest.raises(ValueError, match="7 finite"):
        f.move_to_joints([0.0] * 6)
    f._tcp = lambda: np.array([0.5, 0.0, 0.0])  # below the Z floor
    out = f.move_to_joints([0.1, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8])
    assert out["control_steps"] == 1 and "workspace" in out["error"]
    # A trajectory runs its waypoints with CaP-X's per-waypoint budget and stops at the error.
    out = f.move_along_trajectory([target, target])
    assert out["control_steps"] == 1 and out["waypoints"] == 2 and "error" in out


def test_solve_ik_and_traj_plan_ask_genesis_ik_for_the_tcp_and_move_nothing():
    """solve_ik: 7 joints for a TCP pose (default: the held orientation); traj_plan: IK waypoints
    along the straight line in 2 cm steps, each seeded with the previous solution."""
    f = _moving_facade()
    asked: list = []

    def ik(**kw):
        asked.append(kw)
        return np.arange(9, dtype=np.float64) * float(kw["pos"][2])

    f._robot.inverse_kinematics = ik
    q = f.solve_ik([0.5, 0.0, 0.1])
    assert q == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert asked[0]["quat"].tolist() == f._hold_quat.tolist()
    assert asked[0]["local_point"] == [0.0, 0.0, g.TCP_OFFSET_M]
    start = [0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.3]
    traj = f.traj_plan(start, [0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.2])
    assert len(traj) == 6 and all(len(p) == 7 for p in traj)
    assert "init_qpos" in asked[-1], "seeded with the previous waypoint"
    assert f._steps == 0, "nothing moved"
