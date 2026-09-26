"""ManiSkill env server helpers that need no simulator."""

import numpy as np

from pi_embodied_services.robots.maniskill import env_server as ms


class _Actor:
    def __init__(self, name: str, scene_id: int):
        self.name = name
        self.per_scene_id = np.array([scene_id])
        self.hidden = True

    def show_visual(self):
        self.hidden = False


def test_pickcube_goal_is_shown_to_the_cameras_and_required_in_view():
    """PickCube succeeds only with the cube inside ``goal_site``, which ManiSkill keeps in
    ``_hidden_objects`` (never rendered by the sensor cameras): the server shows it and the
    visibility gate requires it, and the task text names it."""
    env = type("Env", (), {})()
    env.cube, env.goal_site, env.other = (
        _Actor("cube", 1),
        _Actor("goal_site", 2),
        _Actor("other", 3),
    )
    env._hidden_objects = [env.goal_site, env.other]
    ms.show_goals(env, ms.SHOW_GOALS["PickCube-v1"])
    assert env._hidden_objects == [env.other]
    assert not env.goal_site.hidden and env.other.hidden
    assert "goal_site" in ms.TASK_ACTORS["PickCube-v1"]
    assert "goal" in ms.INSTRUCTIONS["PickCube-v1"]

    # The gate refuses an episode whose goal is out of the agentview.
    facade = object.__new__(ms.ManiskillEnvFacade)
    facade._meta = {"env_id": "PickCube-v1"}
    # A stock scene, as __init__ sets them.
    facade._cameras, facade._rig = ms.CAMERAS, None
    facade._env = type("Wrapped", (), {"unwrapped": env})()
    seg = np.zeros((1, 48, 64, 1), dtype=np.int32)
    seg[0, :10, :10] = 1  # the cube, 100 px; no goal pixels
    obs = {"sensor_data": {"base_camera": {"segmentation": seg}}}
    try:
        facade.check_visible(obs)
        raise AssertionError("an invisible goal passed the gate")
    except RuntimeError as e:
        assert "goal_site" in str(e)
    seg[0, 20:25, 20:25] = 2
    facade.check_visible(obs)
    assert facade._meta["visible_px"] == {"cube": 100, "goal_site": 25}


#: The task table before OpenETA's tasks were added: these rows must stay byte-identical
#: (the eight ids pi's --env-id accepted, and their texts, actors and shown goals).
_FROZEN = {
    "BlockPAP-v1": None,
    "BlockStack-v1": None,
    "PickCube-v1": (
        "pick up the red cube and move it into the green goal sphere",
        ["cube", "goal_site"],
        ["goal_site"],
    ),
    "StackCube-v1": (
        "stack the red cube on top of the green cube",
        ["cubeA", "cubeB"],
        None,
    ),
    "PushCube-v1": ("push the cube to the goal marker", ["obj", "goal_region"], None),
    "PullCube-v1": ("pull the cube to the goal marker", ["obj", "goal_region"], None),
    "PokeCube-v1": (
        "poke the cube to the goal marker",
        ["cube", "peg", "goal_region"],
        None,
    ),
    "LiftPegUpright-v1": ("lift the peg upright", ["peg"], None),
}
#: OpenETA's ManiSkill table (sim/envs/maniskill at 7d4a0a1: every registered env minus
#: locomotion, humanoids and dexterous hands), the part a translation-only Panda can attempt.
_ADDED = [
    "PlaceSphere-v1",
    "StackPyramid-v1",
    "PullCubeTool-v1",
    "PegInsertionSide-v1",
    "PlugCharger-v1",
    "PickSingleYCB-v1",
    # The rest of that table: the Panda's FMB assembly, then the tasks built for another
    # robot (_OTHER_ROBOT: the WidowX AI's fixed PickCube, the panda_stick scenes).
    "FMBAssembly1Easy-v1",
    "PickCubeWidowXAI-v1",
    "PushT-v1",
    "DrawTriangle-v1",
    "DrawSVG-v1",
    "TwoRobotPickCube-v1",
    "TwoRobotStackCube-v1",
    "PutCarrotOnPlateInScene-v1",
    "PutEggplantInBasketScene-v1",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
    "PutSpoonOnTableClothInScene-v1",
]


def test_the_eight_existing_env_ids_are_unchanged_and_openeta_tasks_are_complete():
    assert ms.ENV_IDS[:8] == list(_FROZEN)
    assert ms.ENV_IDS[8:] == _ADDED
    for env_id, row in _FROZEN.items():
        if row is None:
            assert env_id not in ms.INSTRUCTIONS  # a rig: scenes.py owns its text
            continue
        text, actors, goals = row
        assert ms.INSTRUCTIONS[env_id] == text
        assert ms.TASK_ACTORS[env_id] == actors
        assert ms.SHOW_GOALS.get(env_id) == goals
    # Every added task has a text and a visibility list; PickSingleYCB's hidden goal is shown.
    for env_id in _ADDED:
        assert ms.INSTRUCTIONS[env_id] and len(ms.TASK_ACTORS[env_id]) >= 2
    assert ms.SHOW_GOALS["PickSingleYCB-v1"] == ["goal_site"]
    assert "goal" in ms.INSTRUCTIONS["PickSingleYCB-v1"]


# ---- --robot (ROBOTS) ----


def test_robot_table_panda_default_and_gripper_mapping():
    """The Panda is the default arm with the uid, gripper and every env id it always had;
    each robot maps pi's open (> 0) / close (<= 0) to its own gripper action."""
    assert list(ms.ROBOTS) == [
        "panda",
        "xarm6_robotiq",
        "widowxai",
        "panda_stick",
        "panda_pair",
        "widowx250s",
    ]
    panda = ms.ROBOTS["panda"]
    assert panda.uid == "panda_wristcam"
    assert panda.envs == tuple(e for e in ms.INSTRUCTIONS if e not in ms._OTHER_ROBOT)
    # Every other-robot task is its owner's, and only its owner's.
    for env_id, owner in ms._OTHER_ROBOT.items():
        assert [r for r, spec in ms.ROBOTS.items() if env_id in spec.envs] == [owner]
    assert panda.wrist == {"mount": "centered", "rotation": 270, "flip": "none"}
    assert (panda.gripper_action(1.0), panda.gripper_action(-1.0)) == (1.0, -1.0)
    # The Robotiq runs in delta mode: +1 closes, -1 opens.
    xarm = ms.ROBOTS["xarm6_robotiq"]
    assert (xarm.open, xarm.gripper_action(-1), xarm.gripper_action(0)) == (
        -1.0,
        1.0,
        1.0,
    )
    assert xarm.wrist["mount"] == "camera_link" and xarm.ee_joints is None
    # Its last joint's limit is 0: the dense rewards' finger-width divisor; sparse instead.
    assert xarm.reward_mode == "sparse" and panda.reward_mode is None
    wx = ms.ROBOTS["widowxai"]
    assert wx.wrist is None and wx.ee_joints == tuple(f"joint_{i}" for i in range(6))
    assert (wx.gripper_action(1), wx.gripper_action(-1)) == (1.0, -1.0)
    for spec in ms.ROBOTS.values():
        assert set(spec.envs) <= set(ms.INSTRUCTIONS)


def test_robot_spec_refuses_unknown_arms_rigs_and_unmeasured_scenes():
    assert ms.robot_spec("panda", "BlockPAP-v1", rig=True) is ms.ROBOTS["panda"]
    assert (
        ms.robot_spec("xarm6_robotiq", "PlugCharger-v1", rig=False).uid
        == "xarm6_robotiq"
    )
    for robot, env_id, rig, match in [
        ("ur5", "PickCube-v1", False, "unknown robot"),
        ("xarm6_robotiq", "BlockStack-v1", True, "--robot panda only"),
        ("xarm6_robotiq", "PushCube-v1", False, "not PushCube-v1"),
        # Its base is near enough that PullCubeTool starts solved on ~8 % of seeds.
        ("xarm6_robotiq", "PullCubeTool-v1", False, "not PullCubeTool-v1"),
        ("widowxai", "StackCube-v1", False, "not StackCube-v1"),
    ]:
        try:
            ms.robot_spec(robot, env_id, rig)
            raise AssertionError(f"{robot} on {env_id} passed")
        except ValueError as e:
            assert match in str(e), e


def _fake_mani_skill(monkeypatch):
    """Stand-ins for the ManiSkill modules the patch helpers import."""
    import sys
    import types

    class Cfg:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    mods = {
        "sapien": types.SimpleNamespace(Pose=lambda p, q: ("pose", tuple(p), tuple(q))),
        "mani_skill": types.ModuleType("mani_skill"),
        "mani_skill.agents": types.ModuleType("mani_skill.agents"),
        "mani_skill.agents.controllers": types.SimpleNamespace(
            PDEEPosControllerConfig=Cfg
        ),
        "mani_skill.agents.registration": types.SimpleNamespace(REGISTERED_AGENTS={}),
        "mani_skill.sensors": types.ModuleType("mani_skill.sensors"),
        "mani_skill.sensors.camera": types.SimpleNamespace(CameraConfig=Cfg),
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return mods


class _Agent:
    arm_stiffness, arm_damping, arm_force_limit = 1e3, 1e2, 100
    ee_link_name, urdf_path = "ee_gripper_link", "wxai_base.urdf"

    def __init__(self):
        self.robot = type("R", (), {"links_map": {"camera_link": "LINK"}})()

    @property
    def _controller_configs(self):
        return {
            "pd_joint_pos": {"arm": "joint-arm", "gripper": "joint-gripper"},
            "pd_joint_delta_pos": {"arm": "delta-arm", "gripper": "joint-gripper"},
        }

    @property
    def _sensor_configs(self):
        return []


def test_prepare_robot_adds_the_ee_mode_and_the_wrist_camera(monkeypatch):
    """widowxai ships joint control only: pd_ee_delta_pos is ManiSkill's PDEEPosController
    on its six arm joints and TCP link, with the pd_joint_pos gripper; xarm6_robotiq gets
    its wristcam variant's hand camera on camera_link; the Panda keeps the centred mount."""
    mods = _fake_mani_skill(monkeypatch)
    wx = type("WidowXAI", (_Agent,), {})
    xarm = type("XArm6Robotiq", (_Agent,), {})
    agents = mods["mani_skill.agents.registration"].REGISTERED_AGENTS
    agents["widowxai"] = type("A", (), {"agent_cls": wx})
    agents["xarm6_robotiq"] = type("A", (), {"agent_cls": xarm})
    agents["panda_wristcam"] = type("A", (), {"agent_cls": type("P", (_Agent,), {})})

    ms.prepare_robot(ms.ROBOTS["widowxai"], "centered")
    wrapped = wx._controller_configs
    ms.prepare_robot(ms.ROBOTS["widowxai"], "centered")  # idempotent: not wrapped twice
    assert wx._controller_configs is wrapped
    cfg = wx()._controller_configs
    ee = cfg["pd_ee_delta_pos"]
    assert set(ee) == {"arm", "gripper"} and ee["gripper"] == "joint-gripper"
    arm = ee["arm"]
    assert arm.joint_names == [f"joint_{i}" for i in range(6)]
    assert (arm.pos_lower, arm.pos_upper) == (-ms.DELTA_BOUND_M, ms.DELTA_BOUND_M)
    assert (arm.ee_link, arm.urdf_path, arm.stiffness) == (
        "ee_gripper_link",
        "wxai_base.urdf",
        1e3,
    )
    assert cfg["pd_joint_pos"]["arm"] == "joint-arm"  # the stock modes stay
    assert wx()._sensor_configs == []  # no camera link: agentview only

    ms.prepare_robot(ms.ROBOTS["xarm6_robotiq"], "centered")
    (cam,) = xarm()._sensor_configs
    assert (cam.uid, cam.mount, cam.width, cam.height) == (
        "hand_camera",
        "LINK",
        256,
        256,
    )
    assert cam.pose == ("pose", (0.0, 0.0, -0.05), (0.70710678, 0.0, 0.70710678, 0.0))
    assert "pd_ee_delta_pos" not in xarm()._controller_configs  # it has its own

    centred = []
    monkeypatch.setattr(ms, "center_wrist_camera", lambda: centred.append(1))
    ms.prepare_robot(ms.ROBOTS["panda"], "camera_link")
    ms.prepare_robot(ms.ROBOTS["panda"], "centered")
    assert centred == [1]


def _facade(robot: str, wrist: bool = True, root_q=(1.0, 0.0, 0.0, 0.0)):
    facade = object.__new__(ms.ManiskillEnvFacade)
    facade._robot = ms.ROBOTS[robot]
    facade._rig = None
    facade._cameras = ms.CAMERAS
    facade._meta = {"robot": robot, "wrist": wrist, "env_id": "PickCube-v1"}
    pose = type("P", (), {"q": np.array([root_q])})()
    tcp = type("T", (), {"pose": type("P", (), {"p": np.zeros((1, 3))})()})()
    agent = type("A", (), {"robot": type("R", (), {"pose": pose})(), "tcp": tcp})()
    facade._grip = [1.0]
    facade._env = type("E", (), {"unwrapped": type("U", (), {"agent": agent})()})()
    return facade


def test_servo_holds_the_robots_gripper_action_and_the_width_is_its_own():
    """The servo turns pi's open / close into each robot's action; the Robotiq's width is
    the pad distance less its closed-empty 6.7 mm, the others the finger joints' sum."""
    for robot, command, action in [
        ("panda", -1, -1.0),
        ("xarm6_robotiq", -1, 1.0),
        ("xarm6_robotiq", 1, -1.0),
        ("widowxai", 1, 1.0),
    ]:
        f = _facade(robot)
        sent = []
        f.stop_requested = lambda: False
        f._state = lambda: {"tcp_pos": np.zeros(3)}
        f._pack = lambda obs: obs
        f._step = lambda a, sent=sent: (sent.append(a) or "obs", 0.0, False, False, {})
        frames, _ = f.servo([0.01, 0, 0], command, min_steps=2, max_steps=2)
        assert len(sent) == 2 and all(a[-1] == action for a in sent), (robot, sent)
        assert np.allclose(sent[0][:3], [0.01 * 1.3 / ms.DELTA_BOUND_M, 0, 0])

    qpos = np.array([0.1, 0.2, 0.02, 0.03])
    assert abs(_facade("widowxai")._gripper_width(qpos) - 0.05) < 1e-9
    xarm = _facade("xarm6_robotiq")
    links = {
        "left_inner_finger_pad": type(
            "L", (), {"pose": type("P", (), {"p": np.array([0, 0.0467, 0.1])})()}
        )(),
        "right_inner_finger_pad": type(
            "L", (), {"pose": type("P", (), {"p": np.array([0, 0.0, 0.1])})()}
        )(),
    }
    agent = type("A", (), {"robot": type("R", (), {"links_map": links})()})()
    xarm._env = type("E", (), {"unwrapped": type("U", (), {"agent": agent})()})()
    assert abs(xarm._gripper_width(qpos) - 0.04) < 1e-9
    links["left_inner_finger_pad"].pose.p = np.array([0, 0.005, 0.1])
    assert xarm._gripper_width(qpos) == 0.0  # closed on nothing


def _servo_actions(f, target, command=1.0):
    sent = []
    f.stop_requested = lambda: False
    f._state = lambda: {"tcp_pos": np.zeros(3)}
    f._pack = lambda obs: obs
    f._step = lambda a: (sent.append(a) or "obs", 0.0, False, False, {})
    f.servo(target, command, min_steps=1, max_steps=1)
    return sent


def test_a_robot_without_a_gripper_sends_the_translation_alone():
    """panda_stick's pd_ee_delta_pos takes [dx, dy, dz]: no gripper element, whatever the
    command; its width is always 0 and the reset hold is three zeros."""
    stick = ms.ROBOTS["panda_stick"]
    assert stick.gripper is None and stick.wrist is None and stick.open is None
    assert stick.gripper_action(1.0) is None and stick.gripper_action(-1.0) is None
    (a,) = _servo_actions(_facade("panda_stick", wrist=False), [0.0, 0.01, 0.0], -1.0)
    assert a.shape == (3,) and np.allclose(a, [0, 0.13, 0])
    assert list(stick.action(np.zeros(3), 1.0)) == [0.0, 0.0, 0.0]
    assert list(ms.ROBOTS["panda"].action(np.zeros(3), 1.0)) == [0.0, 0.0, 0.0, 1.0]
    assert _facade("panda_stick")._gripper_width(np.array([0.1, 0.2])) == 0.0


def test_the_servo_commands_in_the_robot_base_frame():
    """pd_ee_delta_pos acts in the robot base's frame: a base at the identity rotation gets
    the world error as is (every Panda table scene); one turned about z (RollBall's, the
    two-robot scenes', the SO100's) gets it rotated into its frame."""
    (a,) = _servo_actions(_facade("panda"), [0.01, 0.0, 0.0])
    assert np.array_equal(a[:3], np.array([0.01 * 1.3 / ms.DELTA_BOUND_M, 0.0, 0.0]))
    # A base turned -90 deg about z (RollBall: q = [0.707, 0, 0, -0.707]) faces world -y:
    # world +x is its +y.
    s = np.sqrt(0.5)
    (a,) = _servo_actions(_facade("panda", root_q=(s, 0, 0, -s)), [0.01, 0.0, 0.0])
    assert np.allclose(a[:3], [0.0, 0.13, 0.0]), a


class _Pose:
    def __init__(self, p, q=(1.0, 0.0, 0.0, 0.0)):
        self.p, self.q = np.array([p]), np.array([q])


def _pair_facade():
    """A panda_pair facade whose two arms face each other (left turned +90 deg about z,
    right -90 deg), with a two-agent action space."""
    f = object.__new__(ms.ManiskillEnvFacade)
    f._robot = ms.ROBOTS["panda_pair"]
    f._rig = None
    f._cameras = ms.CAMERAS
    f._meta = {"robot": "panda_pair", "wrist": False, "env_id": "TwoRobotPickCube-v1"}
    f._grip = [1.0, 1.0]
    s = np.sqrt(0.5)
    agents = []
    for y, q in [(-0.12, (s, 0, 0, s)), (0.12, (s, 0, 0, -s))]:
        robot = type("R", (), {"pose": _Pose([0, 0, 0], q)})()
        agents.append(
            type(
                "A",
                (),
                {"robot": robot, "tcp": type("T", (), {"pose": _Pose([0, y, 0.18])})()},
            )()
        )
    multi = type("M", (), {"agents": agents})()
    space = type("S", (), {"spaces": {"panda-0": None, "panda-1": None}})()
    f._env = type(
        "E", (), {"unwrapped": type("U", (), {"agent": multi})(), "action_space": space}
    )()
    return f


def test_a_two_arm_robot_drives_one_arm_and_holds_the_other():
    """panda_pair: the servo moves the named arm in its own base frame; the other gets a
    zero move with its last gripper command; the action is keyed by the env's agent ids."""
    f = _pair_facade()
    sent = []
    f.stop_requested = lambda: False
    f._pack = lambda obs: obs
    f._step = lambda a: (sent.append(a) or "obs", 0.0, False, False, {})
    f.servo([0.01, -0.12, 0.18], -1.0, arm="left", min_steps=1, max_steps=1)
    (a,) = sent
    assert list(a) == ["panda-0", "panda-1"]
    # World +x is the left arm's (turned +90 deg) -y.
    assert np.allclose(a["panda-0"], [0, -0.13, 0, -1.0], atol=1e-6), a
    assert np.allclose(a["panda-1"], [0, 0, 0, 1.0])
    f.servo([0.0, 0.12, 0.19], 1.0, arm="right", min_steps=1, max_steps=1)
    assert np.allclose(
        sent[-1]["panda-0"], [0, 0, 0, -1.0]
    )  # left keeps its closed gripper
    assert np.allclose(sent[-1]["panda-1"], [0, 0, 0.13, 1.0], atol=1e-6)
    for bad in [None, "middle"]:
        try:
            f.servo([0, 0, 0], 1.0, arm=bad)
            raise AssertionError(f"arm {bad!r} passed")
        except ValueError as e:
            assert "left" in str(e)
    try:
        _facade("panda").servo([0, 0, 0], 1.0, arm="left")
        raise AssertionError("an arm on one arm passed")
    except ValueError as e:
        assert "one arm" in str(e)
    # A flat code-mode action is split per arm.
    split = f._split(np.arange(8))
    assert list(split["panda-1"]) == [4, 5, 6, 7]


def test_code_mode_raw_actions_use_pis_gripper_sign_on_every_robot():
    """step / chunk_step take [dx, dy, dz, gripper] with > 0 open on every robot: the
    Panda's passes through unchanged (continuous values too), the Robotiq's gripper sign
    is flipped (its +1 closes), a stick sends the translation alone."""
    f = _facade("panda")
    f._robot = ms.ROBOTS["panda"]
    assert np.array_equal(
        f._split([0.1, -0.2, 0.3, 0.5]), np.float32([0.1, -0.2, 0.3, 0.5])
    )
    x = _facade("xarm6_robotiq")
    assert np.allclose(x._split([0.1, 0, 0, 1.0]), [0.1, 0, 0, -1.0])
    assert np.allclose(x._split([0, 0, 0, -0.4]), [0, 0, 0, 0.4])
    assert np.allclose(_facade("panda_stick")._split([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3])


def test_the_bridge_widowx_runs_an_untargeted_copy_of_the_scenes_mode(monkeypatch):
    """The scene's target-based pose mode sums every commanded delta; the WidowX runs a copy
    acting on the current pose, added to both bridge agents (idempotently)."""
    mods = _fake_mani_skill(monkeypatch)
    agents = mods["mani_skill.agents.registration"].REGISTERED_AGENTS
    Cfg = mods["mani_skill.sensors.camera"].CameraConfig
    classes = {}
    wx = ms.ROBOTS["widowx250s"]
    base = wx.untargeted_from[1]
    for uid in wx.untargeted_from[0]:

        def configs(self, base=base):
            return {base: {"arm": Cfg(use_target=True, pos_lower=-1.0), "gripper": "g"}}

        classes[uid] = type(uid, (), {"_controller_configs": property(configs)})
        agents[uid] = type("A", (), {"agent_cls": classes[uid]})
    ms.prepare_robot(wx, "centered")
    ms.prepare_robot(wx, "centered")
    for cls in classes.values():
        cfg = cls()._controller_configs
        assert cfg[base]["arm"].use_target is True  # the scene's own mode is untouched
        mode = cfg[wx.control_mode]
        assert mode["arm"].use_target is False and mode["gripper"] == "g"
        assert mode["arm"].pos_lower == -1.0
    assert wx.control_mode == "pi_ee_delta_pose_gripper_pd_joint_pos"


def test_the_bridge_widowx_takes_an_unnormalised_pose_action_and_objs_actors():
    """The bridge twins' WidowX 250 S: its pose controller takes [dx, dy, dz] in m, three
    zero rotations, then the gripper (+1 open); its actors live in ``objs`` by model id,
    and every bridge scene reports its table height."""
    wx = ms.ROBOTS["widowx250s"]
    a = wx.action(np.array([0.5, 0.0, -1.0]), -1.0)
    assert np.allclose(a, [0.05, 0.0, -0.1, 0, 0, 0, -1.0])
    assert wx.untargeted_from[1].startswith("arm_pd_ee_target_delta_pose")
    assert (wx.tcp_link, wx.agentview) == ("ee_gripper_link", "3rd_view_camera")
    assert set(wx.envs) == set(ms.BRIDGE_ENVS) == set(ms.TABLE_Z)
    assert set(ms.BRIDGE_ENVS) <= set(ms.FIXED_ROBOT_ENVS)
    env = type("E", (), {"objs": {"eggplant": "EGG"}, "sink": "SINK"})()
    assert ms._actor(env, "objs/eggplant") == "EGG" and ms._actor(env, "sink") == "SINK"
    for env_id in ms.BRIDGE_ENVS:
        assert ms.TASK_ACTORS[env_id][0].startswith("objs/")
    # The Panda's action is unchanged by the pose-controller fields.
    assert list(ms.ROBOTS["panda"].action(np.array([0.5, 0, 0]), 1.0)) == [
        0.5,
        0,
        0,
        1.0,
    ]


def test_a_robot_without_a_wrist_camera_observes_the_agentview_alone():
    f = _facade("widowxai", wrist=False)
    f._rgb = lambda obs, name: f"{name}-image"
    f._state = lambda: {"tcp_pos": np.zeros(3)}
    packed = f._pack({})
    assert packed["agentview"] == "agentview-image" and "wrist" not in packed
    try:
        f.render_camera("wrist")
        raise AssertionError("a wrist render passed")
    except ValueError as e:
        assert "no wrist camera" in str(e)
    assert ms.ManiskillEnvFacade._sensor_configs("stock", wrist=False) == {}
    assert ms.ManiskillEnvFacade._sensor_configs("stock") == {
        "hand_camera": {"width": 256, "height": 256}
    }
    wristed = _facade("xarm6_robotiq")
    wristed._rgb, wristed._state = f._rgb, f._state
    assert wristed._pack({})["wrist"] == "wrist-image"


def test_depth_takes_the_views_orientation_and_letterbox_pixel_for_pixel():
    """render_camera(depth=True): the metric depth lands on the same pixels as the rgb
    (orientation, letterbox bars at 0 = no depth), for the agentview and the rotated wrist."""
    f = _facade("panda")
    f._rig = False
    f._view_size = 64
    f._wrist_rotation, f._wrist_flip = 270, "none"
    h, w = 24, 32  # a 4:3 sensor: bars above and below in the square
    rgb = np.zeros((1, h, w, 3), dtype=np.uint8)
    rgb[0, :, : w // 2] = 255  # the left half bright
    depth = np.full((1, h, w, 1), 500, dtype=np.int16)  # 0.5 m
    depth[0, :, : w // 2] = 1200  # the left half 1.2 m
    f._obs = {
        "sensor_data": {
            f._cameras[n]: {"rgb": rgb, "depth": depth} for n in ("agentview", "wrist")
        }
    }
    for name in ("agentview", "wrist"):
        img, d = f.render_camera(name, depth=True)
        assert d.dtype == np.float32 and d.shape == img.shape[:2] == (64, 64)
        bright = img[..., 0] > 200
        dark = (img[..., 0] < 50) & (d > 0)
        assert np.all(d[bright] == np.float32(1.2)), name
        assert np.all(d[dark] == np.float32(0.5)), name
        # The letterbox bars have no depth.
        assert np.all(d[(img.sum(-1) == 0) & (d == 0)] == 0)
        assert (d == 0).sum() > 0 and name
    # Without depth the rgb alone, as before.
    assert f.render_camera("agentview").shape == (64, 64, 3)
