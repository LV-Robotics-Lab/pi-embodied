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
