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
