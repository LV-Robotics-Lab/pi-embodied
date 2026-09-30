"""Relative humanoid odometry must not disclose the target or world position."""

import numpy as np

from pi_embodied_services.robots.humanclaw.env_server import motion_feedback


def test_actual_motion_in_y_up_frame_and_angle_wrap():
    before = (np.array([10.0, 3.0, -8.0]), 175.0)
    after = (np.array([10.3, 2.8, -7.6]), -165.0)
    initial = (np.array([9.0, 3.1, -9.0]), 140.0)
    assert motion_feedback(before, after, initial) == {
        "horizontal_displacement_m": 0.5,
        "height_change_m": -0.2,
        "turned_left_deg": 20.0,
        "heading_from_start_left_deg": 55.0,
        "height_from_start_m": -0.3,
    }


def test_feedback_does_not_depend_on_absolute_scene_position():
    before = (np.array([0.0, 1.0, 0.0]), -170.0)
    after = (np.array([0.0, 1.0, 0.0]), 170.0)
    offset = np.array([120.0, 200.0, -90.0])
    expected = motion_feedback(before, after, before)
    assert expected["turned_left_deg"] == -20.0
    assert expected["horizontal_displacement_m"] == 0
    shifted_before = (before[0] + offset, before[1])
    shifted_after = (after[0] + offset, after[1])
    assert motion_feedback(shifted_before, shifted_after, shifted_before) == expected
