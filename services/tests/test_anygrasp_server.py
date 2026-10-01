"""Exercise the licensed SDK boundary without proprietary binaries or GPU weights."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pi_embodied_services.components import anygrasp_server as S


def facade(tmp_path, monkeypatch, detector):
    sdk = tmp_path / "sdk"
    license_dir = sdk / "grasp_detection" / "license"
    license_dir.mkdir(parents=True)
    (license_dir / "licenseCfg.json").write_text("{}")
    (license_dir / "test.lic").write_text("test fixture")
    checkpoint = tmp_path / "checkpoint.tar"
    checkpoint.touch()
    configs = []

    def create_detector(config):
        configs.append(config)
        return detector

    monkeypatch.setitem(
        sys.modules, "gsnet", SimpleNamespace(create_detector=create_detector)
    )
    monkeypatch.syspath_prepend(str(sdk / "grasp_detection"))
    instance = S.AnyGraspFacade(
        str(sdk),
        str(checkpoint),
        max_gripper_width=0.08,
        gripper_height=0.03,
        depth_truncation=2.0,
        collision_detection=True,
    )
    assert configs[0].checkpoint_path == str(checkpoint)
    assert configs[0].max_gripper_width == 0.08
    return instance


def test_factory_failure_is_not_a_healthy_server(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="initialization failed"):
        facade(tmp_path, monkeypatch, None)


def test_region_mask_tracks_valid_points_and_new_sdk_returns_group(
    tmp_path, monkeypatch
):
    class Group(list):
        def nms(self):
            return self

        def sort_by_score(self):
            return self

    grasp = SimpleNamespace(
        translation=np.array([0.01, 0, 1.0]),
        rotation_matrix=np.eye(3),
        score=0.9,
        width=0.04,
        height=0.03,
        depth=0.02,
    )
    calls = []

    def get_grasp(points, options):
        calls.append((points, options))
        return Group([grasp])

    server = facade(tmp_path, monkeypatch, SimpleNamespace(get_grasp=get_grasp))
    depth = np.array([[0, 1, 3], [1, 1, np.nan]], dtype=np.float32)
    mask = np.array([[True, True, True], [False, False, False]])
    result = server.plan(
        depth,
        np.diag([100, 100, 1]),
        mask,
        max_candidates=1,
        up_direction_camera=[0, 0, -1],
    )
    points, options = calls[0]
    assert points.shape == (3, 3) and points.dtype == np.float32
    assert options["region_steering"].tolist() == [True, False, False]
    assert options["collision_detection"] is True
    assert options["dense_grasp"] is False
    assert options["approach_steering"].tolist() == [0, 0, 1]
    assert options["approach_thresh"] == pytest.approx(np.pi / 6)
    assert result["backend"] == "anygrasp"
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["source_model"] == "anygrasp"
    assert result["metadata"]["scene_points"] == 3


def test_empty_sdk_prediction_is_a_valid_empty_result(tmp_path, monkeypatch):
    server = facade(
        tmp_path, monkeypatch, SimpleNamespace(get_grasp=lambda *args: None)
    )
    result = server.plan(np.ones((2, 2)), np.eye(3), np.ones((2, 2)))
    assert result["candidates"] == []
    assert result["model_candidate_count"] == 0


def test_invalid_target_depth_fails_before_sdk_call(tmp_path, monkeypatch):
    server = facade(tmp_path, monkeypatch, SimpleNamespace())
    with pytest.raises(ValueError, match="no pixel with depth"):
        server.plan(np.zeros((2, 2)), np.eye(3), np.ones((2, 2)))
