"""The Viser 3D view (components/viser_view.py) against a stand-in Viser and env server."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from pi_embodied_services.components import viser_view as vv


class Handle:
    def __init__(self, kind: str, name: str | None, kw: dict) -> None:
        self.kind, self.name, self.kw = kind, name, kw
        self.removed = False
        self.image = kw.get("image")
        self.content = kw.get("content")

    def remove(self) -> None:
        self.removed = True


class FakeScene:
    def __init__(self) -> None:
        self.calls: list[Handle] = []

    def _add(self, kind: str, name: str, **kw):
        h = Handle(kind, name, kw)
        self.calls.append(h)
        return h

    def add_point_cloud(self, name, **kw):
        return self._add("points", name, **kw)

    def add_frame(self, name, **kw):
        return self._add("frame", name, **kw)

    def add_camera_frustum(self, name, **kw):
        return self._add("frustum", name, **kw)


class FakeGui:
    def __init__(self) -> None:
        self.images: list[Handle] = []
        self.texts: list[Handle] = []

    def add_image(self, image, label=None):
        h = Handle("image", label, {"image": image})
        self.images.append(h)
        return h

    def add_markdown(self, content):
        h = Handle("markdown", None, {"content": content})
        self.texts.append(h)
        return h


class FakeServer:
    def __init__(self) -> None:
        self.scene = FakeScene()
        self.gui = FakeGui()


def camera(depth_value: float = 1.0, size: int = 8) -> dict:
    """A camera at the world origin looking down +z (OpenCV), fx = fy = size, centred."""
    return {
        "rgb": np.full((size, size, 3), 200, np.uint8),
        "depth": np.full((size, size), depth_value, np.float32),
        "intrinsic_K": np.array(
            [[size, 0, size / 2], [0, size, size / 2], [0, 0, 1.0]]
        ),
        "extrinsic_cam2world": np.eye(4),
    }


def libero_obs(**over) -> dict:
    obs = {
        "agentview": camera(),
        "wrist": camera(0.5),
        "eef_pos": [0.1, 0.2, 0.3],
        "eef_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "gripper_width": 0.08,
    }
    obs.update(over)
    return obs


def test_back_project_follows_the_pinhole_model_and_drops_empty_depth():
    view = vv._view("cam", camera(2.0))
    view.depth[0, 0] = 0.0
    view.depth[0, 1] = 9.0  # beyond max_depth
    view.cam2world = np.eye(4)
    view.cam2world[:3, 3] = [1.0, 0.0, 0.0]
    pts, cols = vv.back_project(view, stride=1, max_depth=3.0)
    assert pts.shape == (62, 3) and cols.shape == (62, 3)
    # pixel (row 4, col 4) is the principal point: straight ahead at z = 2, shifted by +x 1
    i = int(np.argmin(np.abs(pts[:, 0] - 1.0) + np.abs(pts[:, 1])))
    np.testing.assert_allclose(pts[i], [1.0, 0.0, 2.0], atol=1e-6)
    # (row 4, col 0): x = (0 - 4) * 2 / 8 = -1
    assert np.isclose(pts[:, 0].min(), 0.0)
    assert (cols == 200).all()
    no_cal = vv._view("cam", {**camera(), "extrinsic_cam2world": None})
    assert len(vv.back_project(no_cal)[0]) == 0


def test_libero_snapshot_reads_both_views_and_the_eef():
    calls = []
    snap = vv.libero_snapshot(lambda m: calls.append(m) or libero_obs())
    assert calls == ["env.get_observation"]
    assert [c.name for c in snap.cameras] == ["agentview", "wrist"]
    np.testing.assert_allclose(snap.eef[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(snap.eef[1], [1, 0, 0, 0])
    assert snap.gripper_width == 0.08


def test_scene_draws_clouds_frustums_eef_and_updates_the_gui_in_place():
    server = FakeServer()
    scene = vv.Scene(server, stride=2)
    n = scene.update(vv.libero_snapshot(lambda m: libero_obs()), "step 1")
    assert n == 2 * 16
    kinds = [(h.kind, h.name) for h in server.scene.calls]
    assert kinds == [
        ("points", "/cameras/agentview/points"),
        ("frustum", "/cameras/agentview/frustum"),
        ("points", "/cameras/wrist/points"),
        ("frustum", "/cameras/wrist/frustum"),
        ("frame", "/eef"),
    ]
    frustum = server.scene.calls[1].kw
    assert frustum["aspect"] == 1.0
    assert np.isclose(frustum["fov"], 2 * np.arctan(0.5))
    assert "gripper width 8.0 cm" in server.gui.texts[0].content
    scene.update(vv.libero_snapshot(lambda m: libero_obs(gripper_width=0.01)), "step 2")
    assert len(server.gui.images) == 1 and len(server.gui.texts) == 1, (
        "updated in place"
    )
    assert "1.0 cm" in server.gui.texts[0].content


def test_grasp_candidates_become_frames_the_active_one_larger_and_replace_the_last_plan():
    server = FakeServer()
    scene = vv.Scene(server)
    cands = [
        {
            "id": "g1",
            "position": [0.5, 0, 0.1],
            "approach": [0, 0, -1],
            "closing": [0, 1, 0],
        },
        {"id": "g2", "eef_position": [0.4, 0, 0.2], "eef_quat_xyzw": [1, 0, 0, 0]},
        {"id": "g3"},  # no pose
    ]
    assert scene.grasps(cands, active="g1") == 2
    first = [h for h in server.scene.calls if h.name.startswith("/grasps/")]
    assert [h.name for h in first] == ["/grasps/g1", "/grasps/g2"]
    assert first[0].kw["axes_length"] > first[1].kw["axes_length"]
    # approach -z as the frame's x axis: R = [[0,0,1],[0,1,0],[-1,0,0]] -> 90 deg about y
    np.testing.assert_allclose(
        np.abs(first[0].kw["wxyz"]), [np.sqrt(0.5), 0, np.sqrt(0.5), 0], atol=1e-6
    )
    np.testing.assert_allclose(first[1].kw["wxyz"], [0, 1, 0, 0])
    assert scene.grasps([], None) == 0
    assert all(h.removed for h in first)


class FakeEnv:
    def __init__(self, obs):
        self.obs = obs
        self.calls = []

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append(method)
        if isinstance(self.obs, Exception):
            raise self.obs
        return self.obs


def test_facade_refresh_status_attach_and_grasps():
    envs = {
        "http://a:1": FakeEnv(libero_obs()),
        "http://b:2": FakeEnv(RuntimeError("busy")),
    }
    server = FakeServer()
    f = vv.ViserViewFacade(
        vv.Scene(server),
        vv.libero_snapshot,
        env="http://a:1",
        url="http://0.0.0.0:8080",
        client=lambda url, token: envs[url],
    )
    assert set(f._rpc) >= {
        "viser.status",
        "viser.refresh",
        "viser.grasps",
        "viser.attach",
    }
    s = f.refresh()
    assert s["updates"] == 1 and s["points"] > 0 and s["env"] == "http://a:1"
    assert f.grasps([{"id": "g1", "position": [0, 0, 0]}], "g1") == {"drawn": 1}
    f.attach("http://b:2")
    with pytest.raises(RuntimeError, match="busy"):
        f.refresh()
    assert f.status()["last_error"] == "RuntimeError: busy"
    assert f.status()["updates"] == 1


def test_poll_thread_keeps_going_after_errors_and_stops_on_the_event():
    env = FakeEnv(RuntimeError("down"))
    f = vv.ViserViewFacade(
        vv.Scene(FakeServer()),
        vv.libero_snapshot,
        env="x",
        interval=0.01,
        client=lambda u, t: env,
    )
    stop = threading.Event()
    t = threading.Thread(target=f.poll_forever, args=(stop,))
    t.start()
    while len(env.calls) < 3:
        pass
    env.obs = libero_obs()
    while f.updates < 1:
        pass
    stop.set()
    t.join(2)
    assert not t.is_alive()


def test_franka_snapshot_places_cameras_with_the_calibration_or_only_the_eef_without():
    obs = {
        "main_images": np.zeros((4, 4, 3), np.uint8),
        "main_depths": np.ones((4, 4, 1), np.float32),
        "extra_view_images": np.zeros((1, 4, 4, 3), np.uint8),
        "extra_view_depths": np.ones((1, 4, 4), np.float32),
    }
    K = [[4, 0, 2], [0, 4, 2], [0, 0, 1]]
    meta = {
        "observation_camera_map": {"main": "w", "extra_0": "e"},
        "cameras": {"w": {"intrinsic_K": K}, "e": {"intrinsic_K": K}},
    }
    state = {"raw_base_state": {"tcp_pose": [0.3, 0, 0.5, 0, 0, 0, 1]}}
    replies = {
        "env.get_observation": obs,
        "env.get_camera_meta": meta,
        "env.get_robot_state": state,
    }
    ext = np.eye(4)
    ext[:3, 3] = [1, 2, 3]
    cal = {"wrist": {"matrix": np.eye(4)}, "external": {"matrix": ext}}
    snap = vv.franka_snapshot(replies.__getitem__, lambda: cal)
    assert [c.name for c in snap.cameras] == ["wrist", "third_person"]
    np.testing.assert_allclose(snap.cameras[0].cam2world[:3, 3], [0.3, 0, 0.5])
    np.testing.assert_allclose(snap.cameras[1].cam2world[:3, 3], [1, 2, 3])
    assert snap.cameras[0].depth.shape == (4, 4)
    bare = vv.franka_snapshot(replies.__getitem__, lambda: None)
    assert all(c.cam2world is None for c in bare.cameras)
    assert bare.eef is not None and bare.notes
    assert vv.Scene(FakeServer()).update(bare) == 0


def test_the_env_token_goes_to_the_client_and_is_sent_with_every_call(monkeypatch):
    seen = []
    f = vv.ViserViewFacade(
        vv.Scene(FakeServer()),
        vv.libero_snapshot,
        env="http://a:1",
        token="t0",
        client=lambda url, token: seen.append((url, token)) or FakeEnv(libero_obs()),
    )
    f.attach("http://b:2", "t1")
    f.attach("http://c:3")
    assert seen == [("http://a:1", "t0"), ("http://b:2", "t1"), ("http://c:3", None)]

    sent = []

    class Reply:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import json

    client = vv._http_client("127.0.0.1:9", "secret")

    def fake_open(req, timeout=None):
        sent.append(json.loads(req.data))
        return Reply(json.dumps({"ok": True, "result": {}}).encode())

    monkeypatch.setattr(client._opener, "open", fake_open)
    client.call("env.get_observation")
    assert sent[0]["token"] == "secret" and sent[0]["method"] == "env.get_observation"
