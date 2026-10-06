from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
import torch
import yaml
from arena_planners import goal, weights
from arena_planners.bridge.chunk import check_chunk, chunk_config_from_manifest

import planner

_PLANNER_DIR = pathlib.Path(planner.__file__).parent
_GOAL = {"x": 3.0, "y": 0.0, "theta": 0.0}
_STRAIGHT = np.tile(np.array([[0.0, 1.0]]), (planner.HORIZON, 1)).cumsum(axis=0)
_RIGHT = np.tile(np.array([[1.0, 1.0]]), (planner.HORIZON, 1)).cumsum(axis=0)


def _manifest() -> dict:
    with open(_PLANNER_DIR / "planner.yaml") as fh:
        return yaml.safe_load(fh)


def test_model_frame_x_is_right_and_y_is_forward():
    points = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, -2.0]])
    out = planner.to_model_frame(np.zeros(3), points)
    np.testing.assert_allclose(out, [[0.0, 2.0], [-2.0, 0.0], [2.0, 0.0]], atol=1e-9)


def test_model_frame_rotates_by_heading():
    out = planner.to_model_frame(np.array([1.0, 1.0, math.pi / 2]), np.array([[1.0, 3.0], [3.0, 1.0]]))
    np.testing.assert_allclose(out, [[0.0, 2.0], [2.0, 0.0]], atol=1e-9)


def test_clip_radius_keeps_direction():
    np.testing.assert_allclose(planner.clip_radius(np.array([30.0, 40.0]), 10.0), [6.0, 8.0])
    np.testing.assert_allclose(planner.clip_radius(np.array([3.0, 4.0]), 10.0), [3.0, 4.0])


def test_padded_repeats_the_oldest_entry():
    assert planner.padded([1, 2], 5) == [1, 1, 1, 1, 2]
    assert planner.padded([1, 2, 3, 4, 5, 6], 5) == [2, 3, 4, 5, 6]


def test_input_waypoints_hold_history_then_goal_in_steps():
    poses = [np.array([x, 0.0, 0.0]) for x in (0.0, 0.5, 1.0, 1.5, 2.0)]
    out = planner.input_waypoints(poses, {"x": 5.0, "y": -1.0, "theta": 0.0}, 0.5, 10.0)
    assert out.shape == (6, 2)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(out[:, 1], [-4.0, -3.0, -2.0, -1.0, 0.0, 6.0], atol=1e-6)


def test_input_waypoints_pad_a_standing_start_with_zeros():
    out = planner.input_waypoints([np.array([2.0, 1.0, 0.3])], {"x": 2.0, "y": 1.0, "theta": 0.0}, 0.5, 10.0)
    np.testing.assert_allclose(out, np.zeros((6, 2)), atol=1e-6)


def test_input_waypoints_clip_the_goal_radius():
    out = planner.input_waypoints([np.zeros(3)], {"x": 60.0, "y": 80.0, "theta": 0.0}, 0.5, 10.0)
    np.testing.assert_allclose(out[-1], [-16.0, 12.0], atol=1e-5)


def test_chunk_scales_steps_to_meters_in_the_bridge_frame():
    waypoints = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]])
    chunk = planner.to_chunk(waypoints, 0.5)
    assert len(chunk) == 5
    np.testing.assert_allclose(chunk[0], [1.0, -0.5])
    np.testing.assert_allclose(chunk[-1], [5.0, -2.5])
    assert check_chunk("waypoints", chunk) == chunk


def test_closest_to_picks_the_path_ending_nearest_the_target():
    paths = np.stack([_STRAIGHT, _RIGHT])
    np.testing.assert_array_equal(planner.closest_to(paths, np.array([6.0, 6.0])), _RIGHT)
    np.testing.assert_array_equal(planner.closest_to(paths, np.array([0.0, 6.0])), _STRAIGHT)


@pytest.mark.parametrize(
    "prompt",
    [
        "go <input_pos1><input_pos2><input_pos3><input_pos4><input_pos5>",
        "go <input_target><input_pos1><input_pos2><input_pos3><input_pos4><input_pos5>",
        "go <input_pos1><input_pos1><input_pos2><input_pos3><input_pos4><input_pos5><input_target>",
    ],
)
def test_prompt_without_ordered_placeholders_is_refused(prompt: str):
    with pytest.raises(ValueError):
        planner.check_prompt(prompt)


def test_default_prompt_passes_the_placeholder_check():
    assert planner.check_prompt(planner.DEFAULT_PROMPT) == planner.DEFAULT_PROMPT


def test_settings_follow_config_and_refuse_unknown_variant():
    settings = planner.Settings.from_config({"variant": "qwen2-vl", "device": "cpu", "step_scale_m": "0.8"})
    assert (settings.variant, settings.device, settings.step_scale_m) == ("qwen2-vl", "cpu", 0.8)
    with pytest.raises(ValueError):
        planner.Settings.from_config({"variant": "qwen3-vl"})
    with pytest.raises(ValueError):
        planner.Settings.from_config({"step_scale_m": 0.0})
    with pytest.raises(ValueError):
        planner.Settings.from_config({"samples": 0})


def test_manifest_config_matches_the_settings_defaults():
    manifest = _manifest()
    settings = planner.Settings.from_config(manifest["config"])
    assert settings == planner.Settings(device="cuda")
    assert set(manifest["config"]) == set(planner.Settings.__dataclass_fields__)


def test_manifest_is_a_point_goal_waypoint_vla():
    manifest = _manifest()
    assert manifest["action_type"] == "waypoints"
    assert goal.declared(manifest) == frozenset({"pose"})
    config = chunk_config_from_manifest(manifest)
    assert config.execute_s == pytest.approx(1.0 / manifest["rate_hz"])
    assert config.speed_mps * config.execute_s == pytest.approx(manifest["config"]["step_scale_m"])
    assert manifest["observations"]["datasources"]["camera"]["params"]["output_size"] == [640, 480]


def test_weights_manifest_mirrors_both_checkpoints():
    entries = weights.read(_PLANNER_DIR)
    dirs = {pathlib.PurePosixPath(entry.dest).parent for entry in entries}
    assert dirs == {pathlib.PurePosixPath("model") / planner.VARIANTS[v][0] for v in planner.VARIANTS}
    assert all(entry.repo == "arena-rosnav/socialnav" for entry in entries)
    names = {pathlib.PurePosixPath(entry.dest).name for entry in entries}
    assert {"model.safetensors", "model-00001-of-00002.safetensors", "tokenizer.json", "config.json"} <= names


class _Recorder:
    """Stands in for Policy and Encoder to capture what step() feeds the model."""

    def __init__(self) -> None:
        self.calls: list = []
        self.token_ids: dict[str, int] = {}

    def batch(self, frames: list) -> dict:
        self.calls.append(("frames", len(frames)))
        return {}

    def predict(self, batch: dict, waypoints: np.ndarray, token_ids: dict, noise: torch.Tensor) -> np.ndarray:
        self.calls.append(("waypoints", waypoints.copy()))
        self.calls.append(("noise", noise.numpy().copy()))
        assert noise.shape == (16, planner.HORIZON, 2)
        return np.stack([_STRAIGHT] * 8 + [_RIGHT] * 8)


def _socialnav() -> tuple[planner.SocialNav, _Recorder]:
    recorder = _Recorder()
    return planner.SocialNav(recorder, recorder, planner.Settings(device="cpu")), recorder  # type: ignore[arg-type]


def test_step_holds_still_without_goal_camera_or_pose(frame: np.ndarray):
    nav, recorder = _socialnav()
    nav.reset(None)
    assert nav.step({"camera": frame, "robot_pose": np.zeros(3)}) == planner.Step()
    nav.reset({"goal_pose": _GOAL})
    assert nav.step({"robot_pose": np.zeros(3)}) == planner.Step()
    assert nav.step({"camera": frame, "robot_pose": None}) == planner.Step()
    assert nav.step({"camera": np.zeros((0,), dtype=np.uint8), "robot_pose": np.zeros(3)}) == planner.Step()
    assert recorder.calls == []


def test_step_feeds_five_frames_and_a_growing_history(frame: np.ndarray):
    nav, recorder = _socialnav()
    nav.reset({"goal_pose": _GOAL})
    for x in (0.0, 0.5):
        result = nav.step({"camera": frame, "robot_pose": np.array([x, 0.0, 0.0])})
        np.testing.assert_allclose(result.chunk, [[0.5 * k, 0.0] for k in range(1, 6)])
    assert [c for c in recorder.calls if c[0] == "frames"] == [("frames", 5), ("frames", 5)]
    first, second = (c[1] for c in recorder.calls if c[0] == "waypoints")
    np.testing.assert_allclose(first[:, 1], [0.0, 0.0, 0.0, 0.0, 0.0, 6.0], atol=1e-6)
    np.testing.assert_allclose(second[:, 1], [-1.0, -1.0, -1.0, -1.0, 0.0, 5.0], atol=1e-6)


def test_step_follows_the_draw_ending_closest_to_the_goal(frame: np.ndarray):
    nav, _ = _socialnav()
    nav.reset({"goal_pose": {"x": 3.0, "y": -3.0, "theta": 0.0}})
    result = nav.step({"camera": frame, "robot_pose": np.zeros(3)})
    np.testing.assert_allclose(result.chunk, [[0.5 * k, -0.5 * k] for k in range(1, 6)])


def test_noise_advances_per_tick_and_restarts_at_reset(frame: np.ndarray):
    nav, recorder = _socialnav()
    features = {"camera": frame, "robot_pose": np.zeros(3)}
    nav.reset({"goal_pose": _GOAL})
    nav.step(features)
    nav.step(features)
    nav.reset({"goal_pose": _GOAL})
    nav.step(features)
    first, second, again = (c[1] for c in recorder.calls if c[0] == "noise")
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(first, again)


def test_reset_drops_the_history(frame: np.ndarray):
    nav, recorder = _socialnav()
    nav.reset({"goal_pose": _GOAL})
    nav.step({"camera": frame, "robot_pose": np.array([1.0, 0.0, 0.0])})
    nav.reset({"goal_pose": _GOAL})
    nav.step({"camera": frame, "robot_pose": np.array([2.0, 0.0, 0.0])})
    waypoints = [c[1] for c in recorder.calls if c[0] == "waypoints"]
    np.testing.assert_allclose(waypoints[-1][:5], np.zeros((5, 2)), atol=1e-6)


def test_step_drops_the_alpha_channel(frame: np.ndarray):
    nav, _ = _socialnav()
    nav.reset({"goal_pose": _GOAL})
    rgba = np.concatenate([frame, np.full(frame.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)
    nav.step({"camera": rgba, "robot_pose": np.zeros(3)})
    assert nav.frames[-1].mode == "RGB"
    assert nav.frames[-1].size == (640, 480)


def test_steering_goal_prefers_the_plan_subgoal():
    goal_pose = {"x": 9.0, "y": 0.0, "theta": 0.0}
    assert planner.steering_goal({}, goal_pose) == goal_pose
    assert planner.steering_goal({"subgoal": None}, goal_pose) == goal_pose
    subgoal = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    assert planner.steering_goal({"subgoal": subgoal}, goal_pose) == {"x": 1.0, "y": 2.0, "theta": 0.5}


def test_manifest_steers_by_the_global_plan_subgoal():
    manifest = _manifest()
    assert manifest["depends"]["global_plan"] is True
    with pytest.raises(ValueError):
        goal.withhold_pose(manifest)
    sources = manifest["observations"]["datasources"]
    assert sources["subgoal"] == {"type": "SubgoalGenerator", "params": {"lookahead": 6.0}}
    assert sources["robot_pose_from_tf"]["type"] == "RobotPoseTFGenerator"
    assert sources["goal_pose"] == {"type": "geometry_msgs/PoseStamped", "params": {"topic": "goal_pose"}}
    assert sources["global_plan"] == {"type": "nav_msgs/Path", "params": {"topic": "plan"}}
    assert sources["costmap"] == {"type": "nav_msgs/OccupancyGrid", "params": {"topic": "global_costmap/costmap"}}
    names = list(sources)
    assert names.index("robot_pose_from_tf") < names.index("subgoal")
