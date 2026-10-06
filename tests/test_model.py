from __future__ import annotations

import pathlib
import time

import numpy as np
import pytest
import torch
import yaml
from arena_planners.sdk import Step

import planner
from scene import street_frame

_DIR = pathlib.Path(planner.__file__).parent


def _missing_weights(variant: str) -> list[str]:
    with open(_DIR / "weights.yaml") as fh:
        dests = [entry["dest"] for entry in yaml.safe_load(fh)["files"]]
    folder = f"model/{planner.VARIANTS[variant][0]}/"
    return [dest for dest in dests if dest.startswith(folder) and not (_DIR / dest).is_file()]


@pytest.fixture(scope="module", params=sorted(planner.VARIANTS))
def socialnav(request: pytest.FixtureRequest) -> planner.SocialNav:
    variant = request.param
    missing = _missing_weights(variant)
    if missing:
        pytest.skip(f"{variant} weights {missing[:2]}... missing, run `arena feature planners add socialnav`")
    if not torch.cuda.is_available():
        pytest.skip("the model runs in bfloat16 on a CUDA device")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    planner.on_init({"device": "cuda", "variant": variant})
    allocated = torch.cuda.memory_allocated() / 2**30
    print(f"\n{variant} load {time.perf_counter() - start:.1f} s, {allocated:.2f} GiB allocated")
    assert planner._planner is not None
    return planner._planner


def _timed_step(features: dict, label: str) -> Step:
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = planner.step(features)
    torch.cuda.synchronize()
    print(f"{label} step {1000 * (time.perf_counter() - start):.0f} ms")
    return result


def _chunk(result: Step) -> np.ndarray:
    assert isinstance(result, Step)
    assert not result.action and not result.command and not result.signal
    chunk = np.asarray(result.chunk)
    assert chunk.shape == (5, 2)
    assert np.isfinite(chunk).all()
    return chunk


def _drive(nav: planner.SocialNav, goal: dict, ticks: int = 5) -> np.ndarray:
    """Drives straight at 0.5 m per tick, the camera lane centered, and returns the last chunk in meters."""
    planner.on_reset("ep", {"goal_pose": goal})
    frame = street_frame()
    result = Step()
    for k in range(ticks):
        result = _timed_step({"camera": frame, "robot_pose": np.array([0.5 * k, 0.0, 0.0])}, f"tick {k}")
    chunk = _chunk(result)
    np.set_printoptions(precision=3, suppress=True)
    print(f"goal {goal} chunk (m):\n{chunk}")
    return chunk


def test_goal_ahead_drives_forward(socialnav: planner.SocialNav):
    chunk = _drive(socialnav, {"x": 12.0, "y": 0.0, "theta": 0.0})
    assert chunk[-1, 0] > 1.0
    assert abs(chunk[-1, 1]) < chunk[-1, 0]
    assert np.all(np.diff(chunk[:, 0]) > 0)


def test_goal_side_orders_the_lateral_offset(socialnav: planner.SocialNav):
    left = _drive(socialnav, {"x": 6.0, "y": 6.0, "theta": 0.0})
    right = _drive(socialnav, {"x": 6.0, "y": -6.0, "theta": 0.0})
    assert left[-1, 1] - right[-1, 1] > 0.2
    assert left[-1, 0] > 0.5 and right[-1, 0] > 0.5


def test_same_observation_gives_same_chunk(socialnav: planner.SocialNav):
    planner.on_reset("ep", {"goal_pose": {"x": 12.0, "y": 0.0, "theta": 0.0}})
    features = {"camera": street_frame(), "robot_pose": np.zeros(3)}
    first = _chunk(planner.step(features))
    planner.on_reset("ep", {"goal_pose": {"x": 12.0, "y": 0.0, "theta": 0.0}})
    np.testing.assert_array_equal(first, _chunk(planner.step(features)))


def test_steady_state_latency_and_memory(socialnav: planner.SocialNav):
    planner.on_reset("ep", {"goal_pose": {"x": 12.0, "y": 0.0, "theta": 0.0}})
    features = {"camera": street_frame(), "robot_pose": np.zeros(3)}
    planner.step(features)
    latencies = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        planner.step(features)
        torch.cuda.synchronize()
        latencies.append(1000 * (time.perf_counter() - start))
    peak = torch.cuda.max_memory_allocated() / 2**30
    variant = socialnav.settings.variant
    print(f"{variant} steady step {np.mean(latencies):.0f} ms (min {min(latencies):.0f}), peak {peak:.2f} GiB")
    assert max(latencies) < 5000


def test_no_goal_holds_still(socialnav: planner.SocialNav):
    planner.on_reset("ep", None)
    assert planner.step({"camera": street_frame(), "robot_pose": np.zeros(3)}) == Step()
