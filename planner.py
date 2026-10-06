"""SocialNav wrapper for the arena_planners bridge."""

from __future__ import annotations

import dataclasses
import logging
import math
import pathlib

import numpy as np
import torch
from arena_planners.sdk import Step, load_manifest, main_loop
from PIL import Image
from transformers import AutoProcessor, ProcessorMixin

from socialnav_model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from socialnav_model.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

_DIR = pathlib.Path(__file__).parent
_FETCH_HINT = "run `arena feature planners add socialnav` (or `python -m arena_planners fetch socialnav`)"

VARIANTS: dict[str, tuple[str, type]] = {
    "qwen2-vl": ("SocialNav-Qwen2-VL-SAFE-GRPO", Qwen2VLForConditionalGeneration),
    "qwen2.5-vl": ("SocialNav-Qwen2.5-VL-SAFE-GRPO", Qwen2_5_VLForConditionalGeneration),
}
CONTEXT = 5
HORIZON = 5
WAYPOINT_TOKENS = tuple(f"<input_pos{i}>" for i in range(1, CONTEXT + 1)) + ("<input_target>",)
EXPERT_KWARGS: dict[str, object] = {
    "action_dim": 2,
    "action_chunk": HORIZON,
    "flow_matching_policy": True,
    "num_flow_steps": 5,
    "action_former": True,
    "query_action_layer": 4,
    "sigma": 0.0,
    "ar_lambda_loss": 1.0,
    "min_value": -1.25,
    "max_value": 1.25,
    "sde_mode": "cps",
}
DEFAULT_PROMPT = (
    "You are a robot navigating to a target point. The images are your last five observations, the last one "
    "being the current view. Your positions at those observations were "
    "<input_pos1><input_pos2><input_pos3><input_pos4><input_pos5> and the target is at <input_target>. "
    "Predict the next five waypoints."
)
_NOISE_SEED = 0

_log = logging.getLogger("socialnav")


def to_model_frame(robot: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map-frame `(N, 2)` points in the robot frame of `robot` `[x, y, theta]`, model axes: x right, y forward."""
    rx, ry, rtheta = (float(v) for v in robot[:3])
    delta = np.asarray(points, dtype=np.float64).reshape(-1, 2) - (rx, ry)
    fwd = delta[:, 0] * math.cos(rtheta) + delta[:, 1] * math.sin(rtheta)
    left = -delta[:, 0] * math.sin(rtheta) + delta[:, 1] * math.cos(rtheta)
    return np.stack((-left, fwd), axis=1)


def clip_radius(point: np.ndarray, radius: float) -> np.ndarray:
    norm = float(np.hypot(*point))
    return point * (radius / norm) if norm > radius else point


def padded(items: list, size: int) -> list:
    """The last `size` items, the oldest repeated in front when fewer exist."""
    kept = items[-size:]
    return [kept[0]] * (size - len(kept)) + kept


def steering_goal(features: dict, goal: dict) -> dict:
    """The global plan's lookahead point when the bridge sends one as `subgoal`, else the episode goal."""
    subgoal = features.get("subgoal")
    if subgoal is None:
        return goal
    x, y, theta = (float(v) for v in np.asarray(subgoal, dtype=np.float64)[:3])
    return {"x": x, "y": y, "theta": theta}


def input_waypoints(poses: list[np.ndarray], goal: dict, step_scale: float, goal_clip: float) -> np.ndarray:
    """Model `input_waypoints` `(6, 2)`: the last 5 poses then the goal, in the current robot frame, in steps."""
    history = padded(poses, CONTEXT)
    current = history[-1]
    positions = to_model_frame(current, np.array([p[:2] for p in history]))
    target = clip_radius(to_model_frame(current, np.array([[goal["x"], goal["y"]]]))[0], goal_clip)
    return (np.vstack((positions, target)) / step_scale).astype(np.float32)


def to_chunk(waypoints: np.ndarray, step_scale: float) -> list[list[float]]:
    """Waypoints `[x, y]` in meters (x forward, y left) from the model output `(5, 2)` in steps (x right, y forward)."""
    wp = np.asarray(waypoints, dtype=np.float64).reshape(-1, 2) * step_scale
    return np.stack((wp[:, 1], -wp[:, 0]), axis=1).tolist()


def closest_to(paths: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The path of `paths` `(N, 5, 2)` whose last waypoint lies closest to `target`."""
    return paths[int(np.linalg.norm(paths[:, -1] - target, axis=1).argmin())]


def check_prompt(prompt: str) -> str:
    """`prompt` with every waypoint placeholder present once, in order."""
    positions = [prompt.find(token) for token in WAYPOINT_TOKENS]
    if any(p < 0 for p in positions) or positions != sorted(positions):
        raise ValueError(f"socialnav: prompt must hold {' '.join(WAYPOINT_TOKENS)} in this order, got {prompt!r}")
    if any(prompt.count(token) != 1 for token in WAYPOINT_TOKENS):
        raise ValueError(f"socialnav: each waypoint placeholder must appear once in {prompt!r}")
    return prompt


def model_dir(variant: str) -> pathlib.Path:
    if variant not in VARIANTS:
        raise ValueError(f"socialnav: variant {variant!r} not in {sorted(VARIANTS)}")
    return _DIR / "model" / VARIANTS[variant][0]


@dataclasses.dataclass(frozen=True)
class Settings:
    """Per-launch planner config: model variant and device, step length, goal clip and the text prompt."""

    variant: str = "qwen2.5-vl"
    device: str = "cuda"
    step_scale_m: float = 0.5
    goal_clip_m: float = 10.0
    samples: int = 16
    prompt: str = DEFAULT_PROMPT

    @classmethod
    def from_config(cls, config: dict) -> Settings:
        settings = cls(
            variant=str(config.get("variant", cls.variant)),
            device=str(config.get("device", cls.device)),
            step_scale_m=float(config.get("step_scale_m", cls.step_scale_m)),
            goal_clip_m=float(config.get("goal_clip_m", cls.goal_clip_m)),
            samples=int(config.get("samples", cls.samples)),
            prompt=check_prompt(str(config.get("prompt", cls.prompt))),
        )
        model_dir(settings.variant)
        if settings.step_scale_m <= 0.0 or settings.goal_clip_m <= 0.0 or settings.samples < 1:
            raise ValueError(f"socialnav: step_scale_m and goal_clip_m above 0, samples at least 1: {settings}")
        return settings


class Encoder:
    """Upstream `citywalker.py`'s chat template and processor call for one five-frame observation."""

    def __init__(self, processor: ProcessorMixin, prompt: str) -> None:
        self.processor = processor
        self.prompt = check_prompt(prompt)
        tokenizer = processor.tokenizer
        self.token_ids = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in WAYPOINT_TOKENS}
        ids = [self.token_ids[token] for token in WAYPOINT_TOKENS]
        if tokenizer.unk_token_id in ids or ids != list(range(ids[0], ids[0] + len(ids))):
            raise ValueError(f"socialnav: waypoint placeholder ids must exist and be contiguous, got {ids}")
        content = [{"type": "image"}] * CONTEXT + [{"type": "text", "text": self.prompt}]
        self.text = (
            processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
            + "<|im_end|>"
        )

    def batch(self, frames: list[Image.Image]) -> dict[str, torch.Tensor]:
        if len(frames) != CONTEXT:
            raise ValueError(f"socialnav: expected {CONTEXT} frames, got {len(frames)}")
        return dict(self.processor(text=[self.text], images=frames, padding=True, return_tensors="pt"))


def load_processor(variant: str) -> ProcessorMixin:
    path = model_dir(variant)
    if not (path / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"socialnav tokenizer missing at {path}, {_FETCH_HINT}")
    return AutoProcessor.from_pretrained(path)


class Policy:
    """A SocialNav checkpoint: Qwen-VL brain with the flow-matching action expert, Euler-integrated in 5 steps."""

    def __init__(self, variant: str, device: torch.device) -> None:
        path = model_dir(variant)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"socialnav weights missing at {path}, {_FETCH_HINT}")
        self.device = device
        self.model, info = VARIANTS[variant][1].from_pretrained(
            path, dtype=torch.bfloat16, output_loading_info=True, **EXPERT_KWARGS
        )
        if info["missing_keys"]:
            raise RuntimeError(f"socialnav: checkpoint lacks {sorted(info['missing_keys'])[:5]}...")
        self.model.to(device).eval()

    @torch.inference_mode()
    def predict(
        self, batch: dict[str, torch.Tensor], waypoints: np.ndarray, token_ids: dict[str, int], noise: torch.Tensor
    ) -> np.ndarray:
        """Paths `(N, 5, 2)` in steps (x right, y forward), one per initial noise of `noise` `(N, 5, 2)`."""
        output = self.model(
            **{k: v.to(self.device) for k, v in batch.items()},
            input_waypoints=torch.from_numpy(waypoints).unsqueeze(0).to(self.device),
            train=False,
            train_branch="fm",
            num_samples=noise.shape[0],
            special_token2id=token_ids,
            select_noise=noise.to(self.device),
        )
        return output[0].float().cpu().numpy()


class SocialNav:
    def __init__(self, policy: Policy, encoder: Encoder, settings: Settings) -> None:
        self.policy = policy
        self.encoder = encoder
        self.settings = settings
        self.reset(None)

    def reset(self, initial_state: dict | None) -> None:
        state = initial_state or {}
        self.goal: dict | None = state.get("goal_pose")
        self.frames: list[Image.Image] = []
        self.generator = torch.Generator().manual_seed(_NOISE_SEED)
        self.poses: list[np.ndarray] = []

    def observe(self, features: dict) -> bool:
        """Records the tick's frame and pose, False when either is missing."""
        image, pose = features.get("camera"), features.get("robot_pose")
        if image is None or pose is None or np.asarray(image).ndim != 3:
            return False
        rgb = np.ascontiguousarray(np.asarray(image)[..., :3], dtype=np.uint8)
        self.frames = (self.frames + [Image.fromarray(rgb)])[-CONTEXT:]
        self.poses = (self.poses + [np.asarray(pose, dtype=np.float64)[:3]])[-CONTEXT:]
        return True

    def step(self, features: dict) -> Step:
        if not self.observe(features) or self.goal is None:
            return Step()
        goal = steering_goal(features, self.goal)
        waypoints = input_waypoints(self.poses, goal, self.settings.step_scale_m, self.settings.goal_clip_m)
        batch = self.encoder.batch(padded(self.frames, CONTEXT))
        noise = torch.randn(self.settings.samples, HORIZON, 2, generator=self.generator)
        paths = self.policy.predict(batch, waypoints, self.encoder.token_ids, noise)
        return Step(chunk=to_chunk(closest_to(paths, waypoints[-1]), self.settings.step_scale_m))


_planner: SocialNav | None = None


def on_init(config: dict) -> None:
    global _planner
    settings = Settings.from_config(config)
    device = torch.device(settings.device)
    policy = Policy(settings.variant, device)
    _planner = SocialNav(policy, Encoder(load_processor(settings.variant), settings.prompt), settings)
    _log.warning("socialnav: loaded %s on %s", settings.variant, device)


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    assert _planner is not None
    _planner.reset(initial_state)


def step(features: dict) -> Step:
    assert _planner is not None
    return _planner.step(features)


if __name__ == "__main__":
    logging.basicConfig()
    manifest = load_manifest(_DIR / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset, on_init=on_init)
