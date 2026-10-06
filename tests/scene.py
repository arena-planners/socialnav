from __future__ import annotations

import numpy as np


def street_frame(shift: float = 0.0, size: tuple[int, int] = (480, 640)) -> np.ndarray:
    """A synthetic street view: sky, sidewalk vanishing ahead, buildings left and right, `shift` slides the lane."""
    height, width = size
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    cols = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    horizon = 0.45
    sky = np.stack([0.55 + 0.3 * (1 - rows), 0.7 + 0.2 * (1 - rows), 0.95 * np.ones_like(rows)], axis=-1)
    sky = np.broadcast_to(sky, (height, width, 3))
    depth = np.clip((rows - horizon) / (1 - horizon), 1e-3, 1.0)
    lane = np.abs(cols - shift) < 0.35 * depth + 0.03
    ground = np.where(lane[..., None], np.array([0.62, 0.6, 0.58]), np.array([0.3, 0.45, 0.25]))
    ground = ground * (0.7 + 0.3 * depth[..., None])
    image = np.where(rows[..., None] < horizon, sky, ground)
    walls = (np.abs(cols) > 0.55 + 0.4 * np.clip(rows - 0.2, 0, 1)) & (rows[..., None][..., 0] < horizon + 0.15)
    image = np.where(walls[..., None], np.array([0.5, 0.42, 0.38]) * (0.6 + 0.4 * rows[..., None]), image)
    noise = np.random.default_rng(0).normal(0.0, 0.02, image.shape).astype(np.float32)
    return (np.clip(image + noise, 0.0, 1.0) * 255).astype(np.uint8)
