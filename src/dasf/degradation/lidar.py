"""2-D LiDAR degradation models (applied to a raw range scan; inf = no return)."""
from __future__ import annotations

import numpy as np


def apply_lidar(ranges: np.ndarray, angles: np.ndarray, deg: dict, rng) -> np.ndarray:
    r = ranges.copy()
    if "blackout" in deg:
        r[:] = np.inf
        return r
    if "rangenoise" in deg:
        r = r + deg["rangenoise"] * rng.standard_normal(r.shape)
    if "dropout" in deg:
        r[rng.random(r.shape) < deg["dropout"]] = np.inf
    if "fov" in deg:
        half = np.deg2rad(deg["fov"]) / 2
        centre = np.pi / 2                       # obstruction on the robot's left side
        d = np.abs((angles - centre + np.pi) % (2 * np.pi) - np.pi)
        r[d < half] = np.inf
    if "sparse" in deg:
        k = int(deg["sparse"])
        keep = np.zeros_like(r, dtype=bool)
        keep[::k] = True
        r[~keep] = np.inf
    return r
