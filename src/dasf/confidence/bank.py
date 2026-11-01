"""Loads trained estimators and serves confidences to the fusion runner.

Camera, IMU and odometry features are method-independent, so their
confidences are computed in batch per episode.  LiDAR features include the
ICP result, which depends on the filter's predicted pose, so LiDAR confidence
is computed in the loop (streaming temporal features).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import estimators as E
from .features import (LIDAR_SCAN_FEATURES, RATES, Temporal, augment_batch,
                       lidar_scan_feature_matrix, sensor_features)
from ..sensors.frontends import ICP_FEATURES

SENSORS = ["cam", "lidar", "imu", "odom"]
LIDAR_NAMES = LIDAR_SCAN_FEATURES + ICP_FEATURES


def est_path(models_dir: Path, sensor: str, name: str) -> Path:
    return Path(models_dir) / sensor / f"{name}.pkl"


class ConfidenceBank:
    def __init__(self, cfg, models_dir: Path, heavy_dir: Path | None = None):
        self.cfg, self.dir, self.heavy_dir = cfg, Path(models_dir), heavy_dir
        self._cache = {}

    def get(self, sensor, name):
        key = (sensor, name)
        if key not in self._cache:
            self._cache[key] = E.load(est_path(self.dir, sensor, name))
        return self._cache[key]

    # -------------------------------------------------------------- batch sensors
    def precompute(self, ep: dict, name: str, feats: dict | None = None) -> dict:
        """sensor -> confidence per delivered sample (cam/imu/odom)."""
        feats = feats or sensor_features(ep)
        out = {}
        for s in ("cam", "imu", "odom", "lidar_rel"):
            f = feats[s]
            if len(f["X"]) == 0:
                out[s] = np.zeros(0)
                continue
            if name == "cnn" and s == "cam":
                out[s] = self._heavy_cam(ep)
                continue
            est = self.get(s, name)
            if name == "cnn" and s == "lidar_rel":
                j = ep["lrel_scan"]
                out[s] = np.clip(est.predict(np.stack([ep["lidar_ranges"][j - 1], ep["lidar_ranges"][j]], 1)), 0, 1)
                continue
            X = f["X"]
            if est.feature_set == "temporal":
                X = augment_batch(X, f["t"], RATES[s], self.cfg)
            out[s] = np.clip(est.predict(X), 0.0, 1.0)
        return out

    def _heavy_cam(self, ep):
        from ..utils import stable_int
        m = ep["meta"]
        key = f"{m['world']}__{m['traj']}__s{m['seed']}__{stable_int(str([e.to_dict() for e in m['events']]))}"
        p = Path(self.heavy_dir) / f"{key}.npz"
        if not p.exists():
            raise FileNotFoundError(f"heavy camera confidences missing: {p} "
                                    "(run experiments/05_heavy_camera_pass.py)")
        z = np.load(p)
        return np.interp(ep["cam_t"], z["t"], z["c"])

    # -------------------------------------------------------------- LiDAR (in-loop)
    def lidar_stream(self, name: str, ep: dict):
        return LidarConfidence(self, name, ep)


class LidarConfidence:
    def __init__(self, bank, name, ep):
        self.name = name
        self.scan_X = lidar_scan_feature_matrix(ep)
        self.ranges = ep["lidar_ranges"]
        self.est = bank.get("lidar", name)
        c = bank.cfg["confidence"]
        self.temporal = Temporal(len(LIDAR_NAMES), RATES["lidar"], c["temporal_tau"], c["temporal_window"]) \
            if getattr(self.est, "feature_set", "static") == "temporal" else None

    def __call__(self, j: int, icp_feats: dict, t: float) -> float:
        if self.name == "cnn":
            return float(self.est.predict(self.ranges[j:j + 1])[0])
        x = np.concatenate([self.scan_X[j], [icp_feats[k] for k in ICP_FEATURES]])
        if self.temporal is not None:
            x = self.temporal.update(x, t)
        return float(np.clip(self.est.predict(x[None, :])[0], 0.0, 1.0))
