"""Ground-truth reliability labels and nominal-noise calibration.

A measurement is labelled RELIABLE (y = 1) when its error w.r.t. ground truth
is consistent with the nominal measurement model:

    d^2 = e^T R0^{-1} e  <=  chi2_m(p)       (p = 0.95 by default)

where e = z - h(x_gt) and R0 is the nominal covariance calibrated on NOMINAL
training episodes only. A second label, `injected`, marks whether a
degradation was being injected; it is used for detection AUROC but not for
training (injection at mild severity can still yield consistent data).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from ..utils import wrap


def sensor_errors(ep: dict, sensor: str, idx=None) -> np.ndarray:
    if sensor == "cam":
        e = np.column_stack([ep["cam_v"] - ep["cam_v_true"], wrap(ep["cam_w"] - ep["cam_w_true"])])
    elif sensor == "lidar_rel":
        e = np.column_stack([ep["lrel_v"] - ep["lrel_v_true"], wrap(ep["lrel_w"] - ep["lrel_w_true"])])
    elif sensor == "odom":
        e = np.column_stack([ep["odom_v"] - ep["odom_v_true"], ep["odom_w"] - ep["odom_w_true"]])
    elif sensor == "imu":
        e = np.column_stack([ep["imu_gyro"] - ep["gt_w"] - ep["imu_bg_nominal"],
                             ep["imu_acc"][:, 0] - ep["gt_a"],
                             ep["imu_acc"][:, 1] - ep["gt_v"] * ep["gt_w"]])
    else:
        raise ValueError(sensor)
    return e if idx is None else e[idx]


def lidar_error(z, gt_pose):
    return np.array([z[0] - gt_pose[0], z[1] - gt_pose[1], float(wrap(z[2] - gt_pose[2]))])


def mahalanobis2(e: np.ndarray, r0) -> np.ndarray:
    e = np.atleast_2d(e)
    return np.sum(e * e / np.asarray(r0)[None, :], axis=1)


def reliable_label(e, r0, p=0.95):
    e = np.atleast_2d(e)
    d2 = mahalanobis2(e, r0)
    return (d2 <= chi2.ppf(p, e.shape[1])).astype(int), d2


def robust_diag_cov(E: np.ndarray, trim: float = 0.01) -> list:
    """Per-axis variance after trimming the most extreme `trim` fraction."""
    out = []
    for j in range(E.shape[1]):
        x = E[:, j][np.isfinite(E[:, j])]
        lo, hi = np.quantile(x, [trim / 2, 1 - trim / 2])
        x = x[(x >= lo) & (x <= hi)]
        out.append(float(np.var(x) + 1e-12))
    return out
