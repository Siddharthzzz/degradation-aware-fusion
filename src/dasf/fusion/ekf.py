"""Planar EKF with confidence-scaled measurement covariances.

State x = [p_x, p_y, theta, v, omega, a, b_g]  (b_g = gyro bias).
Process model (constant acceleration / constant turn rate):
    p_x' = p_x + v cos(theta) dt,  p_y' = p_y + v sin(theta) dt,
    theta' = theta + omega dt,     v' = v + a dt,   omega, a, b_g random walks.

Confidence-aware update (Section VII of the paper): measurement i with
nominal covariance R_i^0 and weight w_i = g(c_i) in (0, 1] is fused with
    R_i^eff = R_i^0 / w_i ,
which is exactly the one-step Gauss-Newton solution of the weighted MAP
problem  min_x ||x - x^-||^2_{P^-} + sum_i w_i ||z_i - h_i(x)||^2_{R_i^0}.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from ..utils import wrap

NX = 7
PX, PY, TH, V, W, A, BG = range(NX)


class EKF:
    def __init__(self, x0, cfg):
        f = cfg["fusion"]
        self.x = np.array(x0, float)
        p0 = f["p0"]
        self.P = np.diag([p0["pos"], p0["pos"], p0["th"], p0["v"], p0["w"], p0["a"], p0["bg"]])
        q = f["q"]
        self.qd = np.array([q["pos"], q["pos"], q["th"], q["v"], q["w"], q["a"], q["bg"]])
        self.I = np.eye(NX)

    def predict(self, dt):
        if dt <= 0:
            return
        x = self.x
        c, s = np.cos(x[TH]), np.sin(x[TH])
        F = self.I.copy()
        F[PX, TH] = -x[V] * s * dt
        F[PX, V] = c * dt
        F[PY, TH] = x[V] * c * dt
        F[PY, V] = s * dt
        F[TH, W] = dt
        F[V, A] = dt
        x[PX] += x[V] * c * dt
        x[PY] += x[V] * s * dt
        x[TH] = wrap(x[TH] + x[W] * dt)
        x[V] += x[A] * dt
        self.P = F @ self.P @ F.T + np.diag(self.qd * dt)

    def innovation(self, z, hx, H, R, ang=None):
        y = np.asarray(z, float) - hx
        if ang is not None:
            y[ang] = wrap(y[ang])
        S = H @ self.P @ H.T + R
        return y, S

    def update(self, z, hx, H, R, ang=None):
        y, S = self.innovation(z, hx, H, R, ang)
        Sinv = np.linalg.inv(S)
        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ y
        self.x[TH] = wrap(self.x[TH])
        IKH = self.I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T          # Joseph form
        return float(y @ Sinv @ y), y


# ---------------------------------------------------------------- measurement models
def h_velocity(x):
    """odometry / camera VO: z = [v, omega]."""
    H = np.zeros((2, NX))
    H[0, V] = 1.0
    H[1, W] = 1.0
    return x[[V, W]].copy(), H


def h_imu(x):
    """z = [gyro, a_x, a_y] = [omega + b_g, a, v*omega]."""
    H = np.zeros((3, NX))
    H[0, W] = 1.0
    H[0, BG] = 1.0
    H[1, A] = 1.0
    H[2, V] = x[W]
    H[2, W] = x[V]
    return np.array([x[W] + x[BG], x[A], x[V] * x[W]]), H


def h_pose(x):
    H = np.zeros((3, NX))
    H[0, PX] = H[1, PY] = H[2, TH] = 1.0
    return x[[PX, PY, TH]].copy(), H


def chi2_gate(m, p):
    return chi2.ppf(p, m)
