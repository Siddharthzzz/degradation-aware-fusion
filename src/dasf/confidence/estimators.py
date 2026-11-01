"""Confidence estimators c_i(t) = P(measurement i is reliable | features).

A. RuleBased   -- robust z-scores against NOMINAL feature statistics, soft
                  ramps, AND over intrinsic checks / OR over cross-sensor
                  agreement checks. Needs no degraded training data.
B. lr          -- logistic regression (standardised features)
C. rf / gbm    -- random forest / histogram gradient boosting
D. mlp         -- 2x32 ReLU MLP
(Heavy neural baselines live in neural.py.)

All learned estimators are optionally Platt-calibrated on the validation split.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------ rule groups
# Each group is a list of (feature, direction); direction +1: large is bad,
# -1: small is bad, 0: far from nominal median in either direction is bad.
# Group confidence = MAX over members (agreement with ANY other sensor is
# enough), sensor confidence = MIN over groups.
# Design note (v2, fixed before any full-study run): scene-dependent
# "availability" features (scan coverage, largest gap, valid-return ratio,
# corner count, correlation-peak ratio) vary strongly with the ENVIRONMENT
# (e.g. the open hall W3) and made the v1 rules reject nominal data in the
# pipeline smoke test.  Rules therefore use only measurement-CONSISTENCY
# checks; the learned models still see every feature.
RULES = {
    "cam": [[("sharpness_lapvar", -1)], [("noise_est", +1)], [("vo_resid", +1)],
            [("vo_valid_frac", -1)], [("vo_lat_vel", +1)], [("vo_yaw_disagree", +1)],
            [("d_w_gyro", +1), ("d_w_odo", +1)]],
    "lidar": [[("roughness", +1)], [("icp_fitness", +1)], [("icp_inlier_ratio", -1)],
              [("icp_corr_trans", +1)], [("icp_corr_rot", +1)]],
    "lidar_rel": [[("roughness", +1)], [("icp_fitness", +1)], [("icp_inlier_ratio", -1)],
                  [("d_w_gyro", +1), ("d_w_odo", +1)]],
    "imu": [[("gyro_hf", +1)], [("acc_hf", +1)], [("stuck_frac", +1)],
            [("d_w_odo", +1), ("d_w_cam", +1)]],
    "odom": [[("v_jerk", +1)], [("zero_while_turning", +1)],
             [("d_w_gyro", +1), ("d_w_cam", +1)], [("d_v_cam", +1), ("d_acc", +1)]],
}


class RuleBased:
    kind = "rule"
    feature_set = "static"

    def __init__(self, sensor, names, z0=3.0, scale=3.0):
        self.sensor, self.names, self.z0, self.scale = sensor, list(names), z0, scale
        self.med = self.mad = None

    def fit(self, X_nominal, y=None):
        self.med = np.median(X_nominal, axis=0)
        mad = 1.4826 * np.median(np.abs(X_nominal - self.med), axis=0)
        # Saturated features (e.g. valid fraction == 1 in nominal data) have
        # MAD = 0; floor the scale so a tiny deviation is not an outlier.
        self.mad = np.maximum.reduce([mad, 0.25 * X_nominal.std(axis=0), 0.02 * np.abs(self.med),
                                      np.full_like(mad, 1e-3)])
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        z = (X - self.med) / self.mad
        c = np.ones(len(X))
        for group in RULES[self.sensor]:
            gc = np.zeros(len(X))
            for name, d in group:
                j = self.names.index(name)
                zz = np.abs(z[:, j]) if d == 0 else d * z[:, j]
                gc = np.maximum(gc, np.exp(-np.maximum(0.0, zz - self.z0) ** 2 / (2 * self.scale ** 2)))
            c = np.minimum(c, gc)
        return c

    def n_params(self):
        return int(2 * len(self.names))


# ------------------------------------------------------------------ learned
def _make(family, seed):
    if family == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    if family == "rf":
        return RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5,
                                      n_jobs=1, random_state=seed)
    if family == "gbm":
        return HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1,
                                              random_state=seed)
    if family == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 32),
                                                             alpha=1e-4, max_iter=300,
                                                             early_stopping=True, random_state=seed))
    raise ValueError(family)


class Platt:
    """1-D logistic recalibration on the logit of the raw score."""

    def __init__(self):
        self.a, self.b = 1.0, 0.0

    def fit(self, p, y):
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        lr = LogisticRegression(C=1e4).fit(z[:, None], y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def __call__(self, p):
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        return 1 / (1 + np.exp(-(self.a * z + self.b)))


class Learned:
    def __init__(self, sensor, family, feature_set, names, seed=0):
        self.sensor, self.kind, self.feature_set, self.names = sensor, family, feature_set, list(names)
        self.model = _make(family, seed)
        self.calib = None
        self.const = None

    def fit(self, X, y, max_n=200_000, seed=0):
        if len(np.unique(y)) < 2:              # degenerate split -> constant predictor
            self.const = float(np.mean(y))
            return self
        if len(X) > max_n:
            idx = np.random.default_rng(seed).choice(len(X), max_n, replace=False)
            X, y = X[idx], y[idx]
        self.model.fit(X, y)
        return self

    def calibrate(self, X, y):
        if self.const is None and len(np.unique(y)) == 2:
            self.calib = Platt().fit(self._raw(X), y)
        return self

    def _raw(self, X):
        if self.const is not None:
            return np.full(len(np.atleast_2d(X)), self.const)
        return self.model.predict_proba(np.atleast_2d(X))[:, 1]

    def predict(self, X):
        p = self._raw(X)
        return self.calib(p) if self.calib is not None else p

    def n_params(self):
        m = self.model
        if self.const is not None:
            return 1
        if self.kind == "lr":
            lr = m[-1]
            return int(lr.coef_.size + lr.intercept_.size + 2 * lr.coef_.size)   # + scaler
        if self.kind == "mlp":
            mlp = m[-1]
            return int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
                       + 2 * mlp.coefs_[0].shape[0])
        if self.kind == "rf":
            return int(sum(e.tree_.node_count for e in m.estimators_) * 2)          # threshold + value/feature
        if self.kind == "gbm":
            return int(sum(len(p.nodes) for it in m._predictors for p in it) * 2)
        return -1


def save(est, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(est, f)


def load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def model_size_bytes(est) -> int:
    return len(pickle.dumps(est))


def single_sample_latency(est, x, n=2000, warmup=100) -> dict:
    """Median/p95 latency of predict() on ONE sample (the deployment case)."""
    x = np.atleast_2d(x)
    for _ in range(warmup):
        est.predict(x)
    ts = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        est.predict(x)
        ts[i] = time.perf_counter() - t0
    return dict(median_us=float(np.median(ts) * 1e6), p95_us=float(np.quantile(ts, 0.95) * 1e6))
