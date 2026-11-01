"""Interpretable, causal sensor-quality features.

Per sensor we build a *base* feature vector at every measurement time; the
*temporal* feature set appends, for each base feature, an exponential moving
average and a rolling standard deviation over the sensor's own recent samples
plus the sample staleness (Ablations 8/9).

All features are causal: cross-sensor features use a zero-order hold of the
other sensor's most recent sample (never future samples).
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy import ndimage

from ..utils import zoh

# ============================================================================ camera image
CAM_IMAGE_FEATURES = ["brightness", "contrast", "sharpness_lapvar", "entropy", "edge_density",
                      "corner_count", "dark_frac", "sat_frac", "noise_est", "frame_diff"]


def camera_image_features(img: np.ndarray, prev: np.ndarray | None) -> dict:
    lap = ndimage.laplace(img)
    hist, _ = np.histogram(img, bins=64, range=(0, 1))
    p = hist / hist.sum()
    p = p[p > 0]
    gx, gy = ndimage.sobel(img, 1), ndimage.sobel(img, 0)
    mag = np.hypot(gx, gy)
    # Harris corner response (Harris & Stephens 1988)
    Sxx = ndimage.gaussian_filter(gx * gx, 1.5)
    Syy = ndimage.gaussian_filter(gy * gy, 1.5)
    Sxy = ndimage.gaussian_filter(gx * gy, 1.5)
    R = Sxx * Syy - Sxy ** 2 - 0.04 * (Sxx + Syy) ** 2
    peaks = (R == ndimage.maximum_filter(R, size=5)) & (R > 1e-3)
    hp = img - ndimage.gaussian_filter(img, 1.0)
    return dict(
        brightness=float(img.mean()),
        contrast=float(img.std()),
        sharpness_lapvar=float(np.log10(lap.var() + 1e-9)),
        entropy=float(-(p * np.log2(p)).sum()),
        edge_density=float((mag > 0.5).mean()),
        corner_count=float(np.log1p(peaks.sum())),
        dark_frac=float((img < 0.03).mean()),
        sat_frac=float((img > 0.97).mean()),
        noise_est=float(1.4826 * np.median(np.abs(hp - np.median(hp)))),
        frame_diff=float(np.abs(img - prev).mean()) if prev is not None else 0.0,
    )


# ============================================================================ LiDAR scan
LIDAR_SCAN_FEATURES = ["valid_ratio", "range_mean", "range_std", "coverage", "max_gap_deg",
                       "roughness", "temporal_change"]


def lidar_scan_features(r: np.ndarray, prev: np.ndarray | None) -> dict:
    ok = np.isfinite(r)
    n = len(r)
    rv = r[ok]
    sectors = ok.reshape(12, -1).sum(1) >= 3
    # longest circular run of invalid beams
    if ok.all():
        best = 0
    elif not ok.any():
        best = n
    else:
        k = int(np.argmax(ok))                       # rotate so the scan starts on a valid beam
        inv = np.concatenate([~np.roll(ok, -k), [False]]).astype(np.int8)
        edges = np.diff(np.concatenate([[0], inv]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        best = int((ends - starts).max()) if len(starts) else 0
    if len(rv) >= 3:
        with np.errstate(invalid="ignore"):          # inf - inf on missing beams -> filtered below
            d2 = np.abs(np.diff(r, 2))
        d2 = d2[np.isfinite(d2)]
        rough = float(np.median(d2)) if len(d2) else 0.5
    else:
        rough = 0.5
    if prev is not None:
        both = ok & np.isfinite(prev)
        tchg = float(np.median(np.abs(r[both] - prev[both]))) if both.sum() > 5 else 1.0
    else:
        tchg = 0.0
    return dict(valid_ratio=float(ok.mean()),
                range_mean=float(rv.mean()) if len(rv) else 0.0,
                range_std=float(rv.std()) if len(rv) else 0.0,
                coverage=float(sectors.mean()),
                max_gap_deg=float(best * 360.0 / n),
                roughness=rough, temporal_change=tchg)


# ============================================================================ temporal augmentation
class Temporal:
    """Streaming EMA + rolling std + staleness. Used identically in batch and in-loop."""

    def __init__(self, dim: int, rate: float, tau: float, window: float):
        self.alpha_tau = tau
        self.buf = deque(maxlen=max(2, int(round(window * rate))))
        self.ema = None
        self.t_prev = None

    def update(self, x: np.ndarray, t: float) -> np.ndarray:
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        a = 1.0 - np.exp(-dt / self.alpha_tau) if self.t_prev is not None else 1.0
        self.ema = x.copy() if self.ema is None else self.ema + a * (x - self.ema)
        self.buf.append(x)
        sd = np.std(np.asarray(self.buf), axis=0)
        self.t_prev = t
        return np.concatenate([x, self.ema, sd, [dt]])


def temporal_names(names):
    return list(names) + [n + "_ema" for n in names] + [n + "_std" for n in names] + ["staleness"]


def augment_batch(X, t, rate, cfg):
    c = cfg["confidence"]
    tmp = Temporal(X.shape[1], rate, c["temporal_tau"], c["temporal_window"])
    return np.array([tmp.update(X[i], t[i]) for i in range(len(X))]).reshape(len(X), -1)


# ============================================================================ per-sensor batch features
def _rolling(fn, x, n):
    """Causal rolling statistic over the last n samples (inclusive)."""
    out = np.empty(len(x))
    for i in range(len(x)):
        out[i] = fn(x[max(0, i - n + 1): i + 1])
    return out


def _roll_std(x, n):
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x * x, 0, 0.0))
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(0, i - n)
    cnt = i - lo
    m = (c1[i] - c1[lo]) / cnt
    v = (c2[i] - c2[lo]) / cnt - m * m
    return np.sqrt(np.maximum(v, 0))


def _roll_mean(x, n):
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(0, i - n)
    return (c1[i] - c1[lo]) / (i - lo)


IMU_FEATURES = ["gyro_std", "acc_std", "gyro_hf", "acc_hf", "stuck_frac",
                "d_w_odo", "d_w_cam", "d_ay", "d_ax", "abs_gyro", "v_odo"]
ODOM_FEATURES = ["d_w_gyro", "d_v_cam", "d_w_cam", "v_jerk", "d_acc", "zero_while_turning",
                 "abs_w", "abs_v"]
CAM_CROSS_FEATURES = ["d_w_gyro", "d_v_odo", "d_w_odo"]


def _nz(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def sensor_features(ep: dict) -> dict:
    """Base (static) feature matrices for camera, IMU and odometry (batch, causal)."""
    ti, gy, ac, iv = ep["imu_t"], ep["imu_gyro"], ep["imu_acc"], ep["imu_valid"]
    to, vo, wo, ov = ep["odom_t"], ep["odom_v"], ep["odom_w"], ep["odom_valid"]
    tc, vc, wc = ep["cam_t"], ep["cam_v"], ep["cam_w"]
    # only delivered samples are visible to other sensors
    ti_v, gy_v, ac_v = ti[iv], gy[iv], ac[iv]
    to_v, vo_v, wo_v = to[ov], vo[ov], wo[ov]
    gy_s = _roll_mean(gy_v, 5) if len(gy_v) else gy_v            # 50 ms smoothing
    ax_s = _roll_mean(ac_v[:, 0], 20) if len(ac_v) else ac_v[:, 0]
    dvo = np.gradient(vo_v, to_v) if len(vo_v) > 2 else np.zeros_like(vo_v)
    dvo_s = _roll_mean(dvo, 10) if len(dvo) else dvo

    # ---------------- IMU (at delivered IMU samples)
    w_odo, _ = zoh(to_v, wo_v, ti_v)
    v_odo, _ = zoh(to_v, vo_v, ti_v)
    w_cam, _ = zoh(tc, wc, ti_v)
    dvo_i, _ = zoh(to_v, dvo_s, ti_v)
    dg = np.abs(np.diff(gy_v, prepend=gy_v[:1]))
    da = np.linalg.norm(np.diff(ac_v, axis=0, prepend=ac_v[:1]), axis=1)
    X_imu = np.column_stack([
        _roll_std(gy_v, 25),
        np.hypot(_roll_std(ac_v[:, 0], 25), _roll_std(ac_v[:, 1], 25)),
        _roll_mean(dg, 25), _roll_mean(da, 25),
        _roll_mean((dg == 0).astype(float), 10),
        _nz(np.abs(gy_v - w_odo)), _nz(np.abs(gy_v - w_cam)),
        _nz(np.abs(ac_v[:, 1] - v_odo * gy_v)), _nz(np.abs(ax_s - dvo_i)),
        np.abs(gy_v), _nz(v_odo),
    ])

    # ---------------- odometry
    g_o, _ = zoh(ti_v, gy_s, to_v)
    vcam_o, _ = zoh(tc, vc, to_v)
    wcam_o, _ = zoh(tc, wc, to_v)
    ax_o, _ = zoh(ti_v, ax_s, to_v)
    X_odom = np.column_stack([
        _nz(np.abs(wo_v - g_o)), _nz(np.abs(vo_v - vcam_o)), _nz(np.abs(wo_v - wcam_o)),
        np.abs(np.diff(vo_v, prepend=vo_v[:1])) * 50.0,
        _nz(np.abs(dvo_s - ax_o)),
        ((np.abs(vo_v) < 1e-9) & (np.abs(wo_v) < 1e-9)) * _nz(np.abs(g_o)),
        np.abs(wo_v), np.abs(vo_v),
    ])

    # ---------------- camera (image + VO features computed at generation time)
    g_c, _ = zoh(ti_v, gy_s, tc)
    vo_c, _ = zoh(to_v, vo_v, tc)
    wo_c, _ = zoh(to_v, wo_v, tc)
    X_cross = np.column_stack([_nz(np.abs(wc - g_c)), _nz(np.abs(vc - vo_c)), _nz(np.abs(wc - wo_c))])
    X_cam = np.column_stack([ep["cam_feats"], X_cross]) if len(tc) else np.zeros((0, len(ep["meta"]["cam_feature_names"]) + 3))
    cam_names = list(ep["meta"]["cam_feature_names"]) + CAM_CROSS_FEATURES

    # ---------------- LiDAR scan-to-scan odometry (map-free track)
    tr = ep["lrel_t"]
    from ..sensors.frontends import ICP_FEATURES
    if len(tr):
        scanX = lidar_scan_feature_matrix(ep)[ep["lrel_scan"]]
        g_r, _ = zoh(ti_v, gy_s, tr)
        vo_r, _ = zoh(to_v, vo_v, tr)
        wo_r, _ = zoh(to_v, wo_v, tr)
        X_lrel = np.column_stack([scanX, ep["lrel_icp"], _nz(np.abs(ep["lrel_w"] - g_r)),
                                  _nz(np.abs(ep["lrel_v"] - vo_r)), _nz(np.abs(ep["lrel_w"] - wo_r))])
    else:
        X_lrel = np.zeros((0, len(LIDAR_SCAN_FEATURES) + len(ICP_FEATURES) + 3))
    lrel_names = LIDAR_SCAN_FEATURES + ICP_FEATURES + ["d_w_gyro", "d_v_odo", "d_w_odo"]

    return {
        "lidar_rel": dict(t=tr, X=_nz(X_lrel), names=lrel_names, idx=np.arange(len(tr))),
        "imu": dict(t=ti_v, X=_nz(X_imu), names=IMU_FEATURES, idx=np.flatnonzero(iv)),
        "odom": dict(t=to_v, X=_nz(X_odom), names=ODOM_FEATURES, idx=np.flatnonzero(ov)),
        "cam": dict(t=tc, X=_nz(X_cam), names=cam_names, idx=np.arange(len(tc))),
    }


def lidar_scan_feature_matrix(ep: dict) -> np.ndarray:
    R = ep["lidar_ranges"].astype(float)
    rows, prev = [], None
    for r in R:
        f = lidar_scan_features(r, prev)
        rows.append([f[k] for k in LIDAR_SCAN_FEATURES])
        prev = r
    return np.array(rows)


RATES = {"imu": 100.0, "odom": 50.0, "cam": 15.0, "lidar": 5.0, "lidar_rel": 5.0}
