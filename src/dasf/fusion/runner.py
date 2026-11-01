"""Runs one fusion method on one cached episode.

Asynchronous multi-rate processing on the 100 Hz filter clock:
  predict -> odometry msgs -> camera msgs -> LiDAR scans (ICP from the
  predicted pose) -> IMU sample.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

import numpy as np

from ..confidence.labels import lidar_error, reliable_label, sensor_errors
from ..sensors.frontends import icp_scan_to_map
from ..simulation.world import localization_map
from ..utils import wrap
from .ekf import EKF, chi2_gate, h_imu, h_pose, h_velocity
from .weighting import make_weight_fn, normalized_weights


def oracle_conf(ep, cfg):
    r0, p = cfg["fusion"]["r0"], cfg["confidence"]["label_prob"]
    out = {}
    for s in ("cam", "imu", "odom", "lidar_rel"):
        e = sensor_errors(ep, s)
        out[s] = reliable_label(e, r0[s], p)[0].astype(float)
    out["imu"] = out["imu"][ep["imu_valid"]]
    out["odom"] = out["odom"][ep["odom_valid"]]
    return out


class _Conf:
    """Maps confidence arrays (per delivered sample) onto raw stream indices."""

    def __init__(self, ep, arrays):
        self.map = {}
        for s, valid in (("imu", ep["imu_valid"]), ("odom", ep["odom_valid"]),
                         ("cam", np.ones(len(ep["cam_t"]), bool)),
                         ("lidar_rel", np.ones(len(ep["lrel_t"]), bool))):
            full = np.full(len(valid), np.nan)
            if arrays is not None and s in arrays:
                full[np.flatnonzero(valid)] = arrays[s]
            self.map[s] = full

    def __call__(self, s, j):
        v = self.map[s][j]
        return None if np.isnan(v) else float(v)


def run_method(ep, spec, cfg, bank=None, world=None, log=False, record_lidar=False, track="map"):
    """track='map': LiDAR = scan-to-map pose (bounded error);
    track='odom': LiDAR = scan-to-scan velocity, no map (drifting odometry frame)."""
    world = world or localization_map(ep["meta"]["world"], cfg["sensors"]["lidar"]["icp"]["map_noise"])
    f = cfg["fusion"]
    r0 = {k: np.asarray(v, float) for k, v in f["r0"].items()}           # statistical (labels, oracle)
    kappa = f.get("r0_scale") or {}
    r0_fuse = {k: v * float(kappa.get(k, 1.0)) for k, v in r0.items()}   # tuned, shared by all methods
    gfun = make_weight_fn(cfg, spec.get("weight"))
    floor = f["weight_floor"]
    conf_name = spec["conf"]
    sensors = set(spec["sensors"])
    t0_wall = time.perf_counter()
    t_conf = 0.0

    # ------------------------------------------------ confidences
    arrays, lidar_conf = None, None
    if conf_name == "oracle":
        arrays = oracle_conf(ep, cfg)
    elif conf_name is not None:
        tc = time.perf_counter()
        arrays = bank.precompute(ep, conf_name)
        t_conf += time.perf_counter() - tc
        if track == "map" and "lidar" in sensors and "lidar" not in spec["conf_off"]:
            lidar_conf = bank.lidar_stream(conf_name, ep)
    for s in spec["conf_off"]:
        if arrays is not None:
            arrays = {k: v for k, v in arrays.items() if k not in (s, s + "_rel")}
    conf = _Conf(ep, arrays)

    ti = ep["imu_t"]
    n = len(ti)
    gt0 = [ep["gt_x"][0], ep["gt_y"][0], ep["gt_th"][0], ep["gt_v"][0], ep["gt_w"][0], 0.0, 0.0]
    est = np.zeros((n, 7))
    Pd = np.zeros((n, 3))
    logs = defaultdict(list)
    lidar_rec = []
    stats = dict(n_skipped=defaultdict(int), n_rejected=defaultdict(int), n_updates=defaultdict(int),
                 icp_fail=0)

    comp = spec["kind"] == "comp"
    if comp:
        pose = np.array(gt0[:3], float)
    else:
        kf = EKF(gt0, cfg)
    gate_thr = {m: chi2_gate(m, f["gate_prob"]) for m in (2, 3)}
    rej_since = {}
    innov = defaultdict(lambda: deque(maxlen=f["iae_window"]))

    def weight(s, c):
        if c is None or s in spec["conf_off"]:
            return 1.0
        return gfun(c)

    def fuse(s, z, hfun, t, c, ang=None, r0_key=None):
        w = weight(s, c)
        if log:
            logs[s].append((t, np.nan if c is None else c, w))
        if w < floor:
            stats["n_skipped"][s] += 1
            return
        hx, H = hfun(kf.x)
        R = np.diag(r0_fuse[r0_key or s] / w)
        if spec["iae"]:
            y, S = kf.innovation(z, hx, H, R, ang)
            innov[s].append((np.outer(y, y), H @ kf.P @ H.T))
            if len(innov[s]) >= 5:
                C = np.mean([a for a, _ in innov[s]], 0) - np.mean([b for _, b in innov[s]], 0)
                R = np.diag(np.maximum(np.diag(R), np.diag(C)))
        if spec["gate"]:
            y, S = kf.innovation(z, hx, H, R, ang)
            nis = float(y @ np.linalg.solve(S, y))
            if nis > gate_thr[len(z)]:
                t_rej = rej_since.setdefault(s, t)
                if t - t_rej < f["gate_max_reject_time"]:
                    stats["n_rejected"][s] += 1
                    return
            rej_since.pop(s, None)
        kf.update(z, hx, H, R, ang)
        stats["n_updates"][s] += 1

    jo = jc = jl = 0
    to, tc_, tl, tr_ = ep["odom_t"], ep["cam_t"], ep["lidar_t"], ep["lrel_t"]
    ang = ep["lidar_angles"]
    t_prev = ti[0]
    # complementary-filter latest sources
    src_w, src_v = {}, {}
    for i in range(n):
        t = ti[i]
        dt = t - t_prev
        t_prev = t
        if not comp:
            kf.predict(dt)
        # ---------------- odometry
        while jo < len(to) and to[jo] <= t + 1e-9:
            if ep["odom_valid"][jo] and "odom" in sensors:
                c = conf("odom", jo)
                z = np.array([ep["odom_v"][jo], ep["odom_w"][jo]])
                if comp:
                    src_w["odom"] = (t, z[1], weight("odom", c))
                    src_v["odom"] = (t, z[0], weight("odom", c))
                else:
                    fuse("odom", z, h_velocity, t, c)
            jo += 1
        # ---------------- camera VO
        while jc < len(tc_) and tc_[jc] <= t + 1e-9:
            if "cam" in sensors and np.isfinite(ep["cam_v"][jc]):
                c = conf("cam", jc)
                z = np.array([ep["cam_v"][jc], ep["cam_w"][jc]])
                if comp:
                    src_w["cam"] = (t, z[1], weight("cam", c))
                    src_v["cam"] = (t, z[0], weight("cam", c))
                else:
                    fuse("cam", z, h_velocity, t, c)
            jc += 1
        # ---------------- LiDAR scan-to-scan velocity (odometry track)
        while track == "odom" and jl < len(tr_) and tr_[jl] <= t + 1e-9:
            if "lidar" in sensors:
                c = conf("lidar_rel", jl)
                z = np.array([ep["lrel_v"][jl], ep["lrel_w"][jl]])
                if comp:
                    src_w["lidar"] = (t, z[1], weight("lidar", c))
                    src_v["lidar"] = (t, z[0], weight("lidar", c))
                else:
                    fuse("lidar", z, h_velocity, t, c, r0_key="lidar_rel")
            jl += 1
        # ---------------- LiDAR scan-to-map ICP (method-dependent initial guess)
        while track == "map" and jl < len(tl) and tl[jl] <= t + 1e-9:
            if "lidar" in sensors:
                prior = pose.copy() if comp else kf.x[:3].copy()
                zl, icp_f = icp_scan_to_map(ep["lidar_ranges"][jl].astype(float), ang, prior, world, cfg)
                if lidar_conf is not None:
                    tq = time.perf_counter()
                    c = lidar_conf(jl, icp_f, t)
                    t_conf += time.perf_counter() - tq
                elif conf_name == "oracle" and zl is not None:
                    c = float(reliable_label(lidar_error(zl, ep["lidar_gt"][jl]), r0["lidar"],
                                             cfg["confidence"]["label_prob"])[0][0])
                else:
                    c = None
                if record_lidar:
                    lidar_rec.append(dict(j=jl, t=t, feats=icp_f, z=zl, prior=prior))
                if zl is None:
                    stats["icp_fail"] += 1
                elif comp:
                    wl = weight("lidar", c)
                    if log:
                        logs["lidar"].append((t, np.nan if c is None else c, wl))
                    alpha = f["comp_alpha_lidar"] * wl
                    d = zl - pose
                    d[2] = wrap(d[2])
                    pose = pose + alpha * d
                    pose[2] = wrap(pose[2])
                else:
                    fuse("lidar", zl, h_pose, t, c, ang=[2])
            jl += 1
        # ---------------- IMU
        if "imu" in sensors and ep["imu_valid"][i]:
            c = conf("imu", i)
            z = np.array([ep["imu_gyro"][i], ep["imu_acc"][i, 0], ep["imu_acc"][i, 1]])
            if comp:
                src_w["imu"] = (t, z[0], weight("imu", c))
            else:
                fuse("imu", z, h_imu, t, c)
        # ---------------- complementary propagation
        if comp:
            ws_w = {k: w for k, (ts, _, w) in src_w.items() if t - ts < 0.2}
            ws_v = {k: w for k, (ts, _, w) in src_v.items() if t - ts < 0.2}
            om = sum(normalized_weights(ws_w)[k] * src_w[k][1] for k in ws_w) if ws_w else 0.0
            vv = sum(normalized_weights(ws_v)[k] * src_v[k][1] for k in ws_v) if ws_v else 0.0
            if dt > 0:
                pose[0] += vv * np.cos(pose[2]) * dt
                pose[1] += vv * np.sin(pose[2]) * dt
                pose[2] = wrap(pose[2] + om * dt)
            est[i, :3] = pose
            est[i, 3], est[i, 4] = vv, om
        else:
            est[i] = kf.x
            Pd[i] = np.diag(kf.P)[:3]
    wall = time.perf_counter() - t0_wall
    out = dict(est=est, Pdiag=Pd, stats=stats, wall_time=wall, conf_time=t_conf,
               sim_time=float(ti[-1] - ti[0]))
    if log:
        out["logs"] = {k: np.array(v) for k, v in logs.items()}
    if record_lidar:
        out["lidar_rec"] = lidar_rec
    return out
