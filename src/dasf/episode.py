"""Episode generation: ground truth + all raw sensor streams + front-end outputs.

An episode is fully determined by (world, trajectory kind, seed, degradation
events).  Everything that does NOT depend on the fusion method is computed
here once and cached; the only method-dependent front-end (LiDAR scan-to-map
ICP, which is initialised from the filter prediction) runs inside the filter
loop on the cached raw scans.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .confidence.features import camera_image_features, CAM_IMAGE_FEATURES
from .degradation.catalog import DegEvent, active_params
from .degradation.imu import apply_imu
from .degradation.lidar import apply_lidar
from .degradation.odometry import apply_encoder, slip_factors
from .sensors.camera import Camera
from .sensors.frontends import ICP_FEATURES, CameraVO, icp_scan_to_scan
from .simulation.robot import generate_trajectory
from .simulation.world import make_world
from .utils import rng_stream, wrap


def interp_pose(gt, tq):
    th = np.unwrap(gt["th"])
    return (np.interp(tq, gt["t"], gt["x"]), np.interp(tq, gt["t"], gt["y"]),
            wrap(np.interp(tq, gt["t"], th)))


def _rel_motion(p0, p1, dt):
    """Body-frame forward velocity, lateral velocity and yaw rate between poses."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    c, s = np.cos(p0[2]), np.sin(p0[2])
    return (c * dx + s * dy) / dt, (-s * dx + c * dy) / dt, wrap(p1[2] - p0[2]) / dt


def _injected(events, sensor, t):
    m = np.zeros(len(t), bool)
    for e in events:
        if e.sensor == sensor:
            m |= e.active(t)
    return m


def generate_episode(cfg, world_name, traj_kind, seed, events: list[DegEvent],
                     save_images_every: int = 0, cam_hook=None, keep_frames: bool = False) -> dict:
    world = make_world(world_name)
    gt = generate_trajectory(world, traj_kind, seed, cfg)
    t = gt["t"]
    n = len(t)
    dt = cfg["sim"]["dt"]
    S = cfg["sensors"]
    ep = {"gt_" + k: v for k, v in gt.items()}

    # ------------------------------------------------------------------ IMU (100 Hz)
    ci = S["imu"]
    r = rng_stream(seed, "imu")
    bg = r.normal(0, ci["gyro_bias_init"]) + np.cumsum(r.normal(0, ci["gyro_bias_rw"] * np.sqrt(dt), n))
    ba = r.normal(0, ci["acc_bias_init"], 2) + np.cumsum(r.normal(0, ci["acc_bias_rw"] * np.sqrt(dt), (n, 2)), 0)
    gyro = gt["w"] + bg + r.normal(0, ci["gyro_noise"], n)
    acc = np.column_stack([gt["a"], gt["v"] * gt["w"]]) + ba + r.normal(0, ci["acc_noise"], (n, 2))
    gyro, acc, ivalid = apply_imu(t, gyro, acc, events, cfg, rng_stream(seed, "imu_deg"))
    ep.update(imu_t=t, imu_gyro=gyro, imu_acc=acc, imu_valid=ivalid,
              imu_bg_nominal=bg, imu_injected=_injected(events, "imu", t))

    # ------------------------------------------------------------------ wheel odometry (50 Hz)
    co, rc = S["odom"], cfg["robot"]
    r = rng_stream(seed, "odom")
    r_nom, b_nom = rc["wheel_radius"], rc["wheel_base"]
    rR = r_nom * (1 + r.normal(0, rc["radius_error_std"]))
    rL = r_nom * (1 + r.normal(0, rc["radius_error_std"]))
    b_true = b_nom * (1 + r.normal(0, rc["base_error_std"]))
    fr, fl = slip_factors(t, events, rng_stream(seed, "odom_slip"))
    dR = (gt["v"] + gt["w"] * b_true / 2) / rR * dt * fr + co["speed_noise"] * dt * r.standard_normal(n)
    dL = (gt["v"] - gt["w"] * b_true / 2) / rL * dt * fl + co["speed_noise"] * dt * r.standard_normal(n)
    phiR = np.concatenate([[0.0], np.cumsum(dR)[:-1]])      # angle at t_i (before tick i motion)
    phiL = np.concatenate([[0.0], np.cumsum(dL)[:-1]])
    k = int(round(ci["rate"] / co["rate"]))
    idx = np.arange(k, n, k)
    tpr = co["ticks_per_rev"]
    tickR = np.floor(phiR[idx] * tpr / (2 * np.pi))
    tickL = np.floor(phiL[idx] * tpr / (2 * np.pi))
    tickR, tickL, ovalid = apply_encoder(t[idx], tickR, tickL, events, rng_stream(seed, "odom_deg"))
    to = t[idx]
    dtR = np.diff(np.concatenate([[0.0], tickR])) * 2 * np.pi / tpr
    dtL = np.diff(np.concatenate([[0.0], tickL])) * 2 * np.pi / tpr
    dto = np.diff(np.concatenate([[0.0], to]))
    v_o = r_nom * (dtR + dtL) / 2 / dto
    w_o = r_nom * (dtR - dtL) / b_nom / dto
    v_true = np.array([gt["v"][i - k:i].mean() for i in idx])
    w_true = np.array([gt["w"][i - k:i].mean() for i in idx])
    ep.update(odom_t=to, odom_v=v_o, odom_w=w_o, odom_valid=ovalid,
              odom_v_true=v_true, odom_w_true=w_true, odom_injected=_injected(events, "odom", to))

    # ------------------------------------------------------------------ LiDAR raw scans (5 Hz)
    cl = S["lidar"]
    r = rng_stream(seed, "lidar")
    rd = rng_stream(seed, "lidar_deg")
    tl = np.arange(1, int(t[-1] * cl["rate"]) + 1) / cl["rate"]
    ang = np.arange(cl["n_beams"]) * 2 * np.pi / cl["n_beams"]
    scans = np.empty((len(tl), cl["n_beams"]), np.float32)
    px, py, pth = interp_pose(gt, tl)
    for j, tj in enumerate(tl):
        d, _, _ = world.raycast((px[j], py[j]), pth[j] + ang, max_range=50.0)
        d = d + cl["range_noise"] * r.standard_normal(len(d))
        d[r.random(len(d)) < cl["nominal_dropout"]] = np.inf
        d = apply_lidar(d, ang, active_params(events, "lidar", tj), rd)
        d[(d > cl["range_max"]) | (d < cl["range_min"])] = np.inf
        scans[j] = d
    ep.update(lidar_t=tl, lidar_angles=ang, lidar_ranges=scans,
              lidar_gt=np.column_stack([px, py, pth]),
              lidar_injected=_injected(events, "lidar", tl))

    # ------------------------------------------------------------------ LiDAR scan-to-scan odometry
    # (map-free track).  Method-independent: initialised with the previous
    # scan-to-scan result (constant-velocity guess), never with the filter.
    rel, rel_f, init = [], [], np.zeros(3)
    for j in range(1, len(tl)):
        x, f = icp_scan_to_scan(scans[j - 1].astype(float), scans[j].astype(float), ang, init, cfg)
        dtl = tl[j] - tl[j - 1]
        if x is None:
            init = np.zeros(3)
            continue
        vt, _, wt = _rel_motion((px[j - 1], py[j - 1], pth[j - 1]), (px[j], py[j], pth[j]), dtl)
        rel.append((tl[j], x[0] / dtl, x[2] / dtl, vt, wt, j))
        rel_f.append([f[k_] for k_ in ICP_FEATURES])
        init = x
    rel = np.array(rel) if rel else np.zeros((0, 6))
    ep.update(lrel_t=rel[:, 0], lrel_v=rel[:, 1], lrel_w=rel[:, 2], lrel_v_true=rel[:, 3],
              lrel_w_true=rel[:, 4], lrel_scan=rel[:, 5].astype(int),
              lrel_icp=np.array(rel_f, float).reshape(len(rel), -1),
              lrel_injected=_injected(events, "lidar", rel[:, 0]))

    # ------------------------------------------------------------------ camera + VO (15 Hz)
    cam = Camera(cfg)
    vo = CameraVO(cam, cfg)
    r = rng_stream(seed, "cam")
    rd = rng_stream(seed, "cam_deg")
    tc = np.arange(0, int(t[-1] * S["camera"]["rate"]) + 1) / S["camera"]["rate"]
    cx, cy, cth = interp_pose(gt, tc)
    w_gt_c = np.interp(tc, t, gt["w"])
    rows, feats, images, image_t = [], [], [], []
    frames, frames_t = [], []
    prev_img, prev_t, prev_pose, last_v = None, None, None, 0.0
    for j, tj in enumerate(tc):
        deg = active_params(events, "cam", tj)
        irr = cam.irradiance(world, (cx[j], cy[j], cth[j]))
        img = cam.expose(irr, r, deg, w_gt_c[j], rd)          # always draw -> CRN across conditions
        pose = (cx[j], cy[j], cth[j])
        if "dropout" in deg:
            continue
        if "frozen" in deg and prev_img is not None:
            img = prev_img.copy()
        if cam_hook is not None:
            cam_hook(tj, img)
        if keep_frames:   # exactly the frames cam_hook sees; img is k/255, so uint8 is lossless
            frames.append(np.round(img * 255).astype(np.uint8))
            frames_t.append(tj)
        if save_images_every and j % save_images_every == 0 and prev_img is not None:
            images.append((np.stack([prev_img, img]) * 255).astype(np.uint8))   # [prev, cur]
            image_t.append(tj)
        if prev_img is not None:
            dtc = tj - prev_t
            v_c, w_c, lat, f_vo = vo.estimate(prev_img, img, dtc, init_v=last_v)
            vt, _, wt = _rel_motion(prev_pose, pose, dtc)
            f_im = camera_image_features(img, prev_img)
            rows.append((tj, v_c, w_c, vt, wt, dtc))
            feats.append([f_im[k_] for k_ in CAM_IMAGE_FEATURES] + [f_vo[k_] for k_ in CameraVO.FEATURES])
            last_v = v_c if np.isfinite(v_c) and abs(v_c) < 1.0 else 0.0
        prev_img, prev_t, prev_pose = img, tj, pose
    rows = np.array(rows) if rows else np.zeros((0, 6))
    ep.update(cam_t=rows[:, 0], cam_v=rows[:, 1], cam_w=rows[:, 2], cam_v_true=rows[:, 3],
              cam_w_true=rows[:, 4], cam_dt=rows[:, 5],
              cam_feats=np.array(feats, float).reshape(len(rows), -1),
              cam_injected=_injected(events, "cam", rows[:, 0]))
    if save_images_every:
        ep["cam_images"] = np.array(images)
        ep["cam_images_t"] = np.array(image_t)
        ep["cam_images_injected"] = _injected(events, "cam", np.array(image_t))
    if keep_frames:
        ep["cam_frames"] = np.array(frames, np.uint8)
        ep["cam_frames_t"] = np.array(frames_t)
    ep["meta"] = dict(world=world_name, traj=traj_kind, seed=int(seed), events=list(events),
                      cam_feature_names=CAM_IMAGE_FEATURES + CameraVO.FEATURES)
    return ep


def save_episode(ep: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(ep["meta"])
    meta["events"] = [e.to_dict() for e in meta["events"]]
    np.savez_compressed(path, **{**ep, "meta": json.dumps(meta)})


def load_episode(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        ep = {k: z[k] for k in z.files}
    ep["meta"] = json.loads(str(ep["meta"]))
    ep["meta"]["events"] = [DegEvent(**e) for e in ep["meta"]["events"]]
    return ep
