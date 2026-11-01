"""Experiment management: episode specs per split, parallel execution,
R0 calibration, confidence dataset construction."""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .confidence.features import (LIDAR_SCAN_FEATURES, RATES, augment_batch,
                                  lidar_scan_feature_matrix, sensor_features)
from .confidence.labels import lidar_error, reliable_label, robust_diag_cov, sensor_errors
from .degradation.catalog import parse_condition, random_training_events
from .episode import generate_episode, load_episode, save_episode
from .sensors.frontends import ICP_FEATURES
from .utils import ROOT, resolve, rng_stream, stable_int

BATCH_SENSORS = ("cam", "imu", "odom", "lidar_rel")
ALL_CONF_SENSORS = BATCH_SENSORS + ("lidar",)


# ============================================================================ episode specs
def episode_specs(cfg: dict, split: str) -> list[dict]:
    sp = cfg["splits"][split]
    T = cfg["sim"]["duration"]
    t_on, t_off = cfg["sim"]["deg_window"]
    specs = []
    for world in sp["worlds"]:
        for traj in sp["trajs"]:
            for seed in range(*sp["seeds"]):
                if sp["events"] == "nominal":
                    specs.append(dict(split=split, world=world, traj=traj, seed=seed,
                                      condition="nominal", events=[], save_images_every=0))
                elif sp["events"] == "random":
                    rng = rng_stream(stable_int(seed, world, traj), "train_events")
                    evs = random_training_events(rng, T)
                    specs.append(dict(split=split, world=world, traj=traj, seed=seed,
                                      condition="random", events=evs,
                                      save_images_every=sp.get("save_images_every", 0)))
                else:
                    for cond in cfg["conditions"]:
                        specs.append(dict(split=split, world=world, traj=traj, seed=seed,
                                          condition=cond, events=parse_condition(cond, t_on, t_off),
                                          save_images_every=0))
    return specs


def episode_id(spec) -> str:
    return f"{spec['condition'].replace('+', '_')}__{spec['world']}__{spec['traj']}__s{spec['seed']}"


def episode_path(cfg, spec) -> Path:
    return resolve(cfg, "episodes") / spec["split"] / f"{episode_id(spec)}.npz"


def _gen_one(args):
    cfg, spec, overwrite = args
    p = episode_path(cfg, spec)
    if p.exists() and not overwrite:
        return str(p), "cached"
    keep = bool(cfg.get("store_cam_frames")) and spec["split"] in ("val", "test")
    ep = generate_episode(cfg, spec["world"], spec["traj"], spec["seed"], spec["events"],
                          save_images_every=spec["save_images_every"], keep_frames=keep)
    ep["meta"]["condition"] = spec["condition"]
    ep["meta"]["split"] = spec["split"]
    save_episode(ep, p)
    return str(p), "generated"


def pmap(fn, items, workers):
    if workers <= 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items, chunksize=1))


def default_workers():
    return max(1, (os.cpu_count() or 2) - 1)


def generate_split(cfg, split, workers, overwrite=False, log=print):
    specs = episode_specs(cfg, split)
    log(f"[{split}] {len(specs)} episodes")
    res = pmap(_gen_one, [(cfg, s, overwrite) for s in specs], workers)
    log(f"[{split}] generated {sum(r[1] == 'generated' for r in res)}, cached {sum(r[1] == 'cached' for r in res)}")
    return [r[0] for r in res]


def split_paths(cfg, split):
    return sorted((resolve(cfg, "episodes") / split).glob("*.npz"))


# ============================================================================ calibration
def _uninjected_errors(ep, sensor):
    e = sensor_errors(ep, sensor)
    inj = ep[{"cam": "cam_injected", "imu": "imu_injected", "odom": "odom_injected",
              "lidar_rel": "lrel_injected"}[sensor]]
    if sensor == "imu":
        keep = ep["imu_valid"] & ~inj
    elif sensor == "odom":
        keep = ep["odom_valid"] & ~inj
    else:
        keep = ~inj
    return e[keep & np.all(np.isfinite(e), axis=1)]


def _lidar_record(args):
    """Run the data-collection filter (fixed-R0 EKF, map track) and record ICP
    features + errors for every scan."""
    from .fusion.methods import METHODS
    from .fusion.runner import run_method
    cfg, path = args
    ep = load_episode(Path(path))
    out = run_method(ep, METHODS["ekf"], cfg, record_lidar=True, track="map")
    rows = []
    for r in out["lidar_rec"]:
        j = r["j"]
        err = lidar_error(r["z"], ep["lidar_gt"][j]) if r["z"] is not None else np.full(3, np.nan)
        rows.append(np.concatenate([[j, r["t"], float(ep["lidar_injected"][j]),
                                     float(r["z"] is not None)],
                                    [r["feats"][k] for k in ICP_FEATURES], err]))
    return path, np.array(rows)


def calibrate_r0(cfg, workers, log=print):
    """Fit nominal R0 per sensor on UNINJECTED samples of the TRAIN split."""
    paths = split_paths(cfg, "train")
    E = {s: [] for s in BATCH_SENSORS}
    for p in paths:
        ep = load_episode(p)
        for s in BATCH_SENSORS:
            E[s].append(_uninjected_errors(ep, s))
    r0 = {s: robust_diag_cov(np.vstack(E[s])) for s in BATCH_SENSORS}
    # LiDAR map track: needs filter-in-the-loop ICP (data-collection EKF uses
    # the calibrated R0 of the other sensors and the default LiDAR R0).
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["fusion"]["r0"].update(r0)
    recs = pmap(_lidar_record, [(cfg2, str(p)) for p in paths], workers)
    El = []
    for _, R in recs:
        if len(R):
            ok = (R[:, 2] == 0) & (R[:, 3] == 1)
            El.append(R[ok][:, -3:])
    r0["lidar"] = robust_diag_cov(np.vstack(El))
    scale = tune_r0_scale(cfg, r0, workers, log)
    out = dict(r0=r0, r0_scale=scale,
               source="r0: train split, uninjected samples, 1% trimmed per-axis variance; "
                      "r0_scale: coordinate descent on the nominal 'tune' split (plain EKF, both tracks)",
               n_samples={s: int(sum(len(e) for e in E[s])) for s in BATCH_SENSORS})
    p = ROOT / cfg["paths"]["calibration"]
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"), indent=2)
    log(f"calibrated R0 -> {p}")
    for s, v in r0.items():
        log(f"  {s:10s} std = {np.sqrt(v)}")
    return r0


SCALE_GRID = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


def tune_r0_scale(cfg, r0, workers, log=print, passes=2):
    """Measurement errors are time-correlated (e.g. wheel radius/base errors,
    VO drift), so the white-noise variance R0 over-weights high-rate sensors.
    A per-sensor inflation kappa_s (R = kappa_s R0) is tuned ONCE on nominal
    episodes of a dedicated tuning split, then frozen and shared by every
    method.  Labels keep the statistical R0."""
    from .fusion.methods import METHODS
    paths = split_paths(cfg, "tune")
    if not paths:
        log("no tune split -> r0_scale = 1")
        return {s: 1.0 for s in r0}
    base = json.loads(json.dumps(cfg))
    base["fusion"]["r0"] = r0

    def score(scale):
        c = json.loads(json.dumps(base))
        c["fusion"]["r0_scale"] = scale
        res = pmap(run_episode, [(c, str(p), [("ekf", METHODS["ekf"])], c["tracks"], None) for p in paths], workers)
        return float(np.mean([np.log(max(r["pos_rmse"], 1e-4)) for rs in res for r in rs]))

    scale = {s: 1.0 for s in r0}
    best = score(scale)
    for _ in range(passes):
        for s in r0:
            for k in SCALE_GRID:
                if k == scale[s]:
                    continue
                trial = dict(scale, **{s: k})
                v = score(trial)
                if v < best - 1e-3:
                    best, scale = v, trial
        log(f"  r0_scale after pass: {scale}  (tune score {best:.4f})")
    return scale


# ============================================================================ datasets
def _dataset_one(args):
    """All confidence-training rows for one episode."""
    cfg, path = args
    ep = load_episode(Path(path))
    r0, p = cfg["fusion"]["r0"], cfg["confidence"]["label_prob"]
    feats = sensor_features(ep)
    meta = ep["meta"]
    eid = Path(path).stem
    out = {}
    for s in BATCH_SENSORS:
        f = feats[s]
        if len(f["X"]) == 0:
            continue
        e = sensor_errors(ep, s)[f["idx"]]
        y, d2 = reliable_label(e, r0[s], p)
        inj_key = {"cam": "cam_injected", "imu": "imu_injected", "odom": "odom_injected",
                   "lidar_rel": "lrel_injected"}[s]
        inj = ep[inj_key][f["idx"]]
        Xt = augment_batch(f["X"], f["t"], RATES[s], cfg)
        good = np.all(np.isfinite(e), axis=1)
        out[s] = dict(X=f["X"][good], Xt=Xt[good], y=y[good], d2=d2[good], inj=inj[good],
                      t=f["t"][good], names=f["names"])
        if s == "lidar_rel":
            j = ep["lrel_scan"][good]
            out[s]["raw"] = np.stack([ep["lidar_ranges"][j - 1], ep["lidar_ranges"][j]], 1).astype(np.float16)
        if s == "cam" and "cam_images" in ep:
            ti = ep["cam_images_t"]
            k = np.searchsorted(f["t"][good], ti)
            ok = (k < good.sum()) & (np.abs(f["t"][good][np.minimum(k, good.sum() - 1)] - ti) < 1e-6)
            out[s]["img_idx"] = k[ok]
            out[s]["imgs"] = ep["cam_images"][ok]
    # LiDAR map track (data-collection filter in the loop)
    _, R = _lidar_record((cfg, path))
    if len(R):
        scanX = lidar_scan_feature_matrix(ep)
        J = R[:, 0].astype(int)
        X = np.column_stack([scanX[J], R[:, 4:4 + len(ICP_FEATURES)]])
        Xt = augment_batch(X, R[:, 1], RATES["lidar"], cfg)      # over ALL scans (incl. ICP failures)
        ok = R[:, 3] == 1
        y, d2 = reliable_label(R[ok][:, -3:], r0["lidar"], p)
        out["lidar"] = dict(X=X[ok], Xt=Xt[ok], y=y, d2=d2, inj=R[ok][:, 2].astype(bool), t=R[ok][:, 1],
                            names=LIDAR_SCAN_FEATURES + ICP_FEATURES,
                            raw=ep["lidar_ranges"][J[ok]].astype(np.float16))
    return eid, meta, out


def build_datasets(cfg, split, workers, log=print):
    paths = split_paths(cfg, split)
    res = pmap(_dataset_one, [(cfg, str(p)) for p in paths], workers)
    ddir = resolve(cfg, "datasets")
    for s in ALL_CONF_SENSORS:
        parts = [(eid, meta, o[s]) for eid, meta, o in res if s in o]
        if not parts:
            continue
        cat = lambda k: np.concatenate([o[k] for _, _, o in parts]) if k in parts[0][2] else None  # noqa: E731
        ev_kind = np.concatenate([np.full(len(o["y"]), "|".join(sorted({f"{e.sensor}.{e.kind}" for e in m["events"]})) or "none",
                                          dtype=object) for _, m, o in parts])
        epi = np.concatenate([np.full(len(o["y"]), eid, dtype=object) for eid, _, o in parts])
        arrays = dict(X=cat("X"), Xt=cat("Xt"), y=cat("y"), d2=cat("d2"), inj=cat("inj"), t=cat("t"),
                      episode=epi.astype(str), events=ev_kind.astype(str),
                      names=np.array(parts[0][2]["names"]))
        for k in ("raw", "imgs"):
            v = cat(k)
            if v is not None:
                arrays[k] = v
        if s == "cam" and "img_idx" in parts[0][2]:
            offs = np.cumsum([0] + [len(o["y"]) for _, _, o in parts[:-1]])
            arrays["img_rows"] = np.concatenate([o["img_idx"] + off for (_, _, o), off in zip(parts, offs)])
        np.savez_compressed(ddir / f"{split}_{s}.npz", **arrays)
        log(f"  {split}/{s}: {len(arrays['y'])} rows, reliable={arrays['y'].mean():.3f}, "
            f"injected={arrays['inj'].mean():.3f}")


_BANKS = {}


def get_bank(cfg):
    from .confidence.bank import ConfidenceBank
    key = str(resolve(cfg, "models"))
    if key not in _BANKS:
        _BANKS[key] = ConfidenceBank(cfg, resolve(cfg, "models"), resolve(cfg, "heavy_conf"))
    return _BANKS[key]


def run_episode(args):
    """Run (name, spec) methods on one episode for the given tracks -> metric rows."""
    from .evaluation.metrics import episode_metrics
    from .fusion.runner import run_method
    cfg, path, named_specs, tracks, save_trace_dir = args
    ep = load_episode(Path(path))
    meta = ep["meta"]
    evs = meta["events"]
    if meta.get("condition", "random") == "random" or not evs:
        t_on, t_off = cfg["sim"]["deg_window"]
        if evs:
            t_on, t_off = min(e.t_on for e in evs), max(e.t_off for e in evs)
    else:
        t_on, t_off = evs[0].t_on, evs[0].t_off
    bank = get_bank(cfg)
    rows, traces = [], {}
    for name, spec in named_specs:
        for track in tracks:
            try:
                out = run_method(ep, spec, cfg, bank=bank, track=track, log=save_trace_dir is not None)
            except FileNotFoundError as exc:          # e.g. heavy model not trained
                rows.append(dict(method=name, track=track, error=str(exc)[:200]))
                continue
            m = episode_metrics(ep, out, cfg, t_on, t_off)
            m.update(method=name, track=track, episode=Path(path).stem, condition=meta.get("condition"),
                     world=meta["world"], traj=meta["traj"], seed=meta["seed"])
            rows.append(m)
            if save_trace_dir is not None:
                traces[f"{name}__{track}__est"] = out["est"].astype(np.float32)
                for s, L in out.get("logs", {}).items():
                    traces[f"{name}__{track}__log_{s}"] = L.astype(np.float32)
    if save_trace_dir is not None and traces:
        d = Path(save_trace_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"{Path(path).stem}.npz", **traces)
    return rows


def load_dataset(cfg, split, sensor):
    with np.load(resolve(cfg, "datasets") / f"{split}_{sensor}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
