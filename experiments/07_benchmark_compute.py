"""Step 7: computational cost -- the evidence for (or against) "lightweight".

Measures, on THIS machine (report CPU model; repeat on the target SBC, e.g. a
Raspberry Pi 4, for the paper's deployment claim):
  * parameters, serialized model size, single-sample inference latency
    (median / p95 over 2000 calls, 1 thread), peak Python heap during inference
  * confidence feature-extraction latency per measurement
  * end-to-end filter real-time factor and CPU utilisation per method

    python experiments/07_benchmark_compute.py --config configs/matrix.yaml
"""
import json
import os
import platform
import time
import tracemalloc

import numpy as np
import pandas as pd
from _common import parse

os.environ.setdefault("OMP_NUM_THREADS", "1")

from dasf.confidence import estimators as E  # noqa: E402
from dasf.confidence.bank import est_path  # noqa: E402
from dasf.confidence.features import (augment_batch, camera_image_features,  # noqa: E402
                                      lidar_scan_features, sensor_features, RATES)
from dasf.episode import generate_episode, load_episode  # noqa: E402
from dasf.experiment import ALL_CONF_SENSORS, get_bank, load_dataset, split_paths  # noqa: E402
from dasf.fusion.methods import resolve as resolve_method  # noqa: E402
from dasf.fusion.runner import run_method  # noqa: E402
from dasf.utils import resolve  # noqa: E402


def timeit(fn, n=200, warm=20):
    for _ in range(warm):
        fn()
    ts = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        fn()
        ts[i] = time.perf_counter() - t0
    return float(np.median(ts) * 1e6), float(np.quantile(ts, 0.95) * 1e6)


def bench_estimators(cfg):
    rows = []
    mdir = resolve(cfg, "models")
    names = ["rule"] + [f"{f}_{s}" for f in cfg["confidence"]["families"]
                        for s in cfg["confidence"]["feature_sets"]] + ["cnn"]
    for s in ALL_CONF_SENSORS:
        te = load_dataset(cfg, "test", s)
        for n in names:
            p = est_path(mdir, s, n)
            if not p.exists():
                continue
            est = E.load(p)
            if n == "cnn":
                from dasf.confidence import neural as N
                if s == "cam":
                    x = te["imgs"][:1] if "imgs" in te else None
                    if x is None:
                        tr = load_dataset(cfg, "train", "cam")
                        x = tr["imgs"][:1]
                    est_in = N.camera_input(x)
                    net_est = N.TorchEstimator(s, est.net, None)
                    lat = N.torch_latency(net_est, est_in)
                    gpu = N.torch_latency(net_est, est_in, device="cuda") if N.torch.cuda.is_available() else {}
                else:
                    x = te["raw"][:1].astype(np.float32) if s.startswith("lidar") else te["Xt"][:1]
                    lat = N.torch_latency(est, x)
                    gpu = N.torch_latency(est, x, device="cuda") if N.torch.cuda.is_available() else {}
            else:
                x = (te["X"] if est.feature_set == "static" else te["Xt"])[:1]
                lat = E.single_sample_latency(est, x)
                gpu = {}
            tracemalloc.start()
            for _ in range(200):
                est.predict(x) if n != "cnn" or s != "cam" else None
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            rows.append(dict(sensor=s, estimator=n, n_params=est.n_params(), size_bytes=E.model_size_bytes(est),
                             latency_median_us=lat["median_us"], latency_p95_us=lat["p95_us"],
                             gpu_latency_median_us=gpu.get("median_us", np.nan),
                             peak_heap_bytes=peak, rate_hz=RATES[s],
                             cpu_duty_pct=100 * lat["median_us"] * 1e-6 * RATES[s]))
            print(f"  {s:9s} {n:14s} params {rows[-1]['n_params']:>9d}  {lat['median_us']:9.1f} us")
    return pd.DataFrame(rows)


def bench_features(cfg):
    ep = load_episode(split_paths(cfg, "test")[0])
    from dasf.sensors.camera import Camera
    from dasf.simulation.world import make_world
    cam, w = Camera(cfg), make_world(ep["meta"]["world"])
    rng = np.random.default_rng(0)
    img0 = cam.expose(cam.irradiance(w, (1, 1, 0)), rng, {}, 0.0)
    img1 = cam.expose(cam.irradiance(w, (1.02, 1, 0.01)), rng, {}, 0.0)
    r = ep["lidar_ranges"][0].astype(float)
    rows = [dict(item="camera image features / frame", **dict(zip(["median_us", "p95_us"], timeit(lambda: camera_image_features(img1, img0))))),
            dict(item="LiDAR scan features / scan", **dict(zip(["median_us", "p95_us"], timeit(lambda: lidar_scan_features(r, r)))))]
    t0 = time.perf_counter()
    F = sensor_features(ep)
    dt = time.perf_counter() - t0
    nmeas = sum(len(v["X"]) for v in F.values())
    rows.append(dict(item="all batch features / measurement (amortised)", median_us=1e6 * dt / nmeas, p95_us=np.nan))
    t0 = time.perf_counter()
    augment_batch(F["imu"]["X"], F["imu"]["t"], 100.0, cfg)
    rows.append(dict(item="temporal augmentation / IMU sample", median_us=1e6 * (time.perf_counter() - t0) / len(F["imu"]["X"]), p95_us=np.nan))
    return pd.DataFrame(rows)


def bench_filters(cfg, sel, n_eps=3):
    rows = []
    paths = split_paths(cfg, "test")[:n_eps]
    bank = get_bank(cfg)
    for name in ["ekf", "ekf_gate", "ekf_iae", "rule", "proposed", "heavy_cnn"]:
        try:
            spec = resolve_method(name, sel)
        except RuntimeError:
            continue
        for tr in cfg["tracks"]:
            wall = cpu = sim = ct = 0.0
            ok = True
            for p in paths:
                ep = load_episode(p)
                c0, w0 = time.process_time(), time.perf_counter()
                try:
                    out = run_method(ep, spec, cfg, bank=bank, track=tr)
                except FileNotFoundError:
                    ok = False
                    break
                cpu += time.process_time() - c0
                wall += time.perf_counter() - w0
                sim += out["sim_time"]
                ct += out["conf_time"]
            if ok:
                rows.append(dict(method=name, track=tr, rtf=wall / sim, cpu_util_pct=100 * cpu / wall,
                                 conf_time_frac=ct / wall, wall_s_per_sim_s=wall / sim))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    args, cfg = parse(__doc__)
    tdir = resolve(cfg, "tables")
    sel_p = resolve(cfg, "models") / "selection.json"
    sel = json.load(open(sel_p)) if sel_p.exists() else None
    info = dict(platform=platform.platform(), processor=platform.processor(), python=platform.python_version(),
                cpu_count=os.cpu_count())
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        pass
    json.dump(info, open(tdir / "compute_platform.json", "w"), indent=2)
    bench_estimators(cfg).to_csv(tdir / "compute_estimators.csv", index=False)
    bench_features(cfg).to_csv(tdir / "compute_features.csv", index=False)
    bench_filters(cfg, sel).to_csv(tdir / "compute_filters.csv", index=False)
    print(f"compute benchmarks written to {tdir}")
