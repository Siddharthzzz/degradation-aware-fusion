"""Step 8: generate Figs. 2-18 from measured outputs.

Missing inputs are reported and the corresponding figure is skipped -- no
figure is ever drawn from placeholder numbers.

    python experiments/08_make_figures.py --config configs/matrix.yaml
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from _common import parse

from dasf.confidence import estimators as E
from dasf.confidence.bank import est_path
from dasf.experiment import load_dataset
from dasf.utils import resolve
from dasf.visualization import figures as F

spec = importlib.util.spec_from_file_location("an", Path(__file__).with_name("06_analyze.py"))
AN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AN)


def attempt(label, fn):
    try:
        fn()
        print(f"  ok      {label}")
    except FileNotFoundError as exc:
        print(f"  SKIPPED {label}: missing input ({exc})")
    except Exception as exc:                          # noqa: BLE001
        print(f"  FAILED  {label}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    args, cfg = parse(__doc__, lambda ap: ap.add_argument("--split", default="test"))
    out = resolve(cfg, "figures")
    runs, tdir = resolve(cfg, "runs"), resolve(cfg, "tables")
    sc = cfg["showcase"]
    eid = lambda c: f"{c.replace('+', '_')}__{sc['world']}__{sc['traj']}__s{sc['seed']}"  # noqa: E731
    ep_dir = resolve(cfg, "episodes") / args.split
    show = {c: ep_dir / f"{eid(c)}.npz" for c in sc["conditions"] if (ep_dir / f"{eid(c)}.npz").exists()}
    tr_dir = runs / "traces"

    attempt("Fig 2 robot/sensors", lambda: F.fig2_robot(cfg, out))
    attempt("Figs 3-4 sensor data", lambda: F.fig3_4_sensor_examples(cfg, out, show))
    multi = [c for c in sc["conditions"] if "+" in c]
    attempt("Figs 5-6 confidence/weights", lambda: F.fig5_6_conf_weights(
        cfg, out, tr_dir / f"{eid(multi[0] if multi else sc['conditions'][-1])}.npz"))
    for cond, name, title in (("nominal", "fig07_traj_nominal", "Nominal"),
                              ("cam.lowlight.3", "fig08_traj_camera", "Camera low light (3)"),
                              ("lidar.fov.3", "fig09_traj_lidar", "LiDAR FOV blocked (3)"),
                              ("imu.bias.2", "fig10_traj_imu", "IMU gyro bias (2)")):
        attempt(f"{name}", lambda c=cond, n=name, t=title: F.fig_traj(
            cfg, out, ep_dir / f"{eid(c)}.npz", tr_dir / f"{eid(c)}.npz", n, t))

    metrics_csv = runs / f"{args.split}_metrics.csv"
    if metrics_csv.exists():
        df, sel = AN.load(cfg, args.split)
        main = [m for m in AN.MAIN_BASELINES + ["proposed", "oracle"] if m in set(df.method)]
        attempt("Fig 11 RMSE vs severity", lambda: F.fig_severity(cfg, out, df, "pos_rmse", "position RMSE (m)", "fig11_rmse_severity", main))
        attempt("Fig 12 orientation vs severity", lambda: F.fig_severity(cfg, out, df, "ori_rmse_deg", "orientation RMSE (deg)", "fig12_orientation_severity", main))
        attempt("Fig 13 recovery", lambda: F.fig13_recovery(cfg, out, df, [m for m in main if m != "oracle"]))
        rob_p = tdir / "robustness_increase.csv"
        attempt("Fig 18 heatmap", lambda: F.fig18_heatmap(cfg, out, pd.read_csv(rob_p), main))
        abl_p = tdir / "tests_ablations.csv"
        attempt("Fig 17 ablations", lambda: F.fig17_ablation(cfg, out, pd.read_csv(abl_p)))
    else:
        print(f"  SKIPPED Figs 11-13, 17-18: {metrics_csv} not found (run 05 and 06)")

    comp_p = tdir / "compute_estimators.csv"
    attempt("Figs 14-15 compute", lambda: F.fig14_15_compute(cfg, out, pd.read_csv(comp_p)))

    def calib():
        sel = json.load(open(resolve(cfg, "models") / "selection.json"))
        prop = f"{sel['family']}_{sel['feature_set']}"
        preds = {}
        for s in ("cam", "lidar", "lidar_rel", "imu", "odom"):
            te = load_dataset(cfg, "test", s)
            preds[s] = {}
            for lab, n in (("rule", "rule"), ("proposed", prop), ("cnn", "cnn")):
                p = est_path(resolve(cfg, "models"), s, n)
                if not p.exists() or (n == "cnn" and s == "cam"):
                    continue
                est = E.load(p)
                X = te["raw"].astype(np.float32) if n == "cnn" and s.startswith("lidar") else (
                    te["X"] if getattr(est, "feature_set", "") == "static" else te["Xt"])
                preds[s][lab] = (est.predict(X), te["y"])
        F.fig16_calibration(cfg, out, preds)
    attempt("Fig 16 calibration", calib)
    print(f"figures -> {out}")
