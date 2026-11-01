"""Step 3: train confidence estimators, calibrate on VAL, and select the
proposed configuration on VAL (never on TEST).

Pre-registered selection rule (fixed before any test run):
  score(config) = mean over val episodes and both tracks of log(position RMSE)
  1. choose (family, feature set) with weight function = linear;
  2. for that estimator choose the weight function g in {linear, power, gate, exp};
  3. single_best (Baseline 1) = single-sensor method with the best score, per track.
Ties (|delta score| < 0.01) are broken toward fewer parameters.

    python experiments/03_train_confidence.py --config configs/matrix.yaml
"""
import json

import numpy as np
import pandas as pd
from _common import parse

from dasf.confidence import estimators as E
from dasf.confidence.bank import est_path
from dasf.confidence.features import temporal_names
from dasf.evaluation.metrics import confidence_metrics
from dasf.experiment import ALL_CONF_SENSORS, load_dataset, pmap, run_episode, split_paths
from dasf.fusion.methods import METHODS
from dasf.utils import resolve


def train_all(cfg, log=print):
    mdir = resolve(cfg, "models")
    rows = []
    for s in ALL_CONF_SENSORS:
        tr, va = load_dataset(cfg, "train", s), load_dataset(cfg, "val", s)
        names = list(tr["names"])
        nominal = ~tr["inj"].astype(bool)
        rule = E.RuleBased(s, names, cfg["confidence"]["rule_z0"], cfg["confidence"]["rule_scale"]).fit(tr["X"][nominal])
        E.save(rule, est_path(mdir, s, "rule"))
        cands = [("rule", rule, "X")]
        for fam in cfg["confidence"]["families"]:
            for fs in cfg["confidence"]["feature_sets"]:
                key = "X" if fs == "static" else "Xt"
                nm = names if fs == "static" else temporal_names(names)
                est = E.Learned(s, fam, fs, nm).fit(tr[key], tr["y"])
                est.calibrate(va[key], va["y"])
                E.save(est, est_path(mdir, s, f"{fam}_{fs}"))
                cands.append((f"{fam}_{fs}", est, key))
        for name, est, key in cands:
            c = est.predict(va[key])
            m = confidence_metrics(c, va["y"], va["inj"].astype(bool), va["d2"])
            m.update(sensor=s, estimator=name, n_params=est.n_params(), size_bytes=E.model_size_bytes(est))
            rows.append(m)
            log(f"  {s:9s} {name:14s} val Brier {m['brier']:.4f} ECE {m['ece']:.4f} "
                f"AUROC(unrel) {m['auroc_unreliable']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(mdir / "val_confidence_metrics.csv", index=False)
    return df


def score_val(cfg, named_specs, workers):
    paths = split_paths(cfg, "val")
    res = pmap(run_episode, [(cfg, str(p), named_specs, cfg["tracks"], None) for p in paths], workers)
    df = pd.DataFrame([r for rs in res for r in rs if "error" not in r])
    df["log_rmse"] = np.log(df["pos_rmse"].clip(lower=1e-4))
    return df


def select(cfg, workers, conf_df, log=print):
    mdir = resolve(cfg, "models")
    fams, fsets = cfg["confidence"]["families"], cfg["confidence"]["feature_sets"]
    size = {f"{fa}_{fs}": conf_df[conf_df.estimator == f"{fa}_{fs}"]["n_params"].sum()
            for fa in fams for fs in fsets}
    specs = []
    for fa in fams:
        for fs in fsets:
            sp = dict(METHODS["learned_" + fa])
            sp["conf"], sp["weight"] = f"{fa}_{fs}", "linear"
            specs.append((f"cand_{fa}_{fs}", sp))
    singles = [(k, METHODS[k]) for k in METHODS if k.startswith("single_")]
    df = score_val(cfg, specs + singles, workers)
    df.to_csv(mdir / "val_fusion_scores.csv", index=False)
    sc = df.groupby("method")["log_rmse"].mean()
    log("validation scores (mean log RMSE):\n" + sc.sort_values().to_string())
    cand = sc[[n for n, _ in specs]].sort_values()
    best = cand.index[0]
    close = [n for n in cand.index if cand[n] - cand.iloc[0] < 0.01]
    best = min(close, key=lambda n: size[n[5:]])
    fa, fs = best[5:].split("_")
    # weight function
    wspecs = []
    for w in ("linear", "power", "gate", "exp"):
        sp = dict(METHODS["learned_" + fa])
        sp["conf"], sp["weight"] = f"{fa}_{fs}", w
        wspecs.append((f"w_{w}", sp))
    dw = score_val(cfg, wspecs, workers)
    dw.to_csv(mdir / "val_weight_scores.csv", index=False)
    wsc = dw.groupby("method")["log_rmse"].mean().sort_values()
    log("weight functions:\n" + wsc.to_string())
    single_best = {tr: df[(df.track == tr) & df.method.str.startswith("single_")]
                   .groupby("method")["log_rmse"].mean().idxmin() for tr in cfg["tracks"]}
    sel = dict(family=fa, feature_set=fs, weight=wsc.index[0][2:], single_best=single_best,
               rule="pre-registered: mean val log(pos RMSE), ties<0.01 -> fewer params",
               val_scores={k: float(v) for k, v in sc.items()},
               val_weight_scores={k: float(v) for k, v in wsc.items()})
    json.dump(sel, open(mdir / "selection.json", "w"), indent=2)
    log(f"SELECTED: {sel['family']} / {sel['feature_set']} / g={sel['weight']}; single_best={single_best}")
    return sel


if __name__ == "__main__":
    args, cfg = parse(__doc__, lambda ap: ap.add_argument("--skip-train", action="store_true"))
    conf_df = train_all(cfg) if not args.skip_train else \
        pd.read_csv(resolve(cfg, "models") / "val_confidence_metrics.csv")
    select(cfg, args.workers, conf_df)
