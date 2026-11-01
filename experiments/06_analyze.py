"""Step 6: statistics + tables (CSV and LaTeX) from measured runs.

Independence: the same (world, trajectory, seed) block appears under every
condition, so pooled tests first average over conditions within a block and
then compare methods across the 30 independent blocks.

    python experiments/06_analyze.py --config configs/matrix.yaml
"""
import json

import numpy as np
import pandas as pd
from _common import parse

from dasf.confidence import estimators as E
from dasf.confidence.bank import est_path
from dasf.evaluation.metrics import confidence_metrics
from dasf.evaluation.stats import friedman, holm, paired_compare, summarize
from dasf.experiment import ALL_CONF_SENSORS, load_dataset
from dasf.utils import resolve

GROUPS = ["nominal", "cam", "lidar", "imu", "odom", "multi"]
MAIN_BASELINES = ["single_best", "fixed_complementary", "ekf", "ekf_gate", "ekf_iae", "rule", "heavy_cnn"]
LABEL = dict(single_best="Single best sensor", fixed_complementary="Fixed-weight fusion", ekf="EKF",
             ekf_gate="EKF + $\\chi^2$ gating", ekf_iae="Innovation-adaptive EKF", rule="Rule-based conf.",
             proposed="Proposed (learned conf.)", heavy_cnn="Heavy neural conf.", oracle="Oracle conf. (UB)")


def group_of(cond):
    if cond == "nominal":
        return "nominal"
    return "multi" if "+" in cond else cond.split(".")[0]


def load(cfg, split):
    runs = resolve(cfg, "runs")
    df = pd.read_csv(runs / f"{split}_metrics.csv")
    if "error" in df:
        df = df[df["error"].isna()].copy()
    sel = json.load(open(runs / f"{split}_run_info.json"))["selection"]
    # Baseline 1: relabel the validation-selected single sensor per track
    extra = []
    for tr, m in (sel or {}).get("single_best", {}).items():
        d = df[(df.method == m) & (df.track == tr)].copy()
        d["method"] = "single_best"
        extra.append(d)
    df = pd.concat([df] + extra, ignore_index=True)
    df["group"] = df["condition"].map(group_of)
    df["block"] = df["world"] + "|" + df["traj"] + "|" + df["seed"].astype(str)
    return df, sel


def fmt(s, prec=3):
    return "TBD" if not np.isfinite(s["mean"]) else f"{s['mean']:.{prec}f} $\\pm$ {s['std']:.{prec}f}"


def table_main(df, metric, tdir, name, methods):
    out, tex = [], []
    for tr in sorted(df.track.unique()):
        for m in methods:
            row = dict(track=tr, method=m)
            for g in GROUPS:
                x = df[(df.track == tr) & (df.method == m) & (df.group == g)][metric]
                s = summarize(x)
                row[g] = fmt(s)
                row[g + "_median"] = s["median"]
            out.append(row)
            tex.append(f"{tr} & {LABEL.get(m, m)} & " + " & ".join(row[g] for g in GROUPS) + " \\\\")
    pd.DataFrame(out).to_csv(tdir / f"{name}.csv", index=False)
    open(tdir / f"{name}.tex", "w").write("\n".join(tex) + "\n")


def block_means(df, method, track, groups, metric):
    d = df[(df.method == method) & (df.track == track) & (df.group.isin(groups))]
    return d.groupby("block")[metric].mean()


def table_tests(df, metric, tdir, name, target, others, groups):
    rows = []
    for tr in sorted(df.track.unique()):
        a = block_means(df, target, tr, groups, metric)
        for b in others:
            bb = block_means(df, b, tr, groups, metric)
            idx = a.index.intersection(bb.index)
            if len(idx) == 0:
                continue
            r = paired_compare(a[idx].values, bb[idx].values)
            r.update(track=tr, target=target, baseline=b, metric=metric, groups="+".join(groups))
            rows.append(r)
    res = pd.DataFrame(rows)
    if len(res):
        for tr in res.track.unique():
            k = res.track == tr
            res.loc[k, "p_holm"] = holm(res.loc[k, "p_wilcoxon"].values)
        tex = [f"{r.track} & {LABEL.get(r.baseline, r.baseline)} & {r.n} & {r.median_diff:.4f} "
               f"[{r.median_diff_ci[0]:.4f}, {r.median_diff_ci[1]:.4f}] & {r.ratio_of_means:.3f} "
               f"[{r.ratio_ci[0]:.3f}, {r.ratio_ci[1]:.3f}] & {r.effect_rank_biserial:.2f} & "
               f"{r.p_holm:.2g} \\\\" for r in res.itertuples()]
        open(tdir / f"{name}.tex", "w").write("\n".join(tex) + "\n")
    res.to_csv(tdir / f"{name}.csv", index=False)
    return res


def table_severity(df, tdir, methods):
    d = df[df.group.isin(["cam", "lidar", "imu", "odom"])].copy()
    d["type"] = d.condition.str.rsplit(".", n=1).str[0]
    d["level"] = d.condition.str.rsplit(".", n=1).str[1].astype(int)
    g = d[d.method.isin(methods)].groupby(["track", "type", "level", "method"])["pos_rmse"]
    t = g.agg(["mean", "std", "median", "count"]).reset_index()
    t.to_csv(tdir / "severity.csv", index=False)
    return t


def table_robustness(df, tdir, methods):
    nom = df[df.condition == "nominal"][["method", "track", "block", "pos_rmse"]].rename(columns={"pos_rmse": "nom"})
    d = df[df.condition != "nominal"].merge(nom, on=["method", "track", "block"])
    d["increase_pct"] = 100 * (d.pos_rmse - d.nom) / d.nom
    t = d[d.method.isin(methods)].groupby(["track", "condition", "method"])["increase_pct"].median().reset_index()
    t.to_csv(tdir / "robustness_increase.csv", index=False)
    return t


def table_recovery(df, tdir, methods):
    d = df[df.method.isin(methods) & (df.condition != "nominal")]
    rows = []
    for (tr, g, m), x in d.groupby(["track", "group", "method"]):
        rec = x[x.recovery_status == "recovered"]
        rows.append(dict(track=tr, group=g, method=m, n=len(x),
                         median_recovery_s=float(rec.recovery_time.median()) if len(rec) else np.nan,
                         frac_disturbed=float((x.recovery_status != "undisturbed").mean()),
                         frac_censored=float((x.recovery_status == "censored").mean()),
                         failure_rate=float(x.failure.mean()), success_rate=float(1 - x.failure.mean())))
    t = pd.DataFrame(rows)
    t.to_csv(tdir / "recovery_failure.csv", index=False)
    return t


def table_friedman(df, tdir, methods):
    rows = []
    for tr in sorted(df.track.unique()):
        mats = [block_means(df, m, tr, GROUPS[1:], "pos_rmse") for m in methods]
        M = pd.concat(mats, axis=1, keys=methods).dropna()
        st, p, ranks = friedman(M.values)
        rows.append(dict(track=tr, n_blocks=len(M), stat=st, p=p, **{f"rank_{m}": r for m, r in zip(methods, ranks)}))
    pd.DataFrame(rows).to_csv(tdir / "friedman.csv", index=False)


def table_confidence(cfg, tdir, sel):
    rows = []
    mdir = resolve(cfg, "models")
    names = ["rule"] + [f"{f}_{s}" for f in cfg["confidence"]["families"] for s in cfg["confidence"]["feature_sets"]] + ["cnn"]
    for s in ALL_CONF_SENSORS:
        te = load_dataset(cfg, "test", s)
        for n in names:
            p = est_path(mdir, s, n)
            if not p.exists():
                continue
            est = E.load(p)
            if n == "cnn":
                if s == "cam":
                    continue                                   # evaluated via the re-render pass only
                X = te["raw"].astype(np.float32) if s.startswith("lidar") else te["Xt"]
            else:
                X = te["X"] if est.feature_set == "static" else te["Xt"]
            c = est.predict(X)
            m = confidence_metrics(c, te["y"], te["inj"].astype(bool), te["d2"])
            m.update(sensor=s, estimator=n, n_params=est.n_params())
            rows.append(m)
    t = pd.DataFrame(rows)
    t.to_csv(tdir / "confidence_quality.csv", index=False)
    return t


NONINF_MARGIN = 1.05     # pre-registered: proposed is non-inferior if upper CI of RMSE ratio < 1.05


def hypothesis_H3_H4(df, tdir, present):
    rows = []
    for tr in sorted(df.track.unique()):
        # H3 accuracy side: proposed vs heavy model (non-inferiority) and vs EKF on NOMINAL data
        for other, groups, tag in (("heavy_cnn", GROUPS[1:], "H3_vs_heavy_degraded"),
                                   ("ekf", ["nominal"], "H3_nominal_cost_vs_ekf")):
            if other not in present:
                continue
            a = block_means(df, "proposed", tr, groups, "pos_rmse")
            b = block_means(df, other, tr, groups, "pos_rmse")
            idx = a.index.intersection(b.index)
            r = paired_compare(a[idx].values, b[idx].values)
            rows.append(dict(track=tr, test=tag, ratio=r["ratio_of_means"], ci_lo=r["ratio_ci"][0],
                             ci_hi=r["ratio_ci"][1], non_inferior=bool(r["ratio_ci"][1] < NONINF_MARGIN),
                             p_wilcoxon=r["p_wilcoxon"], n=r["n"]))
        # H4: relative improvement over EKF, single-sensor groups vs 4-sensor degradation
        if "ekf" in present:
            def rel_impr(groups, cond_filter=None):
                d = df[(df.track == tr) & df.group.isin(groups) & df.method.isin(["ekf", "proposed"])]
                if cond_filter is not None:
                    d = d[d.condition.map(cond_filter)]
                p = d.pivot_table(index=["block", "condition"], columns="method", values="pos_rmse")
                p["impr"] = (p["ekf"] - p["proposed"]) / p["ekf"]
                return p.groupby(level="block")["impr"].mean()
            single = rel_impr(["cam", "lidar", "imu", "odom"])
            multi = rel_impr(["multi"])
            idx = single.index.intersection(multi.index)
            if len(idx):
                r = paired_compare(-single[idx].values, -multi[idx].values)
                rows.append(dict(track=tr, test="H4_single_minus_multi_rel_improvement",
                                 ratio=float(np.median(single[idx] - multi[idx])), ci_lo=np.nan, ci_hi=np.nan,
                                 non_inferior=np.nan, p_wilcoxon=r["p_wilcoxon"], n=r["n"]))
    pd.DataFrame(rows).to_csv(tdir / "hypotheses_H3_H4.csv", index=False)


if __name__ == "__main__":
    args, cfg = parse(__doc__, lambda ap: ap.add_argument("--split", default="test"))
    tdir = resolve(cfg, "tables")
    df, sel = load(cfg, args.split)
    present = set(df.method)
    main = [m for m in MAIN_BASELINES + ["proposed", "oracle"] if m in present]
    for metric, nm in (("pos_rmse", "main_pos_rmse"), ("ori_rmse_deg", "main_ori_rmse"),
                       ("rpe_trans_1s", "main_rpe1"), ("pos_rmse_deg_win", "main_pos_rmse_degwin")):
        table_main(df, metric, tdir, nm, main)
    degraded = GROUPS[1:]
    table_tests(df, "pos_rmse", tdir, "tests_proposed_vs_baselines", "proposed",
                [m for m in MAIN_BASELINES if m in present], degraded)
    for g in degraded:
        table_tests(df, "pos_rmse", tdir, f"tests_group_{g}", "proposed",
                    [m for m in ("ekf", "rule", "ekf_gate") if m in present], [g])
    table_tests(df, "pos_rmse", tdir, "tests_nominal_cost", "proposed", ["ekf"], ["nominal"])
    abl = sorted(m for m in present if m.startswith("abl_"))
    table_tests(df, "pos_rmse", tdir, "tests_ablations", "proposed", abl + ["rule", "ekf"], degraded)
    # H2: recovery (censored runs enter with the remaining episode time -> conservative)
    table_tests(df, "recovery_time", tdir, "tests_H2_recovery", "proposed",
                [m for m in ("ekf", "ekf_gate", "rule") if m in present], degraded)
    hypothesis_H3_H4(df, tdir, present)
    table_severity(df, tdir, main)
    table_robustness(df, tdir, main)
    table_recovery(df, tdir, main)
    table_friedman(df, tdir, [m for m in main if m != "oracle"])
    table_confidence(cfg, tdir, sel)
    print(f"tables written to {tdir}")
