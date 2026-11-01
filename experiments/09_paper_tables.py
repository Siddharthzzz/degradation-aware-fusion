"""Step 9: LaTeX rows for the paper tables that 06/07 do not emit directly
(recovery, confidence quality, ablations, compute) and \\newcommand macros for
the headline numbers quoted in the text.  Everything is read from tables/ and
results/runs/; nothing is typed by hand.

    python experiments/09_paper_tables.py --config configs/reduced.yaml
"""
import json

import numpy as np
import pandas as pd
from _common import parse

from dasf.utils import resolve

TRACKS = ["map", "odom"]
REC_METHODS = [("ekf", "EKF"), ("ekf_gate", "EKF + $\\chi^2$ gating"), ("ekf_iae", "IAE-EKF"),
               ("rule", "Rule-based"), ("proposed", "Proposed"), ("heavy_cnn", "Heavy neural"),
               ("oracle", "Oracle")]
ABL = [("ekf", "No confidence (EKF)"), ("rule", "Rule-based confidence"),
       ("abl_no_cam_conf", "Without camera confidence"), ("abl_no_lidar_conf", "Without LiDAR confidence"),
       ("abl_no_imu_conf", "Without IMU confidence"), ("abl_no_odom_conf", "Without odometry confidence"),
       ("abl_static", "Static instead of temporal features"), ("abl_w_power", "$g=c^2$ instead of gate"),
       ("abl_w_exp", "$g=\\exp(4(c-1))$ instead of gate"),
       ("abl_conf_complementary", "Convex (sum-to-one) weighting"),
       ("abl_conf_plus_gate", "Confidence + $\\chi^2$ gating")]
SENSORS = [("cam", "Camera"), ("lidar", "LiDAR (map)"), ("lidar_rel", "LiDAR (scan)"), ("imu", "IMU"),
           ("odom", "Odometry")]


def ci(s):
    return tuple(float(v) for v in str(s).strip("()").split(","))


def pfmt(p):
    if not np.isfinite(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def macro(name, val):
    return f"\\newcommand{{\\{name}}}{{{val}}}"


if __name__ == "__main__":
    args, cfg = parse(__doc__)
    T = resolve(cfg, "tables")
    runs = resolve(cfg, "runs")
    df = pd.read_csv(runs / "test_metrics.csv")
    if "error" in df:
        df = df[df["error"].isna()]
    sel = json.load(open(resolve(cfg, "models") / "selection.json"))
    M = []

    # ---------------------------------------------------------------- recovery / failure
    deg = df[df.condition != "nominal"]
    rows = []
    for tr in TRACKS:
        for m, lab in REC_METHODS:
            x = deg[(deg.track == tr) & (deg.method == m)]
            if not len(x):
                continue
            rec = x[x.recovery_status == "recovered"].recovery_time
            dist = (x.recovery_status != "undisturbed").mean()
            cens = (x.recovery_status == "censored").mean()
            med = f"{rec.median():.2f}" if len(rec) else "--"
            rows.append(f"{tr} & {lab} & {med} & {100 * dist:.0f}\\% & {100 * cens:.1f}\\% & "
                        f"{100 * x.failure.mean():.1f}\\%\\\\")
    open(T / "paper_recovery.tex", "w").write("\n".join(rows) + "\n")

    # ---------------------------------------------------------------- confidence quality
    cq = pd.read_csv(T / "confidence_quality.csv")
    prop = f"{sel['family']}_{sel['feature_set']}"
    rows = []
    for s, slab in SENSORS:
        for est, elab in (("rule", "Rule-based"), (prop, "Proposed"), ("cnn", "Heavy")):
            r = cq[(cq.sensor == s) & (cq.estimator == est)]
            if not len(r):
                continue
            r = r.iloc[0]
            rows.append(f"{slab} & {elab} & {r.brier:.3f} & {r.ece:.3f} & {r.auroc_unreliable:.3f} & "
                        f"{r.auroc_injected:.3f} & {r.spearman_conf_vs_neg_d2:.2f}\\\\")
    open(T / "paper_confidence.tex", "w").write("\n".join(rows) + "\n")
    # best learned family per sensor by Brier (reported in text)
    fam = cq[~cq.estimator.isin(["rule", "cnn"])]
    for s, _ in SENSORS:
        f = fam[fam.sensor == s]
        if len(f):
            b = f.loc[f.brier.idxmin()]
            M.append(macro(f"BestConf{s.replace('_', '')}", b.estimator.replace("_", "/")))

    # ---------------------------------------------------------------- ablations (both tracks)
    ab = pd.read_csv(T / "tests_ablations.csv")
    rows = []
    for m, lab in ABL:
        cells = []
        for tr in TRACKS:
            r = ab[(ab.track == tr) & (ab.baseline == m)]
            if not len(r):
                cells += ["--", "--"]
                continue
            r = r.iloc[0]
            lo, hi = ci(r.ratio_ci)
            cells += [f"{r.ratio_of_means:.2f} [{lo:.2f}, {hi:.2f}]", pfmt(r.p_holm)]
        rows.append(f"{lab} & " + " & ".join(cells) + "\\\\")
    open(T / "paper_ablation.tex", "w").write("\n".join(rows) + "\n")

    # ---------------------------------------------------------------- compute
    ce = pd.read_csv(T / "compute_estimators.csv")
    fs = sel["feature_set"]
    rows = []
    for est, lab in (("rule", "Rule-based"), (f"lr_{fs}", "Logistic regression"), (f"rf_{fs}", "Random forest"),
                     (f"gbm_{fs}", "Gradient boosting"), (f"mlp_{fs}", "MLP $2\\times32$"), ("cnn", "Heavy neural")):
        r = ce[ce.estimator == est]
        if not len(r):
            continue
        lat = r.latency_median_us.median()
        latcell = f"{lat:.0f}"
        if est == "cnn" and r.gpu_latency_median_us.notna().any():
            latcell += f" / {r.gpu_latency_median_us.median():.0f}"
        star = "$^\\star$" if est == prop else ""
        rows.append(f"{lab}{star} & {int(r.n_params.sum()):,} & {r.size_bytes.sum() / 1024:.1f} & {latcell} & "
                    f"{r.cpu_duty_pct.sum():.2f}\\\\".replace(",", "{,}"))
        key = "Prop" if est == prop else ("Heavy" if est == "cnn" else None)
        if key:
            M += [macro(f"{key}Params", f"{int(r.n_params.sum()):,}".replace(",", "{,}")),
                  macro(f"{key}LatUs", f"{lat:.0f}"), macro(f"{key}Duty", f"{r.cpu_duty_pct.sum():.2f}"),
                  macro(f"{key}SizeKB", f"{r.size_bytes.sum() / 1024:.0f}")]
    open(T / "paper_compute.tex", "w").write("\n".join(rows) + "\n")
    cf = pd.read_csv(T / "compute_filters.csv")
    for tr in TRACKS:
        r = cf[(cf.method == "proposed") & (cf.track == tr)]
        if len(r):
            M.append(macro(f"PropRTF{tr}", f"{r.rtf.iloc[0]:.3f}"))

    # ---------------------------------------------------------------- headline numbers
    tp = pd.read_csv(T / "tests_proposed_vs_baselines.csv")
    for tr in TRACKS:
        for b, key in (("ekf", "Ekf"), ("fixed_complementary", "Fixed"), ("rule", "Rule"), ("ekf_gate", "Gate"),
                       ("ekf_iae", "Iae"), ("heavy_cnn", "Heavy"), ("single_best", "Single")):
            r = tp[(tp.track == tr) & (tp.baseline == b)]
            if not len(r):
                continue
            r = r.iloc[0]
            lo, hi = ci(r.ratio_ci)
            M += [macro(f"R{key}{tr}", f"{r.ratio_of_means:.2f}"),
                  macro(f"R{key}{tr}CI", f"[{lo:.2f}, {hi:.2f}]"),
                  macro(f"P{key}{tr}", pfmt(r.p_holm)),
                  macro(f"Pct{key}{tr}", f"{100 * (1 - r.ratio_of_means):.0f}")]
    nc = pd.read_csv(T / "tests_nominal_cost.csv")
    for tr in TRACKS:
        r = nc[nc.track == tr]
        if len(r):
            r = r.iloc[0]
            lo, hi = ci(r.ratio_ci)
            M += [macro(f"RNom{tr}", f"{r.ratio_of_means:.2f}"), macro(f"RNom{tr}CI", f"[{lo:.2f}, {hi:.2f}]"),
                  macro(f"PNom{tr}", pfmt(r.p_holm if np.isfinite(r.p_holm) else r.p_wilcoxon))]
    h = pd.read_csv(T / "hypotheses_H3_H4.csv")
    for tr in TRACKS:
        r = h[(h.track == tr) & (h.test == "H3_vs_heavy_degraded")]
        if len(r):
            M += [macro(f"HNonInf{tr}", "yes" if bool(r.non_inferior.iloc[0]) else "no"),
                  macro(f"HRatioCIhi{tr}", f"{r.ci_hi.iloc[0]:.2f}")]
        r = h[(h.track == tr) & (h.test == "H4_single_minus_multi_rel_improvement")]
        if len(r):
            M += [macro(f"HFour{tr}", f"{100 * r.ratio.iloc[0]:.1f}"), macro(f"HFourP{tr}", pfmt(r.p_wilcoxon.iloc[0]))]
    rec = pd.read_csv(T / "tests_H2_recovery.csv")
    for tr in TRACKS:
        r = rec[(rec.track == tr) & (rec.baseline == "ekf")]
        if len(r):
            r = r.iloc[0]
            lo, hi = ci(r.ratio_ci)
            M += [macro(f"HTwoDiff{tr}", f"{r.median_diff:.2f}"), macro(f"HTwoP{tr}", pfmt(r.p_holm)),
                  macro(f"HTwoR{tr}", f"{r.ratio_of_means:.2f}"), macro(f"HTwoR{tr}CI", f"[{lo:.2f}, {hi:.2f}]")]
    # held-out world W3 only (plan L6): proposed vs EKF over degraded conditions
    from dasf.evaluation.stats import paired_compare
    for tr in TRACKS:
        d = deg[(deg.track == tr) & (deg.world == "W3_featurepoor") & deg.method.isin(["ekf", "proposed"])].copy()
        d["block"] = d.world + "|" + d.traj + "|" + d.seed.astype(str)
        p = d.groupby(["block", "method"]).pos_rmse.mean().unstack()
        if {"ekf", "proposed"} <= set(p.columns) and len(p):
            r = paired_compare(p.proposed.values, p.ekf.values)
            M += [macro(f"RWthree{tr}", f"{r['ratio_of_means']:.2f}"),
                  macro(f"RWthree{tr}CI", f"[{r['ratio_ci'][0]:.2f}, {r['ratio_ci'][1]:.2f}]"),
                  macro(f"NWthree{tr}", str(r["n"]))]
    M.append(macro("SelFamily", sel["family"].upper()))
    M.append(macro("SelFeat", sel["feature_set"]))
    M.append(macro("SelWeight", sel["weight"]))
    M.append(macro("NTestEpisodes", str(df.drop_duplicates(["condition", "world", "traj", "seed"]).shape[0])))
    open(T / "paper_numbers.tex", "w").write("\n".join(M) + "\n")

    # ================================================================ compact tables (5-page version)
    # main: mean position RMSE per group, best deployable method per column in bold
    grp = lambda c: "nominal" if c == "nominal" else ("multi" if "+" in c else c.split(".")[0])
    d = df.copy()
    d["group"] = d.condition.map(grp)
    sb = (sel or {}).get("single_best", {})
    main = [("single_best", "Single best"), ("fixed_complementary", "Fixed-weight"), ("ekf", "EKF"),
            ("ekf_gate", "EKF+$\\chi^2$"), ("ekf_iae", "IAE-EKF"), ("rule", "Rule conf."),
            ("heavy_cnn", "Heavy CNN"), ("proposed", "\\textbf{Proposed}"), ("oracle", "Oracle (UB)")]
    G = ["nominal", "cam", "lidar", "imu", "odom", "multi"]
    rows = []
    for tr in TRACKS:
        tab = {}
        for m, _ in main:
            mm = sb.get(tr, m) if m == "single_best" else m
            x = d[(d.track == tr) & (d.method == mm)]
            tab[m] = [x[x.group == g].pos_rmse.mean() for g in G]
        best = [min(tab[m][j] for m, _ in main if m != "oracle") for j in range(len(G))]
        for i, (m, lab) in enumerate(main):
            cells = [(f"\\textbf{{{v:.3f}}}" if (m != "oracle" and v == best[j]) else f"{v:.3f}")
                     for j, v in enumerate(tab[m])]
            mm = sb.get(tr, m) if m == "single_best" else m
            x = d[(d.track == tr) & (d.method == mm) & (d.group != "nominal")]
            cells.append(f"{100 * x.failure.mean():.0f}")
            head = f"\\multirow{{{len(main)}}}{{*}}{{\\rotatebox{{90}}{{{dict(map='map', odom='map-free')[tr]}}}}}" if i == 0 else ""
            rows.append(f"{head} & {lab} & " + " & ".join(cells) + "\\\\")
        if tr != TRACKS[-1]:
            rows.append("\\midrule")
    open(T / "paper5_main.tex", "w").write("\n".join(rows) + "\n")

    # comparison: proposed / X (ratio of means, 95% CI), both tracks; * = p_Holm < 0.05
    def cell(tab, tr, b):
        r = tab[(tab.track == tr) & (tab.baseline == b)]
        if not len(r):
            return "--"
        r = r.iloc[0]
        lo, hi = ci(r.ratio_ci)
        star = "$^\\ast$" if np.isfinite(r.p_holm) and r.p_holm < 0.05 else ""
        return f"{r.ratio_of_means:.2f} [{lo:.2f}, {hi:.2f}]{star}"
    rows = ["\\multicolumn{3}{@{}l}{\\emph{Baselines}}\\\\"]
    for b, lab in (("single_best", "Single best sensor"), ("fixed_complementary", "Fixed-weight fusion"),
                   ("ekf", "EKF"), ("ekf_gate", "EKF + $\\chi^2$ gating"), ("ekf_iae", "IAE-EKF"),
                   ("rule", "Rule-based conf."), ("heavy_cnn", "Heavy CNN conf.")):
        rows.append(f"{lab} & {cell(tp, 'map', b)} & {cell(tp, 'odom', b)}\\\\")
    rows.append("\\midrule\\multicolumn{3}{@{}l}{\\emph{Ablations (variant of the proposed method)}}\\\\")
    for b, lab in (("abl_no_cam_conf", "No camera conf."), ("abl_no_lidar_conf", "No LiDAR conf."),
                   ("abl_no_imu_conf", "No IMU conf."), ("abl_no_odom_conf", "No odometry conf."),
                   ("abl_static", "Static features"), ("abl_w_power", "$g=c^2$"),
                   ("abl_w_exp", "$g=e^{4(c-1)}$"), ("abl_conf_complementary", "Convex weighting"),
                   ("abl_conf_plus_gate", "Conf.\\ + $\\chi^2$ gating")):
        rows.append(f"{lab} & {cell(ab, 'map', b)} & {cell(ab, 'odom', b)}\\\\")
    open(T / "paper5_compare.tex", "w").write("\n".join(rows) + "\n")

    # edge efficiency: cost of each estimator + fused accuracy (ratio of mean block RMSE vs EKF, degraded)
    blk = deg.assign(block=deg.world + "|" + deg.traj + "|" + deg.seed.astype(str))
    bm = {tr: blk[blk.track == tr].groupby(["method", "block"]).pos_rmse.mean().unstack(0) for tr in TRACKS}
    rr = lambda m: " / ".join(f"{bm[tr][m].mean() / bm[tr]['ekf'].mean():.2f}" for tr in TRACKS)
    rows = []
    for est, lab, meths in (("rule", "Rule-based", ["rule"]),
                            (f"lr_{fs}", "Logistic regression", ["learned_lr"]),
                            (f"rf_{fs}", "Random forest", ["learned_rf"]),
                            (f"gbm_{fs}", "Gradient boosting", ["learned_gbm"]),
                            (f"mlp_{fs}", "MLP $2\\times32$", ["learned_mlp", "proposed"]),
                            ("cnn", "Heavy CNN/MLP", ["heavy_cnn"])):
        r = ce[ce.estimator == est]
        lat = f"{r.latency_median_us.median():.0f}"
        if est == "cnn" and r.gpu_latency_median_us.notna().any():
            lat += f"/{r.gpu_latency_median_us.median():.0f}$^\\dagger$"
        cost = (f"{int(r.n_params.sum()):,}".replace(",", "{,}") + f" & {r.size_bytes.sum() / 1024:,.0f}".replace(",", "{,}")
                + f" & {lat} & {r.cpu_duty_pct.sum():.1f}")
        for k, m in enumerate(meths):
            name = lab + (", $g=c$" if m == "learned_mlp" else (", gate$^\\star$" if m == "proposed" else ""))
            rows.append(f"{name} & " + (cost if k == 0 else "\\multicolumn{4}{c}{(same estimator)}") + f" & {rr(m)}\\\\")
    open(T / "paper5_efficiency.tex", "w").write("\n".join(rows) + "\n")

    # recovery / failure, both tracks side by side
    rows = []
    for m, lab in REC_METHODS:
        cells = []
        for tr in TRACKS:
            x = deg[(deg.track == tr) & (deg.method == m)]
            cells += [f"{100 * (x.recovery_status == 'censored').mean():.0f}", f"{100 * x.failure.mean():.0f}"]
        rows.append(f"{lab} & " + " & ".join(cells) + "\\\\")
    open(T / "paper5_recovery.tex", "w").write("\n".join(rows) + "\n")
    print(f"paper tables + {len(M)} macros written to {T}")
