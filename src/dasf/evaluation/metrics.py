"""Trajectory, robustness and confidence metrics.

Localisation errors are computed in the map/world frame WITHOUT trajectory
alignment: the initial pose is known in both tracks, so an SE(2)/Umeyama
alignment (needed for monocular VO evaluation, Zhang & Scaramuzza 2018)
would hide real errors.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from ..utils import wrap

ERR_CAP = 100.0          # m, caps diverged estimates so means stay finite


def _rmse(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan


def rpe(gt, est, k):
    """Relative pose error over index offset k -> (trans RMSE [m], rot RMSE [deg])."""
    if len(gt) <= k:
        return np.nan, np.nan
    i, j = np.arange(len(gt) - k), np.arange(k, len(gt))

    def rel(P):
        dx, dy = P[j, 0] - P[i, 0], P[j, 1] - P[i, 1]
        c, s = np.cos(P[i, 2]), np.sin(P[i, 2])
        return c * dx + s * dy, -s * dx + c * dy, wrap(P[j, 2] - P[i, 2])

    gx, gy, gt_ = rel(gt)
    ex, ey, et = rel(est)
    return _rmse(np.hypot(ex - gx, ey - gy)), float(np.rad2deg(_rmse(wrap(et - gt_))))


def recovery_time(t, err, t_off, thr, hold):
    """Time after t_off until err stays <= thr for `hold` seconds.

    Returns (seconds, status): status 'undisturbed' (err never exceeded thr in
    the degradation window -> 0 s), 'recovered', or 'censored' (never
    recovered before the episode ended -> value = remaining duration).
    """
    after = t >= t_off
    ta, ea = t[after], err[after]
    if len(ta) == 0:
        return np.nan, "censored"
    dt = np.median(np.diff(t))
    n_hold = max(1, int(round(hold / dt)))
    ok = ea <= thr
    run = 0
    for k in range(len(ok)):
        run = run + 1 if ok[k] else 0
        if run >= n_hold:
            return float(ta[k - n_hold + 1] - t_off), "recovered"
    return float(ta[-1] - t_off), "censored"


def episode_metrics(ep, out, cfg, t_on=None, t_off=None) -> dict:
    ev = cfg["evaluation"]
    t = ep["imu_t"]
    gt = np.column_stack([ep["gt_x"], ep["gt_y"], ep["gt_th"]])
    est = out["est"][:, :3]
    e_pos = np.minimum(np.hypot(est[:, 0] - gt[:, 0], est[:, 1] - gt[:, 1]), ERR_CAP)
    e_pos = np.where(np.isfinite(e_pos), e_pos, ERR_CAP)
    e_ori = np.abs(wrap(est[:, 2] - gt[:, 2]))
    dt = cfg["sim"]["dt"]
    m = dict(pos_rmse=_rmse(e_pos), pos_mae=float(np.mean(e_pos)), pos_max=float(e_pos.max()),
             pos_final=float(e_pos[-1]), ori_rmse_deg=float(np.rad2deg(_rmse(e_ori))))
    path = float(np.sum(np.hypot(np.diff(gt[:, 0]), np.diff(gt[:, 1]))))
    m["drift_pct"] = 100.0 * m["pos_final"] / max(path, 1e-6)
    for d in ev["rpe_deltas"]:
        tr, rr = rpe(gt, np.where(np.isfinite(est), est, 0.0), int(round(d / dt)))
        m[f"rpe_trans_{d:g}s"], m[f"rpe_rot_{d:g}s_deg"] = tr, rr
    # windows
    if t_on is None:
        t_on, t_off = cfg["sim"]["deg_window"]
    n0, n1 = cfg["sim"]["nominal_eval_window"][0], t_on
    wn = (t >= n0) & (t < n1)
    wd = (t >= t_on) & (t < t_off)
    wp = t >= t_off
    m["pos_rmse_nominal_win"] = _rmse(e_pos[wn])
    m["pos_rmse_deg_win"] = _rmse(e_pos[wd])
    m["pos_rmse_post_win"] = _rmse(e_pos[wp])
    m["ori_rmse_deg_win_deg"] = float(np.rad2deg(_rmse(e_ori[wd])))
    p95 = float(np.quantile(e_pos[wn], 0.95)) if wn.any() else np.nan
    thr = max(ev["recovery_factor"] * p95, p95 + ev["recovery_margin"])
    disturbed = bool((e_pos[wd] > thr).any()) if wd.any() else False
    if disturbed:
        rt, status = recovery_time(t, e_pos, t_off, thr, ev["recovery_hold"])
    else:
        rt, status = 0.0, "undisturbed"
    m.update(recovery_time=rt, recovery_status=status, recovery_thr=thr,
             failure=bool(e_pos.max() > ev["failure_error"]),
             path_length=path)
    Pd = out.get("Pdiag")
    if Pd is not None and np.any(Pd):
        ex = est[:, 0] - gt[:, 0]
        ey = est[:, 1] - gt[:, 1]
        eth = wrap(est[:, 2] - gt[:, 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            nees = ex ** 2 / Pd[:, 0] + ey ** 2 / Pd[:, 1] + eth ** 2 / Pd[:, 2]   # diagonal approx.
        m["nees_mean"] = float(np.nanmean(np.where(np.isfinite(nees), nees, np.nan)))
    m.update(wall_time=out["wall_time"], rtf=out["wall_time"] / out["sim_time"],
             conf_time=out["conf_time"])
    for k in ("n_skipped", "n_rejected", "n_updates"):
        for s, v in out["stats"][k].items():
            m[f"{k}_{s}"] = v
    m["icp_fail"] = out["stats"]["icp_fail"]
    return m


# ============================================================================ confidence metrics
def ece(p, y, n_bins=15):
    """Expected calibration error (Naeini et al. 2015; Guo et al. 2017), equal-width bins."""
    p, y = np.asarray(p), np.asarray(y)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        s = idx == b
        if s.any():
            tot += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(tot)


def reliability_curve(p, y, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        s = idx == b
        if s.any():
            rows.append((0.5 * (bins[b] + bins[b + 1]), p[s].mean(), y[s].mean(), int(s.sum())))
    return np.array(rows)


def confidence_metrics(c, y_reliable, injected, d2) -> dict:
    c = np.asarray(c, float)
    out = dict(n=len(c), frac_reliable=float(np.mean(y_reliable)),
               brier=float(np.mean((c - y_reliable) ** 2)), ece=ece(c, y_reliable),
               acc_at_0p5=float(np.mean((c >= 0.5) == (y_reliable == 1))))
    unrel = 1 - y_reliable
    for name, pos in (("unreliable", unrel), ("injected", injected.astype(int))):
        if 0 < pos.sum() < len(pos):
            out[f"auroc_{name}"] = float(roc_auc_score(pos, 1 - c))
            out[f"auprc_{name}"] = float(average_precision_score(pos, 1 - c))
            out[f"prevalence_{name}"] = float(pos.mean())
        else:
            out[f"auroc_{name}"] = out[f"auprc_{name}"] = np.nan
    rho = spearmanr(c, -np.asarray(d2)).correlation if np.std(c) > 0 else np.nan
    out["spearman_conf_vs_neg_d2"] = float(rho)
    return out
