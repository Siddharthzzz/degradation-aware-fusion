"""Statistical analysis for PAIRED experiments.

Design: every method is run on exactly the same episodes (same world,
trajectory, noise seed, degradation), so comparisons are paired by episode.
Localisation errors are positive and typically right-skewed with occasional
divergences, so the PRIMARY test is the two-sided Wilcoxon signed-rank test
(Wilcoxon 1945) on paired differences, with the matched-pairs rank-biserial
correlation as effect size and percentile-bootstrap CIs (Efron & Tibshirani
1993) for the median paired difference and the ratio of means.  A paired
t-test is reported only as a secondary check when a Shapiro-Wilk test does
not reject normality of the (log) differences.  Family-wise error across the
comparisons in one table is controlled with Holm's step-down procedure
(Holm 1979).  Omnibus comparison of k methods over many blocks uses the
Friedman test (Friedman 1937) as recommended by Demsar (2006).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(x, stat=np.median, n=10_000, alpha=0.05, seed=0):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)])
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def ratio_of_means_ci(a, b, n=10_000, alpha=0.05, seed=0):
    """CI of mean(a)/mean(b) with paired resampling."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return float(np.quantile(r, alpha / 2)), float(np.quantile(r, 1 - alpha / 2))


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation (Kerby 2014 formulation):
    r = (sum of ranks of positive diffs - sum of ranks of negative diffs) / total rank sum."""
    d = np.asarray(d)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def paired_compare(a, b, seed=0) -> dict:
    """Compare method A against B on paired samples (lower is better).

    Negative median difference / negative effect size => A better than B.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    out = dict(n=int(len(d)), mean_a=float(a.mean()), mean_b=float(b.mean()),
               median_a=float(np.median(a)), median_b=float(np.median(b)),
               median_diff=float(np.median(d)), median_diff_ci=bootstrap_ci(d, seed=seed),
               ratio_of_means=float(a.mean() / b.mean()) if b.mean() > 0 else np.nan,
               ratio_ci=ratio_of_means_ci(a, b, seed=seed) if b.mean() > 0 else (np.nan, np.nan),
               effect_rank_biserial=rank_biserial(d),
               frac_a_better=float(np.mean(d < 0)))
    if len(d) >= 6 and np.any(d != 0):
        out["p_wilcoxon"] = float(stats.wilcoxon(a, b, zero_method="pratt", alternative="two-sided").pvalue)
        ld = np.log(np.maximum(a, 1e-9)) - np.log(np.maximum(b, 1e-9))
        sh = stats.shapiro(ld).pvalue if len(ld) >= 8 else np.nan
        out["p_shapiro_logdiff"] = float(sh)
        out["p_ttest_log"] = float(stats.ttest_rel(np.log(np.maximum(a, 1e-9)),
                                                   np.log(np.maximum(b, 1e-9))).pvalue)
    else:
        out["p_wilcoxon"] = out["p_shapiro_logdiff"] = out["p_ttest_log"] = np.nan
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaNs passed through)."""
    p = np.asarray(pvals, float)
    adj = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        adj[i] = running
    return adj


def friedman(matrix):
    """matrix: (n_blocks, k_methods), lower is better. Returns stat, p, mean ranks."""
    M = np.asarray(matrix, float)
    M = M[np.all(np.isfinite(M), axis=1)]
    if len(M) < 2 or M.shape[1] < 3:
        return np.nan, np.nan, np.full(M.shape[1], np.nan)
    st, p = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    ranks = np.apply_along_axis(stats.rankdata, 1, M).mean(0)
    return float(st), float(p), ranks


def summarize(x) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, mean=np.nan, std=np.nan, median=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    lo, hi = bootstrap_ci(x, np.mean)
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                median=float(np.median(x)), iqr=float(np.subtract(*np.quantile(x, [0.75, 0.25]))),
                ci_lo=lo, ci_hi=hi)
