"""Confidence-to-weight functions g: [0,1] -> [0,1] (Ablation 10).

    linear : g(c) = c
    power  : g(c) = c^gamma                       (sharper down-weighting)
    gate   : g(c) = 1[c >= tau]                   (hard accept / reject)
    exp    : g(c) = exp(beta (c - 1))             (never fully rejects)

A measurement with g(c) < weight_floor is skipped. The effective covariance
is R_eff = R0 / max(g(c), weight_floor).
"""
from __future__ import annotations

import numpy as np


def make_weight_fn(cfg: dict, name: str | None = None):
    f = cfg["fusion"]
    name = name or f["weight_function"]
    if name == "linear":
        return lambda c: float(c)
    if name == "power":
        gam = f["weight_power"]
        return lambda c: float(c) ** gam
    if name == "gate":
        tau = f["weight_gate_tau"]
        return lambda c: 1.0 if c >= tau else 0.0
    if name == "exp":
        beta = f["weight_exp_beta"]
        return lambda c: float(np.exp(beta * (c - 1.0)))
    raise ValueError(name)


def normalized_weights(ws: dict) -> dict:
    """Convex weights sum_i w_i = 1 (used only by the complementary baseline)."""
    s = sum(ws.values())
    if s <= 0:
        return {k: 1.0 / len(ws) for k in ws}
    return {k: v / s for k, v in ws.items()}
