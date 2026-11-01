"""Wheel-odometry degradation models.

Slip is modelled on the ENCODER side (wheels turn more than the ground
motion implies), so the ground-truth trajectory is unchanged and paired
comparisons across conditions remain valid.
"""
from __future__ import annotations

import numpy as np


def slip_factors(t, events, rng):
    """Per-wheel multiplicative factor >= 1 applied to wheel-angle increments."""
    fr = np.ones(len(t))
    fl = np.ones(len(t))
    for e in events:
        if e.sensor != "odom" or e.kind not in ("slip", "bumps"):
            continue
        m = e.active(t)
        if not m.any():
            continue
        n = m.sum()
        if e.kind == "slip":
            # slowly varying, asymmetric slip ratio s in [0, 0.95)
            base = e.value
            wobble = np.convolve(rng.standard_normal(n), np.ones(50) / 50, mode="same") * 3.0
            sr = np.clip(base * (1.0 + 0.5 * wobble), 0.0, 0.95)
            sl = np.clip(0.5 * base * (1.0 - 0.5 * wobble), 0.0, 0.95)
            fr[m] /= 1.0 - sr
            fl[m] /= 1.0 - sl
        elif e.kind == "bumps":
            dt = np.median(np.diff(t))
            hits = rng.random(n) < e.value * dt * 2.0
            spike = np.convolve(hits.astype(float), np.ones(15), mode="same")
            amp = 1.0 + e.value * 0.5 * spike * rng.uniform(0.5, 1.5, n)
            side = rng.random(n) < 0.5
            fr[m] *= np.where(side, amp, 1.0)
            fl[m] *= np.where(~side, amp, 1.0)
    return fr, fl


def apply_encoder(t_s, ticks_r, ticks_l, events, rng):
    """Encoder-noise and stuck/dropout on sampled tick counts."""
    tr, tl = ticks_r.astype(float), ticks_l.astype(float)
    valid = np.ones(len(t_s), bool)
    for e in events:
        if e.sensor != "odom":
            continue
        m = e.active(t_s)
        if not m.any():
            continue
        if e.kind == "encnoise":
            s = 2.0 * np.sqrt(e.value ** 2 - 1.0)        # ticks std
            tr[m] += s * rng.standard_normal(m.sum())
            tl[m] += s * rng.standard_normal(m.sum())
        elif e.kind == "stuck":
            i0 = max(np.argmax(m) - 1, 0)
            tr[m], tl[m] = tr[i0], tl[i0]
        elif e.kind == "dropout":
            valid[m] = False
    return np.round(tr), np.round(tl), valid
