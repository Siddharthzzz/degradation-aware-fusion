"""Differential-drive ground truth: route planning on a roadmap + pure pursuit.

The ground-truth trajectory depends ONLY on (world, trajectory kind, seed).
It is identical across all degradation conditions and all fusion methods, so
every comparison is paired.
"""
from __future__ import annotations

import numpy as np

from ..utils import rng_stream, wrap


def _route(world, kind: str, seed: int, n_nodes: int = 40):
    rng = rng_stream(seed, "route")
    if kind == "loop":
        start = int(rng.integers(len(world.loop)))
        seq = [world.loop[(start + i) % len(world.loop)] for i in range(n_nodes)]
        if rng.random() < 0.5:                 # random direction of travel
            seq = [world.loop[(start - i) % len(world.loop)] for i in range(n_nodes)]
    elif kind == "random":
        nbr = {i: set() for i in range(len(world.waypoints))}
        for i, j in world.edges:
            nbr[i].add(j)
            nbr[j].add(i)
        cur = int(rng.choice(world.loop))
        seq, prev = [cur], None
        for _ in range(n_nodes - 1):
            opts = sorted(nbr[cur] - ({prev} if len(nbr[cur]) > 1 else set()))
            prev, cur = cur, int(rng.choice(opts))
            seq.append(cur)
    else:
        raise ValueError(kind)
    return world.waypoints[seq]


def generate_trajectory(world, kind: str, seed: int, cfg: dict) -> dict:
    """Simulate the robot following a roadmap route with pure pursuit.

    Returns arrays at dt: t, x, y, th, v, w, a (a = dv/dt).
    """
    rc, sc = cfg["robot"], cfg["sim"]
    dt, T = sc["dt"], sc["duration"]
    n = int(round(T / dt)) + 1
    path = _route(world, kind, seed)
    rng = rng_stream(seed, "speed")
    v_nom = rc["v_nominal"] * (1.0 + 0.1 * rng.uniform(-1, 1))

    x, y = path[0]
    th = np.arctan2(path[1, 1] - y, path[1, 0] - x)
    v = w = 0.0
    seg = 0
    out = np.zeros((n, 7))
    L = rc["lookahead"]
    for k in range(n):
        t = k * dt
        # advance segment when the projection passes its end
        while seg < len(path) - 2:
            a, b = path[seg], path[seg + 1]
            ab = b - a
            u = np.dot([x - a[0], y - a[1]], ab) / np.dot(ab, ab)
            if u >= 1.0 or np.hypot(x - b[0], y - b[1]) < 0.5 * L:
                seg += 1
            else:
                break
        a, b = path[seg], path[seg + 1]
        ab = b - a
        u = np.clip(np.dot([x - a[0], y - a[1]], ab) / np.dot(ab, ab), 0, 1)
        # lookahead point along the polyline
        rem, s_idx, p = L, seg, a + u * ab
        while True:
            q = path[s_idx + 1]
            dseg = np.linalg.norm(q - p)
            if dseg >= rem or s_idx >= len(path) - 2:
                tgt = p + (q - p) * min(1.0, rem / max(dseg, 1e-9))
                break
            rem -= dseg
            s_idx += 1
            p = path[s_idx]
        dx, dy = tgt[0] - x, tgt[1] - y
        lx = np.cos(th) * dx + np.sin(th) * dy
        ly = -np.sin(th) * dx + np.cos(th) * dy
        head_err = np.arctan2(ly, lx)
        if abs(head_err) > np.deg2rad(60):         # turn in place
            v_des, w_des = 0.0, np.sign(head_err) * 0.6 * rc["w_max"]
        else:
            kappa = 2.0 * ly / max(lx * lx + ly * ly, 1e-6)
            v_des = v_nom / (1.0 + 1.5 * abs(kappa))
            w_des = v_des * kappa
        v_new = v + np.clip(v_des - v, -rc["a_max"] * dt, rc["a_max"] * dt)
        w_new = w + np.clip(w_des - w, -rc["alpha_max"] * dt, rc["alpha_max"] * dt)
        v_new = np.clip(v_new, 0.0, rc["v_max"])
        w_new = np.clip(w_new, -rc["w_max"], rc["w_max"])
        acc = (v_new - v) / dt if k > 0 else 0.0
        v, w = v_new, w_new
        out[k] = (t, x, y, th, v, w, acc)
        # exact unicycle integration over dt
        if abs(w) > 1e-9:
            x += v / w * (np.sin(th + w * dt) - np.sin(th))
            y -= v / w * (np.cos(th + w * dt) - np.cos(th))
        else:
            x += v * np.cos(th) * dt
            y += v * np.sin(th) * dt
        th = float(wrap(th + w * dt))
    keys = ["t", "x", "y", "th", "v", "w", "a"]
    return {k: out[:, i].copy() for i, k in enumerate(keys)}
