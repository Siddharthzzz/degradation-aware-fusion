"""Shared helpers: config loading, angles, RNG streams, causal lookups."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------- config
def _deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load configs/default.yaml, then overlay `path` (which may `inherit`)."""
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml", encoding="utf-8"))
    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        over = yaml.safe_load(open(p, encoding="utf-8")) or {}
        chain = []
        while over is not None:
            chain.append(over)
            parent = over.pop("inherit", None)
            over = yaml.safe_load(open(ROOT / parent, encoding="utf-8")) if parent else None
        for layer in reversed(chain):
            cfg = _deep_update(cfg, layer)
    calib = ROOT / cfg["paths"].get("calibration", "results/calibration.json")
    if calib.exists():
        cal = json.load(open(calib))
        cfg["fusion"]["r0"] = cal["r0"]
        cfg["fusion"]["r0_scale"] = cal.get("r0_scale", {})
        cfg["fusion"]["r0_source"] = str(calib)
    else:
        cfg["fusion"]["r0_source"] = "default.yaml (uncalibrated)"
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------- math
def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def rot(th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


# ----------------------------------------------------------------------------- RNG
def stable_int(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:12], 16)


def rng_stream(seed: int, name: str) -> np.random.Generator:
    """Independent, reproducible stream per (seed, purpose).

    Nominal sensor noise uses streams keyed only by the episode seed, so the
    same seed yields identical nominal noise under every degradation
    condition (common random numbers -> paired comparisons).
    """
    return np.random.default_rng([seed, stable_int(name)])


# ----------------------------------------------------------------------------- causal lookup
def zoh(t_src, v_src, t_q):
    """Zero-order hold: last source value with t_src <= t_q (causal).

    Returns (values, age). Values before the first sample are NaN.
    """
    t_src = np.asarray(t_src)
    v_src = np.asarray(v_src, dtype=float)
    idx = np.searchsorted(t_src, t_q, side="right") - 1
    ok = idx >= 0
    idc = np.clip(idx, 0, None)
    vals = v_src[idc].copy()
    age = np.where(ok, np.asarray(t_q) - t_src[idc], np.inf)
    vals[~ok] = np.nan
    return vals, age


class Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter()
        dt, self.t = now - self.t, now
        return dt
