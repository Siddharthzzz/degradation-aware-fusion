"""IMU degradation models, applied to full-rate streams (vectorised over time).

gyro: (N,), acc: (N,2) body-frame [ax, ay].
"""
from __future__ import annotations

import numpy as np


def apply_imu(t, gyro, acc, events, cfg, rng):
    c = cfg["sensors"]["imu"]
    gyro, acc = gyro.copy(), acc.copy()
    valid = np.ones(len(t), bool)
    for e in events:
        if e.sensor != "imu":
            continue
        m = e.active(t)
        if not m.any():
            continue
        n = m.sum()
        if e.kind == "noise":
            k = np.sqrt(e.value ** 2 - 1.0)
            gyro[m] += k * c["gyro_noise"] * rng.standard_normal(n)
            acc[m] += k * c["acc_noise"] * rng.standard_normal((n, 2))
        elif e.kind == "bias":
            gyro[m] += e.value
            acc[m, 0] += 10 * e.value
        elif e.kind == "biasramp":
            gyro[m] += e.value * (t[m] - e.t_on)
            acc[m, 0] += 10 * e.value * (t[m] - e.t_on)
        elif e.kind == "vibration":
            f = rng.uniform(15.0, 35.0)
            ph = 2 * np.pi * f * t[m]
            acc[m, 0] += e.value * np.sin(ph) + 0.5 * e.value * rng.standard_normal(n)
            acc[m, 1] += e.value * np.cos(1.3 * ph) + 0.5 * e.value * rng.standard_normal(n)
            gyro[m] += 0.05 * e.value * rng.standard_normal(n)
        elif e.kind == "stuck":
            i0 = max(np.argmax(m) - 1, 0)
            gyro[m], acc[m] = gyro[i0], acc[i0]
        elif e.kind == "dropout":
            valid[m] = False
    return gyro, acc, valid
