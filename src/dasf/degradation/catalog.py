"""Degradation catalogue: types, severity levels and scheduling.

A degradation event is (sensor, type, severity value, t_on, t_off).  Severity
LEVELS (1 = mild, 2 = medium, 3 = severe, 4 = extreme where defined) map to the
physical parameter values below.  All values are DESIGN CHOICES.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# sensor -> type -> list of parameter values for severity levels 1..K
CATALOG = {
    "cam": {
        "lowlight":    [0.30, 0.10, 0.03, 0.01],   # illumination scale (0.01 = severe darkness)
        "defocus":     [1.5, 3.0, 6.0],             # Gaussian blur sigma [px]
        "motionblur":  [5, 11, 21],                 # horizontal kernel length [px]
        "noise":       [0.04, 0.10, 0.20],          # extra additive noise std [0-1 intensity]
        "occlusion":   [0.25, 0.50, 0.75],          # fraction of image width covered (lower band)
        "frozen":      [1.0],                       # stuck frame (driver hang)
        "dropout":     [1.0],                       # no frames delivered
    },
    "lidar": {
        "dropout":     [0.10, 0.30, 0.50, 0.80],    # extra random beam-dropout probability
        "rangenoise":  [0.03, 0.10, 0.30],          # extra range noise std [m]
        "fov":         [90.0, 180.0, 270.0],        # obstructed sector [deg]
        "sparse":      [2, 4, 8],                   # keep every k-th beam
        "blackout":    [1.0],                       # no valid returns
    },
    "imu": {
        "noise":       [3.0, 10.0, 30.0],           # noise std multiplier
        "bias":        [0.02, 0.05, 0.10],          # gyro bias step [rad/s]; accel step = 10x [m/s^2]
        "biasramp":    [0.001, 0.003, 0.010],       # gyro bias slope [rad/s^2]
        "vibration":   [0.5, 1.5, 4.0],             # accel vibration amplitude [m/s^2]
        "stuck":       [1.0],                       # output frozen at last value
        "dropout":     [1.0],                       # no samples delivered
    },
    "odom": {
        "slip":        [0.10, 0.30, 0.60],          # mean slip ratio (asymmetric between wheels)
        "encnoise":    [3.0, 10.0, 30.0],           # encoder noise multiplier
        "bumps":       [0.5, 1.5, 3.0],             # uneven floor: spike rate [1/s] x amplitude
        "stuck":       [1.0],                       # frozen encoder readings
        "dropout":     [1.0],                       # no samples delivered
    },
}


@dataclass
class DegEvent:
    sensor: str
    kind: str
    value: float
    t_on: float
    t_off: float
    level: int = 0

    def active(self, t):
        return (np.asarray(t) >= self.t_on) & (np.asarray(t) < self.t_off)

    def to_dict(self):
        return asdict(self)


def event(sensor, kind, level, t_on, t_off) -> DegEvent:
    vals = CATALOG[sensor][kind]
    return DegEvent(sensor, kind, float(vals[level - 1]), t_on, t_off, level)


def parse_condition(cond: str, t_on: float, t_off: float) -> list[DegEvent]:
    """Condition strings: 'nominal', 'cam.lowlight.2', 'cam.lowlight.2+odom.slip.2'."""
    if cond == "nominal":
        return []
    evs = []
    for part in cond.split("+"):
        s, k, lvl = part.split(".")
        evs.append(event(s, k, int(lvl), t_on, t_off))
    return evs


def random_training_events(rng: np.random.Generator, duration: float,
                           exclude: tuple = ()) -> list[DegEvent]:
    """Training episodes: 0-3 events, random type, CONTINUOUS random severity
    (interpolated between catalogue levels), random timing. Disjoint seeds from test.
    """
    n = rng.choice([0, 1, 1, 2, 2, 3])
    evs = []
    for _ in range(n):
        sensor = rng.choice(list(CATALOG))
        kinds = [k for k in CATALOG[sensor] if f"{sensor}.{k}" not in exclude]
        if not kinds:
            continue
        kind = rng.choice(kinds)
        vals = np.asarray(CATALOG[sensor][kind], float)
        if len(vals) == 1:
            val = vals[0]
        else:
            pos = rng.uniform(0, len(vals) - 1)
            i = int(np.floor(pos))
            j = min(i + 1, len(vals) - 1)
            val = vals[i] + (pos - i) * (vals[j] - vals[i])
            if kind in ("sparse", "motionblur"):
                val = float(np.round(val))
        dur = rng.uniform(5.0, 30.0)
        t_on = rng.uniform(5.0, max(6.0, duration - dur - 2.0))
        evs.append(DegEvent(str(sensor), str(kind), float(val), float(t_on), float(t_on + dur), -1))
    return evs


def active_params(events, sensor, t):
    """Dict kind -> value for events of `sensor` active at scalar time t."""
    return {e.kind: e.value for e in events if e.sensor == sensor and e.t_on <= t < e.t_off}
