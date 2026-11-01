"""2-D indoor worlds: wall segments, procedural textures, ray casting, map lookup.

Three layouts provide *environmental* variation (independent of injected
degradation):
  W1_office       -- two rooms, several obstacles, rich texture
  W2_corridor     -- ring corridor; long straight stretches are geometrically
                     degenerate for a 3.5 m LiDAR (cf. Zhang et al. 2016)
  W3_featurepoor  -- open hall, low-contrast floor/walls, few LiDAR returns
                     in the centre
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils import stable_int


# ----------------------------------------------------------------------------- hashing
def hash01(i, j, k):
    """Deterministic integer hash -> [0,1). Vectorised, wraps on overflow."""
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    with np.errstate(over="ignore"):
        h = (i * np.int64(73856093)) ^ (j * np.int64(19349663)) ^ (k * np.int64(83492791))
        h = (h ^ (h >> 13)) * np.int64(1274126177)
        h = h ^ (h >> 16)
    return (h & 0xFFFF).astype(np.float64) / 65536.0


@dataclass
class World:
    name: str
    segs: np.ndarray                 # (M,4) x1 y1 x2 y2
    bounds: tuple                    # xmin xmax ymin ymax
    waypoints: np.ndarray            # (N,2) roadmap nodes
    loop: list                       # node indices of the closed loop route
    edges: list                      # undirected roadmap edges (i,j)
    wall_contrast: float = 1.0
    floor_contrast: float = 1.0
    tex_seed: int = 0
    tex: np.ndarray = field(default=None, repr=False)
    _grid: dict = field(default=None, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.tex_seed)
        m = len(self.segs)
        # per-segment texture parameters: base, a1 f1 p1, a2 f2 p2, block amp
        self.tex = np.column_stack([
            rng.uniform(0.35, 0.75, m),
            rng.uniform(0.05, 0.15, m), rng.uniform(1.5, 6.0, m), rng.uniform(0, 2 * np.pi, m),
            rng.uniform(0.03, 0.10, m), rng.uniform(4.0, 12.0, m), rng.uniform(0, 2 * np.pi, m),
            rng.uniform(0.15, 0.35, m),
        ])
        a, b = self.segs[:, :2], self.segs[:, 2:]
        e = b - a
        self.seg_len = np.linalg.norm(e, axis=1)
        self.seg_dir = e / self.seg_len[:, None]
        self.seg_nrm = np.column_stack([-self.seg_dir[:, 1], self.seg_dir[:, 0]])

    # ------------------------------------------------------------------ geometry
    def raycast(self, origin, angles, max_range=30.0):
        """Return (distance, segment index, arclength along segment) per ray."""
        o = np.asarray(origin, float)
        d = np.column_stack([np.cos(angles), np.sin(angles)])          # (N,2)
        a = self.segs[:, :2]
        e = self.segs[:, 2:] - a                                          # (M,2)
        ao = a - o                                                        # (M,2)
        den = d[:, 0:1] * e[None, :, 1] - d[:, 1:2] * e[None, :, 0]       # (N,M)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ao[None, :, 0] * e[None, :, 1] - ao[None, :, 1] * e[None, :, 0]) / den
            u = (ao[None, :, 0] * d[:, 1:2] - ao[None, :, 1] * d[:, 0:1]) / den
        ok = (np.abs(den) > 1e-12) & (t > 1e-6) & (u >= 0.0) & (u <= 1.0)
        t = np.where(ok, t, np.inf)
        idx = np.argmin(t, axis=1)
        dist = t[np.arange(len(angles)), idx]
        s = u[np.arange(len(angles)), idx] * self.seg_len[idx]
        dist = np.where(dist > max_range, np.inf, dist)
        return dist, idx, s

    def point_seg_dist(self, pts):
        """Distance from points (P,2) to every segment -> (P,M) and closest pts."""
        a = self.segs[:, :2]
        e = self.segs[:, 2:] - a
        ap = pts[:, None, :] - a[None]
        u = np.clip((ap * e[None]).sum(-1) / (self.seg_len ** 2)[None], 0.0, 1.0)
        c = a[None] + u[..., None] * e[None]
        return np.linalg.norm(pts[:, None, :] - c, axis=-1), c

    def build_grid(self, res=0.02):
        """Nearest-segment lookup grid (used by ICP correspondence search)."""
        if self._grid is not None and self._grid["res"] == res:
            return self._grid
        xmin, xmax, ymin, ymax = self.bounds
        xs = np.arange(xmin - 0.5, xmax + 0.5, res) + res / 2
        ys = np.arange(ymin - 0.5, ymax + 0.5, res) + res / 2
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        pts = np.column_stack([X.ravel(), Y.ravel()])
        best = np.full(len(pts), np.inf)
        arg = np.zeros(len(pts), dtype=np.int32)
        a = self.segs[:, :2]
        e = self.segs[:, 2:] - a
        for m in range(len(self.segs)):
            ap = pts - a[m]
            u = np.clip(ap @ e[m] / self.seg_len[m] ** 2, 0, 1)
            d = np.linalg.norm(ap - u[:, None] * e[m], axis=1)
            better = d < best
            best[better], arg[better] = d[better], m
        self._grid = dict(res=res, x0=xs[0] - res / 2, y0=ys[0] - res / 2,
                          nx=len(xs), ny=len(ys), seg=arg.reshape(len(ys), len(xs)),
                          dist=best.reshape(len(ys), len(xs)))
        return self._grid

    def clearance(self, p):
        d, _ = self.point_seg_dist(np.atleast_2d(p))
        return d.min(axis=1)

    # ------------------------------------------------------------------ appearance
    def wall_albedo(self, seg, s, z):
        T = self.tex[seg]
        val = (T[..., 0]
               + T[..., 1] * np.sin(2 * np.pi * T[..., 2] * s + T[..., 3])
               + T[..., 4] * np.sin(2 * np.pi * T[..., 5] * (s + 0.7 * z) + T[..., 6]))
        blk = hash01(np.floor(s / 0.12), np.floor(z / 0.12), seg + self.tex_seed) - 0.5
        blk2 = hash01(np.floor(s / 0.04), np.floor(z / 0.04), seg + 7919) - 0.5
        val = T[..., 0] + self.wall_contrast * (val - T[..., 0] + T[..., 7] * blk + 0.08 * blk2)
        skirt = z < 0.08                                   # dark skirting board
        val = np.where(skirt, 0.15, val)
        return np.clip(val, 0.02, 0.98)

    def floor_albedo(self, x, y):
        tile = 0.30
        gx, gy = np.mod(x, tile), np.mod(y, tile)
        grout = (np.minimum(gx, tile - gx) < 0.008) | (np.minimum(gy, tile - gy) < 0.008)
        t1 = hash01(np.floor(x / tile), np.floor(y / tile), self.tex_seed + 11) - 0.5
        t2 = hash01(np.floor(x / 0.05), np.floor(y / 0.05), self.tex_seed + 13) - 0.5
        t3 = hash01(np.floor(x / 0.02), np.floor(y / 0.02), self.tex_seed + 17) - 0.5
        val = 0.5 + self.floor_contrast * (0.12 * t1 + 0.20 * t2 + 0.10 * t3)
        val = np.where(grout, 0.5 - 0.25 * self.floor_contrast, val)
        return np.clip(val, 0.02, 0.98)

    def light(self, x, y):
        """Spatially varying ambient light (ceiling-lamp pattern), in [0.75, 1.0]."""
        return 0.875 + 0.125 * np.cos(2 * np.pi * x / 4.0) * np.cos(2 * np.pi * y / 4.0)


# ----------------------------------------------------------------------------- layouts
def _box(cx, cy, w, h):
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    return [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]


def _loop_edges(n):
    return [(i, (i + 1) % n) for i in range(n)]


def make_world(name: str) -> World:
    if name == "W1_office":
        segs = _box(5, 4, 10, 8) + [(0, 4, 3.5, 4), (6.5, 4, 10, 4)]
        segs += _box(2.0, 2.0, 0.8, 0.8) + _box(7.5, 2.0, 0.6, 1.0) + _box(2.5, 6.2, 1.0, 0.6)
        segs += _box(7.8, 6.0, 0.8, 0.8) + _box(5.0, 2.2, 0.4, 0.4)
        wp = [(1.0, 1.0), (4.0, 1.0), (6.3, 1.0), (9.0, 1.0), (9.0, 3.2), (5.0, 3.3),
              (5.0, 5.0), (9.0, 5.0), (9.0, 7.2), (1.0, 7.2), (1.0, 5.0), (4.2, 5.0),
              (4.2, 3.0), (1.0, 3.0)]
        loop = list(range(len(wp)))
        edges = _loop_edges(len(wp)) + [(5, 12), (6, 11)]
        return World(name, np.array(segs, float), (0, 10, 0, 8), np.array(wp), loop, edges,
                     wall_contrast=1.0, floor_contrast=1.0, tex_seed=stable_int(name) % 10000)
    if name == "W2_corridor":
        segs = _box(6, 3, 12, 6) + _box(6, 3, 8.8, 2.8)
        segs += _box(3.0, 0.1, 0.4, 0.2) + _box(9.0, 0.1, 0.4, 0.2) + _box(6.0, 5.9, 0.4, 0.2)
        wp = [(0.8, 0.8), (6.0, 0.8), (11.2, 0.8), (11.2, 3.0), (11.2, 5.2), (6.0, 5.2),
              (0.8, 5.2), (0.8, 3.0)]
        loop = list(range(len(wp)))
        return World(name, np.array(segs, float), (0, 12, 0, 6), np.array(wp), loop,
                     _loop_edges(len(wp)), wall_contrast=0.8, floor_contrast=0.8,
                     tex_seed=stable_int(name) % 10000)
    if name == "W3_featurepoor":
        segs = _box(5, 4, 10, 8) + _box(3.2, 5.2, 0.6, 0.6) + _box(6.8, 2.8, 0.6, 0.6)
        segs += _box(5.0, 6.0, 0.8, 0.4)
        wp = [(1.5, 1.5), (5.0, 1.5), (8.5, 1.5), (8.5, 4.0), (8.5, 6.5), (5.0, 7.3),
              (1.5, 6.5), (1.5, 4.0), (5.0, 4.0)]
        loop = list(range(8))
        edges = _loop_edges(8) + [(1, 8), (8, 7), (8, 3)]
        return World(name, np.array(segs, float), (0, 10, 0, 8), np.array(wp), loop, edges,
                     wall_contrast=0.35, floor_contrast=0.35, tex_seed=stable_int(name) % 10000)
    raise ValueError(f"unknown world {name}")


WORLDS = ["W1_office", "W2_corridor", "W3_featurepoor"]

_MAPS = {}


def localization_map(name: str, map_noise: float) -> World:
    """The robot's prior map: the true walls with a FIXED per-world vertex
    perturbation (std `map_noise`), modelling mapping/as-built errors.
    Cached so the ICP lookup grid is built once per process."""
    key = (name, map_noise)
    if key not in _MAPS:
        w = make_world(name)
        rng = np.random.default_rng(stable_int("map", name))
        verts = {}
        segs = w.segs.copy()
        for m in range(len(segs)):
            for c in (0, 2):
                v = (round(segs[m, c], 6), round(segs[m, c + 1], 6))
                if v not in verts:
                    verts[v] = rng.normal(0, map_noise, 2)
                segs[m, c:c + 2] += verts[v]
        _MAPS[key] = World(name + "_map", segs, w.bounds, w.waypoints, w.loop, w.edges,
                           w.wall_contrast, w.floor_contrast, w.tex_seed)
    return _MAPS[key]
