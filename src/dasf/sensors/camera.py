"""Level pinhole RGB-camera model (rendered as grayscale luminance).

Rendering: per-column ray casting against the wall segments, per-pixel
wall/floor/ceiling selection, procedural textures, ambient lighting.
Image formation: auto-exposure gain (capped), shot + read noise, 8-bit
quantisation.  Optical degradations (defocus, motion blur, lens occlusion)
are applied to the irradiance BEFORE noise; low light scales irradiance, so
darkness produces the realistic low-SNR / high-gain image.
"""
from __future__ import annotations

import numpy as np

from ..degradation.camera import apply_optical, apply_post


class Camera:
    def __init__(self, cfg: dict):
        c = cfg["sensors"]["camera"]
        self.c = c
        self.W, self.H = c["width"], c["height"]
        self.f = (self.W / 2) / np.tan(np.deg2rad(c["hfov_deg"]) / 2)
        self.cx, self.cy = (self.W - 1) / 2, (self.H - 1) / 2
        self.h = c["mount_height"]
        # Pixels integrate irradiance over their area: render on an ss x ss
        # sub-pixel grid and box-average (anti-aliasing). Without this, point-
        # sampled texture aliases and biases the VO toward zero motion.
        self.ss = int(c.get("supersample", 3))
        off = (np.arange(self.ss) + 0.5) / self.ss - 0.5
        u = (np.arange(self.W)[:, None] + off[None, :]).ravel()
        v = (np.arange(self.H)[:, None] + off[None, :]).ravel()
        self.col_az_ss = np.arctan2(self.cx - u, self.f)
        self.n_h = np.hypot(self.f, self.cx - u)
        self.rows = v[:, None]
        self.cols = u[None, :]
        self.col_az = np.arctan2(self.cx - np.arange(self.W), self.f)   # pixel-centre azimuths (VO)

    def irradiance(self, world, pose) -> np.ndarray:
        x, y, th = pose
        D, seg, s = world.raycast((x, y), th + self.col_az_ss, max_range=50.0)
        D = np.where(np.isfinite(D), D, 50.0)
        Hs, Ws = self.rows.shape[0], self.cols.shape[1]
        s_w = D[None, :] / self.n_h[None, :]
        z_w = self.h + (self.cy - self.rows) * s_w
        below = self.rows > self.cy
        with np.errstate(divide="ignore"):
            s_f = np.where(below, self.h / np.maximum(self.rows - self.cy, 1e-9), np.inf)
        wall = (s_w < s_f) & (z_w <= self.c["wall_height"]) & (z_w >= 0)
        floor = below & ~wall
        segb = np.broadcast_to(seg[None, :], (Hs, Ws))
        sb = np.broadcast_to(s[None, :], (Hs, Ws))
        alb = np.full((Hs, Ws), 0.25)                              # ceiling / far
        # textures are per-pixel, so evaluating them only where visible is exact
        z_wb = np.broadcast_to(z_w, (Hs, Ws))
        alb[wall] = world.wall_albedo(segb[wall], sb[wall], z_wb[wall])
        # floor points in the world frame
        fx = self.f * np.where(floor, s_f, 0.0)
        fy = (self.cx - self.cols) * np.where(floor, s_f, 0.0)
        wx = x + np.cos(th) * fx - np.sin(th) * fy
        wy = y + np.sin(th) * fx + np.cos(th) * fy
        alb[floor] = world.floor_albedo(wx[floor], wy[floor])
        alb = alb.reshape(self.H, self.ss, self.W, self.ss).mean(axis=(1, 3))
        return alb * world.light(x, y)

    def expose(self, irr, rng, deg: dict, w_true: float, rng_deg=None) -> np.ndarray:
        """Irradiance -> 8-bit image in [0,1] under active degradations `deg`."""
        illum = deg.get("lowlight", 1.0)
        sig = apply_optical(irr * illum, deg, w_true, self)
        gain = float(np.clip(self.c["ae_target"] / max(sig.mean(), 1e-6), 1.0, self.c["max_gain"]))
        std = np.sqrt(self.c["shot_noise"] * sig + self.c["read_noise"] ** 2)
        img = gain * (sig + std * rng.standard_normal(sig.shape))
        img = apply_post(img, deg, rng_deg if rng_deg is not None else rng)
        return np.round(np.clip(img, 0, 1) * 255) / 255.0
