"""Camera degradation models.

Optical (before sensor noise): defocus (Gaussian PSF), motion blur (horizontal
box PSF, added to the natural rotational blur), lens occlusion (opaque dark
blob over the lower image band -- where the ground-plane VO looks).
Post (after sensor noise): extra additive noise (e.g., EMI / compression).
Low light is handled as an irradiance scale in Camera.expose.
Frozen frames and dropout are handled in the episode generator.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def apply_optical(sig: np.ndarray, deg: dict, w_true: float, cam) -> np.ndarray:
    out = sig
    # natural rotational motion blur during a 10 ms exposure
    nat = abs(w_true) * 0.010 * cam.f
    extra = deg.get("motionblur", 0.0)
    L = int(round(nat + extra))
    if L >= 2:
        out = ndimage.uniform_filter1d(out, size=L, axis=1, mode="nearest")
    if "defocus" in deg:
        out = ndimage.gaussian_filter(out, deg["defocus"], mode="nearest")
    if "occlusion" in deg:
        frac = deg["occlusion"]
        H, W = out.shape
        mask = np.zeros_like(out)
        mask[int(0.45 * H):, : int(round(frac * W))] = 1.0
        mask = ndimage.gaussian_filter(mask, 2.0)
        out = out * (1 - mask) + 0.01 * mask
    return out


def apply_post(img: np.ndarray, deg: dict, rng) -> np.ndarray:
    if "noise" in deg:
        img = img + deg["noise"] * rng.standard_normal(img.shape)
    return img
