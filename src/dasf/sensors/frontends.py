"""Measurement front-ends. These are REAL algorithms operating on the rendered
images / ray-cast scans, so measurement errors under degradation emerge from
the data rather than from a hand-written error model.

* CameraVO   -- planar visual odometry for a level monocular camera:
                (1) yaw increment from phase correlation (Kuglin & Hines 1975)
                    of the upper image band resampled to uniform azimuth;
                (2) metric forward/lateral motion from direct photometric
                    Gauss-Newton alignment of a ground-plane patch
                    (known camera height -> metric scale).
* icp_scan_to_map -- point-to-line ICP (Besl & McKay 1992 family) of a 2-D
                scan against the known wall map, initialised from the
                filter's predicted pose.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..utils import wrap


# ============================================================================ camera VO
def phase_correlation(a, b):
    """Return (shift_rows, shift_cols, peak_ratio) with b(x) ~= a(x - shift)."""
    H, W = a.shape
    win = np.outer(np.hanning(H), np.hanning(W))
    A = np.fft.fft2((a - a.mean()) * win)
    B = np.fft.fft2((b - b.mean()) * win)
    R = B * np.conj(A)
    R /= np.abs(R) + 1e-9
    r = np.real(np.fft.ifft2(R))
    i, j = np.unravel_index(np.argmax(r), r.shape)
    peak = r[i, j]

    def sub(v_m, v_0, v_p):
        den = v_m - 2 * v_0 + v_p
        return 0.0 if abs(den) < 1e-12 else 0.5 * (v_m - v_p) / den

    di = sub(r[(i - 1) % H, j], peak, r[(i + 1) % H, j])
    dj = sub(r[i, (j - 1) % W], peak, r[i, (j + 1) % W])
    si = i if i <= H // 2 else i - H
    sj = j if j <= W // 2 else j - W
    rr = r.copy()
    for a_ in (-1, 0, 1):
        for b_ in (-1, 0, 1):
            rr[(i + a_) % H, (j + b_) % W] = -np.inf
    second = rr.max()
    ratio = peak / max(second, 1e-6) if second > 0 else 10.0
    return si + di, sj + dj, float(min(ratio, 10.0))


class CameraVO:
    FEATURES = ["vo_resid", "vo_valid_frac", "vo_log_mineig", "vo_peak_ratio",
                "vo_lat_vel", "vo_yaw_disagree", "vo_converged", "vo_iters"]

    def __init__(self, cam, cfg):
        self.cam = cam
        v = cfg["sensors"]["camera"]["vo"]
        self.iters = v["gn_iters"]
        self.sigmas = v["pyramid_sigmas"]
        xs = np.linspace(*v["ipm_x"], v["ipm_n"])
        ys = np.linspace(*v["ipm_y"], v["ipm_n"])
        X, Y = np.meshgrid(xs, ys)
        q = np.column_stack([X.ravel(), Y.ravel()])
        u, vv = self._proj(q)
        ok = (u > 2) & (u < cam.W - 3) & (vv > cam.cy + 2) & (vv < cam.H - 3)
        self.q = q[ok]
        # uniform-azimuth resampling of the upper band (rotation -> pure shift)
        psi0, psi1 = cam.col_az[0], cam.col_az[-1]
        self.dpsi = (psi0 - psi1) / (cam.W - 1)
        psi = psi0 - self.dpsi * np.arange(cam.W)
        self.u_of_psi = cam.cx - cam.f * np.tan(psi)
        self.band = int(cam.cy) - 2

    def _proj(self, p):
        c = self.cam
        u = c.cx - c.f * p[:, 1] / p[:, 0]
        v = c.cy + c.f * c.h / p[:, 0]
        return u, v

    def _cyl(self, img):
        rows = np.arange(self.band)[:, None] * np.ones((1, self.cam.W))
        cols = np.ones((self.band, 1)) * self.u_of_psi[None, :]
        return ndimage.map_coordinates(img, [rows, cols], order=1, mode="nearest")

    def estimate(self, prev, cur, dt, init_v=0.0):
        """Relative motion prev->cur. Returns (v, w, lateral_v, features dict)."""
        _, sj, ratio = phase_correlation(self._cyl(prev), self._cyl(cur))
        dth_pc = sj * self.dpsi
        xi = np.array([init_v * dt, 0.0, dth_pc])
        resid, nvalid, mineig, conv, it_total = np.nan, 0, 0.0, 0.0, 0
        for sig in self.sigmas:
            Ip = ndimage.gaussian_filter(prev, sig)
            Ic = ndimage.gaussian_filter(cur, sig)
            gv, gu = np.gradient(Ip)
            uc, vc = self._proj(self.q)
            ic = ndimage.map_coordinates(Ic, [vc, uc], order=1)
            ic = (ic - ic.mean()) / (ic.std() + 1e-6)
            for _ in range(self.iters):
                it_total += 1
                c, s = np.cos(xi[2]), np.sin(xi[2])
                px = c * self.q[:, 0] - s * self.q[:, 1] + xi[0]
                py = s * self.q[:, 0] + c * self.q[:, 1] + xi[1]
                p = np.column_stack([px, py])
                u, v = self._proj(p)
                ok = (u > 1) & (u < self.cam.W - 2) & (v > self.cam.cy + 1) & (v < self.cam.H - 2) & (px > 0.1)
                nvalid = int(ok.sum())
                if nvalid < 30:
                    break
                ip = ndimage.map_coordinates(Ip, [v[ok], u[ok]], order=1)
                sd = ip.std() + 1e-6
                r = (ip - ip.mean()) / sd - ic[ok]
                Iu = ndimage.map_coordinates(gu, [v[ok], u[ok]], order=1) / sd
                Iv = ndimage.map_coordinates(gv, [v[ok], u[ok]], order=1) / sd
                f, h = self.cam.f, self.cam.h
                pxo, pyo = px[ok], py[ok]
                du_dx, du_dy = f * pyo / pxo ** 2, -f / pxo
                dv_dx = -f * h / pxo ** 2
                gx = Iu * du_dx + Iv * dv_dx
                gy = Iu * du_dy
                qx, qy = self.q[ok, 0], self.q[ok, 1]
                dpx_dth = -s * qx - c * qy
                dpy_dth = c * qx - s * qy
                J = np.column_stack([gx, gy, gx * dpx_dth + gy * dpy_dth])
                wts = 1.0 / np.maximum(1.0, np.abs(r))          # Huber, k = 1
                Hm = (J * wts[:, None]).T @ J
                g = (J * wts[:, None]).T @ r
                try:
                    step = -np.linalg.solve(Hm + 1e-6 * np.eye(3), g)
                except np.linalg.LinAlgError:
                    break
                xi += step
                resid = float(np.sqrt(np.mean(r ** 2)))
                ev = np.linalg.eigvalsh(Hm / max(nvalid, 1))
                mineig = float(ev[0])
                conv = float(np.linalg.norm(step[:2]) < 1e-4 and abs(step[2]) < 1e-4)
                if conv:
                    break
        feats = dict(
            vo_resid=resid if np.isfinite(resid) else 3.0,
            vo_valid_frac=nvalid / max(len(self.q), 1),
            vo_log_mineig=float(np.log10(max(mineig, 1e-12))),
            vo_peak_ratio=ratio,
            vo_lat_vel=abs(xi[1]) / dt,
            vo_yaw_disagree=abs(wrap(xi[2] - dth_pc)) / dt,
            vo_converged=conv,
            vo_iters=float(it_total),
        )
        return xi[0] / dt, xi[2] / dt, xi[1] / dt, feats


# ============================================================================ LiDAR ICP
ICP_FEATURES = ["icp_fitness", "icp_inlier_ratio", "icp_log_mineig", "icp_log_cond",
                "icp_corr_trans", "icp_corr_rot", "icp_iters", "icp_converged"]


def _gn_icp(p, x0, corr, ic):
    """Robust point-to-line Gauss-Newton ICP.

    p: (N,2) points in the moving frame; corr(w) -> (closest pts, normals,
    usable mask) for points w in the reference frame. Returns (x or None, feats).
    """
    x = np.array(x0, float)
    sched = ic["reject_dist"]
    per_stage = max(1, ic["max_iters"] // len(sched))
    stages = [thr for thr in sched for _ in range(per_stage)]
    it, conv, k = 0, 0.0, 0
    H, inl, d = np.zeros((3, 3)), np.zeros(len(p), bool), np.zeros(len(p))
    while k < len(stages):
        thr = stages[k]
        it += 1
        c, s = np.cos(x[2]), np.sin(x[2])
        w = np.column_stack([c * p[:, 0] - s * p[:, 1] + x[0], s * p[:, 0] + c * p[:, 1] + x[1]])
        cp, n, usable = corr(w)
        dist = np.linalg.norm(w - cp, axis=1)
        d = ((w - cp) * n).sum(1)
        inl = usable & (dist < thr)
        if inl.sum() < ic["min_inliers"]:
            return None, None
        dth = np.column_stack([-s * p[:, 0] - c * p[:, 1], c * p[:, 0] - s * p[:, 1]])
        J = np.column_stack([n[:, 0], n[:, 1], (n * dth).sum(1)])[inl]
        r = d[inl]
        wt = ic["huber"] / np.maximum(np.abs(r), ic["huber"])
        H = (J * wt[:, None]).T @ J
        gv = (J * wt[:, None]).T @ r
        try:
            step = -np.linalg.solve(H + 1e-9 * np.eye(3), gv)
        except np.linalg.LinAlgError:
            return None, None
        x += step
        x[2] = float(wrap(x[2]))
        small = np.linalg.norm(step[:2]) < 1e-4 and abs(step[2]) < 1e-4
        if small and thr == sched[-1]:
            conv = 1.0
            break
        k = stages.index(thr) + per_stage if small else k + 1
    return x, _icp_feats(H, inl, d, x, x0, it, conv)


def _failed_feats():
    f = {k: 0.0 for k in ICP_FEATURES}
    f.update(icp_fitness=1.0, icp_log_mineig=-12.0, icp_log_cond=12.0)
    return f


def icp_scan_to_map(ranges, angles, pose0, world, cfg):
    """Scan-to-map ICP. Returns (pose or None, features dict)."""
    ic = cfg["sensors"]["lidar"]["icp"]
    g = world.build_grid(ic["grid_res"])
    ok = np.isfinite(ranges)
    if ok.sum() < ic["min_inliers"]:
        return None, _failed_feats()
    p = np.column_stack([ranges[ok] * np.cos(angles[ok]), ranges[ok] * np.sin(angles[ok])])

    def corr(w):
        gi = np.clip(((w[:, 1] - g["y0"]) / g["res"]).astype(int), 0, g["ny"] - 1)
        gj = np.clip(((w[:, 0] - g["x0"]) / g["res"]).astype(int), 0, g["nx"] - 1)
        sg = g["seg"][gi, gj]
        a = world.segs[sg, :2]
        e = world.segs[sg, 2:] - a
        u = np.clip(((w - a) * e).sum(1) / world.seg_len[sg] ** 2, 0, 1)
        return a + u[:, None] * e, world.seg_nrm[sg], np.ones(len(w), bool)

    x, f = _gn_icp(p, pose0, corr, ic)
    return (None, _failed_feats()) if x is None else (x, f)


def scan_normals(pts):
    """Normals from neighbouring beams (beam-ordered points); invalid -> nan."""
    n = len(pts)
    nrm = np.full((n, 2), np.nan)
    if n < 3:
        return nrm
    t = pts[2:] - pts[:-2]
    L = np.linalg.norm(t, axis=1)
    good = (L > 1e-4) & (L < 0.25)
    tt = t / np.maximum(L, 1e-9)[:, None]
    nrm[1:-1] = np.where(good[:, None], np.column_stack([-tt[:, 1], tt[:, 0]]), np.nan)
    return nrm


def icp_scan_to_scan(prev_r, cur_r, angles, init, cfg):
    """Relative pose of the current scan frame in the previous one."""
    from scipy.spatial import cKDTree
    ic = cfg["sensors"]["lidar"]["icp"]
    okp, okc = np.isfinite(prev_r), np.isfinite(cur_r)
    if okp.sum() < ic["min_inliers"] or okc.sum() < ic["min_inliers"]:
        return None, _failed_feats()
    P = np.column_stack([prev_r[okp] * np.cos(angles[okp]), prev_r[okp] * np.sin(angles[okp])])
    Q = np.column_stack([cur_r[okc] * np.cos(angles[okc]), cur_r[okc] * np.sin(angles[okc])])
    N = scan_normals(P)
    keep = np.isfinite(N[:, 0])
    if keep.sum() < ic["min_inliers"]:
        return None, _failed_feats()
    P, N = P[keep], N[keep]
    tree = cKDTree(P)

    def corr(w):
        _, j = tree.query(w)
        return P[j], N[j], np.ones(len(w), bool)

    x, f = _gn_icp(Q, init, corr, ic)
    return (None, _failed_feats()) if x is None else (x, f)


def _icp_feats(H, inl, d, x, pose0, it, conv):
    ev = np.linalg.eigvalsh(H / max(inl.sum(), 1))
    return dict(
        icp_fitness=float(np.sqrt(np.mean(d[inl] ** 2))),
        icp_inlier_ratio=float(inl.mean()),
        icp_log_mineig=float(np.log10(max(ev[0], 1e-12))),
        icp_log_cond=float(np.log10(max(ev[-1], 1e-12) / max(ev[0], 1e-12))),
        icp_corr_trans=float(np.hypot(x[0] - pose0[0], x[1] - pose0[1])),
        icp_corr_rot=float(abs(wrap(x[2] - pose0[2]))),
        icp_iters=float(it),
        icp_converged=conv,
    )
