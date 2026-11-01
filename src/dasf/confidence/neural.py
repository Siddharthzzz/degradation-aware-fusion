"""Heavy neural confidence baseline (Baseline 5).

Camera : ResNet-18 (He et al. 2016 architecture, trained from scratch) on the
         raw image pair [current, current - previous]  (2 x 96 x 160).
LiDAR  : 1-D CNN on the raw 360-beam scan [range, validity mask].
IMU / odometry : wide MLP (256-256) on the temporal feature set.

Requires PyTorch (+ torchvision for ResNet-18). The purpose of this baseline
is the accuracy-vs-compute trade-off, not to be a tuned state of the art.
"""
from __future__ import annotations

import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:                                  # pragma: no cover
    HAVE_TORCH = False


def _device():
    return "cuda" if HAVE_TORCH and torch.cuda.is_available() else "cpu"


if HAVE_TORCH:
    class LidarCNN(nn.Module):
        def __init__(self, in_ch=2):
            super().__init__()
            def blk(i, o, s):
                return nn.Sequential(nn.Conv1d(i, o, 5, s, 2, padding_mode="circular"),
                                     nn.BatchNorm1d(o), nn.ReLU())
            self.f = nn.Sequential(blk(in_ch, 32, 1), blk(32, 64, 2), blk(64, 128, 2), blk(128, 128, 2),
                                   nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                   nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x):
            return self.f(x).squeeze(-1)

    def camera_net(arch="resnet18"):
        if arch == "resnet18":
            import torchvision
            m = torchvision.models.resnet18(weights=None, num_classes=1)
            m.conv1 = nn.Conv2d(2, 64, 7, 2, 3, bias=False)
            return _Squeeze(m)
        return _Squeeze(nn.Sequential(
            nn.Conv2d(2, 16, 5, 2, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1)))

    class _Squeeze(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x).squeeze(-1)

    class WideMLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.mu = nn.Parameter(torch.zeros(d), requires_grad=False)
            self.sd = nn.Parameter(torch.ones(d), requires_grad=False)
            self.f = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                   nn.Linear(256, 1))

        def forward(self, x):
            return self.f((x - self.mu) / self.sd).squeeze(-1)


def lidar_input(ranges: np.ndarray) -> np.ndarray:
    """(N,360) -> (N,2,360) [range/3.5, valid];  (N,2,360) scan pairs -> (N,4,360)."""
    r = np.asarray(ranges, np.float32)
    if r.ndim == 1:
        r = r[None]
    if r.ndim == 3:
        return np.concatenate([lidar_input(r[:, 0]), lidar_input(r[:, 1])], 1)
    ok = np.isfinite(r)
    return np.stack([np.where(ok, r / 3.5, 0.0), ok.astype(np.float32)], 1).astype(np.float32)


def camera_input(pairs_u8: np.ndarray) -> np.ndarray:
    """(N,2,H,W) uint8 [prev, cur] -> (N,2,H,W) float [cur, cur-prev]."""
    p = pairs_u8.astype(np.float32) / 255.0
    return np.stack([p[:, 1], p[:, 1] - p[:, 0]], 1)


def train_binary(net, X, y, Xv, yv, epochs=15, bs=256, lr=1e-3, seed=0, log=print):
    torch.manual_seed(seed)
    dev = _device()
    net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    n = len(Xt)
    best, best_state = np.inf, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(Xt[b].to(dev)), yt[b].to(dev))
            loss.backward()
            opt.step()
        pv = predict_net(net, Xv)
        vl = float(-np.mean(yv * np.log(pv + 1e-7) + (1 - yv) * np.log(1 - pv + 1e-7)))
        log(f"  epoch {ep + 1}/{epochs}  val BCE {vl:.4f}")
        if vl < best:
            best, best_state = vl, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net.cpu().eval()


def predict_net(net, X, bs=512):
    dev = next(net.parameters()).device
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(net(torch.from_numpy(X[i:i + bs]).to(dev))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


class TorchEstimator:
    """Wraps a torch net with the same predict() interface as Learned."""
    kind = "cnn"

    def __init__(self, sensor, net, input_fn=None, feature_set="raw"):
        self.sensor, self.net, self.input_fn, self.feature_set = sensor, net.cpu().eval(), input_fn, feature_set
        self.calib = None

    def predict(self, X):
        X = self.input_fn(X) if self.input_fn is not None else np.atleast_2d(X).astype(np.float32)
        p = predict_net(self.net, X)
        return self.calib(p) if self.calib is not None else p

    def n_params(self):
        return int(sum(p.numel() for p in self.net.parameters()))


def torch_latency(est, x, n=500, warmup=50, device="cpu", threads=1):
    torch.set_num_threads(threads)
    net = est.net.to(device).eval()
    X = torch.from_numpy(est.input_fn(x) if est.input_fn else np.atleast_2d(x).astype(np.float32)).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            net(X)
        if device == "cuda":
            torch.cuda.synchronize()
        ts = np.empty(n)
        for i in range(n):
            t0 = time.perf_counter()
            net(X)
            if device == "cuda":
                torch.cuda.synchronize()
            ts[i] = time.perf_counter() - t0
    est.net.cpu()
    return dict(median_us=float(np.median(ts) * 1e6), p95_us=float(np.quantile(ts, 0.95) * 1e6))
