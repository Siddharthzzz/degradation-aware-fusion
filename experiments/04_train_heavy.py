"""Step 4 (Baseline 5): train heavy neural confidence models and run the camera
re-render pass that produces CNN camera confidences for val/test episodes.

    python experiments/04_train_heavy.py --config configs/matrix.yaml [--arch resnet18|small]
"""
from pathlib import Path

import numpy as np
from _common import parse

from dasf.confidence import estimators as E
from dasf.confidence import neural as N
from dasf.confidence.bank import est_path
from dasf.episode import generate_episode, load_episode
from dasf.experiment import load_dataset, pmap, split_paths
from dasf.utils import resolve, stable_int


def heavy_key(meta):
    ev = [e.to_dict() for e in meta["events"]]
    return f"{meta['world']}__{meta['traj']}__s{meta['seed']}__{stable_int(str(ev))}"


def train(cfg, arch, epochs, log=print):
    mdir = resolve(cfg, "models")
    # camera: raw image pairs
    tr, va = load_dataset(cfg, "train", "cam"), load_dataset(cfg, "val", "cam")
    Xtr, ytr = N.camera_input(tr["imgs"]), tr["y"][tr["img_rows"]]
    Xva, yva = N.camera_input(va["imgs"]), va["y"][va["img_rows"]]
    log(f"camera CNN ({arch}): {len(Xtr)} train / {len(Xva)} val images")
    net = N.train_binary(N.camera_net(arch), Xtr, ytr, Xva, yva, epochs=epochs, log=log)
    est = N.TorchEstimator("cam", net, None)
    est.calib = E.Platt().fit(est.predict(Xva), yva)
    E.save(est, est_path(mdir, "cam", "cnn"))
    # LiDAR (map track: single scan; odometry track: scan pair)
    for s, in_ch in (("lidar", 2), ("lidar_rel", 4)):
        tr, va = load_dataset(cfg, "train", s), load_dataset(cfg, "val", s)
        Xtr, Xva = N.lidar_input(tr["raw"].astype(np.float32)), N.lidar_input(va["raw"].astype(np.float32))
        log(f"{s} 1-D CNN: {len(Xtr)} train scans")
        net = N.train_binary(N.LidarCNN(in_ch), Xtr, tr["y"], Xva, va["y"], epochs=epochs, log=log)
        est = N.TorchEstimator(s, net, N.lidar_input)
        est.calib = E.Platt().fit(est.predict(va["raw"].astype(np.float32)), va["y"])
        E.save(est, est_path(mdir, s, "cnn"))
    # IMU / odometry: wide MLP on the temporal feature set
    for s in ("imu", "odom"):
        tr, va = load_dataset(cfg, "train", s), load_dataset(cfg, "val", s)
        net = N.WideMLP(tr["Xt"].shape[1])
        net.mu.data = N.torch.tensor(tr["Xt"].mean(0), dtype=N.torch.float32)
        net.sd.data = N.torch.tensor(tr["Xt"].std(0) + 1e-6, dtype=N.torch.float32)
        log(f"{s} wide MLP: {len(tr['Xt'])} rows")
        net = N.train_binary(net, tr["Xt"].astype(np.float32), tr["y"], va["Xt"].astype(np.float32),
                             va["y"], epochs=max(3, epochs // 3), bs=2048, log=log)
        est = N.TorchEstimator(s, net, None, feature_set="temporal")
        est.calib = E.Platt().fit(est.predict(va["Xt"]), va["y"])
        E.save(est, est_path(mdir, s, "cnn"))


def _cam_pass(args):
    cfg, path = args
    import torch
    torch.set_num_threads(1)
    ep = load_episode(Path(path))
    meta = ep["meta"]
    out = Path(resolve(cfg, "heavy_conf")) / f"{heavy_key(meta)}.npz"
    if out.exists():
        return
    est = E.load(est_path(resolve(cfg, "models"), "cam", "cnn"))
    ts, cs, prev = [], [], [None]

    def hook(t, img):
        p = prev[0] if prev[0] is not None else img
        pair = (np.stack([p, img])[None] * 255).astype(np.uint8)
        cs.append(float(est.predict(N.camera_input(pair))[0]))
        ts.append(t)
        prev[0] = img

    if "cam_frames" in ep:   # frames stored at generation time: replay instead of re-rendering
        for tj, f in zip(ep["cam_frames_t"], ep["cam_frames"]):
            hook(float(tj), f.astype(np.float64) / 255.0)
    else:
        generate_episode(cfg, meta["world"], meta["traj"], meta["seed"], meta["events"], cam_hook=hook)
    np.savez(out, t=np.array(ts), c=np.array(cs))


if __name__ == "__main__":
    def extra(ap):
        ap.add_argument("--arch", default="resnet18", choices=["resnet18", "small"])
        ap.add_argument("--epochs", type=int, default=15)
        ap.add_argument("--skip-train", action="store_true")
    args, cfg = parse(__doc__, extra)
    if not N.HAVE_TORCH:
        raise SystemExit("PyTorch not installed: Baseline 5 cannot be run (report as not run).")
    if not args.skip_train:
        train(cfg, args.arch, args.epochs)
    for split in ("val", "test"):
        paths = split_paths(cfg, split)
        print(f"camera re-render pass [{split}]: {len(paths)} episodes")
        pmap(_cam_pass, [(cfg, str(p)) for p in paths], args.workers)
