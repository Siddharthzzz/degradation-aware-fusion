"""Figures 2-18. Every data figure reads MEASURED outputs (episode files,
run traces, metric CSVs, benchmark CSVs); nothing is typed in by hand.
Fig. 1 (architecture) is a TikZ drawing inside the paper.
"""
from __future__ import annotations

from ast import literal_eval

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge

from ..degradation.catalog import active_params, parse_condition
from ..episode import load_episode
from ..evaluation.metrics import reliability_curve
from ..evaluation.stats import bootstrap_ci
from ..sensors.camera import Camera
from ..simulation.world import make_world
from ..utils import rng_stream
from .style import METHOD_STYLE, NEUTRAL, SENSOR_STYLE, save, setup

setup()


def _shade(ax, cfg, t_on=None, t_off=None):
    if t_on is None:
        t_on, t_off = cfg["sim"]["deg_window"]
    ax.axvspan(t_on, t_off, color="#e4e3df", alpha=0.6, lw=0, zorder=0)


def _mstyle(m):
    return METHOD_STYLE.get(m, (NEUTRAL, "o", m))


# ---------------------------------------------------------------- Fig. 2
def fig2_robot(cfg, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 3.2))
    rc, S = cfg["robot"], cfg["sensors"]
    L = S["lidar"]["range_max"]
    a.add_patch(Wedge((0, 0), L, 0, 360, color=SENSOR_STYLE["lidar"][0], alpha=0.12, label="LiDAR 360°, 0.12-3.5 m"))
    h = S["camera"]["hfov_deg"] / 2
    a.add_patch(Wedge((0, 0), 2.0, -h, h, color=SENSOR_STYLE["cam"][0], alpha=0.35, label=f"Camera HFOV {2*h:.1f}°"))
    vo = S["camera"]["vo"]
    a.add_patch(Rectangle((vo["ipm_x"][0], vo["ipm_y"][0]), vo["ipm_x"][1] - vo["ipm_x"][0],
                          vo["ipm_y"][1] - vo["ipm_y"][0], fill=False, ls="--", ec="#0b0b0b", label="Ground-plane VO patch"))
    a.add_patch(Rectangle((-0.14, -0.15), 0.28, 0.30, color="#52514e"))
    for s in (-1, 1):
        a.add_patch(Rectangle((-0.033, s * rc["wheel_base"] / 2 - 0.01), 0.066, 0.02, color="#0b0b0b"))
    a.set_xlim(-1.2, 3.8)
    a.set_ylim(-2.2, 2.2)
    a.set_aspect("equal")
    a.set_xlabel("x (m, forward)")
    a.set_ylabel("y (m, left)")
    a.set_title("(a) Sensor footprint (top view)")
    a.legend(loc="lower right")
    rows = [("IMU", f"{S['imu']['rate']:.0f} Hz", f"gyro σ {S['imu']['gyro_noise']} rad/s, acc σ {S['imu']['acc_noise']} m/s²"),
            ("Wheel enc.", f"{S['odom']['rate']:.0f} Hz", f"{S['odom']['ticks_per_rev']} ticks/rev"),
            ("Camera", f"{S['camera']['rate']:.0f} Hz", f"{S['camera']['width']}×{S['camera']['height']} px, h={S['camera']['mount_height']} m"),
            ("LiDAR", f"{S['lidar']['rate']:.0f} Hz", f"{S['lidar']['n_beams']} beams, σ {S['lidar']['range_noise']} m"),
            ("Base", "—", f"r={rc['wheel_radius']} m, b={rc['wheel_base']} m, v≤{rc['v_max']} m/s")]
    b.axis("off")
    tb = b.table(cellText=rows, colLabels=["Sensor", "Rate", "Model (Table II)"], loc="center", cellLoc="left")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    tb.scale(1, 1.5)
    b.set_title("(b) Simulated sensor suite (design parameters)")
    save(fig, cfg, out, "fig02_robot_sensors")


# ---------------------------------------------------------------- Figs. 3/4
def _render(cfg, world, pose, deg, seed=0):
    cam = Camera(cfg)
    return cam.expose(cam.irradiance(world, pose), rng_stream(seed, "fig"), deg, 0.0, rng_stream(seed, "figd"))


def fig3_4_sensor_examples(cfg, out, episodes):
    """episodes: dict condition -> episode path (showcase set)."""
    ep = load_episode(episodes["nominal"])
    world = make_world(ep["meta"]["world"])
    t_show = float(np.mean(cfg["sim"]["deg_window"]))          # inside the degradation window
    k = min(int(np.searchsorted(ep["gt_t"], t_show)), len(ep["gt_t"]) - 1)
    pose = (ep["gt_x"][k], ep["gt_y"][k], ep["gt_th"][k])
    jl = min(int(np.searchsorted(ep["lidar_t"], t_show)), len(ep["lidar_t"]) - 1)

    def scan_ax(ax, e, title):
        r = e["lidar_ranges"][jl].astype(float)
        a = e["lidar_angles"]
        ok = np.isfinite(r)
        ax.scatter(r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok]), s=2, color=SENSOR_STYLE["lidar"][0])
        ax.plot(0, 0, marker=">", color="#0b0b0b")
        ax.set_aspect("equal")
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
        ax.set_title(title)
        ax.set_xlabel("x (m)")

    # Fig. 3 nominal
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.9))
    ax[0].imshow(_render(cfg, world, pose, {}), cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("Camera (nominal)")
    ax[0].axis("off")
    scan_ax(ax[1], ep, "LiDAR scan")
    ax[2].plot(ep["imu_t"], ep["imu_gyro"], lw=0.6, color=SENSOR_STYLE["imu"][0], label="gyro z")
    ax[2].plot(ep["gt_t"], ep["gt_w"], lw=0.8, color="#0b0b0b", label="truth")
    ax[2].set_xlabel("t (s)")
    ax[2].set_ylabel("ω (rad/s)")
    ax[2].legend()
    ax[3].plot(ep["odom_t"], ep["odom_v"], lw=0.6, color=SENSOR_STYLE["odom"][0], label="odometry v")
    ax[3].plot(ep["gt_t"], ep["gt_v"], lw=0.8, color="#0b0b0b", label="truth")
    ax[3].set_xlabel("t (s)")
    ax[3].set_ylabel("v (m/s)")
    ax[3].legend()
    save(fig, cfg, out, "fig03_nominal_sensor_data")

    # Fig. 4 degraded
    cam_degs = [("cam.lowlight.3", "Low light (3)"), ("cam.defocus.3", "Defocus (3)"),
                ("cam.occlusion.2", "Occlusion (2)"), ("cam.noise.3", "Noise (3)")]
    fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.8))
    for i, (c, title) in enumerate(cam_degs):
        deg = active_params(parse_condition(c, 0, 1e9), "cam", 1.0)
        ax[0, i].imshow(_render(cfg, world, pose, deg), cmap="gray", vmin=0, vmax=1)
        ax[0, i].set_title(title)
        ax[0, i].axis("off")
    for i, (c, title) in enumerate([("lidar.dropout.3", "LiDAR dropout 50%"), ("lidar.fov.3", "LiDAR FOV blocked 270°")]):
        if c in episodes:
            scan_ax(ax[1, i], load_episode(episodes[c]), title)
        else:
            ax[1, i].axis("off")
            ax[1, i].text(0.5, 0.5, f"{title}:\nshowcase episode\nnot generated", ha="center", va="center", fontsize=7)
    if "imu.bias.2" in episodes:
        e = load_episode(episodes["imu.bias.2"])
        ax[1, 2].plot(e["imu_t"], e["imu_gyro"], lw=0.5, color=SENSOR_STYLE["imu"][0], label="gyro z")
        ax[1, 2].plot(e["gt_t"], e["gt_w"], lw=0.8, color="#0b0b0b", label="truth")
        _shade(ax[1, 2], cfg)
        ax[1, 2].set_title("IMU gyro bias step")
        ax[1, 2].set_xlabel("t (s)")
        ax[1, 2].legend()
    if "odom.slip.3" in episodes:
        e = load_episode(episodes["odom.slip.3"])
        ax[1, 3].plot(e["odom_t"], e["odom_v"], lw=0.5, color=SENSOR_STYLE["odom"][0], label="odometry v")
        ax[1, 3].plot(e["gt_t"], e["gt_v"], lw=0.8, color="#0b0b0b", label="truth")
        _shade(ax[1, 3], cfg)
        ax[1, 3].set_title("Wheel slip (3)")
        ax[1, 3].set_xlabel("t (s)")
        ax[1, 3].legend()
    fig.tight_layout()
    save(fig, cfg, out, "fig04_degraded_sensor_data")


# ---------------------------------------------------------------- Figs. 5/6
def fig5_6_conf_weights(cfg, out, trace_path, method="proposed"):
    z = np.load(trace_path)
    for idx, name, col in ((1, "fig05_confidence_over_time", "Confidence $c_i(t)$"),
                           (2, "fig06_weights_over_time", "Weight $w_i(t)=g(c_i)$")):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
        for ax, tr in zip(axes, ("map", "odom")):
            for s, (color, lab) in SENSOR_STYLE.items():
                k = f"{method}__{tr}__log_{s}"
                if k not in z:
                    continue
                L = z[k]
                # 0.5 s moving median for readability; raw values in the npz
                t = L[:, 0]
                v = pd.Series(L[:, idx]).rolling(max(1, int(0.5 * len(t) / (t[-1] - t[0] + 1e-9))), min_periods=1).median()
                ax.plot(t, v, color=color, label=lab, lw=1.2)
            _shade(ax, cfg)
            ax.set_title(f"{tr} track")
            ax.set_xlabel("t (s)")
            ax.set_ylim(-0.05, 1.05)
        axes[0].set_ylabel(col)
        axes[0].legend(loc="lower left")
        save(fig, cfg, out, name)


# ---------------------------------------------------------------- Figs. 7-10
def fig_traj(cfg, out, ep_path, trace_path, name, title, methods=("ekf", "rule", "proposed", "oracle")):
    ep = load_episode(ep_path)
    z = np.load(trace_path)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    t = ep["gt_t"]
    t_on, t_off = cfg["sim"]["deg_window"]
    for ax, tr in zip(axes, ("map", "odom")):
        ax.plot(ep["gt_x"], ep["gt_y"], color="#0b0b0b", lw=2.0, label="Ground truth")
        for m in methods:
            k = f"{m}__{tr}__est"
            if k not in z:
                continue
            c, mk, lab = _mstyle(m)
            e = z[k]
            ax.plot(e[:, 0], e[:, 1], color=c, lw=1.0, label=lab, ls="--" if m == "oracle" else "-")
            w = (t >= t_on) & (t < t_off)
            ax.plot(e[w, 0], e[w, 1], color=c, lw=2.2, alpha=0.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{title} — {tr} track")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    axes[1].legend(loc="best")
    fig.text(0.5, -0.02, "Thick translucent segments: degradation window.", ha="center", fontsize=7)
    save(fig, cfg, out, name)


# ---------------------------------------------------------------- Figs. 11/12
def fig_severity(cfg, out, df, metric, ylabel, name, methods):
    d = df[df.group.isin(["cam", "lidar", "imu", "odom"])].copy()
    d["type"] = d.condition.str.rsplit(".", n=1).str[0]
    d["level"] = d.condition.str.rsplit(".", n=1).str[1].astype(int)
    nom = df[df.condition == "nominal"]
    for tr in sorted(d.track.unique()):
        types = sorted(d.type.unique())
        nc = 5
        nr = int(np.ceil(len(types) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(7.2, 1.6 * nr), sharey=False, squeeze=False)
        for ax, ty in zip(axes.flat, types):
            for m in methods:
                c, mk, lab = _mstyle(m)
                x0 = nom[(nom.track == tr) & (nom.method == m)][metric]
                pts = [(0, x0)] + [(lv, d[(d.track == tr) & (d.method == m) & (d.type == ty) & (d.level == lv)][metric])
                                   for lv in sorted(d[d.type == ty].level.unique())]
                xs, mu, lo, hi = [], [], [], []
                for lv, x in pts:
                    if len(x) == 0:
                        continue
                    ci = bootstrap_ci(x.values, np.mean, n=2000)
                    xs.append(lv)
                    mu.append(x.mean())
                    lo.append(x.mean() - ci[0])
                    hi.append(ci[1] - x.mean())
                if xs:
                    ax.errorbar(xs, mu, yerr=[lo, hi], color=c, marker=mk or "o", ms=4, lw=1.2, capsize=2,
                                label=lab, ls="--" if m == "oracle" else "-")
            ax.set_title(ty, fontsize=7)
            ax.set_yscale("log")
            ax.set_xticks(range(0, 5))
        for ax in axes.flat[len(types):]:
            ax.axis("off")
        axes[0, 0].set_ylabel(ylabel)
        h, lab = axes[0, 0].get_legend_handles_labels()
        for ax in axes[-1]:
            ax.set_xlabel("severity (0 = nominal)", fontsize=7)
        fig.tight_layout()
        fig.legend(h, lab, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.0), frameon=False)
        save(fig, cfg, out, f"{name}_{tr}")


# ---------------------------------------------------------------- Fig. 13
def fig13_recovery(cfg, out, df, methods):
    d = df[(df.condition != "nominal") & df.method.isin(methods)]
    for tr in sorted(d.track.unique()):
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        groups = ["cam", "lidar", "imu", "odom", "multi"]
        w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            c, mk, lab = _mstyle(m)
            data = [d[(d.track == tr) & (d.method == m) & (d.group == g) & (d.recovery_status != "undisturbed")]
                    .recovery_time.dropna().values for g in groups]
            pos = np.arange(len(groups)) + (i - len(methods) / 2 + 0.5) * w
            bp = ax.boxplot([x if len(x) else [np.nan] for x in data], positions=pos, widths=w * 0.85,
                            patch_artist=True, showfliers=False, medianprops=dict(color="#0b0b0b"))
            for b in bp["boxes"]:
                b.set(facecolor=c, alpha=0.7, edgecolor=c)
            ax.plot([], [], color=c, lw=6, label=lab)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups)
        ax.set_ylabel("recovery time after degradation ends (s)")
        ax.set_title(f"Recovery time (disturbed runs only; censored runs at episode end) — {tr} track")
        ax.legend(ncol=4, loc="upper left")
        save(fig, cfg, out, f"fig13_recovery_{tr}")


# ---------------------------------------------------------------- Figs. 14/15
def fig14_15_compute(cfg, out, comp):
    agg = comp.groupby("estimator").agg(latency=("latency_median_us", "median"), p95=("latency_p95_us", "median"),
                                        params=("n_params", "sum"), size=("size_bytes", "sum")).sort_values("latency")
    for col, lab, name in (("latency", "single-sample inference latency (µs, median over sensors)", "fig14_latency"),
                           ("params", "parameters (sum over sensors)", "fig15_model_size")):
        fig, ax = plt.subplots(figsize=(7.2, 2.4))
        colors = ["#e34948" if e == "cnn" else ("#eb6834" if e == "rule" else "#2a78d6") for e in agg.index]
        ax.bar(range(len(agg)), agg[col], color=colors, width=0.7)
        ax.set_yscale("log")
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg.index, rotation=30, ha="right")
        ax.set_ylabel(lab)
        if col == "params":
            for i, v in enumerate(agg["size"]):
                ax.text(i, agg[col].iloc[i], f"{v / 1024:.0f} kB", ha="center", va="bottom", fontsize=6)
        save(fig, cfg, out, name)


# ---------------------------------------------------------------- Fig. 16
def fig16_calibration(cfg, out, preds: dict):
    """preds: sensor -> {estimator: (c, y)} on the TEST dataset."""
    sensors = list(preds)
    fig, axes = plt.subplots(1, len(sensors), figsize=(7.2, 2.0), squeeze=False)
    cols = {"rule": "#eb6834", "proposed": "#e34948", "cnn": "#008300"}
    for ax, s in zip(axes[0], sensors):
        ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=0.8, ls=":")
        for est, (c, y) in preds[s].items():
            rc = reliability_curve(c, y)
            if len(rc):
                ax.plot(rc[:, 1], rc[:, 2], marker="o", ms=3, color=cols.get(est, "#2a78d6"), label=est)
        ax.set_title(s)
        ax.set_xlabel("predicted confidence")
        ax.set_aspect("equal")
    axes[0, 0].set_ylabel("observed reliable fraction")
    axes[0, 0].legend()
    save(fig, cfg, out, "fig16_calibration")


# ---------------------------------------------------------------- Fig. 17
def fig17_ablation(cfg, out, tests):
    for tr in sorted(tests.track.unique()):
        d = tests[tests.track == tr].sort_values("ratio_of_means")
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        y = np.arange(len(d))
        lo = d.ratio_of_means - d.ratio_ci.map(lambda s: literal_eval(s)[0] if isinstance(s, str) else s[0])
        hi = d.ratio_ci.map(lambda s: literal_eval(s)[1] if isinstance(s, str) else s[1]) - d.ratio_of_means
        ax.errorbar(d.ratio_of_means, y, xerr=[lo, hi], fmt="o", color="#2a78d6", capsize=2)
        ax.axvline(1.0, color=NEUTRAL, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(d.baseline)
        ax.set_xlabel("mean RMSE(proposed) / mean RMSE(variant)   (<1: proposed better; 95% bootstrap CI)")
        ax.set_title(f"Ablations, all degraded conditions — {tr} track")
        save(fig, cfg, out, f"fig17_ablation_{tr}")


# ---------------------------------------------------------------- Fig. 18
def fig18_heatmap(cfg, out, rob, methods):
    for tr in sorted(rob.track.unique()):
        d = rob[(rob.track == tr) & rob.method.isin(methods)].pivot(index="method", columns="condition",
                                                                     values="increase_pct")
        d = d.reindex([m for m in methods if m in d.index])
        fig, ax = plt.subplots(figsize=(7.2, 0.35 * len(d) + 1.6))
        v = np.log10(1 + np.clip(d.values, 0, None) / 100)
        im = ax.imshow(v, aspect="auto", cmap="Blues")
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([_mstyle(m)[2] for m in d.index])
        ax.set_xticks(range(d.shape[1]))
        ax.set_xticklabels(d.columns, rotation=90, fontsize=5)
        cb = fig.colorbar(im, ax=ax, fraction=0.02)
        cb.set_label("log10(1 + median error increase)")
        ax.set_title(f"Median % position-RMSE increase vs. same-seed nominal run — {tr} track")
        ax.grid(False)
        save(fig, cfg, out, f"fig18_robustness_heatmap_{tr}")
