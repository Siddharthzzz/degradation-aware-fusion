"""Consistent figure style. Method colours follow the ENTITY (never rank):
fixed categorical order, validated reference palette (light mode, print),
plus a distinct marker per method so identity never relies on colour alone.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#6b6a66"
INK, INK2 = "#0b0b0b", "#52514e"

METHOD_STYLE = {
    "single_best":         (PALETTE[3], "v", "Single best sensor"),
    "fixed_complementary": (PALETTE[4], "D", "Fixed-weight fusion"),
    "ekf":                 (PALETTE[0], "o", "EKF"),
    "ekf_gate":            (PALETTE[2], "s", "EKF + χ² gating"),
    "ekf_iae":             (PALETTE[6], "^", "Innov.-adaptive EKF"),
    "rule":                (PALETTE[1], "P", "Rule-based conf."),
    "proposed":            (PALETTE[7], "*", "Proposed"),
    "heavy_cnn":           (PALETTE[5], "X", "Heavy neural conf."),
    "oracle":              (NEUTRAL, "", "Oracle (upper bound)"),
}
SENSOR_STYLE = {"cam": (PALETTE[0], "Camera"), "lidar": (PALETTE[1], "LiDAR"),
                "imu": (PALETTE[2], "IMU"), "odom": (PALETTE[3], "Wheel odom.")}


def setup():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.5, "lines.markersize": 5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def watermark(fig, cfg):
    if cfg.get("smoke"):
        fig.text(0.5, 0.5, "SMOKE TEST - NOT A RESULT", fontsize=18, color="#e34948", alpha=0.35,
                 ha="center", va="center", rotation=25, transform=fig.transFigure)


def save(fig, cfg, outdir, name):
    watermark(fig, cfg)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
