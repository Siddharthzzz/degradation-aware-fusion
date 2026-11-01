"""Registry of every evaluated method (baselines, proposed, ablations, oracle).

`conf` names an estimator family+feature set (e.g. 'lr_temporal'), 'rule',
'cnn', 'oracle' or None.  'PROPOSED' is resolved to the estimator selected on
the VALIDATION split (experiments/03_train_confidence.py writes it to
results/models/selection.json) -- the test split is never used for selection.
"""
from __future__ import annotations

ALL = ("cam", "lidar", "imu", "odom")


def M(kind="ekf", sensors=ALL, conf=None, conf_off=(), weight=None, gate=False, iae=False, group=""):
    return dict(kind=kind, sensors=tuple(sensors), conf=conf, conf_off=tuple(conf_off),
                weight=weight, gate=gate, iae=iae, group=group)


METHODS = {
    # ---- Baseline 1: single sensors (best one chosen on validation nominal data)
    "single_lidar": M(sensors=("lidar",), group="B1"),
    "single_odom": M(sensors=("odom",), group="B1"),
    "single_cam": M(sensors=("cam",), group="B1"),
    "single_imu": M(sensors=("imu",), group="B1"),
    # ---- Baseline 2: fixed-weight (convex, sum-to-one) complementary fusion
    "fixed_complementary": M(kind="comp", group="B2"),
    # ---- Baseline 3: standard EKF (fixed R0), and classical robust/adaptive EKFs
    "ekf": M(group="B3"),
    "ekf_gate": M(gate=True, group="B3"),
    "ekf_iae": M(iae=True, group="B3"),
    # ---- Baseline 4: rule-based confidence
    "rule": M(conf="rule", group="B4"),
    # ---- Proposed candidates (one is selected on validation)
    "learned_lr": M(conf="lr_temporal", group="P"),
    "learned_rf": M(conf="rf_temporal", group="P"),
    "learned_gbm": M(conf="gbm_temporal", group="P"),
    "learned_mlp": M(conf="mlp_temporal", group="P"),
    # ---- Baseline 5: heavy neural confidence
    "heavy_cnn": M(conf="cnn", group="B5"),
    # ---- Upper bound (not deployable)
    "oracle": M(conf="oracle", group="UB"),
    # ---- Ablations (all relative to PROPOSED)
    "abl_no_cam_conf": M(conf="PROPOSED", conf_off=("cam",), group="A"),
    "abl_no_lidar_conf": M(conf="PROPOSED", conf_off=("lidar",), group="A"),
    "abl_no_imu_conf": M(conf="PROPOSED", conf_off=("imu",), group="A"),
    "abl_no_odom_conf": M(conf="PROPOSED", conf_off=("odom",), group="A"),
    "abl_static": M(conf="PROPOSED_STATIC", group="A"),
    "abl_w_power": M(conf="PROPOSED", weight="power", group="A"),
    "abl_w_gate": M(conf="PROPOSED", weight="gate", group="A"),
    "abl_w_exp": M(conf="PROPOSED", weight="exp", group="A"),
    "abl_conf_complementary": M(kind="comp", conf="PROPOSED", group="A"),
    "abl_conf_plus_gate": M(conf="PROPOSED", gate=True, group="A"),
}

# Names used in the paper's main tables
MAIN = ["single_best", "fixed_complementary", "ekf", "ekf_gate", "ekf_iae", "rule",
        "proposed", "heavy_cnn", "oracle"]


def resolve(name: str, selection: dict | None) -> dict:
    """Return a concrete method spec; resolves PROPOSED placeholders."""
    if name == "proposed":
        if not selection:
            raise RuntimeError("no validation selection found; run 03_train_confidence.py")
        spec = dict(METHODS["learned_" + selection["family"]])
        spec["conf"] = f"{selection['family']}_{selection['feature_set']}"
        spec["weight"] = selection.get("weight")
        return spec
    if name == "single_best":
        if not selection or "single_best" not in selection:
            raise RuntimeError("single_best not selected yet")
        return dict(METHODS[selection["single_best"]])
    spec = dict(METHODS[name])
    if spec["conf"] in ("PROPOSED", "PROPOSED_STATIC"):
        if not selection:
            raise RuntimeError("no validation selection found")
        fs = "static" if spec["conf"] == "PROPOSED_STATIC" else selection["feature_set"]
        if spec["conf"] == "PROPOSED_STATIC" and selection["feature_set"] == "static":
            fs = "temporal"          # ablate in the other direction
        spec["conf"] = f"{selection['family']}_{fs}"
        if spec["weight"] is None:
            spec["weight"] = selection.get("weight")
    return spec
