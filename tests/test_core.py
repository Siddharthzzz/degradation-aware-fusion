"""Unit tests for the components whose correctness every result depends on.

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dasf.confidence.estimators import RuleBased  # noqa: E402
from dasf.confidence.features import Temporal, lidar_scan_features  # noqa: E402
from dasf.degradation.catalog import CATALOG, parse_condition  # noqa: E402
from dasf.degradation.lidar import apply_lidar  # noqa: E402
from dasf.evaluation.metrics import ece, recovery_time, rpe  # noqa: E402
from dasf.evaluation.stats import holm, paired_compare, rank_biserial  # noqa: E402
from dasf.fusion.ekf import EKF, h_pose, h_velocity  # noqa: E402
from dasf.fusion.weighting import make_weight_fn, normalized_weights  # noqa: E402
from dasf.sensors.camera import Camera  # noqa: E402
from dasf.sensors.frontends import CameraVO, icp_scan_to_map, icp_scan_to_scan, phase_correlation  # noqa: E402
from dasf.simulation.robot import generate_trajectory  # noqa: E402
from dasf.simulation.world import WORLDS, make_world  # noqa: E402
from dasf.utils import load_config, wrap, zoh  # noqa: E402

CFG = load_config()


# ---------------------------------------------------------------- simulation
@pytest.mark.parametrize("name", WORLDS)
def test_roadmap_clearance(name):
    """Every roadmap edge keeps >= 0.35 m from walls (robot footprint ~0.3 m)."""
    w = make_world(name)
    for i, j in w.edges:
        a, b = w.waypoints[i], w.waypoints[j]
        pts = a + np.linspace(0, 1, 50)[:, None] * (b - a)
        assert w.clearance(pts).min() >= 0.35, (name, i, j)


def test_raycast_box():
    w = make_world("W1_office")
    d, _, _ = w.raycast((1.0, 1.0), np.array([np.pi, -np.pi / 2]))
    assert np.allclose(d, [1.0, 1.0], atol=1e-9)


def test_trajectory_consistent_and_collision_free():
    w = make_world("W1_office")
    gt = generate_trajectory(w, "loop", 3, CFG)
    dt = CFG["sim"]["dt"]
    # heading integrates omega
    dth = wrap(np.diff(gt["th"]))
    assert np.allclose(dth, gt["w"][:-1] * dt, atol=1e-9)
    assert w.clearance(np.column_stack([gt["x"], gt["y"]])).min() > 0.2
    assert gt["v"].max() <= CFG["robot"]["v_max"] + 1e-9


def test_trajectory_reproducible():
    w = make_world("W2_corridor")
    a = generate_trajectory(w, "random", 11, CFG)
    b = generate_trajectory(w, "random", 11, CFG)
    assert np.array_equal(a["x"], b["x"])


# ---------------------------------------------------------------- front-ends
def test_phase_correlation_shift():
    rng = np.random.default_rng(0)
    a = rng.random((32, 64))
    b = np.roll(a, 5, axis=1)
    _, sj, _ = phase_correlation(a, b)
    assert abs(sj - 5) < 0.3


def test_camera_vo_known_motion():
    w = make_world("W1_office")
    cam, vo = Camera(CFG), CameraVO(Camera(CFG), CFG)
    rng = np.random.default_rng(1)
    dt = 1 / 15
    est = []
    for k in range(4):
        p0 = (2.0 + 0.5 * k, 1.0, 0.0)
        p1 = (p0[0] + 0.2 * dt, 1.0, 0.3 * dt)
        i0 = cam.expose(cam.irradiance(w, p0), rng, {}, 0.0)
        i1 = cam.expose(cam.irradiance(w, p1), rng, {}, 0.0)
        v, om, _, _ = vo.estimate(i0, i1, dt)
        est.append((v, om))
    v, om = np.mean(est, 0)
    assert abs(v - 0.2) < 0.03 and abs(om - 0.3) < 0.05


def test_icp_scan_to_map_recovers_offset():
    w = make_world("W1_office")
    ang = np.arange(360) * np.pi / 180
    pose = np.array([4.0, 2.8, 0.3])
    r, _, _ = w.raycast(pose[:2], pose[2] + ang)
    r[r > 3.5] = np.inf
    x, f = icp_scan_to_map(r, ang, pose + [0.1, -0.08, 0.05], w, CFG)
    assert x is not None and np.linalg.norm(x[:2] - pose[:2]) < 0.01 and abs(wrap(x[2] - pose[2])) < 0.01
    assert f["icp_inlier_ratio"] > 0.9


def test_icp_scan_to_scan():
    w = make_world("W1_office")
    ang = np.arange(360) * np.pi / 180
    p0, p1 = np.array([4.0, 2.8, 0.3]), np.array([4.03, 2.81, 0.33])
    r0, _, _ = w.raycast(p0[:2], p0[2] + ang)
    r1, _, _ = w.raycast(p1[:2], p1[2] + ang)
    r0[r0 > 3.5] = np.inf
    r1[r1 > 3.5] = np.inf
    x, _ = icp_scan_to_scan(r0, r1, ang, np.zeros(3), CFG)
    c, s = np.cos(p0[2]), np.sin(p0[2])
    d = p1[:2] - p0[:2]
    rel = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], p1[2] - p0[2]])
    assert np.allclose(x, rel, atol=5e-3)


# ---------------------------------------------------------------- degradation
def test_condition_parsing_and_catalog():
    evs = parse_condition("cam.lowlight.2+odom.slip.3", 30, 60)
    assert evs[0].value == CATALOG["cam"]["lowlight"][1] and evs[1].value == CATALOG["odom"]["slip"][2]
    assert parse_condition("nominal", 0, 1) == []


def test_lidar_fov_obstruction():
    ang = np.arange(360) * np.pi / 180
    r = apply_lidar(np.ones(360), ang, {"fov": 90.0}, np.random.default_rng(0))
    assert np.isinf(r).sum() in (89, 90, 91)


def test_lidar_gap_feature():
    r = np.ones(360)
    r[350:] = np.inf
    r[:5] = np.inf
    f = lidar_scan_features(r, None)
    assert f["max_gap_deg"] == 15.0


# ---------------------------------------------------------------- fusion
def test_ekf_confidence_scaling_equals_covariance_scaling():
    """R_eff = R0 / w must equal an update with covariance R0/w (definition)."""
    x0 = [0, 0, 0, 0.2, 0.1, 0, 0]
    a, b = EKF(x0, CFG), EKF(x0, CFG)
    z = np.array([0.25, 0.05])
    hx, H = h_velocity(a.x)
    a.update(z, hx, H, np.diag([1e-4, 1e-3]) / 0.25)
    b.update(z, hx, H, np.diag([4e-4, 4e-3]))
    assert np.allclose(a.x, b.x) and np.allclose(a.P, b.P)


def test_ekf_pose_update_wraps_angle():
    k = EKF([0, 0, np.pi - 0.01, 0, 0, 0, 0], CFG)
    hx, H = h_pose(k.x)
    k.update(np.array([0, 0, -np.pi + 0.01]), hx, H, np.eye(3) * 1e-6, ang=[2])
    assert abs(abs(k.x[2]) - np.pi) < 0.02


def test_weight_functions():
    for name in ("linear", "power", "gate", "exp"):
        g = make_weight_fn(CFG, name)
        assert 0 <= g(0.0) <= g(0.5) <= g(1.0) <= 1.0 + 1e-12
    assert abs(sum(normalized_weights({"a": 1, "b": 3}).values()) - 1) < 1e-12


# ---------------------------------------------------------------- confidence / features
def test_temporal_features_causal():
    t = Temporal(1, 10.0, 1.0, 1.0)
    out1 = [t.update(np.array([v]), i * 0.1) for i, v in enumerate([1, 1, 1])]
    t2 = Temporal(1, 10.0, 1.0, 1.0)
    out2 = [t2.update(np.array([v]), i * 0.1) for i, v in enumerate([1, 1, 1, 100])]
    assert np.allclose(out1, out2[:3])            # a future sample never changes past features


def test_zoh_is_causal():
    v, age = zoh(np.array([0.0, 1.0]), np.array([5.0, 7.0]), np.array([-0.5, 0.5, 1.0, 2.0]))
    assert np.isnan(v[0]) and list(v[1:]) == [5.0, 7.0, 7.0]


def test_rule_based_monotone():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (500, 5))
    names = ["gyro_hf", "acc_hf", "stuck_frac", "d_w_odo", "d_w_cam"]
    r = RuleBased("imu", names).fit(X)
    lo = r.predict(np.zeros((1, 5)))
    hi = r.predict(np.array([[20.0, 0, 0, 0, 0]]))
    assert lo[0] > 0.99 and hi[0] < 0.05


# ---------------------------------------------------------------- metrics / stats
def test_rpe_zero_for_identical():
    P = np.column_stack([np.linspace(0, 1, 50), np.zeros(50), np.zeros(50)])
    assert rpe(P, P, 5) == (0.0, 0.0)


def test_recovery_time():
    t = np.arange(0, 10, 0.1)
    e = np.where((t >= 3) & (t < 6), 1.0, 0.0)
    rt, st = recovery_time(t, e, 5.0, 0.5, 1.0)
    assert st == "recovered" and abs(rt - 1.0) < 1e-9


def test_ece_perfect():
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert ece(p, np.array([0, 0, 1, 1])) == 0.0


def test_holm():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_paired_compare_direction():
    rng = np.random.default_rng(0)
    b = rng.uniform(1, 2, 30)
    a = b * 0.8
    r = paired_compare(a, b)
    assert r["median_diff"] < 0 and r["p_wilcoxon"] < 0.001 and rank_biserial(a - b) == -1.0
