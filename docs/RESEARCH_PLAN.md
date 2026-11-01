# Research plan, protocol and results checklist

Companion to `paper/main.tex`. Covers deliverables B–L of the brief: methodology, maths
summary, experiment matrix, required tables/figures, statistics, repository/code
architecture, data to collect, the measured-vs-pending checklist, and what must be run
before any claim can be made.

**Status (2026-09-27): no experimental result has been measured.** The code runs end to end
on a smoke configuration (`configs/smoke.yaml`, disjoint seeds, 40 s episodes); smoke outputs
are pipeline checks and must never be quoted.

---

## B. Experimental methodology (summary)

1. **Simulate** paired episodes: ground truth depends only on (world, trajectory, seed);
   nominal sensor noise depends only on the seed (common random numbers); each degradation
   uses its own random stream. So every method and every condition see the same robot motion
   and the same nominal noise.
2. **Front-ends** are real algorithms (ground-plane direct VO, point-to-line ICP), so degraded
   measurement errors emerge from the data.
3. **Calibrate** nominal covariances R0 on uninjected training samples; tune one shared
   per-sensor inflation kappa on a nominal-only tuning split with the plain EKF; freeze both.
4. **Label** each measurement reliable iff its chi-square distance to ground truth under R0 is
   below the 95% quantile.
5. **Train** rule-based (nominal statistics only) and learned estimators (LR, RF, GBM, MLP ×
   static/temporal features) on the train split; Platt-calibrate on validation.
6. **Select** the proposed configuration on validation with the pre-registered rule (mean
   validation log position RMSE, ties < 0.01 → fewer parameters).
7. **Test**: run all 26 methods on all 1980 test episodes in both tracks; compute metrics.
8. **Analyse** with block-aggregated paired non-parametric tests + Holm correction.

## C. Mathematical formulation (summary; full version in paper §III, §VII)

* State `x = [p_x, p_y, θ, v, ω, a, b_g]`, CA/CTR process model, Joseph-form EKF.
* Measurement models: odom/cam `[v, ω]`; IMU `[ω + b_g, a, vω]`; LiDAR map `[p_x, p_y, θ]`;
  LiDAR odometry `[v, ω]`.
* Label: `y = 1[eᵀ R0⁻¹ e ≤ χ²_m(0.95)]`, `e = z − h(x_gt)`.
* Confidence: `c = P̂(y = 1 | φ)`, Platt-calibrated.
* Fusion: `R_eff = κ R0 / g(c)`, skip if `g(c) < 0.02`; equivalent to the weighted MAP
  problem `min ‖x − x⁻‖²_{P⁻⁻¹} + Σ g(c_i) ‖z_i − h_i(x)‖²_{(κ_i R0_i)⁻¹}`.
* `g ∈ {c, c², 1[c ≥ 0.5], exp(4(c − 1))}`; the sum-to-one convex alternative is evaluated as
  a complementary filter (Baseline 2 and an ablation).

## D. Experiment matrix

| Factor | Levels |
|---|---|
| Conditions | 66 = 1 nominal + 18 camera + 14 LiDAR + 14 IMU + 11 odometry + 8 multi (exact list: `configs/matrix.yaml`) |
| Worlds | W1 office, W2 corridor, W3 feature-poor (never used for training/tuning/selection) |
| Trajectories | loop, random tour |
| Test seeds | 3000–3004 |
| Episodes | 66 × 3 × 2 × 5 = **1980**, 90 s, degradation in [30, 60) s |
| Tracks | map (scan-to-map LiDAR), odometry (scan-to-scan LiDAR, no map) |
| Methods | 26 (see `src/dasf/fusion/methods.py`) |
| Method runs | 1980 × 26 × 2 ≈ 103 000 |
| Other splits | tune 40 nominal (1500–1509), train 480 random-degradation (1000–1119, W1/W2), val 60 (2000–2014) |

Severity levels per type: paper Table I / `src/dasf/degradation/catalog.py`.

**Optional experiments (supported, not in the main matrix):**
* E-gen: leave-one-degradation-type-out training (`random_training_events(exclude=...)`).
* E-seeds: 10 instead of 5 test seeds if CIs are too wide (change `splits.test.seeds`).
* E-SBC: repeat step 07 on a Raspberry Pi 4 / Jetson class board.

## E. Required tables (auto-generated → `tables/`)

| Paper table | File | Content |
|---|---|---|
| III main | `main_pos_rmse.{csv,tex}` | mean ± std position RMSE per method × group × track |
| (suppl.) | `main_ori_rmse`, `main_rpe1`, `main_pos_rmse_degwin` | orientation RMSE, RPE 1 s, window RMSE |
| IV tests | `tests_proposed_vs_baselines` | Δ, ratio, CIs, r_rb, Holm p |
| per group | `tests_group_{cam,lidar,imu,odom,multi}` | proposed vs EKF / rule / gating |
| H2 | `tests_H2_recovery`, `recovery_failure.csv` | recovery time, censoring, failure/success |
| H3/H4 | `hypotheses_H3_H4.csv`, `tests_nominal_cost` | non-inferiority, nominal cost, single vs multi |
| severity | `severity.csv` | per type × level × method |
| robustness | `robustness_increase.csv` | median % increase vs same-seed nominal |
| Friedman | `friedman.csv` | omnibus test + mean ranks |
| V confidence | `confidence_quality.csv` | Brier, ECE, AUROC/AUPRC, Spearman |
| VI ablation | `tests_ablations` | ratio, CI, Holm p |
| VII compute | `compute_estimators.csv`, `compute_features.csv`, `compute_filters.csv`, `compute_platform.json` | params, size, latency, heap, duty, RTF, CPU% |

## F. Required figures (auto-generated → `figures/`)

| Fig | File | x | y | Aggregation / error bars |
|---|---|---|---|---|
| 1 | TikZ in paper | – | – | architecture (not data) |
| 2 | `fig02_robot_sensors` | x [m] | y [m] | design parameters (not data) |
| 3 | `fig03_nominal_sensor_data` | image; scan x,y [m]; t [s] | ω [rad/s], v [m/s] | single showcase episode |
| 4 | `fig04_degraded_sensor_data` | as Fig 3 | as Fig 3 | single episodes per degradation |
| 5 | `fig05_confidence_over_time` | t [s] | c_i [–] | one episode, 0.5 s moving median; grey = window |
| 6 | `fig06_weights_over_time` | t [s] | w_i = g(c_i) | as Fig 5 |
| 7 | `fig07_traj_nominal` | x [m] | y [m] | one episode, both tracks |
| 8–10 | `fig08_traj_camera`, `fig09_traj_lidar`, `fig10_traj_imu` | x [m] | y [m] | one episode; window highlighted |
| 11 | `fig11_rmse_severity_{map,odom}` | severity level | position RMSE [m], log | mean over 30 episodes, 95% bootstrap CI |
| 12 | `fig12_orientation_severity_*` | severity level | orientation RMSE [deg] | as Fig 11 |
| 13 | `fig13_recovery_*` | degradation group | recovery time [s] | box plots, disturbed runs |
| 14 | `fig14_latency` | estimator | median latency [µs], log | median over sensors; p95 in CSV |
| 15 | `fig15_model_size` | estimator | parameters, log (kB annotated) | sum over sensors |
| 16 | `fig16_calibration` | predicted confidence | observed reliable fraction | 15 equal-width bins, test split |
| 17 | `fig17_ablation_*` | RMSE ratio | ablation | ratio of means, 95% paired bootstrap CI |
| 18 | `fig18_robustness_heatmap_*` | condition | method | median % increase, log colour |

Legends: every multi-series figure carries a legend; methods have fixed colours and markers
(`src/dasf/visualization/style.py`); smoke-config figures are watermarked.

## G. Statistical analysis plan (pre-registered)

* **Unit of analysis.** A block = (world, trajectory, seed); 30 blocks. Pooled comparisons
  average conditions within a block first (conditions share seeds → not independent).
* **Primary test.** Two-sided Wilcoxon signed-rank on paired block values (errors are
  positive, skewed, with divergences → no normality assumption).
* **Effect sizes.** Median paired difference and ratio of means, each with 95% percentile
  bootstrap CI (10 000 paired resamples); matched-pairs rank-biserial correlation.
* **Secondary.** Paired t-test on log errors only when Shapiro–Wilk does not reject normality
  of log differences.
* **Multiplicity.** Holm within each table/family (per track).
* **Omnibus.** Friedman test over main methods; post-hoc via the Holm-corrected Wilcoxon
  tests (Demšar 2006).
* **Why not ANOVA.** Repeated-measures ANOVA assumes normal, homoscedastic residuals, which
  diverging runs violate; a linear mixed model on log RMSE with random block effects is an
  acceptable robustness check, not the primary analysis.
* **H1** supported if proposed < EKF and < fixed-weight on pooled degraded blocks,
  p_Holm < 0.05, in a track; reported per track and per group.
* **H2** supported if recovery time is lower vs EKF (p_Holm < 0.05) without higher censoring.
* **H3** supported if (a) latency and CPU duty are ≥ 10× below the heavy model and the filter
  RTF stays < 0.1, and (b) the upper 95% CI of RMSE(proposed)/RMSE(heavy) < 1.05.
* **H4** supported if the relative improvement over EKF is larger in single-sensor groups than
  in the 4-sensor condition (Wilcoxon on block differences).
* **Nominal cost.** Report RMSE ratio vs EKF on nominal; a significant increase is a negative
  result to be reported.
* Report all outcomes including non-significant and adverse ones.

## H. Repository structure

```
senosrfail/
├── configs/            default.yaml (all parameters) · matrix.yaml (full study) · smoke.yaml
├── simulation/         README: Tier 1 (implemented) / Tier 2 ROS 2+Gazebo plan
├── src/dasf/
│   ├── simulation/     world.py (maps, textures, ray casting, prior map) · robot.py (trajectories)
│   ├── sensors/        camera.py (renderer, exposure) · frontends.py (VO, ICP)
│   ├── degradation/    catalog.py · camera.py · lidar.py · imu.py · odometry.py
│   ├── confidence/     features.py · labels.py · estimators.py · neural.py · bank.py
│   ├── fusion/         ekf.py · weighting.py · methods.py · runner.py
│   ├── evaluation/     metrics.py · stats.py
│   ├── visualization/  style.py · figures.py
│   ├── episode.py      episode generation / IO
│   └── experiment.py   splits, calibration, datasets, parallel runs
├── experiments/        01…08 pipeline steps · run_all.py
├── results/            episodes/ datasets/ models/ runs/ heavy_conf/ calibration.json (generated)
├── tables/  figures/   generated outputs (smoke outputs in */smoke)
├── notebooks/          exploratory analysis only (never a source of reported numbers)
├── tests/              unit tests (pytest)
├── docs/               this plan
└── paper/              main.tex · references.bib · CITATION_VERIFICATION.md
```

## I. Code architecture / pipeline

| Step | Script | Input → Output |
|---|---|---|
| 1 | `01_generate_episodes.py` | configs → `results/episodes/{tune,train,val,test}/*.npz` |
| 2 | `02_calibrate_and_build_datasets.py` | train/tune episodes → `calibration.json` (R0, κ); all splits → `results/datasets/{split}_{sensor}.npz` |
| 3 | `03_train_confidence.py` | datasets → `results/models/{sensor}/{estimator}.pkl`, `val_*.csv`, `selection.json` |
| 4 | `04_train_heavy.py` | datasets → CNN models; re-rendered camera pass → `results/heavy_conf/*.npz` |
| 5 | `05_run_fusion.py` | test episodes + models → `results/runs/test_metrics.csv`, showcase traces |
| 6 | `06_analyze.py` | metrics + test datasets → `tables/*` |
| 7 | `07_benchmark_compute.py` | models → `tables/compute_*` |
| 8 | `08_make_figures.py` | everything → `figures/*` |

`python experiments/run_all.py --config configs/matrix.yaml` runs all steps.
Rough cost from the smoke run (to be re-measured): generation ≈ 15–20 s per 90 s episode per
core; a method run ≈ 0.3–2 s (map-track ICP dominates). Full study ≈ 2 500 episodes and
≈ 10⁵ method runs → on the order of a day of single-core CPU, a few hours with 14 workers.

## J. Exact data that must be collected

Per **episode** (`results/episodes/<split>/<id>.npz`):
`gt_{t,x,y,th,v,w,a}` (100 Hz); `imu_{t,gyro,acc,valid,bg_nominal,injected}`;
`odom_{t,v,w,valid,v_true,w_true,injected}`; `cam_{t,v,w,v_true,w_true,dt,feats,injected}`
(+ `cam_images` pairs for train/val); `lidar_{t,angles,ranges,gt,injected}`;
`lrel_{t,v,w,v_true,w_true,scan,icp,injected}`; `meta` (world, trajectory, seed, events,
condition, split).

Per **confidence sample** (`results/datasets/<split>_<sensor>.npz`): static features `X`,
temporal features `Xt`, label `y`, `d2`, `injected`, `t`, episode id, event types, raw input
(`raw`/`imgs`) for the heavy model.

Per **method run** (`results/runs/test_metrics.csv`, one row per method × track × episode):
pos RMSE/MAE/max/final, orientation RMSE, drift %, RPE 1 s/5 s (trans, rot), window RMSEs,
recovery time + status + threshold, failure flag, path length, NEES, wall time, RTF,
confidence time, per-sensor update/skip/reject counts, ICP failures, identifiers.

Per **showcase episode**: full estimated trajectories and per-measurement (t, c, w) logs.

Per **estimator**: parameters, serialized size, single-sample latency (median, p95), GPU
latency (heavy), Python heap peak, CPU duty; per method: RTF, CPU utilisation, confidence-time
fraction; platform description.

Provenance to archive with the paper: git commit, `configs/*.yaml`, `calibration.json`,
`selection.json`, `test_run_info.json`, `pip freeze`.

## K. Checklist — measured vs pending

| Item | Status |
|---|---|
| Simulator, degradations, front-ends, EKF, estimators, metrics, stats, figures | **Implemented**; 24 unit tests pass |
| End-to-end pipeline execution | **Verified on smoke config only** (not results) |
| Citations | 82 entries checked against Crossref/publisher records (see `paper/CITATION_VERIFICATION.md`); 1 (Kuglin & Hines 1975) confirmed only via secondary sources |
| R0 calibration, κ tuning (full study) | PENDING |
| Estimator training + validation selection (full study) | PENDING |
| Heavy baseline training | PENDING |
| Test runs, all tables (III–VII) | PENDING — every cell TBD |
| Figures 3–18 from the full study | PENDING (Fig. 2 is design-only) |
| H1, H2, H3, H4 | PENDING — no claim may be made |
| Compute on target SBC | PENDING |
| Leave-one-type-out generalisation | PENDING (optional) |
| Tier-2 Gazebo replication | NOT IMPLEMENTED (specified) |
| Real-robot validation | NOT PLANNED in this study |

Design changes made after smoke testing (disclose in the paper; all before any full-study
run, on seeds disjoint from the study):
1. Camera renderer anti-aliasing (3×3 supersampling): point-sampled texture biased VO
   forward speed by ≈ −30% on a known-motion test.
2. Shared per-sensor covariance inflation κ tuned on a nominal split: without it, the EKF was
   worse than LiDAR-only in the map-free track because odometry errors are time-correlated.
3. Rule-based estimator restricted to measurement-consistency checks (availability features
   flagged nominal data in the held-out feature-poor world); robust-scale floor for saturated
   features.
4. Added IMU/odometry dropout conditions.

## L. What must be run before the paper can claim an improvement

1. `python -m pytest tests -q` passes on the exact commit used.
2. Full pipeline on `configs/matrix.yaml` (steps 1–8), no manual edits to outputs; archive
   provenance files (J).
3. Confirm the selection in `results/models/selection.json` was produced before step 5 and
   from validation data only.
4. Inspect `tests_proposed_vs_baselines`: an improvement may be claimed **only** for the tracks
   and groups where proposed beats the EKF with Holm-adjusted p < 0.05 **and** the CI of the
   ratio of means excludes 1. Report magnitude with CIs, not just p.
5. Check the nominal-cost test; if the proposed method is significantly worse on nominal
   data, say so in the abstract.
6. Check the held-out world W3 separately (filter `world == W3_featurepoor`); if the benefit
   disappears there, the claim must be restricted to seen environments.
7. "Lightweight" requires step 7 numbers (ideally on the target SBC) and H3(a).
8. Compare with the oracle to report how much of the achievable gain is realised.
9. Replace every TBD only through the generated files; re-read the abstract, discussion and
   conclusion so that they state exactly what was measured, including negative results.
10. State clearly: simulation-only evidence; Tier-2/real-robot validation is future work.
