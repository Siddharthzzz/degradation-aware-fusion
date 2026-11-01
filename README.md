# Degradation-Aware Multi-Sensor Fusion with Lightweight Confidence Estimation

Code for the paper *"Degradation-Aware Multi-Sensor Fusion with Lightweight Confidence Estimation for Resource-Constrained Mobile Robots"* (PCDS 2026).

Lightweight, Platt-calibrated per-measurement confidence estimators map causal sensor-quality features (camera, 2-D LiDAR, IMU, wheel odometry, and cross-sensor consistency) to a confidence. That confidence scales or gates the measurement covariance of an extended Kalman filter (EKF): `R_eff = kappa * R0 / g(c)`. The study is **simulation-only**.

## Repository layout

| Path | Contents |
|---|---|
| `src/dasf/simulation/` | 2-D world (W1 office, W2 corridor, W3 hall) and TurtleBot3-like robot |
| `src/dasf/sensors/` | Rendered camera, ray-cast LiDAR, IMU, encoders; VO and ICP front-ends |
| `src/dasf/degradation/` | Physical-stage degradation injection (camera, LiDAR, IMU, odometry) and the condition catalog |
| `src/dasf/confidence/` | Quality features, reliability labels, rule/LR/RF/GB/MLP estimators, heavy CNN baseline |
| `src/dasf/fusion/` | EKF, confidence weighting (`g = c, c^2, gate, exp`), all baselines, episode runner |
| `src/dasf/evaluation/` | Metrics (RMSE, failure, recovery) and statistics (paired bootstrap, Wilcoxon, Holm) |
| `experiments/01…09` | Pipeline: generate → calibrate/datasets → train/select → heavy baseline → fusion → analysis → compute benchmark → figures → paper tables |
| `configs/` | `default.yaml` (all parameters), `reduced.yaml` (the reported run), `matrix.yaml` (full 66-condition design), `smoke.yaml` (quick check) |
| `tables/`, `figures/` | Tables and figures produced by the reported run |
| `results/` | Selection record, calibration, and per-run test metrics (`results/runs/test_metrics.csv`) |
| `tests/` | Unit tests |
| `docs/RESEARCH_PLAN.md` | Methodology, condition matrix, statistics plan |

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision          # only for the heavy CNN baseline
python -m pytest tests -q
```

Tested with Python 3.13 on Windows.

## Reproducing the paper

```bash
# quick end-to-end pipeline check (minutes; outputs go to */smoke and must not be reported)
python experiments/run_all.py --config configs/smoke.yaml --heavy-args "--arch small --epochs 2"

# the reported, compute-budgeted run (25 conditions x 3 environments x 2 trajectories x 2 seeds)
python experiments/run_all.py --config configs/reduced.yaml

# the full planned design (66 conditions, 5 seeds; many CPU-hours)
python experiments/run_all.py --config configs/matrix.yaml
```

All random seeds are set in the configuration files. The pipeline regenerates the episode data (about 11 GB) and the trained models, which are therefore not stored in this repository. Tables are written to `tables/` and figures to `figures/`.

Compute measurements in the paper were taken on one thread of an Intel Core Ultra 9 185H (heavy model also on an RTX 4070 Laptop GPU) with scikit-learn and PyTorch. No embedded board was used.

## Citation

```bibtex
@inproceedings{siddharth2026degradation,
  author    = {Manda Siddharth and Akhil Varun Varma Kalidindi and Rishi Kamishetty and
               Bilakanti, Mukesh Reddy and Lakshmi Pavan Rangisetti and Veeraiah Talagondapati},
  title     = {Degradation-Aware Multi-Sensor Fusion with Lightweight Confidence Estimation
               for Resource-Constrained Mobile Robots},
  booktitle = {Proc. PCDS},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`).
