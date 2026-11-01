"""Run the full pipeline in order.

    python experiments/run_all.py --config configs/smoke.yaml      # ~minutes, pipeline check only
    python experiments/run_all.py --config configs/matrix.yaml     # full study (hours; see README)
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_generate_episodes.py", "02_calibrate_and_build_datasets.py", "03_train_confidence.py",
         "04_train_heavy.py", "05_run_fusion.py", "06_analyze.py", "07_benchmark_compute.py",
         "08_make_figures.py"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/matrix.yaml")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--from-step", type=int, default=1)
    ap.add_argument("--heavy-args", default="", help="extra args for 04, e.g. '--arch small --epochs 3'")
    a = ap.parse_args()
    for i, s in enumerate(STEPS, 1):
        if i < a.from_step:
            continue
        cmd = [sys.executable, str(HERE / s), "--config", a.config]
        if a.workers:
            cmd += ["--workers", str(a.workers)]
        if s.startswith("04") and a.heavy_args:
            cmd += a.heavy_args.split()
        print(f"\n===== [{i}/{len(STEPS)}] {' '.join(cmd[1:])}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            if s.startswith("04"):
                print("Heavy baseline step failed; continuing (Baseline 5 will be reported as not run).")
                continue
            sys.exit(r.returncode)
