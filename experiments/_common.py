"""Shared CLI plumbing for the experiment scripts."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dasf.experiment import default_workers  # noqa: E402
from dasf.utils import load_config  # noqa: E402


def parse(desc, extra=None):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default="configs/matrix.yaml")
    ap.add_argument("--workers", type=int, default=default_workers())
    if extra:
        extra(ap)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.get("smoke"):
        print("*** SMOKE CONFIG: outputs are pipeline checks, NOT paper results ***")
    return args, cfg
