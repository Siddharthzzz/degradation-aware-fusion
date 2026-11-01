"""Step 2: (a) calibrate nominal R0 on uninjected TRAIN samples; (b) build the
confidence datasets (features, labels) for train / val / test.

    python experiments/02_calibrate_and_build_datasets.py --config configs/matrix.yaml
"""
from _common import parse

from dasf.experiment import build_datasets, calibrate_r0
from dasf.utils import load_config

if __name__ == "__main__":
    args, cfg = parse(__doc__, lambda ap: ap.add_argument("--skip-calibration", action="store_true"))
    if not args.skip_calibration:
        calibrate_r0(cfg, args.workers)
        cfg = load_config(args.config)                  # reload: picks up calibration.json
    print("R0 source:", cfg["fusion"]["r0_source"])
    for split in ("train", "val", "test"):
        build_datasets(cfg, split, args.workers)
