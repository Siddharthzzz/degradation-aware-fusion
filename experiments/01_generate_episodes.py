"""Step 1: simulate and cache all episodes (ground truth + sensor streams + front-ends).

    python experiments/01_generate_episodes.py --config configs/matrix.yaml --splits train val test
"""
from _common import parse

from dasf.experiment import generate_split

if __name__ == "__main__":
    args, cfg = parse(__doc__, lambda ap: (ap.add_argument("--splits", nargs="+", default=["tune", "train", "val", "test"]),
                                           ap.add_argument("--overwrite", action="store_true")))
    for split in args.splits:
        generate_split(cfg, split, args.workers, args.overwrite)
