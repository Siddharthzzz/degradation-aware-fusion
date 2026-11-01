"""Step 5: run every method on every TEST episode, both tracks; write metric rows.

    python experiments/05_run_fusion.py --config configs/matrix.yaml
"""
import json
from pathlib import Path

import pandas as pd
from _common import parse

from dasf.experiment import pmap, run_episode, split_paths
from dasf.fusion.methods import resolve as resolve_method
from dasf.utils import resolve

if __name__ == "__main__":
    args, cfg = parse(__doc__, lambda ap: (ap.add_argument("--split", default="test"),
                                           ap.add_argument("--methods", nargs="*")))
    sel_p = resolve(cfg, "models") / "selection.json"
    selection = json.load(open(sel_p)) if sel_p.exists() else None
    names = (args.methods or cfg["methods"]) + ["proposed"]
    named = []
    for n in names:
        try:
            named.append((n, resolve_method(n, selection)))
        except RuntimeError as exc:
            print(f"skipping {n}: {exc}")
    sc = cfg["showcase"]
    show = {f"{c.replace('+', '_')}__{sc['world']}__{sc['traj']}__s{sc['seed']}" for c in sc["conditions"]}
    runs = resolve(cfg, "runs")
    jobs = [(cfg, str(p), named, cfg["tracks"], str(runs / "traces") if p.stem in show else None)
            for p in split_paths(cfg, args.split)]
    print(f"{len(jobs)} episodes x {len(named)} methods x {len(cfg['tracks'])} tracks")
    res = pmap(run_episode, jobs, args.workers)
    df = pd.DataFrame([r for rs in res for r in rs])
    out = runs / f"{args.split}_metrics.csv"
    df.to_csv(out, index=False)
    errs = df[df.get("error").notna()] if "error" in df else []
    print(f"wrote {out} ({len(df)} rows; {len(errs)} method runs skipped with errors)")
    json.dump(dict(selection=selection, methods=[n for n, _ in named], r0=cfg["fusion"]["r0"],
                   r0_source=cfg["fusion"]["r0_source"]), open(runs / f"{args.split}_run_info.json", "w"), indent=2)
