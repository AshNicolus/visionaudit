"""Paired per-fold deltas of CV runs against a baseline run (docs/cv/<name>.json)."""
import json
import sys
from pathlib import Path

import numpy as np

CV = Path(__file__).resolve().parent / "docs" / "cv"
# delta sign: positive is better (box_l1 is flipped)
SIGN = {"acc": 1, "iou": 1, "box_l1": -1}


# options a candidate may change; anything else must match the baseline exactly
KNOBS = {"name", "ema", "crop_prob", "label_smoothing", "weight_decay", "warmup", "width",
         "deep", "tb", "out_dir", "ann_path", "meta_path", "img_dir"}


def load(name):
    path = Path(name) if name.endswith(".json") else CV / f"{name}.json"
    return json.loads(path.read_text())


def check_comparable(base, cand):
    """Refuse to pair runs that differ in folds, seeds, schedule or data."""
    a, b = base["args"], cand["args"]
    bad = {k: (a.get(k), b.get(k)) for k in (a.keys() | b.keys()) - KNOBS if a.get(k) != b.get(k)}
    if bad:
        raise SystemExit(f"not comparable ({base['name']} vs {cand['name']}): {bad}")


def compare(base, cand, mode="plain"):
    rows = []
    for m, s in SIGN.items():
        d = np.array([s * (c[mode][m] - b[mode][m]) for b, c in zip(base["folds"], cand["folds"])])
        mean, std = d.mean(), d.std(ddof=1)
        verdict = "better" if mean > std else "worse" if mean < -std else "no change"
        rows.append((m, d, mean, std, verdict))
    return rows


def main(base_name, names):
    base = load(base_name)
    for mode in ("plain", "tta"):
        print(f"== {mode} vs {base_name}")
        for n in names:
            cand = load(n)
            check_comparable(base, cand)
            s = cand[mode]
            print(f"{n}: " + " ".join(f"{k} {v['mean']:.3f}+-{v['std']:.3f}" for k, v in s.items()))
            for m, d, mean, std, verdict in compare(base, cand, mode):
                print(f"   {m:7s} deltas {np.round(d, 3).tolist()} "
                      f"mean {mean:+.3f} sd {std:.3f} -> {verdict}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
