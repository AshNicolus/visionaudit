"""Paired per-fold deltas of CV runs against a baseline run (docs/cv/<name>.json)."""
import json
import sys
from pathlib import Path

import numpy as np

CV = Path(__file__).resolve().parent / "docs" / "cv"
# delta sign: positive is better (box_l1 is flipped)
SIGN = {"acc": 1, "iou": 1, "box_l1": -1}


def load(name):
    return json.loads((CV / f"{name}.json").read_text())


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
            s = cand[mode]
            print(f"{n}: " + " ".join(f"{k} {v['mean']:.3f}+-{v['std']:.3f}" for k, v in s.items()))
            for m, d, mean, std, verdict in compare(base, cand, mode):
                print(f"   {m:7s} deltas {np.round(d, 3).tolist()} mean {mean:+.3f} sd {std:.3f} -> {verdict}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
