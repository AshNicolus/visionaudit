"""5-fold stratified group cross-validation on the training code in train.py.

Folds are fixed (seed 0) and grouped so same-series and near-duplicate images never straddle
train and test. Inside each training split a group-aware inner val (about 1/6) drives early
stopping and best-checkpoint choice; the held-out fold is only scored once, with those
weights. Same rule for every config, so runs are paired fold by fold.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

import train as T
from dataset import CocoDataset, load_coco
from groups import build_groups

HERE = Path(__file__).resolve().parent
FOLD_SEED = 0
N_FOLDS = 5


def image_classes(ann_path):
    """Class index per image (every image holds one class), in load_coco order."""
    images, by_image, cat_index = load_coco(ann_path)
    y = [cat_index[by_image[im["id"]][0]["category_id"]] for im in images]
    return images, np.array(y)


def make_folds(ann_path, meta_path, img_dir, fold_seed=FOLD_SEED):
    """List of (train_idx, inner_val_idx, test_idx) with grouped, stratified splits."""
    images, y = image_classes(ann_path)
    g = build_groups([im["file_name"] for im in images], meta_path, img_dir)
    outer = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=fold_seed)
    folds = []
    for k, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y, g)):
        inner = StratifiedGroupKFold(6, shuffle=True, random_state=fold_seed + k)
        a, b = next(inner.split(np.zeros(len(tr)), y[tr], g[tr]))
        folds.append((tr[a], tr[b], te))
    return folds, g


def run_fold(k, split, args, rep=0):
    tr, val, te = split
    mk = lambda idx, train, seed: CocoDataset(  # noqa: E731
        args.ann_path, args.img_dir, idx.tolist(), args.img_size, train=train, seed=seed,
        crop_prob=args.crop_prob if train else 0.0)
    seed = 100 + k + 1000 * rep
    T.set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    tr_loader = DataLoader(mk(tr, True, seed), batch_size=args.batch_size, shuffle=True,
                           generator=gen)
    val_loader = DataLoader(mk(val, False, 0), batch_size=args.batch_size)
    te_loader = DataLoader(mk(te, False, 0), batch_size=args.batch_size)
    model = T.CustomCNN(3, width=args.width, deep=args.deep)
    run_dir = HERE / "runs" / "cv" / args.name / f"fold{k}_s{rep}"
    hist = T.train(model, tr_loader, val_loader, args.epochs, args.lr, args.patience, seed,
                   run_dir=run_dir, box_weight=5.0, label_smoothing=args.label_smoothing,
                   weight_decay=args.weight_decay, warmup=args.warmup,
                   ema_decay=args.ema, tb=args.tb, min_epoch=args.min_epoch)
    dev = next(model.parameters()).device
    crit = T.DetectionLoss().to(dev)
    out = {}
    for tta in (False, True):
        loss, acc, l1, iou = T.evaluate(model, te_loader, crit, dev, tta=tta)
        out["tta" if tta else "plain"] = {"loss": loss, "acc": acc, "box_l1": l1, "iou": iou}
    best = min(hist, key=lambda r: r["val_loss"])
    return {"seed_rep": rep, "n_train": len(tr), "n_inner_val": len(val), "n_test": len(te),
            "epochs_run": len(hist), "best_epoch": best["epoch"], **out}


def average_reps(k, reps):
    """Fold result = mean of the per-seed results; the seeds are kept for the record."""
    out = {"fold": k, "n_test": reps[0]["n_test"], "seeds": reps}
    for mode in ("plain", "tta"):
        out[mode] = {m: float(np.mean([r[mode][m] for r in reps])) for m in reps[0][mode]}
    out["best_epochs"] = [r["best_epoch"] for r in reps]
    return out


def summarize(folds, mode):
    keys = ("acc", "iou", "box_l1")
    return {k: {"mean": float(np.mean([f[mode][k] for f in folds])),
                "std": float(np.std([f[mode][k] for f in folds]))} for k in keys}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--ema", type=float, default=0.0)
    p.add_argument("--crop-prob", type=float, default=0.0)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--deep", action="store_true")
    p.add_argument("--fold-seed", type=int, default=FOLD_SEED)
    p.add_argument("--seeds", type=int, default=3, help="training seeds per fold, averaged")
    p.add_argument("--min-epoch", type=int, default=5)
    p.add_argument("--tb", action="store_true", help="write TensorBoard logs per fold")
    p.add_argument("--ann-path", default=str(HERE / "data/coco/annotations.json"))
    p.add_argument("--img-dir", default=str(HERE / "data/images"))
    p.add_argument("--meta-path", default=str(HERE / "data/raw_meta.json"))
    p.add_argument("--out-dir", default=str(HERE / "docs/cv"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    folds, groups = make_folds(args.ann_path, args.meta_path, args.img_dir, args.fold_seed)
    print(f"{len(groups)} images, {len(set(groups))} groups", flush=True)
    results = []
    for k, split in enumerate(folds):
        t0 = time.time()
        r = average_reps(k, [run_fold(k, split, args, rep) for rep in range(args.seeds)])
        results.append(r)
        plain = {m: round(v, 3) for m, v in r["plain"].items()}
        print(f"fold {k}: plain {plain} tta acc {r['tta']['acc']:.3f} "
              f"best_ep {r['best_epochs']} ({time.time() - t0:.0f}s)", flush=True)
    meta = vars(args) | {"ann_sha256": hashlib.sha256(Path(args.ann_path).read_bytes()).hexdigest()}
    out = {"name": args.name, "args": meta, "folds": results,
           "plain": summarize(results, "plain"), "tta": summarize(results, "tta")}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / f"{args.name}.json", "w") as f:
        json.dump(out, f, indent=1)
    for mode in ("plain", "tta"):
        print(mode, {k: f"{v['mean']:.3f}+-{v['std']:.3f}" for k, v in out[mode].items()})


if __name__ == "__main__":
    main()
