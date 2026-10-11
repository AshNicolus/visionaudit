# Checkpoint, 2026-10-11

## Branch `improve/data-ensemble` (latest, on top of main)

- v5 dataset: 341 images / 383 annotations, sha `d28681b6`, all 221 new labels reviewed by eye (74 dropped), 7 of 40 sampled old labels wrong and removed; the remaining old labels are unreviewed. Files in `docs/data_v5/`.
- CV on v5 (fold_seed 0, 3 seeds): base acc 0.537 / iou 0.477, ema_crop 0.567 / 0.485 (paired +0.030 +- 0.028, +0.008 +- 0.004). Results in `docs/cv/v5/`.
- Ensemble of 5 (`cv.py --ensemble`, `train.py --ensemble N`): acc +0.007 +- 0.026, iou +0.003 +- 0.001 vs the single-model mean; not worth it.
- Not reached: 150 images per class (drone 76). Still open: review the rest of the old labels, rerun `docs/final/` on v5.
- Not merged; push only.

Older notes below, from `improve/model-v2` (merged).

## On main (done)

Pipeline, auditor with the vLLM backend, demo output, README, repo renamed to VisionAudit (PRs 1 to 8).

## On branch `docs/baseline-v2`

Baseline v2 on the 190-image set: 3 seeds, logs, archived annotations (`docs/baseline_v2/`). Not merged yet.

## On branch `improve/model-v2`

- Grouped 5-fold CV (`cv.py`, `groups.py`): stratified, near-duplicates and same-series images kept together, inner val for early stopping.
- Options added to the model/training code: EMA, warmup, box-crop augmentation, flip TTA, wider/deeper variant.
- 3 seeds per fold and a checkpoint floor (min epoch 5) as defaults; `compare_cv.py` refuses mismatched runs.
- Round 1 (1 seed, no floor) is kept in `docs/cv/round1_1seed/` and is not trustworthy: baseline fold 2 was degenerate (best epoch 1, acc 0.237).

## What to trust

- Grouped-CV baseline sits near acc 0.46. The earlier 0.667 came from one 38-image split and was optimistic.
- Only crop's IoU gain (+0.014, sd 0.007) passed the 1-sd bar in round 1, and it is small.
- EMA and wider/deeper gains were fold-2 artefacts. Do not claim them.

## CV results (done)

v4 matrix, v4clean (4 configs) and a fold_seed 1 confirmation are in `docs/cv/` and written up in `docs/experiments.md`.
ema_crop is best: IoU +0.012 to +0.016 in 10 of 10 folds, accuracy not established. `train.py --preset ema_crop` reproduces it.
Final run logs are in `docs/final/`.

## Uncommitted, left out on purpose

- Nothing. The v4 labels were reviewed (`docs/data_v4/review.json`: 19 checked, 8 dropped, clean file of 201 images). The stray `docs/cv/ema_crop.json` of unknown origin was moved out of the repo.

## Open items from the monitor

1. Done: plain is the headline, TTA reported beside it.
2. Done: fold_seed 1 confirmation (IoU confirmed, accuracy not).
3. Done: `--preset ema_crop`; TensorBoard scalars and per-epoch samples are written by default.
Still open:
- Only 19 labels were reviewed by eye; the other Grounding DINO + SAM labels are not.
- widedeep and sched were not rerun on v4clean.
- `best_epochs` in the JSONs made before the cv.py fix can be earlier than the epoch used.
- Rebuild folds and rerun the baseline on any new image set; keep dataset tables separate.
- Rebase onto main (README was renamed in PR 8).

## Resume

```bash
cd ~/augle-improve && git status
./.venv/bin/python -u cv.py --name base --out-dir docs/cv/v4clean --ann-path data/coco/annotations_v4_clean.json
./.venv/bin/python compare_cv.py docs/cv/v4clean/base.json docs/cv/v4clean/ema_crop.json
```

Also rotate the vLLM API key that was pasted into the terminal.
