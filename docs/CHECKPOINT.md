# Checkpoint, 2026-10-11

Where things stand on `improve/model-v2`. Resume from here.

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

## Interrupted

The round-2 matrix (base, crop, ema, ema_crop; 3 seeds, floor 5) was stopped part way and has no results. Rerun it.

## Uncommitted, left out on purpose

- `docs/data_v4/` (new drone labels, unreviewed), `docs/relabel_log_new_drones.json`, edits to `relabel_missed.py`: review the 13 new drone labels on a contact sheet before using them.
- `docs/cv/ema_crop.json`: timestamp predates the matrix start, so its origin is unclear. Check or delete.

## Open items from the monitor

1. Pick plain vs TTA once and report both.
2. Confirm the winner with a second FOLD_SEED.
3. Expose the winning options in the `train.py` CLI, and make the shipped run write TensorBoard scalars and per-epoch augmented samples.
4. Rebuild folds and rerun the baseline on any new image set; keep v2 and v3 tables separate.
5. Rebase onto main (README was renamed in PR 8).

## Resume

```bash
cd ~/augle-improve && git status
./.venv/bin/python -u cv.py --name base --out-dir docs/cv   # then crop, ema, ema_crop
./.venv/bin/python compare_cv.py
```

Also rotate the vLLM API key that was pasted into the terminal.
