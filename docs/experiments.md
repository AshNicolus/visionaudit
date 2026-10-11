# Experiments

Command: `python train.py --epochs 40 --patience 8 --seed S`, seeds 42 / 1 / 2. The val split depends
on the seed, so seeds see different val images. Numbers are the best-val-loss epoch; mean ± std (population) over
3 seeds. Val is tiny (34 or 38 images, one image is about 3 points of accuracy) and was also used to pick
what to keep, so treat every difference below about 8 points as noise and all of this as tentative.

The seed-42 baseline run reproduces `docs/baseline/summary.json` (epoch 8, val_loss 1.007, acc 0.529, l1 0.156).
Per-seed val_acc is `s42 / s1 / s2`. Raw per-run output is not stored (runs/ is gitignored); only the shipped
run is in `docs/improved/`.

## Original data (171 images, 34 val)

| Change | val_acc per seed | val_loss | val_acc | val_box_l1 | val_iou | Kept |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | .529 / .529 / .618 | 0.923 ± 0.066 | 0.559 ± 0.042 | 0.145 ± 0.012 | 0.450 ± 0.042 | - |
| Class-weighted CE | .559 / .471 / .647 | 0.913 ± 0.071 | 0.559 ± 0.072 | 0.150 ± 0.012 | 0.440 ± 0.048 | no, no change |
| Colour jitter, blur, scale 0.7-1.3 | .559 / .647 / .706 | 0.912 ± 0.083 | 0.637 ± 0.060 | 0.145 ± 0.015 | 0.445 ± 0.056 | tried, dropped below |
| Label smoothing 0.1, weight decay 1e-2 | .559 / .618 / .647 | 0.972 (smoothed, not comparable) | 0.608 ± 0.037 | 0.145 ± 0.010 | 0.452 ± 0.046 | no |
| Warmup 3 ep, box-head dropout | .559 / .647 / .676 | 0.895 ± 0.100 | 0.627 ± 0.050 | 0.150 ± 0.015 | 0.445 ± 0.047 | no, within noise |

The jitter row is +7.8 points of acc over 3 seeds, about 2 standard errors, and loss did not move
(0.923 vs 0.912 is well inside the 0.07 std). Paired deltas vs baseline: +.030 / +.118 / +.088.
Warmup and label smoothing were run on top of the jitter change, so they are compared with the jitter row, not the baseline.
The two regularisation rows are not better than jitter alone.

## More data (190 images, 212 annotations, 38 val)

Re-scraped to 80 per class, re-annotated, COCO regenerated and validated. Drone is still the smallest class
(47 images; Grounding DINO missed many drone photos). Counts: circuit_board 77 images, robot_arm 66, drone 47.

| Change | val_acc per seed | val_loss | val_acc | val_box_l1 | val_iou | Kept |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline recipe | .658 / .763 / .579 | 0.849 ± 0.127 | 0.667 ± 0.075 | 0.130 ± 0.008 | 0.487 ± 0.027 | yes (shipped) |
| Colour jitter, blur, scale 0.7-1.3 | .605 / .579 / .579 | 0.886 ± 0.061 | 0.588 ± 0.012 | 0.133 ± 0.008 | 0.476 ± 0.015 | no |
| Colour jitter, blur, scale 0.8-1.2 | .605 / .684 / .632 | 0.863 ± 0.096 | 0.640 ± 0.033 | 0.132 ± 0.007 | 0.486 ± 0.012 | no |

On the new data the stronger augmentation is worse by 8 points (paired deltas -.053 / -.184 / 0), reversing its
gain on the old data. The two results cancel, so the jitter effect is not established; the branch ships the
baseline augmentation. The extra data is the only change with a plausible effect (acc 0.559 to 0.667,
but the val sets differ, so this is not a clean comparison either).

## Grouped cross-validation

`cv.py`: 5-fold stratified group CV (near-duplicates and same-series images stay together), an inner val
(about 1/6 of train) picks the checkpoint, the held-out fold is scored once. 3 training seeds per fold are averaged,
checkpoint floor at epoch 5, 40 epochs, patience 8. Headline metric is **plain** (no TTA), decided before the
v4clean runs; flip-TTA is reported beside it. Deltas are paired per fold (`compare_cv.py`, which refuses runs that
differ in data, folds, seeds or schedule). Mean ± std over the 5 folds. A mean delta under about 1 std of the fold
deltas is called "no change". Only 5 folds of 38-42 images, so one image is about 2.5 points of accuracy.

Reporting fix: `best_epoch` in the JSONs was the argmin over the whole history, but the weights come from the
checkpoint that respects the epoch-5 floor. `cv.py` now reports the floor-respecting epoch. Metrics are unaffected;
`best_epochs` in the committed JSONs (all runs before this change) can be earlier than the epoch actually used.

### Round 1 (1 seed, no floor, 190 images)

Kept in `docs/cv/round1_1seed/`, not trustworthy: baseline fold 2 was degenerate (best epoch 1, acc 0.237).
Baseline acc 0.463 ± 0.128, iou 0.448. The EMA and wider/deeper gains seen there came from that fold.

### v2 (190 images, annotations_v2, sha 0c2dba57)

Baseline only: acc 0.553 ± 0.075, iou 0.467 ± 0.023, box_l1 0.136 ± 0.011 (TTA acc 0.549). The earlier
0.667 on one 38-image split was optimistic.

### v4 (201 images, includes 13 new drone labels, sha d2acad01; fold_seed 0)

Some v4 labels were not reviewed when this ran, so these numbers are superseded by v4clean.

| Run | acc | iou | box_l1 | Paired delta vs base (acc / iou / box_l1) |
| --- | --- | --- | --- | --- |
| base | 0.535 ± 0.079 | 0.470 ± 0.010 | 0.139 ± 0.005 | - |
| crop | 0.532 ± 0.071 | 0.481 ± 0.015 | 0.136 ± 0.006 | -0.003 ± 0.038 / +0.011 ± 0.008 / +0.004 ± 0.005 |
| ema | 0.560 ± 0.091 | 0.469 ± 0.009 | 0.138 ± 0.005 | +0.025 ± 0.031 / -0.001 ± 0.003 / +0.002 ± 0.002 |
| ema_crop | 0.568 ± 0.075 | 0.484 ± 0.010 | 0.134 ± 0.004 | +0.034 ± 0.018 / +0.014 ± 0.006 / +0.006 ± 0.004 |
| wider+deeper | 0.517 ± 0.067 | 0.468 ± 0.007 | 0.139 ± 0.003 | -0.017 ± 0.037 / -0.001 ± 0.009 / 0.000 ± 0.003 |
| warmup 3, wd 1e-2, ls 0.1 | 0.525 ± 0.063 | 0.467 ± 0.010 | 0.138 ± 0.005 | -0.009 ± 0.039 / -0.002 ± 0.008 / +0.002 ± 0.004 |

(box_l1 deltas are signed so positive is better.) ema_crop was better on all 3 metrics in all 5 folds. EMA alone
is not a win: its accuracy gain is mostly fold 2 (+0.079). Wider/deeper and the schedule changes: no change.

### v4clean (201 images, 228 annotations, sha 3ce6fb6d; review in `docs/data_v4/review.json`)

The 8 wrong v4 images were dropped after review. fold_seed 0, 3 seeds, floor 5. Per-fold deltas are folds 0 to 4.

| Run | acc | iou | box_l1 | TTA acc / iou |
| --- | --- | --- | --- | --- |
| base | 0.508 ± 0.068 | 0.467 ± 0.011 | 0.135 ± 0.005 | 0.506 / 0.469 |
| crop | 0.553 ± 0.063 | 0.477 ± 0.011 | 0.135 ± 0.006 | 0.548 / 0.478 |
| ema | 0.553 ± 0.080 | 0.470 ± 0.014 | 0.133 ± 0.005 | 0.553 / 0.470 |
| ema_crop | 0.563 ± 0.077 | 0.480 ± 0.011 | 0.134 ± 0.005 | 0.558 / 0.481 |

Paired plain deltas vs base:

| Run | acc per fold | acc mean ± sd | iou per fold | iou mean ± sd | box_l1 mean ± sd |
| --- | --- | --- | --- | --- | --- |
| crop | 0 / +.041 / +.025 / +.108 / +.050 | +0.045 ± 0.040 (better, marginal) | +.008 / +.011 / +.010 / +.005 / +.015 | +0.010 ± 0.003 (better) | 0.000 ± 0.004 (no change) |
| ema | -.008 / +.024 / +.067 / +.117 / +.025 | +0.045 ± 0.048 (no change) | 0 / -.001 / +.001 / +.006 / +.008 | +0.003 ± 0.004 (no change) | +0.002 ± 0.002 (no change) |
| ema_crop | +.033 / +.057 / +.017 / +.192 / -.025 | +0.055 ± 0.082 (no change) | +.015 / +.015 / +.005 / +.007 / +.019 | +0.012 ± 0.006 (better) | +0.001 ± 0.003 (no change) |

Fold 3 (+0.19 for ema_crop, +0.11 for the others) carries the accuracy mean, so accuracy is not established.
IoU is the consistent signal: ema_crop and crop are better in 5 of 5 folds. TTA changes nothing (within 0.005).

### v4clean, confirmation with fold_seed 1 (base and ema_crop only)

| Run | acc | iou | box_l1 |
| --- | --- | --- | --- |
| base | 0.569 ± 0.043 | 0.465 ± 0.026 | 0.136 ± 0.008 |
| ema_crop | 0.600 ± 0.041 | 0.481 ± 0.028 | 0.133 ± 0.011 |

Paired plain deltas: acc +.081 / +.017 / +.042 / +.017 / 0.000, mean +0.031 ± 0.032 (no change);
iou +.021 / +.016 / +.019 / +.017 / +.008, mean +0.016 ± 0.005 (better, 5 of 5);
box_l1 +0.003 ± 0.006 (no change). With TTA the accuracy delta is +0.035 ± 0.033, still about 1 std.

### Verdict

ema_crop is the best configuration and it holds up on a second fold split, but the claim is narrow. IoU improves by
about +0.012 to +0.016 (a quarter of the 0.03 fold-to-fold spread) in all 10 folds over the two fold seeds. Accuracy
moves in the same direction (+0.055 and +0.031, 9 of 10 folds not worse) but is under 1 std in both, so it is not
established. Box L1 is no change. Most of the IoU gain comes from the box-crop augmentation (crop alone: +0.010);
EMA adds little by itself. Absolute accuracy is 0.5 to 0.6 and base accuracy shifts 0.06 between fold seeds, which is
larger than any effect here.

Limits: about 200 images and 5 folds; 3 seeds average training noise but not data noise; the options were chosen
on v4 and then re-run on the cleaned set, so v4clean is a replication on the same images minus 8, not new data;
fold_seed 1 shares those images; the clean set is still Grounding DINO + SAM labels, only the 19 changed ones were
reviewed by eye; widedeep and sched were not rerun on v4clean; comparisons across datasets are not valid and
`compare_cv.py` refuses them.

Shipping recipe: `python train.py --preset ema_crop --ann-path data/coco/annotations_v4_clean.json --seed 42`
(EMA 0.99, crop 0.5, 40 epochs, patience 8, floor 5). Train defaults are unchanged. One run on the clean set is in
`docs/final/` (stopped at epoch 21, selected epoch 13: val acc 0.575, iou 0.531 on its own val split, which is not
comparable to the CV numbers).

## Limits

- The 171-image `annotations.json` used for the baseline was overwritten by the re-scrape and was not archived, so the baseline cannot be rebuilt exactly. Its metrics are in `docs/baseline/`.
- The shipped 190-image file has sha256 `0c2dba57ab20aad0965ef8a295f9197bd3acf8f8af6e4e6029e56f53003e4e6e`.
- Full per-epoch logs are kept only for the shipped run (`docs/improved/`). Other rows list per-seed accuracy only.
- New labels are unreviewed Grounding DINO + SAM output.
