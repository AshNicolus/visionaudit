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
