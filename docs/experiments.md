# Experiments

Each row: best-val-loss epoch per seed (42, 1, 2), mean ± std over the three seeds. The val split
changes with the seed, so only compare rows against each other. `python train.py --epochs 40 --patience 8`.

| Change | val_loss | val_acc | val_box_l1 | val_iou | Kept |
| --- | --- | --- | --- | --- | --- |
| Baseline | 0.923 ± 0.066 | 0.559 ± 0.042 | 0.145 ± 0.012 | 0.450 ± 0.042 | - |
| Class-weighted CE (inverse frequency) | 0.913 ± 0.071 | 0.559 ± 0.072 | 0.150 ± 0.012 | 0.440 ± 0.048 | no, same acc, noisier |
| Colour jitter + blur + scale 0.7-1.3 | 0.912 ± 0.083 | 0.637 ± 0.060 | 0.145 ± 0.015 | 0.445 ± 0.056 | yes, acc +8 pts |
| Label smoothing 0.1 + weight decay 1e-2 | 0.972 ± 0.058 (smoothed, not comparable) | 0.608 ± 0.037 | 0.145 ± 0.010 | 0.452 ± 0.046 | no, acc below aug-only |
| Warmup 3 epochs + dropout before box head | 0.895 ± 0.100 | 0.627 ± 0.050 | 0.150 ± 0.015 | 0.445 ± 0.047 | no, within noise |
