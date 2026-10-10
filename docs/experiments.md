# Experiments

Each row: best-val-loss epoch per seed (42, 1, 2), mean ± std over the three seeds. The val split
changes with the seed, so only compare rows against each other. `python train.py --epochs 40 --patience 8`.

| Change | val_loss | val_acc | val_box_l1 | val_iou | Kept |
| --- | --- | --- | --- | --- | --- |
| Baseline | 0.923 ± 0.066 | 0.559 ± 0.042 | 0.145 ± 0.012 | 0.450 ± 0.042 | - |
| Class-weighted CE (inverse frequency) | 0.913 ± 0.071 | 0.559 ± 0.072 | 0.150 ± 0.012 | 0.440 ± 0.048 | no, same acc, noisier |
