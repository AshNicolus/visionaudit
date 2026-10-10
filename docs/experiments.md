# Experiments

Each row: best-val-loss epoch per seed (42, 1, 2), mean ± std over the three seeds. The val split
changes with the seed, so only compare rows against each other. `python train.py --epochs 40 --patience 8`.

| Change | val_loss | val_acc | val_box_l1 | val_iou | Kept |
| --- | --- | --- | --- | --- | --- |
