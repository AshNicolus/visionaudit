"""Grouping, EMA, LR schedule and fold-leakage tests (no data needed)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import groups  # noqa: E402
import train  # noqa: E402


def test_series_key_strips_numbers_and_brackets():
    a = groups.series_key("https://commons.wikimedia.org/wiki/File:Robot_Arm_(3).jpg")
    b = groups.series_key("https://commons.wikimedia.org/wiki/File:Robot_arm_12.JPG")
    assert a == b == "robot arm"


def test_near_duplicates_share_a_group(tmp_path):
    import json
    import cv2
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "a.jpg"), base)
    cv2.imwrite(str(tmp_path / "b.jpg"), cv2.GaussianBlur(base, (3, 3), 0))
    cv2.imwrite(str(tmp_path / "c.jpg"), rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    meta = [{"class": "x", "file_name": f"{n}.jpg", "page": f"File:{t}.jpg"}
            for n, t in (("a", "Alpha"), ("b", "Beta"), ("c", "Gamma"))]
    (tmp_path / "m.json").write_text(json.dumps(meta))
    g = groups.build_groups(["a.jpg", "b.jpg", "c.jpg"], tmp_path / "m.json", tmp_path)
    assert g[0] == g[1] != g[2]


def test_lr_factor_warms_up_then_decays():
    f = [train.lr_factor(e, 10, 3) for e in range(10)]
    assert f[0] < f[1] < f[2] < f[3] == 1.0 and f[-1] < f[4]


def test_ema_lags_the_model():
    m = torch.nn.Linear(2, 2)
    ema = train.EMA(m, 0.9)
    old = ema.model.weight.detach().clone()
    with torch.no_grad():
        m.weight.add_(1.0)
    ema.update(m)
    delta = (ema.model.weight - old).abs().mean().item()
    assert 0 < delta < 1.0


def test_variants_forward_and_tta_shapes():
    x = torch.randn(2, 3, 64, 64)
    for kw in ({}, {"width": 1.5, "deep": True}):
        model = train.CustomCNN(3, **kw).eval()
        logits, box = train.predict(model, x, tta=True)
        assert logits.shape == (2, 3) and box.shape == (2, 4)


def test_early_stopping_ignores_epochs_before_floor(tmp_path):
    m = torch.nn.Linear(1, 1)
    es = train.EarlyStopping(patience=2, path=tmp_path / "b.pt", min_epoch=3)
    assert not es.step(0.1, m) and not es.step(0.1, m)  # epochs 1-2 never become "best"
    assert not (tmp_path / "b.pt").exists()
    es.step(0.9, m)  # epoch 3 is the first allowed checkpoint
    assert (tmp_path / "b.pt").exists() and es.best == 0.9


def test_compare_ensemble_pairs_ens_against_single():
    import compare_cv
    run = {"folds": [{"plain": {"acc": 0.5, "iou": 0.4, "box_l1": 0.2},
                      "ens": {"acc": 0.6, "iou": 0.5, "box_l1": 0.1}} for _ in range(3)]}
    rows = {m: mean for m, _, mean, _, _ in compare_cv.compare_ensemble(run)}
    assert abs(rows["acc"] - 0.1) < 1e-9 and abs(rows["box_l1"] - 0.1) < 1e-9
