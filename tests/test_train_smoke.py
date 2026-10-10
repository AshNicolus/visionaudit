"""Smoke tests for train.py using a fake loader (CPU, no data needed)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: E402


def fake_loader(n_batches=3, bs=4, size=64, classes=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [{
        "image": torch.randn(bs, 3, size, size, generator=g),
        "label": torch.randint(0, classes, (bs,), generator=g),
        "box": torch.rand(bs, 4, generator=g) * 0.5 + 0.25,
        "mask": torch.zeros(bs, 1, size, size),
    } for _ in range(n_batches)]


def test_forward_shapes():
    logits, box = train.CustomCNN(3)(fake_loader()[0]["image"])
    assert logits.shape == (4, 3) and box.shape == (4, 4)
    assert box.min() >= 0 and box.max() <= 1


def test_training_changes_params_and_logs(tmp_path):
    model = train.CustomCNN(3)
    before = [p.detach().clone() for p in model.parameters()]
    history = train.train(model, fake_loader(), fake_loader(seed=1), epochs=2, lr=1e-2,
                          patience=5, run_dir=tmp_path, device=torch.device("cpu"))
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(h["train_loss"])) for h in history)
    assert any(not torch.equal(b, a) for b, a in zip(before, model.parameters()))
    assert (tmp_path / "best.pt").exists()
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert {"epoch", "train_loss", "val_loss", "val_acc", "val_box_l1", "lr"} <= set(eval(lines[0]))


def test_early_stopping_triggers(tmp_path):
    model = torch.nn.Linear(1, 1)
    stopper = train.EarlyStopping(patience=2, path=tmp_path / "b.pt")
    assert not stopper.step(1.0, model)
    assert not stopper.step(1.5, model)
    assert stopper.step(1.5, model)
    assert stopper.best == 1.0


def test_early_stopping_in_loop(tmp_path):
    history = train.train(train.CustomCNN(3), fake_loader(), fake_loader(seed=1), epochs=50,
                          lr=0.0, patience=2, run_dir=tmp_path, device=torch.device("cpu"))
    assert len(history) < 50
