"""Custom CNN, custom loss and a manual training loop with TensorBoard logging."""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dataset import MEAN, STD

HERE = Path(__file__).resolve().parent


def conv_bn(c_in, c_out, stride=1):
    """3x3 conv + BatchNorm + ReLU."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class ResidualBlock(nn.Module):
    """Two 3x3 convs with a hand-written identity skip connection."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = conv_bn(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + x)


class CustomCNN(nn.Module):
    """Six-conv backbone with a class head and a normalized box head."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(3, 32, stride=2),
            conv_bn(32, 64, stride=2),
            ResidualBlock(64),
            conv_bn(64, 128, stride=2),
            conv_bn(128, 256, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))
        self.box_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Linear(64, 4))

    def forward(self, x):
        z = self.pool(self.features(x)).flatten(1)
        return self.cls_head(z), torch.sigmoid(self.box_head(z))


class DetectionLoss(nn.Module):
    """Weighted CrossEntropy on classes plus SmoothL1 on boxes."""

    def __init__(self, cls_weight=1.0, box_weight=5.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.ce = nn.CrossEntropyLoss()
        self.l1 = nn.SmoothL1Loss()

    def forward(self, logits, boxes, labels, target_boxes):
        cls_loss = self.ce(logits, labels)
        return self.cls_weight * cls_loss + self.box_weight * self.l1(boxes, target_boxes)


class EarlyStopping:
    """Stops when val loss has not improved for `patience` epochs; saves the best model."""

    def __init__(self, patience=5, min_delta=0.0, path="runs/best.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = Path(path)
        self.best = float("inf")
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, val_loss, model):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad_epochs = 0
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.path)
        else:
            self.bad_epochs += 1
            self.should_stop = self.bad_epochs >= self.patience
        return self.should_stop


def set_seed(seed):
    """Seed python, numpy and torch; make cudnn deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Pick cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over the training data; returns (mean loss, first batch used)."""
    model.train()
    total, count, first = 0.0, 0, None
    for batch in loader:
        if first is None:
            first = batch
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        boxes = batch["box"].to(device)
        optimizer.zero_grad()
        logits, pred = model(images)
        loss = criterion(logits, pred, labels, boxes)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
        count += images.size(0)
    return total / max(count, 1), first


def evaluate(model, loader, criterion, device):
    """Returns (val_loss, accuracy, mean box L1) without gradients."""
    model.eval()
    loss_sum, correct, l1_sum, count = 0.0, 0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            boxes = batch["box"].to(device)
            logits, pred = model(images)
            n = images.size(0)
            loss_sum += criterion(logits, pred, labels, boxes).item() * n
            correct += (logits.argmax(1) == labels).sum().item()
            l1_sum += (pred - boxes).abs().mean(1).sum().item()
            count += n
    count = max(count, 1)
    return loss_sum / count, correct / count, l1_sum / count


def draw_box_sample(image, box):
    """De-normalize a CHW tensor and draw a [cx, cy, w, h] box; returns CHW uint8 RGB."""
    img = image.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * np.array(STD) + np.array(MEAN), 0, 1)
    img = np.ascontiguousarray((img * 255).astype(np.uint8))
    h, w = img.shape[:2]
    cx, cy, bw, bh = box.detach().cpu().tolist()
    p1 = (int((cx - bw / 2) * w), int((cy - bh / 2) * h))
    p2 = (int((cx + bw / 2) * w), int((cy + bh / 2) * h))
    cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
    return img.transpose(2, 0, 1)


def log_samples(writer, batch, epoch, n=4):
    """Write n augmented samples from a batch the epoch already used."""
    for i in range(min(n, batch["image"].size(0))):
        writer.add_image(f"samples/{i}", draw_box_sample(batch["image"][i], batch["box"][i]), epoch)


def append_metrics(path, record):
    """Append one JSON line to the metrics file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def train(model, train_loader, val_loader, epochs=30, lr=1e-3, patience=5, seed=42,
          run_dir="runs", device=None, box_weight=5.0):
    """Full training run; returns the list of per-epoch metric dicts."""
    set_seed(seed)
    device = device or get_device()
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    model.to(device)
    criterion = DetectionLoss(box_weight=box_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    stopper = EarlyStopping(patience, path=run_dir / "best.pt")
    writer = SummaryWriter(str(run_dir / "tb"))
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, first_batch = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_l1 = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                  "val_acc": val_acc, "val_box_l1": val_l1, "lr": current_lr}
        history.append(record)
        append_metrics(metrics_path, record)
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        writer.add_scalar("val/box_l1", val_l1, epoch)
        writer.add_scalar("lr", current_lr, epoch)
        log_samples(writer, first_batch, epoch)
        print(f"epoch {epoch} train {train_loss:.4f} val {val_loss:.4f} acc {val_acc:.3f} "
              f"l1 {val_l1:.4f} lr {current_lr:.2e}")
        if stopper.step(val_loss, model):
            print(f"early stopping at epoch {epoch}")
            break
    writer.close()
    if stopper.path.exists():  # return the best weights, not the last
        model.load_state_dict(torch.load(stopper.path, map_location=device))
    return history


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ann-path", default=str(HERE / "data/coco/annotations.json"))
    p.add_argument("--img-dir", default=str(HERE / "data/images"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from dataset import get_dataloaders

    train_loader, val_loader, num_classes = get_dataloaders(
        ann_path=args.ann_path, img_dir=args.img_dir,
        img_size=args.img_size, batch_size=args.batch_size, seed=args.seed)
    set_seed(args.seed)
    model = CustomCNN(num_classes)
    train(model, train_loader, val_loader, args.epochs, args.lr, args.patience, args.seed)


if __name__ == "__main__":
    main()
