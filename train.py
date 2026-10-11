"""Custom CNN, custom loss and a manual training loop with TensorBoard logging."""
import argparse
import copy
import json
import math
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
    """Six-conv backbone with a class head and a normalized box head.

    width scales every channel count; deep adds a second residual block at 128 channels.
    """

    def __init__(self, num_classes, width=1.0, deep=False):
        super().__init__()
        c = [round(ch * width) for ch in (32, 64, 128, 256)]
        layers = [conv_bn(3, c[0], stride=2), conv_bn(c[0], c[1], stride=2),
                  ResidualBlock(c[1]), conv_bn(c[1], c[2], stride=2)]
        if deep:
            layers.append(ResidualBlock(c[2]))
        layers.append(conv_bn(c[2], c[3], stride=2))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(c[3], num_classes))
        self.box_head = nn.Sequential(nn.Linear(c[3], 64), nn.ReLU(inplace=True), nn.Linear(64, 4))

    def forward(self, x):
        z = self.pool(self.features(x)).flatten(1)
        return self.cls_head(z), torch.sigmoid(self.box_head(z))


class DetectionLoss(nn.Module):
    """Weighted CrossEntropy on classes plus SmoothL1 on boxes."""

    def __init__(self, cls_weight=1.0, box_weight=5.0, label_smoothing=0.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.l1 = nn.SmoothL1Loss()

    def forward(self, logits, boxes, labels, target_boxes):
        cls_loss = self.ce(logits, labels)
        return self.cls_weight * cls_loss + self.box_weight * self.l1(boxes, target_boxes)


class EMA:
    """Exponential moving average of the weights (and a copy of the BN buffers)."""

    def __init__(self, model, decay=0.99):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))  # short ramp
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1 - d)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def lr_factor(epoch, epochs, warmup):
    """LR multiplier for a 0-based epoch: linear warmup, then cosine to zero."""
    if epoch < warmup:
        return (epoch + 1) / (warmup + 1)
    return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(epochs - warmup, 1)))


class EarlyStopping:
    """Stops when val loss has not improved for `patience` epochs; saves the best model."""

    def __init__(self, patience=5, min_delta=0.0, path="runs/best.pt", min_epoch=1):
        self.min_epoch = min_epoch  # no checkpoint or patience counting before this epoch
        self.epoch = 0
        self.patience = patience
        self.min_delta = min_delta
        self.path = Path(path)
        self.best = float("inf")
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, val_loss, model):
        self.epoch += 1
        if self.epoch < self.min_epoch:
            return False
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


def train_one_epoch(model, loader, criterion, optimizer, device, ema=None):
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
        if ema is not None:
            ema.update(model)
        total += loss.item() * images.size(0)
        count += images.size(0)
    return total / max(count, 1), first


def cxcywh_iou(a, b):
    """Per-row IoU of two (N,4) cxcywh tensors."""
    a1, a2 = a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2
    b1, b2 = b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2
    wh = (torch.minimum(a2, b2) - torch.maximum(a1, b1)).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return inter / union.clamp(min=1e-9)


def predict(model, images, tta=False):
    """(logits, boxes); with tta, average class probabilities and boxes with a flipped view.

    Returned values stay logits-shaped: the averaged probabilities go through log so
    argmax and cross entropy behave.
    """
    logits, box = model(images)
    if not tta:
        return logits, box
    f_logits, f_box = model(torch.flip(images, dims=[3]))
    f_box = torch.stack([1 - f_box[:, 0], f_box[:, 1], f_box[:, 2], f_box[:, 3]], 1)
    prob = (logits.softmax(1) + f_logits.softmax(1)) / 2
    return prob.clamp_min(1e-9).log(), (box + f_box) / 2


def evaluate(model, loader, criterion, device, tta=False):
    """Returns (val_loss, accuracy, mean box L1, mean box IoU) without gradients."""
    model.eval()
    loss_sum, correct, l1_sum, iou_sum, count = 0.0, 0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            boxes = batch["box"].to(device)
            logits, pred = predict(model, images, tta)
            n = images.size(0)
            loss_sum += criterion(logits, pred, labels, boxes).item() * n
            correct += (logits.argmax(1) == labels).sum().item()
            l1_sum += (pred - boxes).abs().mean(1).sum().item()
            iou_sum += cxcywh_iou(pred, boxes).sum().item()
            count += n
    count = max(count, 1)
    return loss_sum / count, correct / count, l1_sum / count, iou_sum / count


def evaluate_ensemble(models, loader, criterion, device):
    """Same metrics as evaluate for a seed ensemble: class probabilities and boxes are averaged.

    Returns (loss, accuracy, mean box L1, mean box IoU); loss uses the log of the mean
    probabilities, like predict(tta=True).
    """
    for m in models:
        m.eval()
    loss_sum, correct, l1_sum, iou_sum, count = 0.0, 0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            boxes = batch["box"].to(device)
            outs = [m(images) for m in models]
            prob = torch.stack([o[0].softmax(1) for o in outs]).mean(0)
            pred = torch.stack([o[1] for o in outs]).mean(0)
            logits = prob.clamp_min(1e-9).log()
            n = images.size(0)
            loss_sum += criterion(logits, pred, labels, boxes).item() * n
            correct += (logits.argmax(1) == labels).sum().item()
            l1_sum += (pred - boxes).abs().mean(1).sum().item()
            iou_sum += cxcywh_iou(pred, boxes).sum().item()
            count += n
    count = max(count, 1)
    return loss_sum / count, correct / count, l1_sum / count, iou_sum / count


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
          run_dir="runs", device=None, box_weight=5.0, label_smoothing=0.0, weight_decay=1e-4,
          warmup=0, ema_decay=0.0, tb=True, min_epoch=1):
    """Full training run; returns the list of per-epoch metric dicts."""
    set_seed(seed)
    device = device or get_device()
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    model.to(device)
    criterion = DetectionLoss(box_weight=box_weight, label_smoothing=label_smoothing).to(device)
    eval_criterion = DetectionLoss(box_weight=box_weight).to(device)  # val loss is never smoothed
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if warmup:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: lr_factor(e, epochs, warmup))
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema = EMA(model, ema_decay) if ema_decay else None
    stopper = EarlyStopping(patience, path=run_dir / "best.pt", min_epoch=min_epoch)
    writer = SummaryWriter(str(run_dir / "tb")) if tb else None
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, first_batch = train_one_epoch(
            model, train_loader, criterion, optimizer, device, ema)
        scored = ema.model if ema else model
        val_loss, val_acc, val_l1, val_iou = evaluate(scored, val_loader, eval_criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                  "val_acc": val_acc, "val_box_l1": val_l1, "val_iou": val_iou, "lr": current_lr}
        history.append(record)
        append_metrics(metrics_path, record)
        if writer:
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalar("val/acc", val_acc, epoch)
            writer.add_scalar("val/box_l1", val_l1, epoch)
            writer.add_scalar("val/iou", val_iou, epoch)
            writer.add_scalar("lr", current_lr, epoch)
            log_samples(writer, first_batch, epoch)
        print(f"epoch {epoch} train {train_loss:.4f} val {val_loss:.4f} acc {val_acc:.3f} "
              f"l1 {val_l1:.4f} iou {val_iou:.3f} lr {current_lr:.2e}")
        if stopper.step(val_loss, scored):
            print(f"early stopping at epoch {epoch}")
            break
    if writer:
        writer.close()
    if stopper.path.exists():  # return the best weights, not the last
        model.load_state_dict(torch.load(stopper.path, map_location=device))
    return history


# the CV winner (docs/experiments.md): EMA plus box-crop, 40 epochs, patience 8, floor 5
EMA_CROP = {"ema": 0.99, "crop_prob": 0.5, "epochs": 40, "patience": 8, "min_epoch": 5}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["ema_crop"],
                   help="CV-winning recipe; explicit flags still override it")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay, 0 = off")
    p.add_argument("--crop-prob", type=float, default=0.0, help="zoom-in crop around the box")
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--deep", action="store_true")
    p.add_argument("--ensemble", type=int, default=1,
                   help="train N seeds, keep each best checkpoint, report the ensemble")
    p.add_argument("--min-epoch", type=int, default=1, help="no best-checkpoint before this epoch")
    p.add_argument("--run-dir", default=str(HERE / "runs"))
    p.add_argument("--ann-path", default=str(HERE / "data/coco/annotations.json"))
    p.add_argument("--img-dir", default=str(HERE / "data/images"))
    pre, _ = p.parse_known_args(argv)
    if pre.preset == "ema_crop":
        p.set_defaults(**EMA_CROP)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from dataset import get_dataloaders

    train_loader, val_loader, num_classes = get_dataloaders(
        ann_path=args.ann_path, img_dir=args.img_dir,
        img_size=args.img_size, batch_size=args.batch_size, seed=args.seed,
        crop_prob=args.crop_prob)
    models = []
    for i in range(args.ensemble):
        seed = args.seed + i
        set_seed(seed)
        model = CustomCNN(num_classes, width=args.width, deep=args.deep)
        run_dir = args.run_dir if args.ensemble == 1 else f"{args.run_dir}/seed{seed}"
        train(model, train_loader, val_loader, args.epochs, args.lr, args.patience, seed,
              run_dir=run_dir, label_smoothing=args.label_smoothing,
              weight_decay=args.weight_decay, warmup=args.warmup, ema_decay=args.ema,
              min_epoch=args.min_epoch)
        models.append(model)
    if args.ensemble > 1:
        dev = next(models[0].parameters()).device
        loss, acc, l1, iou = evaluate_ensemble(models, val_loader, DetectionLoss().to(dev), dev)
        print(f"ensemble of {len(models)}: val {loss:.4f} acc {acc:.3f} l1 {l1:.4f} iou {iou:.3f}")


if __name__ == "__main__":
    main()
