# VisionAudit

Image data pipeline: scrape, auto-label, train a custom CNN and audit the dataset with an LLM.

Classes: robot arm, circuit board, drone.

## Setup

Python 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
export ANTHROPIC_API_KEY=...   # only needed for llm_auditor.py (anthropic backend)
```

## Run

```bash
python collect_and_annotate.py   # scrape, label, write data/coco/annotations.json
python train.py --preset ema_crop --ann-path data/coco/annotations_v5_clean.json
tensorboard --logdir runs/tb
python llm_auditor.py --demo
pytest -q
```

## Files

| File | Purpose |
| --- | --- |
| `collect_and_annotate.py` | Playwright scraper (Wikimedia Commons, CC images) and Grounding DINO + SAM labelling, validated COCO JSON |
| `dataset.py` | COCO parser, OpenCV letterbox, normalize and affine augmentation that moves boxes and masks together |
| `train.py` | Custom CNN, loss, scheduler, early stopping, TensorBoard, `--ensemble N` |
| `cv.py`, `compare_cv.py` | Grouped 5-fold CV, optional seed ensemble, paired per-fold comparison |
| `scrape_more.py`, `annotate_v5.py`, `make_v5_clean.py` | Top-up scrape with dedupe, labelling with review sheets, clean file from review |
| `llm_auditor.py` | RAG over the COCO file and metrics log, with tool calling |
| `tests/` | Geometry, COCO schema, training and auditor tests |

## Pipeline notes

- Images come from Wikimedia Commons; licence and URL per file are in `data/raw_meta.json`.
- Labels: `IDEA-Research/grounding-dino-tiny` for boxes, `facebook/sam-vit-base` for masks.
- `validate_coco()` runs before the file is written.
- Augmentation boxes are the bounding box of the four transformed corners, masks use nearest-neighbour warps. Tests check both against ground truth.

## Dataset

Shipped set (v5): 341 images, 383 annotations (robot_arm 133 images, circuit_board 132, drone 76), file
`docs/data_v5/annotations_v5_clean.json`, sha256 `d28681b67fa4b1f75db649fe54c67eb196aeb29361d1dd000a4ffe2104b4497a`.
Licences and URLs per file: `docs/data_v5/raw_meta.json` (CC BY, CC BY-SA, public domain only; NC/ND are rejected).
The earlier v4clean set (201 images, sha `3ce6fb6d`, `docs/data_v4/`) is kept for the older tables.

The scrape is not deterministic. To use the reference file, copy it to `data/coco/annotations_v5_clean.json` and
point `--ann-path` at it; fetch images with `scrape_more.py`, label with `annotate_v5.py`, apply the review with
`make_v5_clean.py`. The target of 150 images per class was not reached (drone 76).

Labels are Grounding DINO + SAM. Reviewed by eye: all 221 labels added in v5 (74 dropped, reasons in
`docs/data_v5/review.json`), 19 older relabels, and a random 40 of the older labels, which had 7 wrong (17.5%);
those 7 were removed. The rest of the older labels (about 150 images) are not reviewed and likely hold more errors.

## Model

The CNN predicts the class and one normalized box (the largest object) per image.

- Six 3x3 conv layers, stride-2 downsampling, 32 to 256 channels, BatchNorm and ReLU. Receptive field is 47px.
- One hand-written residual block at 64 channels to keep gradients flowing.
- Global average pooling feeds a class head and a sigmoid box head.
- Why 3x3 kernels: stacked small kernels give a larger receptive field with fewer parameters than one large kernel.
- Why six conv layers: small enough to train from scratch on a few hundred images, deep enough for a 47px receptive field.
- Why the residual skip: the identity path at 64 channels keeps gradients flowing and speeds early convergence.
- Loss is CrossEntropy plus weighted SmoothL1.
- Options, off by default: EMA of the weights (`--ema`), zoom-in box-crop augmentation (`--crop-prob`), flip TTA, warmup, wider/deeper variant.

## Training run

Evaluation is grouped 5-fold CV on the v5 clean set (`cv.py`; near-duplicates stay in one fold, plain metric, mean +- std
over folds, deltas paired per fold). Details: `docs/experiments.md`.

| fold_seed 0, v5 | val_acc | val_iou | paired delta vs baseline (acc / iou) |
| --- | --- | --- | --- |
| baseline (3 seeds) | 0.537 +- 0.087 | 0.477 +- 0.012 | - |
| ema_crop (3 seeds) | 0.567 +- 0.078 | 0.485 +- 0.013 | +0.030 +- 0.028 / +0.008 +- 0.004 |

Seed ensemble (5 seeds, ema_crop, same folds): single-model mean acc 0.572, iou 0.486; ensemble acc 0.578, iou 0.489.
Paired: acc +0.007 +- 0.026 (no change), iou +0.003 +- 0.001 (5 of 5 folds but tiny). Not worth 5x cost;
`train.py --ensemble N` is available and off by default.

Verdict: ema_crop's IoU gain reproduces on the bigger set (+0.008, 5 of 5 folds; +0.012 to +0.016 on v4clean), the
accuracy gain is marginal (about 1 std, one fold carries it), the ensemble does not help accuracy. Accuracy sits at
0.54 to 0.58 and is limited by data and label noise. v5 numbers are not comparable with the v4clean numbers
(different images); the older tables are in `docs/experiments.md`.

Shipped recipe (EMA 0.99, crop 0.5, 40 epochs, patience 8). The log in `docs/final/` is a seed-42 run on v4clean (not
rerun on v5):

```bash
python train.py --preset ema_crop --ann-path data/coco/annotations_v4_clean.json --seed 42   # the logged run
python train.py --preset ema_crop --ann-path data/coco/annotations_v5_clean.json --seed 42   # same recipe on v5
```

```
epoch 13 train 0.8780 val 1.1326 acc 0.575 l1 0.1169 iou 0.531 lr 7.94e-04
epoch 21 train 0.7740 val 1.3222 acc 0.550 l1 0.1147 iou 0.521 lr 5.00e-04
early stopping at epoch 21
```

Selected epoch 13 (val acc 0.575, IoU 0.531 on its own 40-image split; not comparable to the CV numbers).

## Auditor

Tools: `find_small_boxes`, `class_balance`, `bbox_area_stats`. Output of the tools on the v4clean annotations:

```
class_balance -> circuit_board 92, robot_arm 80, drone 56, imbalance ratio 1.64
find_small_boxes(max_area_fraction=0.02) -> 16 of 228 boxes (circuit_board 3, robot_arm 3, drone 10)
bbox_area_stats -> median area fraction: circuit_board 0.36, robot_arm 0.14, drone 0.08
```

LLM demo (two tool-call scenarios and a retrieval question), saved to `docs/demo_output.txt`:

```bash
python llm_auditor.py --backend anthropic --demo | tee docs/demo_output.txt
```

It also runs against a self-hosted vLLM server (OpenAI-compatible, `Qwen/Qwen2.5-14B-Instruct`, started with `--enable-auto-tool-choice --tool-call-parser hermes`). `--backend openai` is the default when `LLM_BASE_URL` is set. Config comes from env vars only; the model may emit invalid tool-call JSON, which is sent back as the tool result so it can retry.

```bash
ssh -p <port> -L 20001:localhost:20001 <user>@<host> -N   # keep open in another terminal
export LLM_BASE_URL=http://localhost:20001/v1 LLM_API_KEY=... LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
python llm_auditor.py --backend openai --demo | tee docs/demo_output.txt
```

### End-to-end demo

First lines of the saved run (the full file is `docs/demo_output.txt`):

```
=== Tool scenario 1: small boxes ===
Q: How many bounding boxes cover less than 0.1% of their image, and which classes are they in?
  [retrieved 5 chunks]
    - Image robot_arm_0010.jpg (1200x1600) has 1 boxes: robot_arm x1. Smallest box covers 25.32% of the image, largest 25.32%.
    - Image robot_arm_0018.jpg (1600x1200) has 1 boxes: robot_arm x1. Smallest box covers 1.75% of the image, largest 1.75%.
    - Image circuit_board_0019.jpg (1600x875) has 1 boxes: circuit_board x1. Smallest box covers 58.65% of the image, largest 58.65%.
    - Image robot_arm_0155.jpg (1600x1200) has 1 boxes: robot_arm x1. Smallest box covers 20.25% of the image, largest 20.25%.
    - Image robot_arm_0151.jpg (1600x1200) has 1 boxes: robot_arm x1. Smallest box covers 22.25% of the image, largest 22.25%.
  [tool] find_small_boxes({"max_area_fraction": 0.001, "limit": 100}) -> {"total_matches": 0, "total_boxes": 383, "matches_by_class": {}, "examples": []}
A: There are no bounding boxes that cover less than 0.1% of their respective images out of the total 383 boxes in the dataset. The `find_small_boxes` tool did not return any examples meeting the criteria.

=== Tool scenario 2: class imbalance ===
Q: Is the dataset class-balanced? Give counts and the imbalance ratio, and say what it means for training.
  [retrieved 5 chunks]
    - Epoch 12: train_loss 0.7941368422152834, val_loss 1.2256991624832154, val_acc 0.625, val_box_l1 0.11763055324554443, lr 0.0008247240241650918.
```
