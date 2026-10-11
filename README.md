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
python train.py --preset ema_crop --ann-path data/coco/annotations_v4_clean.json
tensorboard --logdir runs/tb
python llm_auditor.py --demo
pytest -q
```

## Files

| File | Purpose |
| --- | --- |
| `collect_and_annotate.py` | Playwright scraper (Wikimedia Commons, CC images) and Grounding DINO + SAM labelling, validated COCO JSON |
| `dataset.py` | COCO parser, OpenCV letterbox, normalize and affine augmentation that moves boxes and masks together |
| `train.py` | Custom CNN, loss, scheduler, early stopping, TensorBoard |
| `llm_auditor.py` | RAG over the COCO file and metrics log, with tool calling |
| `tests/` | Geometry, COCO schema, training and auditor tests |

## Pipeline notes

- Images come from Wikimedia Commons; licence and URL per file are in `data/raw_meta.json`.
- Labels: `IDEA-Research/grounding-dino-tiny` for boxes, `facebook/sam-vit-base` for masks.
- `validate_coco()` runs before the file is written.
- Augmentation boxes are the bounding box of the four transformed corners, masks use nearest-neighbour warps. Tests check both against ground truth.

## Dataset

Shipped set: 201 images, 228 annotations (circuit_board 92, robot_arm 80, drone 56), file
`docs/data_v4/annotations_v4_clean.json`, sha256 `3ce6fb6d4fba8bf04edcc35eca52b2725971be7a4d7390a7c47ebbe6b007de01`.
`train.py` with seed 42 splits it 161 train / 40 val.

Rebuild, in order (the scrape is not deterministic, so images can differ; the committed JSON is the reference):

```bash
python collect_and_annotate.py                      # data/coco/annotations.json (v2: 190 images)
python scrape_more.py --cls drone                   # more drone images
python relabel_missed.py --src data/coco/annotations_v2.json --out data/coco/annotations_v3.json   # extra prompts for misses
# v4 = docs/data_v4/annotations.json (sha d2acad01...); clean = v4 minus the 8 images dropped in docs/data_v4/review.json
cp docs/data_v4/annotations_v4_clean.json data/coco/
sha256sum data/coco/annotations_v4_clean.json
```

I reviewed 19 relabelled images by eye on contact sheets and dropped 8 of them (`docs/data_v4/review.json`).
The remaining Grounding DINO + SAM labels have not been reviewed.

## Model

The CNN predicts the class and one normalized box (the largest object) per image.

- Six 3x3 conv layers, stride-2 downsampling, 32 to 256 channels, BatchNorm and ReLU. Receptive field is 47px.
- One hand-written residual block at 64 channels to keep gradients flowing.
- Global average pooling feeds a class head and a sigmoid box head.
- Loss is CrossEntropy plus weighted SmoothL1.
- Options, off by default: EMA of the weights (`--ema`), zoom-in box-crop augmentation (`--crop-prob`), flip TTA, warmup, wider/deeper variant.

## Training run

Evaluation is grouped 5-fold CV on the clean set (`cv.py`; near-duplicates stay in one fold, 3 training seeds per fold,
plain metric, mean +- std over folds, deltas paired per fold). Details, other configs and limits: `docs/experiments.md`.

| fold_seed 0 | val_acc | val_iou | paired delta acc / iou vs baseline |
| --- | --- | --- | --- |
| baseline | 0.508 +- 0.068 | 0.467 +- 0.011 | - |
| crop | 0.553 +- 0.063 | 0.477 +- 0.011 | +0.045 +- 0.040 / +0.010 +- 0.003 |
| ema_crop | 0.563 +- 0.077 | 0.480 +- 0.011 | +0.055 +- 0.082 / +0.012 +- 0.006 |

Confirmation with fold_seed 1 (baseline vs ema_crop): acc 0.569 vs 0.600 (delta +0.031 +- 0.032), IoU 0.465 vs 0.481
(delta +0.016 +- 0.005).

Verdict: the IoU gain is established (better in 10 of 10 folds over both fold splits), but it is small, about +0.012 to
+0.016, and mostly comes from box-crop. The accuracy gain is not established: it is under 1 std in both splits and one
fold (+0.19) carries the mean. Absolute accuracy is only 0.5 to 0.6. The earlier single-split numbers (0.667 etc.) were
optimistic; they are kept only in `docs/experiments.md`.

Shipped recipe (EMA 0.99, crop 0.5, 40 epochs, patience 8), seed 42, log in `docs/final/`:

```bash
python train.py --preset ema_crop --ann-path data/coco/annotations_v4_clean.json --seed 42
```

```
epoch 13 train 0.8780 val 1.1326 acc 0.575 l1 0.1169 iou 0.531 lr 7.94e-04
epoch 21 train 0.7740 val 1.3222 acc 0.550 l1 0.1147 iou 0.521 lr 5.00e-04
early stopping at epoch 21
```

Selected epoch 13 (val acc 0.575, IoU 0.531 on its own 40-image split; not comparable to the CV numbers).

## Auditor

Tools: `find_small_boxes`, `class_balance`, `bbox_area_stats`. Output of the tools on the real annotations:

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
ssh -p 31319 -L 20001:localhost:20001 ubuntu@117.18.102.47 -N   # keep open in another terminal
export LLM_BASE_URL=http://localhost:20001/v1 LLM_API_KEY=... LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
python llm_auditor.py --backend openai --demo | tee docs/demo_output.txt
```
