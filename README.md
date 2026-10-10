# Augle AI Intern Assignment

Scrape, auto-label, train and audit an image dataset end to end. Classes: robot arm, circuit board, drone.

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
python train.py --epochs 40 --patience 8
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

Training baseline (v2, seeds 42/1/2, rebuildable with data hashes and commands): `docs/baseline_v2/`.

Dataset: 190 images, 212 annotations (circuit_board 92, robot_arm 69, drone 51), split 152 train / 38 val. (The first version had 171 images; the scrape was extended to 80 per class.)

## Model

The CNN predicts the class and one normalized box (the largest object) per image.

- Six 3x3 conv layers, stride-2 downsampling, 32 to 256 channels, BatchNorm and ReLU. Receptive field is 47px.
- One hand-written residual block at 64 channels to keep gradients flowing.
- Global average pooling feeds a class head and a sigmoid box head.
- Loss is CrossEntropy plus weighted SmoothL1.

## Training run

Full log in `docs/improved/train_log.txt` (metrics in `docs/improved/metrics.jsonl`, baseline in `docs/baseline/`).
`python train.py --epochs 40 --patience 8`, seed 42. Early stopping triggered at epoch 25, best epoch 17.
`metrics.jsonl` now also has `val_iou`.

```
epoch 1 train 1.1752 val 1.1687 acc 0.316 l1 0.1409 iou 0.452 lr 1.00e-03
epoch 17 train 0.9070 val 0.8503 acc 0.658 l1 0.1218 iou 0.519 lr 6.55e-04
early stopping at epoch 25
```

| | data | val_loss | val_acc | val_box_l1 | val_iou |
| --- | --- | --- | --- | --- | --- |
| Baseline, seed 42 | 171 images, 34 val | 1.007 | 0.529 | 0.156 | 0.392 |
| Final, seed 42 | 190 images, 38 val | 0.850 | 0.658 | 0.122 | 0.519 |
| Baseline, mean of seeds 42/1/2 | 171 images | 0.923 +- 0.066 | 0.559 +- 0.042 | 0.145 +- 0.012 | 0.450 +- 0.042 |
| Final, mean of seeds 42/1/2 | 190 images | 0.849 +- 0.127 | 0.667 +- 0.075 | 0.130 +- 0.008 | 0.487 +- 0.027 |

The val sets differ between rows (the split depends on the data size and the seed), and val has only 34 to 38 images, so
these are tentative. The only change kept is more data. Class-weighted loss, stronger augmentation, label smoothing, weight
decay, warmup and head dropout did not give a result that held up; details and per-seed numbers are in `docs/experiments.md`.

## Auditor

Tools: `find_small_boxes`, `class_balance`, `bbox_area_stats`. Output of the tools on the real annotations:

```
class_balance -> circuit_board 92, robot_arm 69, drone 51, imbalance ratio 1.8
find_small_boxes(max_area_fraction=0.02) -> 13 of 212 boxes (circuit_board 3, robot_arm 3, drone 7)
bbox_area_stats -> median area fraction: circuit_board 0.36, robot_arm 0.14, drone 0.09
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
