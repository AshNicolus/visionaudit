# Augle AI Intern Assignment

Scrape, auto-label, train and audit an image dataset end to end. Classes: robot arm, circuit board, drone.

## Setup

Python 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
export ANTHROPIC_API_KEY=...   # only needed for llm_auditor.py
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

Dataset: 171 images, 191 annotations (circuit_board 87, robot_arm 66, drone 38), split 137 train / 34 val.

## Model

The CNN predicts the class and one normalized box (the largest object) per image.

- Six 3x3 conv layers, stride-2 downsampling, 32 to 256 channels, BatchNorm and ReLU. Receptive field is 47px.
- One hand-written residual block at 64 channels to keep gradients flowing.
- Global average pooling feeds a class head and a sigmoid box head.
- Loss is CrossEntropy plus weighted SmoothL1.

## Training run

Full log in `docs/train_log.txt`. Early stopping triggered at epoch 16.

```
epoch 1 train 1.2010 val 1.1667 acc 0.382 l1 0.1518 lr 1.00e-03
epoch 8 train 0.9214 val 1.0071 acc 0.529 l1 0.1562 lr 9.26e-04
epoch 15 train 0.8948 val 1.1337 acc 0.588 l1 0.1694 lr 7.27e-04
early stopping at epoch 16
```

With 137 training images the model only reaches about 50 to 59 percent validation accuracy. More data would help most.

## Auditor

Tools: `find_small_boxes`, `class_balance`, `bbox_area_stats`. Output of the tools on the real annotations:

```
class_balance -> circuit_board 87, robot_arm 66, drone 38, imbalance ratio 2.29
find_small_boxes(max_area_fraction=0.02) -> 11 of 191 boxes (circuit_board 3, robot_arm 3, drone 5)
bbox_area_stats -> median area fraction: circuit_board 0.33, robot_arm 0.14, drone 0.06
```

LLM demo (two tool-call scenarios and a retrieval question), saved to `docs/demo_output.txt`:

```bash
python llm_auditor.py --demo | tee docs/demo_output.txt
```
