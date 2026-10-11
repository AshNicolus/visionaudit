"""Label the images added after v4clean with Grounding DINO + SAM and draw review sheets.

Prompt ladder: the default prompt at 0.35/0.25, then the prompts that worked best in earlier
relabel runs at 0.25/0.20. Nothing lower. The result is raw: every new label is reviewed by
eye before it goes into the clean file.
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

import collect_and_annotate as C

LADDER = {
    "drone": [("a drone.", 0.35, 0.25), ("a quadcopter.", 0.25, 0.20),
              ("an unmanned aerial vehicle.", 0.25, 0.20)],
    "robot_arm": [("a robot arm.", 0.35, 0.25), ("a robotic arm.", 0.25, 0.20)],
    "circuit_board": [("a circuit board.", 0.35, 0.25),
                      ("a printed circuit board.", 0.25, 0.20)],
}


def sheet(items, path, cols=4, cell=360):
    """Contact sheet of (name, image, annotations) with boxes and mask outlines."""
    tiles = []
    for name, img, anns in items:
        im = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for a in anns:
            x, y, w, h = map(int, a["bbox"])
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), max(2, im.shape[1] // 250))
            for p in a["segmentation"]:
                cv2.polylines(im, [np.array(p, np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 2)
        s = cell / max(im.shape[:2])
        im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
        canvas = np.zeros((cell + 24, cell, 3), np.uint8)
        canvas[24:24 + im.shape[0], :im.shape[1]] = im
        cv2.putText(canvas, name.replace(".jpg", ""), (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        tiles.append(canvas)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data/coco/annotations_v4_clean.json")
    ap.add_argument("--before", default="docs/data_v5/raw_meta_before.json")
    ap.add_argument("--out", default="data/coco/annotations_v5_raw.json")
    ap.add_argument("--log", default="docs/data_v5/label_log.json")
    ap.add_argument("--sheets", default="/tmp/lg/sheets")
    ap.add_argument("--per-sheet", type=int, default=12)
    args = ap.parse_args()
    meta = json.load(open(C.META_PATH))
    old = {r["file_name"] for r in json.load(open(args.before))}
    coco = json.load(open(args.src))
    cat_ids = {c["name"]: c["id"] for c in coco["categories"]}
    device = C.pick_device()
    dino_proc, dino, sam_proc, sam = C.load_models(device)
    ann_id = max(a["id"] for a in coco["annotations"]) + 1
    image_id = max(im["id"] for im in coco["images"]) + 1
    log, shown, missed = [], [], []
    for rec in meta:
        if rec["file_name"] in old:
            continue
        image = Image.open(os.path.join(C.IMG_DIR, rec["file_name"])).convert("RGB")
        w, h = image.size
        for prompt, bt, tt in LADDER[rec["class"]]:
            boxes, scores = C.detect_boxes(image, prompt, dino_proc, dino, device, bt, tt)
            kept = C.filter_boxes(boxes, scores, w, h) if len(boxes) else []
            if not kept:
                continue
            masks = C.segment_boxes(image, kept, sam_proc, sam, device)
            anns = []
            for m in masks:
                a = C.make_annotation(ann_id + len(anns), image_id, cat_ids[rec["class"]], m)
                if a:
                    anns.append(a)
            if not anns:
                continue
            ann_id += len(anns)
            coco["annotations"] += anns
            coco["images"].append({"id": image_id, "file_name": rec["file_name"], "width": w,
                                   "height": h, "license": 1, "coco_url": rec["url"],
                                   "flickr_url": "", "date_captured": ""})
            image_id += 1
            log.append({"file": rec["file_name"], "prompt": prompt, "box_thr": bt,
                        "text_thr": tt, "objects": len(anns)})
            shown.append((rec["file_name"], image, anns))
            print("labelled", rec["file_name"], prompt, len(anns), flush=True)
            break
        else:
            missed.append(rec["file_name"])
            print("missed", rec["file_name"], flush=True)
    C.validate_coco(coco, C.IMG_DIR)
    C.write_coco(coco, args.out)
    json.dump({"labelled": log, "missed": missed}, open(args.log, "w"), indent=1)
    os.makedirs(args.sheets, exist_ok=True)
    for i in range(0, len(shown), args.per_sheet):
        sheet(shown[i:i + args.per_sheet], f"{args.sheets}/new_{i // args.per_sheet + 1:02d}.png")
    print(C.class_counts(coco), len(coco["images"]), len(coco["annotations"]))


if __name__ == "__main__":
    main()
