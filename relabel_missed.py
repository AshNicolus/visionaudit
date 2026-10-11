"""Second labelling pass for downloaded images Grounding DINO missed with the default prompt.

Tries extra prompts and lower thresholds, appends the hits to the COCO file and writes a
log of which prompt/threshold labelled which image. Licences are unchanged: these images
already passed the LICENSE_OK filter when they were scraped.
"""
import argparse
import json
import os

from PIL import Image

import collect_and_annotate as C

# (prompt, box threshold, text threshold), tried in order; first with a detection wins
EXTRA = {
    "drone": [("a drone.", 0.35, 0.25), ("a drone.", 0.25, 0.20), ("a quadcopter.", 0.25, 0.20),
              ("an unmanned aerial vehicle.", 0.25, 0.20), ("a drone. a quadcopter.", 0.20, 0.20)],
    "robot_arm": [("a robotic arm.", 0.25, 0.20), ("an industrial robot.", 0.25, 0.20)],
    "circuit_board": [("a printed circuit board.", 0.25, 0.20)],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--src", default="data/coco/annotations_v2.json")
    ap.add_argument("--out", default="data/coco/annotations_v3.json")
    ap.add_argument("--skip-meta", help="meta file; its images are not retried")
    ap.add_argument("--log", default="docs/relabel_log.json")
    args = ap.parse_args()
    meta = json.load(open(C.META_PATH))
    coco = json.load(open(args.src))
    have = {im["file_name"] for im in coco["images"]}
    if args.skip_meta:
        have |= {r["file_name"] for r in json.load(open(args.skip_meta))}
    cat_ids = {c["name"]: c["id"] for c in coco["categories"]}
    device = C.pick_device()
    dino_proc, dino, sam_proc, sam = C.load_models(device)
    ann_id = max(a["id"] for a in coco["annotations"]) + 1
    log = []
    for rec in meta:
        if rec["file_name"] in have:
            continue
        image = Image.open(os.path.join(C.IMG_DIR, rec["file_name"])).convert("RGB")
        w, h = image.size
        for prompt, bt, tt in EXTRA[rec["class"]]:
            boxes, scores = C.detect_boxes(image, prompt, dino_proc, dino, device, bt, tt)
            kept = C.filter_boxes(boxes, scores, w, h) if len(boxes) else []
            if not kept:
                continue
            masks = C.segment_boxes(image, kept, sam_proc, sam, device)
            image_id = max(im["id"] for im in coco["images"]) + 1
            anns = []
            for mask in masks:
                a = C.make_annotation(ann_id + len(anns), image_id, cat_ids[rec["class"]], mask)
                if a:
                    anns.append(a)
            if not anns:
                continue
            ann_id += len(anns)
            coco["annotations"] += anns
            coco["images"].append({"id": image_id, "file_name": rec["file_name"], "width": w,
                                   "height": h, "license": 1, "coco_url": rec["url"],
                                   "flickr_url": "", "date_captured": ""})
            log.append({"file": rec["file_name"], "prompt": prompt, "box_thr": bt,
                        "text_thr": tt, "objects": len(anns)})
            print("labelled", rec["file_name"], prompt, len(anns), flush=True)
            break
        else:
            print("still missed", rec["file_name"], flush=True)
    C.validate_coco(coco, C.IMG_DIR)
    C.write_coco(coco, args.out)
    json.dump(log, open(args.log, "w"), indent=1)
    print(C.class_counts(coco), len(coco["images"]), len(coco["annotations"]))


if __name__ == "__main__":
    main()
