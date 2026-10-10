"""Scrape images with Playwright, auto-label them with Grounding DINO + SAM, write COCO JSON."""
import argparse
import json
import os
import re
import time
from datetime import datetime, timezone

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(ROOT, "data", "images")
META_PATH = os.path.join(ROOT, "data", "raw_meta.json")
COCO_PATH = os.path.join(ROOT, "data", "coco", "annotations.json")

COMMONS = "https://commons.wikimedia.org"
USER_AGENT = "AugleAssessmentBot/0.1 (educational dataset project)"
DELAY_S = 0.7
MAX_SIDE = 1600  # saved images are downscaled to this
MIN_SIDE = 720  # smaller sources are rejected as not high-res
MAX_BYTES = 25 * 1024 * 1024
LICENSE_OK = re.compile(r"\bCC\b|Creative Commons|public domain|\bPD\b", re.I)

# class name -> Commons categories to crawl and the Grounding DINO text prompt
CLASSES = {
    "robot_arm": {
        "categories": ["Industrial_robots", "Robotic_arms", "KUKA_robots", "ABB_robots"],
        "prompt": "a robot arm.",
    },
    "circuit_board": {
        "categories": ["Printed_circuit_boards", "Circuit_boards", "Motherboards"],
        "prompt": "a circuit board.",
    },
    "drone": {
        "categories": ["Unmanned_aerial_vehicles", "DJI_Phantom", "Quadcopters"],
        "prompt": "a drone.",
    },
}

DINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam-vit-base"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
MAX_OBJECTS = 4
MIN_BOX_FRAC = 0.03  # box area as a fraction of the image


# ---- scraping -------------------------------------------------------------

def goto(page, url, tries=3):
    """page.goto with retries; Commons is occasionally slow."""
    for attempt in range(tries):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            return True
        except Exception:
            time.sleep(3 * (attempt + 1))
    return False


def category_files(page, category, limit):
    """Collect File: page URLs from a Commons category, following 'next page' links."""
    url, found = f"{COMMONS}/wiki/Category:{category}", []
    while url and len(found) < limit:
        if not goto(page, url):
            break
        links = page.eval_on_selector_all(
            "#mw-category-media a.mw-file-description", "els => els.map(e => e.href)")
        found += [u for u in links if re.search(r"\.(jpe?g|png)$", u, re.I)]
        nxt = page.query_selector("a:has-text('next page')")
        url = nxt.get_attribute("href") if nxt else None
        url = COMMONS + url if url and url.startswith("/") else url
        time.sleep(DELAY_S)
    return found[:limit]


def file_info(page, file_url):
    """Read the original-resolution link and license from a Commons file page."""
    if not goto(page, file_url):
        return None
    link = page.query_selector("div.fullImageLink a")
    if link is None:
        return None
    lic = page.query_selector_all(".licensetpl_short")
    license_name = lic[0].inner_text().strip() if lic else ""
    dims = re.search(r"Original file\W+([\d,]+)\s*[×x]\s*([\d,]+)", page.inner_text("body"))
    size = [int(g.replace(",", "")) for g in dims.groups()] if dims else None
    return {"page": file_url, "url": link.get_attribute("href"), "license": license_name,
            "size": size}


def thumb_url(url, width):
    """Wikimedia thumbnail URL of the same file at a standard width."""
    m = re.match(r"(https://upload\.wikimedia\.org/wikipedia/commons)/(\w/\w\w/)([^?]+)", url)
    return f"{m[1]}/thumb/{m[2]}{m[3]}/{width}px-{m[3]}" if m else None


def fetch_image(request, info):
    """Try the original, then standard-width thumbnails (huge originals get rate-limited)."""
    big = info["size"] and max(info["size"]) > 3000
    urls = [] if big else [info["url"]]
    urls += [u for u in (thumb_url(info["url"], 1920), thumb_url(info["url"], 1280)) if u]
    for url in urls:
        data = fetch_bytes(request, url)
        if data:
            return data
    return None


def fetch_bytes(request, url, retries=2):
    """GET with back-off on 429/5xx; returns bytes or None."""
    for attempt in range(retries):
        resp = request.get(url, timeout=60000)
        if resp.status == 200:
            if int(resp.headers.get("content-length", 0)) > MAX_BYTES:
                return None
            return resp.body()
        if resp.status in (429, 500, 502, 503):
            time.sleep(3 * (attempt + 1))
            continue
        return None
    return None


def save_image(data, path):
    """Decode, reject small images, downscale to MAX_SIDE, write JPEG. Returns (w, h) or None."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or min(img.shape[:2]) < MIN_SIDE:
        return None
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return img.shape[1], img.shape[0]


def scrape_images(per_class=80, headless=True, budget_s=None):
    """Download up to per_class licensed high-res images per class; returns metadata records."""
    from playwright.sync_api import sync_playwright

    os.makedirs(IMG_DIR, exist_ok=True)
    records = _load_meta()
    deadline = time.time() + budget_s if budget_s else float("inf")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        ctx = browser.new_context(user_agent=USER_AGENT)
        page = ctx.new_page()
        for cls, cfg in CLASSES.items():
            have = {r["page"] for r in records}
            n = sum(r["class"] == cls for r in records)
            for cat in cfg["categories"]:
                if n >= per_class:
                    break
                for fpage in category_files(page, cat, per_class * 3):
                    if n >= per_class or time.time() > deadline:
                        break
                    if fpage in have:
                        continue
                    rec = _download_one(page, ctx.request, cls, fpage)
                    have.add(fpage)
                    if rec:
                        records.append(rec)
                        n += 1
                        _save_meta(records)
                        print(f"[{cls}] {n}/{per_class} {rec['file_name']}", flush=True)
        browser.close()
    return records


def next_index(cls):
    """Lowest unused numeric suffix for this class's image files."""
    used = {int(m.group(1)) for f in os.listdir(IMG_DIR)
            if (m := re.fullmatch(rf"{cls}_(\d+)\.jpg", f))}
    return max(used, default=-1) + 1


def _download_one(page, request, cls, fpage):
    """Fetch one file page + image; returns a metadata record or None."""
    info = file_info(page, fpage)
    time.sleep(DELAY_S)
    if not info or not LICENSE_OK.search(info["license"]):
        return None
    if info["size"] and min(info["size"]) < MIN_SIDE:
        return None
    data = fetch_image(request, info)
    time.sleep(DELAY_S)
    if data is None:
        return None
    name = f"{cls}_{next_index(cls):04d}.jpg"
    size = save_image(data, os.path.join(IMG_DIR, name))
    if size is None:
        return None
    return {"class": cls, "file_name": name, "width": size[0], "height": size[1],
            "page": fpage, "url": info["url"], "license": info["license"]}


def _load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            recs = json.load(f)
        return [r for r in recs if os.path.exists(os.path.join(IMG_DIR, r["file_name"]))]
    return []


def _save_meta(records):
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(records, f, indent=1)


# ---- annotation -----------------------------------------------------------

def pick_device():
    """cuda > mps > cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(device):
    """Load Grounding DINO and SAM with their processors."""
    from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel,
                              SamProcessor)
    dino_proc = AutoProcessor.from_pretrained(DINO_ID)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(device).eval()
    sam_proc = SamProcessor.from_pretrained(SAM_ID)
    sam = SamModel.from_pretrained(SAM_ID).to(device).eval()
    return dino_proc, dino, sam_proc, sam


def detect_boxes(image, prompt, dino_proc, dino, device):
    """Grounding DINO boxes (xyxy px) and scores for one PIL image and text prompt."""
    import torch
    inputs = dino_proc(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = dino(**inputs)
    res = dino_proc.post_process_grounded_object_detection(
        out, inputs["input_ids"], threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]])[0]
    return res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()


def filter_boxes(boxes, scores, w, h):
    """Drop tiny and near-duplicate boxes; keep the top MAX_OBJECTS by score."""
    order = np.argsort(-scores)
    kept = []
    for i in order:
        x1, y1, x2, y2 = np.clip(boxes[i], 0, [w, h, w, h])
        if (x2 - x1) * (y2 - y1) < MIN_BOX_FRAC * w * h:
            continue
        box = np.array([x1, y1, x2, y2])
        if all(_iou(box, k) < 0.6 for k in kept):
            kept.append(box)
        if len(kept) == MAX_OBJECTS:
            break
    return kept


def _iou(a, b):
    iw = max(min(a[2], b[2]) - max(a[0], b[0]), 0)
    ih = max(min(a[3], b[3]) - max(a[1], b[1]), 0)
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def segment_boxes(image, boxes, sam_proc, sam, device):
    """SAM masks (bool, N x H x W) for the given xyxy boxes."""
    import torch
    inputs = sam_proc(image, input_boxes=[[b.tolist() for b in boxes]], return_tensors="pt")
    feed = {k: v.to(device, torch.float32 if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()}  # MPS has no float64
    with torch.no_grad():
        out = sam(**feed, multimask_output=False)
    masks = sam_proc.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"], inputs["reshaped_input_sizes"])[0]
    return masks[:, 0].numpy().astype(bool)


def mask_to_polygons(mask, min_area=64):
    """Binary mask -> list of flat COCO polygons (external contours, simplified)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(c) >= 3:
            polys.append([float(v) for v in c.flatten()])
    return polys


def make_annotation(ann_id, image_id, cat_id, mask):
    """COCO annotation dict from a SAM mask; the bbox is the mask's own tight bbox."""
    polys = mask_to_polygons(mask)
    if not polys:
        return None
    ys, xs = np.nonzero(mask)
    x, y = float(xs.min()), float(ys.min())
    w, h = float(xs.max() + 1 - x), float(ys.max() + 1 - y)
    return {"id": ann_id, "image_id": image_id, "category_id": cat_id,
            "bbox": [x, y, w, h], "area": float(mask.sum()), "iscrowd": 0,
            "segmentation": polys}


def annotate_images(records, device=None):
    """Run DINO + SAM over every record; returns a COCO dict."""
    from PIL import Image
    device = device or pick_device()
    print(f"annotating on {device}", flush=True)
    dino_proc, dino, sam_proc, sam = load_models(device)
    cat_ids = {name: i + 1 for i, name in enumerate(CLASSES)}
    coco = {
        "info": {"description": "Auto-labelled with Grounding DINO + SAM",
                 "date_created": datetime.now(timezone.utc).isoformat()},
        "licenses": [{"id": 1, "name": "Wikimedia Commons (per-file, see raw_meta.json)"}],
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n, "supercategory": "object"} for n, i in cat_ids.items()],
    }
    ann_id = 1
    for rec in records:
        image = Image.open(os.path.join(IMG_DIR, rec["file_name"])).convert("RGB")
        w, h = image.size
        prompt = CLASSES[rec["class"]]["prompt"]
        boxes, scores = detect_boxes(image, prompt, dino_proc, dino, device)
        kept = filter_boxes(boxes, scores, w, h) if len(boxes) else []
        if not kept:
            print(f"skip (no detection) {rec['file_name']}", flush=True)
            continue
        masks = segment_boxes(image, kept, sam_proc, sam, device)
        image_id = len(coco["images"]) + 1
        anns = []
        for mask in masks:
            ann = make_annotation(ann_id + len(anns), image_id, cat_ids[rec["class"]], mask)
            if ann:
                anns.append(ann)
        if not anns:
            continue
        ann_id += len(anns)
        coco["annotations"] += anns
        coco["images"].append({"id": image_id, "file_name": rec["file_name"], "width": w,
                               "height": h, "license": 1, "coco_url": rec["url"],
                               "flickr_url": "", "date_captured": ""})
        print(f"labelled {rec['file_name']}: {len(anns)} objects", flush=True)
    return coco


# ---- validation -----------------------------------------------------------

def _fail(msg):
    raise ValueError(f"invalid COCO: {msg}")


def _check_unique(items, name):
    ids = [it.get("id") for it in items]
    if any(not isinstance(i, int) or isinstance(i, bool) for i in ids):
        _fail(f"{name} ids must be integers")
    if len(set(ids)) != len(ids):
        _fail(f"duplicate {name} ids")


def _check_annotation(a, images, cat_ids):
    """Validate one annotation against its image."""
    for key in ("id", "image_id", "category_id", "bbox", "area", "iscrowd", "segmentation"):
        if key not in a:
            _fail(f"annotation {a.get('id')} missing '{key}'")
    if a["image_id"] not in images:
        _fail(f"annotation {a['id']} references unknown image {a['image_id']}")
    if a["category_id"] not in cat_ids:
        _fail(f"annotation {a['id']} references unknown category {a['category_id']}")
    if a["iscrowd"] not in (0, 1):
        _fail(f"annotation {a['id']} iscrowd must be 0 or 1")
    box = a["bbox"]
    if not (isinstance(box, list) and len(box) == 4
            and all(isinstance(v, (int, float)) and np.isfinite(v) for v in box)):
        _fail(f"annotation {a['id']} bbox must be 4 finite numbers")
    x, y, w, h = box
    img = images[a["image_id"]]
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        _fail(f"annotation {a['id']} has degenerate or negative bbox {box}")
    if x + w > img["width"] + 1e-3 or y + h > img["height"] + 1e-3:
        _fail(f"annotation {a['id']} bbox exceeds image bounds")
    if (not isinstance(a["area"], (int, float)) or a["area"] <= 0
            or a["area"] > img["width"] * img["height"]):
        _fail(f"annotation {a['id']} has invalid area {a['area']}")
    _check_polygons(a, img)


def _check_polygons(a, img):
    segm = a["segmentation"]
    if a["iscrowd"] == 0 and (not isinstance(segm, list) or not segm):
        _fail(f"annotation {a['id']} needs a non-empty polygon list")
    if isinstance(segm, list):
        for poly in segm:
            if not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2:
                _fail(f"annotation {a['id']} has a malformed polygon")
            xs, ys = poly[0::2], poly[1::2]
            if min(xs) < 0 or min(ys) < 0 or max(xs) > img["width"] or max(ys) > img["height"]:
                _fail(f"annotation {a['id']} polygon outside image")


def validate_coco(coco, img_dir=None):
    """Raise ValueError on any COCO schema problem; return the dict if it is valid."""
    for key in ("images", "annotations", "categories"):
        if not isinstance(coco.get(key), list) or not coco[key]:
            _fail(f"'{key}' must be a non-empty list")
    for name in ("images", "annotations", "categories"):
        _check_unique(coco[name], name)
    for im in coco["images"]:
        for key in ("file_name", "width", "height"):
            if key not in im:
                _fail(f"image {im['id']} missing '{key}'")
        if im["width"] <= 0 or im["height"] <= 0:
            _fail(f"image {im['id']} has non-positive size")
        if img_dir and not os.path.exists(os.path.join(img_dir, im["file_name"])):
            _fail(f"image file missing: {im['file_name']}")
    for c in coco["categories"]:
        if not c.get("name"):
            _fail(f"category {c['id']} missing name")
    images = {im["id"]: im for im in coco["images"]}
    cat_ids = {c["id"] for c in coco["categories"]}
    for a in coco["annotations"]:
        _check_annotation(a, images, cat_ids)
    return coco


def write_coco(coco, path=COCO_PATH):
    """Write COCO JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(coco, f)


def class_counts(coco):
    """Per-class (images, annotations) counts."""
    names = {c["id"]: c["name"] for c in coco["categories"]}
    out = {n: {"images": set(), "annotations": 0} for n in names.values()}
    for a in coco["annotations"]:
        out[names[a["category_id"]]]["images"].add(a["image_id"])
        out[names[a["category_id"]]]["annotations"] += 1
    return {n: (len(v["images"]), v["annotations"]) for n, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=80)
    ap.add_argument("--skip-scrape", action="store_true", help="reuse data/images + raw_meta.json")
    ap.add_argument("--budget-s", type=int, help="stop scraping after this many seconds")
    ap.add_argument("--scrape-only", action="store_true", help="stop before annotation")
    args = ap.parse_args()
    records = _load_meta() if args.skip_scrape else scrape_images(args.per_class, budget_s=args.budget_s)
    if args.scrape_only:
        return
    coco = annotate_images(records)
    validate_coco(coco, IMG_DIR)
    write_coco(coco)
    print("valid COCO written to", COCO_PATH)
    print("class: (images, annotations)", class_counts(coco))
    print("total images", len(coco["images"]), "annotations", len(coco["annotations"]))


if __name__ == "__main__":
    main()
