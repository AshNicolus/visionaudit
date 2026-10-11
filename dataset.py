"""COCO-backed PyTorch Dataset with OpenCV preprocessing and box/mask-aware augmentation."""
import json
import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_BOX_SIDE = 2.0  # px; smaller boxes after clipping are dropped
MIN_AREA_KEPT = 0.2  # fraction of the unclipped box that must stay visible


# ---- geometry -------------------------------------------------------------

def letterbox_params(h, w, size):
    """Return (scale, pad_x, pad_y) that fit an h x w image into a size x size square."""
    scale = size / max(h, w)
    pad_x = (size - round(w * scale)) // 2
    pad_y = (size - round(h * scale)) // 2
    return scale, pad_x, pad_y


def letterbox_matrix(h, w, size):
    """3x3 matrix mapping source pixels to letterboxed pixels."""
    scale, pad_x, pad_y = letterbox_params(h, w, size)
    return np.array([[scale, 0, pad_x], [0, scale, pad_y], [0, 0, 1]], dtype=np.float64)


def affine_matrix(size, angle=0.0, scale=1.0, tx=0.0, ty=0.0, flip=False):
    """3x3 matrix: optional horizontal flip, then rotate/scale about the centre, then shift."""
    c = size / 2
    rot = cv2.getRotationMatrix2D((c, c), angle, scale)
    m = np.vstack([rot, [0, 0, 1]])
    m[:2, 2] += (tx, ty)
    if flip:
        f = np.array([[-1, 0, size], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        m = m @ f
    return m


def warp_image(img, m, size, interp=cv2.INTER_LINEAR):
    """Warp an image or mask by a 3x3 edge-coordinate matrix onto a size x size canvas.

    Boxes live in edge coordinates (pixel i spans i..i+1) while OpenCV indexes pixel
    centres, so the matrix is conjugated by a half-pixel shift.
    """
    t = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], dtype=np.float64)
    m_cv = np.linalg.inv(t) @ m @ t
    return cv2.warpAffine(img, m_cv[:2], (size, size), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def warp_mask(mask, m, size):
    """Warp a binary mask with nearest-neighbour sampling so it stays binary."""
    return warp_image(mask, m, size, cv2.INTER_NEAREST)


def warp_box(box, m, size):
    """AABB of the 4 transformed corners of an xyxy box, clipped to the image.

    Returns (clipped_box, visible_fraction); the fraction is clipped/unclipped area.
    """
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float64)
    pts = corners @ m.T
    lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
    full = max(hi[0] - lo[0], 1e-9) * max(hi[1] - lo[1], 1e-9)
    lo, hi = np.clip(lo, 0, size), np.clip(hi, 0, size)
    clipped = np.array([lo[0], lo[1], hi[0], hi[1]], dtype=np.float32)
    frac = max(hi[0] - lo[0], 0) * max(hi[1] - lo[1], 0) / full
    return clipped, frac


def is_valid_box(box, frac):
    """False for boxes that are degenerate or mostly cropped out."""
    w, h = box[2] - box[0], box[3] - box[1]
    return w >= MIN_BOX_SIDE and h >= MIN_BOX_SIDE and frac >= MIN_AREA_KEPT


def box_iou(a, b):
    """IoU of two xyxy boxes."""
    iw = max(min(a[2], b[2]) - max(a[0], b[0]), 0)
    ih = max(min(a[3], b[3]) - max(a[1], b[1]), 0)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def transform_sample(img, boxes, masks, m, size):
    """Apply matrix m to image, boxes (N,4 xyxy) and masks (N,H,W); drop degenerate objects.

    Returns (img, boxes, masks, keep_indices).
    """
    out_img = warp_image(img, m, size)
    new_boxes, new_masks, keep = [], [], []
    for i, (box, mask) in enumerate(zip(boxes, masks)):
        wb, frac = warp_box(box, m, size)
        wm = warp_mask(mask, m, size)
        if is_valid_box(wb, frac) and wm.any():
            new_boxes.append(wb)
            new_masks.append(wm)
            keep.append(i)
    boxes_out = np.array(new_boxes, dtype=np.float32).reshape(-1, 4)
    masks_out = np.array(new_masks, dtype=np.uint8).reshape(-1, size, size)
    return out_img, boxes_out, masks_out, keep


def random_affine_matrix(size, rng, max_rot=20, scale_range=(0.8, 1.2), max_shift=0.1):
    """Sample a random flip/rotate/scale/translate matrix."""
    return affine_matrix(
        size,
        angle=rng.uniform(-max_rot, max_rot),
        scale=rng.uniform(*scale_range),
        tx=rng.uniform(-max_shift, max_shift) * size,
        ty=rng.uniform(-max_shift, max_shift) * size,
        flip=rng.random() < 0.5,
    )


def _largest(boxes):
    """Index of the largest xyxy box."""
    return int(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).argmax())


def box_crop_matrix(size, box, rng, zoom=(1.2, 2.0), jitter=0.15, flip_p=0.5):
    """Zoom in about the box centre, then shift the centre near the middle of the canvas.

    Same 3x3 form as every other transform, so boxes and masks follow through
    transform_sample; objects that end up mostly cropped out are dropped there.
    """
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    s = rng.uniform(*zoom)
    # keep the box inside the canvas when it fits, otherwise just zoom less
    longest = max(box[2] - box[0], box[3] - box[1], 1.0)
    s = min(s, 0.95 * size / longest)
    s = max(s, 1.0)
    tx = size / 2 + rng.uniform(-jitter, jitter) * size
    ty = size / 2 + rng.uniform(-jitter, jitter) * size
    m = np.array([[s, 0, tx - s * cx], [0, s, ty - s * cy], [0, 0, 1]], dtype=np.float64)
    if rng.random() < flip_p:
        m = np.array([[-1, 0, size], [0, 1, 0], [0, 0, 1]], dtype=np.float64) @ m
    return m


# ---- preprocessing --------------------------------------------------------

def normalize_image(img_bgr):
    """BGR uint8 HxWx3 -> normalized RGB float32 3xHxW."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1).copy()


def polygons_to_mask(segmentation, h, w, bbox=None):
    """Rasterise COCO polygons, clipped to the bbox; fall back to the filled bbox."""
    mask = np.zeros((h, w), dtype=np.uint8)
    polys = [np.round(np.array(p).reshape(-1, 2)).astype(np.int32)
             for p in (segmentation or []) if len(p) >= 6]
    if polys:
        cv2.fillPoly(mask, polys, 1)  # fills the outline too, so clip to the box below
    if bbox is not None:
        x, y, bw, bh = bbox
        window = np.zeros_like(mask)
        window[int(y):int(math.ceil(y + bh)), int(x):int(math.ceil(x + bw))] = 1
        mask = mask & window if polys else window
    return mask


def letterbox_sample(img, boxes, masks, size):
    """Letterbox image, xyxy boxes and masks into size x size (masks use nearest)."""
    h, w = img.shape[:2]
    m = letterbox_matrix(h, w, size)
    scale, left, top = letterbox_params(h, w, size)
    new = (max(round(w * scale), 1), max(round(h * scale), 1))
    right, bottom = size - new[0] - left, size - new[1] - top

    def fit(a, interp):
        a = cv2.resize(a, new, interpolation=interp)
        return cv2.copyMakeBorder(a, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    out = fit(img, cv2.INTER_AREA)
    out_boxes = np.array([warp_box(b, m, size)[0] for b in boxes], dtype=np.float32).reshape(-1, 4)
    out_masks = np.array([fit(k, cv2.INTER_NEAREST) for k in masks], dtype=np.uint8)
    return out, out_boxes, out_masks.reshape(-1, size, size)


# ---- dataset --------------------------------------------------------------

def load_coco(ann_path):
    """Parse COCO JSON into (images, per-image annotation lists, cat_id -> index)."""
    with open(ann_path) as f:
        coco = json.load(f)
    cat_ids = sorted(c["id"] for c in coco["categories"])
    cat_index = {cid: i for i, cid in enumerate(cat_ids)}
    by_image = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    images = [im for im in coco["images"] if im["id"] in by_image]
    return images, by_image, cat_index


class CocoDataset(Dataset):
    """Returns the largest object per image: class, normalized cxcywh box and binary mask."""

    def __init__(self, ann_path, img_dir, indices=None, img_size=224, train=False, seed=0,
                 crop_prob=0.0):
        self.images, self.by_image, self.cat_index = load_coco(ann_path)
        if indices is not None:
            self.images = [self.images[i] for i in indices]
        self.img_dir, self.size, self.train = img_dir, img_size, train
        self.rng = np.random.default_rng(seed)
        self.crop_prob = crop_prob  # chance of a zoom-in crop around the largest box

    def __len__(self):
        return len(self.images)

    @property
    def num_classes(self):
        return len(self.cat_index)

    def _load(self, idx):
        """Read image and build xyxy boxes, masks, class indices at original resolution."""
        info = self.images[idx]
        img = cv2.imread(os.path.join(self.img_dir, info["file_name"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(info["file_name"])
        h, w = img.shape[:2]
        anns = self.by_image[info["id"]]
        boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                           a["bbox"][1] + a["bbox"][3]] for a in anns], dtype=np.float32)
        masks = np.array([polygons_to_mask(a.get("segmentation"), h, w, a["bbox"])
                          for a in anns], dtype=np.uint8)
        labels = np.array([self.cat_index[a["category_id"]] for a in anns], dtype=np.int64)
        return img, boxes, masks, labels

    def _augment(self, img, boxes, masks, labels):
        """Random affine; falls back to the input if no object survives."""
        if self.crop_prob and self.rng.random() < self.crop_prob:
            m = box_crop_matrix(self.size, boxes[_largest(boxes)], self.rng)
        else:
            m = random_affine_matrix(self.size, self.rng)
        out_img, out_boxes, out_masks, keep = transform_sample(img, boxes, masks, m, self.size)
        if not keep:
            return img, boxes, masks, labels
        return out_img, out_boxes, out_masks, labels[keep]

    def __getitem__(self, idx):
        img, boxes, masks, labels = self._load(idx)
        img, boxes, masks = letterbox_sample(img, boxes, masks, self.size)
        if self.train:
            img, boxes, masks, labels = self._augment(img, boxes, masks, labels)
        k = _largest(boxes)
        x1, y1, x2, y2 = boxes[k] / self.size
        box = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float32)
        return {
            "image": torch.from_numpy(normalize_image(img)),
            "label": torch.tensor(labels[k], dtype=torch.long),
            "box": torch.from_numpy(box),
            "mask": torch.from_numpy(masks[k].astype(np.float32))[None],
        }


def split_indices(n, val_split, seed):
    """Seeded train/val index split."""
    perm = np.random.RandomState(seed).permutation(n)
    n_val = int(round(n * val_split))
    return perm[n_val:].tolist(), perm[:n_val].tolist()


def get_dataloaders(ann_path="data/coco/annotations.json", img_dir="data/images",
                    img_size=224, batch_size=16, val_split=0.2, seed=42, crop_prob=0.0):
    """Build (train_loader, val_loader, num_classes); only train is augmented."""
    n = len(load_coco(ann_path)[0])
    train_idx, val_idx = split_indices(n, val_split, seed)
    train_ds = CocoDataset(ann_path, img_dir, train_idx, img_size, train=True, seed=seed,
                           crop_prob=crop_prob)
    val_ds = CocoDataset(ann_path, img_dir, val_idx, img_size, train=False)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_ds.num_classes
