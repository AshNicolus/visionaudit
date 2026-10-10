"""COCO schema tests: validate_coco accepts good files and rejects each class of defect."""
import copy
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import collect_and_annotate as ca  # noqa: E402


def good_coco():
    return {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 80}],
        "categories": [{"id": 1, "name": "drone", "supercategory": "object"}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 40.0, 30.0],
            "area": 1200.0, "iscrowd": 0,
            "segmentation": [[10, 10, 50, 10, 50, 40, 10, 40]],
        }],
    }


def test_valid_passes():
    assert ca.validate_coco(good_coco())


def mutate(fn):
    doc = copy.deepcopy(good_coco())
    fn(doc)
    return doc


BAD = {
    "empty annotations": lambda d: d.update(annotations=[]),
    "missing categories": lambda d: d.pop("categories"),
    "duplicate ann id": lambda d: d["annotations"].append(copy.deepcopy(d["annotations"][0])),
    "duplicate image id": lambda d: d["images"].append(copy.deepcopy(d["images"][0])),
    "unknown image": lambda d: d["annotations"][0].update(image_id=9),
    "unknown category": lambda d: d["annotations"][0].update(category_id=9),
    "bbox wrong length": lambda d: d["annotations"][0].update(bbox=[1, 2, 3]),
    "bbox zero width": lambda d: d["annotations"][0].update(bbox=[10, 10, 0, 5]),
    "bbox negative": lambda d: d["annotations"][0].update(bbox=[-1, 10, 5, 5]),
    "bbox out of bounds": lambda d: d["annotations"][0].update(bbox=[90, 10, 40, 5]),
    "bbox nan": lambda d: d["annotations"][0].update(bbox=[10, 10, float("nan"), 5]),
    "bad iscrowd": lambda d: d["annotations"][0].update(iscrowd=2),
    "zero area": lambda d: d["annotations"][0].update(area=0),
    "missing segmentation": lambda d: d["annotations"][0].pop("segmentation"),
    "short polygon": lambda d: d["annotations"][0].update(segmentation=[[1, 2, 3, 4]]),
    "polygon outside": lambda d: d["annotations"][0].update(segmentation=[[0, 0, 500, 0, 5, 5]]),
    "image no size": lambda d: d["images"][0].pop("width"),
}


@pytest.mark.parametrize("name", list(BAD))
def test_invalid_raises(name):
    with pytest.raises(ValueError):
        ca.validate_coco(mutate(BAD[name]))


def test_missing_image_file_detected(tmp_path):
    with pytest.raises(ValueError):
        ca.validate_coco(good_coco(), img_dir=str(tmp_path))


def test_mask_to_polygons_roundtrip():
    mask = np.zeros((60, 80), np.uint8)
    mask[10:40, 20:60] = 1
    polys = ca.mask_to_polygons(mask)
    assert len(polys) == 1 and len(polys[0]) >= 6
    xs, ys = polys[0][0::2], polys[0][1::2]
    assert min(xs) == 20 and max(xs) == 59 and min(ys) == 10 and max(ys) == 39


def test_make_annotation_is_valid():
    mask = np.zeros((80, 100), bool)
    mask[10:40, 10:50] = True
    ann = ca.make_annotation(1, 1, 1, mask)
    assert ann["bbox"] == [10.0, 10.0, 40.0, 30.0] and ann["area"] == 1200.0
    doc = good_coco()
    doc["annotations"] = [ann]
    ca.validate_coco(doc)


def test_filter_boxes_dedups_and_drops_tiny():
    boxes = np.array([[0, 0, 500, 500], [5, 5, 495, 495], [0, 0, 5, 5]], np.float32)
    kept = ca.filter_boxes(boxes, np.array([0.9, 0.8, 0.7]), 1000, 1000)
    assert len(kept) == 1


@pytest.mark.skipif(not os.path.exists(ca.COCO_PATH), reason="dataset not generated")
def test_real_annotations_are_valid():
    with open(ca.COCO_PATH) as f:
        coco = json.load(f)
    ca.validate_coco(coco, ca.IMG_DIR)
    assert len(coco["images"]) >= 150 and len(coco["categories"]) >= 3


def test_annotation_polygon_roundtrips_through_dataset_mask():
    import dataset as ds
    mask = np.zeros((80, 100), bool)
    mask[10:40, 10:50] = True
    ann = ca.make_annotation(1, 1, 1, mask)
    back = ds.polygons_to_mask(ann["segmentation"], 80, 100, ann["bbox"])
    assert (back.astype(bool) == mask).all()
