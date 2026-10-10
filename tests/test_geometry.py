"""Geometry tests: a synthetic box+mask pushed through the affine pipeline."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataset as ds  # noqa: E402

SIZE = 200


def make_sample(box=(60, 80, 140, 120)):
    """Grey image with one filled rectangle; mask matches the box exactly."""
    x1, y1, x2, y2 = box
    mask = np.zeros((SIZE, SIZE), np.uint8)
    mask[y1:y2, x1:x2] = 1
    img = np.full((SIZE, SIZE, 3), 100, np.uint8)
    img[mask > 0] = 255
    return img, np.array([box], np.float32), mask[None]


def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def test_identity_keeps_box_and_mask():
    img, boxes, masks = make_sample()
    _, b, mk, keep = ds.transform_sample(img, boxes, masks, ds.affine_matrix(SIZE), SIZE)
    assert keep == [0]
    assert ds.box_iou(b[0], boxes[0]) == pytest.approx(1.0)
    assert (mk[0] == masks[0]).all()


def test_flip_mirrors_box():
    img, boxes, masks = make_sample()
    m = ds.affine_matrix(SIZE, flip=True)
    _, b, _, _ = ds.transform_sample(img, boxes, masks, m, SIZE)
    expected = np.array([SIZE - 140, 80, SIZE - 60, 120], np.float32)
    assert ds.box_iou(b[0], expected) > 0.99


def test_translate_shifts_box():
    img, boxes, masks = make_sample()
    m = ds.affine_matrix(SIZE, tx=15, ty=-10)
    _, b, _, _ = ds.transform_sample(img, boxes, masks, m, SIZE)
    assert ds.box_iou(b[0], np.array([75, 70, 155, 110], np.float32)) > 0.99


def test_scale_about_centre():
    img, boxes, masks = make_sample()
    m = ds.affine_matrix(SIZE, scale=1.5)
    _, b, _, _ = ds.transform_sample(img, boxes, masks, m, SIZE)
    expected = np.array([100 - 40 * 1.5, 100 - 20 * 1.5, 100 + 40 * 1.5, 100 + 20 * 1.5])
    assert ds.box_iou(b[0], expected) > 0.99


def test_rotation_90_swaps_sides():
    img, boxes, masks = make_sample()
    m = ds.affine_matrix(SIZE, angle=90)
    _, b, mk, _ = ds.transform_sample(img, boxes, masks, m, SIZE)
    expected = np.array([80, 60, 120, 140], np.float32)
    assert ds.box_iou(b[0], expected) > 0.95
    assert ds.box_iou(b[0], mask_bbox(mk[0])) > 0.95


def test_rotation_45_box_is_aabb_of_corners():
    img, boxes, masks = make_sample()
    m = ds.affine_matrix(SIZE, angle=45)
    _, b, mk, _ = ds.transform_sample(img, boxes, masks, m, SIZE)
    half = (40 + 20) / np.sqrt(2)
    expected = np.array([100 - half, 100 - half, 100 + half, 100 + half])
    assert ds.box_iou(b[0], expected) > 0.99
    # AABB must enclose the rotated mask.
    mb = mask_bbox(mk[0])
    assert b[0][0] <= mb[0] + 1 and b[0][1] <= mb[1] + 1
    assert b[0][2] >= mb[2] - 1 and b[0][3] >= mb[3] - 1


def test_mask_stays_binary_after_warp():
    img, boxes, masks = make_sample()
    m = ds.affine_matrix(SIZE, angle=33, scale=1.1, tx=5)
    _, _, mk, _ = ds.transform_sample(img, boxes, masks, m, SIZE)
    assert set(np.unique(mk)) <= {0, 1}


def test_box_clipped_to_image():
    img, boxes, masks = make_sample()
    m = ds.affine_matrix(SIZE, tx=60)
    _, b, _, keep = ds.transform_sample(img, boxes, masks, m, SIZE)
    assert keep == [0]
    assert b[0][2] == SIZE
    assert (b >= 0).all() and (b <= SIZE).all()


def test_object_pushed_off_image_is_dropped():
    img, boxes, masks = make_sample()
    m = ds.affine_matrix(SIZE, tx=400)
    _, b, mk, keep = ds.transform_sample(img, boxes, masks, m, SIZE)
    assert keep == [] and len(b) == 0 and len(mk) == 0


def test_mostly_cropped_box_is_dropped():
    img, boxes, masks = make_sample()
    m = ds.affine_matrix(SIZE, tx=130)  # ~12% stays inside
    _, _, _, keep = ds.transform_sample(img, boxes, masks, m, SIZE)
    assert keep == []


def test_degenerate_thin_box_is_invalid():
    assert not ds.is_valid_box(np.array([10, 10, 11, 50]), 1.0)


def test_letterbox_preserves_aspect_and_centres():
    img = np.full((100, 200, 3), 255, np.uint8)
    box = np.array([[0, 0, 200, 100]], np.float32)
    mask = np.ones((1, 100, 200), np.uint8)
    out, b, mk = ds.letterbox_sample(img, box, mask, 100)
    assert out.shape == (100, 100, 3)
    assert ds.box_iou(b[0], np.array([0, 25, 100, 75], np.float32)) > 0.98
    assert mk[0, :20].sum() == 0 and mk[0, 40:60].all()


def test_normalize_shape_and_range():
    img = np.zeros((8, 8, 3), np.uint8)
    out = ds.normalize_image(img)
    assert out.shape == (3, 8, 8) and out.dtype == np.float32
    assert out.min() < 0  # zero pixels land below the mean


def test_random_affine_is_seeded():
    a = ds.random_affine_matrix(SIZE, np.random.default_rng(1))
    b = ds.random_affine_matrix(SIZE, np.random.default_rng(1))
    assert np.allclose(a, b)


def test_mask_matches_box_exactly_under_flip():
    img, boxes, masks = make_sample()
    m = ds.affine_matrix(SIZE, flip=True)
    _, b, mk, _ = ds.transform_sample(img, boxes, masks, m, SIZE)
    assert (mask_bbox(mk[0]) == b[0]).all()  # cols 60..139, not 61..141


def test_polygon_mask_not_wider_than_bbox():
    poly = [[10, 10, 50, 10, 50, 40, 10, 40]]
    mask = ds.polygons_to_mask(poly, 80, 100, [10, 10, 40, 30])
    assert (mask_bbox(mask) == [10, 10, 50, 40]).all()
    assert mask.sum() == 40 * 30


def test_letterbox_mask_matches_box():
    img = np.zeros((100, 200, 3), np.uint8)
    mask = np.zeros((1, 100, 200), np.uint8)
    mask[0, 20:60, 40:120] = 1
    box = np.array([[40, 20, 120, 60]], np.float32)
    _, b, mk = ds.letterbox_sample(img, box, mask, 100)
    assert np.abs(mask_bbox(mk[0]) - b[0]).max() <= 1


def test_box_crop_keeps_box_and_mask_consistent():
    img, boxes, masks = make_sample()
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = ds.box_crop_matrix(SIZE, boxes[0], rng)
        _, b, mk, keep = ds.transform_sample(img, boxes, masks, m, SIZE)
        assert keep == [0]
        assert b.min() >= 0 and b.max() <= SIZE
        assert np.abs(mask_bbox(mk[0]) - b[0]).max() <= 1  # fractional scale: within a pixel
        # zooming in: the box is never smaller than the original
        assert (b[0, 2] - b[0, 0]) >= (boxes[0, 2] - boxes[0, 0]) - 1
