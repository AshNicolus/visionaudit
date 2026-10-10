"""Leakage-safe grouping: same Commons series and near-duplicate images share a group."""
import json
import os
import re
import urllib.parse

import cv2
import numpy as np

DHASH_MAX = 10  # hamming distance (of 64 bits) at or below which two images are one group


def series_key(page_url):
    """Commons page title with the extension and trailing numbers/brackets removed."""
    title = urllib.parse.unquote(page_url.rsplit("File:", 1)[-1])
    title = re.sub(r"\.(jpe?g|png)$", "", title, flags=re.I).replace("_", " ")
    prev = None
    while prev != title:
        prev = title
        title = re.sub(r"[\s,.\-]*(\([^()]*\)|\[[^\[\]]*\]|\d+)\s*$", "", title)
    return title.strip().lower()


def dhash(path):
    """64-bit difference hash as a numpy bool vector."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (small[:, 1:] > small[:, :-1]).flatten()


def hash_distance(a, b):
    return int((a != b).sum())


def build_groups(file_names, meta_path, img_dir):
    """One group id per file name: union of same series key and dHash distance <= DHASH_MAX."""
    with open(meta_path) as f:
        meta = {r["file_name"]: r for r in json.load(f)}
    parent = list(range(len(file_names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    keys = [(meta[n]["class"], series_key(meta[n]["page"])) for n in file_names]
    first = {}
    for i, k in enumerate(keys):
        if k in first:
            union(i, first[k])
        else:
            first[k] = i
    hashes = [dhash(os.path.join(img_dir, n)) for n in file_names]
    for i in range(len(file_names)):
        for j in range(i):
            if hash_distance(hashes[i], hashes[j]) <= DHASH_MAX:
                union(i, j)
    roots = {}
    return np.array([roots.setdefault(find(i), len(roots)) for i in range(len(file_names))])
