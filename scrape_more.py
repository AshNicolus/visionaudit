"""Top up classes beyond the first scrape, same licence filter, skipping duplicates.

A downloaded image is dropped if its sha256 matches an existing file or its dHash is within
groups.DHASH_MAX of one (existing images and ones added in this run).
"""
import argparse
import hashlib
import os
import re
import urllib.parse

import collect_and_annotate as C
from groups import DHASH_MAX, dhash, hash_distance

EXTRA_CATS = {
    "robot_arm": ["Fanuc_robots", "Yaskawa_robots", "Articulated_robots", "SCARA_robots",
                  "Delta_robots", "Robot_arms", "Kawasaki_robots", "Collaborative_robots"],
    "circuit_board": ["Raspberry_Pi", "Arduino", "Single-board_computers", "Motherboards_by_brand",
                      "Electronic_circuit_boards", "Printed_circuit_boards_by_type"],
    "drone": ["DJI_Inspire", "Micro_air_vehicles", "Camera_drones", "Consumer_drones",
              "Parrot_AR.Drone", "Racing_drones", "Agricultural_drones", "Delivery_drones"],
}


# page titles of people, logos and other off-class shots
SKIP_TITLE = re.compile(r"portrait|logo|speaks?\b|interview|conference|"
                        r"\bmeeting\b|ceremony|visit|award|minister|president|CEO|staff|"
                        r"student|crowd|forum|summit|exhibition|fair\b|\bWEF\b", re.I)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def install_dedupe():
    """Wrap C._download_one so duplicate downloads are deleted and not recorded."""
    seen = {}
    for n in os.listdir(C.IMG_DIR):
        p = os.path.join(C.IMG_DIR, n)
        seen[n] = (sha(p), dhash(p))
    orig = C._download_one
    skipped = []

    def guarded(page, request, cls, fpage):
        title = urllib.parse.unquote(fpage.rsplit("File:", 1)[-1])
        if SKIP_TITLE.search(title):
            skipped.append(title)
            return None
        return orig(page, request, cls, fpage)
    stats = {"sha": 0, "dhash": 0}

    def wrapped(page, request, cls, fpage):
        rec = guarded(page, request, cls, fpage)
        if rec is None:
            return None
        p = os.path.join(C.IMG_DIR, rec["file_name"])
        s, d = sha(p), dhash(p)
        for n, (s2, d2) in seen.items():
            kind = "sha" if s == s2 else "dhash" if hash_distance(d, d2) <= DHASH_MAX else None
            if kind:
                stats[kind] += 1
                print(f"dup of {n} ({kind}), dropped {rec['file_name']}", flush=True)
                os.remove(p)
                return None
        seen[rec["file_name"]] = (s, d)
        return rec

    C._download_one = wrapped
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="drone")
    ap.add_argument("--target", type=int, default=130, help="downloaded images for the class")
    ap.add_argument("--budget-s", type=int, default=900)
    ap.add_argument("--extra-cats", action="store_true")
    a = ap.parse_args()
    cfg = dict(C.CLASSES[a.cls])
    if a.extra_cats:
        cfg["categories"] = cfg["categories"] + EXTRA_CATS[a.cls]
    C.CLASSES = {a.cls: cfg}
    stats = install_dedupe()
    recs = C.scrape_images(a.target, budget_s=a.budget_s)
    print(len(recs), stats)


if __name__ == "__main__":
    main()
