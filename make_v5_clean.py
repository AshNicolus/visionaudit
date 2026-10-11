"""Apply the eyeball review verdicts to the raw v5 labels and write the clean COCO file.

The drop list is the result of reviewing every new label on contact sheets (annotate_v5.py).
"""
import collections
import hashlib
import json
import re

import collect_and_annotate as C

DROPS = {
    "person, not a board": "circuit_board_0082",
    "board lost in clutter, or tiny sensor on a toy": "circuit_board_0089 circuit_board_0143",
    "illustration or art, not a physical board": "circuit_board_0095 circuit_board_0131 "
                                                 "circuit_board_0141 circuit_board_0164",
    "pile of many boards, whole-frame box": " ".join(f"circuit_board_{i:04d}" for i in (
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 138)),
    "dark or blurred, box does not fit": "circuit_board_0120",
    "macro of components, not a board": "circuit_board_0125 circuit_board_0126 "
                                        "circuit_board_0127 circuit_board_0157 circuit_board_0158",
    "micrograph": "circuit_board_0133 circuit_board_0134",
    "wrong object (cabinet, machine, cables, pendant, welding)": " ".join(
        f"robot_arm_{i:04d}" for i in (98, 99, 150, 152, 169, 170, 171, 172, 173, 176, 177, 178,
                                       122)),
    "circuit board filed under robot_arm": "robot_arm_0101",
    "not an arm (mobile robots, car, sculpture, people)": "robot_arm_0107 robot_arm_0109 "
                                                          "robot_arm_0113 robot_arm_0124 "
                                                          "robot_arm_0131",
    "dark image, label unusable": "robot_arm_0110 robot_arm_0111",
    "messy or overlapping boxes, or tiny box on a crowded scene": " ".join(
        f"robot_arm_{i:04d}" for i in (81, 126, 130, 132, 139, 143, 154, 158, 162, 163, 165, 166,
                                       167)),
    "insect or toy, not a drone": "drone_0108 drone_0109 drone_0110",
    "part, battery, tool or phone, not a drone": "drone_0112 drone_0115 drone_0123 drone_0125",
    "box on person, car, animal or windsock, not the drone": "drone_0132 drone_0135 drone_0136 "
                                                             "drone_0137 drone_0145",
}


def main(raw_path="data/coco/annotations_v5_raw.json", log_path="docs/data_v5/label_log.json",
         out_path="docs/data_v5/annotations_v5_clean.json",
         review_path="docs/data_v5/review.json",
         old_review="docs/data_v5/old_sample_review.json"):
    drop = {n + ".jpg": r for r, names in DROPS.items() for n in names.split()}
    old = json.load(open(old_review))["wrong"]  # wrong labels found in the v4clean sample
    raw = json.load(open(raw_path))
    log = json.load(open(log_path))
    new = {x["file"]: x for x in log["labelled"]}
    assert all(d in new for d in drop), [d for d in drop if d not in new]
    gone = {im["id"] for im in raw["images"] if im["file_name"] in drop or im["file_name"] in old}
    raw["images"] = [i for i in raw["images"] if i["id"] not in gone]
    raw["annotations"] = [a for a in raw["annotations"] if a["image_id"] not in gone]
    C.validate_coco(raw, C.IMG_DIR)
    C.write_coco(raw, out_path)
    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    by = lambda files: dict(collections.Counter(re.sub(r"_\d+\.jpg", "", f) for f in files))  # noqa
    out = {"note": "Every label added after v4clean was reviewed by eye on contact sheets (green "
                   "box, red mask outline). Older labels: random 40-image sample in "
                   "old_sample_review.json.",
           "clean_file": out_path, "sha256": sha, "images": len(raw["images"]),
           "annotations": len(raw["annotations"]),
           "class_counts_images_annotations": C.class_counts(raw),
           "old_sample_dropped": old, "new_labelled": len(new), "new_by_class": by(new),
           "dropped": len(drop),
           "dropped_by_class": by(drop), "detector_missed": log["missed"],
           "reviews": [{"file": f, "verdict": "drop", "reason": r}
                       for f, r in sorted(drop.items())]
           + [{"file": f, "verdict": "keep", "prompt": x["prompt"], "box_thr": x["box_thr"]}
              for f, x in new.items() if f not in drop]}
    json.dump(out, open(review_path, "w"), indent=1)
    print(sha, out["images"], out["annotations"], out["class_counts_images_annotations"],
          out["new_by_class"], out["dropped_by_class"])


if __name__ == "__main__":
    main()
