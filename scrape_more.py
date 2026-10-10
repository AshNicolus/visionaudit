"""Top up one class (default drone) beyond the first scrape, same licence filter."""
import argparse

import collect_and_annotate as C

ap = argparse.ArgumentParser()
ap.add_argument("--cls", default="drone")
ap.add_argument("--target", type=int, default=130)
ap.add_argument("--budget-s", type=int, default=900)
a = ap.parse_args()
C.CLASSES = {a.cls: C.CLASSES[a.cls]}
recs = C.scrape_images(a.target, budget_s=a.budget_s)
print(len(recs))
