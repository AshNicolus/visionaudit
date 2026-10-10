"""RAG data auditor: embeds COCO annotations and training metrics, answers with tool calling."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

MODEL = "claude-sonnet-5-5"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COCO = "data/coco/annotations.json"
DEFAULT_METRICS = "runs/metrics.jsonl"
DEFAULT_DB = "chroma_db"
MAX_TURNS = 8

SYSTEM_PROMPT = (
    "You audit an object-detection dataset and its training run. "
    "Use the retrieved context for facts about individual images, classes and epochs, "
    "and call tools for anything that needs counting or filtering over the whole dataset. "
    "Never guess numbers; quote them from context or tool output. "
    "If the evidence is missing, say so."
)


# --- loading ---

def load_coco(path):
    """Read a COCO JSON file into a dict."""
    return json.loads(Path(path).read_text())


def load_metrics(path):
    """Read one JSON object per line; a missing file gives an empty list."""
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# --- chunking ---

def _class_names(coco):
    return {c["id"]: c["name"] for c in coco.get("categories", [])}


def image_chunks(coco):
    """One document per image summarising its annotations."""
    names = _class_names(coco)
    by_image = {}
    for ann in coco.get("annotations", []):
        by_image.setdefault(ann["image_id"], []).append(ann)
    chunks = []
    for img in coco.get("images", []):
        anns = by_image.get(img["id"], [])
        counts = Counter(names.get(a["category_id"], "?") for a in anns)
        frac = [a["area"] / (img["width"] * img["height"]) for a in anns]
        text = (
            f"Image {img['file_name']} ({img['width']}x{img['height']}) has "
            f"{len(anns)} boxes: "
            + (", ".join(f"{n} x{c}" for n, c in sorted(counts.items())) or "none")
            + "."
        )
        if frac:
            text += f" Smallest box covers {min(frac):.2%} of the image, largest {max(frac):.2%}."
        chunks.append({"id": f"image-{img['id']}", "text": text, "kind": "image"})
    return chunks


def class_chunks(coco):
    """One document per class with count, share and mean box area."""
    stats = class_balance_from(coco)
    total = sum(r["count"] for r in stats["classes"]) or 1
    areas = {}
    for a in coco.get("annotations", []):
        areas.setdefault(a["category_id"], []).append(a["area"])
    ids = {v: k for k, v in _class_names(coco).items()}
    chunks = []
    for row in stats["classes"]:
        vals = areas.get(ids[row["name"]], [])
        mean = sum(vals) / len(vals) if vals else 0
        text = (
            f"Class {row['name']}: {row['count']} annotations "
            f"({row['count'] / total:.1%} of all), mean box area {mean:.0f} px."
        )
        chunks.append({"id": f"class-{row['name']}", "text": text, "kind": "class"})
    return chunks


def metric_chunks(metrics):
    """One document per epoch line."""
    chunks = []
    for m in metrics:
        text = (
            f"Epoch {m['epoch']}: train_loss {m['train_loss']}, val_loss {m['val_loss']}, "
            f"val_acc {m['val_acc']}, val_box_l1 {m['val_box_l1']}, lr {m['lr']}."
        )
        chunks.append({"id": f"epoch-{m['epoch']}", "text": text, "kind": "epoch"})
    return chunks


def build_chunks(coco, metrics):
    """All documents to index."""
    return image_chunks(coco) + class_chunks(coco) + metric_chunks(metrics)


# --- vector store ---

def make_embedder():
    """Local sentence-transformers embedding function for chromadb."""
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def ingest(coco, metrics, db_path=DEFAULT_DB, embedder=None, name="audit"):
    """Rebuild the persistent collection from the given data and return it."""
    import chromadb
    from chromadb.errors import NotFoundError

    client = chromadb.PersistentClient(path=str(db_path))
    try:
        client.delete_collection(name)
    except NotFoundError:  # collection did not exist yet
        pass
    col = client.create_collection(name, embedding_function=embedder or make_embedder())
    chunks = build_chunks(coco, metrics)
    if chunks:
        col.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"kind": c["kind"]} for c in chunks],
        )
    return col


def retrieve(col, query, k=5):
    """Top-k chunk texts for a natural-language query."""
    if col.count() == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(k, col.count()))
    return res["documents"][0]


# --- tools (pure Python, no API key needed) ---

def _image_sizes(coco):
    return {i["id"]: i["width"] * i["height"] for i in coco.get("images", [])}


def find_small_boxes(coco, max_area_fraction=None, max_area_px=None, limit=20):
    """Boxes below an area threshold: a fraction of the image, or absolute pixels."""
    if max_area_fraction is None and max_area_px is None:
        max_area_fraction = 0.001
    sizes = _image_sizes(coco)
    names = _class_names(coco)
    files = {i["id"]: i["file_name"] for i in coco.get("images", [])}
    hits = []
    for a in coco.get("annotations", []):
        frac = a["area"] / sizes[a["image_id"]]
        if max_area_fraction is not None and frac > max_area_fraction:
            continue
        if max_area_px is not None and a["area"] > max_area_px:
            continue
        hits.append({
            "annotation_id": a["id"],
            "image": files[a["image_id"]],
            "class": names.get(a["category_id"], "?"),
            "area_px": a["area"],
            "area_fraction": round(frac, 5),
        })
    hits.sort(key=lambda h: h["area_px"])
    by_class = dict(Counter(h["class"] for h in hits))
    return {"total_matches": len(hits), "total_boxes": len(coco.get("annotations", [])),
            "matches_by_class": by_class, "examples": hits[:limit]}


def class_balance_from(coco):
    """Per-class counts, sorted descending, plus imbalance ratio (max/min non-zero)."""
    names = _class_names(coco)
    counts = Counter(a["category_id"] for a in coco.get("annotations", []))
    rows = sorted(({"name": names[cid], "count": counts.get(cid, 0)} for cid in names),
                  key=lambda r: -r["count"])
    nonzero = [r["count"] for r in rows if r["count"] > 0]
    ratio = round(max(nonzero) / min(nonzero), 2) if nonzero else None
    empty = [r["name"] for r in rows if r["count"] == 0]
    return {"classes": rows, "imbalance_ratio": ratio, "classes_with_no_boxes": empty}


def bbox_area_stats(coco):
    """Overall and per-class box area stats as a fraction of image area."""
    sizes = _image_sizes(coco)
    names = _class_names(coco)
    per = {}
    for a in coco.get("annotations", []):
        cls = names.get(a["category_id"], "?")
        per.setdefault(cls, []).append(a["area"] / sizes[a["image_id"]])

    def summ(v):
        v = sorted(v)
        return {"n": len(v), "min": round(v[0], 5), "median": round(v[len(v) // 2], 5),
                "mean": round(sum(v) / len(v), 5), "max": round(v[-1], 5)}

    allv = [x for v in per.values() for x in v]
    return {"overall": summ(allv) if allv else None,
            "per_class": {k: summ(v) for k, v in sorted(per.items())}}


TOOL_SCHEMAS = [
    {
        "name": "find_small_boxes",
        "description": "Find bounding boxes smaller than a threshold. Give max_area_fraction "
                       "(fraction of image area, e.g. 0.001) or max_area_px (absolute pixels, "
                       "boxes with area <= this value). Returns match count, per-class counts "
                       "and the smallest examples.",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_area_fraction": {"type": "number",
                                      "description": "e.g. 0.001 for 0.1% of the image"},
                "max_area_px": {"type": "number",
                                "description": "box area in pixels, inclusive upper bound"},
                "limit": {"type": "integer", "description": "max examples to return"},
            },
        },
    },
    {
        "name": "class_balance",
        "description": "Per-class annotation counts and the imbalance ratio "
                       "(largest class / smallest non-empty class).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "bbox_area_stats",
        "description": "Min/median/mean/max box area as a fraction of image area, "
                       "overall and per class.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def run_tool(coco, name, args):
    """Dispatch a tool call by name; raises KeyError for unknown tools."""
    if name == "find_small_boxes":
        return find_small_boxes(coco, **args)
    if name == "class_balance":
        return class_balance_from(coco)
    if name == "bbox_area_stats":
        return bbox_area_stats(coco)
    raise KeyError(name)


# --- LLM loop ---

def get_client():
    """Anthropic client, or exit with a one-line message when no key is set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set; --demo/--query need it "
                 "(tools and retrieval work offline).")
    import anthropic

    return anthropic.Anthropic()


def execute_tool_use(coco, block, log):
    """Run one tool_use block and build its tool_result block."""
    try:
        out = run_tool(coco, block.name, block.input or {})
        content, is_error = json.dumps(out), False
    except Exception as exc:  # report to the model instead of crashing the loop
        content, is_error = f"{type(exc).__name__}: {exc}", True
    log(f"  [tool] {block.name}({json.dumps(block.input)}) -> {content[:300]}")
    return {"type": "tool_result", "tool_use_id": block.id, "content": content,
            "is_error": is_error}


def answer(client, coco, col, question, k=5, log=print, model=MODEL):
    """Retrieve context, then run the tool-use loop until the model stops."""
    context = retrieve(col, question, k) if col is not None else []
    log(f"  [retrieved {len(context)} chunks]")
    for c in context:
        log(f"    - {c}")
    prompt = "Context:\n" + "\n".join(f"- {c}" for c in context) + f"\n\nQuestion: {question}"
    messages = [{"role": "user", "content": prompt}]
    for _ in range(MAX_TURNS):
        resp = client.messages.create(
            model=model, max_tokens=4096, system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS, messages=messages,
        )
        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = [execute_tool_use(coco, b, log) for b in resp.content if b.type == "tool_use"]
            messages.append({"role": "user", "content": results})
            continue
        if resp.stop_reason not in ("end_turn", "stop_sequence"):
            log(f"  [stopped: {resp.stop_reason}]")
        return "".join(b.text for b in resp.content if b.type == "text")
    return "Stopped: tool loop exceeded the turn limit."


DEMO_QUESTIONS = [
    ("Tool scenario 1: small boxes",
     "How many bounding boxes cover less than 0.1% of their image, and which classes are they in?"),
    ("Tool scenario 2: class imbalance",
     "Is the dataset class-balanced? Give counts and the imbalance ratio, "
     "and say what it means for training."),
    ("Pure RAG: training metrics",
     "How did validation loss and accuracy change across epochs, "
     "and is there any sign of overfitting?"),
]


def run_demo(client, coco, col, log=print):
    """Run the fixed demo questions."""
    for title, q in DEMO_QUESTIONS:
        log(f"\n=== {title} ===\nQ: {q}")
        log("A: " + answer(client, coco, col, q, log=log))


# --- CLI ---

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco", default=DEFAULT_COCO)
    p.add_argument("--metrics", default=DEFAULT_METRICS)
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--demo", action="store_true", help="run the scripted demo")
    p.add_argument("--query", help="ask one question")
    p.add_argument("-k", type=int, default=5, help="chunks to retrieve")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not (args.demo or args.query):
        sys.exit("Pass --demo or --query '...'.")
    client = get_client()
    coco = load_coco(args.coco)
    col = ingest(coco, load_metrics(args.metrics), args.db)
    if args.demo:
        run_demo(client, coco, col)
    else:
        print(answer(client, coco, col, args.query, k=args.k))


if __name__ == "__main__":
    main()
