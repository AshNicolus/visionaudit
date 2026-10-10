"""Offline tests for the auditor tools, chunking, retrieval and the tool-use loop."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from chromadb import EmbeddingFunction

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm_auditor as la  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def coco():
    return la.load_coco(FIX / "annotations.json")


@pytest.fixture
def metrics():
    return la.load_metrics(FIX / "metrics.jsonl")


class HashEmbedder(EmbeddingFunction):
    """Deterministic bag-of-words embedding so tests need no model download."""

    def __init__(self):
        pass

    def __call__(self, input):
        out = []
        for text in input:
            v = [0.0] * 64
            for w in text.lower().replace(":", " ").replace(",", " ").split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
            out.append(v)
        return out

    @staticmethod
    def name():
        return "hash-test"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return HashEmbedder()


def test_find_small_boxes_fraction(coco):
    res = la.find_small_boxes(coco, max_area_fraction=0.001)
    assert res["total_boxes"] == 18
    assert res["total_matches"] == 2  # 20 px and 18 px of 20000
    assert res["examples"][0]["area_px"] <= res["examples"][1]["area_px"]


def test_find_small_boxes_pixels_and_default(coco):
    assert la.find_small_boxes(coco, max_area_px=64)["total_matches"] == 3
    assert la.find_small_boxes(coco)["total_matches"] == 2


def test_class_balance(coco):
    res = la.class_balance_from(coco)
    rows = [(r["name"], r["count"]) for r in res["classes"]]
    assert rows == [("cat", 12), ("dog", 5), ("bird", 1)]
    assert res["imbalance_ratio"] == 12.0
    assert res["classes_with_no_boxes"] == []


def test_class_balance_empty_class():
    data = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "a"}]}
    res = la.class_balance_from(data)
    assert res["imbalance_ratio"] is None and res["classes_with_no_boxes"] == ["a"]


def test_bbox_area_stats(coco):
    res = la.bbox_area_stats(coco)
    assert res["overall"]["n"] == 18
    assert set(res["per_class"]) == {"cat", "dog", "bird"}
    assert res["overall"]["min"] <= res["overall"]["median"] <= res["overall"]["max"]


def test_run_tool_unknown(coco):
    with pytest.raises(KeyError):
        la.run_tool(coco, "nope", {})


def test_chunks(coco, metrics):
    chunks = la.build_chunks(coco, metrics)
    kinds = [c["kind"] for c in chunks]
    assert kinds.count("image") == 8 and kinds.count("class") == 3 and kinds.count("epoch") == 5
    assert len({c["id"] for c in chunks}) == len(chunks)


def test_retrieval_roundtrip(coco, metrics, tmp_path):
    col = la.ingest(coco, metrics, tmp_path / "db", embedder=HashEmbedder())
    assert col.count() == 16
    docs = la.retrieve(col, "Epoch 5 val_loss", k=3)
    assert any(d.startswith("Epoch 5") for d in docs)


def test_missing_metrics_is_empty(tmp_path):
    assert la.load_metrics(tmp_path / "none.jsonl") == []


def test_no_key_exits_with_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        la.get_client()
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_tool_loop_with_fake_client(coco):
    tool_block = SimpleNamespace(type="tool_use", id="t1", name="class_balance", input={})
    text_block = SimpleNamespace(type="text", text="Imbalance is 12x.")
    replies = [
        SimpleNamespace(stop_reason="tool_use", content=[tool_block]),
        SimpleNamespace(stop_reason="end_turn", content=[text_block]),
    ]
    seen = []

    class FakeMessages:
        def create(self, **kw):
            seen.append(json.loads(json.dumps(kw["messages"], default=lambda o: "obj")))
            return replies.pop(0)

    client = SimpleNamespace(messages=FakeMessages())
    out = la.answer(client, coco, None, "balance?", log=lambda *_: None)
    assert out == "Imbalance is 12x."
    result = seen[1][-1]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t1"
    assert '"imbalance_ratio": 12.0' in result["content"]


def test_real_embedder_if_available(coco, metrics, tmp_path):
    try:
        emb = la.make_embedder()
        emb(["warmup"])
    except Exception:
        pytest.skip("sentence-transformers model not available offline")
    col = la.ingest(coco, metrics, tmp_path / "db", embedder=emb)
    assert la.retrieve(col, "which class has the fewest boxes", k=3)
