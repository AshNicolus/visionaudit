# Augle AI Intern Assignment

Scrape, auto-label, train and audit an image dataset end to end.

Setup, architecture notes and a sample run will be added as each phase lands.

## Layout

| File | Purpose |
| --- | --- |
| `collect_and_annotate.py` | Playwright scraper and zero-shot annotation (COCO JSON) |
| `dataset.py` | PyTorch Dataset and OpenCV preprocessing |
| `train.py` | Custom CNN and training loop |
| `llm_auditor.py` | RAG auditor with tool calling |
| `tests/` | Geometry and schema tests |
