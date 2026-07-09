"""End-to-end smoke test: embed a handful of real dataset images, index them,
query, and assert sane top-K behavior. Uses the real materialized split
manifest produced by `ml/scripts/build_splits.py` (skips if not present).
"""
import json
from pathlib import Path

import pytest
from PIL import Image

from ml.embeddings.backbone import embed_image
from ml.index.vector_store import VectorStore

MANIFEST_PATH = Path("ml/data/manifests/split_manifest.json")


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="split manifest not built yet")
def test_embed_index_query_smoke():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    gallery_records = [r for r in manifest["records"] if r["split"] == "gallery"]

    # Take 3 images each from 5 distinct classes for the gallery, and hold
    # out one extra image per class as a query — a small but real smoke test.
    by_class: dict[int, list[dict]] = {}
    for r in gallery_records:
        by_class.setdefault(r["label"], []).append(r)
    chosen_classes = [c for c, recs in by_class.items() if len(recs) >= 4][:5]
    assert len(chosen_classes) == 5, "expected at least 5 classes with >=4 gallery images"

    store = VectorStore(model_version="smoke-test")
    queries = []
    for cls in chosen_classes:
        recs = sorted(by_class[cls], key=lambda r: r["id"])[:4]
        gallery_recs, query_rec = recs[:3], recs[3]
        for rec in gallery_recs:
            with Image.open(rec["image_path"]) as img:
                vector = embed_image(img)
            store.add(rec["id"], vector, {"label": rec["label"], "label_name": rec["label_name"]})
        with Image.open(query_rec["image_path"]) as img:
            query_vector = embed_image(img)
        queries.append((query_rec, query_vector))

    assert len(store) == 15

    hits = 0
    for query_rec, query_vector in queries:
        results = store.query(query_vector, top_k=3)
        assert len(results) == 3
        assert results[0].score >= results[1].score >= results[2].score
        if results[0].metadata["label"] == query_rec["label"]:
            hits += 1

    # Sanity bound, not a strict accuracy claim: a frozen zero-shot backbone
    # over 5 classes / tiny gallery should get most top-1s right.
    assert hits >= 3, f"expected at least 3/5 top-1 hits from a sane embedding space, got {hits}"
