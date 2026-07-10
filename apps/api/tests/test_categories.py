"""Categories aggregation tests (`categories_service.get_categories` +
`GET /api/categories`). Runs against the real embeddings-cache metadata when
present; skipped otherwise (like the ML-dependent garden test) so the suite
never fails just because the pipeline hasn't been built in this environment."""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.app import categories_service
from apps.api.app.main import app

EMBEDDINGS_CACHE = Path("ml/data/embeddings_cache/metadata.json")

client = TestClient(app)

_HEX = re.compile(r"^#[0-9a-f]{6}$")


@pytest.fixture(autouse=True)
def _clear_cache():
    """Isolate the module-level categories cache from other test modules."""
    categories_service.reset_categories_cache()
    yield
    categories_service.reset_categories_cache()


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_get_categories_returns_102_valid_entries():
    entries = categories_service.get_categories()

    # Oxford-102 → 102 distinct species, sorted by name, no duplicates.
    assert len(entries) == 102
    names = [e["label_name"] for e in entries]
    assert names == sorted(names)
    assert len(set(names)) == 102

    for e in entries:
        assert isinstance(e["label"], int)
        assert isinstance(e["label_name"], str) and e["label_name"]
        assert isinstance(e["gallery_count"], int) and e["gallery_count"] >= 0
        assert isinstance(e["total_count"], int) and e["total_count"] > 0
        assert e["gallery_count"] <= e["total_count"]
        assert isinstance(e["specimen_id"], str) and e["specimen_id"]
        assert _HEX.match(e["color"]), f"bad color for {e['label_name']}: {e['color']}"


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_representative_specimen_id_is_a_real_gallery_specimen():
    import json

    metadata = json.loads(EMBEDDINGS_CACHE.read_text(encoding="utf-8"))
    specimens = metadata["specimens"]
    for e in categories_service.get_categories():
        meta = specimens[e["specimen_id"]]
        assert meta["label_name"] == e["label_name"]
        # A species with any gallery specimen must be represented by one.
        if e["gallery_count"] > 0:
            assert meta["split"] == "gallery"


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_endpoint_shape_and_total_specimens_sums_correctly():
    resp = client.get("/api/categories")
    assert resp.status_code == 200
    body = resp.json()

    assert body["count"] == len(body["categories"]) == 102
    assert body["total_specimens"] == sum(c["total_count"] for c in body["categories"])
    # 3268 specimens across all 102 species in the Oxford-102 build.
    assert body["total_specimens"] == 3268


def test_endpoint_returns_503_when_metadata_missing(tmp_path, monkeypatch):
    # Point the service at an empty dir: metadata.json is absent, so the
    # endpoint must surface a clean 503 rather than crashing or caching empty.
    import types

    monkeypatch.setattr(
        categories_service, "settings", types.SimpleNamespace(embeddings_cache_dir=str(tmp_path))
    )
    categories_service.reset_categories_cache()
    resp = client.get("/api/categories")
    assert resp.status_code == 503
