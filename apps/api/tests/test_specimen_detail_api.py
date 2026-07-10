"""Specimen-detail tests (`specimen_service.get_specimen_detail` +
`GET /api/specimens/{specimen_id}`). Runs against the real embeddings-cache
metadata when present; skipped otherwise (like the other ML-dependent tests) so
the suite never fails just because the pipeline hasn't been built here. The
missing-cache -> 503 and unknown-id -> 404 contracts are checked without the
cache via monkeypatch."""
import json
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.app import specimen_service
from apps.api.app.main import app

EMBEDDINGS_CACHE = Path("ml/data/embeddings_cache/metadata.json")

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Isolate the module-level specimens cache from other test modules."""
    specimen_service.reset_specimen_detail_cache()
    yield
    specimen_service.reset_specimen_detail_cache()


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_specimen_detail_returns_expected_fields_for_known_id():
    metadata = json.loads(EMBEDDINGS_CACHE.read_text(encoding="utf-8"))
    specimen_id, meta = next(iter(metadata["specimens"].items()))

    resp = client.get(f"/api/specimens/{specimen_id}")
    assert resp.status_code == 200
    body = resp.json()

    assert body["specimen_id"] == specimen_id
    assert body["label"] == meta["label"]
    assert body["label_name"] == meta["label_name"]
    assert body["split"] == meta["split"]
    assert body["image_url"] == f"/api/specimen/{specimen_id}/image"


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_specimen_detail_image_url_resolves_to_the_thumbnail_endpoint():
    metadata = json.loads(EMBEDDINGS_CACHE.read_text(encoding="utf-8"))
    specimen_id = next(
        sid for sid, m in metadata["specimens"].items() if m["split"] == "gallery"
    )
    body = client.get(f"/api/specimens/{specimen_id}").json()
    # The advertised image_url must actually serve an image.
    img_resp = client.get(body["image_url"])
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] == "image/jpeg"


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_specimen_detail_unknown_id_returns_404():
    resp = client.get("/api/specimens/does-not-exist")
    assert resp.status_code == 404


def test_specimen_detail_returns_503_when_metadata_missing(tmp_path, monkeypatch):
    # Point the service at an empty dir: metadata.json is absent, so the
    # endpoint must surface a clean 503 rather than crashing.
    monkeypatch.setattr(
        specimen_service, "settings", types.SimpleNamespace(embeddings_cache_dir=str(tmp_path))
    )
    specimen_service.reset_specimen_detail_cache()
    resp = client.get("/api/specimens/anything")
    assert resp.status_code == 503
