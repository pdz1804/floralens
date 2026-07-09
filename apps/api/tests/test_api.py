"""API tests: /health and /api/search against the real gallery index."""
import base64
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from apps.api.app.main import app
from apps.api.app.search_service import (
    reset_gallery_store_cache,
    reset_specimen_image_index_cache,
)

EMBEDDINGS_CACHE = Path("ml/data/embeddings_cache/metadata.json")

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_version" in body


def test_api_health_ok():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_search_rejects_missing_body():
    resp = client.post("/api/search")
    assert resp.status_code == 400


def test_search_rejects_invalid_base64():
    resp = client.post("/api/search", json={"image_base64": "not-valid-base64!!"})
    assert resp.status_code == 400


def test_search_rejects_undecodable_image_bytes():
    resp = client.post(
        "/api/search", json={"image_base64": base64.b64encode(b"not an image").decode()}
    )
    assert resp.status_code == 400


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_search_multipart_returns_top1_correct_class():
    reset_gallery_store_cache()
    metadata = json.loads(EMBEDDINGS_CACHE.read_text(encoding="utf-8"))
    # Pick a gallery-indexed specimen: querying with its own (near-identical)
    # image should surface its own class at rank 1.
    gallery_specimen = next(
        (sid, m) for sid, m in metadata["specimens"].items() if m["split"] == "gallery"
    )
    specimen_id, meta = gallery_specimen
    img_bytes = Path(meta["image_path"]).read_bytes()

    resp = client.post(
        "/api/search",
        files={"file": ("query.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) > 0
    assert body["results"][0]["label_name"] == meta["label_name"]
    assert body["results"][0]["score"] > 0.99  # querying with the exact same image


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_search_base64_json_path_works():
    reset_gallery_store_cache()
    metadata = json.loads(EMBEDDINGS_CACHE.read_text(encoding="utf-8"))
    specimen_id, meta = next(
        (sid, m) for sid, m in metadata["specimens"].items() if m["split"] == "gallery"
    )
    with Image.open(meta["image_path"]) as img:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode()

    resp = client.post("/api/search", json={"image_base64": encoded})
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["label_name"] == meta["label_name"]


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_specimen_image_serves_known_id():
    reset_specimen_image_index_cache()
    metadata = json.loads(EMBEDDINGS_CACHE.read_text(encoding="utf-8"))
    specimen_id = next(
        sid for sid, m in metadata["specimens"].items() if m["split"] == "gallery"
    )
    resp = client.get(f"/api/specimen/{specimen_id}/image")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 0


def test_specimen_image_unknown_id_404():
    resp = client.get("/api/specimen/does-not-exist/image")
    assert resp.status_code == 404


def test_specimen_image_rejects_traversal_id():
    # A crafted id that is not a known specimen must never resolve to a file.
    resp = client.get("/api/specimen/..%2F..%2F..%2Fetc%2Fpasswd/image")
    assert resp.status_code in (404, 400)
