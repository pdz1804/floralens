"""Embedding-endpoint tests (`POST /api/embed`). Input-validation (400) tests
run everywhere; the real embedding path is gated on the (gitignored) embeddings
cache like the search tests, since it loads the backbone/model and is only
meaningful in a built environment."""
import base64
import io
import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from apps.api.app.config import settings
from apps.api.app.main import app
from apps.api.app.search_service import reset_gallery_store_cache

EMBEDDINGS_CACHE = Path("ml/data/embeddings_cache/metadata.json")

client = TestClient(app)


def test_embed_rejects_missing_body():
    resp = client.post("/api/embed")
    assert resp.status_code == 400


def test_embed_rejects_invalid_base64():
    resp = client.post("/api/embed", json={"image_base64": "not-valid-base64!!"})
    assert resp.status_code == 400


def test_embed_rejects_undecodable_image_bytes():
    resp = client.post(
        "/api/embed", json={"image_base64": base64.b64encode(b"not an image").decode()}
    )
    assert resp.status_code == 400


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_embed_multipart_returns_normalized_vector():
    reset_gallery_store_cache()
    metadata = json.loads(EMBEDDINGS_CACHE.read_text(encoding="utf-8"))
    specimen_id, meta = next(
        (sid, m) for sid, m in metadata["specimens"].items() if m["split"] == "gallery"
    )
    img_bytes = Path(meta["image_path"]).read_bytes()

    resp = client.post(
        "/api/embed",
        files={"file": ("query.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["model_version"] == settings.model_version or body["model_version"] == "baseline"
    embedding = body["embedding"]
    assert isinstance(embedding, list) and len(embedding) == body["dim"] > 0
    assert all(isinstance(x, (int, float)) for x in embedding)
    # The returned vector must be (unit) L2-normalized.
    norm = math.sqrt(sum(x * x for x in embedding))
    assert norm == pytest.approx(1.0, abs=1e-4)


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_embed_base64_json_path_matches_multipart():
    reset_gallery_store_cache()
    metadata = json.loads(EMBEDDINGS_CACHE.read_text(encoding="utf-8"))
    specimen_id, meta = next(
        (sid, m) for sid, m in metadata["specimens"].items() if m["split"] == "gallery"
    )
    with Image.open(meta["image_path"]) as img:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
    raw = buf.getvalue()

    multipart = client.post(
        "/api/embed", files={"file": ("q.jpg", io.BytesIO(raw), "image/jpeg")}
    ).json()
    json_body = client.post(
        "/api/embed", json={"image_base64": base64.b64encode(raw).decode()}
    ).json()

    # Both input paths run the same deterministic embed -> identical dim/version
    # and near-identical vectors.
    assert multipart["dim"] == json_body["dim"]
    assert multipart["model_version"] == json_body["model_version"]
    for a, b in zip(multipart["embedding"], json_body["embedding"]):
        assert a == pytest.approx(b, abs=1e-5)
