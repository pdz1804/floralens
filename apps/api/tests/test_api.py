"""API tests: /health and /api/search against the real gallery index."""
import base64
import io
import json
import re
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
    # Additive description field (curated botanical description; display only).
    assert "description" in body["results"][0]
    assert isinstance(body["results"][0]["description"], str)
    assert len(body["results"][0]["description"]) > 0


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


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_specimen_image_unknown_id_404():
    # Resolving a specimen id loads the gallery index first, so this needs the
    # (gitignored) embeddings cache present.
    resp = client.get("/api/specimen/does-not-exist/image")
    assert resp.status_code == 404


def test_specimen_image_rejects_traversal_id():
    # A crafted id that is not a known specimen must never resolve to a file.
    resp = client.get("/api/specimen/..%2F..%2F..%2Fetc%2Fpasswd/image")
    assert resp.status_code in (404, 400)


def _tiny_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (64, 48), color=(180, 90, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_preprocess_preview_returns_steps_and_images():
    resp = client.post(
        "/api/preprocess-preview",
        files={"file": ("query.jpg", io.BytesIO(_tiny_jpeg_bytes()), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["steps"]) > 0
    for step in body["steps"]:
        assert step["name"] and step["description"]
    assert isinstance(body["before_png_b64"], str) and len(body["before_png_b64"]) > 0
    assert isinstance(body["after_png_b64"], str) and len(body["after_png_b64"]) > 0
    # Both must decode as valid base64.
    base64.b64decode(body["before_png_b64"], validate=True)
    base64.b64decode(body["after_png_b64"], validate=True)


def test_preprocess_preview_per_step_filmstrip_has_valid_images():
    # Each step must carry its own after-step image (the UI filmstrip), same
    # exact shape as build_preprocess_preview: {name, description, image_png_b64}.
    resp = client.post(
        "/api/preprocess-preview",
        files={"file": ("query.jpg", io.BytesIO(_tiny_jpeg_bytes()), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    expected_step_names = {
        "exif_autoorient", "convert_rgb", "resize", "center_crop", "white_balance", "clahe",
    }
    assert {s["name"] for s in body["steps"]} == expected_step_names
    for step in body["steps"]:
        assert isinstance(step["image_png_b64"], str) and len(step["image_png_b64"]) > 0
        decoded = base64.b64decode(step["image_png_b64"], validate=True)
        # Must be a real PNG (magic bytes), not an empty/garbage payload.
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_preprocess_preview_rejects_missing_file():
    resp = client.post("/api/preprocess-preview")
    assert resp.status_code == 422  # FastAPI validation: required 'file' field missing


def test_search_rejects_decompression_bomb():
    # A tiny uniform PNG that decodes to >40MP must be rejected before load(),
    # not allowed to allocate hundreds of MB (OOM DoS).
    from apps.api.app import search_service

    search_service.reset_gallery_store_cache()
    bomb = Image.new("RGB", (7000, 7000), (10, 20, 30))  # 49MP, compresses tiny
    buf = io.BytesIO()
    bomb.save(buf, format="PNG")
    resp = client.post(
        "/api/search", files={"file": ("bomb.png", io.BytesIO(buf.getvalue()), "image/png")}
    )
    assert resp.status_code == 400


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_gallery_store_rejects_preprocessing_mismatch(monkeypatch):
    # A stale cache built with an OLDER preprocessing (backbone name unchanged)
    # must fail loud — query/gallery would otherwise diverge silently.
    from apps.api.app import search_service

    search_service.reset_gallery_store_cache()
    monkeypatch.setattr(search_service, "preprocess_fingerprint", lambda: "different_prep")
    with pytest.raises(RuntimeError, match="preprocessing mismatch"):
        search_service.get_gallery_store()
    search_service.reset_gallery_store_cache()


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_gallery_store_rejects_backbone_mismatch(monkeypatch):
    # If the active backbone differs from the one the gallery was embedded with,
    # the store build must fail loud (same-dim swap would silently corrupt search).
    from apps.api.app import search_service

    search_service.reset_gallery_store_cache()
    monkeypatch.setattr(search_service, "backbone_name", lambda: "some_other_backbone")
    with pytest.raises(RuntimeError, match="backbone mismatch"):
        search_service.get_gallery_store()
    search_service.reset_gallery_store_cache()


def test_pipeline_snapshot_returns_required_keys():
    resp = client.get("/api/pipeline")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("dataset", "preprocessing", "backbone", "eval", "calibration", "promotion", "model_version"):
        assert key in body
    assert len(body["preprocessing"]) > 0
    assert body["backbone"]["frozen"] is True


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_galaxy_returns_points_with_coords_and_color():
    resp = client.get("/api/galaxy")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == len(body["points"])
    assert body["count"] >= 100
    hex_color_re = re.compile(r"^#[0-9a-f]{6}$")
    for p in body["points"]:
        assert isinstance(p["specimen_id"], str) and p["specimen_id"]
        for axis in ("x", "y", "z"):
            assert isinstance(p[axis], (int, float))
            assert -1.0 - 1e-6 <= p[axis] <= 1.0 + 1e-6
        assert isinstance(p["label"], int)
        assert isinstance(p["label_name"], str) and p["label_name"]
        assert hex_color_re.match(p["color"])


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_galaxy_colors_are_stable_per_species():
    resp = client.get("/api/galaxy")
    assert resp.status_code == 200
    points = resp.json()["points"]
    color_by_label: dict[str, str] = {}
    for p in points:
        seen = color_by_label.setdefault(p["label_name"], p["color"])
        assert seen == p["color"], f"{p['label_name']} has inconsistent colors"
