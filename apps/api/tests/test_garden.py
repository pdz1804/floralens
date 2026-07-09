"""My Garden CRUD tests (PRD Phase 7) — add/list/remove + persistence, via
`GET/POST/DELETE /api/garden`. Runs against an isolated tmp SQLite file per
test (never touches a real garden.db)."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.app import garden_service
from apps.api.app.main import app

EMBEDDINGS_CACHE = Path("ml/data/embeddings_cache/metadata.json")

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_garden_db(tmp_path, monkeypatch):
    """Every test gets its own empty SQLite file — no cross-test bleed, and
    the real garden.db (if any) is never touched by the test suite."""
    monkeypatch.setattr(garden_service, "_DB_PATH", tmp_path / "garden.db")
    yield


def test_list_garden_empty_by_default():
    resp = client.get("/api/garden")
    assert resp.status_code == 200
    assert resp.json() == {"items": []}


def test_add_unknown_specimen_rejected():
    resp = client.post("/api/garden", json={"specimen_id": "not-a-real-specimen"})
    assert resp.status_code == 404


def test_add_rejects_empty_specimen_id():
    resp = client.post("/api/garden", json={"specimen_id": ""})
    assert resp.status_code == 422  # pydantic min_length=1


def test_remove_unknown_specimen_404():
    resp = client.delete("/api/garden/not-saved")
    assert resp.status_code == 404


def test_garden_crud_roundtrip(monkeypatch):
    # Bypass the real dataset-metadata lookup so this test never depends on
    # the embeddings cache being built.
    monkeypatch.setattr(
        garden_service, "resolve_label_name", lambda sid: "Test Rose" if sid == "sid-1" else None
    )

    add_resp = client.post("/api/garden", json={"specimen_id": "sid-1"})
    assert add_resp.status_code == 201
    body = add_resp.json()
    assert body["specimen_id"] == "sid-1"
    assert body["label_name"] == "Test Rose"
    assert body["saved_at"]

    list_resp = client.get("/api/garden")
    assert list_resp.status_code == 200
    items = list_resp.json()["items"]
    assert len(items) == 1
    assert items[0]["specimen_id"] == "sid-1"

    # Saving the same specimen again is an idempotent refresh, not a duplicate.
    client.post("/api/garden", json={"specimen_id": "sid-1"})
    assert len(client.get("/api/garden").json()["items"]) == 1

    remove_resp = client.delete("/api/garden/sid-1")
    assert remove_resp.status_code == 204
    assert client.get("/api/garden").json()["items"] == []

    # Removing again (already gone) is a clean 404, not a crash.
    assert client.delete("/api/garden/sid-1").status_code == 404


def test_garden_persists_to_disk(monkeypatch):
    # Proves durability: a fresh call into the service layer (simulating a
    # new request after a restart) reads the same on-disk SQLite file.
    monkeypatch.setattr(garden_service, "resolve_label_name", lambda sid: "Test Tulip")
    client.post("/api/garden", json={"specimen_id": "sid-2"})

    items = garden_service.list_specimens()
    assert any(i.specimen_id == "sid-2" and i.label_name == "Test Tulip" for i in items)


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_add_real_gallery_specimen_resolves_label_name():
    garden_service.reset_specimen_catalog_cache()
    metadata = json.loads(EMBEDDINGS_CACHE.read_text(encoding="utf-8"))
    specimen_id, meta = next(
        (sid, m) for sid, m in metadata["specimens"].items() if m["split"] == "gallery"
    )

    resp = client.post("/api/garden", json={"specimen_id": specimen_id})
    assert resp.status_code == 201
    assert resp.json()["label_name"] == meta["label_name"]

    garden_service.reset_specimen_catalog_cache()  # don't leak into other test modules
