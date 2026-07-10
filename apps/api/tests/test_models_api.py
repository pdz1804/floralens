"""Model-registry tests (`models_service.get_models` + `GET /api/models`).

The endpoint is read-only/best-effort and does NOT need the (gitignored)
embeddings cache or the backbone, so the core shape tests run everywhere; a
"not-built" test points the service at an empty models dir and asserts a
sensible baseline-only response instead of a crash."""
import types

from fastapi.testclient import TestClient

from apps.api.app import models_service
from apps.api.app.config import settings
from apps.api.app.main import app

client = TestClient(app)


def test_models_endpoint_shape_and_active_flag():
    resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()

    assert body["active"] == settings.model_version
    assert isinstance(body["models"], list) and body["models"]

    names = [m["name"] for m in body["models"]]
    # The synthetic baseline is always present and listed first.
    assert names[0] == "baseline"
    assert len(names) == len(set(names))  # no duplicate versions

    # Exactly the versions matching settings.model_version are marked active.
    active = [m for m in body["models"] if m["active"]]
    assert all(m["name"] == settings.model_version for m in active)
    assert len(active) <= 1

    for m in body["models"]:
        assert isinstance(m["name"], str) and m["name"]
        assert isinstance(m["active"], bool)
        # Optional, best-effort fields — present or null, never wrong-typed.
        assert m["backbone"] is None or isinstance(m["backbone"], str)
        assert m["metrics"] is None or isinstance(m["metrics"], dict)


def test_active_version_is_listed_and_flagged():
    body = client.get("/api/models").json()
    by_name = {m["name"]: m for m in body["models"]}
    # The active model_version must appear in the list and carry active=True
    # (the default finetuned candidate has an artifact dir; baseline is synthetic).
    assert settings.model_version in by_name
    assert by_name[settings.model_version]["active"] is True


def test_models_endpoint_baseline_only_when_no_artifacts(tmp_path, monkeypatch):
    # Point the service at an empty models dir: no ModelVersion artifacts exist,
    # so the endpoint must still return a sensible response (just the baseline),
    # never a 500. The embeddings cache is also absent under tmp_path.
    monkeypatch.setattr(
        models_service,
        "settings",
        types.SimpleNamespace(
            model_version="baseline",
            models_dir=str(tmp_path / "models"),
            embeddings_cache_dir=str(tmp_path / "cache"),
        ),
    )
    resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] == "baseline"
    assert [m["name"] for m in body["models"]] == ["baseline"]
    assert body["models"][0]["active"] is True
