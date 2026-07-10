"""Per-user auth scaffold tests (backend-only, default-off; see auth.py):
`resolve_user` / `issue_token`, the `/api/auth/*` dev-scaffold endpoints, and
per-user data isolation in the garden store (garden_service.py) + the
assistant's memory namespace (assistant_service.py / memory_service.py).

Every test either leaves `FLORALENS_JWT_SECRET` unset (auth off -- proves
today's single-user behavior is byte-for-byte unchanged) or sets it via
`monkeypatch` (auto-restored) to exercise the per-user path. Uses an isolated
tmp SQLite file per test (same convention as test_garden.py) so the real
garden.db is never touched, and the shared naturalist in_memory bucket is
cleared before/after (same convention as test_memory.py).
"""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest
from agent_core import Scope
from fastapi.testclient import TestClient
from starlette.requests import Request

from apps.api.app import assistant_service, garden_service
from apps.api.app.auth import DEFAULT_USER, issue_token, resolve_user
from apps.api.app.main import app, assistant_registries
from apps.api.app.memory_service import memory_scope_and_namespace

EMBEDDINGS_CACHE = Path("ml/data/embeddings_cache/metadata.json")

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_garden_db(tmp_path, monkeypatch):
    """Every test gets its own empty SQLite file -- matches test_garden.py's
    convention; the real garden.db is never touched by this suite."""
    monkeypatch.setattr(garden_service, "_DB_PATH", tmp_path / "garden.db")
    yield


@pytest.fixture(autouse=True)
def _auth_off_by_default(monkeypatch):
    """Every test starts from auth-off; tests that need auth on set the env
    var themselves via monkeypatch (auto-restored at teardown either way)."""
    monkeypatch.delenv("FLORALENS_JWT_SECRET", raising=False)
    yield


async def _clear_namespace_bucket(namespace: str) -> None:
    provider = assistant_registries.memory.get("in_memory")
    scope = Scope("user")
    items = await provider.all(scope, namespace)
    ids = [it.id for it in items if it.id is not None]
    if ids:
        await provider.delete(scope, namespace, ids)


@pytest.fixture(autouse=True)
def clean_memory_buckets():
    """The naturalist manifest's default namespace ("floralens") plus the
    per-user buckets this file exercises ("floralens:alice",
    "floralens:bob") -- cleared before/after so this file never bleeds into
    test_memory.py or between its own tests."""
    namespaces = ["floralens", "floralens:alice", "floralens:bob"]

    async def _clear_all():
        for ns in namespaces:
            await _clear_namespace_bucket(ns)

    asyncio.run(_clear_all())
    yield
    asyncio.run(_clear_all())


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _request_with_bearer(token: str | None) -> Request:
    headers = [(b"authorization", f"Bearer {token}".encode())] if token else []
    return Request({"type": "http", "headers": headers})


# --------------------------------------------------------------------------- #
# Auth OFF (no FLORALENS_JWT_SECRET) -- single-user behavior, unchanged.
# --------------------------------------------------------------------------- #
def test_resolve_user_returns_default_user_when_secret_unset():
    request = Request({"type": "http", "headers": []})
    assert asyncio.run(resolve_user(request)) == DEFAULT_USER == "public"


def test_auth_me_returns_default_user_when_secret_unset():
    resp = client.get("/api/auth/me")
    assert resp.status_code == 200
    assert resp.json() == {"user_id": "public"}


def test_garden_works_without_a_token_when_auth_off(monkeypatch):
    monkeypatch.setattr(
        garden_service, "resolve_label_name", lambda sid: "Test Rose" if sid == "sid-1" else None
    )
    add_resp = client.post("/api/garden", json={"specimen_id": "sid-1"})
    assert add_resp.status_code == 201

    items = client.get("/api/garden").json()["items"]
    assert len(items) == 1
    assert items[0]["specimen_id"] == "sid-1"


def test_auth_token_endpoint_404s_when_auth_not_configured():
    resp = client.post("/api/auth/token", json={"user_id": "alice"})
    assert resp.status_code in (404, 501)


# --------------------------------------------------------------------------- #
# Auth ON (FLORALENS_JWT_SECRET set).
# --------------------------------------------------------------------------- #
def test_issue_token_round_trips_through_resolve_user(monkeypatch):
    monkeypatch.setenv("FLORALENS_JWT_SECRET", "test-secret")
    token = issue_token("alice")
    request = _request_with_bearer(token)
    assert asyncio.run(resolve_user(request)) == "alice"


def test_resolve_user_rejects_missing_token_when_auth_on(monkeypatch):
    monkeypatch.setenv("FLORALENS_JWT_SECRET", "test-secret")
    request = _request_with_bearer(None)
    with pytest.raises(Exception) as exc_info:
        asyncio.run(resolve_user(request))
    assert getattr(exc_info.value, "status_code", None) == 401

    resp = client.get("/api/auth/me")
    assert resp.status_code == 401


def test_resolve_user_rejects_invalid_token_when_auth_on(monkeypatch):
    monkeypatch.setenv("FLORALENS_JWT_SECRET", "test-secret")
    resp = client.get("/api/auth/me", headers=_auth_header("not-a-real-jwt"))
    assert resp.status_code == 401


def test_resolve_user_rejects_wrong_signature_token_when_auth_on(monkeypatch):
    monkeypatch.setenv("FLORALENS_JWT_SECRET", "secret-a")
    forged = issue_token("alice")  # signed with secret-a

    monkeypatch.setenv("FLORALENS_JWT_SECRET", "secret-b")  # rotate the secret
    resp = client.get("/api/auth/me", headers=_auth_header(forged))
    assert resp.status_code == 401


def test_resolve_user_rejects_expired_token_when_auth_on(monkeypatch):
    monkeypatch.setenv("FLORALENS_JWT_SECRET", "test-secret")
    expired = issue_token("alice", expires_in_s=-10)
    resp = client.get("/api/auth/me", headers=_auth_header(expired))
    assert resp.status_code == 401


def test_auth_me_with_valid_token_returns_the_sub_claim(monkeypatch):
    monkeypatch.setenv("FLORALENS_JWT_SECRET", "test-secret")
    token = issue_token("alice")
    resp = client.get("/api/auth/me", headers=_auth_header(token))
    assert resp.status_code == 200
    assert resp.json() == {"user_id": "alice"}


def test_auth_token_endpoint_issues_a_usable_token_when_configured(monkeypatch):
    monkeypatch.setenv("FLORALENS_JWT_SECRET", "test-secret")
    resp = client.post("/api/auth/token", json={"user_id": "alice"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["token"]

    me = client.get("/api/auth/me", headers=_auth_header(body["token"]))
    assert me.status_code == 200
    assert me.json() == {"user_id": "alice"}


def test_issue_token_raises_without_a_configured_secret():
    with pytest.raises(RuntimeError):
        issue_token("alice")


# --------------------------------------------------------------------------- #
# Per-user data isolation: garden (SQLite, composite (user_id, specimen_id)
# primary key).
# --------------------------------------------------------------------------- #
def test_garden_is_isolated_per_user(monkeypatch):
    monkeypatch.setenv("FLORALENS_JWT_SECRET", "test-secret")
    monkeypatch.setattr(garden_service, "resolve_label_name", lambda sid: "Test Rose")

    alice_headers = _auth_header(issue_token("alice"))
    bob_headers = _auth_header(issue_token("bob"))

    add_resp = client.post("/api/garden", json={"specimen_id": "sid-shared"}, headers=alice_headers)
    assert add_resp.status_code == 201

    # Bob doesn't see alice's specimen...
    assert client.get("/api/garden", headers=bob_headers).json()["items"] == []
    # ...and can independently save the SAME specimen_id (composite PK).
    bob_add = client.post("/api/garden", json={"specimen_id": "sid-shared"}, headers=bob_headers)
    assert bob_add.status_code == 201

    assert len(client.get("/api/garden", headers=alice_headers).json()["items"]) == 1
    assert len(client.get("/api/garden", headers=bob_headers).json()["items"]) == 1

    # Alice deleting hers doesn't affect bob's independent copy.
    del_resp = client.delete("/api/garden/sid-shared", headers=alice_headers)
    assert del_resp.status_code == 204
    assert client.get("/api/garden", headers=alice_headers).json()["items"] == []
    assert len(client.get("/api/garden", headers=bob_headers).json()["items"]) == 1


# --------------------------------------------------------------------------- #
# Per-user data isolation: assistant memory namespace.
# --------------------------------------------------------------------------- #
def test_default_user_memory_namespace_matches_todays_single_bucket():
    """With auth off (DEFAULT_USER), the namespace must be the exact string
    the naturalist manifest declares -- unchanged from before this scaffold,
    so test_memory.py's assertions keep holding."""
    scope, namespace = memory_scope_and_namespace(DEFAULT_USER)
    assert scope == "user"
    assert namespace == "floralens"


def test_memory_namespace_is_distinct_per_user():
    scope_a, ns_a = memory_scope_and_namespace("alice")
    scope_b, ns_b = memory_scope_and_namespace("bob")
    _, ns_default = memory_scope_and_namespace(DEFAULT_USER)

    assert ns_a != ns_b
    assert ns_a != ns_default
    assert ns_b != ns_default
    assert scope_a == scope_b == "user"


def test_compile_naturalist_scopes_memory_provider_lookup_per_user():
    manifests_alice = assistant_service.load_naturalist_manifests("alice")
    manifests_bob = assistant_service.load_naturalist_manifests("bob")
    manifests_default = assistant_service.load_naturalist_manifests()

    assert manifests_alice["naturalist"].memory.namespace == "floralens:alice"
    assert manifests_bob["naturalist"].memory.namespace == "floralens:bob"
    assert manifests_default["naturalist"].memory.namespace == "floralens"


# --------------------------------------------------------------------------- #
# Garden schema migration: an existing OLD-schema garden.db (specimen_id PK,
# no user_id column) must be rebuilt in place, preserving every row, now
# owned by DEFAULT_USER -- data is never lost.
# --------------------------------------------------------------------------- #
def test_old_schema_garden_db_migrates_and_preserves_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "old_garden.db"
    monkeypatch.setattr(garden_service, "_DB_PATH", db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE garden ("
            "specimen_id TEXT PRIMARY KEY, label_name TEXT NOT NULL, saved_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO garden (specimen_id, label_name, saved_at) VALUES (?, ?, ?)",
            ("legacy-sid", "Legacy Rose", "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()

    # Any service call triggers the lazy, idempotent migration.
    items = garden_service.list_specimens(DEFAULT_USER)
    assert len(items) == 1
    assert items[0].specimen_id == "legacy-sid"
    assert items[0].label_name == "Legacy Rose"

    conn = sqlite3.connect(db_path)
    try:
        cols = {c[1] for c in conn.execute("PRAGMA table_info(garden)").fetchall()}
        assert "user_id" in cols
        row = conn.execute(
            "SELECT user_id, specimen_id, label_name FROM garden WHERE specimen_id = ?",
            ("legacy-sid",),
        ).fetchone()
        assert row == (DEFAULT_USER, "legacy-sid", "Legacy Rose")
    finally:
        conn.close()

    # Migration doesn't leak the backfilled row to an unrelated user.
    assert garden_service.list_specimens("someone-else") == []


def test_garden_migration_is_idempotent_across_repeated_connections(tmp_path, monkeypatch):
    db_path = tmp_path / "old_garden_2.db"
    monkeypatch.setattr(garden_service, "_DB_PATH", db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE garden ("
            "specimen_id TEXT PRIMARY KEY, label_name TEXT NOT NULL, saved_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO garden (specimen_id, label_name, saved_at) VALUES (?, ?, ?)",
            ("sid-x", "Old Tulip", "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()

    with garden_service._connection():
        pass  # first connection: triggers the migration
    with garden_service._connection() as c2:  # second connection: must be a no-op, not an error
        rows = c2.execute("SELECT specimen_id FROM garden").fetchall()
    assert rows == [("sid-x",)]


# --------------------------------------------------------------------------- #
# Real-gallery-data variant, guarded exactly like test_garden.py's own
# equivalent -- skips cleanly when the (gitignored) embeddings cache hasn't
# been built.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_garden_isolation_holds_for_a_real_gallery_specimen(monkeypatch):
    import json

    monkeypatch.setenv("FLORALENS_JWT_SECRET", "test-secret")
    garden_service.reset_specimen_catalog_cache()
    try:
        metadata = json.loads(EMBEDDINGS_CACHE.read_text(encoding="utf-8"))
        specimen_id, _ = next(
            (sid, m) for sid, m in metadata["specimens"].items() if m["split"] == "gallery"
        )

        alice_headers = _auth_header(issue_token("alice"))
        bob_headers = _auth_header(issue_token("bob"))

        resp = client.post("/api/garden", json={"specimen_id": specimen_id}, headers=alice_headers)
        assert resp.status_code == 201
        assert client.get("/api/garden", headers=bob_headers).json()["items"] == []
    finally:
        garden_service.reset_specimen_catalog_cache()
