"""Hardening tests (PRD Phase 9): opt-in API-key auth, per-IP rate limiting,
and secret redaction. All auth/rate-limit tests are careful to restore the
default (no env set) state afterward, since the rest of the test suite
depends on that default-open behavior.
"""
import json

import pytest
from agent_core import ModelProvider, ModelResponse
from fastapi.testclient import TestClient

from apps.api.app import rate_limit
from apps.api.app.main import _assistant_error_event, app, assistant_registries
from apps.api.app.redaction import RedactingLogFilter, redact_secrets

client = TestClient(app)


class _FakeOpenAI(ModelProvider):
    """Deterministic offline stand-in (same pattern as test_assistant.py's
    `_FakeOpenAI`) so this file's assistant calls never spend a real LLM call
    or depend on another test module having run first."""

    provider = "openai"

    async def complete(self, messages, tools=None, **cfg):
        return ModelResponse(text="(offline hardening test answer)", usage={"input_tokens": 1, "output_tokens": 1})


@pytest.fixture(autouse=True)
def _fake_openai_provider():
    original = assistant_registries.models.get("openai") if assistant_registries.models.has("openai") else None
    assistant_registries.models.register("openai", _FakeOpenAI(), overwrite=True)
    yield
    if original is not None:
        assistant_registries.models.register("openai", original, overwrite=True)


# --------------------------------------------------------------------------- #
# Auth: default-open (no FLORALENS_API_KEY) behavior must be unchanged.
# --------------------------------------------------------------------------- #
def test_garden_open_by_default_without_api_key(monkeypatch):
    monkeypatch.delenv("FLORALENS_API_KEY", raising=False)
    resp = client.get("/api/garden")
    assert resp.status_code == 200


def test_memory_open_by_default_without_api_key(monkeypatch):
    monkeypatch.delenv("FLORALENS_API_KEY", raising=False)
    resp = client.get("/api/memory")
    assert resp.status_code == 200


def test_health_and_search_never_require_a_key_even_when_configured(monkeypatch):
    # Read-only endpoints stay public regardless of FLORALENS_API_KEY.
    monkeypatch.setenv("FLORALENS_API_KEY", "s3cret-key")
    try:
        assert client.get("/health").status_code == 200
        assert client.get("/api/pipeline").status_code == 200
        # Malformed body still gets a 400 from validation, never a 401 — proves
        # no auth dependency is attached to /api/search.
        assert client.post("/api/search").status_code == 400
    finally:
        monkeypatch.delenv("FLORALENS_API_KEY", raising=False)


# --------------------------------------------------------------------------- #
# Auth: when FLORALENS_API_KEY is set, protected endpoints require it.
# --------------------------------------------------------------------------- #
def test_garden_requires_key_when_configured(monkeypatch):
    monkeypatch.setenv("FLORALENS_API_KEY", "s3cret-key")
    try:
        no_key = client.get("/api/garden")
        assert no_key.status_code == 401

        wrong_key = client.get("/api/garden", headers={"X-API-Key": "wrong"})
        assert wrong_key.status_code == 401

        right_key = client.get("/api/garden", headers={"X-API-Key": "s3cret-key"})
        assert right_key.status_code == 200

        bearer = client.get("/api/garden", headers={"Authorization": "Bearer s3cret-key"})
        assert bearer.status_code == 200
    finally:
        monkeypatch.delenv("FLORALENS_API_KEY", raising=False)


def test_memory_requires_key_when_configured(monkeypatch):
    monkeypatch.setenv("FLORALENS_API_KEY", "s3cret-key")
    try:
        assert client.get("/api/memory").status_code == 401
        assert client.get("/api/memory", headers={"X-API-Key": "s3cret-key"}).status_code == 200
        assert client.delete("/api/memory").status_code == 401
        assert client.delete("/api/memory", headers={"X-API-Key": "s3cret-key"}).status_code == 200
    finally:
        monkeypatch.delenv("FLORALENS_API_KEY", raising=False)


def test_assistant_requires_key_when_configured(monkeypatch):
    monkeypatch.setenv("FLORALENS_API_KEY", "s3cret-key")
    try:
        resp = client.post("/api/assistant", json={"message": "hi"})
        assert resp.status_code == 401
    finally:
        monkeypatch.delenv("FLORALENS_API_KEY", raising=False)


# --------------------------------------------------------------------------- #
# Rate limiting: generous default doesn't trip normal use; a tight override
# (via env) does trip and returns 429.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _reset_rate_limits():
    rate_limit.reset_rate_limits()
    yield
    rate_limit.reset_rate_limits()


def test_search_rate_limit_returns_429_when_exceeded(monkeypatch):
    monkeypatch.setenv("FLORALENS_RATE_LIMIT_SEARCH_PER_MIN", "2")
    try:
        # Cheap invalid-body requests: they still cost a rate-limit token
        # (the dependency runs before body validation) without needing the
        # embeddings cache to be built.
        first = client.post("/api/search", json={"image_base64": "bm90LXZhbGlk"})
        second = client.post("/api/search", json={"image_base64": "bm90LXZhbGlk"})
        third = client.post("/api/search", json={"image_base64": "bm90LXZhbGlk"})
        assert first.status_code != 429
        assert second.status_code != 429
        assert third.status_code == 429
    finally:
        monkeypatch.delenv("FLORALENS_RATE_LIMIT_SEARCH_PER_MIN", raising=False)


def test_assistant_rate_limit_returns_429_when_exceeded(monkeypatch):
    monkeypatch.setenv("FLORALENS_RATE_LIMIT_ASSISTANT_PER_MIN", "1")
    try:
        first = client.post("/api/assistant", json={"message": "hi"})
        second = client.post("/api/assistant", json={"message": "hi"})
        assert first.status_code == 200
        assert second.status_code == 429
    finally:
        monkeypatch.delenv("FLORALENS_RATE_LIMIT_ASSISTANT_PER_MIN", raising=False)


def test_search_default_rate_limit_does_not_trip_normal_use():
    # Default is generous (120/min) — a handful of quick calls must never 429.
    for _ in range(5):
        resp = client.post("/api/search", json={"image_base64": "bm90LXZhbGlk"})
        assert resp.status_code != 429


# --------------------------------------------------------------------------- #
# Secret redaction.
# --------------------------------------------------------------------------- #
def test_redact_secrets_scrubs_openai_and_tavily_keys():
    text = "failed with OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwx and TAVILY_API_KEY=tvly-1234567890abcdef"
    redacted = redact_secrets(text)
    assert "sk-abcdefghijklmnopqrstuvwx" not in redacted
    assert "tvly-1234567890abcdef" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_secrets_scrubs_bearer_token():
    text = "Authorization: Bearer sk-live-abcdefgh12345678"
    redacted = redact_secrets(text)
    assert "abcdefgh12345678" not in redacted


def test_redact_secrets_scrubs_generic_key_value_pairs():
    text = 'config: {"api_key": "abcd1234efgh5678"}'
    redacted = redact_secrets(text)
    assert "abcd1234efgh5678" not in redacted
    assert "api_key" in redacted  # key name preserved, only the value scrubbed


def test_redact_secrets_leaves_ordinary_text_untouched():
    text = "search returned 5 results for a rose query"
    assert redact_secrets(text) == text


def test_assistant_error_event_redacts_key_in_detail():
    # Proves a key present in an error's input/detail never survives into the
    # SSE payload sent to the client.
    event = _assistant_error_event("upstream failed, key=sk-abcdefghijklmnop was rejected")
    assert "sk-abcdefghijklmnop" not in event
    payload = json.loads(event.removeprefix("data: ").strip())
    assert payload["type"] == "error"
    assert "[REDACTED]" in payload["detail"]


def test_redacting_log_filter_scrubs_log_record_message():
    import logging

    record = logging.LogRecord(
        name="test", level=logging.ERROR, pathname=__file__, lineno=1,
        msg="auth failed with api_key=abcd1234secretvalue", args=(), exc_info=None,
    )
    RedactingLogFilter().filter(record)
    assert "abcd1234secretvalue" not in record.getMessage()
