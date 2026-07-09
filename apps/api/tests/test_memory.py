"""Assistant memory inspector tests (PRD Phase 7 / Epic E4) — list + clear
via `GET/DELETE /api/memory`, against the exact `in_memory` MemoryProvider
instance the naturalist manifest (agents/naturalist.yaml `memory:` block)
declares. Each test clears the shared bucket before/after so tests never
bleed into each other or into a real chat session's memories."""
import asyncio

import pytest
from agent_core import MemoryItem, ModelProvider, ModelResponse, Scope
from fastapi.testclient import TestClient

from apps.api.app.main import app, assistant_registries
from apps.api.app.memory_service import memory_scope_and_namespace

client = TestClient(app)


def _provider_scope_namespace():
    scope, namespace = memory_scope_and_namespace()
    return assistant_registries.memory.get("in_memory"), Scope(scope), namespace


async def _clear_bucket() -> None:
    provider, scope, namespace = _provider_scope_namespace()
    items = await provider.all(scope, namespace)
    ids = [it.id for it in items if it.id is not None]
    if ids:
        await provider.delete(scope, namespace, ids)


@pytest.fixture(autouse=True)
def clean_memory():
    asyncio.run(_clear_bucket())
    yield
    asyncio.run(_clear_bucket())


def test_list_memory_empty_by_default():
    resp = client.get("/api/memory")
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    # Values come straight from agents/naturalist.yaml's memory: block.
    assert body["scope"] == "user"
    assert body["namespace"] == "floralens"


def test_list_memory_returns_seeded_items():
    provider, scope, namespace = _provider_scope_namespace()
    asyncio.run(
        provider.add(scope, namespace, [MemoryItem(text="User said: hi\nAssistant answered: hello")])
    )

    resp = client.get("/api/memory")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert "hello" in items[0]["text"]
    assert items[0]["id"]  # provider assigns an id on add


def test_clear_memory_removes_all_items():
    provider, scope, namespace = _provider_scope_namespace()
    asyncio.run(provider.add(scope, namespace, [MemoryItem(text="a"), MemoryItem(text="b")]))

    resp = client.delete("/api/memory")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 2
    assert client.get("/api/memory").json()["items"] == []


def test_clear_memory_is_idempotent_on_empty_bucket():
    resp = client.delete("/api/memory")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 0


class _FakeOpenAI(ModelProvider):
    """Deterministic offline stand-in — same pattern as test_assistant.py's
    `_FakeOpenAI`. Registered/restored via a fixture (not module-level, like
    test_assistant.py does) because `assistant_registries.models` is process
    -global: a bare module-level `register(..., overwrite=True)` here would
    permanently clobber test_assistant.py's own fake whenever pytest collects
    this module first, breaking that file's assertions depending on run order."""

    provider = "openai"

    async def complete(self, messages, tools=None, **cfg):
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return ModelResponse(text=f"(memory test answer for: {last_user})", usage={"input_tokens": 1, "output_tokens": 1})


@pytest.fixture
def fake_openai_model():
    original = assistant_registries.models.get("openai") if assistant_registries.models.has("openai") else None
    assistant_registries.models.register("openai", _FakeOpenAI(), overwrite=True)
    yield
    if original is not None:
        assistant_registries.models.register("openai", original, overwrite=True)


def test_assistant_turn_is_persisted_to_the_same_memory_the_inspector_reads(fake_openai_model):
    resp = client.post("/api/assistant", json={"message": "remember I like peonies"})
    assert resp.status_code == 200

    items = client.get("/api/memory").json()["items"]
    assert any("peonies" in it["text"] for it in items)
