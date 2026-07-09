"""Naturalist assistant tests (PRD P5-6 / US-6 cross-product reuse).

Runs fully offline: the real `openai` provider slot in `assistant_registries`
is swapped for a deterministic fake before any test in this module runs (same
pattern AgentForge's own API tests use — see
agentforge/apps/api/tests/test_health.py::test_run_streams_error_event_on_runtime_failure),
so no test spends a real LLM call. The gallery tool, however, reads the real
on-disk embeddings cache (skipped if it hasn't been built yet).
"""
import asyncio
import json
from pathlib import Path

import pytest
from agent_core import ModelProvider, ModelResponse, resolve_manifest
from fastapi.testclient import TestClient

from apps.api.app.assistant_service import GalleryFactsTool, load_naturalist_manifests
from apps.api.app.main import app, assistant_registries

EMBEDDINGS_CACHE = Path("ml/data/embeddings_cache/metadata.json")

client = TestClient(app)


class _FakeOpenAI(ModelProvider):
    """Deterministic stand-in for the real OpenAI provider — no network, no
    tool calls, so the supervisor answers directly in one step."""

    provider = "openai"

    async def complete(self, messages, tools=None, **cfg):
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return ModelResponse(
            text=f"(offline test answer for: {last_user}) source: https://example.com/care-guide",
            usage={"input_tokens": 1, "output_tokens": 1},
        )


# Registered once at module import so every test in this file runs offline.
assistant_registries.models.register("openai", _FakeOpenAI(), overwrite=True)


def test_naturalist_manifests_resolve():
    manifests = load_naturalist_manifests()
    resolve_manifest(
        manifests["naturalist"], assistant_registries, known_agents={"care_advisor"}
    )
    resolve_manifest(
        manifests["care_advisor"], assistant_registries, known_agents={"care_advisor"}
    )


def test_assistant_streams_answer_offline():
    resp = client.post("/api/assistant", json={"message": "How do I care for a rose?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert '"type": "run_started"' in body
    assert '"type":"answer"' in body  # TraceEvent.model_dump_json (no space after ':')
    assert "example.com/care-guide" in body  # the fake model's cited source
    assert '"type": "done"' in body


def test_assistant_rejects_empty_message():
    resp = client.post("/api/assistant", json={"message": ""})
    assert resp.status_code == 422


def test_assistant_preserves_thread_id():
    resp = client.post(
        "/api/assistant", json={"message": "identify this bloom", "thread_id": "t-42"}
    )
    assert resp.status_code == 200
    assert '"type":"answer"' in resp.text


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_gallery_facts_tool_returns_real_gallery_data():
    metadata = json.loads(EMBEDDINGS_CACHE.read_text(encoding="utf-8"))
    gallery_name = next(
        m["label_name"] for m in metadata["specimens"].values() if m["split"] == "gallery"
    )
    query_term = gallery_name.split()[0]

    tool = GalleryFactsTool()
    result = asyncio.run(tool.run(species=query_term))

    assert result.ok
    assert gallery_name in result.output
    assert gallery_name in result.meta["matched_species"]


def test_gallery_facts_tool_handles_no_match_gracefully():
    tool = GalleryFactsTool()
    result = asyncio.run(tool.run(species="zzz-not-a-real-species-zzz"))
    assert result.ok  # a clean "no match" is not a tool error
    assert result.meta["matched_species"] == []
    assert "no gallery specimens found" in result.output.lower()


def test_gallery_facts_tool_rejects_empty_species():
    tool = GalleryFactsTool()
    with pytest.raises(Exception):  # pydantic ValidationError via validate_args
        asyncio.run(tool.run(species=""))
