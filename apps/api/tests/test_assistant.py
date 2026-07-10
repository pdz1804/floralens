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
import re
from pathlib import Path

import pytest
from agent_core import ModelProvider, ModelResponse, resolve_manifest, sqlite_checkpointer
from agent_core.errors import UnknownReferenceError
from fastapi.testclient import TestClient

from apps.api.app.assistant_service import (
    GalleryFactsTool,
    build_floralens_registries,
    compile_naturalist,
    load_naturalist_manifests,
)
from apps.api.app.main import app, assistant_registries

EMBEDDINGS_CACHE = Path("ml/data/embeddings_cache/metadata.json")

# The naturalist supervisor's three sub-agents (PRD D2 four-role team) are each
# exposed to it as an `ask_<id>` delegation tool by `compile_naturalist`.
_EXPECTED_SUBAGENT_TOOLS = {"ask_identifier", "ask_researcher", "ask_care_advisor"}
# The runtime-enforced output guardrails attached to the naturalist manifest.
_EXPECTED_GUARDRAILS = ["no_medical_dosage", "educational_disclaimer", "no_secret_exfil"]

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
    known = set(manifests)  # naturalist + identifier + researcher + care_advisor
    for manifest in manifests.values():
        # Resolves the model provider, prompt, tools, sub_agents AND the
        # supervisor's guardrails against the real assistant registries.
        resolve_manifest(manifest, assistant_registries, known_agents=known)


def test_naturalist_manifest_declares_guardrails():
    """The supervisor manifest carries the three built-in guardrails by their
    exact registered names, and each resolves against the registries."""
    manifests = load_naturalist_manifests()
    assert manifests["naturalist"].guardrails == _EXPECTED_GUARDRAILS
    for name in _EXPECTED_GUARDRAILS:
        assert assistant_registries.guardrails.has(name)


def test_compile_naturalist_wires_all_three_subagents_and_guardrails():
    """compile_naturalist exposes each sub-agent as an `ask_<id>` tool and
    attaches the configured guardrails to the compiled supervisor."""
    agent = compile_naturalist(build_floralens_registries())
    assert _EXPECTED_SUBAGENT_TOOLS <= set(agent._tools)
    attached = [g.name for g in agent._guardrails]
    assert attached == _EXPECTED_GUARDRAILS


def test_unknown_guardrail_name_fails_fast():
    """Sanity: an unregistered guardrail name is rejected at compile time
    (fail-fast, exactly like an unknown tool/prompt) rather than ignored."""
    manifests = load_naturalist_manifests()
    naturalist = manifests["naturalist"].model_copy(
        update={"guardrails": ["no_such_guardrail"]}
    )
    registries = build_floralens_registries()
    with pytest.raises(UnknownReferenceError):
        resolve_manifest(naturalist, registries, known_agents=set(manifests))


class _FixedAnswerModel(ModelProvider):
    """Returns a fixed answer with no tool calls, so the supervisor emits it in
    one step and the manifest's guardrails run over exactly this text."""

    provider = "openai"

    def __init__(self, text: str) -> None:
        self._text = text

    async def complete(self, messages, tools=None, **cfg):
        return ModelResponse(text=self._text, usage={"input_tokens": 1, "output_tokens": 1})


def _run_naturalist_with_answer(raw_answer: str) -> str:
    """Compile the naturalist against fresh registries whose model returns
    ``raw_answer`` verbatim, run it once offline, and return the guardrailed
    final answer."""
    registries = build_floralens_registries()
    registries.models.register("openai", _FixedAnswerModel(raw_answer), overwrite=True)
    agent = compile_naturalist(registries)

    async def _drive():
        try:
            return await agent.arun("tell me about this plant")
        finally:
            await agent.aclose()

    return asyncio.run(_drive()).answer


def test_guardrail_blocks_medical_dosage_in_final_answer():
    """A raw answer stating a specific dose is replaced by the no_medical_dosage
    refusal — the concrete dosage never survives to the user."""
    answer = _run_naturalist_with_answer("Give the plant 500 mg of aspirin daily.")
    assert "500 mg" not in answer
    assert "can't provide a specific medical dosage" in answer.lower()


def test_guardrail_redacts_secret_in_final_answer():
    """A leaked API key in the raw answer is redacted by no_secret_exfil before
    it reaches the user."""
    leaked = "sk-abcdef0123456789ABCDEF"
    answer = _run_naturalist_with_answer(f"Sure, the key is {leaked} — enjoy!")
    assert leaked not in answer
    assert "[REDACTED]" in answer


def test_guardrail_appends_educational_disclaimer():
    """A dosage/secret-free answer passes through but gains the educational
    disclaimer appended by educational_disclaimer."""
    answer = _run_naturalist_with_answer("Roses like full sun and well-drained soil.")
    assert "Roses like full sun" in answer
    assert "educational purposes only" in answer.lower()


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


# --------------------------------------------------------------------------- #
# Opt-in durable short-term thread memory (PRD Phase 7 / E3): compile_naturalist
# now accepts an agent_core checkpointer. These tests build their own fresh
# registries so they never clobber the module-level `_FakeOpenAI` the streaming
# tests above rely on.
# --------------------------------------------------------------------------- #
class _CountingModel(ModelProvider):
    """Answers directly (no tool calls) and reports how many messages it was
    given. With a checkpointer resuming a thread, turn 2 sees turn 1's persisted
    history, so its message count is strictly higher than a fresh turn's."""

    provider = "openai"

    async def complete(self, messages, tools=None, **cfg):
        return ModelResponse(
            text=f"messages_seen={len(messages)}",
            usage={"input_tokens": 1, "output_tokens": 1},
        )


def _messages_seen(answer: str) -> int:
    # Extract just the integer: the naturalist manifest's educational_disclaimer
    # guardrail appends a note to the final answer, so a plain split("=") would
    # capture that trailing text too.
    return int(re.search(r"messages_seen=(\d+)", answer).group(1))


def _compile_counting_naturalist(checkpointer):
    registries = build_floralens_registries()
    registries.models.register("openai", _CountingModel(), overwrite=True)
    return compile_naturalist(registries, checkpointer)


def test_checkpointer_enables_thread_resume(tmp_path):
    """With a durable checkpointer, a second turn on the same thread_id resumes
    the first turn's persisted state — even when compiled by a *separate* agent
    instance (mirroring the endpoint's fresh-agent-per-request model)."""
    db_path = str(tmp_path / "threads.sqlite")

    async def _drive():
        # Two independently compiled agents sharing one on-disk db, both driven
        # in the same event loop (the AsyncSqliteSaver connection is loop-bound).
        first = _compile_counting_naturalist(sqlite_checkpointer(db_path))
        try:
            r1 = await first.arun("how do I care for a rose?", thread_id="conv-1")
        finally:
            await first.aclose()
        second = _compile_counting_naturalist(sqlite_checkpointer(db_path))
        try:
            r2 = await second.arun("and a tulip?", thread_id="conv-1")
        finally:
            await second.aclose()
        return r1, r2

    r1, r2 = asyncio.run(_drive())
    # Turn 2 resumed turn 1's history (prior user+assistant turns persisted),
    # so the model saw strictly more messages than the fresh first turn.
    assert _messages_seen(r2.answer) > _messages_seen(r1.answer)


def test_without_checkpointer_runs_stay_single_shot():
    """Default (no checkpointer): repeated runs on the same thread_id do NOT
    accumulate history — behavior is exactly as before this feature."""

    async def _drive():
        agent = _compile_counting_naturalist(None)
        try:
            r1 = await agent.arun("first", thread_id="conv-1")
            r2 = await agent.arun("second", thread_id="conv-1")
        finally:
            await agent.aclose()
        return r1, r2

    r1, r2 = asyncio.run(_drive())
    assert _messages_seen(r2.answer) == _messages_seen(r1.answer)


def test_streaming_with_checkpointer_still_yields_answer(tmp_path):
    """The endpoint's streaming path (astream + aclose in a finally) still
    produces a normal answer event when a checkpointer is configured — no
    regression from wiring one in."""
    db_path = str(tmp_path / "stream.sqlite")

    async def _drive():
        agent = _compile_counting_naturalist(sqlite_checkpointer(db_path))
        event_types = []
        try:
            async for event in agent.astream("hello", thread_id="s-1"):
                event_types.append(event.type)
        finally:
            await agent.aclose()
        return event_types

    event_types = asyncio.run(_drive())
    assert "answer" in event_types
