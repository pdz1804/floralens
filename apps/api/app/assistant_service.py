"""FloraLens naturalist assistant service.

**Cross-product reuse proof (PRD US-6 / AgentForge P12):** this module builds
the naturalist agent entirely on top of AgentForge's Unified Agent Core
(`agent_core`), installed *editable* from the sibling agentforge repo into
this app's venv (`pip install -e ../agentforge/packages/agent-core[openai]`,
see requirements.txt). The manifest schema, registries, LangGraph ReAct
runtime, and OpenAI/Echo model providers are the exact same code that powers
AgentForge itself — nothing here forks or re-implements them. FloraLens only
adds one domain tool (`GalleryFactsTool`, below) and two manifests/prompts
under `agents/` + `prompts/`.
"""
from __future__ import annotations

from pathlib import Path

from agent_core import (
    AgentManifest,
    BaseTool,
    CompiledAgent,
    Registries,
    ToolResult,
    build_default_registries,
    compile_agent,
    load_manifest_file,
)
from agent_core.checkpoint import CheckpointerArg
from pydantic import BaseModel, Field

from apps.api.app.auth import DEFAULT_USER
from apps.api.app.config import settings
from ml.descriptions.loader import get_description
from ml.embeddings.cache import load_embeddings

# assistant_service.py -> app -> api -> apps -> <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGENTS_DIR = _REPO_ROOT / "agents"
_PROMPTS_DIR = _REPO_ROOT / "prompts"


class GalleryFactsArgs(BaseModel):
    species: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Flower species/common name to look up, e.g. 'rose' or 'sunflower'.",
    )


class GalleryFactsTool(BaseTool):
    """Looks up a species in the FloraLens gallery, read-only.

    Reuses the same embeddings-cache metadata `search_service.get_gallery_store`
    is built from (`ml.embeddings.cache.load_embeddings`) and the curated
    per-species description loader (`ml.descriptions.loader.get_description`)
    — no gallery data is duplicated or mutated, and no model/tool file outside
    this module's ownership is modified.
    """

    name = "gallery_facts"
    description = (
        "Look up a flower species in the FloraLens gallery by name (substring "
        "match, e.g. 'rose' or 'sunflower'). Returns the curated botanical "
        "description and how many gallery specimens exist for each match."
    )
    args_schema = GalleryFactsArgs

    async def run(self, **kwargs) -> ToolResult:
        args = self.validate_args(**kwargs)
        query = args.species.strip().lower()
        try:
            _, metadata = load_embeddings(settings.embeddings_cache_dir)
        except FileNotFoundError as exc:
            return ToolResult(
                ok=False, error=f"gallery index not built yet: {exc}"
            )

        counts: dict[str, int] = {}
        for meta in metadata.get("specimens", {}).values():
            if meta.get("split") != "gallery":
                continue
            name = meta["label_name"]
            if query in name.lower():
                counts[name] = counts.get(name, 0) + 1

        if not counts:
            return ToolResult(
                ok=True,
                output=f"No gallery specimens found matching '{args.species}'.",
                meta={"matched_species": []},
            )

        lines = []
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            line = f"- {name}: {count} gallery specimen(s)"
            desc = get_description(name)
            if desc:
                line += f" — {desc}"
            lines.append(line)
        return ToolResult(
            ok=True,
            output="\n".join(lines),
            meta={"matched_species": sorted(counts)},
        )


def build_floralens_registries() -> Registries:
    """Default agent-core registries (echo/openai/anthropic models, web_search,
    embedding_search, ...) plus FloraLens's one domain tool and its
    `agents/`/`prompts/` manifests — proving the shared core extends without
    any core code edits (AgentForge PRD §8.5)."""
    registries = build_default_registries(prompts_dir=_PROMPTS_DIR)
    registries.tools.register("gallery_facts", GalleryFactsTool())
    return registries


def namespaced_for_user(base_namespace: str, user_id: str) -> str:
    """Per-user memory namespace (per-user auth scaffold, see auth.py).

    `DEFAULT_USER` maps to `base_namespace` UNCHANGED — the exact bucket the
    naturalist manifest declares today (agents/naturalist.yaml's
    `memory.namespace`) — so auth-off/single-user behavior (the default) is
    byte-for-byte identical to before this scaffold existed, and existing
    memory tests asserting that literal namespace keep passing. Any other
    resolved user_id gets its own suffixed bucket, so different users never
    see each other's memories."""
    if user_id == DEFAULT_USER:
        return base_namespace
    return f"{base_namespace}:{user_id}"


def _apply_memory_provider_override(manifest: AgentManifest, user_id: str = DEFAULT_USER) -> None:
    """Point the naturalist's long-term memory at the configured backend (Gap
    G12) and, additively, at `user_id`'s namespace bucket (per-user auth
    scaffold). The manifest declares `in_memory` (the dependency-free default);
    when `FLORALENS_MEMORY_PROVIDER` selects a different backend (e.g. `mem0`),
    swap the provider name in place so both the compiled agent and the
    /api/memory inspector read the exact same store. With the env var unset (or
    set to the manifest's own provider) the provider swap is a no-op. The
    namespace is always rewritten via `namespaced_for_user`, which is itself a
    no-op for `DEFAULT_USER` — so with auth off, default behavior is unchanged.
    The provider name is validated against the registry downstream by
    `compile_agent`, so an unknown value fails loudly rather than silently."""
    if manifest.memory is None:
        return
    if settings.memory_provider != manifest.memory.provider:
        manifest.memory.provider = settings.memory_provider
    manifest.memory.namespace = namespaced_for_user(manifest.memory.namespace, user_id)


def load_naturalist_manifests(user_id: str = DEFAULT_USER) -> dict[str, AgentManifest]:
    """Load the naturalist supervisor + its three sub-agent manifests.

    The supervisor (naturalist) delegates to the four-role team (PRD D2):
    `identifier` (species ID), `researcher` (cited botanical facts), and
    `care_advisor` (FloraLens gallery facts). Each is exposed to the supervisor
    as an `ask_<id>` tool by `compile_naturalist`.

    `user_id` (default `DEFAULT_USER`, i.e. today's single-user behavior)
    scopes the naturalist's long-term memory namespace to that user — see
    `_apply_memory_provider_override` / `namespaced_for_user`."""
    manifests = {
        "naturalist": load_manifest_file(_AGENTS_DIR / "naturalist.yaml"),
        "identifier": load_manifest_file(_AGENTS_DIR / "identifier.yaml"),
        "researcher": load_manifest_file(_AGENTS_DIR / "researcher.yaml"),
        "care_advisor": load_manifest_file(_AGENTS_DIR / "care_advisor.yaml"),
    }
    _apply_memory_provider_override(manifests["naturalist"], user_id)
    return manifests


def compile_naturalist(
    registries: Registries,
    checkpointer: CheckpointerArg = None,
    user_id: str = DEFAULT_USER,
) -> CompiledAgent:
    """Resolve + compile the naturalist manifest (with identifier, researcher,
    and care_advisor as sub-agent tools `ask_identifier` / `ask_researcher` /
    `ask_care_advisor`, plus its runtime output guardrails) against the given
    registries.

    `checkpointer` opts the compiled naturalist into agent_core's durable
    short-term thread memory (PRD Phase 7 / E3): pass a saver built via
    `agent_core.sqlite_checkpointer(...)` and LangGraph persists state per
    request `thread_id`, so same-thread runs resume prior state (multi-turn
    conversations) while different threads stay isolated. Left `None` (the
    default), behavior is exactly as before — single-shot runs, no cross-turn
    state — so the local demo and existing callers are unaffected. Only the
    top-level supervisor is checkpointed; the care_advisor sub-agent stays
    single-shot (see `compile_agent`'s sub-agent handling).

    `user_id` (default `DEFAULT_USER`) scopes the compiled agent's long-term
    memory to that user's namespace bucket (per-user auth scaffold)."""
    manifests = load_naturalist_manifests(user_id)
    return compile_agent(
        manifests["naturalist"],
        registries,
        agents={
            "identifier": manifests["identifier"],
            "researcher": manifests["researcher"],
            "care_advisor": manifests["care_advisor"],
        },
        checkpointer=checkpointer,
    )
