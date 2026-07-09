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
from pydantic import BaseModel, Field

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


def load_naturalist_manifests() -> dict[str, AgentManifest]:
    """Load the naturalist supervisor + its care_advisor sub-agent manifest."""
    return {
        "naturalist": load_manifest_file(_AGENTS_DIR / "naturalist.yaml"),
        "care_advisor": load_manifest_file(_AGENTS_DIR / "care_advisor.yaml"),
    }


def compile_naturalist(registries: Registries) -> CompiledAgent:
    """Resolve + compile the naturalist manifest (with care_advisor as a
    sub-agent tool, `ask_care_advisor`) against the given registries."""
    manifests = load_naturalist_manifests()
    return compile_agent(
        manifests["naturalist"],
        registries,
        agents={"care_advisor": manifests["care_advisor"]},
    )
