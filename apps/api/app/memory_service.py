"""Memory inspector: read/clear the naturalist assistant's long-term memory
(PRD Phase 7, Epic E4 — "view/edit/delete" memory control).

Reuses `agent_core`'s `MemoryProvider` — the exact provider instance, scope
and namespace the compiled naturalist agent itself reads/writes (see
`agent_core.runtime.CompiledAgent._retrieve` / `_persist`), sourced straight
from the naturalist manifest's `memory:` block (agents/naturalist.yaml) so
this module can never drift out of sync with what the assistant actually
uses. No memory logic is duplicated or forked here — this is a thin
list/delete view over the same store.
"""
from __future__ import annotations

from agent_core import Registries, Scope

from apps.api.app.assistant_service import load_naturalist_manifests


class MemoryNotConfiguredError(RuntimeError):
    """Raised if the naturalist manifest has no `memory:` block configured."""


def _memory_config():
    manifest = load_naturalist_manifests()["naturalist"]
    if manifest.memory is None:
        raise MemoryNotConfiguredError("the naturalist manifest has no memory configured")
    return manifest.memory


def memory_scope_and_namespace() -> tuple[str, str]:
    """The (scope, namespace) the naturalist assistant's memory is bucketed
    under — for display alongside the listed items."""
    cfg = _memory_config()
    return cfg.scope.value, cfg.namespace


async def list_memories(registries: Registries) -> list[dict]:
    """All memory items the assistant currently holds, oldest first (as
    returned by the provider)."""
    cfg = _memory_config()
    provider = registries.memory.get(cfg.provider)
    items = await provider.all(Scope(cfg.scope.value), cfg.namespace)
    return [{"id": it.id, "text": it.text, "meta": it.meta} for it in items]


async def clear_memories(registries: Registries) -> int:
    """Delete every memory item in the assistant's bucket. Returns the count
    removed."""
    cfg = _memory_config()
    provider = registries.memory.get(cfg.provider)
    scope = Scope(cfg.scope.value)
    items = await provider.all(scope, cfg.namespace)
    ids = [it.id for it in items if it.id is not None]
    if ids:
        await provider.delete(scope, cfg.namespace, ids)
    return len(ids)
