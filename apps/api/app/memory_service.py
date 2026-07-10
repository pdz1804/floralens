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
from apps.api.app.auth import DEFAULT_USER


class MemoryNotConfiguredError(RuntimeError):
    """Raised if the naturalist manifest has no `memory:` block configured."""


def _memory_config(user_id: str = DEFAULT_USER):
    manifest = load_naturalist_manifests(user_id)["naturalist"]
    if manifest.memory is None:
        raise MemoryNotConfiguredError("the naturalist manifest has no memory configured")
    return manifest.memory


def memory_scope_and_namespace(user_id: str = DEFAULT_USER) -> tuple[str, str]:
    """The (scope, namespace) `user_id`'s assistant memory is bucketed under —
    for display alongside the listed items. `DEFAULT_USER` (the default)
    returns the exact namespace the naturalist manifest declares, unchanged
    from before this scaffold existed."""
    cfg = _memory_config(user_id)
    return cfg.scope.value, cfg.namespace


async def list_memories(registries: Registries, user_id: str = DEFAULT_USER) -> list[dict]:
    """All memory items `user_id`'s assistant bucket currently holds, oldest
    first (as returned by the provider)."""
    cfg = _memory_config(user_id)
    provider = registries.memory.get(cfg.provider)
    items = await provider.all(Scope(cfg.scope.value), cfg.namespace)
    return [{"id": it.id, "text": it.text, "meta": it.meta} for it in items]


async def clear_memories(registries: Registries, user_id: str = DEFAULT_USER) -> int:
    """Delete every memory item in `user_id`'s assistant bucket. Returns the
    count removed."""
    cfg = _memory_config(user_id)
    provider = registries.memory.get(cfg.provider)
    scope = Scope(cfg.scope.value)
    items = await provider.all(scope, cfg.namespace)
    ids = [it.id for it in items if it.id is not None]
    if ids:
        await provider.delete(scope, cfg.namespace, ids)
    return len(ids)
