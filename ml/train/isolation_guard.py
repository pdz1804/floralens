"""Test-set isolation guard (PRD §14.2 rule 3, §14.8): training/selection code
must never read the `test` split. This module is the single choke point
every training/selection entry point calls before touching any specimen ids,
so the guarantee is enforced in code, not just by convention.
"""
from __future__ import annotations

from typing import Any, Iterable


class TestSetLeakageError(RuntimeError):
    """Raised when a training/selection code path references a test-split id."""


def assert_no_test_ids(ids_used: Iterable[str], manifest: dict[str, Any]) -> None:
    """Raise `TestSetLeakageError` if any id in `ids_used` belongs to the test split.

    `ids_used` should be every specimen id a training or model-selection step
    reads (train embeddings, val embeddings/gallery used for early stopping
    and hyperparameter selection). The test split is only ever touched by the
    dedicated Phase 3b one-shot evaluation script, which does not call this
    guard.
    """
    test_ids = {rec["id"] for rec in manifest["records"] if rec["split"] == "test"}
    used = set(ids_used)
    leaked = used & test_ids
    if leaked:
        raise TestSetLeakageError(
            f"training/selection touched {len(leaked)} test-split id(s), e.g. {sorted(leaked)[:5]}"
        )
