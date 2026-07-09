"""Deterministic stratified subset selection for the compute-bound embedding step.

The full split manifest (built over all ~8189 images) is correct at full
scale, but embedding every image on CPU is slow. For Phase 1 we embed a
documented, stratified subset: up to `per_class_cap[split]` images per class
per split, selected deterministically (stable sort by id, no randomness
beyond what's already baked into the manifest's split assignment). `train`
is not embedded at all in Phase 1 — the backbone is frozen and no training
happens yet, so train images are irrelevant until Phase 3.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

DEFAULT_PER_CLASS_CAP = {"gallery": 10, "val": 4, "test": 4, "train": 0}


def select_embedding_subset(
    manifest: dict[str, Any], per_class_cap: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    cap = per_class_cap or DEFAULT_PER_CLASS_CAP
    by_split_class: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in manifest["records"]:
        by_split_class[(rec["split"], rec["label"])].append(rec)

    selected: list[dict[str, Any]] = []
    for (split, _label), recs in by_split_class.items():
        limit = cap.get(split, 0)
        if limit <= 0:
            continue
        recs_sorted = sorted(recs, key=lambda r: r["id"])
        selected.extend(recs_sorted[:limit])
    return sorted(selected, key=lambda r: r["id"])
