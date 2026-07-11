"""Deterministic selection of which manifest records get embedded.

`select_full_eval_set` embeds the FULL gallery/val/test partitions (every
image in those three splits, no per-class cap) — the "bigger dataset"
upgrade: the prior smaller subset (~1801 images, `select_embedding_subset`
below) covered a stratified fraction of each split; embedding the full
gallery/val/test partitions instead uses the entire ~3200+ image eval corpus
at full retrieval-eval scale. `train` is still not embedded here — it is
irrelevant until fine-tuning (`ml.scripts.embed_train_subset` handles the
train-only, augmented, capped subset used to fit the projection head).

`select_embedding_subset` (with a per-class cap) is kept for a fast, smaller
dev/smoke run — e.g. local iteration on preprocessing/backbone code without
paying the full embedding cost.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

DEFAULT_PER_CLASS_CAP = {"gallery": 10, "val": 4, "test": 4, "train": 0}
FULL_EVAL_SPLITS = ("gallery", "val", "test")


def select_full_eval_set(manifest: dict[str, Any], splits: tuple[str, ...] = FULL_EVAL_SPLITS) -> list[dict[str, Any]]:
    """Every record in `splits` (default gallery+val+test), no cap."""
    selected = [rec for rec in manifest["records"] if rec["split"] in splits]
    return sorted(selected, key=lambda r: r["id"])


def select_embedding_subset(
    manifest: dict[str, Any], per_class_cap: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    """Deterministic, per-class-capped subset of manifest records for a fast dev/smoke run.

    For each (split, class) group, records are sorted by id and only the
    first `per_class_cap[split]` are kept; splits with a cap <= 0 (default
    `train`) are skipped entirely. Defaults to `DEFAULT_PER_CLASS_CAP`. The
    returned list is sorted by id, so the selection and its order are stable
    across runs given the same manifest.
    """
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
