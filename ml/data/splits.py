"""Stratified train/val/test/gallery split builder with a near-duplicate guard.

Split policy (PRD §14.1-14.2):
  - Four disjoint partitions, stratified by class: train, val, test, gallery.
    `gallery` plays the role of the "gallery-hold" partition in the PRD data
    model — it is the corpus that gets indexed for production search AND
    reused as the retrieval gallery when evaluating val/test queries (§14.4).
    Reusing one gallery for both eval splits keeps the protocol simple while
    still giving every class enough neighbors to make Recall@K meaningful.
  - Leakage prevention:
      1. Every image id is assigned to exactly one split.
      2. Near-duplicate detection via perceptual hash (imagehash.phash),
         compared only *within each class* (different flower species are
         never near-duplicates of each other, so this keeps an O(n^2)
         algorithm cheap while staying correct). Near-duplicate images are
         unioned into a single "duplicate group" that is assigned to one
         split atomically, so duplicates can never straddle a split boundary.
      3. Augmentation/normalization statistics are computed downstream on
         train only (not implemented in Phase 1 — the backbone is frozen).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml.data.dataset import SpecimenRecord

PHASH_HAMMING_THRESHOLD = 6  # imagehash 64-bit phash; <=6 bits differing ~= near-duplicate
DEFAULT_SEED = 42


@dataclass(frozen=True)
class SplitFractions:
    """Target per-class split proportions, with floors for small classes.

    `train`/`gallery`/`val`/`test` are the desired fraction of each class's
    duplicate-group-deduplicated records assigned to that split. The `min_*`
    floors guarantee val/test/gallery each get enough examples even for
    classes near the dataset's minimum class size, where a raw fraction could
    otherwise round down to zero.
    """

    train: float = 0.60
    gallery: float = 0.20
    val: float = 0.10
    test: float = 0.10
    min_val: int = 3
    min_test: int = 3
    min_gallery: int = 5


def _perceptual_hash(image_path: str) -> str:
    import imagehash
    from PIL import Image

    with Image.open(image_path) as img:
        return str(imagehash.phash(img.convert("RGB")))


class _UnionFind:
    def __init__(self, ids: list[str]) -> None:
        """Initialize each id as its own singleton duplicate group."""
        self._parent = {i: i for i in ids}

    def find(self, x: str) -> str:
        """Return the representative id of `x`'s duplicate group, compressing the path."""
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        """Merge the duplicate groups containing `a` and `b` into one group."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _cluster_near_duplicates(
    records: list[SpecimenRecord], phashes: dict[str, str], threshold: int = PHASH_HAMMING_THRESHOLD
) -> dict[str, str]:
    """Union-find near-duplicates within each class. Returns id -> group_id."""
    import imagehash

    by_class: dict[int, list[SpecimenRecord]] = defaultdict(list)
    for r in records:
        by_class[r.label].append(r)

    uf = _UnionFind([r.id for r in records])
    for class_records in by_class.values():
        hashes = [imagehash.hex_to_hash(phashes[r.id]) for r in class_records]
        n = len(class_records)
        for i in range(n):
            for j in range(i + 1, n):
                if hashes[i] - hashes[j] <= threshold:
                    uf.union(class_records[i].id, class_records[j].id)

    return {r.id: uf.find(r.id) for r in records}


def build_split_manifest(
    records: list[SpecimenRecord],
    class_names: list[str],
    fractions: SplitFractions = SplitFractions(),
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a stratified, leakage-free split manifest + leakage report.

    Returns (manifest, leakage_report). Both are JSON-serializable dicts.
    """
    import random

    rng = random.Random(seed)

    # Exact-duplicate quarantine first (content hash), then near-duplicate
    # clustering (perceptual hash) — both feed the same "duplicate group"
    # concept so a group of exact+near dupes is assigned to one split.
    phashes = {r.id: _perceptual_hash(r.image_path) for r in records}
    group_of = _cluster_near_duplicates(records, phashes)

    # Also union records sharing an exact content hash (covers cases where
    # phash distance narrowly misses the threshold but bytes are identical).
    by_hash: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_hash[r.content_hash].append(r.id)
    hash_groups = {ids[0]: ids for ids in by_hash.values() if len(ids) > 1}
    id_to_final_group: dict[str, str] = dict(group_of)
    for representative, ids in hash_groups.items():
        canonical = group_of[representative]
        for i in ids:
            id_to_final_group[i] = canonical

    # Build class -> list of duplicate-group-ids (each group kept intact).
    groups_by_class: dict[int, dict[str, list[SpecimenRecord]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups_by_class[r.label][id_to_final_group[r.id]].append(r)

    split_assignment: dict[str, str] = {}
    per_class_counts: dict[int, dict[str, int]] = {}

    for label, groups in groups_by_class.items():
        group_ids = list(groups.keys())
        rng.shuffle(group_ids)
        class_size = sum(len(v) for v in groups.values())

        target_val = max(fractions.min_val, round(class_size * fractions.val))
        target_test = max(fractions.min_test, round(class_size * fractions.test))
        target_gallery = max(fractions.min_gallery, round(class_size * fractions.gallery))

        counts = {"train": 0, "val": 0, "test": 0, "gallery": 0}
        for gid in group_ids:
            members = groups[gid]
            # Assign the whole group to whichever under-quota split needs it
            # most (val/test/gallery first, since they have hard minimums;
            # anything left over falls to train).
            if counts["val"] < target_val:
                dest = "val"
            elif counts["test"] < target_test:
                dest = "test"
            elif counts["gallery"] < target_gallery:
                dest = "gallery"
            else:
                dest = "train"
            for m in members:
                split_assignment[m.id] = dest
            counts[dest] += len(members)
        per_class_counts[label] = counts

    manifest_records = []
    for r in records:
        manifest_records.append(
            {
                "id": r.id,
                "image_path": r.image_path,
                "label": r.label,
                "label_name": r.label_name,
                "content_hash": r.content_hash,
                "phash": phashes[r.id],
                "duplicate_group": id_to_final_group[r.id],
                "split": split_assignment[r.id],
            }
        )

    dataset_hash = hashlib.sha1(
        "|".join(f"{r['id']}:{r['content_hash']}:{r['label']}" for r in sorted(manifest_records, key=lambda r: r["id"])).encode(
            "utf-8"
        )
    ).hexdigest()

    manifest = {
        "dataset": "oxford-102-flowers",
        "dataset_hash": dataset_hash,
        "seed": seed,
        "fractions": asdict(fractions),
        "phash_threshold": PHASH_HAMMING_THRESHOLD,
        "num_classes": len(class_names),
        "class_names": class_names,
        "num_records": len(records),
        "records": manifest_records,
        "per_class_counts": {str(k): v for k, v in per_class_counts.items()},
    }

    leakage_report = _build_leakage_report(manifest_records)
    return manifest, leakage_report


def _build_leakage_report(manifest_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, set[str]] = defaultdict(set)
    hash_by_split: dict[str, set[str]] = defaultdict(set)
    group_by_split: dict[str, set[str]] = defaultdict(set)

    for rec in manifest_records:
        split = rec["split"]
        by_split[split].add(rec["id"])
        hash_by_split[split].add(rec["content_hash"])
        group_by_split[split].add(rec["duplicate_group"])

    splits = ["train", "val", "test", "gallery"]
    id_overlaps = {}
    hash_overlaps = {}
    group_overlaps = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            id_overlap = by_split[a] & by_split[b]
            hash_overlap = hash_by_split[a] & hash_by_split[b]
            group_overlap = group_by_split[a] & group_by_split[b]
            if id_overlap:
                id_overlaps[f"{a}∩{b}"] = sorted(id_overlap)
            if hash_overlap:
                hash_overlaps[f"{a}∩{b}"] = sorted(hash_overlap)
            if group_overlap:
                group_overlaps[f"{a}∩{b}"] = sorted(group_overlap)

    counts = {s: len(by_split[s]) for s in splits}
    # Classes covered per split (sanity: every class should appear in every split)
    class_coverage = {
        s: len({r["label"] for r in manifest_records if r["split"] == s}) for s in splits
    }

    return {
        "split_sizes": counts,
        "class_coverage_per_split": class_coverage,
        "id_overlaps": id_overlaps,
        "content_hash_overlaps": hash_overlaps,
        "duplicate_group_overlaps": group_overlaps,
        "is_leakage_free": not id_overlaps and not hash_overlaps and not group_overlaps,
    }


def save_manifest(manifest: dict[str, Any], leakage_report: dict[str, Any], out_dir: str = "ml/data/manifests") -> None:
    """Write the split manifest and leakage report as pretty-printed JSON.

    Creates `out_dir` if it does not exist and writes `split_manifest.json`
    and `leakage_report.json` there, overwriting any existing files.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_path / "leakage_report.json").write_text(json.dumps(leakage_report, indent=2), encoding="utf-8")


def load_manifest(path: str = "ml/data/manifests/split_manifest.json") -> dict[str, Any]:
    """Load a previously saved split manifest JSON file into a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
