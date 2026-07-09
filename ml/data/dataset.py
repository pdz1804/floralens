"""Oxford 102 Flowers ingestion.

We pull images from the `dpdl-benchmark/oxford_flowers102` mirror on the
Hugging Face Hub (verified identical 102-class label schema/order to the
canonical torchvision.datasets.Flowers102, and to the original Oxford VGG
release) rather than the original robots.ox.ac.uk server, which proved slow
and unreliable to download from directly in this environment. This is a
transport-only substitution — same dataset, same license, same content.

We deliberately ignore the mirror's own train/validation/test assignment
(designed for classification, not our retrieval split policy). Instead we
pull the union of all three partitions into one flat pool of
(image_path, label) and build our OWN stratified train/val/test/gallery
split manifest downstream in `ml.data.splits` (PRD §14.1-14.2).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

_HF_DATASET_ID = "dpdl-benchmark/oxford_flowers102"
_HF_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SpecimenRecord:
    """One raw dataset record before split assignment."""

    id: str
    image_path: str
    label: int
    label_name: str
    content_hash: str


def _sha1_of_file(path: Path) -> str:
    hasher = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_specimen_pool(data_dir: str = "ml/data/raw") -> tuple[list[SpecimenRecord], list[str]]:
    """Download (if needed) and return the full 8189-image Oxford-102 pool.

    Images are materialized as JPEG files under
    `<data_dir>/flowers102_hf/images/` so downstream steps (content hashing,
    perceptual hashing, embedding) can treat them as plain files. Each
    record carries a stable id (e.g. "flowers102-00001-00734") and a sha1
    content hash used by the leakage guard to catch exact duplicates across
    splits. Returns (records, class_names).
    """
    from datasets import load_dataset

    images_dir = Path(data_dir) / "flowers102_hf" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: list[SpecimenRecord] = []
    class_names: list[str] = []
    seen_hashes: set[str] = set()

    for split in _HF_SPLITS:
        ds = load_dataset(_HF_DATASET_ID, split=split)
        if not class_names:
            class_names = list(ds.features["label"].names)

        for idx in range(len(ds)):
            example = ds[idx]
            specimen_id = f"flowers102-{split[:2]}{idx:05d}"
            image_path = images_dir / f"{specimen_id}.jpg"
            if not image_path.exists():
                example["image"].convert("RGB").save(image_path, format="JPEG", quality=95)
            content_hash = _sha1_of_file(image_path)
            if content_hash in seen_hashes:
                continue  # exact duplicate across the mirror's own splits; keep first occurrence
            seen_hashes.add(content_hash)
            label = int(example["label"])
            records.append(
                SpecimenRecord(
                    id=specimen_id,
                    image_path=str(image_path),
                    label=label,
                    label_name=class_names[label],
                    content_hash=content_hash,
                )
            )

    return sorted(records, key=lambda r: r.id), class_names
