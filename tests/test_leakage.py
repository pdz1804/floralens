"""Leakage tests (PRD §14.2, §15.2 — build-blocking, hard assert).

Two layers:
1. A hermetic unit test that builds a small synthetic split manifest
   (including an intentional near-duplicate pair) and asserts the guard
   quarantines it to one split.
2. A hard assertion against the REAL split manifest produced by the Phase 1
   data pipeline (`ml/scripts/build_splits.py`): train/val/test/gallery must
   be disjoint by id AND by content hash. This test fails (as intended) if
   the manifest has not been generated yet.
"""
import json
from pathlib import Path

from PIL import Image

from ml.data.dataset import SpecimenRecord
from ml.data.splits import SplitFractions, build_split_manifest

MANIFEST_PATH = Path("ml/data/manifests/split_manifest.json")
LEAKAGE_REPORT_PATH = Path("ml/data/manifests/leakage_report.json")


def _write_solid_color_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), color=color).save(path, format="JPEG")


def test_synthetic_manifest_is_leakage_free_and_quarantines_duplicates(tmp_path):
    # Two classes, several distinct images each, plus one exact-duplicate
    # pair within class 0 (same bytes) that must land in a single split.
    records = []
    class_names = ["class_a", "class_b"]

    for cls in (0, 1):
        for i in range(12):
            img_path = tmp_path / f"c{cls}_{i}.jpg"
            color = (10 * i, 50 + cls * 40, 200 - 5 * i)
            _write_solid_color_image(img_path, color)
            content_hash = _sha1(img_path)
            records.append(
                SpecimenRecord(
                    id=f"id-{cls}-{i}",
                    image_path=str(img_path),
                    label=cls,
                    label_name=class_names[cls],
                    content_hash=content_hash,
                )
            )

    # Force an exact duplicate: id-0-11 gets identical bytes to id-0-0.
    dup_path = tmp_path / "c0_dup.jpg"
    _write_solid_color_image(dup_path, (0, 50, 200))  # same color as c0_0
    records[0] = SpecimenRecord(
        id="id-0-0", image_path=str(tmp_path / "c0_0.jpg"), label=0, label_name="class_a", content_hash=_sha1(tmp_path / "c0_0.jpg")
    )
    dup_record = SpecimenRecord(
        id="id-0-dup",
        image_path=str(dup_path),
        label=0,
        label_name="class_a",
        content_hash=_sha1(dup_path),
    )
    assert dup_record.content_hash == records[0].content_hash
    records.append(dup_record)

    manifest, leakage_report = build_split_manifest(
        records,
        class_names,
        fractions=SplitFractions(train=0.4, gallery=0.3, val=0.15, test=0.15, min_val=1, min_test=1, min_gallery=1),
        seed=7,
    )

    assert leakage_report["is_leakage_free"] is True
    assert leakage_report["id_overlaps"] == {}
    assert leakage_report["content_hash_overlaps"] == {}

    split_by_id = {r["id"]: r["split"] for r in manifest["records"]}
    assert split_by_id["id-0-0"] == split_by_id["id-0-dup"], "exact duplicates must share one split"


def _sha1(path: Path) -> str:
    import hashlib

    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def test_real_split_manifest_is_leakage_free():
    assert LEAKAGE_REPORT_PATH.exists(), (
        "leakage report not found — run `python -m ml.scripts.build_splits` first"
    )
    report = json.loads(LEAKAGE_REPORT_PATH.read_text(encoding="utf-8"))

    assert report["is_leakage_free"] is True
    assert report["id_overlaps"] == {}, f"id overlap across splits: {report['id_overlaps']}"
    assert report["content_hash_overlaps"] == {}, f"hash overlap across splits: {report['content_hash_overlaps']}"
    assert report["duplicate_group_overlaps"] == {}, (
        f"near-duplicate group spans multiple splits: {report['duplicate_group_overlaps']}"
    )

    # Every split should be non-empty and cover a healthy fraction of the 102 classes.
    for split in ("train", "val", "test", "gallery"):
        assert report["split_sizes"][split] > 0
        assert report["class_coverage_per_split"][split] >= 90  # allow minor class-size edge cases


def test_train_val_test_ids_disjoint_from_manifest_records():
    assert MANIFEST_PATH.exists(), "split manifest not found — run `python -m ml.scripts.build_splits` first"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    ids_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set(), "gallery": set()}
    hashes_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set(), "gallery": set()}
    for rec in manifest["records"]:
        ids_by_split[rec["split"]].add(rec["id"])
        hashes_by_split[rec["split"]].add(rec["content_hash"])

    assert ids_by_split["train"].isdisjoint(ids_by_split["val"])
    assert ids_by_split["train"].isdisjoint(ids_by_split["test"])
    assert ids_by_split["val"].isdisjoint(ids_by_split["test"])
    assert hashes_by_split["train"].isdisjoint(hashes_by_split["val"])
    assert hashes_by_split["train"].isdisjoint(hashes_by_split["test"])
    assert hashes_by_split["val"].isdisjoint(hashes_by_split["test"])
