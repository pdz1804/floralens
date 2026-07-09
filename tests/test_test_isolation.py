"""Test-set isolation guard: proves the training/selection code path never
reads the test split (PRD §14.2 rule 3, §14.8, US-6).
"""
import json
from pathlib import Path

import pytest

from ml.train.isolation_guard import TestSetLeakageError, assert_no_test_ids

_SYNTHETIC_MANIFEST = {
    "records": [
        {"id": "a-train-1", "split": "train"},
        {"id": "a-train-2", "split": "train"},
        {"id": "a-val-1", "split": "val"},
        {"id": "a-gallery-1", "split": "gallery"},
        {"id": "a-test-1", "split": "test"},
    ]
}


def test_train_and_val_ids_pass():
    assert_no_test_ids(["a-train-1", "a-train-2", "a-val-1", "a-gallery-1"], _SYNTHETIC_MANIFEST)


def test_test_id_raises():
    with pytest.raises(TestSetLeakageError):
        assert_no_test_ids(["a-train-1", "a-test-1"], _SYNTHETIC_MANIFEST)


def test_error_message_names_leaked_ids():
    with pytest.raises(TestSetLeakageError, match="a-test-1"):
        assert_no_test_ids(["a-test-1"], _SYNTHETIC_MANIFEST)


MANIFEST_PATH = Path("ml/data/manifests/split_manifest.json")


def test_real_sweep_experiment_logs_never_reference_test_ids():
    """If a hyperparameter-sweep run log exists, prove none of its recorded
    training/selection ids belong to the real test split. This is the
    build-blocking guard for US-6 applied to the actual Phase 3 artifacts."""
    experiments_dir = Path("ml/eval/reports/experiments")
    if not experiments_dir.exists() or not MANIFEST_PATH.exists():
        pytest.skip("no sweep experiment logs / manifest yet - run ml.scripts.run_finetune_sweep first")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    test_ids = {r["id"] for r in manifest["records"] if r["split"] == "test"}

    checked_any = False
    for path in experiments_dir.glob("*.json"):
        log = json.loads(path.read_text(encoding="utf-8"))
        ids_used = set(log.get("train_ids", [])) | set(log.get("val_ids", [])) | set(log.get("gallery_ids", []))
        if not ids_used:
            continue
        checked_any = True
        leaked = ids_used & test_ids
        assert not leaked, f"{path} referenced test-split ids: {sorted(leaked)[:5]}"

    if not checked_any:
        pytest.skip("sweep logs did not record id lists")
