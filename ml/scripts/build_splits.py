"""CLI entry point: ingest Oxford-102, build the stratified split manifest +
leakage report, and write both to ml/data/manifests/.

Usage:
    venv/Scripts/python.exe -m ml.scripts.build_splits
"""
from __future__ import annotations

import json
import logging
import sys
import time

from ml.data.dataset import load_specimen_pool
from ml.data.splits import build_split_manifest, save_manifest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    t0 = time.time()
    logger.info("loading specimen pool (downloads on first run)...")
    records, class_names = load_specimen_pool()
    logger.info("loaded %d records across %d classes in %.1fs", len(records), len(class_names), time.time() - t0)

    t0 = time.time()
    logger.info("building stratified split manifest + leakage guard (perceptual hash)...")
    manifest, leakage_report = build_split_manifest(records, class_names)
    logger.info("split built in %.1fs: %s", time.time() - t0, leakage_report["split_sizes"])

    if not leakage_report["is_leakage_free"]:
        logger.error("LEAKAGE DETECTED: %s", json.dumps(leakage_report, indent=2))
        sys.exit(1)

    save_manifest(manifest, leakage_report)
    logger.info("manifest + leakage report saved to ml/data/manifests/")
    logger.info("leakage_free=%s class_coverage=%s", leakage_report["is_leakage_free"], leakage_report["class_coverage_per_split"])


if __name__ == "__main__":
    main()
