"""CLI entry point: embed the FULL gallery/val/test partitions from the split
manifest with the frozen backbone (+ CV preprocessing), and cache vectors +
metadata. This is the "bigger dataset" upgrade — every gallery/val/test image
is embedded (not a stratified per-class subset), at whatever backbone is
currently active (`ml.embeddings.backbone`, DINOv3 preferred / DINOv2
fallback — see that module's docstring).

Usage:
    venv/Scripts/python.exe -m ml.scripts.build_embeddings_index
"""
from __future__ import annotations

import logging

from ml.data.splits import load_manifest
from ml.data.subset import select_full_eval_set
from ml.embeddings.cache import embed_records, save_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    manifest = load_manifest()
    subset = select_full_eval_set(manifest)
    logger.info(
        "selected %d/%d records to embed (full gallery+val+test, no per-class cap)",
        len(subset),
        manifest["num_records"],
    )
    vectors, metadata = embed_records(subset, model_version="baseline")
    save_embeddings(vectors, metadata)
    logger.info("saved embeddings cache: %d vectors", len(vectors))


if __name__ == "__main__":
    main()
