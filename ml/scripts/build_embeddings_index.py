"""CLI entry point: select a stratified embedding subset from the split
manifest, embed it with the frozen backbone, and cache vectors + metadata.

Usage:
    venv/Scripts/python.exe -m ml.scripts.build_embeddings_index
"""
from __future__ import annotations

import logging

from ml.data.splits import load_manifest
from ml.data.subset import DEFAULT_PER_CLASS_CAP, select_embedding_subset
from ml.embeddings.cache import embed_records, save_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    manifest = load_manifest()
    subset = select_embedding_subset(manifest, DEFAULT_PER_CLASS_CAP)
    logger.info(
        "selected %d/%d records to embed (per-class caps: %s)",
        len(subset),
        manifest["num_records"],
        DEFAULT_PER_CLASS_CAP,
    )
    vectors, metadata = embed_records(subset, model_version="baseline")
    save_embeddings(vectors, metadata)
    logger.info("saved embeddings cache: %d vectors", len(vectors))


if __name__ == "__main__":
    main()
