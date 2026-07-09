"""Search service: builds the gallery vector index from the cached embeddings
and exposes `search_image(pil_image, top_k)` used by the /api/search route.
"""
from __future__ import annotations

import functools
import io
import logging

import numpy as np
from PIL import Image

from apps.api.app.config import settings
from ml.embeddings.backbone import embed_image
from ml.embeddings.cache import load_embeddings
from ml.index.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SearchResultItem:
    __slots__ = ("specimen_id", "label", "label_name", "score")

    def __init__(self, specimen_id: str, label: int, label_name: str, score: float) -> None:
        self.specimen_id = specimen_id
        self.label = label
        self.label_name = label_name
        self.score = score


@functools.lru_cache(maxsize=1)
def get_gallery_store() -> VectorStore:
    """Build (once) the in-memory gallery vector store from the embeddings cache.

    Only specimens tagged split=="gallery" are indexed, matching the PRD
    rule that production search never queries against held-out val/test
    query partitions (§14.7).
    """
    vectors, metadata = load_embeddings(settings.embeddings_cache_dir)
    model_version = metadata.get("model_version", settings.model_version)
    store = VectorStore(model_version=model_version)
    specimens = metadata["specimens"]
    indexed = 0
    for specimen_id, vector in vectors.items():
        meta = specimens[specimen_id]
        if meta["split"] != "gallery":
            continue
        store.add(
            specimen_id,
            vector,
            {
                "label": meta["label"],
                "label_name": meta["label_name"],
                "image_path": meta["image_path"],
            },
        )
        indexed += 1
    logger.info("gallery vector store ready: %d specimens (model_version=%s)", indexed, model_version)
    return store


def reset_gallery_store_cache() -> None:
    """Test helper: clears the cached gallery store singleton."""
    get_gallery_store.cache_clear()


def strip_exif_and_load(image_bytes: bytes) -> Image.Image:
    """Decode image bytes and return a clean RGB copy with EXIF metadata dropped."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img.load()
        clean = Image.new("RGB", img.size)
        clean.paste(img.convert("RGB"))
        return clean


def search_image(image: Image.Image, top_k: int | None = None) -> tuple[str, list[SearchResultItem]]:
    """Embed the query image and return (model_version, ranked results)."""
    top_k = top_k or settings.default_top_k
    store = get_gallery_store()
    vector: np.ndarray = embed_image(image)
    results = store.query(vector, top_k=top_k)
    items = [
        SearchResultItem(
            specimen_id=r.id,
            label=r.metadata["label"],
            label_name=r.metadata["label_name"],
            score=r.score,
        )
        for r in results
    ]
    return store.model_version, items
