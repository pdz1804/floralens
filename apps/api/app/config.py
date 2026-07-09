"""API configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_version: str = os.environ.get("MODEL_VERSION", "baseline")
    embeddings_cache_dir: str = os.environ.get(
        "EMBEDDINGS_CACHE_DIR", "ml/data/embeddings_cache"
    )
    max_upload_bytes: int = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    default_top_k: int = int(os.environ.get("DEFAULT_TOP_K", "12"))


settings = Settings()
