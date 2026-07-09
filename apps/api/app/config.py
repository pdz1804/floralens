"""API configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # "baseline" = frozen zero-shot backbone, no projection head (Phase 0-1
    # behavior). Any other value is looked up as a ModelVersion artifact
    # directory under `models_dir` (Phase 3b promotion switch); if that
    # directory doesn't exist, search_service falls back to baseline so an
    # unset/typo'd env var never breaks search.
    #
    # Default is "finetuned_arcface_v1": Phase 3b's promotion gate PROMOTEd
    # this candidate on 2026-07-09 (see ml/eval/reports/promotion_decision.json)
    # — test Recall@5 0.9902 vs baseline 0.9779, val/test gap 0.0025, ECE
    # 0.0004. Override with MODEL_VERSION=baseline to roll back.
    model_version: str = os.environ.get("MODEL_VERSION", "finetuned_arcface_v1")
    embeddings_cache_dir: str = os.environ.get(
        "EMBEDDINGS_CACHE_DIR", "ml/data/embeddings_cache"
    )
    models_dir: str = os.environ.get("MODELS_DIR", "ml/models")
    max_upload_bytes: int = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    default_top_k: int = int(os.environ.get("DEFAULT_TOP_K", "12"))


settings = Settings()
