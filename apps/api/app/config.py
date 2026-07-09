"""API configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # "baseline" = frozen zero-shot backbone, no projection head. Any other
    # value is looked up as a ModelVersion artifact directory under
    # `models_dir` (promotion switch); if that directory doesn't exist,
    # search_service falls back to baseline so an unset/typo'd env var never
    # breaks search.
    #
    # Default is "finetuned_arcface_dinov2_v1": the ML-pipeline-upgrade
    # promotion gate PROMOTEd this candidate on 2026-07-09 (see
    # ml/eval/reports/promotion_decision.json and
    # ml/models/finetuned_arcface_dinov2_v1/model_card.md) — backbone
    # upgraded to DINOv2 ViT-L/14 (DINOv3 is gated/unreachable in this
    # environment; see ml/embeddings/backbone.py), full gallery/val/test
    # (3268 images) + CV preprocessing re-embedded, test Recall@5 0.9976 vs
    # zero-shot baseline 0.9976 and vs the prior active OpenCLIP-backbone
    # candidate 0.9902, val/test gap 0.0012, ECE 0.0001. This supersedes
    # "finetuned_arcface_v1" (OpenCLIP-backbone, 512-d — dimensionally
    # incompatible with the new 1024-d DINOv2 gallery, archived at
    # ml/models/archive_openclip_v1/). Override with MODEL_VERSION=baseline
    # to roll back to the zero-shot DINOv2 backbone.
    model_version: str = os.environ.get("MODEL_VERSION", "finetuned_arcface_dinov2_v1")
    embeddings_cache_dir: str = os.environ.get(
        "EMBEDDINGS_CACHE_DIR", "ml/data/embeddings_cache"
    )
    models_dir: str = os.environ.get("MODELS_DIR", "ml/models")
    max_upload_bytes: int = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    default_top_k: int = int(os.environ.get("DEFAULT_TOP_K", "12"))


settings = Settings()
