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
    # Default is "finetuned_arcface_dinov2_v2": the color-preserving-
    # preprocessing + GPU retrain promotion gate PROMOTEd this candidate on
    # 2026-07-10 (see ml/eval/reports/promotion_decision.json and
    # ml/models/finetuned_arcface_dinov2_v2/model_card.md) — same DINOv2
    # ViT-L/14 backbone as v1 (DINOv3 is gated/unreachable in this
    # environment; see ml/embeddings/backbone.py), but the grey-world white
    # balance in ml/preprocess/pipeline.py was fixed (full-strength
    # correction was destroying saturated flower color, e.g. turning a red
    # rose brown) and the full gallery/val/test (3268 images) + train subset
    # were re-embedded on GPU under the fixed preprocessing. Test Recall@5
    # 0.9976 vs zero-shot baseline 0.9976 (re-measured under the same fixed
    # preprocessing) and vs the prior active "..._v1" candidate 0.9976,
    # val/test gap 0.0012, ECE 0.00005. This supersedes "finetuned_arcface_
    # dinov2_v1" (same backbone, pre-colorfix preprocessing, archived at
    # ml/models/archive_dinov2_v1/), which itself superseded
    # "finetuned_arcface_v1" (OpenCLIP-backbone, 512-d, archived at
    # ml/models/archive_openclip_v1/). Override with MODEL_VERSION=baseline
    # to roll back to the zero-shot DINOv2 backbone.
    model_version: str = os.environ.get("MODEL_VERSION", "finetuned_arcface_dinov2_v2")
    embeddings_cache_dir: str = os.environ.get(
        "EMBEDDINGS_CACHE_DIR", "ml/data/embeddings_cache"
    )
    models_dir: str = os.environ.get("MODELS_DIR", "ml/models")
    max_upload_bytes: int = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    default_top_k: int = int(os.environ.get("DEFAULT_TOP_K", "12"))
    # Device the embedding backbone (and any training run) executes on:
    # "auto" (default) picks cuda if available else cpu; "cuda"/"cpu" force a
    # choice (see ml.device.resolve_device, the single source of truth this
    # value is fed into). Raw string here — resolution happens lazily where
    # torch is actually imported, so importing this module never pulls in torch.
    device: str = os.environ.get("FLORALENS_DEVICE", "auto")
    # Opt-in durable short-term thread memory for the naturalist assistant
    # (PRD Phase 7 / E3). Unset by default -> the assistant stays single-shot
    # (each request starts from fresh state, no cross-turn resume), exactly as
    # before. When set to a filesystem path (or ":memory:"), compile_naturalist
    # wires agent_core's durable SQLite checkpointer
    # (agent_core.sqlite_checkpointer), so a multi-turn conversation keyed by
    # the request thread_id resumes prior state across requests — and, for a
    # file path, across process restarts. Empty string is treated as unset.
    checkpoint_db: str | None = os.environ.get("FLORALENS_CHECKPOINT_DB") or None
    # Opt-in long-term memory backend for the naturalist assistant (Gap G12).
    # Default "in_memory" is the process-local, dependency-free provider the
    # naturalist manifest (agents/naturalist.yaml) already declares, so unset ->
    # behavior is byte-for-byte identical to before (offline demo + existing
    # tests unaffected). Set FLORALENS_MEMORY_PROVIDER=mem0 to switch to
    # agent_core's semantic/persistent mem0 backend instead — both providers are
    # already registered by build_default_registries, and assistant_service
    # overrides the manifest's provider with this value at load time so the
    # compiled agent AND the /api/memory inspector read the same store. mem0 is
    # lazily imported (see agent_core.memory.mem0_provider): selecting it
    # requires `pip install 'mem0ai>=0.1'` + OPENAI_API_KEY; the default path
    # never imports mem0, so a missing package can never break the app.
    memory_provider: str = os.environ.get("FLORALENS_MEMORY_PROVIDER", "in_memory")


settings = Settings()
