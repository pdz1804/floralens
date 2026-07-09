"""FloraLens API — FastAPI app.

Endpoints (Phase 0-1 scope):
  GET  /health            - liveness + active model_version
  GET  /api/health        - alias, same payload (matches PRD API surface prefix)
  POST /api/search        - image (multipart file OR JSON base64) -> top-K matches

Additive (PRD P5-6 — naturalist multi-agent assistant, reusing AgentForge's
Unified Agent Core, see assistant_service.py):
  POST /api/assistant (SSE) - chat with the naturalist agent team
"""
from __future__ import annotations

import base64
import binascii
import json
import logging

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent_core.errors import AgentCoreError

from apps.api.app.assistant_service import build_floralens_registries, compile_naturalist
from apps.api.app.config import settings
from apps.api.app.galaxy_service import get_galaxy_points
from apps.api.app.pipeline_service import build_preprocess_preview, get_pipeline_snapshot
from apps.api.app.search_service import (
    get_specimen_image_path,
    search_image,
    strip_exif_and_load,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FloraLens API", version="0.1.0")

# Built once at startup; tests may swap in a fake model provider via
# `assistant_registries.models.register(..., overwrite=True)` to run the
# naturalist agent offline (no API key / network spend).
assistant_registries = build_floralens_registries()

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


class HealthResponse(BaseModel):
    status: str
    model_version: str


class SearchResultOut(BaseModel):
    specimen_id: str
    label: int
    label_name: str
    score: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity, 0..1 range in practice")
    # Calibrated by a promoted model (Phase 3b); null when the baseline is active.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    band: str | None = None
    # Curated botanical description (display only); null if the label has no
    # curated entry. Never derived from or fed into embeddings/training.
    description: str | None = None


class SearchResponse(BaseModel):
    model_version: str
    results: list[SearchResultOut]


def _health_payload() -> HealthResponse:
    return HealthResponse(status="ok", model_version=settings.model_version)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return _health_payload()


@app.get("/api/health", response_model=HealthResponse)
def api_health() -> HealthResponse:
    return _health_payload()


class SearchBase64Request(BaseModel):
    image_base64: str


def _validate_and_decode_bytes(raw: bytes, declared_content_type: str | None) -> bytes:
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="empty image payload")
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"image exceeds max size of {settings.max_upload_bytes} bytes",
        )
    if declared_content_type and declared_content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"unsupported content type: {declared_content_type}")
    return raw


@app.post("/api/search", response_model=SearchResponse)
async def search(request: Request, file: UploadFile | None = File(default=None)) -> SearchResponse:
    content_type = request.headers.get("content-type", "")

    if file is not None:
        raw = await file.read()
        raw = _validate_and_decode_bytes(raw, file.content_type)
    elif content_type.startswith("application/json"):
        body = await request.json()
        try:
            payload = SearchBase64Request.model_validate(body)
        except Exception as exc:  # pydantic ValidationError
            raise HTTPException(status_code=400, detail=f"invalid request body: {exc}") from exc
        try:
            raw = base64.b64decode(payload.image_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="invalid base64 image data") from exc
        raw = _validate_and_decode_bytes(raw, None)
    else:
        raise HTTPException(
            status_code=400,
            detail="provide either a multipart 'file' upload or a JSON body with 'image_base64'",
        )

    try:
        image = strip_exif_and_load(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc

    try:
        model_version, results = search_image(image)
    except FileNotFoundError as exc:
        logger.error("gallery index unavailable: %s", exc)
        raise HTTPException(
            status_code=503, detail="search index not built yet; run the embedding pipeline first"
        ) from exc

    return SearchResponse(
        model_version=model_version,
        results=[
            SearchResultOut(
                specimen_id=r.specimen_id,
                label=r.label,
                label_name=r.label_name,
                score=r.score,
                confidence=r.confidence,
                band=r.band,
                description=r.description,
            )
            for r in results
        ],
    )


class PreprocessStepOut(BaseModel):
    name: str
    description: str
    # Base64 PNG of the image immediately after this step ran (bounded to
    # ~256px on the longest side) — powers the UI's per-step filmstrip.
    image_png_b64: str


class PreprocessPreviewResponse(BaseModel):
    steps: list[PreprocessStepOut]
    before_png_b64: str
    after_png_b64: str


@app.post("/api/preprocess-preview", response_model=PreprocessPreviewResponse)
async def preprocess_preview(file: UploadFile = File(...)) -> PreprocessPreviewResponse:
    """Run the CV preprocessing pipeline on an uploaded image and return the
    applied steps (each with its own after-step image, for a filmstrip) plus
    base64-encoded before/after PNGs, for the Pipeline page's preview. Same
    size/content-type bounds as /api/search."""
    raw = await file.read()
    raw = _validate_and_decode_bytes(raw, file.content_type)
    try:
        preview = build_preprocess_preview(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not process image: {exc}") from exc
    return PreprocessPreviewResponse(**preview)


@app.get("/api/pipeline")
def pipeline() -> dict:
    """Read-only snapshot of the active ML pipeline (dataset scale,
    preprocessing steps, backbone, val/test eval, calibration, promotion
    decision) sourced from the real eval reports / model card on disk — for
    the app's Pipeline page. No fixed response_model: the shape mirrors
    whatever the underlying eval/calibration/promotion JSON reports contain."""
    return get_pipeline_snapshot()


class GalaxyPointOut(BaseModel):
    specimen_id: str
    x: float
    y: float
    z: float
    label: int
    label_name: str
    color: str = Field(..., description="Stable per-species hex color, e.g. '#3fa06a'")


class GalaxyResponse(BaseModel):
    points: list[GalaxyPointOut]
    count: int


@app.get("/api/galaxy", response_model=GalaxyResponse)
def galaxy() -> GalaxyResponse:
    """3D projection of the gallery embeddings (PCA — see
    ml/scripts/build_galaxy_projection.py) powering the Galaxy tab's
    fly-through point cloud. The projection is built once (lazily, on first
    call) if missing, then served from an in-memory cache on every call after."""
    try:
        points = get_galaxy_points()
    except FileNotFoundError as exc:
        logger.error("galaxy projection source embeddings unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="embeddings cache not built yet; run the embedding pipeline first",
        ) from exc
    return GalaxyResponse(points=[GalaxyPointOut(**p) for p in points], count=len(points))


@app.get("/api/specimen/{specimen_id}/image")
def specimen_image(specimen_id: str) -> FileResponse:
    """Serve a matched specimen's thumbnail image so the UI can preview results.

    Only ids present in the dataset resolve (traversal-safe); unknown ids 404.
    """
    path = get_specimen_image_path(specimen_id)
    if path is None:
        raise HTTPException(status_code=404, detail="specimen image not found")
    return FileResponse(
        path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"}
    )


class AssistantRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str = "default"


def _assistant_error_event(detail: str) -> str:
    return f"data: {json.dumps({'type': 'error', 'detail': detail})}\n\n"


@app.post("/api/assistant")
async def assistant(req: AssistantRequest) -> StreamingResponse:
    """Chat with the naturalist agent team, streaming trace + answer as SSE.

    Compiles the `naturalist` manifest (agents/naturalist.yaml, delegating to
    the `care_advisor` sub-agent) via the shared `agent_core.compile_agent` —
    the exact same compile/stream path AgentForge's own `/api/runs` uses (see
    assistant_service.py). Mirrors that endpoint's SSE contract: a
    `run_started` event, one event per trace step, then `done` (or a
    structured `error` event on failure, never a broken stream).
    """

    async def event_stream():
        yield f"data: {json.dumps({'type': 'run_started'})}\n\n"
        try:
            agent = compile_naturalist(assistant_registries)
        except AgentCoreError as exc:
            yield _assistant_error_event(str(exc))
            return
        except Exception:
            logger.exception("failed to compile the naturalist agent")
            yield _assistant_error_event("failed to prepare the naturalist assistant")
            return

        try:
            async for event in agent.astream(req.message, thread_id=req.thread_id):
                yield f"data: {event.model_dump_json()}\n\n"
        except AgentCoreError as exc:
            yield _assistant_error_event(str(exc))
            return
        except Exception:
            logger.exception("naturalist assistant run failed")
            yield _assistant_error_event("internal error during assistant run")
            return

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
