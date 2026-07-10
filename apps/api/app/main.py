"""FloraLens API — FastAPI app.

Core endpoints (health + image search):
  GET  /health            - liveness + active model_version
  GET  /api/health        - alias, same payload (matches PRD API surface prefix)
  POST /api/search        - image (multipart file OR JSON base64) -> top-K matches

Additive (PRD P5-6 — naturalist multi-agent assistant, reusing AgentForge's
Unified Agent Core, see assistant_service.py):
  POST /api/assistant (SSE) - chat with the naturalist agent team

Additive (PRD Phase 7 — "My Garden" + assistant memory inspector):
  GET/POST/DELETE /api/garden  - save/list/remove a specimen (garden_service.py)
  GET/DELETE      /api/memory  - inspect/clear the assistant's memory (memory_service.py)

Additive (PRD Phase 9 — opt-in hardening; see auth.py, rate_limit.py,
redaction.py): sensitive/mutating endpoints (assistant, garden, memory) sit
behind `require_api_key`, a no-op unless `FLORALENS_API_KEY` is set, so the
local demo and existing tests are unaffected by default. `/api/search` and
`/api/assistant` are additionally per-IP rate limited, and secrets are
redacted from the assistant trace/error stream and from all log output.

Additive (per-user auth scaffold + data isolation — opt-in, see auth.py):
  POST /api/auth/token - DEV-ONLY: mint a bearer token for a user_id
  GET  /api/auth/me    - resolve the caller's user_id from its bearer token
/api/garden, /api/memory, and /api/assistant now resolve a `user_id` (via
`auth.resolve_user`) and scope their data to it. With `FLORALENS_JWT_SECRET`
unset (the default), resolution is a no-op returning `auth.DEFAULT_USER` for
every caller, so single-user behavior is unchanged from before this scaffold.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import sys
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from agent_core import sqlite_checkpointer
from agent_core.errors import AgentCoreError

from apps.api.app import garden_service, memory_service, models_service, specimen_service
from apps.api.app.env_loader import load_env_files
from apps.api.app.assistant_service import build_floralens_registries, compile_naturalist
from apps.api.app.auth import issue_token, jwt_auth_configured, require_api_key, resolve_user
from apps.api.app.categories_service import get_categories
from apps.api.app.config import settings
from apps.api.app.galaxy_service import get_galaxy_points
from apps.api.app.memory_service import MemoryNotConfiguredError
from apps.api.app.pipeline_service import build_preprocess_preview, get_pipeline_snapshot
from apps.api.app.rate_limit import assistant_rate_limit, search_rate_limit
from apps.api.app.redaction import RedactingLogFilter, redact_secrets
from apps.api.app.search_service import (
    get_specimen_image_path,
    search_image,
    strip_exif_and_load,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Scrubs secrets (API keys, bearer tokens, ...) from every log record
# process-wide, in case one ends up in an exception message (PRD Phase 9).
# The filter must sit on the HANDLERS, not the root logger: a logger's own
# filters only run for records logged directly on it, so records propagated
# up from named child loggers (logging.getLogger(__name__)) bypass a
# root-logger filter but still pass through the root's handlers.
_redacting_filter = RedactingLogFilter()
for _handler in logging.getLogger().handlers:
    _handler.addFilter(_redacting_filter)

app = FastAPI(title="FloraLens API", version="0.1.0")

# Local-dev convenience: the naturalist assistant reuses agent_core + its OpenAI/
# Tavily keys, but FloraLens has no .env of its own, so a plain `uvicorn` would
# start keyless and only fail on the first assistant run. Load FloraLens's own
# .env first (if any), then the sibling agentforge/.env for the shared keys.
# Skipped under pytest so tests never inherit a developer's .env; already-set
# env vars always win.
if "pytest" not in sys.modules:
    _fl_root = Path(__file__).resolve().parents[3]
    load_env_files(_fl_root / ".env", _fl_root.parent / "agentforge" / ".env")

# Built once at startup; tests may swap in a fake model provider via
# `assistant_registries.models.register(..., overwrite=True)` to run the
# naturalist agent offline (no API key / network spend).
assistant_registries = build_floralens_registries()

# Opt-in durable short-term thread memory (PRD Phase 7 / E3): when
# FLORALENS_CHECKPOINT_DB is set, every /api/assistant run is compiled with
# agent_core's durable SQLite checkpointer, so runs sharing a thread_id resume
# prior conversation state across requests (and, for a file path, across
# restarts). Built once here as a lightweight spec (the real async saver is
# materialized lazily, per run, inside the running event loop). Unset (the
# default) -> None -> single-shot runs, unchanged from before.
_assistant_checkpointer = (
    sqlite_checkpointer(settings.checkpoint_db) if settings.checkpoint_db else None
)

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


# --------------------------------------------------------------------------- #
# Per-user auth scaffold (additive, opt-in — see auth.py). `/api/auth/token`
# is a DEV-ONLY stand-in for a real login/OAuth flow, kept deliberately
# minimal so the rest of the scaffold (resolve_user, per-user garden/memory
# isolation) can be exercised end-to-end before that flow lands.
# --------------------------------------------------------------------------- #
class AuthTokenRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=200)


class AuthTokenResponse(BaseModel):
    token: str
    token_type: str = "bearer"


@app.post(
    "/api/auth/token",
    response_model=AuthTokenResponse,
    dependencies=[Depends(require_api_key)],
)
def issue_auth_token(req: AuthTokenRequest) -> AuthTokenResponse:
    """**DEV SCAFFOLD — NOT real authentication.** Mints a bearer token for
    whatever `user_id` the caller supplies, with no credential check beyond
    the shared `FLORALENS_API_KEY` (if one is configured via
    `require_api_key`) — there is no login, password, or OAuth flow behind
    it. A follow-up round MUST replace this endpoint with real
    login/OAuth(OIDC) before any multi-tenant/production deployment: as
    written, anyone able to call this endpoint can mint a token for ANY
    user_id and fully impersonate that user against /api/garden, /api/memory,
    and /api/assistant. It exists solely so the per-user data-isolation
    plumbing can be tested and demoed today. 404s when
    `FLORALENS_JWT_SECRET` isn't configured (auth off)."""
    if not jwt_auth_configured():
        raise HTTPException(
            status_code=404,
            detail="auth not configured: set FLORALENS_JWT_SECRET to enable token issuance",
        )
    token = issue_token(req.user_id)
    return AuthTokenResponse(token=token, token_type="bearer")


class AuthMeResponse(BaseModel):
    user_id: str


@app.get("/api/auth/me", response_model=AuthMeResponse)
def auth_me(user: str = Depends(resolve_user)) -> AuthMeResponse:
    """The caller's resolved user_id — `auth.DEFAULT_USER` when
    `FLORALENS_JWT_SECRET` is unset (auth off), else the `sub` claim of a
    valid bearer token (401 otherwise)."""
    return AuthMeResponse(user_id=user)


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


async def _load_query_image(request: Request, file: UploadFile | None) -> Image.Image:
    """Read a query image from the request (multipart 'file' OR a JSON body with
    'image_base64'), apply the same size/content-type bounds as the rest of the
    API, and return the preprocessed, ready-to-embed PIL image. Shared by
    /api/search and /api/embed so both accept identical inputs and raise the
    same 400s."""
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
        return strip_exif_and_load(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc


@app.post("/api/search", response_model=SearchResponse, dependencies=[Depends(search_rate_limit)])
async def search(request: Request, file: UploadFile | None = File(default=None)) -> SearchResponse:
    image = await _load_query_image(request, file)

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


class CategoryOut(BaseModel):
    label: int
    label_name: str
    gallery_count: int
    total_count: int
    # A representative gallery specimen id — fetch its thumbnail via
    # /api/specimen/{specimen_id}/image.
    specimen_id: str
    color: str = Field(..., description="Stable per-species hex color (matches the Galaxy tab)")


class CategoriesResponse(BaseModel):
    categories: list[CategoryOut]
    count: int
    total_specimens: int


@app.get("/api/categories", response_model=CategoriesResponse)
def categories() -> CategoriesResponse:
    """The 102 dataset species, one entry each (gallery/total counts, a
    representative gallery thumbnail id, and the same stable per-species color
    as the Galaxy tab), sorted by name — for the Categories tab. Sourced from
    the embeddings-cache metadata; built once and cached on first call."""
    try:
        entries = get_categories()
    except FileNotFoundError as exc:
        logger.error("categories source metadata unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="embeddings cache not built yet; run the embedding pipeline first",
        ) from exc
    return CategoriesResponse(
        categories=[CategoryOut(**e) for e in entries],
        count=len(entries),
        total_specimens=sum(e["total_count"] for e in entries),
    )


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


# --------------------------------------------------------------------------- #
# Model registry, specimen detail, and query embedding (PRD §11 REST surface).
# --------------------------------------------------------------------------- #
class ModelInfoOut(BaseModel):
    name: str
    active: bool
    # Best-effort, read from each version's metadata / eval report; omitted
    # (null) when the artifact doesn't carry it.
    backbone: str | None = None
    metrics: dict | None = None


class ModelsResponse(BaseModel):
    active: str
    models: list[ModelInfoOut]


@app.get("/api/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    """List the available model versions (the synthetic zero-shot 'baseline'
    plus every ModelVersion artifact directory under settings.models_dir),
    marking which one is active (settings.model_version) and surfacing each
    version's key retrieval metrics when its metadata/eval report has them.
    Read-only/best-effort — never loads the backbone; an environment without
    trained artifacts still returns the baseline entry."""
    data = models_service.get_models()
    return ModelsResponse(
        active=data["active"],
        models=[ModelInfoOut(**m) for m in data["models"]],
    )


class SpecimenDetailResponse(BaseModel):
    specimen_id: str
    label: int
    label_name: str
    split: str
    image_url: str


@app.get("/api/specimens/{specimen_id}", response_model=SpecimenDetailResponse)
def specimen_detail(specimen_id: str) -> SpecimenDetailResponse:
    """Richer detail for one specimen (label, species name, split, and the URL
    of its thumbnail) sourced from the embeddings-cache metadata. 404 for an
    unknown id; 503 if the cache hasn't been built yet (same contract as
    /api/categories)."""
    try:
        detail = specimen_service.get_specimen_detail(specimen_id)
    except FileNotFoundError as exc:
        logger.error("specimen detail source metadata unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="embeddings cache not built yet; run the embedding pipeline first",
        ) from exc
    if detail is None:
        raise HTTPException(status_code=404, detail="specimen not found")
    return SpecimenDetailResponse(**detail)


class EmbedResponse(BaseModel):
    model_version: str
    dim: int
    embedding: list[float]


def _embed_query_image(image: Image.Image) -> tuple[str, np.ndarray]:
    """Embed a query image via the exact backbone/projection path /api/search
    uses (search_service): the frozen backbone embedding, then — if a promoted
    candidate is active — projected through its head into the gallery's space.
    Returns (effective model_version, L2-normalized vector)."""
    from apps.api.app import search_service
    from ml.embeddings.backbone import embed_image

    vector = embed_image(image)
    candidate = search_service._load_candidate(settings.model_version)
    if candidate:
        from ml.train.model_io import project_embeddings

        head, _, _ = candidate
        vector = project_embeddings(head, vector.reshape(1, -1))[0]
        model_version = settings.model_version
    else:
        model_version = "baseline"

    # Backbone and head both emit L2-normalized vectors; re-normalize defensively
    # so the response's `embedding` is guaranteed unit-norm regardless of head.
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return model_version, vector


@app.post("/api/embed", response_model=EmbedResponse, dependencies=[Depends(search_rate_limit)])
async def embed(request: Request, file: UploadFile | None = File(default=None)) -> EmbedResponse:
    """Return the normalized embedding vector (and its dimensionality) for an
    uploaded image, using the same input handling and embedding path as
    /api/search. Accepts a multipart 'file' OR a JSON body with 'image_base64'.
    Rate-limited like /api/search."""
    image = await _load_query_image(request, file)
    try:
        model_version, vector = _embed_query_image(image)
    except FileNotFoundError as exc:
        logger.error("embedding backbone/model artifact unavailable: %s", exc)
        raise HTTPException(
            status_code=503, detail="embedding model not available; build the pipeline first"
        ) from exc
    return EmbedResponse(
        model_version=model_version,
        dim=int(vector.shape[0]),
        embedding=[float(x) for x in vector.tolist()],
    )


class AssistantRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str = "default"


def _assistant_error_event(detail: str) -> str:
    return f"data: {json.dumps({'type': 'error', 'detail': redact_secrets(detail)})}\n\n"


@app.post(
    "/api/assistant",
    dependencies=[Depends(require_api_key), Depends(assistant_rate_limit)],
)
async def assistant(req: AssistantRequest, user: str = Depends(resolve_user)) -> StreamingResponse:
    """Chat with the naturalist agent team, streaming trace + answer as SSE.

    Compiles the `naturalist` manifest (agents/naturalist.yaml, delegating to
    the `care_advisor` sub-agent) via the shared `agent_core.compile_agent` —
    the exact same compile/stream path AgentForge's own `/api/runs` uses (see
    assistant_service.py). Mirrors that endpoint's SSE contract: a
    `run_started` event, one event per trace step, then `done` (or a
    structured `error` event on failure, never a broken stream).

    `user` (resolved via `auth.resolve_user`) scopes the compiled agent's
    long-term memory to that user's namespace bucket; `auth.DEFAULT_USER`
    when auth is off, unchanged from before this scaffold.
    """

    async def event_stream():
        # Bound before the compile try so the finally can release it even if
        # compilation failed. A fresh agent is compiled per request, so without
        # aclose every run with a durable checkpointer set would leak a sqlite
        # connection + its background thread (no-op in the default setup).
        agent = None
        yield f"data: {json.dumps({'type': 'run_started'})}\n\n"
        try:
            try:
                agent = compile_naturalist(assistant_registries, _assistant_checkpointer, user)
            except AgentCoreError as exc:
                yield _assistant_error_event(str(exc))
                return
            except Exception:
                logger.exception("failed to compile the naturalist agent")
                yield _assistant_error_event("failed to prepare the naturalist assistant")
                return

            try:
                async for event in agent.astream(req.message, thread_id=req.thread_id):
                    # Redact before it ever leaves the process: a tool/model step
                    # could echo back a key from its input or a misconfigured env.
                    yield f"data: {redact_secrets(event.model_dump_json())}\n\n"
            except AgentCoreError as exc:
                yield _assistant_error_event(str(exc))
                return
            except Exception:
                logger.exception("naturalist assistant run failed")
                yield _assistant_error_event("internal error during assistant run")
                return

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        finally:
            if agent is not None:
                await agent.aclose()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# --------------------------------------------------------------------------- #
# My Garden (PRD Phase 7) — save/list/remove specimens, durable across
# restarts (see garden_service.py for the SQLite-backed store).
# --------------------------------------------------------------------------- #
class GardenAddRequest(BaseModel):
    specimen_id: str = Field(..., min_length=1, max_length=200)


class GardenItemOut(BaseModel):
    specimen_id: str
    label_name: str
    saved_at: str


class GardenListResponse(BaseModel):
    items: list[GardenItemOut]


@app.get("/api/garden", response_model=GardenListResponse, dependencies=[Depends(require_api_key)])
def list_garden(user: str = Depends(resolve_user)) -> GardenListResponse:
    items = garden_service.list_specimens(user)
    return GardenListResponse(
        items=[GardenItemOut(specimen_id=i.specimen_id, label_name=i.label_name, saved_at=i.saved_at) for i in items]
    )


@app.post(
    "/api/garden", response_model=GardenItemOut, status_code=201, dependencies=[Depends(require_api_key)]
)
def add_to_garden(req: GardenAddRequest, user: str = Depends(resolve_user)) -> GardenItemOut:
    label_name = garden_service.resolve_label_name(req.specimen_id)
    if label_name is None:
        raise HTTPException(status_code=404, detail=f"unknown specimen_id: {req.specimen_id}")
    item = garden_service.add_specimen(req.specimen_id, label_name, user)
    return GardenItemOut(specimen_id=item.specimen_id, label_name=item.label_name, saved_at=item.saved_at)


@app.delete(
    "/api/garden/{specimen_id}",
    status_code=204,
    response_model=None,
    dependencies=[Depends(require_api_key)],
)
def remove_from_garden(specimen_id: str, user: str = Depends(resolve_user)) -> None:
    if not garden_service.remove_specimen(specimen_id, user):
        raise HTTPException(status_code=404, detail="specimen not saved in the garden")


# --------------------------------------------------------------------------- #
# Assistant memory inspector (PRD Phase 7 / Epic E4) — view/clear what the
# naturalist assistant remembers (memory_service.py; same MemoryProvider
# instance + scope/namespace the compiled naturalist agent itself uses).
# --------------------------------------------------------------------------- #
class MemoryItemOut(BaseModel):
    id: str | None
    text: str
    meta: dict = Field(default_factory=dict)


class MemoryListResponse(BaseModel):
    items: list[MemoryItemOut]
    scope: str
    namespace: str


class MemoryDeleteResponse(BaseModel):
    deleted: int


def _memory_unavailable(exc: MemoryNotConfiguredError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


@app.get("/api/memory", response_model=MemoryListResponse, dependencies=[Depends(require_api_key)])
async def list_memory(user: str = Depends(resolve_user)) -> MemoryListResponse:
    try:
        items = await memory_service.list_memories(assistant_registries, user)
        scope, namespace = memory_service.memory_scope_and_namespace(user)
    except MemoryNotConfiguredError as exc:
        raise _memory_unavailable(exc) from exc
    return MemoryListResponse(
        items=[MemoryItemOut(id=i["id"], text=i["text"], meta=i["meta"]) for i in items],
        scope=scope,
        namespace=namespace,
    )


@app.delete("/api/memory", response_model=MemoryDeleteResponse, dependencies=[Depends(require_api_key)])
async def clear_memory(user: str = Depends(resolve_user)) -> MemoryDeleteResponse:
    try:
        deleted = await memory_service.clear_memories(assistant_registries, user)
    except MemoryNotConfiguredError as exc:
        raise _memory_unavailable(exc) from exc
    return MemoryDeleteResponse(deleted=deleted)
