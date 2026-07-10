# FloraLens API Reference

## Overview

The FloraLens API is a FastAPI application running on port 8100. All endpoints are RESTful (except `/api/assistant` which uses SSE streaming). Authentication and rate limiting are optional hardening features controlled by environment variables.

**Base URL:** `http://localhost:8100`

---

## Health Check

### GET /health

Liveness check + active model version.

**Response:** 200 OK
```json
{
  "status": "ok",
  "model_version": "finetuned_arcface_dinov2_v2"
}
```

---

### GET /api/health

Alias for `/health` (same payload). Matches PRD API surface prefix.

**Response:** 200 OK
```json
{
  "status": "ok",
  "model_version": "finetuned_arcface_dinov2_v2"
}
```

---

## Search

### POST /api/search

Upload or paste a flower image to find visually similar species. Returns top-K gallery matches ranked by calibrated confidence scores.

**Rate Limited:** Yes (per IP, configurable via `FLORALENS_SEARCH_LIMIT` env)

**Request:**

Option 1: Multipart file upload
```bash
curl -X POST http://localhost:8100/api/search \
  -F "file=@flower.jpg"
```

Option 2: Base64 JSON
```bash
curl -X POST http://localhost:8100/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
  }'
```

**Request Body (JSON option):**
```json
{
  "image_base64": "base64-encoded PNG/JPEG/WebP/BMP image"
}
```

**Supported Content Types:** `image/jpeg`, `image/png`, `image/webp`, `image/bmp`

**Size Limit:** Default 10 MB (configurable via `FLORALENS_MAX_UPLOAD_BYTES` env)

**Response:** 200 OK
```json
{
  "model_version": "finetuned_arcface_dinov2_v2",
  "results": [
    {
      "specimen_id": "ox102_rose_001",
      "label": 75,
      "label_name": "Rose",
      "score": 0.892,
      "confidence": 0.87,
      "band": "high",
      "description": "Rosa species with deep red petals, a symbol of love and romance."
    },
    {
      "specimen_id": "ox102_rose_002",
      "label": 75,
      "label_name": "Rose",
      "score": 0.871,
      "confidence": 0.81,
      "band": "high",
      "description": "Rosa damascena, prized for fragrance and ornamental value."
    }
  ]
}
```

**Response Schema:**

| Field | Type | Description |
|---|---|---|
| `model_version` | string | Active model version (e.g., "finetuned_arcface_dinov2_v2") |
| `results[].specimen_id` | string | Unique specimen identifier in the dataset |
| `results[].label` | integer | Class label (0–101, 102 Oxford Flowers classes) |
| `results[].label_name` | string | Species name (e.g., "Rose", "Tulip") |
| `results[].score` | float | Raw cosine similarity (-1 to 1, typically 0–1 range) |
| `results[].confidence` | float\|null | Calibrated probability (0–1); null if baseline model active |
| `results[].band` | string\|null | Confidence band: "high" (≥0.70), "medium" (0.40–0.69), "low" (<0.40); null if baseline |
| `results[].description` | string\|null | Curated botanical description; null if not available for this species |

**Error Responses:**

| Status | Detail |
|---|---|
| 400 | Empty image payload |
| 400 | Image exceeds max size (10 MB default) |
| 400 | Unsupported content type |
| 400 | Invalid base64 data |
| 400 | Could not decode image |
| 503 | Search index not built yet; run the embedding pipeline first |

---

## Image & Specimen

### GET /api/specimen/{specimen_id}/image

Fetch a specimen's thumbnail image. Used by the UI to preview search results and gallery items.

**Parameters:**
- `specimen_id` (path, required): Specimen ID (e.g., "ox102_rose_001")

**Response:** 200 OK (image/jpeg)
- JPEG image, ~256–512px on longest side
- Cache-Control: `public, max-age=3600`

**Error Responses:**

| Status | Detail |
|---|---|
| 404 | Specimen image not found |

---

### POST /api/preprocess-preview

Show the preprocessing pipeline steps applied to an uploaded image. Returns a filmstrip of before/after images at each step, useful for educational/debugging purposes.

**Request:** Multipart file upload (same as `/api/search`)
```bash
curl -X POST http://localhost:8100/api/preprocess-preview \
  -F "file=@flower.jpg"
```

**Response:** 200 OK
```json
{
  "steps": [
    {
      "name": "EXIF Strip",
      "description": "Remove embedded metadata",
      "image_png_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    },
    {
      "name": "Resize & Crop",
      "description": "Resize to 518×518, center-crop to 448×448",
      "image_png_b64": "..."
    },
    {
      "name": "White Balance",
      "description": "Apply color-preserving grey-world white balance",
      "image_png_b64": "..."
    }
  ],
  "before_png_b64": "...",
  "after_png_b64": "..."
}
```

**Response Schema:**

| Field | Type | Description |
|---|---|---|
| `steps[].name` | string | Pipeline step name |
| `steps[].description` | string | What this step does |
| `steps[].image_png_b64` | string | Base64-encoded PNG after this step (~256px on longest side) |
| `before_png_b64` | string | Original image, base64-encoded PNG |
| `after_png_b64` | string | Final preprocessed image, base64-encoded PNG |

**Error Responses:**

| Status | Detail |
|---|---|
| 400 | Could not process image |

---

### GET /api/pipeline

Read-only snapshot of the active ML pipeline (model card, dataset scale, preprocessing, backbone, calibration, promotion decision). Sourced from disk artifacts (`ml/eval/reports/`, `ml/models/`).

**Response:** 200 OK
```json
{
  "dataset": {
    "name": "Oxford 102 Flowers",
    "total_specimens": 8185,
    "classes": 102,
    "source": "Hugging Face Hub (dpdl-benchmark/oxford_flowers102)"
  },
  "splits": {
    "train": 1530,
    "val": 818,
    "test": 818,
    "gallery": 1632
  },
  "preprocessing": {
    "exif_strip": true,
    "resize": "518×518",
    "center_crop": "448×448",
    "white_balance": "grey-world (color-preserving)",
    "normalize": "ImageNet"
  },
  "backbone": {
    "name": "DINOv2 ViT-L/14",
    "frozen": true,
    "output_dim": 1024,
    "source": "OpenCLIP"
  },
  "head": {
    "type": "ArcFace projection head",
    "input_dim": 1024,
    "output_dim": 256,
    "margin": 0.5,
    "scale": 30.0,
    "training_loss": "ArcFace (additive angular margin)"
  },
  "calibration": {
    "method": "isotonic regression",
    "fit_split": "validation",
    "eval_split": "test",
    "ece_before": 0.499,
    "ece_after": 0.00005
  },
  "baseline_eval": {
    "recall_1": 0.9535,
    "recall_5": 0.9926,
    "recall_10": 0.9963,
    "map_10": 0.9083,
    "mrr": 0.9709
  },
  "finetuned_eval": {
    "recall_1": 0.9762,
    "recall_5": 0.9976,
    "recall_10": 0.9988,
    "map_10": 0.9471,
    "mrr": 0.9863
  },
  "promotion_decision": "PROMOTE",
  "active_model_version": "finetuned_arcface_dinov2_v2"
}
```

**Schema:** Mirrors the actual ML model card JSON; exact structure varies by pipeline run.

**Error Responses:**

| Status | Detail |
|---|---|
| 503 | Model card artifacts unavailable |

---

## Catalog & Exploration

### GET /api/categories

The 102 flower species, one entry each (gallery count, total count, representative specimen, stable color). Sorted by species name. Sourced from embeddings metadata and cached on first call.

**Response:** 200 OK
```json
{
  "categories": [
    {
      "label": 0,
      "label_name": "Anthurium",
      "gallery_count": 18,
      "total_count": 162,
      "specimen_id": "ox102_anthurium_001",
      "color": "#d64444"
    },
    {
      "label": 1,
      "label_name": "Artichoke",
      "gallery_count": 19,
      "total_count": 171,
      "specimen_id": "ox102_artichoke_001",
      "color": "#7cb342"
    }
  ],
  "count": 102,
  "total_specimens": 1632
}
```

**Response Schema:**

| Field | Type | Description |
|---|---|---|
| `categories[].label` | integer | Class label (0–101) |
| `categories[].label_name` | string | Species name |
| `categories[].gallery_count` | integer | Number of specimens in the gallery (searchable index) |
| `categories[].total_count` | integer | Total specimens of this species across all splits |
| `categories[].specimen_id` | string | Representative gallery specimen (for thumbnail) |
| `categories[].color` | string | Stable hex color for UI consistency (matches galaxy tab) |
| `count` | integer | Total number of species (102) |
| `total_specimens` | integer | Total gallery specimens (1,632) |

**Error Responses:**

| Status | Detail |
|---|---|
| 503 | Embeddings cache not built yet; run the embedding pipeline first |

---

### GET /api/gallery

3D projection of gallery embeddings (PCA) for the embedding galaxy visualization. Points are returned with stable per-species colors.

**Response:** 200 OK
```json
{
  "points": [
    {
      "specimen_id": "ox102_rose_001",
      "x": 0.123,
      "y": -0.456,
      "z": 0.789,
      "label": 75,
      "label_name": "Rose",
      "color": "#e84c3d"
    },
    {
      "specimen_id": "ox102_rose_002",
      "x": 0.124,
      "y": -0.455,
      "z": 0.790,
      "label": 75,
      "label_name": "Rose",
      "color": "#e84c3d"
    }
  ],
  "count": 1632
}
```

**Response Schema:**

| Field | Type | Description |
|---|---|---|
| `points[].specimen_id` | string | Specimen ID |
| `points[].x` | float | 3D PCA projection coordinate |
| `points[].y` | float | 3D PCA projection coordinate |
| `points[].z` | float | 3D PCA projection coordinate |
| `points[].label` | integer | Class label (0–101) |
| `points[].label_name` | string | Species name |
| `points[].color` | string | Stable hex color (matches categories tab) |
| `count` | integer | Total number of points (1,632) |

**Projection:** Fit on train+gallery embeddings, applied to all gallery points. Deterministic; same projection used consistently across sessions.

**Error Responses:**

| Status | Detail |
|---|---|
| 503 | Embeddings cache not built yet; run the embedding pipeline first |

---

## Naturalist Assistant

### POST /api/assistant (SSE)

Chat with the naturalist multi-agent team. Streams responses as Server-Sent Events (SSE). Requires optional API key (if `FLORALENS_API_KEY` env is set) and is rate-limited per IP.

**Rate Limited:** Yes (per IP)
**Requires API Key:** Yes (if `FLORALENS_API_KEY` env is set; otherwise optional)

**Request:** JSON
```bash
curl -X POST http://localhost:8100/api/assistant \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I care for a rose?",
    "thread_id": "user-123"
  }'
```

**Request Body:**
```json
{
  "message": "Your question (1–4000 characters)",
  "thread_id": "default"
}
```

| Field | Type | Default | Description |
|---|---|---|
| `message` | string | required | User query |
| `thread_id` | string | "default" | Conversation thread ID; use same ID to resume a conversation |

**Response:** 200 OK (text/event-stream)

The response is a stream of SSE events. Each event is a JSON object prefixed with `data:`.

```
data: {"type": "run_started"}

data: {"type": "step", "step_type": "agent_call", "agent_id": "naturalist_supervisor", "step_num": 1, "thought": "The user is asking about rose care..."}

data: {"type": "step", "step_type": "tool_use", "agent_id": "identifier", "tool_name": "embedding_search", "input": {"image": null, "query": "rose"}, "output": [...]}

data: {"type": "step", "step_type": "tool_use", "agent_id": "researcher", "tool_name": "web_search", "input": {"query": "rose care guide"}, "output": {"results": [...]}}

data: {"type": "step", "step_type": "answer", "agent_id": "care_advisor", "content": "Roses thrive with..."}

data: {"type": "done"}
```

**SSE Event Schema:**

| Event Type | Fields | Description |
|---|---|---|
| `run_started` | (none) | Agent compilation started |
| `step` | `step_type`, `agent_id`, step-specific fields | Reasoning, tool call, or final answer |
| `done` | (none) | Stream complete, assistant ready for next message |
| `error` | `detail` | Structured error; terminates stream |

**Agent Team:**
- **Naturalist Supervisor:** Routes user query to identifier, researcher, or care_advisor
- **Identifier:** Calls `embedding_search_tool` to find similar specimens
- **Researcher:** Calls `web_search_tool` to answer factual questions
- **Care-Advisor:** Synthesizes information + applies guardrails (educational disclaimers, no medical dosage claims)

**Memory:**
- **Long-term:** mem0 provider, per-user scope, namespace `floralens`
- **Short-term:** LangGraph checkpointer (if `FLORALENS_CHECKPOINT_DB` env is set), per thread_id
- Resuming a `thread_id` loads prior conversation state; assistant remembers the user's garden + preferences

**Guardrails:**
- Educational disclaimer for health/care advice
- Refuse medical dosage claims / toxic content
- Web search results are cited (URL + snippet)

**Error Responses:**

| Status | Detail |
|---|---|
| 401 | Unauthorized (if API key required and not provided) |
| 429 | Rate limited (too many requests from this IP) |
| 503 | Agent core compilation failed / API key not configured |

---

## My Garden

### GET /api/garden

List saved plants. Returns all specimens the user has saved.

**Requires API Key:** Yes (if `FLORALENS_API_KEY` env is set)

**Response:** 200 OK
```json
{
  "items": [
    {
      "specimen_id": "ox102_rose_001",
      "label_name": "Rose",
      "saved_at": "2026-07-10T14:32:00Z"
    },
    {
      "specimen_id": "ox102_tulip_005",
      "label_name": "Tulip",
      "saved_at": "2026-07-09T09:15:00Z"
    }
  ]
}
```

**Response Schema:**

| Field | Type | Description |
|---|---|---|
| `items[].specimen_id` | string | Specimen ID |
| `items[].label_name` | string | Species name |
| `items[].saved_at` | string | ISO 8601 timestamp (UTC) |

---

### POST /api/garden

Add a specimen to the garden (save a plant).

**Requires API Key:** Yes (if `FLORALENS_API_KEY` env is set)

**Request:** JSON
```bash
curl -X POST http://localhost:8100/api/garden \
  -H "Content-Type: application/json" \
  -d '{
    "specimen_id": "ox102_rose_001"
  }'
```

**Request Body:**
```json
{
  "specimen_id": "specimen-id-string"
}
```

**Response:** 201 Created
```json
{
  "specimen_id": "ox102_rose_001",
  "label_name": "Rose",
  "saved_at": "2026-07-10T14:32:00Z"
}
```

**Error Responses:**

| Status | Detail |
|---|---|
| 404 | Unknown specimen_id |
| 401 | Unauthorized (if API key required) |

---

### DELETE /api/garden/{specimen_id}

Remove a specimen from the garden.

**Requires API Key:** Yes (if `FLORALENS_API_KEY` env is set)

**Parameters:**
- `specimen_id` (path, required): Specimen ID to remove

**Response:** 204 No Content

**Error Responses:**

| Status | Detail |
|---|---|
| 404 | Specimen not saved in the garden |
| 401 | Unauthorized (if API key required) |

---

## Memory Inspector

### GET /api/memory

Inspect the assistant's long-term memories (mem0 provider). Shows what the assistant remembers about the user (preferences, past plants, etc.).

**Requires API Key:** Yes (if `FLORALENS_API_KEY` env is set)

**Response:** 200 OK
```json
{
  "items": [
    {
      "id": "mem_001",
      "text": "User prefers indoor plants in zone 6b",
      "meta": {"source": "assistant_chat", "created_at": "2026-07-08T10:00:00Z"}
    },
    {
      "id": "mem_002",
      "text": "User has a rose garden with 15 specimens",
      "meta": {"source": "garden_save", "created_at": "2026-07-09T14:30:00Z"}
    }
  ],
  "scope": "user",
  "namespace": "floralens"
}
```

**Response Schema:**

| Field | Type | Description |
|---|---|---|
| `items[].id` | string\|null | Memory ID (null if not retrievable) |
| `items[].text` | string | Memory content |
| `items[].meta` | object | Metadata (source, created_at, etc.) |
| `scope` | string | Memory scope ("user") |
| `namespace` | string | Memory namespace ("floralens") |

**Error Responses:**

| Status | Detail |
|---|---|
| 401 | Unauthorized (if API key required) |
| 503 | Memory provider not configured (mem0 not initialized) |

---

### DELETE /api/memory

Clear all of the user's long-term memories.

**Requires API Key:** Yes (if `FLORALENS_API_KEY` env is set)

**Response:** 200 OK
```json
{
  "deleted": 5
}
```

**Response Schema:**

| Field | Type | Description |
|---|---|---|
| `deleted` | integer | Number of memories deleted |

**Error Responses:**

| Status | Detail |
|---|---|
| 401 | Unauthorized (if API key required) |
| 503 | Memory provider not configured |

---

## Authentication & Rate Limiting

### Optional API Key Authentication

If `FLORALENS_API_KEY` environment variable is set, the following endpoints require the key in the `Authorization` header:

- POST `/api/assistant`
- GET/POST/DELETE `/api/garden`
- GET/DELETE `/api/memory`

**Header Format:**
```
Authorization: Bearer <API_KEY>
```

**Note:** By default (no env var set), authentication is **disabled** — all endpoints are public. This allows local development and demo use without extra setup.

### Rate Limiting

Two endpoints have per-IP rate limiting:

- **POST `/api/search`** — default limit: 30 requests per minute per IP
- **POST `/api/assistant`** — default limit: 10 requests per minute per IP

**Limit Configuration:** Set via `FLORALENS_SEARCH_LIMIT`, `FLORALENS_ASSISTANT_LIMIT` env vars (default: requests/minute per IP).

**Rate Limit Response:** 429 Too Many Requests

```json
{
  "detail": "Rate limit exceeded"
}
```

---

## Examples

### Complete Search Workflow

```bash
# 1. Upload a flower image
curl -X POST http://localhost:8100/api/search \
  -F "file=@rose.jpg"

# Response:
# {
#   "model_version": "finetuned_arcface_dinov2_v2",
#   "results": [
#     {
#       "specimen_id": "ox102_rose_001",
#       "label": 75,
#       "label_name": "Rose",
#       "score": 0.892,
#       "confidence": 0.87,
#       "band": "high",
#       "description": "Rosa species with deep red petals..."
#     }
#   ]
# }

# 2. Fetch the top result's image
curl http://localhost:8100/api/specimen/ox102_rose_001/image -o rose_specimen.jpg

# 3. Add to garden
curl -X POST http://localhost:8100/api/garden \
  -H "Content-Type: application/json" \
  -d '{"specimen_id": "ox102_rose_001"}'

# Response:
# {
#   "specimen_id": "ox102_rose_001",
#   "label_name": "Rose",
#   "saved_at": "2026-07-10T14:32:00Z"
# }

# 4. Chat with the assistant
curl -X POST http://localhost:8100/api/assistant \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I care for this rose?", "thread_id": "user-123"}'

# Response: stream of SSE events
```

---

## See Also

- **[Architecture](architecture.md)** — System design and service module breakdown
- **[ML Documentation](ml.md)** — Embedding model, training, calibration details
- **[Cross-Product Reuse](cross-product-reuse.md)** — Agent core design and extensibility
