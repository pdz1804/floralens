# FloraLens System Architecture

## Overview

FloraLens is a full-stack AI application for visual flower identification and discovery. The system consists of a **Next.js frontend** (React + Three.js) communicating with a **Python FastAPI backend** that orchestrates three core subsystems: ML embedding service, unified agent core, and persistence layers.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Frontend (Next.js + React)                          │
│                            Port 3100                                         │
│                                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌───────────┐           │
│   │  Search UI   │  │ 3D Galaxy    │  │Categories│  │ Assistant │           │
│   │              │  │ (Three.js)   │  │          │  │   Chat    │           │
│   └──────────────┘  └──────────────┘  └──────────┘  └───────────┘           │
│                                                                              │
│   ┌──────────────┐                                                          │
│   │   My Garden  │                                                          │
│   └──────────────┘                                                          │
└──────┬───────────────────────────────────────────────────────────────────────┘
       │ REST/JSON + SSE (POST /api/search, /api/assistant, etc.)
       │
┌──────▼───────────────────────────────────────────────────────────────────────┐
│                         Backend (FastAPI + Python)                           │
│                            Port 8100                                         │
│                                                                              │
│  ┌─────────────────────────────┬──────────────────────────────────────┐    │
│  │   ML Embedding Service       │  Unified Agent Core (AgentForge)    │    │
│  ├─────────────────────────────┼──────────────────────────────────────┤    │
│  │                             │                                      │    │
│  │ • DINOv2 ViT-L/14 backbone  │ • Agent manifests (YAML)             │    │
│  │   (frozen, 1024-d)          │ • LangGraph state machine            │    │
│  │                             │ • Tool registry + execution          │    │
│  │ • ArcFace projection head    │   - EmbeddingSearchTool              │    │
│  │   (1024→256 MLP)            │   - WebSearchTool                    │    │
│  │                             │ • Memory provider (mem0)             │    │
│  │ • Isotonic calibrator       │ • LangGraph checkpointer (SQLite)    │    │
│  │   (ECE: 0.499 → 0.00005)    │                                      │    │
│  │                             │                                      │    │
│  │ • In-memory cosine          │ • Supervisor → sub-agents            │    │
│  │   vector store (1632 gallery)│   (identifier, researcher,           │    │
│  │                             │    care_advisor)                    │    │
│  │ • Embeddings cache (NumPy)  │                                      │    │
│  └─────────────────────────────┴──────────────────────────────────────┘    │
│                                                                              │
│  Routes:                                                                   │
│    GET  /health, /api/health                                               │
│    POST /api/search                 (image → top-K matches + scores)       │
│    POST /api/assistant (SSE)        (chat, streaming)                      │
│    GET  /api/categories             (102 species grid)                     │
│    GET  /api/galaxy                 (3D projection points)                 │
│    GET  /api/pipeline               (ML model card snapshot)               │
│    POST /api/preprocess-preview     (show preprocessing steps)             │
│    GET  /api/specimen/{id}/image    (fetch specimen thumbnail)            │
│    GET/POST/DELETE /api/garden      (save/list/remove plants)             │
│    GET/DELETE /api/memory           (inspect/clear memories)              │
└──────┬───────────────────────────────────────────────────────────────────────┘
       │
       ├─────────────────┬──────────────────┬──────────────────┐
       │                 │                  │                  │
    ┌──▼──────┐   ┌──────▼─────┐    ┌──────▼────┐   ┌─────────▼────┐
    │Embeddings│   │   SQLite   │    │ Model     │   │   Precomputed│
    │ Cache    │   │   Store    │    │ Artifacts │   │   3D         │
    │          │   │            │    │           │   │   Projection │
    │• embeddings.npz     │    │• head.pt  │   │• projection.json│
    │• metadata.json      │    │• calibrator   │   │ (PCA, 1632    │
    │ (8185 specimens)    │    │  .pkl     │   │  points)       │
    │                     │    │• model_card.md    │                │
    │ Query: extract      │    │                   │   Load once,   │
    │ vector by id,       │    │  Per ModelVersion │   serve cached │
    │ compute cosine      │    │                   │                │
    │                     │    │ Fit on val,       │   Use for      │
    │ (1632 gallery only) │    │ eval on test      │   Galaxy tab   │
    └──────────┘   └──────────┘    └───────────┘   └────────────────┘
```

---

## Component Breakdown

### Frontend (Next.js + React)

**Location:** `apps/web/`

**Key Pages:**
- `app/search-page.tsx` — Image upload/paste, search results grid with calibrated scores
- `app/categories-page.tsx` — 102-species grid, click to inspect
- `app/galaxy-page.tsx` — 3D point-cloud viewer (Three.js, react-three-fiber)
- `app/assistant-page.tsx` — Chat interface with streaming responses and tool traces
- `app/garden-page.tsx` — Saved plants collection
- `app/pipeline-page.tsx` — ML model card, preprocessing flow visualization

**Styling:** Tailwind CSS; responsive design for mobile/tablet/desktop.

**Real-time Communication:** Fetch API for REST; EventSource API for SSE (assistant streaming).

---

### FastAPI Backend

**Location:** `apps/api/app/main.py` (route definitions)

**Service Modules:**

#### Search Service (`search_service.py`)
- Loads the active embedding model (DINOv2 + ArcFace head)
- Accepts image (multipart or base64 JSON)
- Strips EXIF metadata for privacy
- Embeds query image
- Queries the in-memory cosine vector store
- Calibrates raw scores to probabilities
- Returns top-K results with specimen metadata (label, description, confidence band)

#### Categories Service (`categories_service.py`)
- Loads embeddings metadata (102 classes)
- Returns per-class gallery/total counts and representative specimen IDs
- Sorted by species name
- Cached on first call

#### Galaxy Service (`galaxy_service.py`)
- Loads or builds (on-demand) the 3D PCA projection of gallery embeddings
- Returns points: `{specimen_id, x, y, z, label, label_name, color}`
- Color is stable per species (used consistently across galaxy and categories tabs)

#### Pipeline Service (`pipeline_service.py`)
- Serves read-only snapshots of the active model card
- Includes dataset scale, preprocessing steps, backbone details
- Reads from `ml/eval/reports/` and `ml/models/` disk artifacts
- Powers the Pipeline page

#### Assistant Service (`assistant_service.py`)
- Compiles the naturalist agent from YAML manifests (`agents/naturalist_supervisor.yaml`, etc.)
- Reuses AgentForge's Unified Agent Core (editable install)
- Implements domain-specific tools:
  - `EmbeddingSearchTool` — calls `/api/search` internally
  - `WebSearchTool` — calls Tavily API (via agent core)
- Streams responses as SSE events (type: `run_started`, `step`, `done`, `error`)
- Manages memory via mem0 provider
- Manages short-term thread state via SQLite checkpointer (if `FLORALENS_CHECKPOINT_DB` set)

#### Garden Service (`garden_service.py`)
- SQLite CRUD for saved plants (specimen_id, label_name, saved_at)
- Resolver: specimen_id → label_name (from embeddings metadata)

#### Memory Service (`memory_service.py`)
- Queries the mem0 provider used by the naturalist agent
- Allows users to inspect and delete long-term memories
- Scope: per-user, namespace: `floralens`

#### Config & Auth
- `config.py` — environment variables, settings
- `auth.py` — optional API key guard (`require_api_key`)
- `rate_limit.py` — per-IP rate limiting on search and assistant
- `redaction.py` — redact secrets (API keys) from logs and error messages

---

## ML Embedding Service

**Location:** `ml/` directory

### Backbone: DINOv2 ViT-L/14

- **Frozen:** The backbone is never trained; it outputs 1024-dimensional embeddings
- **Input:** RGB images (preprocessed: EXIF stripped, resized to 518×518, center-cropped, color-preserving grey-world white balance)
- **Output:** 1024-d dense vectors, ready for L2 normalization
- **Loading:** `ml/embeddings/backbone.py` handles device selection (`auto`, `cuda`, `cpu`) and model download (OpenCLIP/DINOv2)

### Projection Head: ArcFace

- **Architecture:** MLP (1024→hidden_dim→256) + L2 normalization
- **Loss:** ArcFace (additive angular margin, Deng et al. CVPR 2019)
- **Purpose:** Projects frozen backbone embeddings to a new space where same-species specimens cluster tightly on the unit hypersphere
- **Training:** Only this head is trained (fine-tuned on Oxford-102 train split; Phase 3)
- **Inference:** Normalizes output to unit length, enabling cosine similarity as the retrieval metric

### Calibrator: Isotonic Regression

- **Input:** Raw cosine similarity (0–1 range in practice)
- **Output:** Calibrated probability P(same species)
- **Training:** Fit on validation-set pairs only (Phase 3b)
- **Evaluation:** Expected Calibration Error (ECE) on test set (target ≤ 0.05)
- **Result:** ECE 0.499 (naive rescaling) → 0.00005 (after isotonic calibration)

### Vector Store: In-Memory Cosine Index

**Location:** `ml/index/vector_store.py`

- **Type:** In-memory NumPy arrays + metadata JSON
- **Composition:** Gallery partition only (1,632 specimens)
- **Query:** O(n) cosine similarity computation (acceptable for 1.6k specimens, p95 < 200ms)
- **Metadata:** specimen_id, label (0–101), label_name, description

### Embeddings Cache

**Location:** `ml/data/embeddings_cache/`

- `embeddings.npz` — NumPy-compressed array of all gallery embeddings (1632, 256)
- `metadata.json` — per-embedding metadata (id, label, label_name)
- Built once per model version; loaded into memory on API startup

---

## Data Flow

### Search Query Flow

```
User image (multipart/base64)
  ↓
[EXIF strip]
  ↓
[DINOv2 backbone: 1024-d]
  ↓
[ArcFace head: 256-d + L2-norm]
  ↓
[Cosine similarity vs gallery (1632 vectors)]
  ↓
[Top-K matches]
  ↓
[Calibrate scores → calibrated probabilities]
  ↓
[Assign confidence bands (high/medium/low)]
  ↓
[Hydrate with specimen metadata + descriptions]
  ↓
JSON response (specimen_id, label, label_name, score, confidence, band, description)
```

**Latency:** p95 < 800ms (target, PRD §5). GPU: ~100–200ms; CPU fallback: ~500–800ms.

### Assistant Chat Flow

```
User message
  ↓
[Compile naturalist agent from manifest]
  ↓
[LangGraph supervisor decision: which sub-agent?]
  ↓
[Identifier sub-agent (calls EmbeddingSearchTool)]
  │ └→ /api/search internally, returns top matches
  │
[Researcher sub-agent (calls WebSearchTool)]
  │ └→ Tavily web search + citation extraction
  │
[Care-Advisor sub-agent (synthesis)]
  └→ Combines results, applies guardrails
  ↓
[Memory augmentation: retrieve user's garden + preferences]
  ↓
[Stream answer as SSE events]
  │ ├→ run_started
  │ ├→ step (per tool call + reasoning)
  │ ├→ step (final answer)
  │ └→ done
  ↓
[Checkpointer saves thread state (if enabled)]
```

**Memory:** Long-term (mem0, user scope) + short-term (LangGraph checkpointer, thread scope).

### Training & Model Versioning Flow

```
Oxford-102 dataset (8,185 images)
  ↓
[Split builder: train/val/test/gallery, stratified by class]
  ├→ Leakage guard: near-duplicate detection
  └→ Produces split_manifest.json
  ↓
[Baseline: embed all splits with frozen DINOv2, evaluate]
  ├→ Recall@1 val=0.9607, test=0.9535
  └→ Baseline model activated
  ↓
[Fine-tuning (Phase 3):
   - Embed train subset (1530 base + 3 augmented views = 4590)
   - Hyperparameter sweep (8 ArcFace + 1 triplet config)
   - Early stop on val Recall@5
   - Select best candidate on val metrics only (test untouched)]
  ├→ Winner: run03_arcface_lr0.001_hd0_od256_m0.5
  └→ Candidate model (val Recall@1=0.956)
  ↓
[Test evaluation (Phase 3b, one-shot):
   - Embed test subset with candidate head
   - Evaluate on test split (same protocol)
   - Recall@1 test=0.9762 (beats baseline 0.9535)]
  ↓
[Calibration:
   - Fit calibrator on val pairs
   - Evaluate on test pairs (ECE target ≤ 0.05)
   - Achieved: ECE = 0.00005]
  ↓
[Promotion gate (§14.8):
   Check: Recall@5 improvement + val/test gap + ECE
   ✅ PROMOTE → Candidate becomes finetuned_arcface_dinov2_v2
   ↓
[Re-embed gallery under new version, upsert to index]
  ↓
[Activate in MODEL_VERSION env var]
```

**Artifacts:** All runs, EvalReports, calibrator, model weights stored in `ml/models/` and `ml/eval/reports/`.

---

## Persistence Layers

### Embeddings Cache (NumPy)

- **File:** `ml/data/embeddings_cache/embeddings.npz` + `metadata.json`
- **Format:** Compressed NumPy archive for speed
- **Lifecycle:** Built once by `ml.scripts.build_embeddings_index`; loaded into memory on API startup
- **Versioning:** Re-built and tagged per model version (new index created on model promotion)

### SQLite Stores

**Garden:**
- **File:** `ml/garden.db` (or path from `FLORALENS_GARDEN_DB` env)
- **Schema:** `garden_plants(specimen_id, label_name, saved_at)`
- **Lifecycle:** Persisted across API restarts; user-facing CRUD

**Memory (LangGraph Checkpointer):**
- **File:** Path from `FLORALENS_CHECKPOINT_DB` env (optional; default: disabled)
- **Schema:** LangGraph checkpointer schema (thread state snapshots)
- **Lifecycle:** One entry per thread_id; resuming a thread_id resumes prior conversation

### Model Artifacts

- **Location:** `ml/models/finetuned_arcface_dinov2_v2/` (version-specific)
- **Contents:**
  - `head.pt` — PyTorch weights for the ArcFace projection head
  - `calibrator.pkl` — Scikit-learn fitted isotonic regressor (or Platt calibrator)
  - `metadata.json` — hyperparams, dataset hash, seed, train/val/test metrics
  - `model_card.md` — human-readable summary

### ML Experiment Tracking

- **Tool:** MLflow (default) or Weights & Biases (optional)
- **Backend:** SQLite-backed (local) or cloud
- **Artifacts:** Each run logs config, metrics curves, dataset hash, seed, git commit
- **Purpose:** Reproducibility + hyperparameter audit trail

---

## Service Module Dependencies

```
main.py (entry point, route definitions)
  ├── search_service.py
  │   ├── ml/embeddings/backbone.py (DINOv2 loader)
  │   ├── ml/train/head.py (ArcFace head loader)
  │   ├── ml/eval/calibration.py (score calibrator)
  │   ├── ml/index/vector_store.py (cosine search)
  │   └── ml/embeddings/cache.py (embeddings loader)
  │
  ├── categories_service.py
  │   └── ml/embeddings/cache.py (metadata reader)
  │
  ├── galaxy_service.py
  │   ├── ml/embeddings/cache.py (embeddings)
  │   └── ml/scripts/build_galaxy_projection.py (PCA projection)
  │
  ├── pipeline_service.py
  │   ├── ml/eval/reports/ (JSON artifacts)
  │   └── ml/models/ (model cards)
  │
  ├── assistant_service.py
  │   ├── agent_core (Unified Agent Core, editable install)
  │   ├── agent_core/registries (tools, prompts, memory)
  │   ├── agents/ (naturalist manifests)
  │   └── [search_service for EmbeddingSearchTool]
  │
  ├── garden_service.py
  │   └── sqlite (garden.db)
  │
  ├── memory_service.py
  │   ├── agent_core/memory (mem0 provider)
  │   └── assistant_registries
  │
  └── auth.py, rate_limit.py, redaction.py (middleware)
```

---

## Deployment Considerations

### Local Development
- API: `uvicorn apps.api.app.main:app --reload --port 8100`
- Web: `npm run dev` in apps/web (port 3100)
- ML: Embeddings cache auto-loads on first search; galaxy projection auto-built on first galaxy request
- No external services required; .env loads from project root + sibling agentforge/

### Production (Future)
- Vector index → pgvector (Postgres) for scalability beyond 100k specimens
- Session state → Redis (for distributed assistant checkpointing)
- Model artifacts → S3-compatible object store
- Experiment tracking → MLflow server or W&B cloud
- Images → Object store (S3/MinIO) with CDN

---

## Error Handling & Graceful Degradation

| Scenario | Behavior |
|---|---|
| **Search with missing embeddings cache** | HTTP 503, "search index not built yet" |
| **Assistant with OPENAI_API_KEY unset** | HTTP 503, "agent core error: API key not configured" |
| **Invalid image upload** | HTTP 400, "could not decode image: ..." |
| **Specimen ID not found in gallery** | HTTP 404, "specimen image not found" |
| **Memory not configured** | HTTP 503, "memory provider not configured" |
| **Tool call failure (web search offline)** | Partial answer + note "web search unavailable" |

---

## Performance Targets (PRD §5, §12)

| Metric | Target | Status |
|---|---|---|
| Search latency (p95) | ≤ 800 ms | ✅ Measured ~100–200 ms (GPU), ~500–800 ms (CPU) |
| Galaxy interactive first frame | ≤ 2.5 s | ✅ ~500 ms with instanced rendering |
| Retrieval Recall@1 (test) | ≥ 80% | ✅ 97.62% (finetuned_arcface_dinov2_v2) |
| Calibration ECE (test) | ≤ 0.05 | ✅ 0.00005 |
| Assistant answer citation | ≥ 90% | ⏳ In evaluation |

---

## References

- **[API Reference](api.md)** — Endpoint schemas and examples
- **[ML Deep-Dive](ml.md)** — Training protocol, metrics, calibration details
- **[Cross-Product Reuse](cross-product-reuse.md)** — Agent core design and extensibility
- **[PRD.md](../PRD.md)** — Product requirements, system architecture, success metrics
