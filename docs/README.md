# FloraLens Documentation

Welcome to the FloraLens documentation. This guide covers system architecture, API endpoints, machine learning implementation, and agent integration.

## Getting Started

- **[Quick Start](../README.md#quick-start)** — Set up the API backend and web frontend in minutes.
- **[Project Overview](../README.md)** — Features, capabilities, and high-level architecture.

## Documentation Index

### Core Documentation

- **[Architecture](architecture.md)** — System design, data flow, service modules, and component interaction.
  - FastAPI backend (:8100) + Next.js frontend (:3100)
  - ML embedding service, Unified Agent Core, persistence layers
  - Complete data flow diagram and service breakdown

- **[API Reference](api.md)** — Complete endpoint reference for the FastAPI backend.
  - Health check, image search, preprocessing, pipeline snapshots
  - 3D galaxy, species categories, specimen images
  - Naturalist assistant (SSE streaming)
  - My Garden CRUD, memory inspector
  - Request/response schemas and status codes

- **[Machine Learning](ml.md)** — Deep-dive into the fine-tuned embedding model and evaluation protocol.
  - Dataset: Oxford 102 Flowers (8,185 images, 102 classes)
  - Preprocessing: EXIF stripping, color-preserving white balance, CLAHE
  - **DINOv2 ViT-L/14 backbone** (frozen, 1024-d embeddings)
  - **ArcFace head** training (additive angular margin loss)
  - Isotonic calibration (ECE: 0.499 → 0.00005)
  - Metrics: Recall@1/5/10, mAP, MRR, silhouette
  - Promotion gate and model versioning

- **[Unified Agent Core Reuse](cross-product-reuse.md)** — How FloraLens leverages AgentForge's shared core.
  - Agent manifests (naturalist supervisor, identifier, researcher, care-advisor)
  - Embedding search tool + web search tool
  - Extensibility: adding new agents without core changes
  - Memory integration (mem0 + LangGraph checkpointer)

### Reference

| Document | Purpose | Audience |
|---|---|---|
| [API Reference](api.md) | Endpoint schemas, methods, responses | Backend developers, API integrators |
| [ML Documentation](ml.md) | Model training, evaluation, calibration | ML engineers, data scientists |
| [Architecture](architecture.md) | System design and data flow | Engineers, architects |
| [Agent Core Reuse](cross-product-reuse.md) | Extensibility and agent design | Agent developers |

---

## Key Concepts

### Calibrated Confidence Scores

Search results include a **calibrated confidence score** (0–1 range), not raw cosine similarity. A score of 0.85 means the model is ~85% confident the match is correct, based on a calibrator trained on validation data. This makes scores interpretable and trustworthy.

**Confidence bands** map calibrated scores to user-facing labels:
- **High:** ≥ 0.70 (very confident)
- **Medium:** 0.40–0.69 (somewhat confident)
- **Low:** < 0.40 (low confidence)

### Fine-Tuning Protocol

The embedding model is fine-tuned using **ArcFace loss**, a metric-learning approach that:
1. Freezes the DINOv2 backbone
2. Trains a small projection head (MLP: 1024→256)
3. Uses additive angular margin to sharpen species separation on the unit hypersphere
4. Results in **97.62% Recall@1** on held-out test set

See [ml.md](ml.md) for full training details.

### Unified Agent Core

FloraLens's naturalist assistant is built on an unmodified **Unified Agent Core** (shared with AgentForge). This proves the core is reusable across domains:

- **Agent manifests** (YAML) declare a team: supervisor → identifier, researcher, care-advisor
- **Tools** (embedding search, web search) are pluggable, domain-specific implementations
- **Memory** (mem0 + LangGraph checkpointer) is scoped per user
- **Extensibility:** adding a new agent (e.g., disease diagnosis) requires no core changes

See [cross-product-reuse.md](cross-product-reuse.md) for full details.

---

## Useful Commands

### Run the API Server

```bash
cd apps/api
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m uvicorn apps.api.app.main:app --port 8100
```

### Start the Web Frontend

```bash
cd apps/web
npm install
npm run dev
```

### Run ML Pipeline (rebuild embeddings index)

```bash
python -m ml.scripts.build_splits              # Ingest data, build splits
python -m ml.scripts.build_embeddings_index    # Embed gallery, index
python -m ml.scripts.run_baseline_eval         # Evaluate baseline
```

### Run Tests

```bash
# Python (API + ML):
python -m pytest -v

# JavaScript (frontend):
cd apps/web && npm test
```

---

## Environment Setup

Create a `.env` file in the project root with:

```env
# API Keys (required for assistant)
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

# Optional: hardening (default: disabled)
FLORALENS_API_KEY=your-secret-key

# Optional: device selection (default: auto)
FLORALENS_DEVICE=auto

# Optional: model version (default: finetuned_arcface_dinov2_v2)
MODEL_VERSION=finetuned_arcface_dinov2_v2
```

---

## Architecture Overview

```
┌─────────────────────────── Next.js Frontend (port 3100) ──────────────────┐
│  Search UI  │  3D Galaxy  │  Species Categories  │  Assistant Chat  │  Garden   │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ REST + SSE
┌────────────────────────────────▼──────────────────────────────────────────┐
│                    FastAPI Backend (port 8100)                             │
│                                                                              │
│  ┌──────────────────────────┬──────────────────────────┐                   │
│  │ ML Service               │ Unified Agent Core       │                   │
│  ├──────────────────────────┼──────────────────────────┤                   │
│  │ • DINOv2 backbone        │ • Manifests (YAML)       │                   │
│  │ • ArcFace head           │ • LangGraph runtime      │                   │
│  │ • Calibrator             │ • Tools (embedding,web)  │                   │
│  │ • Vector store (cosine)  │ • Memory (mem0)          │                   │
│  └──────────────────────────┴──────────────────────────┘                   │
│                                                                              │
│  Routes: /api/search, /api/assistant, /api/garden, /api/memory, ...        │
└────────────────────────────────┬──────────────────────────────────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
      ┌────▼─────┐      ┌────────▼──────┐      ┌──────▼──────┐
      │ Embeddings│      │ SQLite Store  │      │ Model Files │
      │ Cache     │      │ (garden/mem)  │      │ (weights,   │
      │ (NumPy)   │      │               │      │  calibrator)│
      └───────────┘      └───────────────┘      └─────────────┘
```

See [architecture.md](architecture.md) for the full diagram with data flows.

---

## Support & Resources

- **[PRD.md](../PRD.md)** — Product requirements, success metrics, non-goals, risk mitigations.
- **[IMPLEMENTATION-PLAN.md](../IMPLEMENTATION-PLAN.md)** — Phased roadmap, exit criteria, build order.
- **[Main README](../README.md)** — Quick start, feature overview, tech stack.

---

## Last Updated

July 10, 2026 — Phase 3b complete (fine-tuning, calibration, promotion gate). All search results now show calibrated confidence scores.
