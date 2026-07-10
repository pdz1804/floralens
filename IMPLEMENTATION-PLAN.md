# Implementation Plan — FloraLens

Phased roadmap for `PRD.md` (v1.1). Each phase lists scope, key files, and exit criteria.
Stack: FastAPI + PyTorch + LangGraph + mem0 (Python) / Next.js + Three.js (TS).
**The ML protocol in PRD §14 is binding on Phases 1, 3, and 3b.**

**Status legend:** ☐ pending · ◐ in-progress · ☑ done

## Phase Overview

| # | Phase | Depends on | Primary skill | Status |
|---|---|---|---|---|
| 0 | Foundation & scaffolding | — | Project setup | ☑ |
| 1 | Dataset, splits & baseline retrieval | 0 | **Data hygiene + eval protocol** | ☑ |
| 2 | Search UI + scored results | 1 | Frontend, UX | ☑ |
| 3 | Fine-tuning + validation (model selection) | 1 | **Training + val discipline** | ☑ |
| 3b | Test eval, calibration & promotion gate | 3 | **Unbiased testing + calibration** | ☑ |
| 4 | 3D specimen viewer + embedding galaxy | 1,2 | Three.js | ☑ |
| 5 | Unified Agent Core integration | 0 | Agent harness | ☑ |
| 6 | Naturalist multi-agent assistant | 5 | LangGraph, prompts, web search | ☑ |
| 7 | My Garden + mem0 memory | 5,6 | Memory | ☑ |
| 8 | Testing, CI gates & ML regression | 1,3b,6 | **Test strategy (PRD §15)** | ☑ |
| 9 | Auth, hardening, observability | all | Platform | ◐ (shared-key + per-user scaffold shipped; login UI/OAuth deferred) |

---

## Phase 0 — Foundation & Scaffolding
**Deliver:** runnable monorepo skeleton, local infra, lint.
- Layout: `apps/web`, `apps/api`, `packages/agent-core`, `ml/` (data/train/eval), `infra/` (docker-compose: Postgres+pgvector, MinIO, MLflow).
- Env config (`.env.example`), no secrets committed; health handshake.
**Exit:** `docker compose up` runs DB + object store + MLflow; web ↔ api health OK.

## Phase 1 — Dataset, Splits & Baseline Retrieval
**Deliver:** versioned data with leakage-free splits + a working zero-shot search evaluated by the real protocol.
- `ml/data/`: ingest Oxford 102; write **DatasetVersion** (content hash) + **split manifest** (train/val/test/gallery-hold), stratified by class (PRD §14.1–14.2).
- **Leakage guard:** near-duplicate detector (perceptual hash + embedding cosine) quarantines dupes to one split; emits a leakage report.
- `ml/embeddings/`: OpenCLIP/DINOv2 loader; `embed_image(img) -> vector`.
- `ml/eval/`: **retrieval eval harness** — query/gallery construction, Recall@K, Precision@K, mAP@K, MRR, silhouette (PRD §14.4).
- Baseline run: embed all splits; produce **baseline EvalReport (val + test)**.
- Index the **gallery partition** in pgvector tagged `model_version="baseline"`; `POST /api/search`.
- Tables: `Specimen` (+`split`), `DatasetVersion`, `Embedding`, `ModelVersion`, `EvalReport`.
**Exit:** leakage test passes (train∩val∩test=∅ by id + hash); baseline EvalReport stored; curl image → top-12 with cosine scores; p95 measured.

## Phase 2 — Search UI + Scored Results
**Deliver:** the paste-a-flower experience (scores still raw cosine until Phase 3b calibration).
- Upload/paste/drag-drop + URL; client validation; server EXIF strip.
- Result grid: thumbnail, label, **score bar + %**, confidence banding (placeholder → calibrated in 3b), filters (color/family).
- Loading/empty/error states; "Explain this match" hook (stub → wired Phase 6).
**Exit:** US-1 flow works end-to-end against baseline model.

## Phase 3 — Fine-Tuning + Validation (Model Selection)
**Deliver:** a fine-tuned model chosen **only** on validation.
- `ml/train/`: projection head + metric loss (**ArcFace first**, then triplet/contrastive as a comparison experiment); hard-negative mining for the triplet variant; **train-only augmentation** (PRD §14.5).
- Hyperparameter search + **early stopping on val Recall@5**; optional stratified k-fold on train+val (test untouched).
- Experiment tracking (MLflow): config, seed, dataset hash, curves per run.
- Produce candidate **ModelVersion** with **val metrics only** so far.
**Exit:** best candidate selected by val metrics; all runs logged/reproducible; **test set never touched in this phase.**

## Phase 3b — Test Evaluation, Calibration & Promotion Gate
**Deliver:** the one-shot unbiased test result + calibrated scores + gated activation.
- Run the eval harness **once** on the **test split** for the selected candidate and the baseline (same protocol) → EvalReports.
- **Score calibrator** (isotonic/Platt) fit on **val**, evaluated on **test** (ECE + reliability diagram) (PRD §14.6).
- **Promotion gate** (PRD §14.8): activate only if test Recall@5 ≥ active, val↔test gap ≤ 5 pts, ECE ≤ 0.05.
- On pass: re-embed gallery under new version, upsert to index; wire calibrated confidence banding into Search UI.
- Precompute UMAP 3D projection for the active version (feeds Phase 4).
**Exit:** US-2 + US-6 met; single baseline-vs-finetuned report with val+test side by side; activation switches search reproducibly; UI shows calibrated %.

## Phase 4 — 3D Specimen Viewer + Embedding Galaxy
**Deliver:** Three.js experiences.
- react-three-fiber specimen viewer (GLTF or pseudo-3D per Open Q#2); orbit/zoom; reduced-motion + no-WebGL fallback.
- Embedding galaxy: instanced point cloud from `/api/galaxy`; color by family; hover tooltip; click → specimen. Projection reproducible (fit documented).
- Perf: LOD, culling, visible-point cap.
**Exit:** US-3 met; ≥10k points interactive on mid laptop.

## Phase 5 — Unified Agent Core Integration
**Deliver:** shared harness wired in (core authored in AgentForge).
- Import `packages/agent-core`: registries + LangGraph runtime.
- Implement `EmbeddingSearchTool`, `WebSearchTool` against `BaseTool`.
- Manifest loader reads `agents/*.yaml`.
**Exit:** trivial single-agent manifest runs a tool call via `/api/assistant`.

## Phase 6 — Naturalist Multi-Agent Assistant
**Deliver:** the agent team + streaming chat.
- Manifests: `naturalist_supervisor`, `identifier`, `researcher`, `care_advisor`; prompts in registry.
- LangGraph StateGraph: Supervisor → sub-agents; tool traces surfaced; SSE streaming.
- Guardrails (disclaimer, no medical dosage) + citations; wire "Explain this match".
**Exit:** US-4 met; ≥90% research answers carry ≥1 citation; adding a stub 5th agent needs no core edit.

## Phase 7 — My Garden + mem0 Memory
**Deliver:** persistence of user context.
- `GardenPlant` CRUD; save-from-result flow.
- mem0 provider (long-term, user namespace) + LangGraph checkpointer (thread state).
- Memory augments assistant answers; inspection UI (view/edit/delete).
**Exit:** US-5 met; answers reflect saved prefs; delete removes from retrieval.

## Phase 8 — Testing, CI Gates & ML Regression
**Deliver:** the full test strategy (PRD §15) as automated gates.
- App tests: unit (embedding/calibration/scoring/validators), integration (search+DB, tools, mem0), contract (Pydantic), e2e (Playwright), 3D smoke, load/perf.
- **ML tests:** data-validation, **build-blocking leakage assertion**, metric-regression on fixed subset, calibration test, determinism test, model-contract canary.
- Agent eval: research-query rubric set; guardrail refusal tests.
- CI: `lint → unit → data-validation → leakage → integration → ML metric-regression (subset) → e2e smoke`; full train/eval nightly/on-demand.
**Exit:** all §15 tests green in CI; leakage/metric-regression gates actually block on violation.

## Phase 9 — Auth, Hardening, Observability
**Deliver:** demo readiness.
- Auth (email/password + OAuth); per-user isolation on all endpoints.
- Rate limiting; structured logging; tool-call tracing; model-version tag on searches.
- Accessibility pass; README + run docs.
**Exit:** all NFRs (§12) met; clean secret scan.
> **Shipped scope (◐):** opt-in **shared-key** auth (`FLORALENS_API_KEY` via `X-API-Key`/`Bearer`, `apps/api/app/auth.py`) on write/read-private endpoints; per-IP rate limiting (`rate_limit.py`) on `/api/search` + `/api/assistant`; secret redaction (`redaction.py`); a **per-user scaffold** (`resolve_user` JWT via `FLORALENS_JWT_SECRET`, `DEFAULT_USER` sentinel, garden + memory isolated per user, dev `/api/auth/token`). **Deferred:** real email/password + OAuth login and a web login surface — the token minter is a scaffold, not production login.

---

## Cross-Phase Guarantees (ML correctness)
- **Test vault:** test split is untouched until Phase 3b, once per model version (enforced by process + determinism test).
- **Leakage gate:** Phase 1 emits + Phase 8 enforces the build-blocking split-disjointness test.
- **Promotion gate:** no model reaches production without passing PRD §14.8.
- **Calibration:** user-facing % is always calibrated (Phase 3b), never raw cosine.

## Suggested Build Order for Learning
0 → 1 (data+eval discipline first — this is the core learning) → 2 → 3 → 3b → 5 → 6 → 4 → 7 → 8 → 9.
Rationale: nail the evaluation protocol before touching training, so every later model claim is trustworthy.

## Decisions (resolved — see PRD §18)
pgvector *(opt-in backend shipped; in-memory cosine remains the default — see Progress Log)* · pseudo-3D specimen cards · **ArcFace first** then triplet comparison · sandbox deferred to AgentForge · Oxford-102 only for v1.
Non-blocking build-time picks: backbone (OpenCLIP ViT-B/32 vs DINOv2) decided by Phase 1 baseline; GPU (local vs cloud).

## Progress Log

### 2026-07-11 — Status reconciliation against shipped code
Phase table was stale (Phases 2, 4–9 still ☐/◐); verified each against the repo and updated to ☑:
- **Phase 2** — Search UI + scored results: `apps/web` + `search_service.py`.
- **Phase 4** — 3D embedding galaxy + species catalogue: `galaxy_service.py`, `categories_service.py`, `/api/galaxy`, `/api/categories`.
- **Phase 5** — Unified Agent Core: `assistant_service.py` imports `agent_core` (editable, from sibling agentforge repo).
- **Phase 6** — Naturalist multi-agent team: `naturalist` supervisor + `identifier`/`researcher`/`care_advisor` manifests (`agents/*.yaml`), output guardrails, live SSE token streaming on `/api/assistant`.
- **Phase 7** — My Garden + memory: `garden_service.py` + `memory_service.py`; `in_memory` default, opt-in **mem0** (`FLORALENS_MEMORY_PROVIDER=mem0`), opt-in **SQLite checkpointer** (`FLORALENS_CHECKPOINT_DB`).
- **Phase 8** — CI + ML gates: `.github/workflows/ci.yml` runs the build-blocking leakage assertion, ML metric-regression gate (Recall@5 / mAP@10), and Playwright e2e (runtime model-dependent e2e is explicitly SKIPPED, not silently passed, on CI runners lacking artifacts/keys).
- **Phase 9** — shipped as ◐: shared-key auth + rate-limit + secret redaction (see Phase 9 note).

**Honest partials / deferred (not overclaimed):**
- **Auth:** shared-key gate plus a per-user scaffold (see 2026-07-11 continuation); email/password login UI and OAuth still deferred.
- **Vector store:** in-memory cosine (numpy) is the default; an opt-in pgvector backend now exists (see 2026-07-11 continuation).
- **Backbone:** OpenCLIP/DINOv2 active; DINOv3 gated behind an HF token.
- **Scale:** Oxford-102 only; iNaturalist-scale is post-v1.
- **Specimen viewer:** GLTF → pseudo-3D cards by decision (Open Q#2).

### 2026-07-11 — Continuation round (pgvector + per-user scaffold + §11 API)
- **PRD §11 API surface** — `GET /api/models`, `GET /api/specimens/{id}`, `POST /api/embed` added to `main.py` (`models_service.py`, `specimen_service.py`), with tests.
- **Vector store (opt-in pgvector)** — `ml/index/pgvector_store.py` (psycopg 3 + pgvector) behind a `VectorIndex` Protocol; `get_gallery_store()` env-selects on `FLORALENS_VECTOR_STORE=pgvector` + `DATABASE_URL`, falling back to in-memory. Validated live against pgvector 0.8.5 (including `/api/search` served end-to-end). In-memory stays the default.
- **Per-user auth scaffold** — `resolve_user` JWT dependency (`FLORALENS_JWT_SECRET`), `DEFAULT_USER="public"` sentinel so single-user behavior is unchanged when unset; garden isolated by composite `(user_id, specimen_id)` with an idempotent migration; per-user memory namespace; dev `POST /api/auth/token` + `GET /api/auth/me`. Still deferred: real login/OAuth and a web login surface — the token minter is an explicit scaffold, not production login.
