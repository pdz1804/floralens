# PRD — FloraLens: AI Visual Naturalist & Plant Discovery Platform

| Field | Value |
|---|---|
| Product | FloraLens |
| Version | 1.1 (Draft — adds rigorous train/val/test + testing strategy) |
| Author | ClaudeKit Engineer |
| Date | 2026-07-06 |
| Status | Planning / For review |
| Type | Learning-oriented full-stack AI product |
| Related | `../agentforge/PRD.md` (shares the Unified Agent Core) |

---

## 1. Executive Summary

FloraLens is a web platform where a user pastes or uploads a photo of a flower/plant and instantly gets **visually similar species ranked by a calibrated similarity score**, an **interactive 3D specimen view**, and an **AI naturalist assistant** (a multi-agent team) that identifies, researches, and advises care — while remembering the user's personal plant collection over time.

The product is deliberately built to **learn and revise a full stack of AI engineering skills**: training/fine-tuning an image embedding model **with correct train/validation/test discipline**, vector similarity search with a **rigorous retrieval-evaluation protocol**, 3D rendering (Three.js), multi-agent orchestration (LangGraph), long-term agent memory (mem0), tool/MCP integration, and web search. It consumes the same **Unified Agent Core** specified for AgentForge, proving the core is reusable across very different domains.

The single most important quality bar for v1: **every model claim (e.g. "80% Top-1") is produced by a documented, leakage-free evaluation protocol, reproducible from a versioned dataset + seed + config.**

## 2. Problem Statement & Learning Goals

**User-facing problem.** Casual gardeners, students, and plant enthusiasts struggle to identify plants from a photo, find visually similar species, and get trustworthy, personalized care guidance in one place. Existing apps identify but rarely explain, rarely visualize in 3D, and never remember *your* garden.

**Engineering learning goals (first-class product driver).** Each headline feature maps to a skill we want to build/revise:

| Feature | Skill exercised |
|---|---|
| Paste-a-flower → ranked similar results with calibrated scores | Image embedding fine-tuning + vector retrieval + **eval protocol** |
| Correct train / validation / test pipeline | **ML methodology: split hygiene, model selection, unbiased test** |
| 3D specimen viewer + 3D embedding "galaxy" | Three.js / WebGL, dimensionality reduction (UMAP/PCA) |
| Naturalist multi-agent assistant | LangGraph multi-agent orchestration, prompts |
| "My Garden" that remembers plants & preferences | mem0 long-term memory, short-term thread state |
| Research answers with citations | Web search tool, tool interface |
| Extensible skills (add "disease diagnosis" later) | Unified Agent Core extension model |

## 3. Target Users & Personas

- **Hobbyist Gardener (primary).** Wants quick ID, visual matches, and care tips; low technical skill; values a friendly assistant that remembers their plants.
- **Botany Student (secondary).** Wants to explore species relationships visually (embedding space), compare specimens in 3D, and get sourced explanations.
- **Builder / Us (internal).** The engineer learning the stack; wants a clean ML evaluation harness and clean agent extension points.

## 4. Goals & Non-Goals

**Goals**
- Sub-second top-K visual similarity search over ≥ 10k flower images with a **calibrated** similarity score.
- A fine-tuned embedding model **measurably and reproducibly** better than the zero-shot baseline on a **held-out test set that is touched exactly once per model version**.
- An interactive 3D specimen viewer and a navigable 3D embedding map.
- A multi-agent naturalist assistant with persistent per-user memory.
- Demonstrate the Unified Agent Core is reusable (same core as AgentForge).

**Non-Goals (v1)**
- Mobile native apps (responsive web only).
- Guaranteed botanical/medical accuracy (educational tool, with disclaimers).
- Real-time collaboration / social feed.
- Monetization, payments, multi-tenant billing.
- Training our own LLM (assistant uses hosted models).

## 5. Success Metrics

All model metrics below are defined by the **evaluation protocol in §14** and computed on the **test split only**, once per model version.

| Metric | Target (v1) | Measured on |
|---|---|---|
| Retrieval **Recall@1** (fine-tuned) | ≥ 80% and > baseline | Test query set vs test gallery (§14.4) |
| Retrieval **Recall@5** | ≥ 95% | Test query set vs test gallery |
| **mAP@10** | ≥ 0.75 | Test query set vs test gallery |
| Score **calibration** (ECE) | ≤ 0.05 | Test set, calibrator fit on val (§14.6) |
| Similarity query latency (p95) | ≤ 800 ms | Production-shaped load test |
| 3D viewer first interactive frame | ≤ 2.5 s | Mid laptop |
| Assistant answer with ≥ 1 valid citation | ≥ 90% | Research-query eval set |
| Add a new agent skill without touching core | < 1 day | "disease diagnosis" extension check |

**Anti-metric (guardrail against cheating):** validation-set and test-set numbers must be reported *side by side*; a val↔test gap > 5 absolute points on Recall@5 is flagged as probable overfitting and blocks promotion.

## 6. Functional Requirements

### Epic A — Visual Similarity Search (the flower example)
- **A1.** User uploads/pastes/drag-drops an image (or an image URL). Validate type/size; strip EXIF.
- **A2.** System computes an embedding via the **active** model version and runs approximate nearest-neighbor search in the vector DB (which holds only the **gallery/index** partition — never raw test-query leakage; see §14.7).
- **A3.** Return top-K (default 12) results with: thumbnail, species label (if known), **calibrated similarity score (0–1) rendered as a bar + %**, and confidence banding (high/medium/low) derived from the calibrator, not raw cosine.
- **A4.** User can filter results by color, family, or dataset; re-rank on filter change.
- **A5.** "Explain this match" opens the assistant with the query + matched image in context.
- **A6.** Admin/internal: activate a model version → triggers versioned re-index (gallery re-embedded under that version).

### Epic B — Embedding Model Training / Validation / Testing
- **B1.** Ingestion + **dataset versioning** (content hash, split manifest) for a labeled flower dataset (Oxford 102 Flowers baseline; optional iNaturalist subset).
- **B2.** **Split builder** producing disjoint train/val/test partitions, stratified by class, with **near-duplicate leakage detection** across splits (§14.2).
- **B3.** Baseline: zero-shot CLIP/DINOv2 embeddings, evaluated with the exact same protocol as fine-tuned models.
- **B4.** Fine-tune (metric learning: triplet / contrastive / ArcFace head); **augmentation applied to train only**; **early stopping and hyperparameter selection on val only**.
- **B5.** **Evaluation harness** computing Recall@K, Precision@K, mAP@K, MRR + embedding-quality (silhouette, centroid separation), separately on val and test.
- **B6.** **Score calibrator** (isotonic/Platt) fit on val, evaluated on test (ECE, reliability curve).
- **B7.** **Model registry + experiment tracking**: each version records base model, method, hyperparams, dataset hash, seed, val metrics, test metrics, calibration; **promotion gate** enforces §5 anti-metric.
- **B8.** Batch-embed the **gallery partition** → upsert to vector DB tagged by version; precompute 3D projection.

### Epic C — 3D Rendering (Three.js)
- **C1.** 3D specimen viewer: GLTF plant model (or pseudo-3D card fallback), orbit/zoom/pan, lighting, reduced-motion fallback.
- **C2.** 3D "Embedding Galaxy": project embeddings to 3D (UMAP/PCA precomputed) → interactive instanced point cloud; click point → open specimen; color by family; hover tooltip. Projection is **fit on train+gallery, applied to all** (documented, so the map is reproducible).
- **C3.** Performance: instanced points for ≥ 10k nodes toward 60 fps; LOD/culling; graceful WebGL-unavailable message.

### Epic D — Naturalist Multi-Agent Assistant (uses Unified Agent Core)
- **D1.** Chat interface bound to a user session and their "Garden".
- **D2.** Agent team (LangGraph): **Supervisor/Router → Identifier** (calls embedding-search tool) **→ Researcher** (web search + citations) **→ Care-Advisor** (care plan).
- **D3.** Streaming responses (token stream + tool-call trace surfaced to UI).
- **D4.** Guardrails: educational disclaimers; refuse toxicity/medical dosage claims; cite sources.
- **D5.** Extensibility: adding "Disease Diagnosis" agent = new manifest + tool registration, no core edits.

### Epic E — "My Garden" Memory (mem0)
- **E1.** Save an identified plant (species, photo, notes, location, added date).
- **E2.** Long-term semantic memory (mem0): remembers preferences (indoor/outdoor, climate zone, likes/dislikes) + past plants; retrieval augments assistant answers.
- **E3.** Short-term memory: conversation thread state via LangGraph checkpointer.
- **E4.** Memory inspection UI: view/edit/delete (user control + privacy).

### Epic F — Accounts & Platform
- **F1.** Auth (email/password + OAuth), per-user isolation.
- **F2.** Upload storage (object store), image processing pipeline.
- **F3.** Rate limiting on search and assistant endpoints.
- **F4.** Observability: request logs, **model-version tag on each search**, tool-call tracing.

## 7. Representative User Stories & Acceptance Criteria

- **US-1 (A):** *Paste a rose photo → 12 similar flowers with calibrated % scores.*
  - AC: results < 1 s p95; score bar + %; invalid input handled; scores come from calibrator not raw cosine.
- **US-2 (B):** *Fine-tune the model and see retrieval improve — provably.*
  - AC: one eval report compares baseline vs fine-tuned on the **same test protocol**; fine-tuned Recall@5 ≥ baseline; val↔test gap ≤ 5 pts; dataset hash + seed + config logged; result reproducible on re-run.
- **US-3 (C):** *Fly through the embedding galaxy and inspect clusters.*
  - AC: ≥ 10k points interactive; click opens specimen; reduced-motion + no-WebGL fallbacks.
- **US-4 (D):** *Ask "how do I care for this?" → sourced, personalized plan.*
  - AC: answer streams; ≥ 1 citation; uses Garden/preferences from memory; disclaimer shown.
- **US-5 (E):** *See and delete what the assistant remembers.*
  - AC: memory list; delete removes it from future retrieval within the session.
- **US-6 (B, correctness):** *No data leakage between splits.*
  - AC: automated test asserts train ∩ val ∩ test = ∅ by image id AND by near-duplicate hash; fails the build if violated.

## 8. System Architecture

```
┌──────────────────────────── Frontend (Next.js + React) ────────────────────────────┐
│  Search UI  │  3D Viewer / Embedding Galaxy (Three.js)  │  Assistant Chat  │ Garden │
└───────────────┬───────────────────────────┬───────────────────────┬────────────────┘
                │ REST/JSON + SSE            │ static GLTF / points   │
┌───────────────▼───────────────────────────▼───────────────────────▼────────────────┐
│                              Backend API (FastAPI, Python)                            │
│  /search  /embed  /models  /specimens  /assistant(SSE)  /garden  /memory  /auth      │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  ML / Embedding Service       │   Unified Agent Core (shared w/ AgentForge)          │
│  - dataset+split versioning   │   - Agent Manifests (YAML)                            │
│  - train / val / test harness │   - Registries: Tools/Prompts/MCP/Memory/Model        │
│  - CLIP/DINOv2 + finetune head│   - LangGraph runtime (StateGraph + checkpointer)     │
│  - eval + score calibrator    │   - Tools: EmbeddingSearchTool, WebSearchTool, ...    │
│  - model registry / tracking  │   - Memory: mem0 provider (+ thread checkpointer)     │
└───────────┬───────────────────┴──────────────────────┬───────────────────────────────┘
            │                                            │
   ┌────────▼─────────┐ ┌──────────────┐ ┌────────────┐ ┌──────▼───────┐ ┌──────────────┐
   │ Vector DB        │ │ Object store │ │ Experiment │ │ mem0 store   │ │ Postgres     │
   │ (Qdrant/pgvector)│ │ (images/GLTF)│ │ tracking   │ │ (long-term)  │ │ (app+ML meta)│
   │  = GALLERY only  │ │              │ │(MLflow/W&B)│ │              │ │              │
   └──────────────────┘ └──────────────┘ └────────────┘ └──────────────┘ └──────────────┘
```

**Key flows**
- *Search:* image → EmbeddingService (active version) → vector → ANN over gallery index → hydrate metadata → calibrate score → JSON.
- *Assistant:* user msg → Agent Core Supervisor (LangGraph) → Identifier (EmbeddingSearchTool) / Researcher (WebSearchTool) / Care-Advisor → memory read/write (mem0) → streamed answer + tool trace.
- *Training:* dataset version → split builder → train (train split) → select/tune (val split) → **final eval (test split, once)** → calibrate (val→test) → registry promotion gate → gallery re-embed + index.

## 9. Unified Agent Core (consumed dependency — full spec in AgentForge PRD §8)

FloraLens declares agents as manifests against the shared core. Minimal manifest:

```yaml
# agents/naturalist_supervisor.yaml
id: naturalist_supervisor
model: { provider: anthropic, name: claude-sonnet-5, temperature: 0.2 }
prompt_ref: prompts/naturalist_supervisor.md
memory: { provider: mem0, scope: user, namespace: floralens }
tools: [embedding_search, web_search]
sub_agents: [identifier, researcher, care_advisor]
guardrails: [educational_disclaimer, no_medical_dosage]
io_schema: { input: NaturalistQuery, output: NaturalistAnswer }
```

**Extension guarantee:** adding "disease_diagnosis" = new `BaseTool` + register + manifest + list under supervisor `sub_agents`. No core code changes. (Interfaces/registries/runtime in AgentForge PRD §8.)

## 10. Data Model (core entities)

- **User**(id, email, auth, climate_zone, prefs_json)
- **Specimen**(id, species, family, color_tags[], image_url, gltf_url?, source_dataset, **split** ∈ {train,val,test,gallery-hold})
- **DatasetVersion**(id, name, content_hash, split_manifest_ref, created_at)
- **Embedding**(id, specimen_id, model_version, vector, projection_3d) — one row per (specimen, model_version)
- **ModelVersion**(id, base_model, method, hyperparams_json, dataset_version_id, seed, val_metrics_json, test_metrics_json, calibration_json, is_active, created_at)
- **EvalReport**(id, model_version_id, split, metrics_json, artifact_ref)
- **GardenPlant**(id, user_id, specimen_id, nickname, notes, location, added_at)
- **Conversation**(id, user_id, thread_state_ref)
- **SearchLog**(id, user_id, model_version, latency_ms, top_k_json)

> The **split** field on Specimen is authoritative: it defines which partition an image belongs to and is the anchor for leakage tests.

## 11. API Surface (representative)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/search` | image (multipart/URL) → ranked matches + calibrated scores |
| POST | `/api/embed` | internal: embed image(s), return vectors |
| GET | `/api/models` / POST `/api/models/activate` | list/activate model version (guarded by promotion gate) |
| GET | `/api/models/{id}/eval` | fetch val+test EvalReport for a version |
| GET | `/api/specimens/{id}` | specimen detail + gltf/projection |
| GET | `/api/galaxy` | 3D projected point cloud payload |
| POST | `/api/assistant` (SSE) | chat with naturalist agent team |
| GET/POST/DELETE | `/api/garden` | manage saved plants |
| GET/DELETE | `/api/memory` | inspect/delete user memories |

## 12. Non-Functional Requirements

- **Performance:** search p95 ≤ 800 ms; galaxy interactive ≤ 2.5 s.
- **Scalability:** vector index ≥ 100k specimens; embeddings versioned so re-index is non-destructive.
- **Security/Privacy:** per-user isolation; EXIF stripped; user can delete memories & uploads; secrets in env, never committed.
- **Reliability:** model version pinned per search; graceful degradation if an agent tool fails (partial + note).
- **Reproducibility (ML):** every model version reproducible from (dataset_version, seed, config); experiment tracking mandatory.
- **Accessibility:** keyboard-navigable, reduced-motion, WebGL fallback, alt text.
- **Observability:** structured logs, tool-call traces, model-version tag on every search.

## 13. Tech Stack

- **Frontend:** Next.js (App Router), React, TypeScript, Three.js (react-three-fiber), Tailwind.
- **Backend:** Python, FastAPI, Pydantic; SSE streaming.
- **ML:** PyTorch, OpenCLIP / DINOv2, metric-learning head (triplet/ArcFace), scikit-learn (calibration, metrics), UMAP; experiment tracking via MLflow (default) or Weights & Biases.
- **Agents:** LangGraph (+ LangChain where useful), Unified Agent Core.
- **Memory:** mem0 (long-term) + LangGraph checkpointer (short-term).
- **Data:** Postgres (+ pgvector) or Qdrant for vectors; object store (S3-compatible/MinIO) for images/GLTF.
- **Testing/ML-ops:** pytest, Great Expectations (or lightweight schema checks) for data validation, DVC (optional) for dataset versioning.
- **Infra:** Docker Compose local; env-based config.

## 14. Machine Learning — Data, Training, Validation & Testing (Authoritative Protocol)

> This section is the contract for **how models are trained and how every reported number is produced.** All Epic B work must conform to it. The guiding principle: **the test set is a vault — opened once per model version, never used for any decision.**

### 14.1 Dataset & versioning
- Primary dataset: **Oxford 102 Flowers** (102 classes). Optional expansion: curated **iNaturalist** flower subset.
- Each ingest produces a **DatasetVersion** with a content hash and an immutable **split manifest** (list of image id → split). Reproducibility depends on this hash.
- Provenance and license recorded per source in `Specimen.source_dataset`.

### 14.2 Split policy (train / validation / test) + leakage prevention
- **Three disjoint partitions**, stratified by class so every class appears in each split:
  - **Train** — used only to fit model weights / the projection head.
  - **Validation (val)** — used only for model selection: hyperparameters, loss/margin choice, embedding dim, early stopping, and **fitting the score calibrator**. Never reported as headline accuracy.
  - **Test** — used only for the **final, one-shot** unbiased evaluation of a finished model version.
- **Leakage prevention (hard rules, enforced by tests — see §17):**
  1. An image id appears in exactly one split.
  2. **Near-duplicate detection** (perceptual hash / embedding cosine > threshold) across splits; duplicates and same-plant/same-shoot photos are quarantined to a single split.
  3. Data **augmentation is applied to train only** (never val/test).
  4. Any normalization/whitening statistics are computed on **train only** and reused for val/test.
  5. The **score calibrator is fit on val, evaluated on test** — never fit on test.
- Optional for small classes: **stratified k-fold cross-validation** on train+val for hyperparameter search; test remains untouched.

### 14.3 Baseline (zero-shot)
- Encode all splits with a frozen backbone (OpenCLIP or DINOv2), no training.
- Run the **identical evaluation protocol** (§14.4) so baseline vs fine-tuned is apples-to-apples.

### 14.4 Retrieval evaluation protocol (how "accuracy" is actually computed)
This is a **retrieval** task, not plain classification, so evaluation uses a query/gallery construction:
- **Gallery (index):** the searchable database of embeddings. For evaluation, the gallery is the held-out **gallery-hold** portion of the corresponding split; in production it is the labeled corpus.
- **Query set:** each evaluation image is a query; its own embedding is **excluded** from its own results.
- **Relevance:** a retrieved item is *relevant* iff it shares the query's **species/class label**.
- **Metrics (computed per query, then averaged):**
  - **Recall@K** — fraction of queries with ≥ 1 relevant item in top-K (K ∈ {1,5,10}).
  - **Precision@K** — fraction of top-K that are relevant.
  - **mAP@K** — mean average precision, rewarding correct ranking order.
  - **MRR** — mean reciprocal rank of the first relevant hit.
- **Embedding-quality diagnostics (unsupervised sanity):** silhouette score, intra-class vs inter-class centroid distance ratio.
- Val and test are each evaluated with this exact protocol; **only test numbers are headline.**

### 14.5 Fine-tuning procedure
1. Attach a projection head to the frozen (or partially unfrozen) backbone.
2. Train with a metric-learning loss — start **triplet/contrastive**, compare against **ArcFace** — using **hard-negative mining** from within-batch classes.
3. Apply train-only augmentation (flip, crop, color jitter, mild rotation).
4. **Early stop** on val Recall@5; **select** the loss/margin/lr/dim by val metrics.
5. Log everything (config, seed, dataset hash, curves) to the experiment tracker.

### 14.6 Score calibration (so the % shown to users means something)
- Raw cosine similarity is **not** a probability. Fit a calibrator (isotonic regression or Platt scaling) mapping cosine → P(same-species) using **val** pairs.
- Evaluate calibration on **test**: **Expected Calibration Error (ECE)** + reliability diagram.
- The UI confidence banding (high/medium/low) is derived from calibrated probability thresholds, not raw cosine.

### 14.7 Indexing & serving (no test leakage into production metrics)
- After a version passes the promotion gate, **re-embed the gallery corpus** under that version and upsert to the vector DB tagged by `model_version`.
- Production search always specifies the **active** `model_version`; switching versions is an explicit, audited admin action.
- The vector index used to *report test metrics* excludes the test queries themselves (§14.4). Production index composition is documented per version.

### 14.8 Model registry & promotion gate
A new version becomes `is_active` **only if all hold:**
1. Test **Recall@5 ≥ current active** (no regression) and ideally beats baseline by the target margin.
2. **val↔test gap ≤ 5 pts** on Recall@5 (overfitting guard, §5 anti-metric).
3. Calibration **ECE ≤ 0.05** on test.
4. EvalReport artifacts (val + test) stored and linked.
Otherwise the previous active version stays; the candidate is archived with its report.

### 14.9 Reproducibility & experiment tracking
- Fixed random seeds (data split, sampler, init) recorded per run.
- Every run logs: dataset_version hash, git commit, full config, hardware, metrics, artifacts.
- Re-running a version's config on the same dataset hash reproduces metrics within a documented tolerance.

### 14.10 Continuous / regression evaluation
- Any retrain triggers a **full re-eval**; the eval report is diffed against the active version.
- A CI "ML regression" job runs a **small fixed eval subset** on every model-code change to catch silent metric drops before full training.

## 15. Testing & Quality Strategy (software + ML)

> ML evaluation (§14) proves the *model* is good. This section proves the *system* is correct. Both are required.

### 15.1 Test pyramid (application)
- **Unit:** embedding function shape/dtype, cosine + calibration math, ranking/scoring, API validators, EXIF stripping.
- **Integration:** `/api/search` against a seeded vector DB; agent tool (`EmbeddingSearchTool`) end-to-end; mem0 read/write.
- **Contract:** API request/response schemas (Pydantic) versioned and asserted.
- **E2E:** upload/paste → ranked results with scores (Playwright); assistant chat streams with citation; save-to-Garden.
- **3D smoke:** galaxy/viewer mount, WebGL fallback path, reduced-motion path.
- **Load/perf:** search p95 ≤ 800 ms under production-shaped concurrency; galaxy interactive budget.

### 15.2 ML-specific tests (data & model correctness)
- **Data-validation tests:** schema, label domain (102 classes), image decodability, class balance report.
- **Leakage assertion test (build-blocking):** `train ∩ val ∩ test = ∅` by id **and** by near-duplicate hash (§14.2). Fails CI if violated. (Satisfies US-6.)
- **Metric-regression test:** fixed eval subset; alert if Recall@5 drops beyond tolerance vs recorded baseline.
- **Calibration test:** ECE ≤ target on a fixed val/test slice.
- **Determinism test:** same seed + dataset hash → identical splits and metrics within tolerance.
- **Model-contract test:** active version embeds a canary image to the expected dimension and near-expected vector (guards accidental model swap).

### 15.3 Agent/LLM evaluation
- Small **research-query eval set** with rubric: answer must contain ≥ 1 resolvable citation and a disclaimer; scored programmatically + occasional human spot-check.
- Guardrail tests: medical-dosage / toxicity prompts are refused.

### 15.4 CI gates
`lint → unit → data-validation → leakage assertion → integration → ML metric-regression (subset) → e2e (smoke)`; full training/eval runs on demand or nightly, not per-commit.

## 16. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Data leakage inflates metrics | Hard split rules + build-blocking leakage test (§14.2, §15.2) |
| Overfitting to val during selection | Test vault opened once; val↔test gap gate (§14.8) |
| Uncalibrated scores mislead users | Calibrator fit on val, ECE-gated on test (§14.6) |
| Fine-tune underperforms baseline | Baseline stays active until promotion gate passes (§14.8) |
| 3D galaxy slow at scale | Instanced rendering, LOD, capped visible points, precomputed projection |
| Agent hallucinates care advice | Web-search grounding + citations + disclaimers + guardrail agent |
| Non-reproducible experiments | Dataset hashing, seeds, experiment tracking mandatory (§14.9) |
| Dataset licensing | Permissive datasets; provenance recorded |

## 17. Skills-Coverage Matrix

| Skill you wanted | Where it lives in FloraLens |
|---|---|
| Train image embedding model + scored similarity | Epic A/B + **§14 (train/val/test protocol)** |
| Correct training / validating / testing | **§14 (splits, leakage, retrieval eval, calibration, promotion) + §15.2 tests** |
| Three.js 3D rendering | Epic C (specimen viewer + embedding galaxy) |
| Multi-agent + skills (extensible) | Epic D + Unified Agent Core (§9) |
| Prompts | Prompt registry, `prompt_ref` in manifests |
| Memory (mem0) | Epic E (long-term) + thread checkpointer (short-term) |
| Harness (unified tools/mcp/prompts/memory) | Unified Agent Core (§9; full spec in AgentForge §8) |
| Web search tool | Researcher agent tool (Epic D) |
| Sandbox for code | Deferred to AgentForge (Open Q#4) |
| MCP connections | Agent Core MCP registry (available; optional v1) |

## 18. Open Questions — RESOLVED (2026-07-06)

All v1 decisions locked; documented here as rationale. Reopen only with new evidence.

1. **Vector store → pgvector.** One DB (Postgres already present), simpler ops, adequate past 100k vectors. Switch to Qdrant only if ANN p95 breaks the 800 ms budget.
2. **Specimen viewer → pseudo-3D image cards (billboard/parallax) for v1.** Sourcing 102 quality GLTF flower models is infeasible; the embedding galaxy is the primary 3D learning surface. Hero GLTF models added later.
3. **Fine-tuning loss → ArcFace first**, then triplet/contrastive as a comparison experiment (both run the §14 protocol). Rationale: stable training + strong class separation on a closed 102-class set without hard-negative-mining overhead.
4. **Sandbox code execution → deferred to AgentForge** for v1 (confirmed). FloraLens stays focused on vision/embeddings.
5. **Dataset → Oxford-102 only for v1.** Establish the leakage-free protocol on one clean dataset; iNaturalist expansion is a post-v1 item.

Remaining true unknowns (resolve during build, non-blocking): exact backbone (OpenCLIP ViT-B/32 vs DINOv2 ViT-S) — decide by baseline eval in Phase 1; GPU availability for fine-tuning (local vs Colab/cloud).

> See `./IMPLEMENTATION-PLAN.md` for the phased build roadmap.
