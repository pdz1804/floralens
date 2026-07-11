# FloraLens Roadmap

A horizon-based roadmap for FloraLens (botanical visual plant-ID) and the shared `agent_core` runtime it
consumes. Organized as **Now / Next / Later**. Shared-platform items also appear in the
[AgentForge roadmap](../../agentforge/docs/roadmap.md); full planning detail lives in
[`plans/260711-1707-superior-improvements-roadmap`](../../plans/260711-1707-superior-improvements-roadmap/plan.md).

**Effort key:** S = small · M = medium · L = large.

## Principles

- **Behavior-frozen:** existing `data-testid`s, the SSE assistant stream, the Three.js galaxy contract,
  and the theme toggle are preserved. New UI is additive.
- **Docs follow the served artifact, never the reverse.** Published ML numbers are sourced from the
  active model artifact; **served artifacts are never edited to fit docs.**
- **Shared core:** FloraLens runs `agent_core` unmodified; core changes ship as tagged `pdz-agent-core`
  releases.
- **CI stays green; secrets never committed.**

---

## Now — finish what the docs promise

**Feature wiring (real gaps, not deferred-by-decision):**

- **"Explain this match"** (M) — a Search result card "Explain" action opens the Assistant tab
  pre-seeded with the query + matched `specimen_id`/`label_name`, invoking the naturalist team.
  *Done when:* click Explain on a rose match → the assistant explains that match.
- **Image-aware Identifier agent** (M) — the identifier sub-agent calls text `gallery_facts`, not the
  image `embedding_search` the PRD advertises, and `/api/assistant` takes no image. Register
  `embedding_search` on `identifier.yaml` and let `/api/assistant` accept an optional image (reuse
  `_embed_query_image` + `search_image`).
  *Done when:* send an image to the assistant → the identifier uses embedding search to name it.
- **Result filters: color / family** (M) — only a confidence-band filter exists; add family/color facets
  (data present in `/api/categories`) + client re-rank.
  *Done when:* filter results to a family or color.
- **Model-admin endpoints** (M) — `GET /api/models` exists (`main.py:440`), but activate/eval-per-version
  do not. Add `POST /api/models/activate` (flip the active version behind the promotion gate + trigger
  re-index) and `GET /api/models/{id}/eval` (per-version report; reports already on disk).
  *Done when:* activate a version via API; fetch a version's eval report.
- **Garden fields** (S) — add `nickname/notes/location` to `garden_service` + form fields.
  *Done when:* save a specimen with a note + location and see them.
- **Memory per-item edit/delete** (S) — `DELETE /api/memory/{id}` + inline edit (mirrors AgentForge's
  memory panel).
  *Done when:* delete one memory item without clearing all.

**Doc-truth reconciliation (docs only — do NOT change served artifacts):**

- **Accuracy numbers** (S) — `docs/ml.md` prints a `test_metrics` block the served `metadata.json` does
  not contain (it has `val_metrics` only); the README headline recall figure must match the live
  `/api/pipeline` value. Source every published number from the served artifact or label it *projected*.
- **Winning hyperparameters** (S) — docs name `run03 (margin 0.5, hidden_dim 0)`; the active artifact is
  `run08 (margin 0.3, head_hidden_dim 256)`. Correct.
- **Dataset scale** (S) — docs say "8,185 images / 1,530 base"; the served dataset is **3,268 (+4,917
  augmented train)**. Correct.
- **API path** (S) — `docs/api.md` documents `GET /api/gallery`; the real route is `/api/galaxy`
  (`main.py:354`). Fix the path + schema block.
- **Named tests** (S) — `docs/ml.md` lists `test_determinism.py` / `test_model_contract.py` that don't
  exist — add them or remove the claim.

---

## Next — product flow, platform, richer UX

**Product:**

- **Image-upload identifier end-to-end** (M) — a first-class upload → identify → explain flow built on
  the Now image-aware assistant (not just an API param).

**Platform hardening (shared with AgentForge):**

- **Real auth** (L) — email/password + OAuth + a web login surface. The shipped JWT is dev-only scaffold
  (the token minter is **not** login). Per-user garden/memory isolation.
- **FloraLens consumes `pdz-agent-core` from PyPI** (S) — today FloraLens installs `agent_core` via an
  editable local path (`requirements.txt:59`). Once 0.1.3 is stable, pin `pdz-agent-core>=0.1.3` and drop
  the editable path (import stays `agent_core`). Depends on the package's undeclared runtime deps being
  folded into its own pyproject.
- **Persistence default-eligible** (L) — Postgres stores + migrations; **pgvector** a default-eligible
  vector backend replacing the in-memory cosine index; a **shared vector service** across both apps.
- **Observability** (M) — structured logging, metrics, error tracking; trace/cost persistence default.
- **CI/CD depth** (M) — provision e2e artifacts/keys in CI (FL currently self-skips when the embeddings
  cache is absent); mypy advisory→blocking; coverage thresholds; promote pip-audit/npm audit to blocking
  once clean; release automation; container images + a deploy target.

**Richer UX:**

- **Batch identify** (M) — identify multiple images in one flow.
- **Shareable results** (M) — a shareable link/card for a match.
- **Collections** (M) — group saved plants beyond the flat garden.
- **Richer species pages** (M) — deeper per-species detail.

---

## Later — scale, ML research

- **iNaturalist-scale index (>100k vectors)** (L) — the in-memory cosine index maxes ~100k
  (`ml.md:616`); requires the Next-phase pgvector/Qdrant shared vector service. Benchmark p95 at scale.
- **DINOv3 backbone** (L, contingent) — HF-token gated; ships as a new `ModelVersion` through the full
  promotion gate (Recall@5 + val/test gap + ECE). Publish only served numbers.
- **UMAP projection option** (M) — alternative to the current PCA galaxy projection; PCA stays default.
- **Multi-dataset support** (L) — generalize the split/leakage/index builders beyond Oxford-102.
- **Research bets** (L each, gate-first) — open-set / OOD recognition (detect flowers outside Oxford-102;
  gate: AUROC on a held-out OOD set); active learning; cross-dataset generalization.

---

## Sequencing

1. Now doc reconciliation first (cheap, no code risk) → feature wiring.
2. Explain-match + image-aware identifier are the highest user-visible value and independent of AgentForge.
3. Real auth + persistence precede per-user features; FL migrates to the PyPI package once 0.1.3 is stable.
4. Scale (>100k index) and ML research follow the shared vector service.

## Open questions

- Real-auth identity provider — shared decision with AgentForge.
- DINOv3 access is Meta-gated; the Later item is contingent on an HF token being granted.
