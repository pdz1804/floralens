# FloraLens — ML Backend (Phase 0-1)

Flower visual-similarity search backend. See `PRD.md` and `IMPLEMENTATION-PLAN.md`
for the full product spec; this README covers what's built so far: **Phase 0
(scaffold)** and **Phase 1 (dataset, splits, embeddings, retrieval eval,
vector index, search API)**.

## Layout

```
apps/api/app/        FastAPI app (health, search)
  main.py            routes
  search_service.py  gallery vector store + query pipeline
  config.py          env-driven settings
apps/api/tests/       API tests (TestClient)
ml/data/              dataset ingestion, split builder, leakage guard
ml/embeddings/         frozen backbone (OpenCLIP ViT-B-32, resnet50 fallback) + cache
ml/index/              in-memory cosine vector store
ml/eval/                retrieval metrics + eval harness + reports/
ml/scripts/             CLI entry points that run the pipeline end-to-end
tests/                 ML unit/integration tests (metrics, leakage, embeddings, smoke)
```

## Setup

```bash
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt
venv/Scripts/python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
venv/Scripts/python.exe -m pip install open-clip-torch==2.31.0
```

Torch is CPU-only by design: the embedding backbone is frozen (no training
in Phase 0-1), so CPU inference is sufficient and keeps the setup simple.

## Data pipeline (run once, in order)

```bash
# 1. Ingest Oxford-102, build the stratified train/val/test/gallery split
#    manifest + leakage report (downloads ~8189 images on first run).
venv/Scripts/python.exe -m ml.scripts.build_splits

# 2. Embed a stratified subset with the frozen backbone and cache vectors.
venv/Scripts/python.exe -m ml.scripts.build_embeddings_index

# 3. Run the retrieval eval protocol (val + test) and save the baseline
#    EvalReport.
venv/Scripts/python.exe -m ml.scripts.run_baseline_eval
```

Outputs:
- `ml/data/manifests/split_manifest.json`, `ml/data/manifests/leakage_report.json`
- `ml/data/embeddings_cache/embeddings.npz`, `ml/data/embeddings_cache/metadata.json`
- `ml/eval/reports/baseline_eval_report.json`

### Dataset source note

The pipeline pulls Oxford-102 Flowers images from the
`dpdl-benchmark/oxford_flowers102` mirror on the Hugging Face Hub rather
than the original `robots.ox.ac.uk` server. The original server download
stalled repeatedly in this environment (~90% through a 345MB transfer,
multiple attempts); the HF mirror has an **identical 102-class label
schema and ordering** (verified against `torchvision.datasets.Flowers102.classes`)
and downloads reliably. This is a transport-only substitution — same
dataset, same content, same license.

### Data scale

- Full pool: 8185 images (of 8189 nominal — 4 exact cross-partition
  duplicates in the source were deduplicated by content hash), 102 classes.
- Split manifest (train/val/test/gallery) is built over the **full 8185**
  and is leakage-free by construction (enforced by `tests/test_leakage.py`).
- The **embedding step embeds a documented stratified subset** (CPU time
  budget): up to 10 gallery / 4 val / 4 test images per class = 1801 images
  total. `train` is not embedded in Phase 1 (backbone is frozen; train
  images are unused until Phase 3 fine-tuning). Caps are set in
  `ml/data/subset.py::DEFAULT_PER_CLASS_CAP` and are trivially raised for a
  full-scale run — the pipeline code itself is correct at full 8185 scale.

## Run the API

```bash
venv/Scripts/python.exe -m uvicorn apps.api.app.main:app --reload --port 8000
```

- `GET /health`, `GET /api/health` — liveness + active `model_version`.
- `POST /api/search` — multipart `file` upload OR JSON `{"image_base64": "..."}`;
  returns top-12 gallery matches (`specimen_id`, `label_name`, cosine `score`).
  EXIF is stripped before embedding.

## Tests

```bash
venv/Scripts/python.exe -m pytest -v
```

30 tests: retrieval metric fixtures, vector store ranking, embedding
determinism/shape, leakage (synthetic + real manifest hard-assert), API
contract/validation/search, and an end-to-end embed→index→query smoke test.

## Baseline results (OpenCLIP ViT-B-32, laion2b_s34b_b79k, zero-shot)

From `ml/eval/reports/baseline_eval_report.json` (gallery = 985 specimens,
408 val queries, 408 test queries):

| Split | Recall@1 | Recall@5 | Recall@10 | mAP@10 | MRR |
|---|---|---|---|---|---|
| val | 0.922 | 0.978 | 0.988 | 0.873 | 0.946 |
| test | 0.914 | 0.978 | 0.990 | 0.869 | 0.943 |

val↔test Recall@5 gap: 0.000 (well under the 5-point overfitting guard in
PRD §5/§14.8). These are zero-shot numbers with no fine-tuning — Phase 3
will compare a fine-tuned model against this baseline using the identical
protocol.
