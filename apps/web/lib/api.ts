// Thin client for the FloraLens backend. Same-origin via Next rewrites.

export type SearchResult = {
  specimen_id: string;
  label: number;
  label_name: string;
  score: number; // cosine similarity
  confidence?: number; // calibrated (added in Phase 3b, optional)
  band?: "high" | "medium" | "low";
  description?: string | null; // curated botanical blurb (display-only, additive)
};

export type SearchResponse = { model_version: string; results: SearchResult[] };
export type Health = { status: string; model_version: string };

// ---- Pipeline transparency (GET /api/pipeline) --------------------------------
// Fields are treated defensively at render time — some keys vary or may be null.
export type PipelineMetrics = Record<string, number> | null;

export type ReliabilityBin = {
  bin_lower: number;
  bin_upper: number;
  mean_confidence: number;
  empirical_accuracy: number;
  count: number;
};

export type PipelineData = {
  dataset: { name: string; total: number; classes: number | null; splits: Record<string, number> };
  preprocessing: { name: string; description: string }[];
  backbone: { name: string; version: string; dim: number | null; frozen: boolean };
  eval: { val: PipelineMetrics; test: PipelineMetrics };
  calibration: { ece_before: number | null; ece_after: number | null; bands: ReliabilityBin[] | null };
  promotion: { decision: string | null; reason: string | null };
  model_version: string;
};

export type PreprocessStep = {
  name: string;
  description: string;
  // PNG (base64) of the image AFTER this step is applied. Optional: older
  // backends omit it, in which case the demo falls back to before/after only.
  image_png_b64?: string;
};
export type PreprocessPreview = {
  steps: PreprocessStep[];
  before_png_b64: string;
  after_png_b64: string;
};

// Client-side upload ceiling (backend enforces its own limit too).
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // 10 MB

export async function getHealth(): Promise<Health> {
  const r = await fetch("/health", { cache: "no-store" });
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export async function searchImage(file: Blob, signal?: AbortSignal): Promise<SearchResponse> {
  const form = new FormData();
  form.append("file", file, "query.jpg");
  const r = await fetch("/api/search", { method: "POST", body: form, signal });
  if (!r.ok) {
    let detail = `search failed (${r.status})`;
    try {
      const j = await r.json();
      if (j?.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return r.json();
}

// Read-only snapshot of the ML pipeline (dataset → preprocessing → backbone →
// index → calibration → eval → promotion). Sourced from on-disk artifacts.
export async function getPipeline(signal?: AbortSignal): Promise<PipelineData> {
  const r = await fetch("/api/pipeline", { cache: "no-store", signal });
  if (!r.ok) throw new Error(`pipeline unavailable (${r.status})`);
  return r.json();
}

// Runs the deterministic preprocessing pipeline on an uploaded image and returns
// before/after PNGs (base64) plus the ordered step list — for the live demo.
export async function preprocessPreview(
  file: Blob,
  signal?: AbortSignal,
): Promise<PreprocessPreview> {
  const form = new FormData();
  form.append("file", file, "query.jpg");
  const r = await fetch("/api/preprocess-preview", { method: "POST", body: form, signal });
  if (!r.ok) {
    let detail = `preview failed (${r.status})`;
    try {
      const j = await r.json();
      if (j?.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return r.json();
}

// ---- Embedding galaxy (GET /api/galaxy) ---------------------------------------
export type GalaxyPoint = {
  specimen_id: string;
  x: number;
  y: number;
  z: number;
  label: number;
  label_name: string;
  color: string; // stable per-species hex color, e.g. "#3fa06a"
};
export type GalaxyData = { points: GalaxyPoint[]; count: number };

// 3D projection of the gallery embeddings (PCA), for the Galaxy tab's
// fly-through point cloud + legend. Sourced from a precomputed artifact —
// cheap, no re-embedding on this path.
export async function getGalaxy(signal?: AbortSignal): Promise<GalaxyData> {
  const r = await fetch("/api/galaxy", { cache: "no-store", signal });
  if (!r.ok) throw new Error(`galaxy unavailable (${r.status})`);
  return r.json();
}

// Provisional confidence banding from raw cosine until calibrated confidence
// (Phase 3b) is present. Prefers the server's calibrated band/confidence.
export function bandFor(r: SearchResult): "high" | "medium" | "low" {
  if (r.band) return r.band;
  const v = r.confidence ?? r.score;
  if (v >= 0.85) return "high";
  if (v >= 0.7) return "medium";
  return "low";
}
