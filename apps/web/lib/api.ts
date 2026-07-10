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

// Per-device timing block in the device benchmark (one for "cpu", one for
// "cuda"). Extra keys (backbone, num_images, notes, …) vary by stage.
export type DeviceTiming = {
  requested_device?: string;
  resolved_device?: string;
  device?: string;
  backbone?: string;
  num_images?: number;
  elapsed_seconds: number;
  images_per_second?: number;
  note?: string;
  [key: string]: string | number | undefined;
};

// GPU-vs-CPU benchmark (GET /api/pipeline → device_benchmark). Null when the
// report hasn't been generated. Additive — older responses omit this key.
export type DeviceBenchmark = {
  batch_size: number | null;
  embed: { cpu?: DeviceTiming; cuda?: DeviceTiming } | null;
  embed_speedup_cuda_over_cpu: number | null;
  train_epoch: { cpu?: DeviceTiming; cuda?: DeviceTiming } | null;
  train_epoch_speedup_cuda_over_cpu: number | null;
  generated_at: string | null;
};

// Finetuning hyperparameter sweep summary (GET /api/pipeline → training).
// Null when the sweep summary hasn't been generated. Additive.
export type TrainingSummary = {
  backbone: string | null;
  num_runs: number | null;
  winner_run_id: string | null;
  winner_config: Record<string, number | string> | null;
  winner_val_metrics: Record<string, number> | null;
  generated_at: string | null;
};

export type PipelineData = {
  dataset: { name: string; total: number; classes: number | null; splits: Record<string, number> };
  preprocessing: { name: string; description: string }[];
  backbone: { name: string; version: string; dim: number | null; frozen: boolean };
  eval: { val: PipelineMetrics; test: PipelineMetrics };
  calibration: { ece_before: number | null; ece_after: number | null; bands: ReliabilityBin[] | null };
  promotion: { decision: string | null; reason: string | null };
  // Additive (kept optional so older backend responses still type-check).
  device_benchmark?: DeviceBenchmark | null;
  training?: TrainingSummary | null;
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

// ---- Opt-in API-key auth (PRD Phase 9) -----------------------------------
// Sent only on the sensitive/mutating endpoints (assistant, garden, memory)
// that the backend actually guards (see apps/api/app/auth.py). Unset by
// default, so the local demo sends no header and is unaffected.
function authHeaders(): Record<string, string> {
  const key = process.env.NEXT_PUBLIC_FLORALENS_API_KEY;
  return key ? { "X-API-Key": key } : {};
}

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

// ---- Species categories (GET /api/categories) ---------------------------------
export type Category = {
  label: number;
  label_name: string;
  gallery_count: number;
  total_count: number;
  specimen_id: string; // representative gallery specimen (thumbnail via /api/specimen/{id}/image)
  color: string; // stable per-species hex color (matches the Galaxy tab)
};
export type CategoriesData = { categories: Category[]; count: number; total_specimens: number };

// The 102 dataset species, one entry each with counts, a representative
// thumbnail specimen id, and the Galaxy-matching color — for the Categories
// tab. Sourced from the embeddings-cache metadata; cheap, no re-embedding.
export async function getCategories(signal?: AbortSignal): Promise<CategoriesData> {
  const r = await fetch("/api/categories", { cache: "no-store", signal });
  if (!r.ok) throw new Error(`categories unavailable (${r.status})`);
  return r.json();
}

// Provisional confidence banding from raw cosine until calibrated confidence
// (Phase 3b) is present. Prefers the server's calibrated band/confidence.
export function bandFor(r: SearchResult): "high" | "medium" | "low" {
  if (r.band) return r.band;
  // Band off the SAME rounded percentage the card displays, so a result can
  // never render "85%" while tagged/grouped medium (0.849 rounds to 85% but
  // is < 0.85). Grouping and the per-card badge both call this, staying in sync.
  const pct = Math.round((r.confidence ?? r.score) * 100);
  if (pct >= 85) return "high";
  if (pct >= 70) return "medium";
  return "low";
}

// ---- Naturalist assistant (POST /api/assistant, SSE) --------------------------
// Reuses AgentForge's Unified Agent Core (agent_core) — see
// apps/api/app/assistant_service.py. One line per Server-Sent Event; each
// decodes to a trace step ("model" | "tool" | "answer" | "limit") or a
// top-level "run_started" | "error" | "done" marker.
export type AssistantEvent = {
  type: string;
  step?: number;
  node?: string;
  detail?: string;
  usage?: Record<string, number>;
};

// Streams one assistant turn, invoking `onEvent` per SSE `data:` line as it
// arrives. Resolves once the stream ends (on `done`, `error`, or the response
// closing) — callers drive UI state from `onEvent`, not the return value.
export async function streamAssistant(
  message: string,
  threadId: string | undefined,
  onEvent: (event: AssistantEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch("/api/assistant", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(threadId ? { message, thread_id: threadId } : { message }),
    signal,
  });
  if (!r.ok || !r.body) {
    let detail = `assistant request failed (${r.status})`;
    try {
      const j = await r.json();
      if (j?.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const processChunk = (chunk: string) => {
    const line = chunk.trim();
    if (!line.startsWith("data:")) return;
    const payload = line.slice("data:".length).trim();
    if (!payload) return;
    try {
      onEvent(JSON.parse(payload) as AssistantEvent);
    } catch {
      /* skip a malformed SSE line rather than aborting the whole stream */
    }
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";
    for (const chunk of chunks) processChunk(chunk);
  }
  // Flush the multibyte tail + any final frame not terminated by a blank line,
  // so the terminal event (often `answer`/`done`) is never silently dropped.
  buffer += decoder.decode();
  for (const chunk of buffer.split("\n\n")) processChunk(chunk);
}

// Extracts unique http(s) URLs from an assistant answer, trimming trailing
// punctuation — powers the "cited sources" list under an answer bubble.
export function extractCitations(text: string): string[] {
  const matches = text.match(/https?:\/\/[^\s)]+/g) ?? [];
  const cleaned = matches.map((u) => u.replace(/[).,;:]+$/, ""));
  return [...new Set(cleaned)];
}

// ---- My Garden (GET/POST/DELETE /api/garden) — PRD Phase 7 ------------------
export type GardenItem = { specimen_id: string; label_name: string; saved_at: string };
export type GardenListResponse = { items: GardenItem[] };

async function readErrorDetail(r: Response, fallback: string): Promise<string> {
  try {
    const j = await r.json();
    if (j?.detail) return typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
  } catch {
    /* keep fallback */
  }
  return fallback;
}

export async function getGarden(signal?: AbortSignal): Promise<GardenListResponse> {
  const r = await fetch("/api/garden", { cache: "no-store", headers: authHeaders(), signal });
  if (!r.ok) throw new Error(await readErrorDetail(r, `garden unavailable (${r.status})`));
  return r.json();
}

export async function addToGarden(specimenId: string, signal?: AbortSignal): Promise<GardenItem> {
  const r = await fetch("/api/garden", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ specimen_id: specimenId }),
    signal,
  });
  if (!r.ok) throw new Error(await readErrorDetail(r, `could not save to garden (${r.status})`));
  return r.json();
}

export async function removeFromGarden(specimenId: string, signal?: AbortSignal): Promise<void> {
  const r = await fetch(`/api/garden/${encodeURIComponent(specimenId)}`, {
    method: "DELETE",
    headers: authHeaders(),
    signal,
  });
  if (!r.ok) throw new Error(await readErrorDetail(r, `could not remove from garden (${r.status})`));
}

// ---- Assistant memory inspector (GET/DELETE /api/memory) — PRD Phase 7 ------
export type MemoryItem = { id: string | null; text: string; meta: Record<string, unknown> };
export type MemoryListResponse = { items: MemoryItem[]; scope: string; namespace: string };
export type MemoryClearResponse = { deleted: number };

export async function getMemory(signal?: AbortSignal): Promise<MemoryListResponse> {
  const r = await fetch("/api/memory", { cache: "no-store", headers: authHeaders(), signal });
  if (!r.ok) throw new Error(await readErrorDetail(r, `memory unavailable (${r.status})`));
  return r.json();
}

export async function clearMemory(signal?: AbortSignal): Promise<MemoryClearResponse> {
  const r = await fetch("/api/memory", { method: "DELETE", headers: authHeaders(), signal });
  if (!r.ok) throw new Error(await readErrorDetail(r, `could not clear memory (${r.status})`));
  return r.json();
}
