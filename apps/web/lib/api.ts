// Thin client for the FloraLens backend. Same-origin via Next rewrites.

export type SearchResult = {
  specimen_id: string;
  label: number;
  label_name: string;
  score: number; // cosine similarity
  confidence?: number; // calibrated (added in Phase 3b, optional)
  band?: "high" | "medium" | "low";
};

export type SearchResponse = { model_version: string; results: SearchResult[] };
export type Health = { status: string; model_version: string };

export async function getHealth(): Promise<Health> {
  const r = await fetch("/health", { cache: "no-store" });
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export async function searchImage(file: Blob): Promise<SearchResponse> {
  const form = new FormData();
  form.append("file", file, "query.jpg");
  const r = await fetch("/api/search", { method: "POST", body: form });
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

// Provisional confidence banding from raw cosine until calibrated confidence
// (Phase 3b) is present. Prefers the server's calibrated band/confidence.
export function bandFor(r: SearchResult): "high" | "medium" | "low" {
  if (r.band) return r.band;
  const v = r.confidence ?? r.score;
  if (v >= 0.85) return "high";
  if (v >= 0.7) return "medium";
  return "low";
}
