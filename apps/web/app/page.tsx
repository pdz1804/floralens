"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  bandFor,
  getHealth,
  searchImage,
  type Health,
  type SearchResult,
} from "@/lib/api";

type Phase = "idle" | "searching" | "done" | "error";

export default function Page() {
  const [health, setHealth] = useState<Health | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [file, setFile] = useState<Blob | null>(null);
  const [urlInput, setUrlInput] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [modelVersion, setModelVersion] = useState<string>("");
  const [bandFilter, setBandFilter] = useState<"all" | "high" | "medium" | "low">("all");
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  const setImage = useCallback(
    (blob: Blob) => {
      if (blob.size === 0) return;
      if (!blob.type.startsWith("image/")) {
        setError("Please choose an image file (jpg, png, webp).");
        setPhase("error");
        return;
      }
      setError(null);
      setFile(blob);
      setPreviewUrl((old) => {
        if (old) URL.revokeObjectURL(old);
        return URL.createObjectURL(blob);
      });
      setResults([]);
      setPhase("idle");
    },
    [],
  );

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) setImage(f);
  }
  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setImage(f);
  }
  async function onLoadUrl() {
    if (!urlInput.trim()) return;
    setError(null);
    try {
      const r = await fetch(urlInput.trim());
      if (!r.ok) throw new Error(`fetch ${r.status}`);
      setImage(await r.blob());
    } catch (e) {
      setError(`Could not load image from URL: ${(e as Error).message}`);
      setPhase("error");
    }
  }

  async function onSearch() {
    if (!file) return;
    setPhase("searching");
    setError(null);
    setResults([]);
    try {
      const res = await searchImage(file);
      setResults(res.results);
      setModelVersion(res.model_version);
      setPhase("done");
    } catch (e) {
      setError((e as Error).message);
      setPhase("error");
    }
  }

  const shown = useMemo(
    () => (bandFilter === "all" ? results : results.filter((r) => bandFor(r) === bandFilter)),
    [results, bandFilter],
  );

  return (
    <>
      <div className="topbar">
        <span className="leaf">🌸</span>
        <h1>FloraLens</h1>
        <span className={`dot ${health ? "ok" : ""}`} data-testid="health-dot" />
        <span className="spacer" />
        <span className="meta" data-testid="health-meta">
          {health ? `model: ${health.model_version}` : "backend offline"}
        </span>
      </div>

      <div className="wrap">
        <div className="layout">
          {/* LEFT: input */}
          <div className="card">
            <h2>Query flower</h2>
            <div className="body">
              <div
                className={`dropzone ${dragging ? "drag" : ""}`}
                data-testid="dropzone"
                onClick={() => fileRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
              >
                <div className="big">📷</div>
                <p>Click, drag &amp; drop, or paste a flower photo</p>
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/*"
                  data-testid="file-input"
                  onChange={onPick}
                  style={{ display: "none" }}
                />
              </div>

              <label htmlFor="url">…or load from URL</label>
              <div style={{ display: "flex", gap: 8 }}>
                <input
                  id="url"
                  type="text"
                  placeholder="https://…/flower.jpg"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  data-testid="url-input"
                />
                <button className="secondary" style={{ width: "auto", marginTop: 0 }} onClick={onLoadUrl}>
                  Load
                </button>
              </div>

              {previewUrl && (
                <div className="preview" data-testid="preview">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={previewUrl} alt="query preview" />
                </div>
              )}

              <button data-testid="search-btn" onClick={onSearch} disabled={!file || phase === "searching"}>
                {phase === "searching" ? "Searching…" : "Find similar flowers"}
              </button>
              {error && <p className="err" data-testid="error" style={{ marginBottom: 0 }}>{error}</p>}
            </div>
          </div>

          {/* RIGHT: results */}
          <div className="card">
            <h2>
              Matches{modelVersion ? ` · model ${modelVersion}` : ""}
            </h2>
            <div className="body">
              {results.length > 0 && (
                <div className="filters" data-testid="filters">
                  {(["all", "high", "medium", "low"] as const).map((b) => (
                    <span
                      key={b}
                      className={`chip ${bandFilter === b ? "active" : ""}`}
                      data-testid={`filter-${b}`}
                      onClick={() => setBandFilter(b)}
                    >
                      {b}
                    </span>
                  ))}
                </div>
              )}

              {phase === "searching" && (
                <div className="state" data-testid="loading">
                  <div className="spinner" />
                  <p>Embedding &amp; searching the gallery…</p>
                </div>
              )}
              {phase === "idle" && results.length === 0 && (
                <div className="state">Choose a flower photo and search to see scored matches.</div>
              )}
              {phase === "done" && results.length === 0 && (
                <div className="state">No matches found.</div>
              )}

              {shown.length > 0 && (
                <div className="grid" data-testid="results">
                  {shown.map((r, i) => {
                    const b = bandFor(r);
                    const pct = Math.round((r.confidence ?? r.score) * 100);
                    return (
                      <div className="result" data-testid="result-card" key={r.specimen_id + i}>
                        <div className="rank">#{results.indexOf(r) + 1} · {r.specimen_id}</div>
                        <div className="name" data-testid="result-name">{r.label_name}</div>
                        <div className="barwrap">
                          <div className="bar" style={{ width: `${Math.max(3, pct)}%` }} />
                        </div>
                        <div className="scoreline">
                          <span className={`band ${b}`}>{b}</span>
                          <span className="pct" data-testid="result-score">{pct}%</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
              {results.length > 0 && (
                <p className="note">
                  Confidence banding is {results[0]?.band ? "calibrated (Phase 3b)" : "provisional from raw cosine (calibrated in Phase 3b)"}.
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
