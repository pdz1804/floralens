"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  bandFor,
  getHealth,
  MAX_UPLOAD_BYTES,
  searchImage,
  type Health,
  type SearchResult,
} from "@/lib/api";
import {
  AlertIcon,
  BloomIcon,
  CameraLeafIcon,
  LinkIcon,
  SearchIcon,
  SparkleIcon,
} from "./icons";
import { ThemeToggle } from "./theme-toggle";
import { ResultDesc } from "./result-desc";
import { PipelinePage } from "./pipeline-page";
import { AboutPage } from "./about-page";

type Phase = "idle" | "searching" | "done" | "error";
type Tab = "search" | "pipeline" | "about";

const TABS: { id: Tab; label: string }[] = [
  { id: "search", label: "Search" },
  { id: "pipeline", label: "Pipeline" },
  { id: "about", label: "About" },
];

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
  const [tab, setTab] = useState<Tab>("search");
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  // Guards against a stale search response overwriting results for a newer image.
  const searchSeq = useRef(0);
  const searchAbort = useRef<AbortController | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
    return () => searchAbort.current?.abort();
  }, []);

  const setImage = useCallback((blob: Blob) => {
    if (blob.size === 0) {
      setError("That file is empty — choose a flower photo.");
      setPhase("error");
      return;
    }
    if (!blob.type.startsWith("image/")) {
      setError("Please choose an image file (jpg, png, webp).");
      setPhase("error");
      return;
    }
    if (blob.size > MAX_UPLOAD_BYTES) {
      setError(`Image is too large (max ${Math.round(MAX_UPLOAD_BYTES / 1024 / 1024)} MB).`);
      setPhase("error");
      return;
    }
    // Invalidate any in-flight search so its late response is ignored.
    searchSeq.current += 1;
    searchAbort.current?.abort();
    setError(null);
    setFile(blob); // preview URL is derived from `file` in the effect below.
    setResults([]);
    setPhase("idle");
  }, []);

  // Own the object-URL lifecycle in one place, keyed on the selected file.
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

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
    const seq = ++searchSeq.current;
    const ctrl = new AbortController();
    searchAbort.current = ctrl;
    setPhase("searching");
    setError(null);
    setResults([]);
    setBandFilter("all");
    try {
      const res = await searchImage(file, ctrl.signal);
      if (seq !== searchSeq.current) return; // superseded by a newer image/search
      setResults(res.results);
      setModelVersion(res.model_version);
      setPhase("done");
    } catch (e) {
      if ((e as Error).name === "AbortError" || seq !== searchSeq.current) return;
      setError((e as Error).message);
      setPhase("error");
    }
  }

  // Roving-focus keyboard nav for the tablist (WAI-ARIA tabs pattern).
  function onTabKey(e: React.KeyboardEvent, i: number) {
    let next = i;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (i + 1) % TABS.length;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = (i - 1 + TABS.length) % TABS.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = TABS.length - 1;
    else return;
    e.preventDefault();
    setTab(TABS[next].id);
    tabRefs.current[next]?.focus();
  }

  const shown = useMemo(
    () => (bandFilter === "all" ? results : results.filter((r) => bandFor(r) === bandFilter)),
    [results, bandFilter],
  );
  // Stable original rank per specimen (avoids O(n²) indexOf during render).
  const rankOf = useMemo(() => {
    const m = new Map<string, number>();
    results.forEach((r, i) => m.set(r.specimen_id, i + 1));
    return m;
  }, [results]);

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="mark" aria-hidden="true">
            <BloomIcon width={22} height={22} />
          </span>
          <div>
            <h1 className="wordmark">FloraLens</h1>
            <div className="tagline">Visual flower discovery &amp; similarity search</div>
          </div>
        </div>
        <nav className="tabs" role="tablist" aria-label="Sections">
          {TABS.map((t, i) => (
            <button
              key={t.id}
              ref={(el) => {
                tabRefs.current[i] = el;
              }}
              type="button"
              role="tab"
              id={`tab-${t.id}`}
              aria-selected={tab === t.id}
              aria-controls={`panel-${t.id}`}
              tabIndex={tab === t.id ? 0 : -1}
              className={`tab ${tab === t.id ? "active" : ""}`}
              data-testid={`tab-${t.id}`}
              onClick={() => setTab(t.id)}
              onKeyDown={(e) => onTabKey(e, i)}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <span className="spacer" />
        <div className="topbar-actions">
          <ThemeToggle />
          <div className="status">
            <span className={`dot ${health ? "ok" : ""}`} data-testid="health-dot" />
            <span className="meta" data-testid="health-meta">
              {health ? `model: ${health.model_version}` : "backend offline"}
            </span>
          </div>
        </div>
      </header>

      <main className="wrap">
        <div
          role="tabpanel"
          id="panel-search"
          aria-labelledby="tab-search"
          hidden={tab !== "search"}
        >
        <section className="hero">
          <span className="eyebrow">
            <BloomIcon width={14} height={14} /> Botanical intelligence
          </span>
          <h2>Find the flowers that look like yours.</h2>
          <p>
            Upload or link a photo and FloraLens embeds it, then surfaces visually similar
            specimens from the gallery — each scored with a calibrated confidence band.
          </p>
        </section>

        <div className="layout">
          {/* LEFT: input */}
          <section className="card left-card" aria-label="Query flower">
            <div className="head">
              <h3 className="title">Query flower</h3>
              <span className="sub">upload · url</span>
            </div>
            <div className="body">
              <div
                className={`dropzone ${dragging ? "drag" : ""}`}
                data-testid="dropzone"
                role="button"
                tabIndex={0}
                aria-label="Upload a flower photo"
                onClick={() => fileRef.current?.click()}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    fileRef.current?.click();
                  }
                }}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
              >
                <span className="ico" aria-hidden="true">
                  <CameraLeafIcon width={26} height={26} />
                </span>
                <p className="lead">Drop a flower photo here</p>
                <p className="hint">Click to browse, drag &amp; drop, or paste an image</p>
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/*"
                  data-testid="file-input"
                  onChange={onPick}
                  style={{ display: "none" }}
                />
              </div>

              <label className="field-label" htmlFor="url">
                <LinkIcon width={15} height={15} /> …or load from a URL
              </label>
              <div className="url-row">
                <input
                  id="url"
                  type="text"
                  placeholder="https://…/flower.jpg"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  data-testid="url-input"
                />
                <button type="button" className="btn btn-ghost" onClick={onLoadUrl}>
                  Load
                </button>
              </div>

              {previewUrl && (
                <div className="preview" data-testid="preview">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={previewUrl} alt="query preview" />
                </div>
              )}

              <button
                type="button"
                className="btn btn-primary"
                data-testid="search-btn"
                onClick={onSearch}
                disabled={!file || phase === "searching"}
              >
                {phase === "searching" ? (
                  <>
                    <span className="spin" aria-hidden="true" /> Searching…
                  </>
                ) : (
                  <>
                    <SearchIcon width={17} height={17} aria-hidden="true" /> Find similar flowers
                  </>
                )}
              </button>
              {error && (
                <p className="err" data-testid="error">
                  <AlertIcon width={16} height={16} aria-hidden="true" />
                  <span>{error}</span>
                </p>
              )}
            </div>
          </section>

          {/* RIGHT: results */}
          <section className="card" aria-label="Matches">
            <div className="head">
              <h3 className="title">Matches</h3>
              {modelVersion && <span className="sub">model {modelVersion}</span>}
            </div>
            <div className="body">
              {results.length > 0 && (
                <div className="filters" data-testid="filters">
                  {(["all", "high", "medium", "low"] as const).map((b) => (
                    <button
                      type="button"
                      key={b}
                      className={`chip ${bandFilter === b ? "active" : ""}`}
                      data-testid={`filter-${b}`}
                      aria-pressed={bandFilter === b}
                      onClick={() => setBandFilter(b)}
                    >
                      {b}
                    </button>
                  ))}
                </div>
              )}

              {phase === "searching" && (
                <div className="grid" data-testid="loading" aria-busy="true" aria-label="Searching the gallery">
                  {Array.from({ length: 6 }).map((_, i) => (
                    <div className="skeleton-card" key={i}>
                      <div className="sk sk-thumb" />
                      <div className="sk-body">
                        <div className="sk sk-line w-70" />
                        <div className="sk sk-bar" />
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {phase === "idle" && results.length === 0 && (
                <div className="state">
                  <span className="state-ico" aria-hidden="true">
                    <SparkleIcon width={26} height={26} />
                  </span>
                  <span className="state-title">Ready when you are</span>
                  <p>Choose a flower photo and search to see scored, ranked matches.</p>
                </div>
              )}
              {phase === "error" && results.length === 0 && (
                <div className="state error">
                  <span className="state-ico" aria-hidden="true">
                    <AlertIcon width={26} height={26} />
                  </span>
                  <span className="state-title">Something went wrong</span>
                  {/* Specific reason is shown once, next to the query panel. */}
                  <p>Check the photo and try again — the details are on the left.</p>
                </div>
              )}
              {phase === "done" && results.length === 0 && (
                <div className="state">
                  <span className="state-ico" aria-hidden="true">
                    <SparkleIcon width={26} height={26} />
                  </span>
                  <span className="state-title">No matches found</span>
                  <p>Try a clearer, closer photo of a single bloom.</p>
                </div>
              )}

              {shown.length > 0 && (
                <div className="grid" data-testid="results">
                  {shown.map((r) => {
                    const b = bandFor(r);
                    const pct = Math.max(0, Math.round((r.confidence ?? r.score) * 100));
                    return (
                      <article className="result" data-testid="result-card" key={r.specimen_id}>
                        <div className="thumb-wrap">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            className="thumb"
                            data-testid="result-thumb"
                            src={`/api/specimen/${encodeURIComponent(r.specimen_id)}/image`}
                            alt={r.label_name}
                            loading="lazy"
                            onError={(e) => {
                              e.currentTarget.style.visibility = "hidden";
                            }}
                          />
                          <span className="rank">#{rankOf.get(r.specimen_id)} · {r.specimen_id}</span>
                        </div>
                        <div className="meta">
                          <h4 className="name" data-testid="result-name">{r.label_name}</h4>
                          <div className="barwrap">
                            <div className={`bar ${b}`} style={{ width: `${Math.max(3, pct)}%` }} />
                          </div>
                          <div className="scoreline">
                            <span className={`band ${b}`}>{b}</span>
                            <span className="pct" data-testid="result-score">{pct}%</span>
                          </div>
                          {r.description ? <ResultDesc text={r.description} /> : null}
                        </div>
                      </article>
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
          </section>
        </div>
        </div>

        {tab === "pipeline" && (
          <div role="tabpanel" id="panel-pipeline" aria-labelledby="tab-pipeline">
            <PipelinePage />
          </div>
        )}
        {tab === "about" && (
          <div role="tabpanel" id="panel-about" aria-labelledby="tab-about" tabIndex={0}>
            <AboutPage />
          </div>
        )}
      </main>
    </>
  );
}
