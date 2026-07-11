"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
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
import { GalaxyPage } from "./galaxy-page";
import { CategoriesPage } from "./categories-page";
import { AssistantPage } from "./assistant-page";
import { GardenPage } from "./garden-page";
import { AboutPage } from "./about-page";

type Phase = "idle" | "searching" | "done" | "error";
type Tab = "search" | "pipeline" | "galaxy" | "categories" | "assistant" | "garden" | "about";

const TABS: { id: Tab; label: string }[] = [
  { id: "search", label: "Search" },
  { id: "pipeline", label: "Pipeline" },
  { id: "galaxy", label: "Galaxy" },
  { id: "categories", label: "Categories" },
  { id: "assistant", label: "Assistant" },
  { id: "garden", label: "Garden" },
  { id: "about", label: "About" },
];

// Display-only band metadata for the grouped results view + filter chips.
// Ordered strongest → weakest so sections always read High → Medium → Low.
type Band = "high" | "medium" | "low";
const BAND_ORDER: Band[] = ["high", "medium", "low"];
const BAND_META: Record<Band, { label: string; note: string }> = {
  high: { label: "High", note: "strong visual matches" },
  medium: { label: "Medium", note: "plausible — verify the details" },
  low: { label: "Low", note: "weaker matches, interpret with care" },
};
const CHIP_LABEL: Record<"all" | Band, string> = {
  all: "All",
  high: "High",
  medium: "Medium",
  low: "Low",
};

// Shared editorial easing (expo-out) for motivated entrance motion.
const EASE = [0.16, 1, 0.3, 1] as const;

export default function Page() {
  const reduce = useReducedMotion() ?? false;
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
  // Read-only bucketing of results by confidence band, preserving score order.
  // Powers the grouped "All" view, per-chip counts and the empty-state hint —
  // derived purely from `results` + `bandFor`, never mutates search state.
  const groups = useMemo(() => {
    const g: Record<Band, SearchResult[]> = { high: [], medium: [], low: [] };
    for (const r of results) g[bandFor(r)].push(r);
    return g;
  }, [results]);
  const counts: Record<Band, number> = {
    high: groups.high.length,
    medium: groups.medium.length,
    low: groups.low.length,
  };
  // Bands to render as sections: every non-empty band in "All", else the one
  // selected band. Empty selection falls through to the band-aware empty state.
  const visibleBands: Band[] =
    bandFilter === "all" ? BAND_ORDER.filter((b) => counts[b] > 0) : [bandFilter];
  const suggestBand = BAND_ORDER.find((b) => counts[b] > 0);
  // Summary reflects the CURRENT view: top of `shown`, not always results[0].
  const shownTop = shown[0];
  const shownDistinct = useMemo(
    () => new Set(shown.map((r) => r.label_name)).size,
    [shown],
  );

  // Single source of truth for a result card so grouped sections and the
  // filtered view stay identical. Display-only — preserves every data-testid.
  const renderCard = (r: SearchResult, i: number) => {
    const b = bandFor(r);
    const pct = Math.max(0, Math.round((r.confidence ?? r.score) * 100));
    return (
      <motion.article
        className="result"
        data-testid="result-card"
        key={r.specimen_id}
        initial={reduce ? false : { opacity: 0, y: 12 }}
        animate={reduce ? undefined : { opacity: 1, y: 0 }}
        transition={reduce ? undefined : { duration: 0.35, ease: EASE, delay: Math.min(i * 0.05, 0.4) }}
        whileHover={reduce ? undefined : { y: -4 }}
        whileTap={reduce ? undefined : { scale: 0.98 }}
      >
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
        </div>
      </motion.article>
    );
  };

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
        <motion.section
          className="hero"
          {...(reduce
            ? {}
            : {
                initial: { opacity: 0, y: 12 },
                animate: { opacity: 1, y: 0 },
                transition: { duration: 0.45, ease: EASE },
              })}
        >
          <span className="eyebrow">
            <BloomIcon width={14} height={14} /> Botanical intelligence
          </span>
          <h2>Find the flowers that look like yours.</h2>
          <p>
            Upload or link a photo and FloraLens embeds it, then surfaces visually similar
            specimens from the gallery — each scored with a calibrated confidence band.
          </p>
        </motion.section>

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
                <span className="input-affix">
                  <LinkIcon width={16} height={16} aria-hidden="true" />
                  <input
                    id="url"
                    type="text"
                    placeholder="https://…/flower.jpg"
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                    data-testid="url-input"
                  />
                </span>
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
                  {(["all", "high", "medium", "low"] as const).map((b) => {
                    const n = b === "all" ? results.length : counts[b];
                    const disabled = b !== "all" && n === 0;
                    return (
                      <button
                        type="button"
                        key={b}
                        className={`chip ${b !== "all" ? `chip-${b}` : ""} ${
                          bandFilter === b ? "active" : ""
                        } ${disabled ? "chip-disabled" : ""}`}
                        data-testid={`filter-${b}`}
                        aria-pressed={bandFilter === b}
                        aria-disabled={disabled || undefined}
                        onClick={() => {
                          if (!disabled) setBandFilter(b);
                        }}
                      >
                        <span className="chip-label">{CHIP_LABEL[b]}</span>
                        <span className="chip-count">{n}</span>
                      </button>
                    );
                  })}
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

              {/* Summary tracks the active filter: top of the CURRENT view. */}
              {phase === "done" && shownTop && (
                <div className="top-match" data-testid="top-match">
                  <div className="top-match-thumb" aria-hidden="true">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`/api/specimen/${encodeURIComponent(shownTop.specimen_id)}/image`}
                      alt=""
                      loading="lazy"
                      onError={(e) => {
                        e.currentTarget.style.visibility = "hidden";
                      }}
                    />
                  </div>
                  <div className="top-match-body">
                    <span className="top-match-label">
                      <SparkleIcon width={13} height={13} aria-hidden="true" />{" "}
                      {bandFilter === "all"
                        ? "Top match"
                        : `Top ${BAND_META[bandFilter].label.toLowerCase()} match`}
                    </span>
                    <div className="top-match-headline">
                      <h4 className="top-match-name">{shownTop.label_name}</h4>
                      <span className={`band ${bandFor(shownTop)}`}>{bandFor(shownTop)}</span>
                      {shownDistinct === 1 && shown.length > 1 && (
                        <span className="top-match-count">
                          all {shown.length} matches
                        </span>
                      )}
                    </div>
                    {shownTop.description ? (
                      <ResultDesc text={shownTop.description} />
                    ) : null}
                  </div>
                </div>
              )}

              {/* Grouped-by-confidence view: every match visible on one page,
                  organised High → Medium → Low. In a filtered view only the
                  selected band's section renders. */}
              {results.length > 0 && shown.length > 0 && (
                <div className="results-groups" data-testid="results">
                  {visibleBands.map((b) =>
                    counts[b] > 0 ? (
                      <section
                        className="band-section"
                        data-testid={`band-section-${b}`}
                        key={b}
                      >
                        <header className="band-section-head">
                          <span className={`band-section-title ${b}`}>
                            <span className="band-dot" aria-hidden="true" />
                            {BAND_META[b].label}
                          </span>
                          <span className={`band-section-count ${b}`}>{counts[b]}</span>
                          <span className="band-section-note">{BAND_META[b].note}</span>
                        </header>
                        <div className="grid">{groups[b].map((r, i) => renderCard(r, i))}</div>
                      </section>
                    ) : null,
                  )}
                </div>
              )}

              {/* A selected band with no members never leaves a blank grid. */}
              {results.length > 0 && shown.length === 0 && bandFilter !== "all" && (
                <div className="state" data-testid="band-empty">
                  <span className="state-ico" aria-hidden="true">
                    <SparkleIcon width={26} height={26} />
                  </span>
                  <span className="state-title">
                    No {BAND_META[bandFilter].label.toLowerCase()}-confidence matches
                  </span>
                  <p>
                    No {BAND_META[bandFilter].label.toLowerCase()}-confidence matches for this
                    photo{suggestBand ? ` — try the ${BAND_META[suggestBand].label} band` : ""}.
                  </p>
                  <button
                    type="button"
                    className="btn btn-ghost band-empty-cta"
                    onClick={() => setBandFilter("all")}
                  >
                    Show all matches
                  </button>
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
        {tab === "galaxy" && (
          <div role="tabpanel" id="panel-galaxy" aria-labelledby="tab-galaxy">
            <GalaxyPage />
          </div>
        )}
        {tab === "categories" && (
          <div role="tabpanel" id="panel-categories" aria-labelledby="tab-categories">
            <CategoriesPage />
          </div>
        )}
        {tab === "assistant" && (
          <div role="tabpanel" id="panel-assistant" aria-labelledby="tab-assistant">
            <AssistantPage />
          </div>
        )}
        {tab === "garden" && (
          <div role="tabpanel" id="panel-garden" aria-labelledby="tab-garden">
            <GardenPage />
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
