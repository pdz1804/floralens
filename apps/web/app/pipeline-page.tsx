"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getPipeline,
  MAX_UPLOAD_BYTES,
  preprocessPreview,
  type PipelineData,
  type PipelineMetrics,
  type PreprocessPreview,
} from "@/lib/api";
import {
  AlertIcon,
  CameraLeafIcon,
  ChartIcon,
  CheckIcon,
  DatabaseIcon,
  GaugeIcon,
  GridIcon,
  LayersIcon,
  WandIcon,
} from "./icons";

/* ---- formatting helpers (defensive: fields vary / may be null) ------------- */
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const pct = (v: unknown, d = 1) => (isNum(v) ? `${(v * 100).toFixed(d)}%` : "—");
const dec = (v: unknown, d = 3) => (isNum(v) ? v.toFixed(d) : "—");
const int = (v: unknown) => (isNum(v) ? Math.round(v).toLocaleString() : "—");

// Curated metric order + display; renders defensively over whatever keys exist.
const METRIC_ROWS: { key: string; label: string; kind: "pct" | "dec" | "int" }[] = [
  { key: "recall@1", label: "Recall@1", kind: "pct" },
  { key: "recall@5", label: "Recall@5", kind: "pct" },
  { key: "recall@10", label: "Recall@10", kind: "pct" },
  { key: "precision@5", label: "Precision@5", kind: "pct" },
  { key: "map@5", label: "mAP@5", kind: "pct" },
  { key: "map@10", label: "mAP@10", kind: "pct" },
  { key: "mrr", label: "MRR", kind: "pct" },
  { key: "silhouette", label: "Silhouette", kind: "dec" },
  { key: "num_queries", label: "Queries", kind: "int" },
  { key: "num_gallery", label: "Gallery", kind: "int" },
];

function fmt(v: unknown, kind: "pct" | "dec" | "int") {
  return kind === "pct" ? pct(v) : kind === "dec" ? dec(v) : int(v);
}

function Stage({
  n,
  icon: Icon,
  title,
  sub,
  children,
}: {
  n: number;
  icon: (p: { width?: number; height?: number }) => JSX.Element;
  title: string;
  sub: string;
  children: React.ReactNode;
}) {
  return (
    <section className="stage">
      <div className="stage-rail" aria-hidden="true">
        <span className="stage-node">
          <Icon width={18} height={18} />
        </span>
      </div>
      <div className="stage-body">
        <header className="stage-head">
          <span className="stage-n">Stage {n}</span>
          <h3>{title}</h3>
          <p className="stage-sub">{sub}</p>
        </header>
        {children}
      </div>
    </section>
  );
}

/* ---- Metrics table (val vs test) ------------------------------------------- */
function MetricsTable({ val, test }: { val: PipelineMetrics; test: PipelineMetrics }) {
  const rows = METRIC_ROWS.filter((r) => (val && r.key in val) || (test && r.key in test));
  if (rows.length === 0) return <p className="muted-note">No evaluation report available yet.</p>;
  return (
    <div className="table-wrap">
      <table className="metrics">
        <thead>
          <tr>
            <th scope="col">Metric</th>
            <th scope="col">Validation</th>
            <th scope="col">Test</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key}>
              <th scope="row">{r.label}</th>
              <td>{fmt(val?.[r.key], r.kind)}</td>
              <td>{fmt(test?.[r.key], r.kind)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---- Live before/after preprocessing demo ---------------------------------- */
function PreprocessDemo() {
  const [phase, setPhase] = useState<"idle" | "running" | "done" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<PreprocessPreview | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => () => abort.current?.abort(), []);

  const run = useCallback(async (blob: Blob) => {
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
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setPhase("running");
    setError(null);
    try {
      const res = await preprocessPreview(blob, ctrl.signal);
      setPreview(res);
      setPhase("done");
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setError((e as Error).message);
      setPhase("error");
    }
  }, []);

  return (
    <div className="demo" data-testid="preprocess-demo">
      <div
        className={`dropzone demo-drop ${dragging ? "drag" : ""}`}
        role="button"
        tabIndex={0}
        aria-label="Upload a photo to preview preprocessing"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f) run(f);
        }}
      >
        <span className="ico" aria-hidden="true">
          {phase === "running" ? <span className="spin" /> : <CameraLeafIcon width={24} height={24} />}
        </span>
        <p className="lead">{phase === "running" ? "Processing…" : "Try it on your own photo"}</p>
        <p className="hint">Click or drop an image to see each transform applied</p>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) run(f);
          }}
        />
      </div>

      {phase === "error" && error && (
        <p className="err" role="alert">
          <AlertIcon width={16} height={16} aria-hidden="true" />
          <span>{error}</span>
        </p>
      )}

      {preview && phase !== "error" && (
        <div className="demo-out">
          <div className="ba">
            <figure>
              <span className="ba-tag">Before</span>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={`data:image/png;base64,${preview.before_png_b64}`} alt="uploaded photo, before preprocessing" />
            </figure>
            <span className="ba-arrow" aria-hidden="true">→</span>
            <figure>
              <span className="ba-tag out">After</span>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={`data:image/png;base64,${preview.after_png_b64}`} alt="the same photo after preprocessing" />
            </figure>
          </div>
          <ol className="demo-steps">
            {preview.steps.map((s, i) => (
              <li key={s.name}>
                <span className="demo-step-n">{i + 1}</span>
                <div>
                  <strong>{s.name.replace(/_/g, " ")}</strong>
                  <span>{s.description}</span>
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

/* ---- Page ------------------------------------------------------------------ */
export function PipelinePage() {
  const [data, setData] = useState<PipelineData | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    getPipeline(ctrl.signal)
      .then((d) => {
        setData(d);
        setStatus("ready");
      })
      .catch((e) => {
        if ((e as Error).name === "AbortError") return;
        setError((e as Error).message);
        setStatus("error");
      });
    return () => ctrl.abort();
  }, []);

  if (status === "loading") {
    return (
      <div className="pipeline" data-testid="pipeline-page">
        <div className="state" aria-busy="true">
          <span className="spin" aria-hidden="true" />
          <span className="state-title">Loading the pipeline…</span>
          <p>Reading the dataset, model, and evaluation artifacts.</p>
        </div>
      </div>
    );
  }

  if (status === "error" || !data) {
    return (
      <div className="pipeline" data-testid="pipeline-page">
        <div className="state error">
          <span className="state-ico" aria-hidden="true">
            <AlertIcon width={26} height={26} />
          </span>
          <span className="state-title">Pipeline unavailable</span>
          <p>{error ?? "Could not load the pipeline snapshot. Is the backend running?"}</p>
        </div>
      </div>
    );
  }

  const { dataset, preprocessing, backbone, eval: ev, calibration, promotion } = data;
  const splitEntries = Object.entries(dataset.splits ?? {});
  const galleryFromEval = ev.test?.["num_gallery"] ?? ev.val?.["num_gallery"];
  const decision = (promotion.decision ?? "").toUpperCase();
  const promoted = decision === "PROMOTE" || decision === "PROMOTED";

  return (
    <div className="pipeline" data-testid="pipeline-page">
      <section className="pipeline-hero">
        <span className="eyebrow">
          <GridIcon width={14} height={14} /> How it works
        </span>
        <h2>From a photo to a calibrated match.</h2>
        <p>
          FloraLens is transparent end to end. Every stage below is sourced from the artifacts
          actually written during training and evaluation — nothing is re-run or estimated here.
        </p>
        <p className="model-chip">
          Active model <code>{data.model_version}</code>
        </p>
      </section>

      <div className="stages">
        {/* 1 — Dataset */}
        <Stage n={1} icon={DatabaseIcon} title="Dataset" sub="The gallery every query is matched against.">
          <div className="stat-row">
            <div className="stat">
              <span className="stat-k">Source</span>
              <span className="stat-v">{dataset.name ?? "—"}</span>
            </div>
            <div className="stat">
              <span className="stat-k">Specimens</span>
              <span className="stat-v">{int(dataset.total)}</span>
            </div>
            <div className="stat">
              <span className="stat-k">Classes</span>
              <span className="stat-v">{int(dataset.classes)}</span>
            </div>
          </div>
          {splitEntries.length > 0 && (
            <div className="splits">
              {splitEntries.map(([k, v]) => (
                <span className="split-chip" key={k}>
                  <em>{k}</em> {int(v)}
                </span>
              ))}
            </div>
          )}
        </Stage>

        {/* 2 — Preprocessing */}
        <Stage
          n={2}
          icon={WandIcon}
          title="Preprocessing"
          sub="A deterministic transform applied identically to gallery and query images."
        >
          <ol className="prep-list">
            {preprocessing.map((s, i) => (
              <li key={s.name}>
                <span className="prep-n">{i + 1}</span>
                <div>
                  <strong>{s.name.replace(/_/g, " ")}</strong>
                  <span>{s.description}</span>
                </div>
              </li>
            ))}
          </ol>
          <div className="demo-frame">
            <span className="demo-frame-label">Live demo</span>
            <PreprocessDemo />
          </div>
        </Stage>

        {/* 3 — Backbone */}
        <Stage n={3} icon={LayersIcon} title="Embedding backbone" sub="A frozen vision model turns each image into a vector.">
          <div className="stat-row">
            <div className="stat">
              <span className="stat-k">Model</span>
              <span className="stat-v mono">{backbone.name ?? "—"}</span>
            </div>
            <div className="stat">
              <span className="stat-k">Embedding dim</span>
              <span className="stat-v">{int(backbone.dim)}</span>
            </div>
            <div className="stat">
              <span className="stat-k">Weights</span>
              <span className="stat-v">
                {backbone.frozen ? <span className="badge frozen">Frozen</span> : "Trainable"}
              </span>
            </div>
          </div>
        </Stage>

        {/* 4 — Index */}
        <Stage n={4} icon={GridIcon} title="Vector index" sub="Query and gallery vectors are compared by cosine similarity.">
          <p className="stage-text">
            Every gallery specimen is embedded once and stored. At search time your photo is embedded
            with the same backbone, then ranked against the gallery by cosine similarity — the raw
            score behind each match.
          </p>
          {isNum(galleryFromEval) && (
            <div className="stat-row">
              <div className="stat">
                <span className="stat-k">Indexed vectors</span>
                <span className="stat-v">{int(galleryFromEval)}</span>
              </div>
              <div className="stat">
                <span className="stat-k">Similarity</span>
                <span className="stat-v">Cosine</span>
              </div>
            </div>
          )}
        </Stage>

        {/* 5 — Calibration */}
        <Stage n={5} icon={GaugeIcon} title="Confidence calibration" sub="Raw scores become honest High / Medium / Low bands.">
          <div className="stat-row">
            <div className="stat">
              <span className="stat-k">ECE before</span>
              <span className="stat-v">{dec(calibration.ece_before, 4)}</span>
            </div>
            <div className="stat accent">
              <span className="stat-k">ECE after</span>
              <span className="stat-v">{dec(calibration.ece_after, 4)}</span>
            </div>
            <div className="stat">
              <span className="stat-k">Reduction</span>
              <span className="stat-v">
                {isNum(calibration.ece_before) && isNum(calibration.ece_after) && calibration.ece_before > 0
                  ? `${(100 * (1 - calibration.ece_after / calibration.ece_before)).toFixed(1)}%`
                  : "—"}
              </span>
            </div>
          </div>
          {calibration.bands && calibration.bands.length > 0 && (
            <div className="reliability" aria-label="Reliability: predicted confidence vs empirical accuracy">
              <div className="reliability-head">
                <span>Confidence</span>
                <span>Accuracy</span>
                <span>Samples</span>
              </div>
              {calibration.bands
                .filter((b) => b.count > 0)
                .map((b) => (
                  <div className="reliability-row" key={`${b.bin_lower}-${b.bin_upper}`}>
                    <span className="rb-range">
                      {Math.round(b.bin_lower * 100)}–{Math.round(b.bin_upper * 100)}%
                    </span>
                    <div className="rb-bars">
                      <span className="rb-bar conf" style={{ width: `${b.mean_confidence * 100}%` }} />
                      <span className="rb-bar acc" style={{ width: `${b.empirical_accuracy * 100}%` }} />
                    </div>
                    <span className="rb-count">{int(b.count)}</span>
                  </div>
                ))}
              <div className="reliability-key">
                <span>
                  <i className="k conf" /> mean confidence
                </span>
                <span>
                  <i className="k acc" /> empirical accuracy
                </span>
              </div>
            </div>
          )}
        </Stage>

        {/* 6 — Evaluation */}
        <Stage n={6} icon={ChartIcon} title="Evaluation" sub="Held-out validation and test — no leakage between splits.">
          <MetricsTable val={ev.val} test={ev.test} />
        </Stage>

        {/* 7 — Promotion */}
        <Stage n={7} icon={CheckIcon} title="Promotion decision" sub="Whether this candidate model shipped as active.">
          {promotion.decision ? (
            <>
              <span className={`promo-badge ${promoted ? "go" : "hold"}`}>
                {promoted ? <CheckIcon width={16} height={16} /> : <AlertIcon width={16} height={16} />}
                {promotion.decision}
              </span>
              {promotion.reason && <p className="promo-reason">{promotion.reason}</p>}
            </>
          ) : (
            <p className="muted-note">No promotion decision recorded yet.</p>
          )}
        </Stage>
      </div>
    </div>
  );
}
