"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getPipeline,
  MAX_UPLOAD_BYTES,
  preprocessPreview,
  type DeviceBenchmark,
  type DeviceTiming,
  type PipelineData,
  type PipelineMetrics,
  type PreprocessPreview,
  type TrainingSummary,
} from "@/lib/api";
import {
  AlertIcon,
  CameraLeafIcon,
  ChartIcon,
  CheckIcon,
  CpuIcon,
  DatabaseIcon,
  GaugeIcon,
  GridIcon,
  LayersIcon,
  WandIcon,
} from "./icons";
import styles from "./pipeline.module.css";

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

/* ---- Model card (at a glance) ---------------------------------------------- */
// Best-effort read of the compute device actually exercised, from whichever
// benchmark stage ran. Honest fallback when no benchmark was written.
function computeDevice(db: DeviceBenchmark | null | undefined): string {
  if (!db) return "—";
  if (db.embed?.cuda || db.train_epoch?.cuda) return "GPU · CUDA";
  if (db.embed?.cpu || db.train_epoch?.cpu) return "CPU";
  return "—";
}

function ModelCard({ data }: { data: PipelineData }) {
  const { eval: ev, calibration, promotion } = data;
  const decision = (promotion.decision ?? "").toUpperCase();
  const promoted = decision === "PROMOTE" || decision === "PROMOTED";
  const testR5 = ev.test?.["recall@5"];
  const device = computeDevice(data.device_benchmark);

  return (
    <section className={styles.modelCard} data-testid="pipeline-model-card">
      <div className={styles.modelCardHead}>
        <span className="eyebrow">
          <GridIcon width={14} height={14} /> At a glance
        </span>
        <h3>Model card</h3>
        <p>
          The active checkpoint in one row — what shipped, how well it scored on the held-out test
          split, how honest its confidence is, and where it ran.
        </p>
      </div>
      <div className={styles.tiles}>
        <div className={styles.tile}>
          <span className={styles.tileK}>Active model</span>
          <span className={`${styles.tileV} ${styles.mono}`}>{data.model_version ?? "—"}</span>
        </div>
        <div className={styles.tile}>
          <span className={styles.tileK}>Promotion</span>
          {promotion.decision ? (
            <span className={`${styles.miniBadge} ${promoted ? styles.go : styles.hold}`}>
              {promoted ? <CheckIcon width={13} height={13} /> : <AlertIcon width={13} height={13} />}
              {promotion.decision}
            </span>
          ) : (
            <span className={styles.tileV}>—</span>
          )}
        </div>
        <div className={`${styles.tile} ${styles.accent}`}>
          <span className={styles.tileK}>Test Recall@5</span>
          <span className={styles.tileV}>{pct(testR5)}</span>
          <span className={styles.tileSub}>held-out split, one pass</span>
        </div>
        <div className={styles.tile}>
          <span className={styles.tileK}>ECE (calibrated)</span>
          <span className={styles.tileV}>{dec(calibration.ece_after, 4)}</span>
          <span className={styles.tileSub}>lower is more honest</span>
        </div>
        <div className={styles.tile}>
          <span className={styles.tileK}>Compute</span>
          <span className={styles.tileV}>{device}</span>
        </div>
      </div>
    </section>
  );
}

/* ---- Device benchmark (GPU vs CPU) ----------------------------------------- */
// One labelled comparison of the same workload on cpu vs cuda. Bars scale to the
// larger of the two timings so the gap reads instantly; renders only the devices
// actually present. `betterHigh` flips which end of the scale is the good one.
function CompareBars({
  cpu,
  cuda,
  value,
  format,
  betterHigh,
}: {
  cpu?: DeviceTiming;
  cuda?: DeviceTiming;
  value: (t: DeviceTiming) => number | undefined;
  format: (v: number | undefined) => string;
  betterHigh: boolean;
}) {
  const rows: { key: "cpu" | "cuda"; label: string; v: number | undefined }[] = [];
  if (cpu) rows.push({ key: "cpu", label: "CPU", v: value(cpu) });
  if (cuda) rows.push({ key: "cuda", label: "GPU", v: value(cuda) });
  const nums = rows.map((r) => r.v).filter(isNum) as number[];
  if (nums.length === 0) return <p className="muted-note">No timing recorded.</p>;
  const max = Math.max(...nums, 1e-9);
  const best = betterHigh ? Math.max(...nums) : Math.min(...nums);

  return (
    <div className={styles.bars}>
      {rows.map((r) => {
        const w = isNum(r.v) ? Math.max(4, (r.v / max) * 100) : 4;
        const isBest = isNum(r.v) && r.v === best;
        return (
          <div className={styles.barRow} key={r.key}>
            <span className={styles.barLabel}>{r.label}</span>
            <span className={styles.barTrack}>
              <span
                className={`${styles.barFill} ${isBest ? styles.cuda : styles.cpu}`}
                style={{ width: `${w}%` }}
              />
            </span>
            <span className={styles.barVal}>{format(r.v)}</span>
          </div>
        );
      })}
    </div>
  );
}

function DeviceBenchmarkSection({ db }: { db: DeviceBenchmark }) {
  const hasEmbed = !!(db.embed?.cpu || db.embed?.cuda);
  const hasTrain = !!(db.train_epoch?.cpu || db.train_epoch?.cuda);
  if (!hasEmbed && !hasTrain) return null;

  const embedSpeedup = db.embed_speedup_cuda_over_cpu;
  const trainSpeedup = db.train_epoch_speedup_cuda_over_cpu;
  const trainNote = db.train_epoch?.cuda?.note ?? db.train_epoch?.cpu?.note;
  const ips = (t: DeviceTiming) => t.images_per_second;
  const secs = (t: DeviceTiming) => t.elapsed_seconds;

  return (
    <section className={styles.section} data-testid="pipeline-device-benchmark">
      <div className={styles.head}>
        <span className="eyebrow">
          <CpuIcon width={14} height={14} /> Compute benchmark
        </span>
        <h3>What the GPU actually buys us.</h3>
        <p>
          The same work, timed on CPU and on CUDA{db.batch_size ? <> at batch size {int(db.batch_size)}</> : null}.
          These are measured elapsed times from a real run — not vendor figures.
        </p>
      </div>

      <div className={styles.benchGrid}>
        {hasEmbed && (
          <article className={styles.benchCard}>
            <div className={styles.benchHead}>
              <span className={styles.benchIco} aria-hidden="true">
                <LayersIcon width={18} height={18} />
              </span>
              <h4 className={styles.benchTitle}>Embedding throughput</h4>
            </div>
            <p className={styles.benchUnit}>images encoded per second — higher is better</p>
            {isNum(embedSpeedup) && (
              <span className={styles.speedup}>
                <em>{embedSpeedup.toFixed(2)}×</em> faster on GPU
              </span>
            )}
            <CompareBars
              cpu={db.embed?.cpu}
              cuda={db.embed?.cuda}
              value={ips}
              format={(v) => (isNum(v) ? `${v.toFixed(2)} img/s` : "—")}
              betterHigh
            />
          </article>
        )}

        {hasTrain && (
          <article className={styles.benchCard}>
            <div className={styles.benchHead}>
              <span className={styles.benchIco} aria-hidden="true">
                <GaugeIcon width={18} height={18} />
              </span>
              <h4 className={styles.benchTitle}>Training epoch</h4>
            </div>
            <p className={styles.benchUnit}>wall-clock seconds for one epoch — lower is better</p>
            {isNum(trainSpeedup) && (
              <span className={styles.speedup}>
                <em>{trainSpeedup.toFixed(2)}×</em> faster on GPU
              </span>
            )}
            <CompareBars
              cpu={db.train_epoch?.cpu}
              cuda={db.train_epoch?.cuda}
              value={secs}
              format={(v) => (isNum(v) ? `${v.toFixed(1)} s` : "—")}
              betterHigh={false}
            />
            {trainNote && <p className={styles.benchNote}>{trainNote}</p>}
          </article>
        )}
      </div>
      {db.generated_at && <p className={styles.benchStamp}>measured {db.generated_at}</p>}
    </section>
  );
}

/* ---- Training / hyperparameter sweep --------------------------------------- */
const CONFIG_ROWS: { key: string; label: string }[] = [
  { key: "loss", label: "Loss" },
  { key: "lr", label: "Learning rate" },
  { key: "head_hidden_dim", label: "Head hidden dim" },
  { key: "output_dim", label: "Output dim" },
  { key: "margin", label: "ArcFace margin" },
  { key: "scale", label: "ArcFace scale" },
  { key: "batch_size", label: "Batch size" },
  { key: "max_epochs", label: "Max epochs" },
];

const WINNER_METRIC_ROWS: { key: string; label: string; kind: "pct" | "dec" }[] = [
  { key: "recall@1", label: "Recall@1", kind: "pct" },
  { key: "recall@5", label: "Recall@5", kind: "pct" },
  { key: "recall@10", label: "Recall@10", kind: "pct" },
  { key: "map@5", label: "mAP@5", kind: "pct" },
  { key: "mrr", label: "MRR", kind: "pct" },
  { key: "silhouette", label: "Silhouette", kind: "dec" },
];

function fmtConfig(key: string, v: number | string | undefined): string {
  if (v === undefined || v === null) return "—";
  if (typeof v === "string") return v;
  if (!Number.isFinite(v)) return "—";
  // Learning-rate-like magnitudes read best in exponential form.
  if ((key === "lr" || key === "learning_rate") && v !== 0 && Math.abs(v) < 0.01) {
    return v.toExponential(1);
  }
  return Number.isInteger(v) ? int(v) : dec(v, 3);
}

function TrainingSweep({ training }: { training: TrainingSummary }) {
  const cfg = training.winner_config ?? {};
  const metrics = training.winner_val_metrics ?? {};
  const cfgRows = CONFIG_ROWS.filter((r) => r.key in cfg);
  const metricRows = WINNER_METRIC_ROWS.filter((r) => r.key in metrics);
  const hasWinner = training.winner_run_id || cfgRows.length > 0 || metricRows.length > 0;
  if (!hasWinner) return null;

  return (
    <section className={styles.section} data-testid="pipeline-training">
      <div className={styles.head}>
        <span className="eyebrow">
          <WandIcon width={14} height={14} /> Hyperparameter sweep
        </span>
        <h3>The winning run, and how it was chosen.</h3>
        <p>
          The ArcFace head was swept across configurations; the checkpoint below won on validation
          and is the one served today. Every value is read from the sweep summary.
        </p>
      </div>

      <div className={styles.sweepMeta}>
        {training.winner_run_id && (
          <span className={styles.runChip}>
            <strong>Winner</strong> <code>{training.winner_run_id}</code>
          </span>
        )}
        {isNum(training.num_runs) && (
          <span className={styles.runChip}>
            {int(training.num_runs)} runs compared
          </span>
        )}
        {training.backbone && (
          <span className={styles.runChip}>
            backbone <code>{training.backbone}</code>
          </span>
        )}
      </div>

      {cfgRows.length > 0 && (
        <>
          <p className={styles.groupLabel}>Winning configuration</p>
          <div className={styles.tiles}>
            {cfgRows.map((r) => (
              <div className={styles.tile} key={r.key}>
                <span className={styles.tileK}>{r.label}</span>
                <span className={`${styles.tileV} ${styles.mono}`}>{fmtConfig(r.key, cfg[r.key])}</span>
              </div>
            ))}
          </div>
        </>
      )}

      {metricRows.length > 0 && (
        <>
          <p className={styles.groupLabel}>Validation metrics</p>
          <div className={styles.tiles}>
            {metricRows.map((r) => (
              <div className={`${styles.tile} ${r.key === "recall@5" ? styles.accent : ""}`} key={r.key}>
                <span className={styles.tileK}>{r.label}</span>
                <span className={styles.tileV}>{fmt(metrics[r.key], r.kind)}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
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
          {preview.steps.some((s) => s.image_png_b64) ? (
            // Filmstrip: original + the image AFTER each transform, so the user
            // SEES each step. Steps without an image are skipped defensively.
            <div
              className="filmstrip"
              data-testid="preprocess-filmstrip"
              role="list"
              aria-label="Each preprocessing step applied in sequence"
            >
              <figure className="film-frame" role="listitem">
                <span className="film-n" aria-hidden="true">0</span>
                <div className="film-img">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={`data:image/png;base64,${preview.before_png_b64}`} alt="your uploaded photo, before any transform" />
                </div>
                <figcaption>
                  <strong>Original</strong>
                  <span>Your photo, before any transform.</span>
                </figcaption>
              </figure>
              {preview.steps.map((s, i) =>
                s.image_png_b64 ? (
                  <figure className="film-frame" role="listitem" key={s.name}>
                    <span className="film-n" aria-hidden="true">{i + 1}</span>
                    <div className="film-img">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={`data:image/png;base64,${s.image_png_b64}`}
                        alt={`after step ${i + 1}: ${s.name.replace(/_/g, " ")}`}
                      />
                    </div>
                    <figcaption>
                      <strong>{s.name.replace(/_/g, " ")}</strong>
                      <span>{s.description}</span>
                    </figcaption>
                  </figure>
                ) : null,
              )}
            </div>
          ) : (
            // Fallback (backend without per-step images): before → after + list.
            <>
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
            </>
          )}
        </div>
      )}
    </div>
  );
}

/* ---- Train / validation / test narrative ----------------------------------- */
function TrainingNarrative({ data }: { data: PipelineData }) {
  const { dataset, backbone, eval: ev, calibration } = data;
  const dim = backbone.dim;
  const splits = dataset.splits ?? {};
  const valR5 = ev.val?.["recall@5"];
  const testR5 = ev.test?.["recall@5"];
  const reduction =
    isNum(calibration.ece_before) && isNum(calibration.ece_after) && calibration.ece_before > 0
      ? `${(100 * (1 - calibration.ece_after / calibration.ece_before)).toFixed(0)}%`
      : null;

  return (
    <section className="tvt" aria-labelledby="tvt-h">
      <div className="tvt-head">
        <span className="eyebrow">
          <LayersIcon width={14} height={14} /> Training discipline
        </span>
        <h3 id="tvt-h">Frozen eyes, a small learned lens, one honest test.</h3>
        <p>
          The interesting question isn&rsquo;t just <em>how</em> FloraLens searches — it&rsquo;s
          what was and wasn&rsquo;t trained, and how each data split was used exactly once for its
          job. That discipline is what makes the confidence you see trustworthy.
        </p>
      </div>
      <div className="tvt-cards">
        <article className="tvt-card">
          <span className="tvt-ico" aria-hidden="true">
            <LayersIcon width={18} height={18} />
          </span>
          <h4>The backbone is frozen</h4>
          <p>
            <strong>{backbone.name ?? "DINOv2"}</strong> is never trained — it&rsquo;s the
            &ldquo;eyes.&rdquo; Every image, gallery and query alike, becomes one{" "}
            <strong>{int(dim)}-d embedding</strong>, at index time and again at query time. Search
            is simply <strong>cosine similarity</strong> between those two vectors.
          </p>
        </article>
        <article className="tvt-card">
          <span className="tvt-ico" aria-hidden="true">
            <WandIcon width={18} height={18} />
          </span>
          <h4>What training actually did</h4>
          <p>
            Only a small <strong>ArcFace projection head</strong> was trained on top of the frozen
            embeddings, using the <em>train</em> split{" "}
            {isNum(splits.train) ? `(${int(splits.train)} images) ` : ""}
            alone. The best checkpoint was picked on <em>validation</em> only — early-stopping on
            Recall@5 {isNum(valR5) ? <>({pct(valR5)})</> : null} — then scored{" "}
            <strong>once</strong> on the held-out <em>test</em> split
            {isNum(testR5) ? <> (Recall@5 {pct(testR5)})</> : null}.
          </p>
        </article>
        <article className="tvt-card">
          <span className="tvt-ico" aria-hidden="true">
            <GaugeIcon width={18} height={18} />
          </span>
          <h4>Then calibrated for honesty</h4>
          <p>
            A <strong>calibrator</strong> was fit on validation so the % you see reflects real
            accuracy{reduction ? <> (ECE cut ~{reduction})</> : null}. Candid note: DINOv2 zero-shot
            is already near-ceiling, so the head&rsquo;s real value is a{" "}
            <strong>trustworthy confidence</strong> and a learned metric space — not big retrieval
            gains.
          </p>
        </article>
      </div>
    </section>
  );
}

/* ---- Architecture / train→val→test flow diagram ---------------------------- */
function FlowNode({
  x,
  y,
  w,
  h,
  title,
  sub,
  cls = "",
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  sub?: string;
  cls?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={12} className={`flow-node ${cls}`} />
      <text x={x + w / 2} y={sub ? y + h / 2 - 3 : y + h / 2 + 5} className="flow-node-t" textAnchor="middle">
        {title}
      </text>
      {sub ? (
        <text x={x + w / 2} y={y + h / 2 + 14} className="flow-node-s" textAnchor="middle">
          {sub}
        </text>
      ) : null}
    </g>
  );
}

function FlowDiagram({ data }: { data: PipelineData }) {
  const { dataset, backbone, eval: ev, promotion } = data;
  const splits = dataset.splits ?? {};
  const known = ["train", "val", "test", "gallery"];
  const splitKeys = [
    ...known.filter((k) => k in splits),
    ...Object.keys(splits).filter((k) => !known.includes(k)),
  ].slice(0, 5);
  const valR5 = ev.val?.["recall@5"];
  const testR5 = ev.test?.["recall@5"];
  const decision = (promotion.decision ?? "").trim();

  // Split pills — laid out dynamically within the splits container.
  const areaX = 246;
  const areaW = 288;
  const gap = 8;
  const n = Math.max(splitKeys.length, 1);
  const pw = (areaW - (n - 1) * gap) / n;

  return (
    <div className="flow-scroll">
      <svg
        className="flow-svg"
        viewBox="0 0 780 716"
        role="img"
        aria-labelledby="flow-title flow-desc"
        data-testid="pipeline-flow-diagram"
      >
        <title id="flow-title">FloraLens training and serving architecture</title>
        <desc id="flow-desc">
          The dataset is split leakage-free into train, validation, test, and gallery. Every image
          is preprocessed and embedded by a frozen DINOv2 backbone. The train split fits an ArcFace
          head, validation selects the model and fits the calibrator, the test split is evaluated
          once, and a promotion gate decides whether the model is served for cosine similarity
          search.
        </desc>
        <defs>
          <marker
            id="flow-arrow"
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path d="M1 1 L9 5 L1 9" className="flow-arrow-head" fill="none" />
          </marker>
        </defs>

        {/* lane backgrounds */}
        <rect x={24} y={352} width={232} height={124} rx={14} className="flow-lane" />
        <rect x={274} y={352} width={232} height={124} rx={14} className="flow-lane" />
        <rect x={524} y={352} width={232} height={124} rx={14} className="flow-lane" />
        {/* lane header pills */}
        <g className="flow-lane-tab train">
          <rect x={102} y={360} width={76} height={22} rx={11} />
          <text x={140} y={375} textAnchor="middle">TRAIN</text>
        </g>
        <g className="flow-lane-tab val">
          <rect x={338} y={360} width={104} height={22} rx={11} />
          <text x={390} y={375} textAnchor="middle">VALIDATION</text>
        </g>
        <g className="flow-lane-tab test">
          <rect x={602} y={360} width={76} height={22} rx={11} />
          <text x={640} y={375} textAnchor="middle">TEST</text>
        </g>

        {/* edges */}
        <path d="M390 74 L390 98" className="flow-edge" markerEnd="url(#flow-arrow)" fill="none" />
        <path d="M390 188 L390 208" className="flow-edge" markerEnd="url(#flow-arrow)" fill="none" />
        <path d="M390 256 L390 276" className="flow-edge" markerEnd="url(#flow-arrow)" fill="none" />
        <path d="M390 332 C 390 360 140 366 140 390" className="flow-edge" markerEnd="url(#flow-arrow)" fill="none" />
        <path d="M390 332 L390 390" className="flow-edge" markerEnd="url(#flow-arrow)" fill="none" />
        <path d="M390 332 C 390 360 640 366 640 390" className="flow-edge" markerEnd="url(#flow-arrow)" fill="none" />
        <path d="M236 420 L292 420" className="flow-edge" markerEnd="url(#flow-arrow)" fill="none" />
        <path d="M486 420 L542 420" className="flow-edge" markerEnd="url(#flow-arrow)" fill="none" />
        <path d="M390 448 L390 496" className="flow-edge" markerEnd="url(#flow-arrow)" fill="none" />
        <path d="M390 546 L390 574" className="flow-edge" markerEnd="url(#flow-arrow)" fill="none" />
        <path d="M640 448 C 640 540 560 601 504 601" className="flow-edge dashed" markerEnd="url(#flow-arrow)" fill="none" />
        <path d="M390 626 L390 648" className="flow-edge" markerEnd="url(#flow-arrow)" fill="none" />
        <text x={556} y={534} className="flow-tag">once</text>

        {/* shared spine */}
        <FlowNode x={290} y={26} w={200} h={48} title="Dataset" sub={`${int(dataset.total)} specimens`} />

        {/* splits container */}
        <rect x={235} y={100} width={310} height={88} rx={14} className="flow-node" />
        <text x={390} y={124} className="flow-node-t" textAnchor="middle">Leakage-free splits</text>
        {splitKeys.map((k, i) => {
          const px = areaX + i * (pw + gap);
          return (
            <g key={k} className={`flow-pill ${known.includes(k) ? k : ""}`}>
              <rect x={px} y={146} width={pw} height={28} rx={14} />
              <text x={px + pw / 2} y={158} textAnchor="middle" className="flow-pill-k">{k}</text>
              <text x={px + pw / 2} y={169} textAnchor="middle" className="flow-pill-v">{int(splits[k])}</text>
            </g>
          );
        })}

        <FlowNode x={280} y={210} w={220} h={46} title="Preprocessing" sub="EXIF · resize · crop · WB · CLAHE" />

        {/* frozen embed with badge */}
        <FlowNode
          x={270}
          y={278}
          w={240}
          h={54}
          title="Frozen DINOv2"
          sub={isNum(backbone.dim) ? `${int(backbone.dim)}-d embedding` : "vector embedding"}
          cls="frozen"
        />
        <g className="flow-badge">
          <rect x={442} y={285} width={58} height={18} rx={9} />
          <text x={471} y={297} textAnchor="middle">frozen</text>
        </g>

        {/* lane nodes */}
        <FlowNode x={44} y={392} w={192} h={56} title="ArcFace head" sub="trained · train split" />
        <FlowNode x={294} y={392} w={192} h={56} title="Model selection" sub={isNum(valR5) ? `early-stop · R@5 ${pct(valR5)}` : "early-stop · Recall@5"} />
        <FlowNode x={544} y={392} w={192} h={56} title="One-shot eval" sub={isNum(testR5) ? `held-out · R@5 ${pct(testR5)}` : "held-out · one pass"} />

        <FlowNode x={280} y={498} w={220} h={48} title="Calibration" sub="fit on val · ECE ↓" />
        <FlowNode x={280} y={576} w={220} h={50} title="Promotion gate" sub={decision ? decision.toLowerCase() : "candidate → active"} />
        <FlowNode x={260} y={650} w={260} h={52} title="Serve" sub="query vs gallery · cosine" cls="serve" />
      </svg>
      <p className="flow-legend">
        Each split is used for exactly one job. The <em>test</em> set is touched a single time
        (dashed path); the gallery is embedded once and indexed, then every query is embedded the
        same way and ranked by cosine similarity.
      </p>
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

      <ModelCard data={data} />

      <TrainingNarrative data={data} />

      <section className="flow-section" aria-labelledby="flow-h">
        <div className="flow-section-head">
          <span className="eyebrow">
            <GridIcon width={14} height={14} /> Architecture at a glance
          </span>
          <h3 id="flow-h">How data flows from split to serve.</h3>
          <p>
            One image path, three disciplined data lanes. Follow it top to bottom: the split you use
            decides the job it does.
          </p>
        </div>
        <FlowDiagram data={data} />
      </section>

      {data.training && <TrainingSweep training={data.training} />}

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

      {data.device_benchmark && <DeviceBenchmarkSection db={data.device_benchmark} />}
    </div>
  );
}
