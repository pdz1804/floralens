"use client";

import {
  BloomIcon,
  ChartIcon,
  GaugeIcon,
  GridIcon,
  LayersIcon,
  LeafIcon,
  SearchIcon,
  SparkleIcon,
  WandIcon,
} from "./icons";
import styles from "./about.module.css";

const STATS = [
  { value: "1024-d", label: "Embedding dimensions per specimen" },
  { value: "3 bands", label: "Calibrated High / Medium / Low confidence" },
  { value: "0 leakage", label: "Strict train / val / test separation" },
  { value: "End-to-end", label: "Every stage exposed on the Pipeline tab" },
] as const;

const FEATURES = [
  {
    icon: SearchIcon,
    title: "Visual similarity search",
    body: "Upload or link any flower photo and FloraLens finds the closest specimens in the gallery by how they actually look — not by tags or filenames.",
  },
  {
    icon: LayersIcon,
    title: "DINOv3-style embeddings",
    body: "Every image is projected into a high-dimensional embedding by a frozen vision backbone, so matches are ranked on rich visual structure.",
  },
  {
    icon: GaugeIcon,
    title: "Calibrated confidence",
    body: "Raw cosine scores are calibrated into honest High / Medium / Low bands, so a percentage means what it says.",
  },
  {
    icon: ChartIcon,
    title: "Leakage-free evaluation",
    body: "Train, validation and test splits stay strictly separate. Reported Recall, mAP and MRR reflect held-out performance, not memorized data.",
  },
  {
    icon: WandIcon,
    title: "Deterministic preprocessing",
    body: "Auto-orient, white balance and CLAHE run identically on gallery and query images, so lighting and framing don't skew the results.",
  },
  {
    icon: GridIcon,
    title: "Transparent pipeline",
    body: "The Pipeline tab shows every stage — dataset, model, index, calibration and the promotion decision — with live metrics.",
  },
] as const;

const STEPS = [
  { n: 1, title: "Add a flower", body: "Drag & drop, browse, or paste an image URL on the Search tab." },
  { n: 2, title: "Search", body: "FloraLens embeds your photo and ranks the gallery by visual similarity." },
  { n: 3, title: "Read the matches", body: "Each card shows a confidence band, a score, and a botanical description." },
  { n: 4, title: "Filter by confidence", body: "Narrow results to High, Medium, or Low bands to focus on the strongest matches." },
] as const;

export function AboutPage() {
  return (
    <div className={styles.page} data-testid="about-page">
      <section className={styles.hero}>
        <span className="eyebrow">
          <BloomIcon width={14} height={14} /> About FloraLens
        </span>
        <h2>A calibrated, visual way to identify flowers.</h2>
        <p className={styles.lead}>
          FloraLens is a visual flower similarity search engine. Give it a photo and it surfaces the
          most visually similar specimens from a curated gallery, each scored with a confidence band
          you can actually trust — transparent end to end, from preprocessing to the model, the
          index, and the evaluation behind every number.
        </p>
      </section>

      <dl className={styles.stats} aria-label="FloraLens at a glance">
        {STATS.map((s) => (
          <div className={styles.stat} key={s.label}>
            <dt className={styles.statValue}>{s.value}</dt>
            <dd className={styles.statLabel}>{s.label}</dd>
          </div>
        ))}
      </dl>

      <section className={styles.section} aria-label="What FloraLens does">
        <header className={styles.sectionHead}>
          <span className={styles.kicker}>
            <SparkleIcon width={13} height={13} /> Capabilities
          </span>
          <h3 className={styles.sectionTitle}>Built for trustworthy visual matching.</h3>
          <p className={styles.sectionLead}>
            Six design choices keep results honest, reproducible, and easy to reason about.
          </p>
        </header>
        <div className={styles.features}>
          {FEATURES.map((f, i) => (
            <article className={styles.feature} key={f.title}>
              <div className={styles.featureTop}>
                <span className={styles.featureIco} aria-hidden="true">
                  <f.icon width={22} height={22} />
                </span>
                <span className={styles.featureNum} aria-hidden="true">
                  {String(i + 1).padStart(2, "0")}
                </span>
              </div>
              <h3>{f.title}</h3>
              <p>{f.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className={styles.section} aria-label="How to use FloraLens">
        <header className={styles.sectionHead}>
          <span className={styles.kicker}>
            <LeafIcon width={13} height={13} /> Getting started
          </span>
          <h3 className={styles.sectionTitle}>Four steps from photo to match.</h3>
        </header>
        <ol className={styles.steps}>
          {STEPS.map((s) => (
            <li className={styles.step} key={s.n}>
              <span className={styles.stepNum} aria-hidden="true">
                {s.n}
              </span>
              <h4>{s.title}</h4>
              <p>{s.body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className={styles.closing} aria-label="Why it matters">
        <span className={styles.closingIco} aria-hidden="true">
          <GaugeIcon width={24} height={24} />
        </span>
        <div>
          <h3>Honest numbers, by design.</h3>
          <p>
            A confidence percentage only helps if it is calibrated and earned on held-out data. Open
            the Pipeline tab to trace every match back through the dataset, the frozen backbone, the
            index and the calibration that produced it.
          </p>
        </div>
      </section>
    </div>
  );
}
