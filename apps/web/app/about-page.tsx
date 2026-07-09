"use client";

import {
  BloomIcon,
  ChartIcon,
  GaugeIcon,
  GridIcon,
  LayersIcon,
  SearchIcon,
  WandIcon,
} from "./icons";

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
    <div className="about" data-testid="about-page">
      <section className="about-hero">
        <span className="eyebrow">
          <BloomIcon width={14} height={14} /> About FloraLens
        </span>
        <h2>A calibrated, visual way to identify flowers.</h2>
        <p>
          FloraLens is a visual flower similarity search engine. Give it a photo and it surfaces
          the most visually similar specimens from a curated gallery, each scored with a confidence
          band you can actually trust. It is built to be transparent end to end — from preprocessing
          to the model, the index, and the evaluation behind every number.
        </p>
      </section>

      <section className="about-grid" aria-label="What FloraLens does">
        {FEATURES.map((f) => (
          <article className="feature" key={f.title}>
            <span className="feature-ico" aria-hidden="true">
              <f.icon width={20} height={20} />
            </span>
            <h3>{f.title}</h3>
            <p>{f.body}</p>
          </article>
        ))}
      </section>

      <section className="about-how" aria-label="How to use FloraLens">
        <h3 className="about-sub">How to use it</h3>
        <ol className="steps">
          {STEPS.map((s) => (
            <li className="step" key={s.n}>
              <span className="step-n" aria-hidden="true">{s.n}</span>
              <div>
                <h4>{s.title}</h4>
                <p>{s.body}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
