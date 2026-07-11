"use client";

import {
  motion,
  useMotionTemplate,
  useMotionValue,
  useReducedMotion,
  type Variants,
} from "motion/react";
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

// Shared editorial easing (expo-out) — used across every reveal so the whole
// page shares one motion signature.
const EASE = [0.16, 1, 0.3, 1] as const;

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

// Stagger container + item variants for grids/lists that reveal on scroll.
const containerVariants: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06 } },
};
const itemVariants: Variants = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: EASE } },
};

type FeatureDef = (typeof FEATURES)[number];

/** Feature card with an optional cursor-follow spotlight driven by motion
 * values (no state → no re-render on mouse move). Degrades to a plain,
 * static card when the user prefers reduced motion. */
function FeatureCard({ f, index, reduce }: { f: FeatureDef; index: number; reduce: boolean }) {
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const spotlight = useMotionTemplate`radial-gradient(220px circle at ${mx}px ${my}px, color-mix(in srgb, var(--primary) 14%, transparent), transparent 72%)`;

  return (
    <motion.article
      className={styles.feature}
      key={f.title}
      variants={reduce ? undefined : itemVariants}
      whileHover={reduce ? undefined : { y: -4 }}
      whileTap={reduce ? undefined : { scale: 0.98 }}
      transition={{ type: "spring", stiffness: 320, damping: 26 }}
      onMouseMove={
        reduce
          ? undefined
          : (e) => {
              const r = e.currentTarget.getBoundingClientRect();
              mx.set(e.clientX - r.left);
              my.set(e.clientY - r.top);
            }
      }
    >
      {!reduce && (
        <motion.span className={styles.spotlight} style={{ background: spotlight }} aria-hidden="true" />
      )}
      <div className={styles.featureTop}>
        <span className={styles.featureIco} aria-hidden="true">
          <f.icon width={22} height={22} />
        </span>
        <span className={styles.featureNum} aria-hidden="true">
          {String(index + 1).padStart(2, "0")}
        </span>
      </div>
      <h3>{f.title}</h3>
      <p>{f.body}</p>
    </motion.article>
  );
}

export function AboutPage() {
  const reduce = useReducedMotion() ?? false;

  // Reveal-on-scroll props, gated on the reduced-motion preference.
  const reveal = reduce
    ? {}
    : {
        initial: { opacity: 0, y: 24 },
        whileInView: { opacity: 1, y: 0 },
        viewport: { once: true, amount: 0.25 },
        transition: { duration: 0.6, ease: EASE },
      };
  const gridReveal = reduce
    ? {}
    : {
        variants: containerVariants,
        initial: "hidden" as const,
        whileInView: "show" as const,
        viewport: { once: true, amount: 0.2 },
      };

  return (
    <div className={styles.page} data-testid="about-page">
      <motion.section
        className={styles.hero}
        {...(reduce
          ? {}
          : {
              initial: { opacity: 0, y: 16 },
              animate: { opacity: 1, y: 0 },
              transition: { duration: 0.6, ease: EASE },
            })}
      >
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
      </motion.section>

      <motion.dl className={styles.stats} aria-label="FloraLens at a glance" {...gridReveal}>
        {STATS.map((s) => (
          <motion.div
            className={styles.stat}
            key={s.label}
            variants={reduce ? undefined : itemVariants}
          >
            <dt className={styles.statValue}>{s.value}</dt>
            <dd className={styles.statLabel}>{s.label}</dd>
          </motion.div>
        ))}
      </motion.dl>

      <motion.section className={styles.section} aria-label="What FloraLens does" {...reveal}>
        <header className={styles.sectionHead}>
          <span className={styles.kicker}>
            <SparkleIcon width={13} height={13} /> Capabilities
          </span>
          <h3 className={styles.sectionTitle}>Built for trustworthy visual matching.</h3>
          <p className={styles.sectionLead}>
            Six design choices keep results honest, reproducible, and easy to reason about.
          </p>
        </header>
        <motion.div className={styles.features} {...gridReveal}>
          {FEATURES.map((f, i) => (
            <FeatureCard f={f} index={i} reduce={reduce} key={f.title} />
          ))}
        </motion.div>
      </motion.section>

      <motion.section className={styles.section} aria-label="How to use FloraLens" {...reveal}>
        <header className={styles.sectionHead}>
          <span className={styles.kicker}>
            <LeafIcon width={13} height={13} /> Getting started
          </span>
          <h3 className={styles.sectionTitle}>Four steps from photo to match.</h3>
        </header>
        <motion.ol className={styles.steps} {...gridReveal}>
          {STEPS.map((s) => (
            <motion.li
              className={styles.step}
              key={s.n}
              variants={reduce ? undefined : itemVariants}
            >
              <span className={styles.stepNum} aria-hidden="true">
                {s.n}
              </span>
              <h4>{s.title}</h4>
              <p>{s.body}</p>
            </motion.li>
          ))}
        </motion.ol>
      </motion.section>

      <motion.section className={styles.closing} aria-label="Why it matters" {...reveal}>
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
      </motion.section>
    </div>
  );
}
