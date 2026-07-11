"use client";

import { motion, useMotionTemplate, useMotionValue, useReducedMotion } from "motion/react";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import {
  BloomIcon,
  CameraLeafIcon,
  ChartIcon,
  ChevronIcon,
  GaugeIcon,
  GithubIcon,
  GridIcon,
  LayersIcon,
  SearchIcon,
  SparkleIcon,
} from "./icons";
import { ThemeToggle } from "./theme-toggle";
import styles from "./landing.module.css";

// Shared editorial easing (expo-out) - one motion signature across the page.
const EASE = [0.16, 1, 0.3, 1] as const;

// Real gallery specimens, served through the same /api proxy the app uses.
// Each frame carries a botanical gradient fallback, so a missing backend
// degrades to a tinted panel rather than a broken image.
const HERO_IDS = ["flowers102-te05815", "flowers102-te00489", "flowers102-te05322", "flowers102-te05617"];
const STRIP_IDS = [
  "flowers102-te05815",
  "flowers102-te00489",
  "flowers102-te05322",
  "flowers102-te05617",
  "flowers102-te01172",
];

const FLOW = [
  { icon: CameraLeafIcon, verb: "Upload", text: "Drop, browse, or paste a link to any flower photo." },
  { icon: LayersIcon, verb: "Embed", text: "A frozen vision backbone turns the image into a dense embedding." },
  { icon: SearchIcon, verb: "Rank", text: "The gallery is scored by visual similarity and ordered best first." },
  { icon: GaugeIcon, verb: "Read", text: "Every match lands in a calibrated High, Medium, or Low band." },
] as const;

function specimenSrc(id: string) {
  return `/api/specimen/${encodeURIComponent(id)}/image`;
}

function hideOnError(e: React.SyntheticEvent<HTMLImageElement>) {
  // Leave the frame's gradient fallback visible instead of a broken glyph.
  e.currentTarget.style.visibility = "hidden";
}

/** Capability cell with an optional cursor-follow spotlight (motion values, no
 *  state, no re-render). Falls back to a plain card under reduced motion. */
function Cell({
  className,
  icon: Icon,
  title,
  children,
  reduce,
  extra,
}: {
  className?: string;
  icon: (p: { width: number; height: number }) => ReactNode;
  title: string;
  children: ReactNode;
  reduce: boolean;
  extra?: ReactNode;
}) {
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const spotlight = useMotionTemplate`radial-gradient(240px circle at ${mx}px ${my}px, color-mix(in srgb, var(--primary) 14%, transparent), transparent 72%)`;
  return (
    <motion.article
      className={`${styles.cell} ${className ?? ""}`}
      variants={reduce ? undefined : itemVariants}
      whileHover={reduce ? undefined : { y: -4 }}
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
      {!reduce && <motion.span className={styles.spotlight} style={{ background: spotlight }} aria-hidden="true" />}
      <span className={styles.cellIco} aria-hidden="true">
        <Icon width={22} height={22} />
      </span>
      <h3 className={styles.cellTitle}>{title}</h3>
      <p className={styles.cellText}>{children}</p>
      {extra}
    </motion.article>
  );
}

const containerVariants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.07 } },
};
const itemVariants = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: EASE } },
};

export default function LandingPage() {
  const reduce = useReducedMotion() ?? false;

  // Gallery filmstrip: native horizontal scroll (scrollbar hidden) driven by
  // prev/next controls + edge fades. Edge state disables controls and hides the
  // matching fade at each end, so the strip always reads as intentional.
  const stripRef = useRef<HTMLDivElement>(null);
  const [atStart, setAtStart] = useState(true);
  const [atEnd, setAtEnd] = useState(false);

  const syncStripEdges = useCallback(() => {
    const el = stripRef.current;
    if (!el) return;
    const { scrollLeft, scrollWidth, clientWidth } = el;
    setAtStart(scrollLeft <= 2);
    // A 2px slack absorbs sub-pixel rounding so the end state latches cleanly.
    setAtEnd(scrollLeft + clientWidth >= scrollWidth - 2);
  }, []);

  useEffect(() => {
    syncStripEdges();
    window.addEventListener("resize", syncStripEdges);
    return () => window.removeEventListener("resize", syncStripEdges);
  }, [syncStripEdges]);

  const scrollStrip = useCallback(
    (dir: 1 | -1) => {
      const el = stripRef.current;
      if (!el) return;
      el.scrollBy({ left: dir * el.clientWidth * 0.8, behavior: reduce ? "auto" : "smooth" });
    },
    [reduce],
  );

  const reveal = reduce
    ? {}
    : {
        initial: { opacity: 0, y: 24 },
        whileInView: { opacity: 1, y: 0 },
        viewport: { once: true, amount: 0.3 },
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
    <div className={styles.page}>
      {/* Safety net: motion reveals render with opacity:0 for hydration. If
          scripting is unavailable, restore full visibility so no section can be
          left permanently hidden. */}
      <noscript>
        <style>{`[style*="opacity"]{opacity:1!important;transform:none!important}`}</style>
      </noscript>
      {/* -------- Sticky header -------- */}
      <header className={styles.nav}>
        <a className={styles.brand} href="#top" aria-label="FloraLens home">
          <span className={styles.brandMark} aria-hidden="true">
            <BloomIcon width={21} height={21} />
          </span>
          <span className={styles.wordmark}>FloraLens</span>
        </a>
        <span className={styles.navSpacer} />
        <nav className={styles.navLinks} aria-label="Page sections">
          <a className={styles.navLink} href="#how">How it works</a>
          <a className={styles.navLink} href="#capabilities">Capabilities</a>
          <a className={styles.navLink} href="#gallery">Gallery</a>
        </nav>
        <div className={styles.navActions}>
          <ThemeToggle />
          <a className={`${styles.btn} ${styles.btnPrimary} ${styles.navBtn}`} href="/app">
            Launch FloraLens
          </a>
        </div>
      </header>

      <main id="top">
        {/* -------- 1. Hero (asymmetric split) -------- */}
        <section className={styles.hero}>
          <motion.div
            {...(reduce
              ? {}
              : {
                  initial: { opacity: 0, y: 20 },
                  animate: { opacity: 1, y: 0 },
                  transition: { duration: 0.6, ease: EASE },
                })}
          >
            <span className={styles.eyebrow}>
              <BloomIcon width={14} height={14} /> Visual plant identification
            </span>
            <h1 className={styles.heroTitle}>
              Find the <em>flower</em> in any photo.
            </h1>
            <p className={styles.heroLead}>
              Upload a photo and FloraLens ranks visually similar specimens from its gallery, each
              scored with a calibrated confidence band.
            </p>
            <div className={styles.heroCtas}>
              <a className={`${styles.btn} ${styles.btnPrimary} ${styles.btnLg}`} href="/app">
                <SearchIcon width={18} height={18} aria-hidden="true" /> Launch FloraLens
              </a>
              <a className={`${styles.btn} ${styles.btnGhost} ${styles.btnLg}`} href="#how">
                See how it works
              </a>
            </div>
          </motion.div>

          <motion.div
            className={styles.heroCluster}
            aria-hidden="true"
            {...(reduce
              ? {}
              : {
                  initial: { opacity: 0, scale: 0.96 },
                  animate: { opacity: 1, scale: 1 },
                  transition: { duration: 0.7, ease: EASE, delay: 0.1 },
                })}
          >
            {HERO_IDS.map((id) => (
              <div className={styles.frame} key={id}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={specimenSrc(id)} alt="" loading="eager" onError={hideOnError} />
              </div>
            ))}
          </motion.div>
        </section>

        {/* -------- 2. How it works (horizontal verb-led flow) -------- */}
        <motion.section className={styles.section} id="how" {...reveal}>
          <div className={styles.sectionHead}>
            <h2 className={styles.sectionTitle}>From a single photo to a ranked, scored match.</h2>
            <p className={styles.sectionLead}>
              No tags, no filenames. FloraLens compares how flowers actually look and shows you how
              much to trust each result.
            </p>
          </div>
          <motion.div className={styles.flow} {...gridReveal}>
            {FLOW.map((s) => (
              <motion.div className={styles.flowStep} key={s.verb} variants={reduce ? undefined : itemVariants}>
                <span className={styles.flowNode} aria-hidden="true">
                  <s.icon width={22} height={22} />
                </span>
                <h3 className={styles.flowVerb}>{s.verb}</h3>
                <p className={styles.flowText}>{s.text}</p>
              </motion.div>
            ))}
          </motion.div>
        </motion.section>

        {/* -------- 3. Capabilities (bento) -------- */}
        <motion.section className={styles.section} id="capabilities" {...reveal}>
          <div className={styles.sectionHead}>
            <h2 className={styles.sectionTitle}>Built so every match can be trusted.</h2>
            <p className={styles.sectionLead}>
              Six choices keep the results honest, reproducible, and easy to reason about.
            </p>
          </div>
          <motion.div className={styles.bento} {...gridReveal}>
            {/* Lead cell: calibrated confidence, with a real mini-band visual. */}
            <Cell
              className={styles.cellLead}
              icon={GaugeIcon}
              title="Calibrated confidence"
              reduce={reduce}
              extra={
                <div className={styles.bands} aria-hidden="true">
                  <div className={styles.bandRow}>
                    <span className={`${styles.bandName} ${styles.high}`}>High</span>
                    <span className={styles.bandTrack}><span className={`${styles.bandFill} ${styles.high}`} style={{ width: "86%" }} /></span>
                  </div>
                  <div className={styles.bandRow}>
                    <span className={`${styles.bandName} ${styles.med}`}>Medium</span>
                    <span className={styles.bandTrack}><span className={`${styles.bandFill} ${styles.med}`} style={{ width: "58%" }} /></span>
                  </div>
                  <div className={styles.bandRow}>
                    <span className={`${styles.bandName} ${styles.low}`}>Low</span>
                    <span className={styles.bandTrack}><span className={`${styles.bandFill} ${styles.low}`} style={{ width: "31%" }} /></span>
                  </div>
                </div>
              }
            >
              Raw cosine scores become honest High, Medium, and Low bands, so a percentage means what
              it says.
            </Cell>

            {/* Photo cell: real gallery specimen with a scrimmed caption. */}
            <motion.article
              className={`${styles.cell} ${styles.cellPhoto}`}
              variants={reduce ? undefined : itemVariants}
              aria-label="A specimen from the FloraLens gallery"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={specimenSrc("flowers102-te01172")} alt="" loading="lazy" onError={hideOnError} />
              <span className={styles.cellPhotoCap}>A curated gallery</span>
            </motion.article>

            <Cell icon={LayersIcon} title="DINOv2 embeddings" reduce={reduce}>
              A frozen vision backbone projects each image into a rich embedding built on visual
              structure.
            </Cell>

            <Cell icon={ChartIcon} title="Leakage-free evaluation" reduce={reduce}>
              Train, validation, and test splits stay separate, so reported scores reflect held-out
              performance.
            </Cell>

            <Cell icon={GridIcon} title="Transparent pipeline" reduce={reduce}>
              Every stage, from preprocessing to the promotion decision, is visible with live metrics
              inside the app.
            </Cell>

            <Cell className={styles.cellWide} icon={SparkleIcon} title="Explore the embedding space" reduce={reduce}>
              Fly through the whole gallery as a 3D point cloud, or ask the naturalist assistant about
              a species beside your matches.
            </Cell>
          </motion.div>
        </motion.section>

        {/* -------- 4. Gallery strip (filmstrip) -------- */}
        <motion.section className={`${styles.section} ${styles.stripSection}`} id="gallery" {...reveal}>
          <div className={styles.sectionHead}>
            <h2 className={styles.sectionTitle}>A gallery of real specimens.</h2>
            <p className={styles.sectionLead}>
              Every search ranks against curated photographs like these.
            </p>
          </div>
          <div
            className={`${styles.stripWrap} ${atStart ? styles.atStart : ""} ${
              atEnd ? styles.atEnd : ""
            }`}
          >
            <button
              type="button"
              className={`${styles.stripNav} ${styles.stripPrev}`}
              aria-label="Show previous specimens"
              onClick={() => scrollStrip(-1)}
              disabled={atStart}
            >
              <ChevronIcon width={20} height={20} className={styles.chevLeft} />
            </button>
            <div
              className={styles.strip}
              ref={stripRef}
              onScroll={syncStripEdges}
              tabIndex={0}
              role="group"
              aria-label="Gallery specimens, scroll horizontally with arrow keys"
              onKeyDown={(e) => {
                if (e.key === "ArrowRight") {
                  e.preventDefault();
                  scrollStrip(1);
                } else if (e.key === "ArrowLeft") {
                  e.preventDefault();
                  scrollStrip(-1);
                }
              }}
            >
              {STRIP_IDS.map((id) => (
                <div className={styles.stripItem} key={id}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={specimenSrc(id)} alt="" loading="lazy" onError={hideOnError} />
                </div>
              ))}
            </div>
            <button
              type="button"
              className={`${styles.stripNav} ${styles.stripNext}`}
              aria-label="Show next specimens"
              onClick={() => scrollStrip(1)}
              disabled={atEnd}
            >
              <ChevronIcon width={20} height={20} className={styles.chevRight} />
            </button>
          </div>
        </motion.section>

        {/* -------- 5. Honest numbers (editorial stats) -------- */}
        <section className={styles.numbers}>
          <motion.div className={styles.numbersInner} {...reveal}>
            <div>
              <p className={styles.numbersLead}>A confidence score only helps if it is earned on held-out data.</p>
              <p className={styles.numbersNote}>Running as a demo on the Oxford-102 flowers dataset.</p>
            </div>
            <div className={styles.statRow}>
              <div className={styles.stat}>
                <span className={styles.statValue}>1024-d</span>
                <span className={styles.statLabel}>Embedding dimensions per specimen</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statValue}>102</span>
                <span className={styles.statLabel}>Flower species in the gallery</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statValue}>3</span>
                <span className={styles.statLabel}>Calibrated confidence bands</span>
              </div>
            </div>
          </motion.div>
        </section>

        {/* -------- 6. Closing CTA band -------- */}
        <section className={styles.cta}>
          <motion.div className={styles.ctaInner} {...reveal}>
            <h2 className={styles.ctaTitle}>
              Identify your <em>first</em> flower.
            </h2>
            <p className={styles.ctaText}>Open the app, add a photo, and read the matches in seconds.</p>
            <a className={`${styles.btn} ${styles.btnPrimary} ${styles.btnLg}`} href="/app">
              <SearchIcon width={18} height={18} aria-hidden="true" /> Launch FloraLens
            </a>
          </motion.div>
        </section>
      </main>

      {/* -------- 7. Footer -------- */}
      <footer className={styles.footer}>
        <div className={styles.footerInner}>
          <div className={styles.footerBrand}>
            <a className={styles.brand} href="#top" aria-label="FloraLens home">
              <span className={styles.brandMark} aria-hidden="true">
                <BloomIcon width={21} height={21} />
              </span>
              <span className={styles.wordmark}>FloraLens</span>
            </a>
            <p className={styles.footerTagline}>
              Visual flower discovery and similarity search, with a confidence band on every match.
            </p>
            <p className={styles.footerNote}>Demo on the Oxford-102 flowers dataset.</p>
          </div>
          <div className={styles.footerLinks}>
            <p className={styles.footerLinksTitle}>Explore</p>
            <a className={styles.footerLink} href="/app">Launch FloraLens</a>
            <a className={styles.footerLink} href="#how">How it works</a>
            <a className={styles.footerLink} href="#capabilities">Capabilities</a>
            <a className={styles.footerLink} href="#gallery">Gallery</a>
          </div>
          <div className={styles.footerLinks}>
            <p className={styles.footerLinksTitle}>Project</p>
            <a
              className={styles.footerLink}
              href="https://github.com/pdz1804/floralens"
              target="_blank"
              rel="noreferrer noopener"
            >
              <GithubIcon width={15} height={15} aria-hidden="true" /> GitHub
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
