"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  addToGarden,
  getCategories,
  getGalaxy,
  type Category,
  type GalaxyPoint,
} from "@/lib/api";
import {
  AlertIcon,
  CheckIcon,
  ChevronIcon,
  GridIcon,
  LayersIcon,
  LeafIcon,
  SearchIcon,
  SparkleIcon,
  TagIcon,
} from "./icons";
import styles from "./categories.module.css";

type Status = "loading" | "ready" | "error";
type SortKey = "name" | "count";
type GalaxyStatus = "idle" | "loading" | "ready" | "error";

const numberFmt = new Intl.NumberFormat("en-US");

export function CategoriesPage() {
  const [categories, setCategories] = useState<Category[]>([]);
  const [status, setStatus] = useState<Status>("loading");
  const [error, setError] = useState<string | null>(null);
  const [totalSpecimens, setTotalSpecimens] = useState(0);

  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SortKey>("name");

  // Expanded species (modal) + its gallery members, sourced lazily from the
  // Galaxy point cloud and cached across opens.
  const [selected, setSelected] = useState<Category | null>(null);
  const [galaxyPoints, setGalaxyPoints] = useState<GalaxyPoint[] | null>(null);
  const [galaxyStatus, setGalaxyStatus] = useState<GalaxyStatus>("idle");

  // Save-to-Garden state (kept at page level so a "saved" confirmation
  // survives closing and re-opening a species).
  const [savedIds, setSavedIds] = useState<Set<string>>(new Set());
  const [savingId, setSavingId] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const closeRef = useRef<HTMLButtonElement>(null);
  const openerRef = useRef<HTMLButtonElement | null>(null);

  // ---- Data load ----------------------------------------------------------
  useEffect(() => {
    const ctrl = new AbortController();
    getCategories(ctrl.signal)
      .then((d) => {
        setCategories(d.categories);
        setTotalSpecimens(d.total_specimens);
        setStatus("ready");
      })
      .catch((e) => {
        if ((e as Error).name === "AbortError") return;
        setError((e as Error).message);
        setStatus("error");
      });
    return () => ctrl.abort();
  }, []);

  // Fetch the Galaxy point cloud once, on first expand — it supplies each
  // species' full set of gallery specimens (filtered by label_name).
  const ensureGalaxy = useCallback(() => {
    setGalaxyStatus((s) => {
      if (s !== "idle") return s;
      getGalaxy()
        .then((d) => {
          setGalaxyPoints(d.points);
          setGalaxyStatus("ready");
        })
        .catch(() => setGalaxyStatus("error"));
      return "loading";
    });
  }, []);

  // ---- Filter + sort ------------------------------------------------------
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = q
      ? categories.filter((c) => c.label_name.toLowerCase().includes(q))
      : categories;
    return [...filtered].sort((a, b) =>
      sort === "name"
        ? a.label_name.localeCompare(b.label_name)
        : b.total_count - a.total_count || a.label_name.localeCompare(b.label_name),
    );
  }, [categories, query, sort]);

  const members = useMemo(() => {
    if (!selected || !galaxyPoints) return [];
    return galaxyPoints.filter((p) => p.label_name === selected.label_name);
  }, [selected, galaxyPoints]);

  // ---- Expand / collapse --------------------------------------------------
  function openSpecies(cat: Category, e: React.MouseEvent<HTMLButtonElement>) {
    openerRef.current = e.currentTarget;
    setSelected(cat);
    setSaveError(null);
    ensureGalaxy();
  }
  const closeSpecies = useCallback(() => {
    setSelected(null);
    setSaveError(null);
    // Return focus to the card that opened the dialog.
    openerRef.current?.focus();
  }, []);

  // Focus the close button + wire Escape while the dialog is open.
  useEffect(() => {
    if (!selected) return;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeSpecies();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected, closeSpecies]);

  // ---- Save to Garden -----------------------------------------------------
  async function onSave(id: string) {
    if (savingId) return;
    setSavingId(id);
    setSaveError(null);
    try {
      await addToGarden(id);
      setSavedIds((cur) => new Set(cur).add(id));
    } catch (e) {
      setSaveError((e as Error).message);
    } finally {
      setSavingId(null);
    }
  }

  const total = categories.length;

  return (
    <div className={styles.page} data-testid="categories-page">
      <section className={styles.hero}>
        <span className="eyebrow">
          <TagIcon width={14} height={14} /> Species catalogue
        </span>
        <h2>Every flower in the gallery, at a glance.</h2>
        <p>
          Browse the {total || 102} species FloraLens can recognise. Search by name, sort
          by how well-represented each species is, and open any card to leaf through its
          gallery specimens.
        </p>
      </section>

      {/* Header stat strip */}
      <section className={styles.stats} aria-label="Catalogue totals">
        <div className={styles.stat}>
          <span className={styles.statValue}>{status === "ready" ? total : "—"}</span>
          <span className={styles.statLabel}>Species</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statValue}>
            {status === "ready" ? numberFmt.format(totalSpecimens) : "—"}
          </span>
          <span className={styles.statLabel}>Specimens</span>
        </div>
      </section>

      {status === "loading" && (
        <div className="state" aria-busy="true">
          <span className={styles.spin} aria-hidden="true" />
          <span className="state-title">Loading the catalogue…</span>
        </div>
      )}

      {status === "error" && (
        <div className="state error">
          <span className="state-ico" aria-hidden="true">
            <AlertIcon width={26} height={26} />
          </span>
          <span className="state-title">Catalogue unavailable</span>
          <p>{error ?? "Could not load the species list. Is the backend running?"}</p>
        </div>
      )}

      {status === "ready" && (
        <>
          {/* Toolbar: search + sort */}
          <div className={styles.toolbar}>
            <div className={styles.searchField}>
              <span className={styles.searchIcon} aria-hidden="true">
                <SearchIcon width={16} height={16} />
              </span>
              <input
                type="text"
                className={styles.searchInput}
                placeholder="Search species by name…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                aria-label="Search species by name"
                data-testid="category-search"
              />
              {query && (
                <button
                  type="button"
                  className={styles.searchClear}
                  onClick={() => setQuery("")}
                  aria-label="Clear search"
                >
                  ×
                </button>
              )}
            </div>

            <div className={styles.sortGroup}>
              <span className={styles.sortLabel} id="cat-sort-label">
                Sort
              </span>
              <div
                className={styles.sortToggle}
                role="group"
                aria-labelledby="cat-sort-label"
              >
                <button
                  type="button"
                  aria-pressed={sort === "name"}
                  onClick={() => setSort("name")}
                >
                  Name A–Z
                </button>
                <button
                  type="button"
                  aria-pressed={sort === "count"}
                  onClick={() => setSort("count")}
                >
                  Most specimens
                </button>
              </div>
            </div>

            <p className={styles.resultCount} role="status" aria-live="polite">
              {query
                ? `${shown.length} of ${total} species`
                : `Showing all ${total} species`}
            </p>
          </div>

          {/* Card grid or no-match state */}
          {shown.length === 0 ? (
            <div className="state">
              <span className="state-ico" aria-hidden="true">
                <SparkleIcon width={26} height={26} />
              </span>
              <span className="state-title">No species match “{query}”</span>
              <p>Try a shorter or different name — for example “lily” or “rose”.</p>
            </div>
          ) : (
            <div className={styles.grid}>
              {shown.map((cat) => {
                const gallery = numberFmt.format(cat.gallery_count);
                const totalCount = numberFmt.format(cat.total_count);
                return (
                  <button
                    type="button"
                    key={cat.label}
                    className={styles.card}
                    data-testid="category-card"
                    onClick={(e) => openSpecies(cat, e)}
                    aria-haspopup="dialog"
                    aria-label={`${cat.label_name} — ${cat.total_count} specimens. Open gallery.`}
                  >
                    <div className={styles.thumbWrap}>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        className={styles.thumb}
                        src={`/api/specimen/${encodeURIComponent(cat.specimen_id)}/image`}
                        alt={cat.label_name}
                        loading="lazy"
                        decoding="async"
                        onError={(e) => {
                          e.currentTarget.style.visibility = "hidden";
                        }}
                      />
                      <span
                        className={styles.accentBar}
                        style={{ background: cat.color }}
                        aria-hidden="true"
                      />
                    </div>
                    <div className={styles.body}>
                      <div className={styles.nameRow}>
                        <span
                          className={styles.colorDot}
                          style={{ background: cat.color }}
                          aria-hidden="true"
                        />
                        <h3 className={styles.name}>{cat.label_name}</h3>
                      </div>
                      <div className={styles.counts}>
                        <span className={styles.countPill} title="Gallery specimens">
                          <GridIcon width={12} height={12} /> <b>{gallery}</b> gallery
                        </span>
                        <span className={styles.countPill} title="Total specimens">
                          <LayersIcon width={12} height={12} /> <b>{totalCount}</b> total
                        </span>
                        <span className={styles.expandHint} aria-hidden="true">
                          Open <ChevronIcon width={13} height={13} />
                        </span>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </>
      )}

      {/* Expanded species dialog */}
      {selected && (
        <div
          className={styles.backdrop}
          onClick={closeSpecies}
          data-testid="category-modal"
        >
          <div
            className={styles.modal}
            role="dialog"
            aria-modal="true"
            aria-labelledby="cat-modal-title"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              ref={closeRef}
              type="button"
              className={styles.modalClose}
              onClick={closeSpecies}
              aria-label="Close species gallery"
            >
              ×
            </button>

            <div className={styles.modalHead}>
              <div className={styles.modalHeadThumb}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`/api/specimen/${encodeURIComponent(selected.specimen_id)}/image`}
                  alt={selected.label_name}
                  loading="lazy"
                  onError={(e) => {
                    e.currentTarget.style.visibility = "hidden";
                  }}
                />
              </div>
              <div className={styles.modalHeadBody}>
                <span className={styles.modalEyebrow}>
                  <TagIcon width={12} height={12} /> Species
                </span>
                <div className={styles.modalTitleRow}>
                  <span
                    className={styles.colorDot}
                    style={{ background: selected.color }}
                    aria-hidden="true"
                  />
                  <h3 className={styles.modalTitle} id="cat-modal-title">
                    {selected.label_name}
                  </h3>
                </div>
                <div className={styles.modalCounts}>
                  <span className={styles.countPill}>
                    <GridIcon width={12} height={12} />{" "}
                    <b>{numberFmt.format(selected.gallery_count)}</b> gallery
                  </span>
                  <span className={styles.countPill}>
                    <LayersIcon width={12} height={12} />{" "}
                    <b>{numberFmt.format(selected.total_count)}</b> total
                  </span>
                </div>
              </div>
            </div>

            <div className={styles.modalActions}>
              {savedIds.has(selected.specimen_id) ? (
                <span className={styles.saveConfirm} role="status">
                  <CheckIcon width={16} height={16} /> Saved to your Garden
                </span>
              ) : (
                <button
                  type="button"
                  className={`btn btn-primary ${styles.saveBtn}`}
                  onClick={() => onSave(selected.specimen_id)}
                  disabled={savingId === selected.specimen_id}
                >
                  {savingId === selected.specimen_id ? (
                    <>
                      <span className="spin" aria-hidden="true" /> Saving…
                    </>
                  ) : (
                    <>
                      <LeafIcon width={16} height={16} aria-hidden="true" /> Save to Garden
                    </>
                  )}
                </button>
              )}
              {saveError && (
                <span className={styles.saveError} role="alert">
                  <AlertIcon width={14} height={14} /> {saveError}
                </span>
              )}
            </div>

            <div className={styles.modalBody}>
              <div className={styles.modalBodyHead}>
                <span className={styles.modalBodyTitle}>Gallery specimens</span>
                {galaxyStatus === "ready" && (
                  <span className={styles.modalBodyCount}>
                    {members.length} shown
                  </span>
                )}
              </div>

              {galaxyStatus === "loading" && (
                <div className={styles.modalLoading} aria-busy="true">
                  <span className={styles.spin} aria-hidden="true" /> Loading specimens…
                </div>
              )}
              {galaxyStatus === "error" && (
                <div className={styles.modalLoading}>
                  <AlertIcon width={16} height={16} /> Could not load this species&rsquo;
                  gallery specimens.
                </div>
              )}
              {galaxyStatus === "ready" && members.length === 0 && (
                <div className={styles.modalLoading}>
                  <SparkleIcon width={16} height={16} /> No additional gallery specimens for
                  this species.
                </div>
              )}
              {galaxyStatus === "ready" && members.length > 0 && (
                <div className={styles.memberGrid}>
                  {members.map((m) => (
                    <figure className={styles.member} key={m.specimen_id}>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={`/api/specimen/${encodeURIComponent(m.specimen_id)}/image`}
                        alt={`${selected.label_name} specimen ${m.specimen_id}`}
                        loading="lazy"
                        decoding="async"
                        onError={(e) => {
                          e.currentTarget.style.visibility = "hidden";
                        }}
                      />
                      <figcaption className={styles.memberId}>{m.specimen_id}</figcaption>
                    </figure>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
