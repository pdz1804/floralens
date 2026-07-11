"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  addToGarden,
  clearMemory,
  getGarden,
  getMemory,
  removeFromGarden,
  type GardenItem,
  type MemoryItem,
} from "@/lib/api";
import { AlertIcon, CheckIcon, LeafIcon, SparkleIcon } from "./icons";
import styles from "./garden.module.css";

type Status = "loading" | "ready" | "error";

// Local, decorative glyphs kept inline so this tab needs no icon-set change.
function CloseGlyph() {
  return (
    <svg
      width={15}
      height={15}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d="M6 6l12 12M18 6 6 18" />
    </svg>
  );
}

export function GardenPage() {
  const [items, setItems] = useState<GardenItem[]>([]);
  const [status, setStatus] = useState<Status>("loading");
  const [error, setError] = useState<string | null>(null);
  const [addId, setAddId] = useState("");
  const [saving, setSaving] = useState(false);
  const [savedName, setSavedName] = useState<string | null>(null);
  const savedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [memStatus, setMemStatus] = useState<Status>("loading");
  const [memError, setMemError] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);

  const loadGarden = useCallback((signal?: AbortSignal) => {
    setStatus((s) => (s === "ready" ? s : "loading"));
    getGarden(signal)
      .then((d) => {
        setItems(d.items);
        setStatus("ready");
      })
      .catch((e) => {
        if ((e as Error).name === "AbortError") return;
        setError((e as Error).message);
        setStatus("error");
      });
  }, []);

  const loadMemory = useCallback((signal?: AbortSignal) => {
    setMemStatus((s) => (s === "ready" ? s : "loading"));
    getMemory(signal)
      .then((d) => {
        setMemories(d.items);
        setMemStatus("ready");
      })
      .catch((e) => {
        if ((e as Error).name === "AbortError") return;
        setMemError((e as Error).message);
        setMemStatus("error");
      });
  }, []);

  useEffect(() => {
    const ctrl = new AbortController();
    loadGarden(ctrl.signal);
    return () => ctrl.abort();
  }, [loadGarden]);

  useEffect(() => {
    const ctrl = new AbortController();
    loadMemory(ctrl.signal);
    return () => ctrl.abort();
  }, [loadMemory]);

  // Clear any pending "saved" toast timer when the tab unmounts.
  useEffect(() => () => {
    if (savedTimer.current) clearTimeout(savedTimer.current);
  }, []);

  async function onAdd() {
    const specimenId = addId.trim();
    if (!specimenId || saving) return;
    setSaving(true);
    setError(null);
    try {
      const saved = await addToGarden(specimenId);
      setAddId("");
      // Friendly, transient confirmation naming what was just saved.
      setSavedName(saved.label_name || specimenId);
      if (savedTimer.current) clearTimeout(savedTimer.current);
      savedTimer.current = setTimeout(() => setSavedName(null), 4000);
      loadGarden();
    } catch (e) {
      setSavedName(null);
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function onRemove(specimenId: string) {
    const removed = items.find((it) => it.specimen_id === specimenId);
    setItems((cur) => cur.filter((it) => it.specimen_id !== specimenId)); // optimistic
    try {
      await removeFromGarden(specimenId);
    } catch (e) {
      // Re-insert ONLY the failed item via a functional update — restoring a
      // whole pre-removal snapshot would resurrect a different item that a
      // concurrent onRemove had already successfully deleted.
      if (removed) {
        setItems((cur) =>
          cur.some((it) => it.specimen_id === specimenId) ? cur : [removed, ...cur],
        );
      }
      setError((e as Error).message);
    }
  }

  async function onClearMemory() {
    if (clearing) return;
    setClearing(true);
    setMemError(null);
    try {
      await clearMemory();
      setMemories([]);
    } catch (e) {
      setMemError((e as Error).message);
    } finally {
      setClearing(false);
    }
  }

  return (
    <div className="garden" data-testid="garden-page">
      <section className="garden-hero">
        <span className="eyebrow">
          <LeafIcon width={14} height={14} /> Your collection
        </span>
        <h2>My Garden</h2>
        <p>
          Specimens you&rsquo;ve saved, and what the naturalist assistant remembers about
          you — both visible and removable here at any time.
        </p>
      </section>

      <div className={styles.layout}>
        {/* ---- Aside: save form + at-a-glance counts ---- */}
        <aside className={styles.aside}>
          <section className="card garden-add">
            <div className="head">
              <h3 className="title">Save a specimen</h3>
              <span className="sub">by id</span>
            </div>
            <div className="body">
              <label className={`field-label ${styles.addLabel}`} htmlFor="garden-add-id">
                Specimen id
              </label>
              <p className={styles.helper}>
                Paste a specimen id below to add that flower to your garden.
              </p>
              <div className={styles.saveRow}>
                <input
                  id="garden-add-id"
                  type="text"
                  placeholder="e.g. gallery_00421"
                  value={addId}
                  onChange={(e) => setAddId(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") onAdd();
                  }}
                  aria-describedby="garden-add-hint"
                  data-testid="garden-add-input"
                />
                <button
                  type="button"
                  className={`btn btn-primary ${styles.saveBtn}`}
                  onClick={onAdd}
                  disabled={!addId.trim() || saving}
                  data-testid="garden-add-btn"
                >
                  {saving ? "Saving…" : "Save"}
                </button>
              </div>

              {savedName && (
                <p className={styles.feedback} role="status" data-testid="garden-saved">
                  <CheckIcon width={16} height={16} aria-hidden="true" />
                  <span>
                    Saved <strong>{savedName}</strong> to your garden
                  </span>
                </p>
              )}
              {error && (
                <p className="err" data-testid="error">
                  <AlertIcon width={16} height={16} aria-hidden="true" />
                  <span>{error}</span>
                </p>
              )}

              <p id="garden-add-hint" className={styles.hint}>
                <span className={styles.hintIco} aria-hidden="true">
                  <SparkleIcon width={15} height={15} />
                </span>
                <span>
                  <strong>Don&rsquo;t know an id?</strong> Every result in the{" "}
                  <b>Search</b> tab and every point in the <b>Galaxy</b> tab shows its
                  specimen id — copy one here to save it.
                </span>
              </p>
            </div>
          </section>

          <div className={styles.stats}>
            <div className={styles.stat}>
              <span className={styles.statNum}>{items.length}</span>
              <span className={styles.statLabel}>Saved</span>
            </div>
            <div className={styles.stat}>
              <span className={styles.statNum}>{memories.length}</span>
              <span className={styles.statLabel}>Memories</span>
            </div>
          </div>
        </aside>

        {/* ---- Main: saved specimens grid ---- */}
        <section className={`card ${styles.main}`}>
          <div className="head">
            <h3 className="title">Saved specimens</h3>
            <span className="sub">{items.length}</span>
          </div>
          <div className="body">
            {status === "loading" && (
              <div className="state" aria-busy="true">
                <span className="spin" aria-hidden="true" />
                <span className="state-title">Loading your garden…</span>
              </div>
            )}
            {status === "error" && (
              <div className="state error">
                <span className="state-ico" aria-hidden="true">
                  <AlertIcon width={26} height={26} />
                </span>
                <span className="state-title">Garden unavailable</span>
                <p>Could not load your saved specimens. Is the backend running?</p>
              </div>
            )}
            {status === "ready" && items.length === 0 && (
              <div className={styles.empty} data-testid="garden-empty">
                <span className={styles.emptyIco} aria-hidden="true">
                  <LeafIcon width={30} height={30} />
                </span>
                <p className={styles.emptyTitle}>Your garden is empty</p>
                <p className={styles.emptyText}>
                  Save flowers you want to revisit and they&rsquo;ll bloom here. Here&rsquo;s
                  how to add your first one:
                </p>
                <div className={styles.tips}>
                  <div className={styles.tip}>
                    <span className={styles.tipNum} aria-hidden="true">
                      1
                    </span>
                    <span>
                      Find a flower in the <b>Search</b> or <b>Galaxy</b> tab.
                    </span>
                  </div>
                  <div className={styles.tip}>
                    <span className={styles.tipNum} aria-hidden="true">
                      2
                    </span>
                    <span>Copy its specimen id.</span>
                  </div>
                  <div className={styles.tip}>
                    <span className={styles.tipNum} aria-hidden="true">
                      3
                    </span>
                    <span>
                      Paste it into <b>Save a specimen</b> and press Save.
                    </span>
                  </div>
                </div>
              </div>
            )}
            {items.length > 0 && (
              <div className={styles.grid}>
                {items.map((it) => (
                  <article
                    className={`result garden-item ${styles.item}`}
                    data-testid="garden-item"
                    key={it.specimen_id}
                  >
                    <div className="thumb-wrap">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        className="thumb"
                        src={`/api/specimen/${encodeURIComponent(it.specimen_id)}/image`}
                        alt={`Saved specimen: ${it.label_name}`}
                        loading="lazy"
                        onError={(e) => {
                          e.currentTarget.style.visibility = "hidden";
                        }}
                      />
                      <button
                        type="button"
                        className={styles.remove}
                        onClick={() => onRemove(it.specimen_id)}
                        aria-label={`Remove ${it.label_name} from your garden`}
                        title="Remove from garden"
                        data-testid="garden-remove-btn"
                      >
                        <CloseGlyph />
                      </button>
                    </div>
                    <div className={`meta ${styles.itemMeta}`}>
                      <h4 className={`name ${styles.itemName}`}>{it.label_name}</h4>
                      <span className={styles.itemId} title={it.specimen_id}>
                        {it.specimen_id}
                      </span>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>

      {/* ---- Assistant memory ---- */}
      <section className="card">
        <div className={styles.memHead}>
          <h3 className={styles.memTitle}>Assistant memory</h3>
          <span className={styles.memCount}>{memories.length}</span>
          <span className={styles.memSpacer} />
          {memories.length > 0 && (
            <button
              type="button"
              className={`btn btn-ghost ${styles.memClear}`}
              onClick={onClearMemory}
              disabled={clearing}
              data-testid="memory-clear-btn"
            >
              {clearing ? "Clearing…" : "Clear all"}
            </button>
          )}
        </div>
        <div className="body">
          {memStatus === "loading" && (
            <div className="state" aria-busy="true">
              <span className="spin" aria-hidden="true" />
              <span className="state-title">Loading memory…</span>
            </div>
          )}
          {memStatus === "error" && (
            <div className="state error">
              <span className="state-ico" aria-hidden="true">
                <AlertIcon width={26} height={26} />
              </span>
              <span className="state-title">Memory unavailable</span>
              <p>{memError}</p>
            </div>
          )}
          {memStatus === "ready" && memories.length === 0 && (
            <div className="state" data-testid="memory-empty">
              <span className="state-ico" aria-hidden="true">
                <SparkleIcon width={26} height={26} />
              </span>
              <span className="state-title">Nothing remembered yet</span>
              <p>Chat with the naturalist assistant and it will remember what you discuss.</p>
            </div>
          )}
          {memories.length > 0 && (
            <>
              <ul className={styles.memList} data-testid="memory-list">
                {memories.map((m) => (
                  <li className={styles.memItem} data-testid="memory-item" key={m.id ?? m.text}>
                    <LeafIcon className={styles.memBullet} width={15} height={15} aria-hidden="true" />
                    <span>{m.text}</span>
                  </li>
                ))}
              </ul>
              {memError && (
                <p className="err" data-testid="error">
                  <AlertIcon width={16} height={16} aria-hidden="true" />
                  <span>{memError}</span>
                </p>
              )}
            </>
          )}
        </div>
      </section>
    </div>
  );
}
