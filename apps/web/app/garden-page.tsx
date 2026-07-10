"use client";

import { useCallback, useEffect, useState } from "react";
import {
  addToGarden,
  clearMemory,
  getGarden,
  getMemory,
  removeFromGarden,
  type GardenItem,
  type MemoryItem,
} from "@/lib/api";
import { AlertIcon, LeafIcon, SparkleIcon } from "./icons";

type Status = "loading" | "ready" | "error";

export function GardenPage() {
  const [items, setItems] = useState<GardenItem[]>([]);
  const [status, setStatus] = useState<Status>("loading");
  const [error, setError] = useState<string | null>(null);
  const [addId, setAddId] = useState("");
  const [saving, setSaving] = useState(false);

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

  async function onAdd() {
    const specimenId = addId.trim();
    if (!specimenId || saving) return;
    setSaving(true);
    setError(null);
    try {
      await addToGarden(specimenId);
      setAddId("");
      loadGarden();
    } catch (e) {
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

      <section className="card garden-add">
        <div className="head">
          <h3 className="title">Save a specimen</h3>
          <span className="sub">by id</span>
        </div>
        <div className="body">
          <label className="field-label" htmlFor="garden-add-id">
            Specimen id
          </label>
          <div className="url-row">
            <input
              id="garden-add-id"
              type="text"
              placeholder="e.g. the id shown under a search result's thumbnail"
              value={addId}
              onChange={(e) => setAddId(e.target.value)}
              data-testid="garden-add-input"
            />
            <button
              type="button"
              className="btn btn-primary"
              onClick={onAdd}
              disabled={!addId.trim() || saving}
              data-testid="garden-add-btn"
            >
              {saving ? "Saving…" : "Save"}
            </button>
          </div>
          {error && (
            <p className="err" data-testid="error">
              <AlertIcon width={16} height={16} aria-hidden="true" />
              <span>{error}</span>
            </p>
          )}
        </div>
      </section>

      <section className="card">
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
            <div className="state" data-testid="garden-empty">
              <span className="state-ico" aria-hidden="true">
                <SparkleIcon width={26} height={26} />
              </span>
              <span className="state-title">Your garden is empty</span>
              <p>Save a specimen id above to start your collection.</p>
            </div>
          )}
          {items.length > 0 && (
            <div className="grid">
              {items.map((it) => (
                <article className="result garden-item" data-testid="garden-item" key={it.specimen_id}>
                  <div className="thumb-wrap">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      className="thumb"
                      src={`/api/specimen/${encodeURIComponent(it.specimen_id)}/image`}
                      alt={it.label_name}
                      loading="lazy"
                      onError={(e) => {
                        e.currentTarget.style.visibility = "hidden";
                      }}
                    />
                  </div>
                  <div className="meta">
                    <h4 className="name">{it.label_name}</h4>
                    <span className="sub">{it.specimen_id}</span>
                    <button
                      type="button"
                      className="btn btn-ghost"
                      onClick={() => onRemove(it.specimen_id)}
                      data-testid="garden-remove-btn"
                    >
                      Remove
                    </button>
                  </div>
                </article>
              ))}
            </div>
          )}
        </div>
      </section>

      <section className="card">
        <div className="head">
          <h3 className="title">Assistant memory</h3>
          <span className="sub">{memories.length}</span>
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
              <ul className="memory-list" data-testid="memory-list">
                {memories.map((m) => (
                  <li className="memory-item" data-testid="memory-item" key={m.id ?? m.text}>
                    {m.text}
                  </li>
                ))}
              </ul>
              {memError && (
                <p className="err" data-testid="error">
                  <AlertIcon width={16} height={16} aria-hidden="true" />
                  <span>{memError}</span>
                </p>
              )}
              <button
                type="button"
                className="btn btn-ghost"
                onClick={onClearMemory}
                disabled={clearing}
                data-testid="memory-clear-btn"
              >
                {clearing ? "Clearing…" : "Clear all memory"}
              </button>
            </>
          )}
        </div>
      </section>
    </div>
  );
}
