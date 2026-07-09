"use client";

import { useEffect, useState } from "react";
import { MonitorIcon, MoonIcon, SunIcon } from "./icons";

type Theme = "system" | "light" | "dark";
const ORDER: Theme[] = ["system", "light", "dark"];
const STORAGE_KEY = "floralens-theme";
const LABEL: Record<Theme, string> = { system: "System", light: "Light", dark: "Dark" };

function readStored(): Theme {
  if (typeof window === "undefined") return "system";
  const t = window.localStorage.getItem(STORAGE_KEY);
  return t === "light" || t === "dark" || t === "system" ? t : "system";
}

/**
 * Accessible, cycling theme control (System → Light → Dark). The active choice
 * is written to <html data-theme> (CSS drives the actual palette; "system"
 * follows prefers-color-scheme) and persisted to localStorage. A no-FOUC inline
 * script in layout.tsx applies the stored value before first paint.
 */
export function ThemeToggle() {
  // Start from "system" so SSR and the client's first render agree; the stored
  // value is read after mount to avoid a hydration mismatch.
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => {
    setTheme(readStored());
  }, []);

  function apply(next: Theme) {
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* storage may be unavailable (private mode) — attribute still applied */
    }
  }

  function cycle() {
    apply(ORDER[(ORDER.indexOf(theme) + 1) % ORDER.length]);
  }

  const next = ORDER[(ORDER.indexOf(theme) + 1) % ORDER.length];
  const Icon = theme === "light" ? SunIcon : theme === "dark" ? MoonIcon : MonitorIcon;

  return (
    <button
      type="button"
      className="theme-toggle"
      data-testid="theme-toggle"
      onClick={cycle}
      title={`Theme: ${LABEL[theme]}`}
      aria-label={`Theme: ${LABEL[theme]}. Switch to ${LABEL[next]}.`}
    >
      <Icon width={18} height={18} aria-hidden="true" />
      <span className="theme-toggle-text">{LABEL[theme]}</span>
    </button>
  );
}
