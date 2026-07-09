"use client";

import { useLayoutEffect, useRef, useState } from "react";

/**
 * Curated botanical description for a result card. Clamped to a few lines with a
 * Read more / less toggle that only appears when the text actually overflows.
 * Display-only — never affects search behavior.
 */
export function ResultDesc({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const [overflowing, setOverflowing] = useState(false);
  const ref = useRef<HTMLParagraphElement>(null);

  // Detect clamp overflow so the toggle is shown only when it's useful.
  useLayoutEffect(() => {
    const el = ref.current;
    if (el) setOverflowing(el.scrollHeight - el.clientHeight > 2);
  }, [text]);

  return (
    <div className="desc-wrap">
      <p
        ref={ref}
        className={`desc ${expanded ? "expanded" : ""}`}
        data-testid="result-desc"
      >
        {text}
      </p>
      {(overflowing || expanded) && (
        <button
          type="button"
          className="desc-toggle"
          aria-expanded={expanded}
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? "Read less" : "Read more"}
        </button>
      )}
    </div>
  );
}
