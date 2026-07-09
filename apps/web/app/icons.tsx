// Presentational inline SVG icons for FloraLens. Purely visual — no logic.
// All icons inherit `currentColor` and accept standard SVG props so callers
// can size / label them. Decorative by default (aria-hidden); pass aria-label
// on icon-only controls for accessibility.

import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

const base = (props: IconProps) => ({
  width: 24,
  height: 24,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
  focusable: false,
  ...props,
});

// Blossom mark used in the wordmark — five petals around a center.
export function BloomIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="12" cy="12" r="2.4" />
      <path d="M12 9.6c0-2.6-1-4.6-2.4-4.6S7.2 6.9 7.2 9.6" />
      <path d="M14.4 12c2.6 0 4.6-1 4.6-2.4S16.7 7.2 14.4 7.2" />
      <path d="M12 14.4c0 2.6 1 4.6 2.4 4.6s2.4-1.9 2.4-4.6" />
      <path d="M9.6 12c-2.6 0-4.6 1-4.6 2.4s1.9 2.4 4.6 2.4" />
    </svg>
  );
}

// Camera with a leaf accent — upload / capture affordance for the dropzone.
export function CameraLeafIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M3 8.5A1.5 1.5 0 0 1 4.5 7h2l1.2-1.8A1 1 0 0 1 8.5 4.7h4a1 1 0 0 1 .8.5L14.5 7h4A1.5 1.5 0 0 1 20 8.5v9A1.5 1.5 0 0 1 18.5 19h-13A1.5 1.5 0 0 1 4 17.5v-9" />
      <circle cx="11.5" cy="12.5" r="3.2" />
    </svg>
  );
}

// Simple leaf — used near the URL loader / secondary contexts.
export function LeafIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M5 19c0-7 4-12 14-13-1 10-6 14-13 14a5 5 0 0 1-1-1Z" />
      <path d="M6 18c3-4 6-6 9-7" />
    </svg>
  );
}

// Magnifier — the primary search action.
export function SearchIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="11" cy="11" r="6.5" />
      <path d="m16 16 4 4" />
    </svg>
  );
}

// Link / URL loader glyph.
export function LinkIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M9.5 14.5 14.5 9.5" />
      <path d="M8 12 6 14a3.5 3.5 0 0 0 5 5l2-2" />
      <path d="M16 12l2-2a3.5 3.5 0 0 0-5-5l-2 2" />
    </svg>
  );
}

// Empty-state illustration cue.
export function SparkleIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M12 4c.5 3.5 1.5 4.5 5 5-3.5.5-4.5 1.5-5 5-.5-3.5-1.5-4.5-5-5 3.5-.5 4.5-1.5 5-5Z" />
      <path d="M18.5 14c.2 1.5.7 2 2.2 2.2-1.5.2-2 .7-2.2 2.2-.2-1.5-.7-2-2.2-2.2 1.5-.2 2-.7 2.2-2.2Z" />
    </svg>
  );
}

// Alert glyph for the error state.
export function AlertIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 8v4.5" />
      <path d="M12 15.8h.01" />
    </svg>
  );
}
