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

// Sun — light theme.
export function SunIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2.5v2M12 19.5v2M4.6 4.6l1.4 1.4M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4 6 18M18 6l1.4-1.4" />
    </svg>
  );
}

// Moon — dark theme.
export function MoonIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M20 14.5A8 8 0 0 1 9.5 4a7 7 0 1 0 10.5 10.5Z" />
    </svg>
  );
}

// Monitor — system (follow OS) theme.
export function MonitorIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <rect x="3" y="4.5" width="18" height="12" rx="1.6" />
      <path d="M8.5 20h7M12 16.5V20" />
    </svg>
  );
}

// Stacked disks — the dataset stage.
export function DatabaseIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <ellipse cx="12" cy="6" rx="7" ry="2.8" />
      <path d="M5 6v6c0 1.5 3.1 2.8 7 2.8s7-1.3 7-2.8V6" />
      <path d="M5 12v6c0 1.5 3.1 2.8 7 2.8s7-1.3 7-2.8v-6" />
    </svg>
  );
}

// Stacked layers — the frozen backbone / embedding model.
export function LayersIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M12 3.5 21 8l-9 4.5L3 8Z" />
      <path d="M3 12l9 4.5L21 12" />
      <path d="M3 16l9 4.5L21 16" />
    </svg>
  );
}

// Grid of cells — the vector index / gallery.
export function GridIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <rect x="3.5" y="3.5" width="7" height="7" rx="1.4" />
      <rect x="13.5" y="3.5" width="7" height="7" rx="1.4" />
      <rect x="3.5" y="13.5" width="7" height="7" rx="1.4" />
      <rect x="13.5" y="13.5" width="7" height="7" rx="1.4" />
    </svg>
  );
}

// Gauge — calibration.
export function GaugeIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M4 16a8 8 0 1 1 16 0" />
      <path d="M12 16l4-4.5" />
      <circle cx="12" cy="16" r="1.1" fill="currentColor" stroke="none" />
    </svg>
  );
}

// Bar chart — evaluation metrics.
export function ChartIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M4 20h16" />
      <rect x="5.5" y="12" width="3.2" height="6" rx="0.8" />
      <rect x="10.4" y="8" width="3.2" height="10" rx="0.8" />
      <rect x="15.3" y="4.5" width="3.2" height="13.5" rx="0.8" />
    </svg>
  );
}

// Wand with a spark — the preprocessing transform.
export function WandIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M5 19 16 8" />
      <path d="M14.5 5.5 18.5 9.5" />
      <path d="M18 3.5c.2 1 .5 1.3 1.5 1.5-1 .2-1.3.5-1.5 1.5-.2-1-.5-1.3-1.5-1.5 1-.2 1.3-.5 1.5-1.5Z" />
    </svg>
  );
}

// Check in a rounded badge — the promotion decision.
export function CheckIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="m8.4 12.2 2.4 2.4 4.8-5.2" />
    </svg>
  );
}

// Paper plane — the assistant chat's send action.
export function SendIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M20 4 3.5 10.8c-.9.4-.8 1.7.1 1.9l6 1.7 1.7 6c.3.9 1.6 1 1.9.1L20 4Z" />
      <path d="M11 13.5 20 4" />
    </svg>
  );
}

// Chevron — expand/collapse affordance.
export function ChevronIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

// Tag — the Categories / species-catalog tab.
export function TagIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M4 12.5V5.5A1.5 1.5 0 0 1 5.5 4h7a2 2 0 0 1 1.4.6l5.5 5.5a1.5 1.5 0 0 1 0 2.1l-6.7 6.7a1.5 1.5 0 0 1-2.1 0L4.6 13.9A2 2 0 0 1 4 12.5Z" />
      <circle cx="8.5" cy="8.5" r="1.2" fill="currentColor" stroke="none" />
    </svg>
  );
}

// CPU / chip — the device (GPU vs CPU) benchmark.
export function CpuIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <rect x="6.5" y="6.5" width="11" height="11" rx="1.6" />
      <rect x="9.5" y="9.5" width="5" height="5" rx="0.8" />
      <path d="M9 3.5v3M15 3.5v3M9 17.5v3M15 17.5v3M3.5 9h3M3.5 15h3M17.5 9h3M17.5 15h3" />
    </svg>
  );
}
