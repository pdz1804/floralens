# FloraLens — Design System

Reference for the current FloraLens web UI (`apps/web`). Written so future work
can match the existing taste. Every value here is defined in
`app/globals.css` (`:root` / `[data-theme]`) or a per-tab CSS Module; token names
below are the source of truth — prefer `var(--token)` over raw literals.

## 1. Direction

**Editorial naturalist.** A calibrated, field-guide feeling: warm off-white
paper, deep-forest brand green, herbarium ochre as the warm secondary, a
whisper of paper grain, generous type, restrained motion. Panels stay flat;
one soft radial glow lives on the body backdrop. Confidence is communicated
honestly through a three-band colour language (High / Medium / Low).

Marketing surfaces (About hero, Search hero) may carry richer motion; dense
data surfaces (Pipeline tables, the Galaxy 3D canvas) stay calm.

## 2. Typography

Fonts are wired via `next/font` in `layout.tsx` and exposed as CSS variables,
with system fallbacks so a build never blocks on a font.

| Role | Stack token | Family | Usage |
|------|-------------|--------|-------|
| Display / headings | `--font-serif-stack` | Playfair Display → Georgia, serif | `h1`–`h3`, hero `h2`, stat values, card titles, species names |
| Body / UI | `--font-sans-stack` | Inter → system-ui | Body copy, controls, labels, taglines |
| Mono | `--font-mono` | SFMono / ui-monospace | Scores, ids, counts, model versions, code |

- Heroes push Playfair presence: `font-weight: 700`, `letter-spacing: -0.025em`.
- Headings default `font-weight: 600`, `letter-spacing: -0.01em`.
- Body base 15px / line-height 1.6. Vietnamese diacritics render correctly in
  both Playfair Display and Inter (both ship the Vietnamese subset).

### Type scale (base 15px)

| Token | px | Typical use |
|-------|----|-------------|
| `--text-2xs` | 11 | Uppercase kickers, meta labels |
| `--text-xs` | 12 | Chips, hints, footnotes |
| `--text-sm` | 13 | Secondary body, controls |
| `--text-md` | 14 | Body copy |
| `--text-base` | 15 | Root body size |
| `--text-lg` | 17 | Card titles |
| `--text-xl` | 20 | Section subheads |
| `--text-2xl` | 24 | Stat values |
| `--text-3xl` | 30 | Large headings |

Hero headings use fluid `clamp()` (e.g. `clamp(30px, 5vw, 44px)`), not fixed steps.

## 3. Colour tokens

Light is default (`:root` / `[data-theme="light"]`); dark applies via
`[data-theme="dark"]` or, when the theme is System/unset, the
`prefers-color-scheme: dark` media query (values kept in sync across both).

### Surfaces & ink

| Token | Light | Dark | Role |
|-------|-------|------|------|
| `--bg` | `#fbfcfa` | `#0f1512` | Page background |
| `--bg-tint` | `#f4f8f3` | `#131b16` | Backdrop glow tint |
| `--panel` | `#ffffff` | `#151d17` | Card / panel surface |
| `--panel-2` | `#f2f6f1` | `#1b241d` | Inset / subtle fill |
| `--panel-3` | `#eaf1e9` | `#212c23` | Deeper inset (thumb bg, tracks) |
| `--ink` | `#182319` | `#e8f0e8` | Primary text |
| `--muted` | `#566b58` | `#9fb3a1` | Secondary text (~5.3:1 on bg) |
| `--faint` | `#5f7261` | `#7d907f` | Tertiary / meta text (≥4.5:1) |

### Lines

| Token | Light | Dark | Role |
|-------|-------|------|------|
| `--line` | `#dde6dc` | `#2a352b` | Default borders / dividers |
| `--line-strong` | `#cdd9cc` | `#354236` | Hover / emphasized borders |

### Brand (deep forest / pine)

| Token | Light | Dark | Role |
|-------|-------|------|------|
| `--primary` | `#1f6b45` | `#43a069` | Brand fills, active tab, CTAs (white label ≥6:1 AA) |
| `--primary-ink` | `#ffffff` | `#08130c` | Text on filled primary |
| `--primary-soft` | `#e4f0e7` | `#1c2e22` | Soft tinted fills, focus ring halo |
| `--accent` | `#338a58` | `#6fce9c` | Hover borders, low-key fills |
| `--ring` | `#1f6b45` | `#6fce9c` | Focus outline |

### Secondary (herbarium ochre / dried terracotta)

| Token | Light | Dark | Role |
|-------|-------|------|------|
| `--secondary` | `#9c6222` | `#d69a4f` | Warm accents, kickers (white label ≥5:1 AA) |
| `--secondary-ink` | `#ffffff` | `#1c1408` | Text on filled secondary |
| `--secondary-soft` | `#f3e6d2` | `#2c2113` | Soft warm fill |

### Confidence bands (text ≥4.5:1 on its soft chip bg)

| Band | Text (L / D) | Soft bg (L / D) |
|------|--------------|-----------------|
| High | `--high` `#237a45` / `#63c98a` | `--high-soft` `#e2f1e6` / `#16281c` |
| Medium | `--med` `#925417` / `#d99f47` | `--med-soft` `#f3e6d2` / `#2c2113` |
| Low | `--low` `#a8432a` / `#e08a6d` | `--low-soft` `#f6e4de` / `#2e1d16` |

### Galaxy WebGL scene (single source of truth for R3F + CSS)

| Token | Value (both themes) | Role |
|-------|---------------------|------|
| `--galaxy-canvas-bg` | `#070b09` | Dark scene background (read by R3F) |
| `--galaxy-dim` | `#5b6b5d` | Legend-dimmed points |
| `--galaxy-tooltip-bg` | `#eef5ee` | Tooltip surface |
| `--galaxy-tooltip-ink` | `#0b120d` | Tooltip text |

## 4. Spacing scale (4px base)

`--space-1` 4 · `--space-2` 8 · `--space-3` 12 · `--space-4` 16 ·
`--space-5` 20 · `--space-6` 24 · `--space-8` 32 · `--space-10` 40 ·
`--space-12` 48 · `--space-16` 64. Raw gap/padding/margin literals snap to these.

## 5. Radius

| Token | Value | Use |
|-------|-------|-----|
| `--r-sm` | 8px | Inputs, chips, inset tiles |
| `--r` | 14px | Cards, result cards |
| `--r-lg` | 20px | Panels, modals, large cards |
| `--pill` | 999px | Chips, buttons, badges, status |

## 6. Elevation (shadow)

| Token | Light | Role |
|-------|-------|------|
| `--shadow-sm` | `0 1px 2px rgba(24,35,25,.05)` | Resting cards |
| `--shadow` | `0 6px 20px -8px …, 0 2px 6px -3px …` | Hover lift, primary panels |
| `--shadow-lg` | `0 18px 44px -18px …, 0 6px 16px -8px …` | Modals, strong hover |

Dark theme uses darker, higher-opacity variants of the same three tokens.

## 7. Z-index

| Layer | z | Where |
|-------|---|-------|
| Body grain / glow | 0 | `body::before`, content lifted to `1` |
| Sticky topbar | 10 | `.topbar` |
| Category modal backdrop | 50 | `.backdrop` (categories module) |

Sticky sidebars (`.left-card`, garden `.aside`, galaxy `.side`) use
`top: 88px` and fall back to `position: static` below 860–900px.

## 8. Motion

Motion is powered by `motion` (motion/react — the Framer Motion successor),
imported per client component. It is **motivated only**: each animation serves
hierarchy, reveal, or feedback. Richer motion on marketing surfaces; restraint
on functional/data surfaces.

### Signature

- **Easing:** `[0.16, 1, 0.3, 1]` (expo-out) — the shared `EASE` constant across
  entrances; interactive lifts use a spring (`stiffness 320, damping 26`).
- **Durations:** entrances 0.30–0.45s; About section reveals 0.5–0.6s.
- **Stagger:** ~0.05–0.06s per item, delay capped at 0.3–0.4s so long lists
  (e.g. many result cards) never crawl.

### Per-surface behaviour

| Surface | Motion | Notes |
|---------|--------|-------|
| About hero | Fade + rise on mount (0.6s) | `about-page.tsx` |
| About sections (stats, capabilities, steps, closing) | Scroll-reveal, `whileInView` `{once:true, amount:0.2–0.25}`, initial `{opacity:0, y:24}` | Stagger via container/item `variants` |
| About feature cards | Reveal + hover lift `y:-4` + tap `scale:.98` + cursor-follow spotlight | Spotlight uses motion values (`useMotionValue` + `useMotionTemplate`), **no state**; `.spotlight` overlay in `about.module.css` |
| Search hero | Fade + rise on mount (0.45s) | `page.tsx` |
| Search result cards | Staggered fade-in on arrival (`initial→animate`, delay `i*0.05` capped 0.4s) + hover lift + tap | Band grouping + testids intact |
| Categories / Garden / Assistant / Pipeline tab roots | Gentle entrance fade/slide `{opacity:0→1, y:8→0}` (0.35s) on mount | Runs when the tab mounts |
| Category cards | Hover lift `y:-4` + tap `scale:.98` | `motion.button`, testid preserved |
| Garden items | Entrance stagger + hover lift + tap | `motion.article` |
| Assistant message rows | Fade + rise (0.3s) as each bubble mounts | Streaming caret, SSE parse and `assistant-answer`/`assistant-citations` sinks untouched |
| Pipeline tiles / bench cards | Calm CSS hover lift `y:-3px` | `pipeline.module.css`; no scale, data stays quiet |
| Galaxy 3D canvas / 2D fallback | **None** | No motion wrapper around the R3F `<Canvas>`; nothing that could remount it |

No `AnimatePresence` exit animations on tab switch; the Search panel stays
mounted via the `hidden` attribute, other panels mount/unmount as before.

### Reduced motion (mandatory)

Every component reads `const reduce = useReducedMotion() ?? false;` and gates
**all** motion props on it — when the user prefers reduced motion, entrances,
hovers, taps, staggers and the spotlight all degrade to fully static (no
transforms, no opacity animation). Belt-and-suspenders: `globals.css` also
nullifies transitions/animations under `@media (prefers-reduced-motion: reduce)`,
and the CSS hover-lifts (result/feature/pipeline) are disabled there too. The
Galaxy 3D capability gate already treats reduced-motion as "3D unavailable" and
falls back to the static 2D scatter.

## 9. Responsiveness

Mobile-first hardening; existing breakpoints kept.

- Full-height backdrop uses `min-height: 100dvh` (with a `100vh` fallback), never
  bare `100vh`/`h-screen`.
- **Breakpoints in use:** 900px (search `380px+1fr` → 1col, left-card static),
  860px (garden/galaxy grids → 1col, sidebars static), 720px (topbar wraps, the
  7-tab pill bar wraps to multiple rows; tabs get ≥40px tap targets), 620px
  (About lead feature + train split stop spanning 2 cols), 560px (tagline /
  toggle text hidden).
- Multi-column grids use `repeat(auto-fit|auto-fill, minmax(…, 1fr))` so they
  collapse to a single column on small screens.
- Horizontal-scroll containers (`.flow-scroll`, `.filmstrip`, `.table-wrap`)
  contain wide diagrams/tables so the page never overflows at 360px.
- The category modal is a bottom sheet under 640px, centred dialog above.
- Tap targets: nav tabs ≥40px on touch; touch (`hover: none`) devices keep the
  garden remove control always visible.

## 10. Accessibility notes

- Colour pairs meet WCAG 2.1 AA (brand/secondary labels ≥5–6:1; band text ≥4.5:1
  on its soft chip).
- Universal `:focus-visible` ring (`--ring`, 2px, offset 2px).
- Tabs follow the WAI-ARIA tabs pattern (roving focus, arrow/Home/End keys).
- Motion honours `prefers-reduced-motion`; the Galaxy honours it as a hard gate.
