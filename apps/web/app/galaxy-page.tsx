"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useThree, type ThreeEvent } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { getGalaxy, type GalaxyData, type GalaxyPoint } from "@/lib/api";
import { AlertIcon, GridIcon, SparkleIcon } from "./icons";

// Neutral color for points dimmed by the legend's species filter — matches
// the botanical theme's --muted family closely enough without importing CSS
// variables into a WebGL color (three.js needs a concrete hex at draw time).
const DIM_COLOR = "#5b6b5d";
const TOP_SPECIES_COUNT = 12;
// World-space point size (points live in a roughly [-1, 1] normalized cube) —
// tuned so the cloud reads clearly at the default camera distance without
// individual points collapsing to sub-pixel dots.
const POINT_SIZE = 0.05;
// Raycaster hit-radius for THREE.Points picking, in the same world units —
// a bit larger than the visual point so hover/click stay easy to land.
const POINT_PICK_THRESHOLD = 0.03;

function detectWebGL(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return !!(
      canvas.getContext("webgl2") ||
      canvas.getContext("webgl") ||
      canvas.getContext("experimental-webgl")
    );
  } catch {
    return false;
  }
}

type Capability = "checking" | "supported" | "unsupported";

/** One-time client-side check: real WebGL context AND no reduced-motion
 * preference. Runs only in an effect (never during SSR/hydration) so the
 * server-rendered markup and the first client render always agree — no
 * hydration mismatch, just a brief "checking" state that settles instantly. */
function use3DCapability(): Capability {
  const [capability, setCapability] = useState<Capability>("checking");
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const webgl = detectWebGL();
    const evaluate = () => setCapability(webgl && !mq.matches ? "supported" : "unsupported");
    evaluate();
    mq.addEventListener("change", evaluate);
    return () => mq.removeEventListener("change", evaluate);
  }, []);
  return capability;
}

type LegendEntry = { label_name: string; color: string; count: number };

function buildLegend(points: GalaxyPoint[]): LegendEntry[] {
  const byName = new Map<string, LegendEntry>();
  for (const p of points) {
    const existing = byName.get(p.label_name);
    if (existing) existing.count += 1;
    else byName.set(p.label_name, { label_name: p.label_name, color: p.color, count: 1 });
  }
  return [...byName.values()].sort((a, b) => b.count - a.count).slice(0, TOP_SPECIES_COUNT);
}

/** Tunes the shared raycaster's Points picking radius once on mount — must
 * run inside the Canvas (useThree only works within the r3f tree). */
function RaycasterTuning({ threshold }: { threshold: number }) {
  const { raycaster } = useThree();
  useEffect(() => {
    raycaster.params.Points = { threshold };
  }, [raycaster, threshold]);
  return null;
}

/** Colored point cloud: a single `THREE.Points` object (one draw call) built
 * from a `BufferGeometry` with both a `position` and a `color` attribute —
 * the canonical three.js pattern for per-point vertex colors. Each point is
 * pickable (hover + click) via r3f's Points raycast, which reports the hit
 * point's index in `event.index`. */
function PointCloud({
  points,
  onHover,
  onSelect,
}: {
  points: (GalaxyPoint & { displayColor: string })[];
  onHover: (specimenId: string | null) => void;
  onSelect: (point: GalaxyPoint) => void;
}) {
  const pointsRef = useRef<THREE.Points>(null);

  const { positions, colors } = useMemo(() => {
    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);
    const c = new THREE.Color();
    points.forEach((p, i) => {
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;
      c.set(p.displayColor);
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    });
    return { positions, colors };
  }, [points]);

  if (points.length === 0) return null;

  return (
    <points
      ref={pointsRef}
      onPointerMove={(e: ThreeEvent<PointerEvent>) => {
        e.stopPropagation();
        onHover(e.index !== undefined ? points[e.index].specimen_id : null);
      }}
      onPointerOut={(e: ThreeEvent<PointerEvent>) => {
        e.stopPropagation();
        onHover(null);
      }}
      onClick={(e: ThreeEvent<MouseEvent>) => {
        e.stopPropagation();
        if (e.index !== undefined) onSelect(points[e.index]);
      }}
    >
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial vertexColors size={POINT_SIZE} sizeAttenuation toneMapped={false} />
    </points>
  );
}

function Galaxy3D({
  points,
  hoveredPoint,
  onHover,
  onSelect,
}: {
  points: (GalaxyPoint & { displayColor: string })[];
  hoveredPoint: GalaxyPoint | null;
  onHover: (specimenId: string | null) => void;
  onSelect: (point: GalaxyPoint) => void;
}) {
  return (
    <Canvas
      data-testid="galaxy-canvas"
      camera={{ position: [0, 0, 2.4], fov: 50 }}
      dpr={[1, 2]}
      gl={{ antialias: true }}
    >
      <color attach="background" args={["#070b09"]} />
      <ambientLight intensity={1.2} />
      <RaycasterTuning threshold={POINT_PICK_THRESHOLD} />
      <PointCloud points={points} onHover={onHover} onSelect={onSelect} />
      {hoveredPoint && (
        <Html position={[hoveredPoint.x, hoveredPoint.y, hoveredPoint.z]} center distanceFactor={2.2}>
          <div className="galaxy-tooltip">{hoveredPoint.label_name}</div>
        </Html>
      )}
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        minDistance={0.15}
        maxDistance={6}
        autoRotate
        autoRotateSpeed={0.35}
      />
    </Canvas>
  );
}

function Galaxy2D({
  points,
  focusSpecies,
  hoveredId,
  onHover,
  onSelect,
}: {
  points: GalaxyPoint[];
  focusSpecies: string | null;
  hoveredId: string | null;
  onHover: (specimenId: string | null) => void;
  onSelect: (point: GalaxyPoint) => void;
}) {
  const size = 640;
  const toScreen = (v: number) => ((v + 1) / 2) * size;
  return (
    <div className="galaxy-2d-fallback" data-testid="galaxy-2d-fallback">
      <svg
        viewBox={`0 0 ${size} ${size}`}
        role="img"
        aria-label="2D scatter plot of gallery embeddings, colored by species"
        className="galaxy-2d-svg"
      >
        <rect x={0} y={0} width={size} height={size} className="galaxy-2d-bg" />
        {points.map((p) => {
          const dimmed = focusSpecies !== null && p.label_name !== focusSpecies;
          const active = p.specimen_id === hoveredId;
          return (
            <circle
              key={p.specimen_id}
              cx={toScreen(p.x)}
              cy={toScreen(-p.y)}
              r={active ? 5.5 : dimmed ? 2 : 3.4}
              fill={dimmed ? DIM_COLOR : p.color}
              opacity={dimmed ? 0.35 : 1}
              className="galaxy-2d-point"
              onMouseEnter={() => onHover(p.specimen_id)}
              onMouseLeave={() => onHover(null)}
              onClick={() => onSelect(p)}
            >
              <title>{p.label_name}</title>
            </circle>
          );
        })}
      </svg>
      <p className="galaxy-2d-note">
        2D projection (x/y of the same PCA space) — hover a point for its species, click to view
        the specimen.
      </p>
    </div>
  );
}

function SpecimenCard({ point, onClose }: { point: GalaxyPoint; onClose: () => void }) {
  return (
    <div className="galaxy-specimen" data-testid="galaxy-specimen-card">
      <button type="button" className="galaxy-specimen-close" onClick={onClose} aria-label="Close specimen preview">
        ×
      </button>
      <div className="galaxy-specimen-thumb">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`/api/specimen/${encodeURIComponent(point.specimen_id)}/image`}
          alt={point.label_name}
          loading="lazy"
          onError={(e) => {
            e.currentTarget.style.visibility = "hidden";
          }}
        />
      </div>
      <div className="galaxy-specimen-body">
        <span className="galaxy-specimen-dot" style={{ background: point.color }} aria-hidden="true" />
        <h4>{point.label_name}</h4>
        <span className="galaxy-specimen-id">{point.specimen_id}</span>
      </div>
    </div>
  );
}

export function GalaxyPage() {
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<GalaxyData | null>(null);
  const [selected, setSelected] = useState<GalaxyPoint | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [focusSpecies, setFocusSpecies] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"3d" | "2d">("2d");
  const capability = use3DCapability();

  useEffect(() => {
    if (capability !== "checking") setViewMode(capability === "supported" ? "3d" : "2d");
  }, [capability]);

  useEffect(() => {
    const ctrl = new AbortController();
    getGalaxy(ctrl.signal)
      .then((d) => {
        setData(d);
        setStatus("ready");
      })
      .catch((e) => {
        if ((e as Error).name === "AbortError") return;
        setError((e as Error).message);
        setStatus("error");
      });
    return () => ctrl.abort();
  }, []);

  const legend = useMemo(() => (data ? buildLegend(data.points) : []), [data]);

  const coloredPoints = useMemo(
    () =>
      (data?.points ?? []).map((p) => ({
        ...p,
        displayColor: !focusSpecies || p.label_name === focusSpecies ? p.color : DIM_COLOR,
      })),
    [data, focusSpecies],
  );

  const hoveredPoint = useMemo(
    () => (hoveredId ? (data?.points.find((p) => p.specimen_id === hoveredId) ?? null) : null),
    [data, hoveredId],
  );

  if (status === "loading") {
    return (
      <div className="galaxy" data-testid="galaxy-page">
        <div className="state" aria-busy="true">
          <span className="spin" aria-hidden="true" />
          <span className="state-title">Loading the embedding galaxy…</span>
          <p>Fetching the gallery&rsquo;s 3D projection.</p>
        </div>
      </div>
    );
  }

  if (status === "error" || !data) {
    return (
      <div className="galaxy" data-testid="galaxy-page">
        <div className="state error">
          <span className="state-ico" aria-hidden="true">
            <AlertIcon width={26} height={26} />
          </span>
          <span className="state-title">Galaxy unavailable</span>
          <p>{error ?? "Could not load the embedding galaxy. Is the backend running?"}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="galaxy" data-testid="galaxy-page">
      <section className="galaxy-hero">
        <span className="eyebrow">
          <SparkleIcon width={14} height={14} /> Explore the embedding space
        </span>
        <h2>Fly through the embedding galaxy.</h2>
        <p>
          Every gallery specimen as a point in 3D — its 1024-dimensional embedding reduced by PCA
          and colored by species. Orbit, zoom in, hover a point for its name, or click one to see
          the specimen.
        </p>
      </section>

      <div className="galaxy-toolbar">
        <span className="galaxy-count">
          <GridIcon width={14} height={14} aria-hidden="true" /> {data.count.toLocaleString()} specimens
        </span>
        {capability === "supported" && (
          <div className="galaxy-view-toggle" role="group" aria-label="Viz mode">
            <button
              type="button"
              className={viewMode === "3d" ? "active" : ""}
              onClick={() => setViewMode("3d")}
            >
              3D
            </button>
            <button
              type="button"
              className={viewMode === "2d" ? "active" : ""}
              onClick={() => setViewMode("2d")}
            >
              2D
            </button>
          </div>
        )}
        {capability === "unsupported" && (
          <span className="galaxy-note">3D view unavailable here (WebGL or reduced-motion) — showing 2D.</span>
        )}
        {focusSpecies && (
          <button type="button" className="btn btn-ghost galaxy-clear-focus" onClick={() => setFocusSpecies(null)}>
            Clear filter: {focusSpecies}
          </button>
        )}
      </div>

      <div className="galaxy-layout">
        <div className="galaxy-viz-wrap">
          {viewMode === "3d" ? (
            <div className="galaxy-canvas-wrap">
              <Galaxy3D
                points={coloredPoints}
                hoveredPoint={hoveredPoint}
                onHover={setHoveredId}
                onSelect={setSelected}
              />
            </div>
          ) : (
            <Galaxy2D
              points={data.points}
              focusSpecies={focusSpecies}
              hoveredId={hoveredId}
              onHover={setHoveredId}
              onSelect={setSelected}
            />
          )}
        </div>

        <aside className="galaxy-side">
          {selected && <SpecimenCard point={selected} onClose={() => setSelected(null)} />}
          <div className="galaxy-legend">
            <h3>Top species</h3>
            <ul>
              {legend.map((l) => (
                <li key={l.label_name}>
                  <button
                    type="button"
                    className={`galaxy-legend-item ${focusSpecies === l.label_name ? "active" : ""}`}
                    onClick={() => setFocusSpecies((cur) => (cur === l.label_name ? null : l.label_name))}
                    aria-pressed={focusSpecies === l.label_name}
                  >
                    <span className="galaxy-legend-dot" style={{ background: l.color }} aria-hidden="true" />
                    <span className="galaxy-legend-name">{l.label_name}</span>
                    <span className="galaxy-legend-count">{l.count}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </aside>
      </div>
    </div>
  );
}
