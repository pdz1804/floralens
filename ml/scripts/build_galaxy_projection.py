"""Reduce the frozen 1024-d gallery embeddings to 3D coordinates for the
"embedding galaxy" fly-through point cloud (PRD US-3 / Phase 4).

Method — PCA (scikit-learn), not t-SNE/UMAP:
  - PCA is a linear, closed-form projection: with `svd_solver="full"` its
    output is fully deterministic for a fixed input matrix (no perplexity,
    iteration count, or random restarts to pin down). t-SNE/UMAP can produce
    visually tighter species clusters, but are stochastic and much slower;
    this viz is illustrative (explore the embedding space, click a point),
    not a metric surface, so exact cluster *distances* aren't load-bearing —
    stable, reproducible positions across rebuilds are what matters (YAGNI).
  - `umap-learn` is not installed in this environment; scikit-learn is
    already a project dependency (eval/calibration code), so PCA introduces
    no new dependency.
  - Coordinates are min-max normalized per axis to [-1, 1] so the web viewer
    can frame the scene with a fixed camera/orbit radius regardless of
    dataset size.

Usage:
    venv/Scripts/python.exe -m ml.scripts.build_galaxy_projection
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from ml.embeddings.cache import load_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_EMBEDDINGS_DIR = "ml/data/embeddings_cache"
DEFAULT_OUT_PATH = Path("ml/data/galaxy_projection.json")
_SEED = 42  # deterministic PCA (full SVD is deterministic regardless; kept for clarity)


def _normalize_axis(values: np.ndarray) -> np.ndarray:
    """Min-max scale one axis to [-1, 1]; a constant axis maps to all zeros."""
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo
    if span == 0:
        return np.zeros_like(values)
    return 2.0 * (values - lo) / span - 1.0


def compute_projection(
    vectors: dict[str, np.ndarray], metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    """Project gallery-split embeddings to normalized 3D points.

    Only `split == "gallery"` specimens are included, matching the set the
    live /api/search vector store indexes (never held-out val/test). Gallery
    ids are sorted first so point order (and therefore the deterministic
    PCA input row order) never depends on dict/JSON iteration order.
    """
    specimens = metadata["specimens"]
    gallery_ids = sorted(sid for sid, meta in specimens.items() if meta["split"] == "gallery")
    if not gallery_ids:
        return []

    matrix = np.stack([vectors[sid] for sid in gallery_ids], axis=0).astype(np.float64)
    n_components = min(3, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full", random_state=_SEED)
    coords = pca.fit_transform(matrix)
    if n_components < 3:
        # Degenerate (tiny) galleries: pad missing axes with zeros rather than
        # failing — the viewer just renders a flatter cloud.
        coords = np.pad(coords, ((0, 0), (0, 3 - n_components)))

    for axis in range(3):
        coords[:, axis] = _normalize_axis(coords[:, axis])

    points: list[dict[str, Any]] = []
    for sid, (x, y, z) in zip(gallery_ids, coords):
        meta = specimens[sid]
        points.append(
            {
                "specimen_id": sid,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "label": meta["label"],
                "label_name": meta["label_name"],
            }
        )
    logger.info(
        "projected %d gallery embeddings to 3D (explained variance ratio: %s)",
        len(points),
        [round(v, 4) for v in pca.explained_variance_ratio_],
    )
    return points


def build_and_save(
    embeddings_dir: str = DEFAULT_EMBEDDINGS_DIR, out_path: Path = DEFAULT_OUT_PATH
) -> list[dict[str, Any]]:
    """Load cached gallery embeddings, project them to 3D, and write the result as JSON.

    Reads vectors/metadata from `embeddings_dir`, computes the normalized 3D
    projection via `compute_projection`, writes it to `out_path` (creating
    parent directories as needed), and returns the same list of points.
    """
    vectors, metadata = load_embeddings(embeddings_dir)
    points = compute_projection(vectors, metadata)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(points, indent=2), encoding="utf-8")
    logger.info("wrote galaxy projection: %s (%d points)", out_path, len(points))
    return points


def load_or_build(
    embeddings_dir: str = DEFAULT_EMBEDDINGS_DIR, out_path: Path = DEFAULT_OUT_PATH
) -> list[dict[str, Any]]:
    """Read the cached projection JSON, building it once if it doesn't exist yet."""
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    return build_and_save(embeddings_dir, out_path)


def main() -> None:
    """Build the embedding galaxy projection and write it to ml/data/galaxy_projection.json."""
    build_and_save()


if __name__ == "__main__":
    main()
