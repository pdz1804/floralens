"""Load a persisted candidate ModelVersion (projection head + metadata) and
project raw backbone embeddings through it. Shared by the Phase 3b test-eval,
calibration, and promotion-gate scripts so the load/project logic exists in
exactly one place.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.train.head import ProjectionHead


def load_candidate_head(candidate_dir: str | Path) -> tuple[ProjectionHead, dict[str, Any]]:
    """Load a persisted candidate's projection head + its metadata.json.

    Reconstructs the `ProjectionHead` with the exact dims recorded at train
    time (`input_dim`, `output_dim`, `head_hidden_dim`), loads the saved
    `head.pt` state dict onto CPU, and puts the head in eval mode (no
    dropout/BN train-time behavior) so it is immediately safe for inference.
    """
    candidate_dir = Path(candidate_dir)
    metadata = json.loads((candidate_dir / "metadata.json").read_text(encoding="utf-8"))
    head = ProjectionHead(
        input_dim=metadata["input_dim"], output_dim=metadata["output_dim"], hidden_dim=metadata["head_hidden_dim"]
    )
    state_dict = torch.load(candidate_dir / "head.pt", map_location="cpu")
    head.load_state_dict(state_dict)
    head.eval()
    return head, metadata


def project_embeddings(head: ProjectionHead, vectors: np.ndarray) -> np.ndarray:
    """Run a batch of raw backbone embeddings through the (eval-mode) head."""
    with torch.no_grad():
        out = head(torch.from_numpy(np.asarray(vectors, dtype=np.float32)))
    return out.numpy().astype(np.float32)
