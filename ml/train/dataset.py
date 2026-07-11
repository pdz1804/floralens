"""Torch Dataset wrapping precomputed frozen-backbone embeddings.

Training operates on embeddings, not raw images: the backbone is frozen
(CPU-only environment, see PRD §14.5 + build constraints), so every image
(including augmented views, see `ml.scripts.embed_train_subset`) is embedded
once through the frozen backbone ahead of time, and only the small
projection head is trained. This keeps every training epoch a cheap
matrix-multiply over cached vectors instead of a full forward pass through
the vision backbone.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """(embedding, label) pairs backed by numpy arrays."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """Validate shapes and store embeddings/labels as torch tensors.

        Embeddings are cast to float32 (matches the head's parameter dtype)
        and labels to int64 (required by `nn.functional.cross_entropy` /
        label-comparison ops downstream).
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D (n, dim), got shape {embeddings.shape}")
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels must be 1D and match embeddings row count")
        self.embeddings = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        """Number of (embedding, label) pairs, as required by `torch.utils.data.Dataset`."""
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (embedding, label) pair at `idx` for `DataLoader` batching."""
        return self.embeddings[idx], self.labels[idx]
