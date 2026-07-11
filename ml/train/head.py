"""Projection head: a small trainable module on top of the frozen backbone.

The backbone (ml.embeddings.backbone) stays frozen throughout Phase 3 — CPU
training makes unfreezing it impractical, and PRD §14.5 only requires a
"projection head + metric-learning loss" on top of it. This module is the
only trainable component: it maps a frozen backbone embedding to a new
embedding space shaped by the metric-learning loss, L2-normalized so cosine
similarity remains the retrieval scoring function end to end.
"""
from __future__ import annotations

import torch
from torch import nn


class ProjectionHead(nn.Module):
    """Maps a frozen backbone embedding to an L2-normalized output embedding.

    `hidden_dim=0` collapses to a single linear layer (a pure metric-learning
    re-projection); `hidden_dim>0` inserts one ReLU hidden layer for extra
    capacity. Both are swept as a hyperparameter in the training script.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 0) -> None:
        """Build the projection network: a single `Linear` when `hidden_dim=0`,
        else `Linear -> ReLU -> Linear` for extra capacity.
        """
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if hidden_dim < 0:
            raise ValueError("hidden_dim must be >= 0")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        if hidden_dim == 0:
            self.net = nn.Linear(input_dim, output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project a batch of backbone embeddings and L2-normalize the output.

        `x` has shape (batch, input_dim); the return has shape
        (batch, output_dim) with unit norm along the last dim, so downstream
        cosine similarity (used by both the losses and the vector store)
        reduces to a plain dot product.
        """
        if x.dim() != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input shape (batch, {self.input_dim}), got {tuple(x.shape)}")
        out = self.net(x)
        return nn.functional.normalize(out, p=2, dim=-1)
