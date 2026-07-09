"""Metric-learning losses for the projection head (PRD §14.5, §18.3: ArcFace first).

`ArcFaceLoss` is the primary loss (additive angular margin, ArcFace paper,
Deng et al. 2019): it treats retrieval as a closed-set classification problem
over the known 102 classes, pulling each class's embeddings toward a learned
per-class direction with an angular margin, which sharpens inter-class
separation on the unit hypersphere — exactly what cosine-similarity retrieval
rewards.

`TripletLoss` (batch-hard mining) is the PRD-mandated comparison experiment
(§14.5 step 2): for each anchor in a batch, mine the hardest positive
(same class, lowest similarity) and hardest negative (different class,
highest similarity) present in that batch, and penalize when the
anchor-negative similarity is not smaller than the anchor-positive
similarity by at least `margin`.
"""
from __future__ import annotations

import torch
from torch import nn


class ArcFaceLoss(nn.Module):
    """Additive angular margin loss over L2-normalized embeddings.

    Holds its own learnable per-class weight matrix (the "class centers");
    training jointly shapes the projection head and these centers.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.3,
        scale: float = 30.0,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or num_classes <= 0:
            raise ValueError("embedding_dim and num_classes must be positive")
        if margin < 0:
            raise ValueError("margin must be >= 0")
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.01)

        self.cos_m = float(torch.cos(torch.tensor(margin)))
        self.sin_m = float(torch.sin(torch.tensor(margin)))
        # Threshold below which cos(theta + m) stops being monotonically
        # decreasing in theta (standard ArcFace numerical-stability guard).
        self.threshold = float(torch.cos(torch.pi - torch.tensor(margin)))
        self.mm = self.sin_m * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() != 2 or embeddings.shape[-1] != self.weight.shape[1]:
            raise ValueError(
                f"expected embeddings shape (batch, {self.weight.shape[1]}), got {tuple(embeddings.shape)}"
            )
        if labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels batch size must match embeddings batch size")

        normalized_weight = nn.functional.normalize(self.weight, p=2, dim=-1)
        cosine = nn.functional.linear(embeddings, normalized_weight)  # (batch, num_classes), already in [-1,1]
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.scale
        return nn.functional.cross_entropy(logits, labels.long())


class TripletLoss(nn.Module):
    """Batch-hard triplet margin loss over cosine distance (1 - cosine_sim)."""

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        if margin < 0:
            raise ValueError("margin must be >= 0")
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() != 2:
            raise ValueError(f"expected embeddings shape (batch, dim), got {tuple(embeddings.shape)}")
        if labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels batch size must match embeddings batch size")

        # embeddings are L2-normalized by the head, so cosine similarity is a plain dot product.
        sim = embeddings @ embeddings.t()  # (batch, batch)
        dist = 1.0 - sim

        labels = labels.view(-1, 1)
        same_class = labels.eq(labels.t())
        diff_class = ~same_class
        self_mask = torch.eye(labels.shape[0], dtype=torch.bool, device=embeddings.device)
        positive_mask = same_class & ~self_mask

        # Batch-hard mining: for each anchor, the farthest same-class point and
        # the nearest different-class point present in this batch.
        hardest_positive = torch.where(positive_mask, dist, torch.full_like(dist, -1.0)).max(dim=1).values
        hardest_negative = torch.where(diff_class, dist, torch.full_like(dist, float("inf"))).min(dim=1).values

        # Anchors with no positive in the batch (can happen with a very small
        # per-class draw) contribute no loss rather than a spurious signal.
        valid = positive_mask.any(dim=1) & diff_class.any(dim=1)
        if not valid.any():
            return embeddings.sum() * 0.0  # zero, but keeps the graph connected for backward()

        losses = torch.relu(hardest_positive - hardest_negative + self.margin)
        return losses[valid].mean()
