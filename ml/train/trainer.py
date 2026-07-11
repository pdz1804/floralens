"""Training loop for the projection head: fit on train, early-stop on val Recall@5.

Reuses `ml.eval.harness.run_retrieval_eval` for the val-side metric so
"early stopping on val Recall@5" is computed by the exact same protocol used
for every other reported number (PRD §14.4) — no separate ad-hoc metric.

Device-aware: `train_head` moves the head + loss module + each training batch
onto the resolved device (`ml.device.resolve_device`, `FLORALENS_DEVICE`
env var, default "auto" = cuda if available else cpu). Val-side evaluation
(`evaluate_val_recall5`) infers the device from the head's own parameters, so
it always matches wherever the head currently lives — no separate device
plumbing needed there. Persisted head state dicts are always moved back to
CPU first, so a candidate trained on GPU loads identically on a CPU-only
inference host.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.device import resolve_device
from ml.eval.harness import EmbeddedSpecimen, run_retrieval_eval
from ml.index.vector_store import VectorStore
from ml.train.dataset import EmbeddingDataset
from ml.train.head import ProjectionHead
from ml.train.losses import ArcFaceLoss, TripletLoss

LossName = Literal["arcface", "triplet"]


@dataclass
class TrainConfig:
    """One hyperparameter configuration for `train_head` (PRD §14.5 sweep).

    loss: metric-learning loss, "arcface" or "triplet".
    lr: Adam learning rate for the head + loss module's parameters.
    head_hidden_dim: `ProjectionHead` hidden width (0 = single linear layer).
    output_dim: projected embedding dimensionality.
    margin: angular margin (ArcFace, radians) or distance margin (triplet).
    scale: ArcFace logit scale factor; unused when loss="triplet".
    batch_size: training batch size.
    max_epochs: hard cap on training epochs (must be >= 1).
    early_stop_patience: epochs without a val Recall@5 improvement (beyond
        `early_stop_min_delta`) tolerated before stopping.
    early_stop_min_delta: minimum val Recall@5 gain counted as an improvement.
    seed: seed for Python/NumPy/torch RNGs and the DataLoader shuffle.
    weight_decay: Adam L2 weight decay.
    """

    loss: LossName = "arcface"
    lr: float = 1e-3
    head_hidden_dim: int = 0
    output_dim: int = 256
    margin: float = 0.3
    scale: float = 30.0  # ArcFace logit scale; unused for triplet
    batch_size: int = 128
    max_epochs: int = 40
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.001
    seed: int = 42
    weight_decay: float = 1e-4


@dataclass
class TrainResult:
    """Outcome of `train_head`: the best (early-stopped) checkpoint plus its
    training history.

    config: the `TrainConfig` that produced this result.
    best_epoch: 1-indexed epoch at which `best_val_recall_at_5` was reached.
    best_val_recall_at_5: highest val Recall@5 seen during training.
    final_val_metrics: full val metric dict (all Recall/Precision/mAP@K, MRR)
        at `best_epoch`, not just the recall@5 used for early stopping.
    epoch_curve: per-epoch `{epoch, train_loss, val_recall@5}` records, useful
        for plotting the training curve.
    head_state_dict: CPU-resident `state_dict()` of the head at `best_epoch`,
        ready to persist or load elsewhere.
    """

    config: TrainConfig
    best_epoch: int
    best_val_recall_at_5: float
    final_val_metrics: dict[str, float]
    epoch_curve: list[dict[str, float]] = field(default_factory=list)
    head_state_dict: dict[str, Any] | None = None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _project_and_normalize(head: ProjectionHead, embeddings: np.ndarray) -> np.ndarray:
    # Infer the device from the head's own parameters so this always matches
    # wherever the caller placed the head (cuda during training, cpu when
    # loading a persisted candidate for inference) with no separate device arg.
    device = next(head.parameters()).device
    head.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(device)
        out = head(tensor)
    return out.cpu().numpy().astype(np.float32)


def evaluate_val_recall5(
    head: ProjectionHead,
    gallery_specimens: list[EmbeddedSpecimen],
    val_specimens: list[EmbeddedSpecimen],
) -> dict[str, float]:
    """Project gallery+val backbone embeddings through `head` and run the
    standard retrieval eval protocol. Only ever called with gallery/val
    specimens — never test (enforced by callers via `isolation_guard`).
    """
    gallery_matrix = np.stack([s.vector for s in gallery_specimens], axis=0)
    val_matrix = np.stack([s.vector for s in val_specimens], axis=0)
    projected_gallery = _project_and_normalize(head, gallery_matrix)
    projected_val = _project_and_normalize(head, val_matrix)

    store = VectorStore(model_version="candidate-in-training")
    for spec, vec in zip(gallery_specimens, projected_gallery):
        store.add(spec.id, vec, {"label": spec.label, "label_name": spec.label_name})

    queries = [
        EmbeddedSpecimen(id=spec.id, label=spec.label, label_name=spec.label_name, vector=vec)
        for spec, vec in zip(val_specimens, projected_val)
    ]
    result = run_retrieval_eval(queries, store)
    return result["aggregate"]


def train_head(
    config: TrainConfig,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    num_classes: int,
    gallery_specimens: list[EmbeddedSpecimen],
    val_specimens: list[EmbeddedSpecimen],
    input_dim: int,
    device: torch.device | None = None,
) -> TrainResult:
    """Train one hyperparameter configuration with early stopping on val Recall@5.

    `device` defaults to `ml.device.resolve_device()` (FLORALENS_DEVICE env
    var, "auto" = cuda if available else cpu). The head and loss module train
    on that device; the returned `head_state_dict` is always moved to CPU so
    it loads identically regardless of which device produced it.
    """
    _set_seed(config.seed)
    device = device or resolve_device()
    if config.max_epochs < 1:
        # Fail loud: with zero epochs the training loop below never runs, so
        # there is no checkpoint to return and the val metrics stay unbound.
        raise ValueError(f"max_epochs must be >= 1, got {config.max_epochs}")

    head = ProjectionHead(
        input_dim=input_dim, output_dim=config.output_dim, hidden_dim=config.head_hidden_dim
    ).to(device)
    if config.loss == "arcface":
        criterion: torch.nn.Module = ArcFaceLoss(
            embedding_dim=config.output_dim, num_classes=num_classes, margin=config.margin, scale=config.scale
        ).to(device)
    elif config.loss == "triplet":
        criterion = TripletLoss(margin=config.margin).to(device)
    else:
        raise ValueError(f"unknown loss: {config.loss}")

    params = list(head.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)

    dataset = EmbeddingDataset(train_embeddings, train_labels)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)

    best_val_recall5 = -1.0
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    best_val_metrics: dict[str, float] = {}
    epochs_without_improvement = 0
    epoch_curve: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        head.train()
        criterion.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch_embeddings, batch_labels in loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            projected = head(batch_embeddings)
            loss = criterion(projected, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            num_batches += 1
        mean_loss = epoch_loss / max(num_batches, 1)

        val_metrics = evaluate_val_recall5(head, gallery_specimens, val_specimens)
        val_recall5 = val_metrics["recall@5"]
        epoch_curve.append({"epoch": epoch, "train_loss": mean_loss, "val_recall@5": val_recall5})

        if val_recall5 > best_val_recall5 + config.early_stop_min_delta:
            best_val_recall5 = val_recall5
            best_epoch = epoch
            # Persisted state always moved to CPU: a candidate trained on GPU
            # must load identically on a CPU-only inference host.
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stop_patience:
                break

    if best_state is None:  # pragma: no cover - defensive; unreachable given max_epochs >= 1
        # The first epoch's val Recall@5 (always >= 0) beats the -1.0 sentinel,
        # so a checkpoint is captured on epoch 1 at the latest. Reaching here
        # would mean the loop never executed, which the max_epochs >= 1 guard
        # above already prevents; raise rather than return an empty result.
        raise RuntimeError("training completed without capturing a checkpoint")

    return TrainResult(
        config=config,
        best_epoch=best_epoch,
        best_val_recall_at_5=best_val_recall5,
        final_val_metrics=best_val_metrics,
        epoch_curve=epoch_curve,
        head_state_dict=best_state,
    )
