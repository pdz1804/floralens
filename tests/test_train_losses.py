"""Unit tests for ArcFace and Triplet losses: known-input shape + gradient sanity."""
import torch

from ml.train.head import ProjectionHead
from ml.train.losses import ArcFaceLoss, TripletLoss


def test_arcface_loss_scalar_and_gradients_flow():
    torch.manual_seed(0)
    head = ProjectionHead(input_dim=32, output_dim=16, hidden_dim=0)
    criterion = ArcFaceLoss(embedding_dim=16, num_classes=4, margin=0.3, scale=30.0)

    x = torch.randn(8, 32)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    embeddings = head(x)
    loss = criterion(embeddings, labels)

    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert head.net.weight.grad is not None
    assert torch.any(head.net.weight.grad != 0)
    assert criterion.weight.grad is not None
    assert torch.any(criterion.weight.grad != 0)


def test_arcface_perfect_alignment_gives_lower_loss_than_random():
    """A class weight matrix aligned with the embeddings should score higher
    (lower loss) than a randomly-initialized one for the same labels."""
    torch.manual_seed(1)
    num_classes, dim = 3, 8
    embeddings = torch.eye(num_classes, dim)  # class i's "embedding" is the i-th basis-ish vector
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    labels = torch.tensor([0, 1, 2])

    aligned = ArcFaceLoss(embedding_dim=dim, num_classes=num_classes, margin=0.3, scale=30.0)
    with torch.no_grad():
        aligned.weight.copy_(embeddings)
    aligned_loss = aligned(embeddings, labels).item()

    misaligned = ArcFaceLoss(embedding_dim=dim, num_classes=num_classes, margin=0.3, scale=30.0)
    with torch.no_grad():
        misaligned.weight.copy_(torch.randn(num_classes, dim))
    misaligned_loss = misaligned(embeddings, labels).item()

    assert aligned_loss < misaligned_loss


def test_arcface_rejects_mismatched_shapes():
    criterion = ArcFaceLoss(embedding_dim=16, num_classes=4)
    import pytest

    with pytest.raises(ValueError):
        criterion(torch.randn(4, 8), torch.tensor([0, 1, 2, 3]))
    with pytest.raises(ValueError):
        criterion(torch.randn(4, 16), torch.tensor([0, 1, 2]))


def test_triplet_loss_scalar_and_gradients_flow():
    torch.manual_seed(0)
    head = ProjectionHead(input_dim=32, output_dim=16, hidden_dim=0)
    criterion = TripletLoss(margin=0.3)

    x = torch.randn(9, 32)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    embeddings = head(x)
    loss = criterion(embeddings, labels)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert head.net.weight.grad is not None
    assert torch.any(head.net.weight.grad != 0)


def test_triplet_loss_zero_when_perfectly_separated_beyond_margin():
    """If every same-class pair is closer than every different-class pair by
    more than the margin, the batch-hard triplet loss should be exactly 0."""
    margin = 0.1
    # Two tight clusters far apart on the unit sphere.
    a = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.99, 0.01]]), dim=-1)
    b = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0], [0.01, 0.99]]), dim=-1)
    embeddings = torch.cat([a, b], dim=0)
    labels = torch.tensor([0, 0, 1, 1])

    criterion = TripletLoss(margin=margin)
    loss = criterion(embeddings, labels)
    assert loss.item() == 0.0
