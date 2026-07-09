"""Unit tests for the projection head: fixed-dim, L2-normalized output."""
import numpy as np
import pytest
import torch

from ml.train.head import ProjectionHead


def test_forward_shape_and_l2_norm_linear():
    head = ProjectionHead(input_dim=512, output_dim=128, hidden_dim=0)
    x = torch.randn(10, 512)
    out = head(x)
    assert out.shape == (10, 128)
    norms = out.norm(dim=-1).detach().numpy()
    np.testing.assert_allclose(norms, np.ones(10), atol=1e-5)


def test_forward_shape_and_l2_norm_mlp():
    head = ProjectionHead(input_dim=512, output_dim=64, hidden_dim=256)
    x = torch.randn(5, 512)
    out = head(x)
    assert out.shape == (5, 64)
    norms = out.norm(dim=-1).detach().numpy()
    np.testing.assert_allclose(norms, np.ones(5), atol=1e-5)


def test_rejects_wrong_input_dim():
    head = ProjectionHead(input_dim=512, output_dim=64)
    with pytest.raises(ValueError):
        head(torch.randn(4, 256))


def test_rejects_invalid_dims():
    with pytest.raises(ValueError):
        ProjectionHead(input_dim=0, output_dim=64)
    with pytest.raises(ValueError):
        ProjectionHead(input_dim=512, output_dim=-1)
    with pytest.raises(ValueError):
        ProjectionHead(input_dim=512, output_dim=64, hidden_dim=-1)
