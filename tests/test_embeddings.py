"""Unit tests for the frozen embedding backbone (ml.embeddings.backbone)."""
import numpy as np
from PIL import Image

from ml.embeddings.backbone import embed_image, embedding_dim


def _synthetic_image(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_embed_image_shape_and_normalization():
    img = _synthetic_image(0)
    vector = embed_image(img)
    assert vector.dtype == np.float32
    assert vector.shape == (embedding_dim(),)
    norm = np.linalg.norm(vector)
    assert abs(norm - 1.0) < 1e-4


def test_embed_image_deterministic_for_same_input():
    img = _synthetic_image(1)
    v1 = embed_image(img)
    v2 = embed_image(img)
    assert np.allclose(v1, v2, atol=1e-6)


def test_embed_image_different_inputs_differ():
    v1 = embed_image(_synthetic_image(2))
    v2 = embed_image(_synthetic_image(3))
    assert not np.allclose(v1, v2, atol=1e-3)


def test_embed_image_handles_non_rgb_mode():
    img = _synthetic_image(4).convert("L")  # grayscale, backbone must convert internally
    vector = embed_image(img)
    assert vector.shape == (embedding_dim(),)
