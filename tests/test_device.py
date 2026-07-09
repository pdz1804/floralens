"""Unit tests for device resolution (ml.device) and device-aware embedding."""
import pytest
import torch

from ml.device import resolve_device


def test_auto_resolves_to_cuda_when_available():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected


def test_default_preference_is_auto(monkeypatch):
    monkeypatch.delenv("FLORALENS_DEVICE", raising=False)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device().type == expected


def test_explicit_cpu_always_returns_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_explicit_cuda_returns_cuda_or_raises_if_unavailable():
    if torch.cuda.is_available():
        assert resolve_device("cuda").type == "cuda"
    else:
        with pytest.raises(RuntimeError):
            resolve_device("cuda")


def test_env_var_is_read_when_no_explicit_preference(monkeypatch):
    monkeypatch.setenv("FLORALENS_DEVICE", "cpu")
    assert resolve_device().type == "cpu"


def test_rejects_unknown_preference():
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_case_and_whitespace_insensitive():
    assert resolve_device(" CPU ").type == "cpu"


def test_embed_runs_on_resolved_device():
    """The active backbone singleton must live on the same device resolve_device()
    picks (verifies the backbone actually wires the device through, not just
    that resolve_device() itself works)."""
    from ml.embeddings.backbone import device_str, embed_image
    from PIL import Image

    img = Image.new("RGB", (48, 48), color=(90, 140, 60))
    vector = embed_image(img)  # exercises the real forward pass end to end
    assert vector.shape[0] > 0

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert device_str() == expected
