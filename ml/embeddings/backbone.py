"""Frozen embedding backbone.

Primary: OpenCLIP ViT-B-32 (pretrained laion2b_s34b_b79k). Falls back to
torchvision resnet50 (ImageNet DEFAULT weights, penultimate 2048-d features)
if the OpenCLIP weights cannot be downloaded. Both paths expose the same
`embed_image(pil_image) -> np.ndarray` interface (L2-normalized, deterministic).

The backbone is frozen (no gradient updates) — Phase 1 only computes and uses
embeddings, it does not train anything, so CPU-only inference is fine.
"""
from __future__ import annotations

import functools
import logging
import threading
from typing import Callable

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_OPEN_CLIP_MODEL_NAME = "ViT-B-32"
_OPEN_CLIP_PRETRAINED = "laion2b_s34b_b79k"
_OPEN_CLIP_CACHE_DIR = "ml/data/raw/open_clip_cache"

_lock = threading.Lock()


class _OpenClipBackbone:
    """Wraps open_clip ViT-B-32; embeds via the visual tower."""

    name = "open_clip_vit_b32_laion2b"
    embedding_dim = 512

    def __init__(self) -> None:
        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            _OPEN_CLIP_MODEL_NAME,
            pretrained=_OPEN_CLIP_PRETRAINED,
            cache_dir=_OPEN_CLIP_CACHE_DIR,
        )
        self.model.eval()
        torch.manual_seed(0)

    @torch.no_grad()
    def embed(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image).unsqueeze(0)
        features = self.model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).numpy().astype(np.float32)


class _ResNet50Backbone:
    """Fallback: torchvision resnet50 penultimate (2048-d) features."""

    name = "resnet50_imagenet_penultimate"
    embedding_dim = 2048

    def __init__(self) -> None:
        from torchvision.models import ResNet50_Weights, resnet50
        from torchvision.transforms import Compose

        weights = ResNet50_Weights.DEFAULT
        base_model = resnet50(weights=weights)
        base_model.eval()
        # Strip the final classification layer -> penultimate 2048-d pooled features.
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.preprocess: Compose = weights.transforms()
        torch.manual_seed(0)

    @torch.no_grad()
    def embed(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image).unsqueeze(0)
        features = self.model(tensor).flatten(1)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).numpy().astype(np.float32)


@functools.lru_cache(maxsize=1)
def _get_backbone():
    """Lazily build and cache the process-wide backbone (thread-safe, singleton)."""
    with _lock:
        try:
            backbone = _OpenClipBackbone()
            logger.info("Loaded embedding backbone: %s", backbone.name)
            return backbone
        except Exception as exc:  # network/weights unavailable -> fallback
            logger.warning(
                "OpenCLIP backbone unavailable (%s); falling back to resnet50.", exc
            )
            backbone = _ResNet50Backbone()
            logger.info("Loaded embedding backbone: %s", backbone.name)
            return backbone


def backbone_name() -> str:
    """Return the active backbone identifier (used for model_version tagging)."""
    return _get_backbone().name


def embedding_dim() -> int:
    """Return the embedding vector dimensionality of the active backbone."""
    return _get_backbone().embedding_dim


def embed_image(image: Image.Image) -> np.ndarray:
    """Embed a PIL image into an L2-normalized float32 vector.

    Deterministic for a given input (frozen weights, eval mode, no dropout/
    randomness in the forward pass). Raises ValueError on invalid input.
    """
    if image is None:
        raise ValueError("image must not be None")
    if image.mode != "RGB":
        image = image.convert("RGB")
    backbone = _get_backbone()
    return backbone.embed(image)


def reset_backbone_cache() -> None:
    """Test helper: clears the cached backbone singleton."""
    _get_backbone.cache_clear()
