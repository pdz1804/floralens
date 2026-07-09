"""Frozen embedding backbone.

Primary target: **DINOv3** (ViT-L/16, `facebook/dinov3-vitl16-pretrain-lvd1689m`
via Hugging Face `transformers`). DINOv3 checkpoints on the Hub are gated
("manual" access review) — verified directly against this environment: an
unauthenticated `from_pretrained()` call returns HTTP 401 "You are trying to
access a gated repo", and no `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` is configured
here. Per the task spec's documented fallback, the active backbone therefore
becomes **DINOv2 ViT-L/14** (`torch.hub.load('facebookresearch/dinov2',
'dinov2_vitl14')`) — same self-supervised DINO family, openly downloadable,
no gating. If DINOv3 access is granted later (an `HF_TOKEN` with approved
access to the gated repo), the chain below picks it up automatically with no
code change; nothing else in the pipeline depends on which DINO generation
is active.

Both DINOv3 and DINOv2 paths, plus the pre-existing OpenCLIP/ResNet50
fallbacks (kept as a last-resort safety net if the whole DINO family is
unreachable, e.g. fully offline), expose the same `embed_image(pil_image) ->
np.ndarray` interface (L2-normalized, deterministic).

CPU-only by design: this environment's GPU has only 4GB VRAM and already
backs a separate live service; reinstalling a CUDA-enabled torch build here
would risk destabilizing that service's environment. All backbones run on
CPU (`torch.cuda.is_available()` is not used). A CUDA build (e.g. torch
2.7.1+cu121) is a documented **future optimization**, not attempted here.

The backbone is frozen (no gradient updates) — only a small projection head
(`ml.train.head`) is ever trained on top of its cached output vectors, so
CPU-only inference is acceptable for both embedding and evaluation.
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

_DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
_DINOV2_HUB_REPO = "facebookresearch/dinov2"
_DINOV2_HUB_MODEL = "dinov2_vitl14"
_OPEN_CLIP_MODEL_NAME = "ViT-B-32"
_OPEN_CLIP_PRETRAINED = "laion2b_s34b_b79k"
_OPEN_CLIP_CACHE_DIR = "ml/data/raw/open_clip_cache"

_lock = threading.Lock()


class _Dinov3Backbone:
    """Primary target backbone: DINOv3 ViT-L/16 (gated on HF Hub; see module
    docstring). Loads via `transformers.AutoModel`/`AutoImageProcessor` so it
    picks up the model's own documented preprocessing (resize/crop/normalize)
    rather than us guessing DINOv3-specific constants."""

    name = "dinov3_vitl16"
    embedding_dim = 1024  # ViT-L hidden size

    def __init__(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(_DINOV3_MODEL_ID)
        self.model = AutoModel.from_pretrained(_DINOV3_MODEL_ID)
        self.model.eval()
        torch.manual_seed(0)

    @torch.no_grad()
    def embed(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]  # CLS token fallback
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        return pooled.squeeze(0).numpy().astype(np.float32)


class _Dinov2Backbone:
    """Documented fallback (task-mandated): DINOv2 ViT-L/14 via torch.hub.
    Used because DINOv3 weights are gated and unreachable in this
    environment without an approved HF token (verified 401 above)."""

    name = "dinov2_vitl14"
    embedding_dim = 1024

    def __init__(self) -> None:
        import os

        from torchvision import transforms

        # Load from the LOCAL torch.hub cache when present — `torch.hub.load`
        # otherwise makes a GitHub round-trip to validate the repo ref on every
        # load, and a transient network error there (e.g. HTTP 504) would
        # silently drop us to a weaker backbone whose vectors don't match the
        # DINOv2-embedded gallery. Weights come from the (cached) checkpoint,
        # so once downloaded this is fully offline.
        cache_dir = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
        if os.path.isdir(cache_dir):
            self.model = torch.hub.load(
                cache_dir, _DINOV2_HUB_MODEL, source="local", trust_repo=True
            )
        else:
            self.model = torch.hub.load(
                _DINOV2_HUB_REPO, _DINOV2_HUB_MODEL, trust_repo=True, skip_validation=True
            )
        self.model.eval()
        # Standard DINOv2 eval-time preprocessing (matches the reference repo):
        # resize short-side-preserving to 256 via bicubic, center-crop 224,
        # ImageNet mean/std normalization.
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        torch.manual_seed(0)

    @torch.no_grad()
    def embed(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image).unsqueeze(0)
        features = self.model(tensor)  # (1, 1024) CLS-token global feature
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).numpy().astype(np.float32)


class _OpenClipBackbone:
    """Further fallback (pre-existing, kept for robustness): OpenCLIP ViT-B-32."""

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
    """Last-resort fallback: torchvision resnet50 penultimate (2048-d) features."""

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


# Ordered by preference: DINOv3 (primary target) -> DINOv2 (documented
# fallback) -> OpenCLIP -> ResNet50 (last-resort, e.g. fully offline).
_BACKBONE_CHAIN: tuple[tuple[str, Callable[[], object]], ...] = (
    ("DINOv3 ViT-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m)", _Dinov3Backbone),
    ("DINOv2 ViT-L/14 (torch.hub facebookresearch/dinov2 — documented fallback)", _Dinov2Backbone),
    ("OpenCLIP ViT-B-32 laion2b (further fallback)", _OpenClipBackbone),
    ("torchvision resnet50 ImageNet (last-resort fallback)", _ResNet50Backbone),
)


@functools.lru_cache(maxsize=1)
def _get_backbone():
    """Lazily build and cache the process-wide backbone (thread-safe, singleton).

    Walks `_BACKBONE_CHAIN` in preference order; the first backbone that
    loads successfully (weights reachable, no exception) becomes active.
    """
    with _lock:
        last_exc: Exception | None = None
        for label, backbone_cls in _BACKBONE_CHAIN:
            try:
                backbone = backbone_cls()
                logger.info("Loaded embedding backbone: %s (%s)", backbone.name, label)
                return backbone
            except Exception as exc:  # network/weights/gating unavailable -> try next
                logger.warning("%s unavailable (%s); trying next backbone in the chain.", label, exc)
                last_exc = exc
        raise RuntimeError("no embedding backbone in the fallback chain could be loaded") from last_exc


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
