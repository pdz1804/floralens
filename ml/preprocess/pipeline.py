"""Deterministic CV preprocessing pipeline applied to every image before it
reaches the embedding backbone — both when building the gallery/train/val/
test embedding caches and on the live `/api/search` query path (PRD A1:
"Validate type/size; strip EXIF").

User-submitted photos are frequently mis-oriented (phone EXIF rotation),
color-cast (indoor tungsten/fluorescent lighting), off-center, or inconsistently
sized. This module fixes those defects with a fixed, ordered sequence of
transforms so every embedded image (gallery, train, val, test, and live
queries) goes through identical, reproducible preprocessing:

  1. EXIF auto-orient  — apply the EXIF Orientation tag, then drop EXIF.
  2. Convert to RGB    — normalize color mode (drop alpha/palette/CMYK etc).
  3. Resize            — longest side to `target_size`, aspect preserved.
  4. Center square crop — consistent square framing for the backbone.
  5. Grey-world white balance — corrects color casts from mixed lighting.
  6. CLAHE (L channel) — contrast-limited adaptive histogram equalization,
     normalizing illumination without blowing out color.

Every step is a pure function of its input (no randomness), so the pipeline
is deterministic: the same input image always produces the same output image
and the same `steps` description list.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

DEFAULT_TARGET_SIZE = 320
_WHITE_BALANCE_GAIN_CLIP = (0.5, 2.0)  # guards against wild correction on near-solid-color images
_CLAHE_CLIP_LIMIT = 2.0
_CLAHE_TILE_GRID = (8, 8)


@dataclass(frozen=True)
class PreprocessStep:
    name: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _exif_autoorient(image: Image.Image) -> Image.Image:
    oriented = ImageOps.exif_transpose(image)
    return oriented if oriented is not None else image


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def _resize_longest_side(image: Image.Image, target_size: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest == 0:
        raise ValueError("image has zero-sized dimension")
    scale = target_size / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.LANCZOS)


# Above this width:height (or height:width) ratio, a longest-side resize +
# center crop would discard almost all content (a 1x1000 strip collapses to
# ~1x1). Real photos never reach this; only pathological inputs do.
_EXTREME_ASPECT = 4.0


def _pad_to_square(image: Image.Image) -> Image.Image:
    """Center the image on a black square canvas (max side) so a later square
    crop keeps its content instead of collapsing an extreme-aspect strip."""
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def _center_square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def _grey_world_white_balance(image: Image.Image) -> Image.Image:
    """Rescale each RGB channel so its mean matches the overall grey mean.

    Classic grey-world assumption: the average color of a natural scene is
    neutral grey, so any deviation of a channel's mean from the overall mean
    is attributed to an illumination color cast and corrected by a per-
    channel gain. Gains are clipped so near-solid-color images (e.g. a
    single large red petal filling the frame) are not over-corrected.
    """
    arr = np.asarray(image, dtype=np.float32)
    channel_means = arr.reshape(-1, 3).mean(axis=0)
    grey_mean = float(channel_means.mean())
    if grey_mean < 1e-6:
        return image  # degenerate (near-black) image; nothing to correct
    gains = grey_mean / np.clip(channel_means, 1e-6, None)
    gains = np.clip(gains, *_WHITE_BALANCE_GAIN_CLIP)
    balanced = np.clip(arr * gains, 0, 255).astype(np.uint8)
    return Image.fromarray(balanced, mode="RGB")


def _clahe_on_luminance(image: Image.Image) -> Image.Image:
    """Apply CLAHE to the L channel of Lab color space, preserving color (a/b)."""
    arr = np.asarray(image)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP_LIMIT, tileGridSize=_CLAHE_TILE_GRID)
    l_equalized = clahe.apply(l_channel)
    lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))
    rgb = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb, mode="RGB")


def preprocess(
    image: Image.Image, target_size: int = DEFAULT_TARGET_SIZE
) -> tuple[Image.Image, list[dict[str, str]]]:
    """Run the full deterministic CV preprocessing pipeline.

    Returns (processed_pil_image, steps) where `steps` is a JSON-serializable
    list of `{name, description}` describing each transform actually applied,
    in order — used both internally (embedding pipeline) and by the
    `/api/preprocess-preview` endpoint to show the user what happened to
    their photo.
    """
    if image is None:
        raise ValueError("image must not be None")
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    steps: list[PreprocessStep] = []

    img = _exif_autoorient(image)
    steps.append(
        PreprocessStep(
            "exif_autoorient",
            "Rotated/flipped the image according to its embedded EXIF orientation "
            "tag so it displays upright, then discarded the EXIF metadata.",
        )
    )

    img = _to_rgb(img)
    steps.append(
        PreprocessStep("convert_rgb", "Converted the image to standard 3-channel RGB color.")
    )

    # Only pathological aspect ratios hit this — normal photos are untouched, so
    # gallery embeddings (all normal-aspect) keep the exact same transform.
    w, h = img.size
    if max(w, h) / max(1, min(w, h)) > _EXTREME_ASPECT:
        img = _pad_to_square(img)
        steps.append(
            PreprocessStep(
                "pad_extreme_aspect",
                "Padded an extreme-aspect image to a square so the crop keeps its subject.",
            )
        )

    img = _resize_longest_side(img, target_size)
    steps.append(
        PreprocessStep(
            "resize",
            f"Resized the longest edge to {target_size}px (aspect ratio preserved).",
        )
    )

    img = _center_square_crop(img)
    steps.append(
        PreprocessStep(
            "center_crop",
            "Cropped to a centered square so the subject is framed consistently.",
        )
    )

    img = _grey_world_white_balance(img)
    steps.append(
        PreprocessStep(
            "white_balance",
            "Applied grey-world white balance to correct color casts from indoor/"
            "outdoor lighting.",
        )
    )

    img = _clahe_on_luminance(img)
    steps.append(
        PreprocessStep(
            "clahe",
            "Applied contrast-limited adaptive histogram equalization (CLAHE) to "
            "the luminance channel to normalize illumination without distorting color.",
        )
    )

    return img, [s.to_dict() for s in steps]


def preprocess_steps_only() -> list[dict[str, str]]:
    """Static step metadata (no image needed) for the `/api/pipeline` snapshot."""
    tiny = Image.new("RGB", (8, 8), color=(128, 128, 128))
    _, steps = preprocess(tiny)
    return steps
