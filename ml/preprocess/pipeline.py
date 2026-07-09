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
  5. Gentle grey-world white balance — mildly corrects color casts from mixed
     lighting (see `_grey_world_white_balance` for why "gentle" matters).
  6. CLAHE (L channel) — mild contrast-limited adaptive histogram
     equalization, normalizing illumination without blowing out color.

Every step is a pure function of its input (no randomness), so the pipeline
is deterministic: the same input image always produces the same output image
and the same `steps` description list.

Color-preserving white balance (bug fix): a *full-strength* grey-world
correction was found to destroy species-diagnostic color — e.g. it turned a
saturated red rose brown and cast the green background cyan (verified on a
real photo, see `docs/` / commit message). The classic grey-world assumption
("the average scene color is neutral grey") badly misfires on a close-up,
single-hue subject: the algorithm reads the rose's own red as an illumination
cast and "corrects" it away. Since flower color is often the single most
diagnostic feature for species identification, this actively hurt retrieval
quality. The fix (`_WHITE_BALANCE_BLEND_ALPHA`): blend only a small fraction
of the full grey-world correction back with the original pixels, so a mild,
real lighting cast still gets faded out, while a saturated single-color
subject stays recognizably its own color. CLAHE's clip limit was also
lowered for the same reason (less aggressive local contrast stretching).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

DEFAULT_TARGET_SIZE = 320
_WHITE_BALANCE_GAIN_CLIP = (0.5, 2.0)  # guards against wild correction on near-solid-color images
# Fraction of the full grey-world correction actually applied (blended with
# the original pixels). Full-strength grey-world (alpha=1.0) destroyed
# species-diagnostic color on saturated single-hue subjects (see module
# docstring); 0.25 still fades a real mixed-lighting cast while keeping a
# saturated red rose clearly red. Chosen from the documented 0.2-0.3 range.
_WHITE_BALANCE_BLEND_ALPHA = 0.25
_CLAHE_CLIP_LIMIT = 1.2  # lowered from 2.0 — gentler local contrast, less color/edge distortion
_CLAHE_TILE_GRID = (8, 8)

# Ordered pipeline identity — bump/derive automatically from the tunables below.
# The gallery is embedded through this exact preprocessing; if a query is later
# preprocessed with different params, query and gallery embeddings live in
# different spaces. The fingerprint lets the search service detect a stale cache
# (built with an older preprocessing) and fail loud instead of degrading silently.
_PIPELINE_STEPS = (
    "exif_autoorient", "convert_rgb", "resize", "center_crop", "white_balance", "clahe",
)


def preprocess_fingerprint() -> str:
    """Short stable hash of the preprocessing params + step order.

    Changing any tunable that alters output pixels (WB blend, CLAHE clip, gain
    clip, target size, step order) changes this fingerprint, so a re-embed is
    required and a mismatch against a cached one is detectable.
    """
    import hashlib

    payload = repr((
        DEFAULT_TARGET_SIZE,
        _WHITE_BALANCE_GAIN_CLIP,
        round(_WHITE_BALANCE_BLEND_ALPHA, 4),
        round(_CLAHE_CLIP_LIMIT, 4),
        _CLAHE_TILE_GRID,
        _PIPELINE_STEPS,
    ))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


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


def _grey_world_white_balance(
    image: Image.Image, blend_alpha: float = _WHITE_BALANCE_BLEND_ALPHA
) -> Image.Image:
    """Gently rescale each RGB channel toward the overall grey mean.

    Classic grey-world assumption: the average color of a natural scene is
    neutral grey, so any deviation of a channel's mean from the overall mean
    is attributed to an illumination color cast and corrected by a per-
    channel gain (gains clipped so near-solid-color images are not wildly
    over-corrected). That assumption fails for a close-up, saturated single-
    hue subject — e.g. a red rose filling the frame — where the "cast" the
    algorithm detects is the subject's real, diagnostic color, not lighting.
    Applying the full correction there turns a red rose brown and a green
    background cyan (verified empirically on a real photo).

    Fix: only blend in `blend_alpha` of the full correction. A real, mild
    lighting cast (indoor tungsten/fluorescent) still gets meaningfully
    reduced; a saturated single-color subject stays recognizably its own
    color because at most `blend_alpha` of the (wrong) correction is applied.
    """
    arr = np.asarray(image, dtype=np.float32)
    channel_means = arr.reshape(-1, 3).mean(axis=0)
    grey_mean = float(channel_means.mean())
    if grey_mean < 1e-6:
        return image  # degenerate (near-black) image; nothing to correct
    gains = grey_mean / np.clip(channel_means, 1e-6, None)
    gains = np.clip(gains, *_WHITE_BALANCE_GAIN_CLIP)
    corrected = arr * gains
    blended = blend_alpha * corrected + (1.0 - blend_alpha) * arr
    balanced = np.clip(blended, 0, 255).astype(np.uint8)
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
    image: Image.Image,
    target_size: int = DEFAULT_TARGET_SIZE,
    capture_steps: bool = False,
) -> tuple[Image.Image, list[dict[str, str]]] | tuple[Image.Image, list[dict[str, str]], list[Image.Image]]:
    """Run the full deterministic CV preprocessing pipeline.

    Returns (processed_pil_image, steps) where `steps` is a JSON-serializable
    list of `{name, description}` describing each transform actually applied,
    in order — used both internally (embedding pipeline) and by the
    `/api/preprocess-preview` endpoint to show the user what happened to
    their photo.

    If `capture_steps=True`, also returns a third element: a list of PIL
    images, one per entry in `steps`, each holding the image state
    *immediately after* that step ran (for a UI filmstrip showing every
    intermediate transform). Callers that don't need the images keep using
    the plain 2-tuple return (default `capture_steps=False`).
    """
    if image is None:
        raise ValueError("image must not be None")
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    steps: list[PreprocessStep] = []
    step_images: list[Image.Image] = []

    def _record(step: PreprocessStep, current: Image.Image) -> None:
        steps.append(step)
        if capture_steps:
            step_images.append(current.copy())

    img = _exif_autoorient(image)
    _record(
        PreprocessStep(
            "exif_autoorient",
            "Rotated/flipped the image according to its embedded EXIF orientation "
            "tag so it displays upright, then discarded the EXIF metadata.",
        ),
        img,
    )

    img = _to_rgb(img)
    _record(
        PreprocessStep("convert_rgb", "Converted the image to standard 3-channel RGB color."), img
    )

    # Only pathological aspect ratios hit this — normal photos are untouched, so
    # gallery embeddings (all normal-aspect) keep the exact same transform.
    w, h = img.size
    if max(w, h) / max(1, min(w, h)) > _EXTREME_ASPECT:
        img = _pad_to_square(img)
        _record(
            PreprocessStep(
                "pad_extreme_aspect",
                "Padded an extreme-aspect image to a square so the crop keeps its subject.",
            ),
            img,
        )

    img = _resize_longest_side(img, target_size)
    _record(
        PreprocessStep(
            "resize",
            f"Resized the longest edge to {target_size}px (aspect ratio preserved).",
        ),
        img,
    )

    img = _center_square_crop(img)
    _record(
        PreprocessStep(
            "center_crop",
            "Cropped to a centered square so the subject is framed consistently.",
        ),
        img,
    )

    img = _grey_world_white_balance(img)
    _record(
        PreprocessStep(
            "white_balance",
            "Applied a gentle (blended), color-preserving grey-world white balance "
            "to fade mild lighting casts without distorting a saturated subject's "
            "true color.",
        ),
        img,
    )

    img = _clahe_on_luminance(img)
    _record(
        PreprocessStep(
            "clahe",
            "Applied mild contrast-limited adaptive histogram equalization (CLAHE) "
            "to the luminance channel to normalize illumination without distorting color.",
        ),
        img,
    )

    if capture_steps:
        return img, [s.to_dict() for s in steps], step_images
    return img, [s.to_dict() for s in steps]


def preprocess_steps_only() -> list[dict[str, str]]:
    """Static step metadata (no image needed) for the `/api/pipeline` snapshot."""
    tiny = Image.new("RGB", (8, 8), color=(128, 128, 128))
    _, steps = preprocess(tiny)
    return steps
