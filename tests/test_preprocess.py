"""Unit tests for the CV preprocessing pipeline (ml.preprocess.pipeline)."""
import colorsys
import io

import numpy as np
import pytest
from PIL import Image

from ml.preprocess.pipeline import preprocess


def _solid_image(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color=color)


def test_extreme_aspect_does_not_collapse():
    # A 1x1000 strip must not collapse to a ~1x1 image (content destroyed);
    # the extreme-aspect pad keeps it a usable square.
    out, steps = preprocess(Image.new("RGB", (1, 1000), (200, 30, 30)))
    assert min(out.size) >= 64  # not degenerate
    assert out.size[0] == out.size[1]  # square
    assert any(s["name"] == "pad_extreme_aspect" for s in steps)


def test_normal_aspect_keeps_standard_steps():
    # A normal photo must NOT trigger the extreme-aspect pad — gallery images
    # (all normal aspect) keep the exact same transform / embedding space.
    _, steps = preprocess(Image.new("RGB", (400, 300), (120, 140, 90)))
    names = [s["name"] for s in steps]
    assert "pad_extreme_aspect" not in names
    assert names == ["exif_autoorient", "convert_rgb", "resize", "center_crop",
                     "white_balance", "clahe"]


def _color_cast_image(size: tuple[int, int] = (120, 90)) -> Image.Image:
    rng = np.random.default_rng(7)
    base = rng.integers(80, 180, size=(size[1], size[0], 3), dtype=np.uint8).astype(np.int16)
    # Strong uniform red cast layered on top of noise so the image isn't degenerate.
    base[:, :, 0] = np.clip(base[:, :, 0] + 70, 0, 255)
    base[:, :, 1] = np.clip(base[:, :, 1] - 40, 0, 255)
    base[:, :, 2] = np.clip(base[:, :, 2] - 40, 0, 255)
    return Image.fromarray(base.astype(np.uint8), mode="RGB")


def _exif_rotated_image() -> Image.Image:
    """Top-red/bottom-blue image tagged with EXIF orientation=3 (180 deg)."""
    arr = np.zeros((80, 80, 3), dtype=np.uint8)
    arr[:40, :, 0] = 255  # top half red
    arr[40:, :, 2] = 255  # bottom half blue
    img = Image.fromarray(arr, mode="RGB")
    exif = img.getexif()
    exif[0x0112] = 3  # Orientation tag: rotate 180
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    buf.seek(0)
    return Image.open(buf)


def test_preprocess_returns_non_empty_steps():
    img = _solid_image((100, 100), (120, 130, 140))
    _out, steps = preprocess(img)
    assert len(steps) > 0
    for step in steps:
        assert "name" in step and "description" in step
        assert step["name"] and step["description"]


def test_preprocess_is_deterministic():
    img = _color_cast_image()
    out1, steps1 = preprocess(img)
    out2, steps2 = preprocess(img)
    assert steps1 == steps2
    assert np.array_equal(np.asarray(out1), np.asarray(out2))


def test_preprocess_output_is_square_rgb():
    img = _solid_image((300, 120), (10, 200, 50))
    out, _steps = preprocess(img, target_size=64)
    assert out.mode == "RGB"
    assert out.size[0] == out.size[1]


def test_exif_autoorient_fixes_rotated_image():
    loaded = _exif_rotated_image()
    assert loaded.getexif().get(0x0112) == 3

    out, steps = preprocess(loaded)
    step_names = [s["name"] for s in steps]
    assert "exif_autoorient" in step_names

    arr = np.asarray(out)
    top_mean = arr[: arr.shape[0] // 2].reshape(-1, 3).mean(axis=0)
    bottom_mean = arr[arr.shape[0] // 2 :].reshape(-1, 3).mean(axis=0)
    # EXIF orientation=3 means the raw pixels (top=red/bottom=blue) must be
    # rotated 180 degrees to display correctly -> after correction, the top
    # half should read blue-dominant and the bottom half red-dominant.
    assert top_mean[2] > top_mean[0]  # top: blue channel dominates
    assert bottom_mean[0] > bottom_mean[2]  # bottom: red channel dominates


def test_color_normalization_reduces_color_cast():
    img = _color_cast_image()
    before_means = np.asarray(img, dtype=np.float64).reshape(-1, 3).mean(axis=0)
    out, _steps = preprocess(img)
    after_means = np.asarray(out, dtype=np.float64).reshape(-1, 3).mean(axis=0)

    before_spread = float(before_means.max() - before_means.min())
    after_spread = float(after_means.max() - after_means.min())
    assert after_spread < before_spread


def test_preprocess_rejects_none_image():
    with pytest.raises(ValueError):
        preprocess(None)  # type: ignore[arg-type]


def test_preprocess_rejects_non_positive_target_size():
    img = _solid_image((50, 50), (1, 2, 3))
    with pytest.raises(ValueError):
        preprocess(img, target_size=0)


# --- capture_steps (per-step filmstrip) ------------------------------------


def test_capture_steps_returns_one_image_per_step():
    img = Image.new("RGB", (400, 300), (120, 140, 90))
    out, steps, step_images = preprocess(img, capture_steps=True)
    assert len(step_images) == len(steps)
    for step_image in step_images:
        assert isinstance(step_image, Image.Image)
        assert step_image.mode == "RGB"
    # Last captured image must match the final returned image exactly.
    assert np.array_equal(np.asarray(step_images[-1]), np.asarray(out))


def test_capture_steps_default_false_keeps_two_tuple_return():
    img = Image.new("RGB", (100, 100), (10, 20, 30))
    result = preprocess(img)
    assert len(result) == 2  # unchanged for callers that don't pass capture_steps


def test_capture_steps_images_reflect_progressive_transforms():
    # The steps that actually mutate pixels (white_balance, clahe) must show a
    # different image than the step before them (convert_rgb is a no-op here
    # since the source is already RGB, so it is deliberately excluded).
    img = _color_cast_image()
    _out, steps, step_images = preprocess(img, capture_steps=True)
    by_name = {s["name"]: image for s, image in zip(steps, step_images)}
    assert not np.array_equal(np.asarray(by_name["center_crop"]), np.asarray(by_name["white_balance"]))
    assert not np.array_equal(np.asarray(by_name["white_balance"]), np.asarray(by_name["clahe"]))


# --- color-preserving white balance (bug fix regression tests) -------------


def _saturated_red_subject_image(size: tuple[int, int] = (200, 200)) -> Image.Image:
    """A greenish background with a solid, highly-saturated red disc in the
    center — mimics a close-up red flower photo where the classic grey-world
    assumption misfires (it reads the red subject itself as a color cast)."""
    arr = np.full((size[1], size[0], 3), (60, 140, 70), dtype=np.uint8)
    cy, cx = size[1] // 2, size[0] // 2
    radius = min(size) // 3
    yy, xx = np.mgrid[0 : size[1], 0 : size[0]]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    arr[mask] = (200, 40, 35)
    return Image.fromarray(arr, mode="RGB")


def _mean_hue_saturation(image: Image.Image) -> tuple[float, float]:
    import cv2

    arr = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float64)
    return float(hsv[..., 0].mean()), float(hsv[..., 1].mean())


def test_saturated_red_subject_stays_red_not_brown():
    img = _saturated_red_subject_image()
    out, _steps = preprocess(img)

    arr = np.asarray(out, dtype=np.float64)
    h, w, _ = arr.shape
    cy, cx = h // 2, w // 2
    patch = arr[cy - 10 : cy + 10, cx - 10 : cx + 10].reshape(-1, 3).mean(axis=0)
    r, g, b = patch
    # A destroyed (brown/grey) rose would have g and b catch up to r (as the
    # buggy full-strength grey-world did). The subject must stay clearly red.
    assert r > g + 50 and r > b + 50, f"red subject color destroyed: mean rgb={patch}"


def test_saturated_red_subject_hue_and_saturation_roughly_preserved():
    img = _saturated_red_subject_image()
    out, _steps = preprocess(img)
    before_hue, before_sat = _mean_hue_saturation(img)
    after_hue, after_sat = _mean_hue_saturation(out)
    # OpenCV hue range is 0-179; a small shift is expected (resize/crop/CLAHE),
    # a hue swing toward brown/desaturated would be much larger.
    assert abs(after_hue - before_hue) < 10
    assert after_sat > before_sat * 0.8


def test_gentle_white_balance_preserves_far_more_color_than_full_strength():
    """Directly demonstrates the fix: blending only a fraction of the grey-
    world correction (the shipped behavior) leaves meaningfully more color
    variation than the old full-strength correction would have."""
    from ml.preprocess.pipeline import _grey_world_white_balance

    img = _color_cast_image()
    gentle = _grey_world_white_balance(img, blend_alpha=0.25)  # shipped default
    full = _grey_world_white_balance(img, blend_alpha=1.0)  # old (buggy) behavior

    gentle_means = np.asarray(gentle, dtype=np.float64).reshape(-1, 3).mean(axis=0)
    full_means = np.asarray(full, dtype=np.float64).reshape(-1, 3).mean(axis=0)
    gentle_spread = float(gentle_means.max() - gentle_means.min())
    full_spread = float(full_means.max() - full_means.min())

    assert gentle_spread > 5 * full_spread
