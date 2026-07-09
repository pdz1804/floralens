"""Unit tests for the CV preprocessing pipeline (ml.preprocess.pipeline)."""
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
