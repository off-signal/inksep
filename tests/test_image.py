"""Image loading behaviour."""

import numpy as np

from inksep.image import load_image, load_image_rgb


def test_load_image_shape_and_dtype(gradient_png):
    gray, w, h = load_image(gradient_png)
    assert gray.dtype == np.float32
    assert gray.shape == (h, w)
    assert gray.min() >= 0.0
    assert gray.max() <= 255.0


def test_load_image_rgb(uniform_black_png):
    rgb = load_image_rgb(uniform_black_png)
    assert rgb.shape == (20, 20, 3)
    assert rgb.dtype == np.uint8


def test_box_blur_softens_isolated_pixels(tmp_path):
    from PIL import Image

    arr = np.zeros((21, 21), dtype=np.uint8)
    arr[10, 10] = 255
    path = tmp_path / "spike.png"
    Image.fromarray(arr, mode="L").convert("RGB").save(path)

    gray, _, _ = load_image(path)
    # A 1px box blur should spread the spike across its 3x3 neighbourhood.
    assert 0 < gray[10, 10] < 255
    assert gray[9, 10] > 0
    assert gray[10, 9] > 0
