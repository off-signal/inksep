"""Image loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter


def load_image(path: str | Path) -> tuple[NDArray[np.float32], int, int]:
    """Load an image and return a blurred grayscale brightness array.

    Applies a 1px box blur before conversion to reduce single-pixel noise
    that would otherwise create isolated stipple points.

    Returns:
        brightness: float32 array of shape (H, W), values 0-255
        width, height: original image dimensions in pixels
    """
    img = Image.open(path).convert("RGB")
    w, h = img.width, img.height
    gray = np.array(img.filter(ImageFilter.BoxBlur(1)).convert("L"), dtype=np.float32)
    return gray, w, h


def load_image_rgb(path: str | Path) -> NDArray[np.uint8]:
    """Load an image as an RGB uint8 array of shape (H, W, 3)."""
    return np.array(Image.open(path).convert("RGB"))
