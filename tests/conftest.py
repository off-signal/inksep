"""Shared fixtures. Forces a headless matplotlib backend before any inksep import."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _save_png(path: Path, arr: np.ndarray, mode: str = "L") -> Path:
    Image.fromarray(arr.astype(np.uint8), mode=mode).convert("RGB").save(path)
    return path


@pytest.fixture
def gradient_png(tmp_path: Path) -> Path:
    """80x60 horizontal gradient: dark (0) on the left, bright (255) on the right."""
    w, h = 80, 60
    xs = np.linspace(0, 255, w, dtype=np.float32)
    arr = np.tile(xs, (h, 1))
    return _save_png(tmp_path / "gradient.png", arr)


@pytest.fixture
def two_tone_png(tmp_path: Path) -> Path:
    """80x60 image: solid black left half, solid white right half."""
    w, h = 80, 60
    arr = np.zeros((h, w), dtype=np.float32)
    arr[:, w // 2 :] = 255.0
    return _save_png(tmp_path / "two_tone.png", arr)


@pytest.fixture
def uniform_white_png(tmp_path: Path) -> Path:
    return _save_png(tmp_path / "white.png", np.full((20, 20), 255, dtype=np.float32))


@pytest.fixture
def uniform_black_png(tmp_path: Path) -> Path:
    return _save_png(tmp_path / "black.png", np.zeros((20, 20), dtype=np.float32))


@pytest.fixture
def gradient_brightness() -> np.ndarray:
    """In-memory 60x80 gradient, no file IO."""
    xs = np.linspace(0, 255, 80, dtype=np.float32)
    return np.tile(xs, (60, 1))
