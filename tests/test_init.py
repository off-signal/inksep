"""Particle initialisation: rejection sampling contract."""

import numpy as np
import pytest

from inksep.core import init_particles
from inksep.image import load_image


def _init(brightness, n, invert=False, seed=42, border=0):
    H, W = brightness.shape
    b = border
    return init_particles(
        brightness, n, float(b), float(W - b), float(b), float(H - b), invert,
        np.random.default_rng(seed),
    )


def test_shape_and_dtype(gradient_brightness):
    pts = _init(gradient_brightness, 100)
    assert pts.shape == (100, 2)
    assert pts.dtype == np.float64


def test_positions_within_bounds(gradient_brightness):
    pts = _init(gradient_brightness, 200, border=6)
    assert pts[:, 0].min() >= 6.0
    assert pts[:, 0].max() <= 80 - 6
    assert pts[:, 1].min() >= 6.0
    assert pts[:, 1].max() <= 60 - 6


def test_seeded_determinism(gradient_brightness):
    a = _init(gradient_brightness, 50, seed=123)
    b = _init(gradient_brightness, 50, seed=123)
    assert np.array_equal(a, b)


def test_different_seeds_differ(gradient_brightness):
    a = _init(gradient_brightness, 50, seed=1)
    b = _init(gradient_brightness, 50, seed=2)
    assert not np.array_equal(a, b)


def test_density_bias_prefers_dark_regions(two_tone_png):
    brightness, _, _ = load_image(two_tone_png)
    pts = _init(brightness, 300, seed=0)
    dark_half = np.mean(pts[:, 0] < 40)
    assert dark_half > 0.9


def test_invert_bias_prefers_bright_regions(two_tone_png):
    brightness, _, _ = load_image(two_tone_png)
    pts = _init(brightness, 300, invert=True, seed=0)
    bright_half = np.mean(pts[:, 0] > 40)
    assert bright_half > 0.9


def test_zero_particles_returns_empty(gradient_brightness):
    pts = _init(gradient_brightness, 0)
    assert pts.shape == (0, 2)


def test_uniform_white_image_raises(uniform_white_png):
    brightness, _, _ = load_image(uniform_white_png)
    with pytest.raises(ValueError, match="uniform white"):
        _init(brightness, 10, invert=False)


def test_uniform_black_inverted_raises(uniform_black_png):
    brightness, _, _ = load_image(uniform_black_png)
    with pytest.raises(ValueError, match="uniform black"):
        _init(brightness, 10, invert=True)
