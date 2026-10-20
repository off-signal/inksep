"""One Lloyd iteration: centroid, damping and separation repulsion."""

import numpy as np
import pytest

from inksep.config import Bounds, StippleConfig
from inksep.core import lloyd_step

BOX = Bounds(x_min=0.0, x_max=100.0, y_min=0.0, y_max=100.0)
EQUILIBRIUM = np.array([[25.0, 50.0], [75.0, 50.0]])


def _cfg(**kwargs) -> StippleConfig:
    defaults: dict = dict(
        n_stipples=2,
        min_dot_size=80.0,
        dot_size_factor=0.0,  # max_dot = 80, dot_scale = 0
        sep_factor=0.0,
        border_width=0,
    )
    defaults.update(kwargs)
    return StippleConfig(**defaults)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def test_shape_preserved(gradient_brightness):
    rng = np.random.default_rng(0)
    pts = np.column_stack(
        [rng.uniform(5, 75, 30), rng.uniform(5, 55, 30)]
    )
    out = lloyd_step(pts, gradient_brightness, BOX, _cfg())
    assert out.shape == (30, 2)


def test_positions_stay_in_bounds(gradient_brightness):
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0, 80, 30), rng.uniform(0, 60, 30)])
    # bounds padded by 1px clip inside lloyd_step
    bounds = Bounds(x_min=0.0, x_max=80.0, y_min=0.0, y_max=60.0)
    out = lloyd_step(pts, gradient_brightness, bounds, _cfg())
    assert out[:, 0].min() >= 1.0
    assert out[:, 0].max() <= 79.0
    assert out[:, 1].min() >= 1.0
    assert out[:, 1].max() <= 59.0


def test_full_damping_is_identity(gradient_brightness):
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(5, 75, 20), rng.uniform(5, 55, 20)])
    out = lloyd_step(pts, gradient_brightness, BOX, _cfg(damping=1.0))
    assert np.array_equal(out, pts)


def test_convergence_movement_shrinks(gradient_brightness):
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(0, 80, 60), rng.uniform(0, 60, 60)])
    bounds = Bounds(x_min=0.0, x_max=80.0, y_min=0.0, y_max=60.0)
    cfg = _cfg(n_stipples=60, sep_factor=0.0)
    movements = []
    prev = pts
    for _ in range(20):
        out = lloyd_step(prev, gradient_brightness, bounds, cfg)
        movements.append(float(np.sqrt(((out - prev) ** 2).sum(axis=1)).mean()))
        prev = out
    assert movements[-1] < movements[0] * 0.25


def test_dark_pair_is_pushed_apart():
    """At Lloyd equilibrium on a black field, repulsion is the only force."""
    black = np.zeros((100, 100), dtype=np.float32)
    sep_off = lloyd_step(EQUILIBRIUM, black, BOX, _cfg(sep_factor=0.0))
    sep_on = lloyd_step(EQUILIBRIUM, black, BOX, _cfg(sep_factor=0.85))

    d_off = _dist(*sep_off)
    d_on = _dist(*sep_on)
    assert d_on > d_off + 5.0
    # Requested separation is 0.85 * 80 = 68 px; repulsion targets that gap.
    assert d_on == pytest.approx(68.0, abs=1.5)


def test_light_pair_is_not_pushed():
    """Brightness-gated: dots above the dark threshold pack freely."""
    bright = np.full((100, 100), 250.0, dtype=np.float32)
    sep_off = lloyd_step(EQUILIBRIUM, bright, BOX, _cfg(sep_factor=0.0))
    sep_on = lloyd_step(EQUILIBRIUM, bright, BOX, _cfg(sep_factor=0.85))
    assert np.array_equal(sep_on, sep_off)


def test_midtone_pair_is_not_pushed():
    """v ≈ 0.5 > sep_dark_threshold 0.5 → free packing."""
    mid = np.full((100, 100), 128.0, dtype=np.float32)
    sep_off = lloyd_step(EQUILIBRIUM, mid, BOX, _cfg(sep_factor=0.0))
    sep_on = lloyd_step(EQUILIBRIUM, mid, BOX, _cfg(sep_factor=0.85))
    assert np.array_equal(sep_on, sep_off)


def test_invert_flips_separation_gating():
    """invert=True: black reads as bright (no push); white reads as dark (pushed)."""
    black = np.zeros((100, 100), dtype=np.float32)
    white = np.full((100, 100), 255.0, dtype=np.float32)
    sep_off = lloyd_step(EQUILIBRIUM, black, BOX, _cfg(sep_factor=0.0))
    black_inv = lloyd_step(EQUILIBRIUM, black, BOX, _cfg(sep_factor=0.85, invert=True))
    white_inv = lloyd_step(EQUILIBRIUM, white, BOX, _cfg(sep_factor=0.85, invert=True))

    assert np.array_equal(black_inv, sep_off)
    assert _dist(*white_inv) == pytest.approx(68.0, abs=1.5)
