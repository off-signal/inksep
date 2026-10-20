"""Configuration validation and derived geometry."""

import pytest

from inksep.config import StippleConfig


def test_defaults():
    cfg = StippleConfig()
    assert cfg.n_stipples == 2000
    assert cfg.max_dot == pytest.approx(1.75 * 5.0)
    assert cfg.dot_scale == pytest.approx(1.75 * 4.0)


def test_derived_geometry():
    cfg = StippleConfig(min_dot_size=2.0, dot_size_factor=3.0)
    assert cfg.max_dot == pytest.approx(8.0)
    assert cfg.dot_scale == pytest.approx(6.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_stipples": 0},
        {"n_stipples": -5},
        {"min_dot_size": 0.0},
        {"min_dot_size": -1.0},
        {"dot_size_factor": -0.1},
        {"white_cutoff": 1.5},
        {"white_cutoff": -0.1},
        {"border_width": -1},
        {"dpi": 0.0},
        {"dpi": -96.0},
        {"step": 0.0},
        {"step": -1.0},
        {"damping": 1.5},
        {"damping": -0.1},
        {"sep_factor": -0.1},
        {"sep_dark_threshold": 0.0},
        {"sep_dark_threshold": 1.5},
    ],
)
def test_invalid_values_rejected(kwargs):
    with pytest.raises(ValueError):
        StippleConfig(**kwargs)


def test_config_is_immutable():
    import dataclasses

    cfg = StippleConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.n_stipples = 5  # type: ignore[misc]
