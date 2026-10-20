"""Configuration objects for inksep."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bounds:
    """Pixel-space drawing bounds. Stipples are kept strictly inside these."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class StippleConfig:
    """All tunable parameters for a stippling run.

    Attributes:
        n_stipples:          Total number of stipple points.
        min_dot_size:        Diameter of the smallest dot in pixels.
        dot_size_factor:     How much bigger dark dots get: largest diameter
                             = smallest × (1 + factor).
        white_cutoff:        Suppress dots in regions brighter than this (0-1).
        invert:              Treat bright pixels as ink (white-on-black output).
        border_width:        Pixel margin kept clear of stipples at image edges.
        dpi:                 Source-image resolution, used to convert pixels
                             to millimetres.
        step:                Sampling grid size in source pixels (larger =
                             faster, coarser).
        damping:             Fraction of each move to skip (0 = take every
                             step; higher = calmer, slower).
        sep_factor:          Minimum distance between dark dots, as a
                             fraction of their radii added together. 0 turns
                             separation off.
        sep_dark_threshold:  Only dots darker than this are held apart;
                             lighter dots pack freely.
        seed:                RNG seed for reproducible particle initialisation.
    """

    n_stipples: int = 2000
    min_dot_size: float = 1.75
    dot_size_factor: float = 4.0
    white_cutoff: float = 0.0
    invert: bool = False
    border_width: int = 6
    dpi: float = 96.0
    step: float = 1.0
    damping: float = 0.0
    sep_factor: float = 0.85
    sep_dark_threshold: float = 0.50
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_stipples < 1:
            raise ValueError(f"n_stipples must be >= 1, got {self.n_stipples}")
        if self.min_dot_size <= 0:
            raise ValueError(f"min_dot_size must be > 0, got {self.min_dot_size}")
        if self.dot_size_factor < 0:
            raise ValueError(f"dot_size_factor must be >= 0, got {self.dot_size_factor}")
        if not 0.0 <= self.white_cutoff <= 1.0:
            raise ValueError(f"white_cutoff must be in [0, 1], got {self.white_cutoff}")
        if self.border_width < 0:
            raise ValueError(f"border_width must be >= 0, got {self.border_width}")
        if self.dpi <= 0:
            raise ValueError(f"dpi must be > 0, got {self.dpi}")
        if self.step <= 0:
            raise ValueError(f"step must be > 0, got {self.step}")
        if not 0.0 <= self.damping <= 1.0:
            raise ValueError(f"damping must be in [0, 1], got {self.damping}")
        if self.sep_factor < 0:
            raise ValueError(f"sep_factor must be >= 0, got {self.sep_factor}")
        if not 0.0 < self.sep_dark_threshold <= 1.0:
            raise ValueError(
                f"sep_dark_threshold must be in (0, 1], got {self.sep_dark_threshold}"
            )

    @property
    def max_dot(self) -> float:
        """Diameter of the largest dot in pixels."""
        return self.min_dot_size * (1.0 + self.dot_size_factor)

    @property
    def dot_scale(self) -> float:
        """Span between max and min dot diameters in pixels."""
        return self.max_dot - self.min_dot_size
