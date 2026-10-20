"""inksep: separation-constrained weighted Voronoi stippling.

Converts a raster image into a stippled SVG using weighted Lloyd relaxation
with an optional physical dot-separation pass. Output is sized in real
millimetres, suitable for pen plotters, laser engravers, and wherever
millimetre-accurate dots matter.

Usage:
    inksep input.png output.svg [options]
"""

from .config import Bounds, StippleConfig
from .core import SeparationReport, compute_separation_report, format_separation_report
from .generator import InkSep

__version__ = "1.0.0"

__all__ = [
    "Bounds",
    "InkSep",
    "SeparationReport",
    "StippleConfig",
    "__version__",
    "compute_separation_report",
    "format_separation_report",
]
