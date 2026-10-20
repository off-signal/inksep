"""SVG export for stipple layouts."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .config import StippleConfig


class SvgSummary(NamedTuple):
    """Statistics about a rendered SVG document."""

    rendered: int
    total: int
    svg_w_mm: float
    svg_h_mm: float
    min_r_px: float
    max_r_px: float
    min_r_mm: float
    max_r_mm: float


def sample_v(
    particles: NDArray[np.float64], brightness: NDArray[np.float32], config: StippleConfig
) -> NDArray[np.float64]:
    """Per-particle brightness in [0, 1], with invert applied."""
    img_H, img_W = brightness.shape
    px = np.clip(np.floor(particles[:, 0]).astype(np.int32), 0, img_W - 1)
    py = np.clip(np.floor(particles[:, 1]).astype(np.int32), 0, img_H - 1)
    v = brightness[py, px].astype(np.float64) / 255.0
    if config.invert:
        v = 1.0 - v
    return v


def visible_mask(
    particles: NDArray[np.float64], brightness: NDArray[np.float32], config: StippleConfig
) -> NDArray[np.bool_]:
    """Boolean mask of particles that survive the white cutoff (drawn to SVG)."""
    return sample_v(particles, brightness, config) < 1.0 - config.white_cutoff


def render_svg(
    particles: NDArray[np.float64],
    brightness: NDArray[np.float32],
    config: StippleConfig,
    *,
    order: NDArray[np.integer] | None = None,
    stroke_width: float | None = None,
    background: bool = True,
) -> str:
    """Render the layout as an SVG document string.

    Dot size follows local brightness; dots brighter than ``white_cutoff``
    are omitted. Each circle carries an inline comment with its ink
    diameter in millimetres.

    ``order`` sets the emission order of the visible dots (e.g. a
    :func:`inksep.tsp.solve_tsp` result); None keeps particle order.
    ``stroke_width``, when set, switches to outline circles sized so the
    outer ink still measures the nominal diameter (the pen-plotter
    convention); None keeps filled discs. ``background`` includes the
    full-canvas white (or black) rectangle; plotters that must not trace
    a border can omit it.
    """
    img_H, img_W = brightness.shape
    W, H = img_W, img_H
    px_per_mm = config.dpi / 25.4
    svg_w_mm = W / px_per_mm
    svg_h_mm = H / px_per_mm
    bg_color = "black" if config.invert else "white"
    dot_color = "white" if config.invert else "black"

    sep_comment = (
        f" | sep_factor={config.sep_factor}"
        f" dark_thresh={config.sep_dark_threshold}"
        if config.sep_factor > 0
        else ""
    )

    min_dot_mm = config.min_dot_size / px_per_mm
    max_dot_mm = config.max_dot / px_per_mm

    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n',
        '<svg xmlns="http://www.w3.org/2000/svg"\n',
        f'     width="{svg_w_mm:.4f}mm" height="{svg_h_mm:.4f}mm"\n',
        f'     viewBox="0 0 {W} {H}">\n',
        "  <desc>Every circle's r is the path radius; ink diameter = 2r when "
        "filled, 2r + stroke-width when stroked. Authored at this exact size; "
        "reposition this group, do not scale it (scaling changes all "
        "millimetre sizes).</desc>\n",
        f"  <!-- inksep | {config.n_stipples} stipples"
        f" | dots {min_dot_mm:.4f}–{max_dot_mm:.4f}mm"
        f" | dpi={config.dpi}{sep_comment} -->\n",
    ]
    if background:
        lines.append(f'  <rect width="{W}" height="{H}" fill="{bg_color}"/>\n')
    lines.append('  <g id="stipples">\n')

    keep = visible_mask(particles, brightness, config)
    idx = np.flatnonzero(keep)
    if order is not None:
        if len(order) != len(idx):
            raise ValueError(
                f"order length {len(order)} != visible dot count {len(idx)}"
            )
        idx = idx[np.asarray(order, dtype=np.intp)]

    v_all = sample_v(particles, brightness, config)
    stroke_px = float(stroke_width) if stroke_width is not None else None
    for i in idx:
        pt = particles[i]
        v = float(v_all[i])
        dot_r = (config.max_dot - v * config.dot_scale) / 2.0
        dot_r_mm = dot_r / px_per_mm
        if stroke_px is not None:
            # Shrink the radius by half the stroke so the outer ink edge
            # lands on the nominal diameter
            draw_r = max(dot_r - stroke_px / 2.0, 0.001)
            attrs = (
                f' r="{draw_r:.4f}" fill="none" stroke="{dot_color}"'
                f' stroke-width="{stroke_px:.4f}"'
            )
            if 2.0 * dot_r < stroke_px:
                # Pen wider than the dot: the stroke defines the ink, so
                # report the pen-limited size (nominal in brackets)
                comment = (
                    f"<!-- d={stroke_px / px_per_mm:.4f}mm"
                    f" (pen-limited; nominal {2.0 * dot_r_mm:.4f}mm) -->"
                )
            else:
                comment = f"<!-- d={2.0 * dot_r_mm:.4f}mm -->"
        else:
            attrs = f' r="{dot_r:.4f}" fill="{dot_color}"'
            comment = f"<!-- d={2.0 * dot_r_mm:.4f}mm -->"
        lines.append(
            f'    <circle cx="{pt[0]:.3f}" cy="{pt[1]:.3f}"{attrs}/>{comment}\n'
        )

    lines.append("  </g>\n</svg>\n")
    return "".join(lines)


def write_svg(
    particles: NDArray[np.float64],
    brightness: NDArray[np.float32],
    config: StippleConfig,
    output_path: str | Path,
    *,
    order: NDArray[np.integer] | None = None,
    stroke_width: float | None = None,
    background: bool = True,
) -> SvgSummary:
    """Render and write the SVG document; returns summary statistics."""
    text = render_svg(
        particles,
        brightness,
        config,
        order=order,
        stroke_width=stroke_width,
        background=background,
    )
    Path(output_path).write_text(text, encoding="utf-8")

    img_H, img_W = brightness.shape
    px_per_mm = config.dpi / 25.4
    rendered = text.count("<circle ")
    return SvgSummary(
        rendered=rendered,
        total=len(particles),
        svg_w_mm=img_W / px_per_mm,
        svg_h_mm=img_H / px_per_mm,
        min_r_px=config.min_dot_size / 2,
        max_r_px=config.max_dot / 2,
        min_r_mm=config.min_dot_size / (2 * px_per_mm),
        max_r_mm=config.max_dot / (2 * px_per_mm),
    )
