"""Core algorithm: particle initialisation, Lloyd relaxation, quality metrics."""

from __future__ import annotations

from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from .config import Bounds, StippleConfig


class SeparationReport(TypedDict):
    """Dot-separation quality metrics, all distances in millimetres."""

    global_min_mm: float
    global_mean_mm: float
    max_dot_mm: float
    ratio: float
    n_overlap: int
    n_overlap_pairs: int
    pct_overlap: float
    max_depth_mm: float
    dark_mean_mm: float
    dark_min_mm: float
    n_dots: int


def init_particles(
    brightness: NDArray[np.float32],
    n: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    invert: bool,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Seed stipple positions by rejection sampling: random pixels are
    accepted in proportion to darkness (or brightness when inverted), so
    dots start concentrated in inked areas.

    Returns:
        Array of shape (n, 2) with initial (x, y) positions.

    Raises:
        ValueError: If the image is uniformly white (or black when
            inverted): there would be nothing to sample.
    """
    if n <= 0:
        return np.empty((0, 2), dtype=np.float64)

    # No pixel can ever be accepted → rejection sampling would loop forever.
    if not invert:
        if not np.any(brightness < 255.0):
            raise ValueError(
                "cannot sample particles: image is uniform white "
                "(no dark regions to stipple); try --invert"
            )
    else:
        if not np.any(brightness > 0.0):
            raise ValueError(
                "cannot sample particles: image is uniform black when inverted "
                "(no bright regions to stipple); drop --invert"
            )

    H, W = brightness.shape
    pts: list[tuple[float, float]] = []
    batch = max(n * 10, 20_000)
    while len(pts) < n:
        xs = rng.uniform(x_min, x_max, batch)
        ys = rng.uniform(y_min, y_max, batch)
        xi = np.floor(xs).astype(np.int32).clip(0, W - 1)
        yi = np.floor(ys).astype(np.int32).clip(0, H - 1)
        p = brightness[yi, xi] / 255.0
        if invert:
            p = 1.0 - p
        accept = rng.random(batch) >= p
        for k in range(batch):
            if accept[k]:
                pts.append((xs[k], ys[k]))
                if len(pts) == n:
                    break
    return np.array(pts, dtype=np.float64)


def lloyd_step(
    particles: NDArray[np.float64],
    brightness: NDArray[np.float32],
    bounds: Bounds,
    config: StippleConfig,
) -> NDArray[np.float64]:
    """Run one Lloyd relaxation iteration with optional separation enforcement.

    Phase 1: centroid. Every sampling-grid pixel joins its nearest dot's
    Voronoi cell; the dot's new target is that cell's ink-weighted centre.

    Phase 2: separation (skipped when sep_factor == 0). Dots closer than
    sep_factor × (r_i + r_j) push apart. The push only affects dark dots
    (full strength at black, none at or above sep_dark_threshold), and the
    lighter dot in a pair yields more than the darker one.

    Phase 3: damping. The dot blends its new target with its current
    position by the damping factor.

    Returns:
        Array of shape (N, 2) with updated particle positions.
    """
    x_min, x_max = bounds.x_min, bounds.x_max
    y_min, y_max = bounds.y_min, bounds.y_max
    invert = config.invert
    step = config.step
    damping = config.damping
    max_dot = config.max_dot
    dot_scale = config.dot_scale
    sep_factor = config.sep_factor
    sep_dark_threshold = config.sep_dark_threshold

    H, W = brightness.shape
    N = len(particles)

    # ── Phase 1: weighted Lloyd centroid ─────────────────────────────
    xs = np.arange(x_min, x_max, step, dtype=np.float32)
    ys = np.arange(y_min, y_max, step, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    gx_flat = gx.ravel()
    gy_flat = gy.ravel()

    grid = np.empty((len(gx_flat), 2), dtype=np.float32)
    grid[:, 0] = gx_flat
    grid[:, 1] = gy_flat

    tree = KDTree(particles)
    _, labels = tree.query(grid, workers=-1)

    xi = np.round(gx_flat).astype(np.int32).clip(0, W - 1)
    yi = np.round(gy_flat).astype(np.int32).clip(0, H - 1)
    bval = brightness[yi, xi]

    density = ((0.001 + bval) if invert else (255.001 - bval)).astype(np.float64)

    wx = np.bincount(labels, weights=density * gx_flat, minlength=N)
    wy = np.bincount(labels, weights=density * gy_flat, minlength=N)
    wd = np.bincount(labels, weights=density, minlength=N)

    mask = wd > 0.0
    safe = np.where(mask, wd, 1.0)
    cx = np.where(mask, wx / safe, particles[:, 0])
    cy = np.where(mask, wy / safe, particles[:, 1])
    centroids = np.column_stack([cx, cy])

    # ── Phase 2: separation repulsion ────────────────────────────────
    if sep_factor > 0.0 and max_dot > 0.0:

        # Brightness at the current positions (before the move)
        px_i = np.clip(np.floor(particles[:, 0]).astype(np.int32), 0, W - 1)
        py_i = np.clip(np.floor(particles[:, 1]).astype(np.int32), 0, H - 1)
        v = brightness[py_i, px_i] / 255.0
        if invert:
            v = 1.0 - v

        # Same radius formula as SVG export: dark pixel = large dot
        radii = (max_dot - v * dot_scale) / 2.0

        # How much each dot cares about separation: 1 at black, 0 at (and
        # above) the threshold
        dark_w = np.clip(1.0 - v / sep_dark_threshold, 0.0, 1.0)

        # Worst case is two black dots at full radius: sep_factor × max_dot
        max_search = sep_factor * max_dot
        pairs = tree.query_pairs(max_search, output_type="ndarray")

        if len(pairs) > 0:
            i_idx = pairs[:, 0]
            j_idx = pairs[:, 1]

            delta = particles[j_idx] - particles[i_idx]
            dist = np.sqrt((delta**2).sum(axis=1))
            req = sep_factor * (radii[i_idx] + radii[j_idx])

            # Only actual violations; coincident dots have no direction to push
            vmask = (dist < req) & (dist > 1e-9)

            if np.any(vmask):
                vi = i_idx[vmask]
                vj = j_idx[vmask]
                vdist = dist[vmask]
                vreq = req[vmask]
                vdelta = delta[vmask]

                unit = vdelta / vdist[:, np.newaxis]
                overlap = vreq - vdist  # how far past the limit

                wi = dark_w[vi]
                wj = dark_w[vj]
                total_w = wi + wj
                safe_t = np.where(total_w < 1e-9, 1.0, total_w)

                # Each dot yields to the other in proportion to the other's
                # darkness, so fi = wj / (wi + wj)
                fi = wj / safe_t
                fj = wi / safe_t

                # A dot in several violating pairs accumulates all the pushes
                rep = np.zeros_like(centroids)
                np.add.at(rep, vi, -unit * (overlap * fi)[:, np.newaxis])
                np.add.at(rep, vj, unit * (overlap * fj)[:, np.newaxis])

                # Light dots are untouched; black dots take the full correction
                centroids = centroids + rep * dark_w[:, np.newaxis]

    # ── Phase 3: damping ─────────────────────────────────────────────
    if damping > 0.0:
        centroids = damping * particles + (1.0 - damping) * centroids

    centroids[:, 0] = np.clip(centroids[:, 0], x_min + 1.0, x_max - 1.0)
    centroids[:, 1] = np.clip(centroids[:, 1], y_min + 1.0, y_max - 1.0)

    return centroids


def compute_separation_report(
    particles: NDArray[np.float64],
    brightness: NDArray[np.float32],
    max_dot: float,
    dot_scale: float,
    sep_factor: float,
    dpi: float,
    invert: bool,
) -> SeparationReport:
    """Dot-separation quality metrics for the final layout, in millimetres.

    Nearest-neighbour gaps (global and for the darkest 10%), plus how many
    dots and pairs overlap and how deeply they merge.
    """
    H, W = brightness.shape
    N = len(particles)
    px_per_mm = dpi / 25.4

    px_i = np.clip(np.floor(particles[:, 0]).astype(np.int32), 0, W - 1)
    py_i = np.clip(np.floor(particles[:, 1]).astype(np.int32), 0, H - 1)
    v = brightness[py_i, px_i] / 255.0
    if invert:
        v = 1.0 - v
    radii = (max_dot - v * dot_scale) / 2.0

    tree = KDTree(particles)
    dists, _ = tree.query(particles, k=2)
    nn_px = cast("NDArray[np.float64]", dists)[:, 1]
    nn_mm = nn_px / px_per_mm
    global_min = np.min(nn_mm)
    max_dot_mm = max_dot / px_per_mm

    # Two dots overlap when their centres are closer than r_i + r_j; the
    # worst case sums to max_dot, so always search that far. Searching only
    # sep_factor × max_dot would miss exactly the overlaps the separation
    # pass failed to fix.
    pairs = tree.query_pairs(max_dot, output_type="ndarray")
    n_overlap_pairs = 0
    n_overlap = 0  # unique dots touching at least one neighbour
    max_depth = 0.0
    if len(pairs) > 0:
        ii, jj = pairs[:, 0], pairs[:, 1]
        pd = np.sqrt(((particles[jj] - particles[ii]) ** 2).sum(axis=1))
        req = radii[ii] + radii[jj]
        ov_mask = pd < req
        n_overlap_pairs = int(np.sum(ov_mask))
        if n_overlap_pairs > 0:
            touched = np.concatenate((ii[ov_mask], jj[ov_mask]))
            n_overlap = int(np.unique(touched).size)
            max_depth = float(np.max((req - pd)[ov_mask])) / px_per_mm

    # Darkest 10% nearest-neighbour
    thresh10 = np.percentile(v, 10)
    dark_mask = v <= thresh10
    dark_pts = particles[dark_mask]
    dark_tree = KDTree(dark_pts)
    dark_d, _ = dark_tree.query(dark_pts, k=2)
    dark_nn_mm = cast("NDArray[np.float64]", dark_d)[:, 1] / px_per_mm

    return SeparationReport(
        global_min_mm=float(global_min),
        global_mean_mm=float(np.mean(nn_mm)),
        max_dot_mm=float(max_dot_mm),
        ratio=float(global_min / max_dot_mm),
        n_overlap=n_overlap,
        n_overlap_pairs=n_overlap_pairs,
        pct_overlap=n_overlap / N * 100,
        max_depth_mm=max_depth,
        dark_mean_mm=float(np.mean(dark_nn_mm)),
        dark_min_mm=float(np.min(dark_nn_mm)),
        n_dots=N,
    )


def format_separation_report(r: SeparationReport) -> str:
    """Render a separation report as a printable multi-line string."""
    bar = "─" * 52
    lines = [
        "",
        f"  {bar}",
        "  SEPARATION REPORT",
        f"  {bar}",
        f"  Min dot spacing  : {r['global_min_mm']:.4f} mm",
        f"  Mean dot spacing : {r['global_mean_mm']:.4f} mm",
        f"  Max dot diameter : {r['max_dot_mm']:.4f} mm",
        f"  Spacing/max-dot  : {r['ratio']:.4f}  "
        f"({'CLEAN ✓' if r['ratio'] >= 1.0 else 'overlap possible'})",
        f"  {bar}",
        f"  Overlapping dots  : {r['n_overlap']:,} / {r['n_dots']:,} "
        f"({r['pct_overlap']:.1f}%) touch ≥1 neighbor",
        f"    from {r['n_overlap_pairs']:,} pairs, max merge depth "
        f"{r['max_depth_mm']:.4f} mm",
        f"  {bar}",
        f"  Darkest 10% mean spacing : {r['dark_mean_mm']:.4f} mm",
        f"  Darkest 10% min  spacing : {r['dark_min_mm']:.4f} mm",
        f"  {bar}",
        "",
    ]
    return "\n".join(lines)
