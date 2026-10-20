"""High-level generator: orchestrates loading, relaxation and export."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .config import Bounds, StippleConfig
from .core import (
    SeparationReport,
    compute_separation_report,
    format_separation_report,
    init_particles,
    lloyd_step,
)
from .image import load_image
from .svg import sample_v, visible_mask, write_svg
from .tsp import open_path_length, solve_tsp

logger = logging.getLogger("inksep")


class InkSep:
    """Stipple an image: load, relax with weighted Voronoi plus separation,
    export SVG. Keyword arguments mirror :class:`~inksep.config.StippleConfig`.
    """

    def __init__(
        self,
        image_path: str | Path,
        n_stipples: int = 2000,
        min_dot_size: float = 1.75,
        dot_size_factor: float = 4.0,
        white_cutoff: float = 0.0,
        invert: bool = False,
        border_width: int = 6,
        dpi: float = 96.0,
        step: float = 1.0,
        damping: float = 0.0,
        sep_factor: float = 0.85,
        sep_dark_threshold: float = 0.50,
        show_preview: bool = True,
        preview_every: int = 1,
        seed: int | None = None,
        optimize_travel: bool = False,
        travel_budget: float = 5.0,
        stroke_width_mm: float | None = None,
        background: bool = True,
    ) -> None:
        self.config = StippleConfig(
            n_stipples=n_stipples,
            min_dot_size=min_dot_size,
            dot_size_factor=dot_size_factor,
            white_cutoff=white_cutoff,
            invert=invert,
            border_width=border_width,
            dpi=dpi,
            step=step,
            damping=damping,
            sep_factor=sep_factor,
            sep_dark_threshold=sep_dark_threshold,
            seed=seed,
        )
        if travel_budget <= 0:
            raise ValueError(f"travel_budget must be > 0, got {travel_budget}")
        if stroke_width_mm is not None and stroke_width_mm <= 0:
            raise ValueError(
                f"stroke_width_mm must be > 0 when set, got {stroke_width_mm}"
            )
        self.optimize_travel = optimize_travel
        self.travel_budget = travel_budget
        self.stroke_width_mm = stroke_width_mm
        self.background = background
        self.show_preview = show_preview
        self.preview_every = max(1, preview_every)
        self.image_path = str(image_path)

        self.brightness, self.img_w, self.img_h = load_image(image_path)
        b = self.config.border_width
        self.bounds = Bounds(
            x_min=float(b),
            x_max=float(self.img_w - b),
            y_min=float(b),
            y_max=float(self.img_h - b),
        )
        logger.info("Image loaded: %d × %d px", self.img_w, self.img_h)

        rng = np.random.default_rng(self.config.seed)
        self.particles = init_particles(
            self.brightness,
            self.config.n_stipples,
            self.bounds.x_min,
            self.bounds.x_max,
            self.bounds.y_min,
            self.bounds.y_max,
            self.config.invert,
            rng,
        )
        seed_msg = f" (seed={self.config.seed})" if self.config.seed is not None else ""
        logger.info("Initialised %d particles%s", self.config.n_stipples, seed_msg)

        # Preview imported here, so matplotlib stays optional
        self.preview: PreviewWindow | None = None
        if show_preview:
            from .preview import PreviewWindow  # deferred: pulls in matplotlib

            self.preview = PreviewWindow(
                brightness=self.brightness,
                n_stipples=self.config.n_stipples,
                generations=0,
                invert=self.config.invert,
                min_dot=self.config.min_dot_size,
                max_dot=self.config.max_dot,
                dot_scale=self.config.dot_scale,
                cutoff=self.config.white_cutoff,
                image_path=self.image_path,
                sep_factor=self.config.sep_factor,
                dpi=self.config.dpi,
            )

    def run(self, generations: int = 100, verbose: bool = True, tol: float = 0.0) -> int:
        """Run the relaxation loop for ``generations`` generations (or until
        the dots settle: mean movement below ``tol`` px for 3 generations in
        a row, when ``tol > 0``). Logs per-generation progress when verbose,
        and a separation report at the end if separation is on.

        Returns the number of generations actually run.
        """
        if generations < 1:
            raise ValueError(f"generations must be >= 1, got {generations}")
        if tol < 0:
            raise ValueError(f"tol must be >= 0, got {tol}")

        if self.preview:
            self.preview.set_generations(generations)

        t_start = time.perf_counter()
        prev_particles = self.particles.copy()
        below_tol = 0
        completed = 0

        for gen in range(1, generations + 1):
            t0 = time.perf_counter()
            self.particles = lloyd_step(
                self.particles, self.brightness, self.bounds, self.config
            )
            gen_time = time.perf_counter() - t0
            elapsed = time.perf_counter() - t_start

            movement = float(
                np.sqrt(((self.particles - prev_particles) ** 2).sum(axis=1)).mean()
            )
            prev_particles = self.particles.copy()

            if verbose:
                sep_tag = (
                    f"  sep={self.config.sep_factor:.2f}" if self.config.sep_factor > 0 else ""
                )
                logger.info(
                    "  Gen %3d/%d  %.0fms  move=%.3fpx%s",
                    gen,
                    generations,
                    gen_time * 1000,
                    movement,
                    sep_tag,
                )

            if self.preview and (gen % self.preview_every == 0 or gen == generations):
                self.preview.update(self.particles, gen, elapsed, gen_time, movement)

            completed = gen

            if tol > 0.0:
                below_tol = below_tol + 1 if movement < tol else 0
                if below_tol >= 3:
                    logger.info(
                        "  Early stop : gen %d  mean move %.4fpx < tol %.4fpx"
                        " for 3 gens  (%d generations skipped)",
                        gen,
                        movement,
                        tol,
                        generations - gen,
                    )
                    break

        logger.info("Stippling complete.")

        if self.config.sep_factor > 0.0:
            report = self.separation_report()
            logger.info("%s", format_separation_report(report).rstrip("\n"))
        return completed

    def separation_report(self) -> SeparationReport:
        """Compute separation quality metrics for the current layout."""
        return compute_separation_report(
            self.particles,
            self.brightness,
            self.config.max_dot,
            self.config.dot_scale,
            self.config.sep_factor,
            self.config.dpi,
            self.config.invert,
        )

    def pen_clamp_stats(self) -> tuple[int, float] | None:
        """How many visible dots are smaller than the pen: ``(count, smallest
        dot diameter in mm)``.

        ``(0, min)`` when nothing clamps; ``None`` when no stroke width is
        set. :meth:`save_svg` calls this for the exact final count. The CLI
        preflight can only give a verdict, because it runs before the dots
        settle.
        """
        if self.stroke_width_mm is None:
            return None
        keep = visible_mask(self.particles, self.brightness, self.config)
        v = sample_v(self.particles[keep], self.brightness, self.config)
        px_per_mm = self.config.dpi / 25.4
        diameters_mm = (self.config.max_dot - v * self.config.dot_scale) / px_per_mm
        if diameters_mm.size == 0:
            return 0, self.config.min_dot_size / px_per_mm
        n_clamped = int(np.sum(diameters_mm < self.stroke_width_mm))
        min_d = float(np.min(diameters_mm))
        return n_clamped, min_d

    def save_svg(self, output_path: str | Path) -> None:
        """Write the final layout as an SVG file.

        Logs the document size and dot count; with ``optimize_travel`` set,
        also the pen-up travel before and after reordering.
        """
        order = None

        if self.optimize_travel:
            keep = visible_mask(self.particles, self.brightness, self.config)
            visible = self.particles[keep]
            if len(visible) >= 2:
                t0 = time.perf_counter()
                idx = solve_tsp(visible, budget_s=self.travel_budget)
                solve_time = time.perf_counter() - t0

                # Map the visible-dot order back to full particle indices
                visible_idx = np.flatnonzero(keep)
                order = visible_idx[idx]

                px_per_mm = self.config.dpi / 25.4
                seq = np.arange(len(visible))
                travel_before_mm = open_path_length(visible, seq) / px_per_mm
                travel_after_mm = open_path_length(visible, idx) / px_per_mm
                pct = (
                    (1.0 - travel_after_mm / travel_before_mm) * 100.0
                    if travel_before_mm > 0
                    else 0.0
                )
                logger.info(
                    "  Travel opt : %.1fs solve  |  pen-up %.2fm → %.2fm  (%.1f%% saved)",
                    solve_time,
                    travel_before_mm / 1000.0,
                    travel_after_mm / 1000.0,
                    pct,
                )

        stroke_px = None
        if self.stroke_width_mm is not None:
            stroke_px = self.stroke_width_mm * self.config.dpi / 25.4
            stats = self.pen_clamp_stats()
            if stats is not None and stats[0] > 0:
                n_clamped, min_d = stats
                logger.warning(
                    "  Pen clamp  : %d dot(s) smaller than stroke width %.2fmm; "
                    "smallest dot is %.2fmm, pen is %.2fmm; clamped dots will "
                    "render at pen width.",
                    n_clamped,
                    self.stroke_width_mm,
                    min_d,
                    self.stroke_width_mm,
                )

        s = write_svg(
            self.particles,
            self.brightness,
            self.config,
            output_path,
            order=order,
            stroke_width=stroke_px,
            background=self.background,
        )

        logger.info("")
        logger.info("SVG saved → %s", output_path)
        logger.info(
            "  Document  : %.2fmm × %.2fmm  (%d×%dpx @ %gdpi)",
            s.svg_w_mm,
            s.svg_h_mm,
            self.img_w,
            self.img_h,
            self.config.dpi,
        )
        logger.info(
            "  Dots      : %d/%d  (%d removed by cutoff)",
            s.rendered,
            s.total,
            s.total - s.rendered,
        )

        if self.preview:
            self.preview.done(output_path)
