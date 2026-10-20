"""Live matplotlib preview window (display layer).

Importing this module selects a matplotlib backend; it is only imported when
a preview is requested, so headless runs never touch matplotlib.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger("inksep")


def _detect_backend() -> tuple[str, bool]:
    import matplotlib as mpl

    env = os.environ.get("MPLBACKEND", "")
    if env:
        return env, env.lower() not in ("agg", "cairo", "svg", "pdf")
    for b in ["TkAgg", "Qt5Agg", "qtagg", "GTK4Agg", "GTK3Agg", "WXAgg", "MacOSX"]:
        try:
            mpl.use(b)
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(0.1, 0.1))
            plt.close(fig)
            return b, True
        except Exception:
            try:
                mpl.use("Agg")
            except Exception:
                pass
    mpl.use("Agg")
    return "Agg", False


_BACKEND, _INTERACTIVE = _detect_backend()

import matplotlib  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.path as mpath  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402

# Bézier control offset that makes a cubic look like a circle on screen
_KAPPA = 0.5522847498307936


def _circles_path(x: np.ndarray, y: np.ndarray, r: np.ndarray) -> mpath.Path:
    """One compound Bézier path with a circle per dot.

    A single vectorised path instead of thousands of individual patches is
    what keeps the preview responsive at 10k+ dots.
    """
    m = len(x)
    if m == 0:
        return mpath.Path(np.empty((0, 2)), np.empty((0,), dtype=np.uint8))
    kr = _KAPPA * r
    verts = np.empty((m, 13, 2), dtype=np.float64)
    verts[:, 0] = np.column_stack((x + r, y))
    verts[:, 1] = np.column_stack((x + r, y + kr))
    verts[:, 2] = np.column_stack((x + kr, y + r))
    verts[:, 3] = np.column_stack((x, y + r))
    verts[:, 4] = np.column_stack((x - kr, y + r))
    verts[:, 5] = np.column_stack((x - r, y + kr))
    verts[:, 6] = np.column_stack((x - r, y))
    verts[:, 7] = np.column_stack((x - r, y - kr))
    verts[:, 8] = np.column_stack((x - kr, y - r))
    verts[:, 9] = np.column_stack((x, y - r))
    verts[:, 10] = np.column_stack((x + kr, y - r))
    verts[:, 11] = np.column_stack((x + r, y - kr))
    verts[:, 12] = np.column_stack((x + r, y))
    codes = np.full(m * 13, mpath.Path.CURVE4, dtype=np.uint8)
    codes[::13] = mpath.Path.MOVETO
    return mpath.Path(verts.reshape(-1, 2), codes)


class PreviewWindow:
    BG_DARK = "#0d0d0d"
    BG_PANEL = "#141414"
    ACCENT = "#e8c97a"
    TEXT_COLOR = "#d4d4d4"
    DIM_COLOR = "#555555"

    def __init__(
        self,
        brightness: np.ndarray,
        n_stipples: int,
        generations: int,
        invert: bool,
        min_dot: float,
        max_dot: float,
        dot_scale: float,
        cutoff: float,
        image_path: str,
        sep_factor: float = 0.0,
        dpi: float = 96.0,
        snapshot_dir: str | Path | None = None,
    ) -> None:
        """Open the preview figure for a stipple run.

        Dots are drawn at true size in image-pixel coordinates, matching the
        exported SVG exactly. ``generations`` may be updated later via
        :meth:`set_generations`. ``dpi`` only affects the mm readout on the
        stats sidebar; ``snapshot_dir`` receives PNG frames when there is no
        display.
        """
        self.brightness = brightness
        self.n = n_stipples
        self.generations = generations
        self.invert = invert
        self.min_dot = min_dot
        self.max_dot = max_dot
        self.dot_scale = dot_scale
        self.cutoff = cutoff
        self.image_path = str(image_path)
        self.sep_factor = sep_factor
        self.dpi = dpi
        self.interactive = _INTERACTIVE
        H, W = brightness.shape
        self.img_w = W
        self.img_h = H
        self._gen_times: list[float] = []
        self._movements: list[float] = []
        self._gen_nums: list[int] = []

        if snapshot_dir is None:
            self.snapshot_dir = tempfile.mkdtemp(prefix="stipple_preview_")
        else:
            self.snapshot_dir = str(snapshot_dir)
            os.makedirs(snapshot_dir, exist_ok=True)

        self._build_figure(W, H)

        if not self.interactive:
            logger.info("\n  [Preview] No display; snapshots → %s/\n", self.snapshot_dir)

    def _build_figure(self, W: int, H: int) -> None:
        aspect = W / H
        fig_w = min(14.0, max(9.0, aspect * 7.0 + 3.5))
        fig_h = fig_w / (aspect + 0.55)
        self.fig = plt.figure(figsize=(fig_w, fig_h), facecolor=self.BG_DARK)
        manager = getattr(self.fig.canvas, "manager", None)
        if self.interactive and manager is not None:
            manager.set_window_title(
                f"inksep: {os.path.basename(self.image_path)}"
            )

        gs = gridspec.GridSpec(
            1,
            2,
            figure=self.fig,
            width_ratios=[aspect, 0.52],
            left=0.01,
            right=0.99,
            bottom=0.04,
            top=0.93,
            wspace=0.04,
        )

        # Main canvas
        self.ax_main = self.fig.add_subplot(gs[0])
        self.ax_main.set_facecolor(self.BG_DARK)
        self.ax_main.set_xlim(0, W)
        self.ax_main.set_ylim(H, 0)
        self.ax_main.set_aspect("equal")
        self.ax_main.axis("off")

        # Background matches the SVG: white, or black when inverted
        canvas_bg = "black" if self.invert else "white"
        self.ax_main.add_patch(
            Rectangle((0, 0), W, H, facecolor=canvas_bg, edgecolor="none", zorder=0)
        )

        # Dots use the SVG radius formula in image-pixel coordinates, so
        # preview and export stay identical at any zoom or display DPI
        dot_color = "white" if self.invert else "black"
        self._dots = PathPatch(
            _circles_path(np.empty(0), np.empty(0), np.empty(0)),
            facecolor=dot_color,
            edgecolor="none",
            alpha=1.0,
            zorder=2,
        )
        self._dots.set_transform(self.ax_main.transData)
        self.ax_main.add_patch(self._dots)

        # Stats sidebar
        self.ax_stats = self.fig.add_subplot(gs[1])
        self.ax_stats.set_facecolor(self.BG_PANEL)
        self.ax_stats.axis("off")
        for sp in self.ax_stats.spines.values():
            sp.set_edgecolor(self.DIM_COLOR)
            sp.set_visible(True)

        y = 0.97
        self.ax_stats.text(
            0.5,
            y,
            "INKSEP",
            transform=self.ax_stats.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=self.ACCENT,
            fontfamily="monospace",
        )
        y -= 0.04
        fname = os.path.basename(self.image_path)
        if len(fname) > 22:
            fname = fname[:19] + "..."
        self.ax_stats.text(
            0.5,
            y,
            fname,
            transform=self.ax_stats.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            color=self.DIM_COLOR,
        )
        if self.sep_factor > 0:
            y -= 0.03
            self.ax_stats.text(
                0.5,
                y,
                f"sep={self.sep_factor:.2f}",
                transform=self.ax_stats.transAxes,
                ha="center",
                va="top",
                fontsize=7,
                color=self.ACCENT,
                fontfamily="monospace",
            )
        y -= 0.06
        self._gen_text = self.ax_stats.text(
            0.5,
            y,
            "Gen  0 / -",
            transform=self.ax_stats.transAxes,
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=self.TEXT_COLOR,
            fontfamily="monospace",
        )
        y -= 0.065
        self._prog_bar_bg = self.ax_stats.barh(
            y, 1.0, left=0.0, height=0.025, transform=self.ax_stats.transAxes, color=self.DIM_COLOR, zorder=1
        )
        self._prog_bar = self.ax_stats.barh(
            y, 0.0, left=0.0, height=0.025, transform=self.ax_stats.transAxes, color=self.ACCENT, zorder=2
        )
        y -= 0.065
        self.ax_stats.plot(
            [0, 1],
            [y + 0.01, y + 0.01],
            color=self.DIM_COLOR,
            linewidth=0.5,
            transform=self.ax_stats.transAxes,
        )
        y -= 0.01

        def stat_row(label: str, init_val: str, dy: float = 0.06) -> matplotlib.text.Text:
            nonlocal y
            self.ax_stats.text(
                0.08,
                y,
                label,
                transform=self.ax_stats.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color=self.DIM_COLOR,
            )
            t = self.ax_stats.text(
                0.92,
                y,
                init_val,
                transform=self.ax_stats.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                fontweight="bold",
                color=self.TEXT_COLOR,
                fontfamily="monospace",
            )
            y -= dy
            return t

        px_per_mm = self.dpi / 25.4
        self._t_dots = stat_row("Stipples", f"{self.n:,}")
        self._t_size = stat_row(
            "Dot range",
            f"{self.min_dot / px_per_mm:.2f}–{self.max_dot / px_per_mm:.2f}mm",
        )
        self._t_elapsed = stat_row("Elapsed", "0.0s")
        self._t_per_gen = stat_row("Per gen", "-")
        self._t_eta = stat_row("ETA", "-")
        self._t_dots_vis = stat_row("Visible", "-")
        self._t_overlaps = stat_row("Overlaps", "-")

        y -= 0.01
        self.ax_stats.plot(
            [0, 1],
            [y + 0.01, y + 0.01],
            color=self.DIM_COLOR,
            linewidth=0.5,
            transform=self.ax_stats.transAxes,
        )
        y -= 0.02

        self._spark_ax = self.ax_stats.inset_axes((0.06, y - 0.18, 0.88, 0.18))
        self._spark_ax.set_facecolor(self.BG_DARK)
        self._spark_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in self._spark_ax.spines.values():
            sp.set_edgecolor(self.DIM_COLOR)
        self._spark_ax.set_xlim(0, max(self.generations, 1))
        (self._spark_line,) = self._spark_ax.plot([], [], color=self.ACCENT, linewidth=1.2)
        self.ax_stats.text(
            0.5,
            y + 0.005,
            "movement / generation",
            transform=self.ax_stats.transAxes,
            ha="center",
            va="bottom",
            fontsize=7,
            color=self.DIM_COLOR,
        )

        self._status_text = self.ax_stats.text(
            0.5,
            0.02,
            "Initialising…",
            transform=self.ax_stats.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=self.ACCENT,
            style="italic",
        )
        self.fig.text(
            0.5,
            0.975,
            f"Weighted Voronoi Stippling: {self.n:,} points",
            ha="center",
            va="top",
            fontsize=10,
            color=self.TEXT_COLOR,
            fontfamily="monospace",
        )

        if self.interactive:
            plt.ion()
            plt.show(block=False)
            self.fig.canvas.flush_events()

    def set_generations(self, generations: int) -> None:
        """Record the total generation count and rescale the sparkline axis."""
        self.generations = generations
        self._spark_ax.set_xlim(0, max(generations, 1))

    def _count_overlaps(self, particles: np.ndarray, radii: np.ndarray) -> int:
        """Fast overlap count for live display."""
        if len(particles) < 2:
            return 0
        tree = KDTree(particles)
        pairs = tree.query_pairs(self.max_dot, output_type="ndarray")
        if len(pairs) == 0:
            return 0
        i, j = pairs[:, 0], pairs[:, 1]
        d = np.sqrt(((particles[j] - particles[i]) ** 2).sum(axis=1))
        return int(np.sum(d < (radii[i] + radii[j])))

    def update(
        self,
        particles: np.ndarray,
        gen: int,
        elapsed: float,
        gen_time: float,
        movement: float,
    ) -> None:
        """Refresh the preview for the given generation."""
        self._gen_times.append(gen_time)
        self._movements.append(movement)
        self._gen_nums.append(gen)

        H, W = self.brightness.shape
        cutoff_s = 1.0 - self.cutoff
        px = np.clip(np.floor(particles[:, 0]).astype(np.int32), 0, W - 1)
        py = np.clip(np.floor(particles[:, 1]).astype(np.int32), 0, H - 1)
        v = self.brightness[py, px] / 255.0
        if self.invert:
            v = 1.0 - v
        visible = v < cutoff_s
        pts_v = particles[visible]
        v_v = v[visible]
        radii_v = (self.max_dot - v_v * self.dot_scale) / 2.0

        self._dots.set_path(_circles_path(pts_v[:, 0], pts_v[:, 1], radii_v))

        self._gen_text.set_text(f"Gen {gen:3d} / {self.generations}")
        for bar in self._prog_bar:
            bar.set_width(gen / self.generations)

        avg_t = float(np.mean(self._gen_times)) if self._gen_times else gen_time
        eta = avg_t * (self.generations - gen)
        self._t_elapsed.set_text(f"{elapsed:.1f}s")
        self._t_per_gen.set_text(f"{gen_time * 1000:.0f}ms")
        self._t_eta.set_text(f"{eta:.0f}s" if gen < self.generations else "done")
        self._t_dots_vis.set_text(f"{len(pts_v):,} / {self.n:,}")

        if self.sep_factor > 0 and len(pts_v) > 1:
            n_ov = self._count_overlaps(pts_v, radii_v)
            pct = n_ov / max(len(pts_v), 1) * 100
            col = "#7ec87e" if pct < 2.0 else (self.ACCENT if pct < 8.0 else "#e87a7a")
            self._t_overlaps.set_text(f"{n_ov} ({pct:.1f}%)")
            self._t_overlaps.set_color(col)
        else:
            self._t_overlaps.set_text("-")

        if len(self._movements) >= 1:
            self._spark_ax.set_ylim(0, max(self._movements) * 1.1 + 1e-9)
            self._spark_line.set_data(self._gen_nums, self._movements)

        if gen == self.generations:
            self._status_text.set_text("Done, saving SVG…")
            self._status_text.set_color("#7ec87e")
        else:
            self._status_text.set_text(f"Running…  ↓{movement:.2f}px avg move")

        if self.interactive:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        else:
            snap = os.path.join(self.snapshot_dir, f"gen_{gen:04d}.png")
            self.fig.savefig(snap, dpi=80, bbox_inches="tight", facecolor=self.BG_DARK)
            if gen == 1 or gen % 10 == 0 or gen == self.generations:
                logger.info("  [Preview] → %s", snap)

    def done(self, output_path: str | Path) -> None:
        """Finalise the preview after the SVG has been saved."""
        self._status_text.set_text(f"Saved: {os.path.basename(str(output_path))}")
        self._status_text.set_color(self.ACCENT)
        if self.interactive:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            logger.info("\n  [Preview] Window open; close it or Ctrl-C to exit.")
            plt.ioff()
            try:
                plt.show(block=True)
            except KeyboardInterrupt:
                pass
        else:
            snap = os.path.join(self.snapshot_dir, "final.png")
            self.fig.savefig(snap, dpi=100, bbox_inches="tight", facecolor=self.BG_DARK)
            logger.info("  [Preview] Final → %s", snap)
        plt.close(self.fig)
