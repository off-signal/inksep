"""Preview window smoke tests (headless Agg; skipped without matplotlib)."""

import re

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")
from matplotlib.patches import Rectangle  # noqa: E402

from inksep.preview import PreviewWindow  # noqa: E402


def _bg_rects(pw):
    return [p for p in pw.ax_main.patches if isinstance(p, Rectangle)]


def _make_preview(tmp_path, brightness, **kwargs):
    defaults = dict(
        brightness=brightness,
        n_stipples=kwargs.pop("n_stipples", 2),
        generations=kwargs.pop("generations", 1),
        invert=False,
        min_dot=2.0,
        max_dot=4.0,
        dot_scale=2.0,
        cutoff=0.0,
        image_path="t.png",
        snapshot_dir=tmp_path / "snaps",
    )
    defaults.update(kwargs)
    return PreviewWindow(**defaults)


def _path_radii(pw):
    """Extract each dot's radius (in image-pixel data units) from the path."""
    return [np.ptp(poly[:, 0]) / 2.0 for poly in pw._dots.get_path().to_polygons()]


def _particles(rng, n=50, w=60, h=40):
    return np.column_stack([rng.uniform(0, w, n), rng.uniform(0, h, n)])


@pytest.fixture
def preview(tmp_path):
    h, w = 40, 60
    rng = np.random.default_rng(0)
    brightness = (rng.random((h, w)) * 255).astype(np.float32)
    pw = _make_preview(
        tmp_path,
        brightness,
        n_stipples=50,
        generations=0,
        min_dot=1.75,
        max_dot=8.75,
        dot_scale=7.0,
        sep_factor=0.85,
    )
    yield pw
    plt.close(pw.fig)


def test_headless_construction(preview):
    assert not preview.interactive  # Agg
    assert preview.n == 50


def test_canvas_is_pure_background_with_no_photo(tmp_path):
    """The preview canvas mirrors the SVG: plain background, no image layer."""
    brightness = np.zeros((20, 30), dtype=np.float32)
    pw = _make_preview(tmp_path, brightness, invert=False)
    try:
        assert len(pw.ax_main.images) == 0
        rects = _bg_rects(pw)
        assert len(rects) == 1  # background rectangle only
        assert rects[0].get_facecolor()[:3] == pytest.approx((1.0, 1.0, 1.0))
        assert pw._dots in pw.ax_main.patches
    finally:
        plt.close(pw.fig)


def test_canvas_background_inverts(tmp_path):
    brightness = np.zeros((20, 30), dtype=np.float32)
    pw = _make_preview(tmp_path, brightness, invert=True)
    try:
        assert _bg_rects(pw)[0].get_facecolor()[:3] == pytest.approx((0.0, 0.0, 0.0))
    finally:
        plt.close(pw.fig)


def test_set_generations_rescales_sparkline(preview):
    """Regression: sparkline x-axis used to stay frozen at (0, 1)."""
    preview.set_generations(8)
    xlim = preview._spark_ax.get_xlim()
    assert xlim == pytest.approx((0.0, 8.0))


def test_update_cycle(preview):
    rng = np.random.default_rng(1)
    preview.set_generations(3)
    for gen in range(1, 4):
        preview.update(_particles(rng), gen, elapsed=gen * 0.1, gen_time=0.05, movement=1.0)
    assert preview._gen_nums == [1, 2, 3]
    assert "3 / 3" in preview._gen_text.get_text()


def test_count_overlaps(tmp_path):
    particles = np.array([[10.0, 10.0], [11.0, 10.0], [50.0, 50.0]])
    radii = np.array([4.0, 4.0, 4.0])
    pw = _make_preview(
        tmp_path, np.zeros((60, 60), dtype=np.float32), min_dot=8.0, max_dot=8.0, dot_scale=0.0, sep_factor=0.85
    )
    try:
        assert pw._count_overlaps(particles, radii) == 1
    finally:
        plt.close(pw.fig)


def test_count_overlaps_edge_cases(tmp_path):
    pw = _make_preview(
        tmp_path, np.zeros((20, 20), dtype=np.float32), min_dot=8.0, max_dot=8.0, dot_scale=0.0, sep_factor=0.85
    )
    try:
        radii = np.array([4.0, 4.0])
        assert pw._count_overlaps(np.array([[5.0, 5.0]]), radii[:1]) == 0  # < 2 dots
        far = np.array([[2.0, 2.0], [18.0, 18.0]])
        assert pw._count_overlaps(far, radii) == 0  # no pair within max_dot
    finally:
        plt.close(pw.fig)


def test_done_saves_final_snapshot(preview, tmp_path):
    preview.set_generations(1)
    rng = np.random.default_rng(2)
    preview.update(_particles(rng), 1, 0.1, 0.05, 0.5)
    preview.done(tmp_path / "final.svg")
    assert (tmp_path / "snaps" / "final.png").exists()
    assert not plt.fignum_exists(preview.fig.number)


def test_long_filename_is_truncated(tmp_path):
    pw = _make_preview(
        tmp_path,
        np.zeros((20, 30), dtype=np.float32),
        min_dot=1.0,
        max_dot=2.0,
        dot_scale=1.0,
        image_path="a_very_long_image_filename_that_exceeds.png",
    )
    try:
        texts = [t.get_text() for t in pw.ax_stats.texts]
        assert any(t.endswith("...") for t in texts)
    finally:
        plt.close(pw.fig)


def test_update_invert_and_empty_visibility(tmp_path):
    pw = _make_preview(
        tmp_path,
        np.full((20, 20), 255.0, dtype=np.float32),
        invert=True,
        sep_factor=0.0,  # overlaps column shows "-"
    )
    try:
        pw.set_generations(1)
        pw.update(np.array([[5.0, 5.0], [10.0, 10.0]]), 1, 0.1, 0.05, 1.0)
        assert pw._t_overlaps.get_text() == "-"
    finally:
        plt.close(pw.fig)


def test_update_suppressed_by_cutoff(tmp_path):
    pw = _make_preview(
        tmp_path,
        np.zeros((20, 20), dtype=np.float32),
        cutoff=1.0,  # cutoff_s = 0 → nothing visible
    )
    try:
        pw.set_generations(1)
        pw.update(np.array([[5.0, 5.0]]), 1, 0.1, 0.05, 1.0)
        assert len(pw._dots.get_path().to_polygons()) == 0
    finally:
        plt.close(pw.fig)


def test_dot_radii_use_exact_svg_formula(tmp_path):
    """Preview circles must carry the exact per-dot radius, in data units."""
    brightness = np.array([[0.0, 255.0], [128.0, 64.0]], dtype=np.float32)
    pw = _make_preview(tmp_path, brightness, max_dot=4.0, dot_scale=2.0)
    try:
        pts = np.array([[0.5, 0.5], [1.5, 1.5], [0.5, 1.5]])
        pw.set_generations(1)
        pw.update(pts, 1, 0.1, 0.05, 1.0)
        # Sampled pixels: (0.5,0.5)→[0,0]=0, (1.5,1.5)→[1,1]=64, (0.5,1.5)→[1,0]=128
        v = np.array([0.0, 64.0, 128.0]) / 255.0
        expected = (4.0 - v * 2.0) / 2.0
        actual = np.array(_path_radii(pw))
        assert actual == pytest.approx(expected, abs=1e-6)
    finally:
        plt.close(pw.fig)


def test_preview_matches_svg_geometry_exactly(tmp_path, gradient_png):
    """End-to-end parity: every <circle r=...> in the exported SVG equals the
    corresponding preview dot radius (SVG rounds r to 4 decimals)."""
    from inksep import InkSep
    from inksep.svg import render_svg

    gen = InkSep(
        str(gradient_png),
        n_stipples=40,
        border_width=2,
        show_preview=False,
        white_cutoff=0.2,
        seed=11,
    )
    gen.run(generations=2, verbose=False)

    svg_text = render_svg(gen.particles, gen.brightness, gen.config)
    svg_r = [float(m) for m in re.findall(r'r="([\d.]+)"', svg_text)]

    pw = _make_preview(
        tmp_path,
        gen.brightness,
        n_stipples=40,
        generations=2,
        min_dot=gen.config.min_dot_size,
        max_dot=gen.config.max_dot,
        dot_scale=gen.config.dot_scale,
        cutoff=gen.config.white_cutoff,
        image_path=str(gradient_png),
        sep_factor=gen.config.sep_factor,
    )
    try:
        pw.set_generations(2)
        pw.update(gen.particles, 2, 0.1, 0.05, 0.5)
        preview_r = _path_radii(pw)
        assert len(preview_r) == len(svg_r) > 0
        assert preview_r == pytest.approx(svg_r, abs=1e-4)
    finally:
        plt.close(pw.fig)


def test_dots_are_opaque(tmp_path):
    pw = _make_preview(tmp_path, np.zeros((20, 30), dtype=np.float32))
    try:
        assert pw._dots.get_alpha() == 1.0
    finally:
        plt.close(pw.fig)


def test_dot_range_overlay_reports_mm(tmp_path):
    """The stats overlay reports the dot range as mm diameters (25.4 dpi → 1 px/mm)."""
    brightness = np.zeros((20, 30), dtype=np.float32)
    pw = _make_preview(tmp_path, brightness, min_dot=2.0, max_dot=4.0, dpi=25.4)
    try:
        assert pw._t_size.get_text() == "2.00–4.00mm"
    finally:
        plt.close(pw.fig)
    # default 96 dpi: 1.75–8.75 px → 0.46–2.32 mm
    pw2 = _make_preview(tmp_path, brightness, min_dot=1.75, max_dot=8.75)
    try:
        assert pw2._t_size.get_text() == "0.46–2.32mm"
    finally:
        plt.close(pw2.fig)
