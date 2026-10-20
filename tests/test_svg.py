"""SVG rendering: structure, units, cutoff, invert, determinism, order, stroke."""

import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from inksep.config import StippleConfig
from inksep.svg import render_svg, visible_mask, write_svg

SVG_NS = "http://www.w3.org/2000/svg"


def _cfg(**kwargs) -> StippleConfig:
    defaults: dict = dict(
        n_stipples=2,
        min_dot_size=2.0,
        dot_size_factor=1.0,  # max_dot = 4, dot_scale = 2
        dpi=25.4,  # 1 px/mm
        border_width=0,
    )
    defaults.update(kwargs)
    return StippleConfig(**defaults)


BRIGHTNESS = np.array([[0.0, 255.0], [255.0, 0.0]], dtype=np.float32)
PARTICLES = np.array([[0.5, 0.5], [1.5, 1.5]])


def test_valid_xml_and_root():
    root = ET.fromstring(render_svg(PARTICLES, BRIGHTNESS, _cfg()))
    assert root.tag == f"{{{SVG_NS}}}svg"


def test_dimensions_in_mm_at_dpi():
    root = ET.fromstring(render_svg(PARTICLES, BRIGHTNESS, _cfg(dpi=25.4)))
    assert root.attrib["width"] == "2.0000mm"
    assert root.attrib["height"] == "2.0000mm"
    assert root.attrib["viewBox"] == "0 0 2 2"
    # 96 dpi: 2 px / (96/25.4 px/mm)
    root96 = ET.fromstring(render_svg(PARTICLES, BRIGHTNESS, _cfg(dpi=96.0)))
    expected = 2 / (96.0 / 25.4)
    assert root96.attrib["width"] == f"{expected:.4f}mm"


def test_circle_count_and_radius_formula():
    text = render_svg(PARTICLES, BRIGHTNESS, _cfg())
    assert text.count("<circle ") == 2
    # Dark pixel (v=0) → r = max_dot/2 = 2 px = 2 mm at 25.4 dpi → d = 4 mm
    assert 'r="2.0000"' in text
    assert "<!-- d=4.0000mm -->" in text


def test_per_dot_comment_is_diameter_mm_not_radius():
    text = render_svg(PARTICLES, BRIGHTNESS, _cfg())
    comments = [c.strip() for c in re.findall(r"<!--(.*?)-->", text, re.S)]
    dot_comments = [c for c in comments if not c.startswith("inksep")]
    assert len(dot_comments) == 2
    assert all(c.startswith("d=") for c in dot_comments)
    assert all(c.endswith("mm") for c in dot_comments)


def test_cutoff_suppresses_bright_dots():
    # cutoff 0.9 → only v < 0.1 survives; the two dark dots remain, bright gone
    text = render_svg(PARTICLES, BRIGHTNESS, _cfg(white_cutoff=0.9))
    assert text.count("<circle ") == 2  # both particles sit on dark pixels
    # particle on a bright pixel only: drop the dark one
    text2 = render_svg(np.array([[1.5, 0.5]]), BRIGHTNESS, _cfg(white_cutoff=0.9))
    assert text2.count("<circle ") == 0


def test_invert_colors():
    # In invert mode only particles on bright pixels are drawn (dark = background)
    bright_particles = np.array([[1.5, 0.5], [0.5, 1.5]])
    root = ET.fromstring(render_svg(bright_particles, BRIGHTNESS, _cfg(invert=True)))
    ns = {"svg": SVG_NS}
    rect = root.find("svg:rect", ns)
    assert rect is not None and rect.attrib["fill"] == "black"
    circles = root.findall("svg:g/svg:circle", ns)
    assert len(circles) == 2
    assert all(c.attrib["fill"] == "white" for c in circles)


def test_header_comment_contains_run_metadata():
    text = render_svg(PARTICLES, BRIGHTNESS, _cfg(n_stipples=42, sep_factor=0.9))
    assert "<!-- inksep | 42 stipples" in text
    assert "sep_factor=0.9" in text
    assert "dpi=25.4" in text
    # dot range in the header is mm diameters, never px
    assert "dots 2.0000–4.0000mm" in text
    header = re.search(r"<!-- inksep.*?-->", text, re.S)
    assert header is not None
    assert "px" not in header.group()


def test_root_desc_declares_reposition_not_scale():
    root = ET.fromstring(render_svg(PARTICLES, BRIGHTNESS, _cfg()))
    ns = {"svg": SVG_NS}
    desc = root.find("svg:desc", ns)
    assert desc is not None and desc.text is not None
    assert "ink diameter = 2r" in desc.text
    assert "reposition this group" in desc.text
    assert "do not scale" in desc.text


def test_no_background_omits_rect():
    text = render_svg(PARTICLES, BRIGHTNESS, _cfg(), background=False)
    root = ET.fromstring(text)
    ns = {"svg": SVG_NS}
    assert root.find("svg:rect", ns) is None
    assert len(root.findall("svg:g/svg:circle", ns)) == 2
    # default keeps the background rect
    with_bg = ET.fromstring(render_svg(PARTICLES, BRIGHTNESS, _cfg()))
    assert with_bg.find("svg:rect", ns) is not None


def test_write_svg_background_flag(tmp_path):
    out = tmp_path / "out.svg"
    write_svg(PARTICLES, BRIGHTNESS, _cfg(), out, background=False)
    assert "<rect" not in out.read_text()


def test_determinism():
    a = render_svg(PARTICLES, BRIGHTNESS, _cfg())
    b = render_svg(PARTICLES, BRIGHTNESS, _cfg())
    assert a == b


def test_write_svg_summary(tmp_path):
    out = tmp_path / "out.svg"
    summary = write_svg(PARTICLES, BRIGHTNESS, _cfg(), out)
    assert out.exists()
    assert summary.rendered == 2
    assert summary.total == 2
    assert summary.svg_w_mm == pytest.approx(2.0)
    assert summary.min_r_mm == pytest.approx(1.0)
    assert summary.max_r_mm == pytest.approx(2.0)


def test_cutoff_removed_count(tmp_path):
    out = tmp_path / "out.svg"
    # One particle on the bright pixel (1.5, 0.5) → removed by cutoff
    pts = np.array([[0.5, 0.5], [1.5, 0.5]])
    summary = write_svg(pts, BRIGHTNESS, _cfg(white_cutoff=0.9), out)
    assert summary.rendered == 1
    assert summary.total == 2


# ── travel-order reordering ─────────────────────────────────────────────


def _circle_coords(text: str) -> list[tuple[float, float]]:
    return [
        (float(m.group(1)), float(m.group(2)))
        for m in re.finditer(r'cx="([\d.]+)" cy="([\d.]+)"', text)
    ]


def test_order_reorders_emission_only():
    """order permutes visible dots; geometry (radii set) is unchanged."""
    base = render_svg(PARTICLES, BRIGHTNESS, _cfg())
    reordered = render_svg(PARTICLES, BRIGHTNESS, _cfg(), order=[1, 0])
    assert _circle_coords(reordered) == list(reversed(_circle_coords(base)))
    # Same multiset of radii
    radii = lambda t: sorted(re.findall(r'r="([\d.]+)"', t))  # noqa: E731
    assert radii(base) == radii(reordered)


def test_order_with_cutoff_indexes_visible_only():
    # Only one dot survives cutoff → order length must be 1, not 2
    pts = np.array([[0.5, 0.5], [1.5, 0.5]])
    cfg = _cfg(white_cutoff=0.9)
    assert int(visible_mask(pts, BRIGHTNESS, cfg).sum()) == 1
    text = render_svg(pts, BRIGHTNESS, cfg, order=[0])
    assert text.count("<circle ") == 1
    assert "0.500" in text  # the surviving dot at (0.5, 0.5)


def test_order_length_mismatch_rejected():
    with pytest.raises(ValueError, match="order length"):
        render_svg(PARTICLES, BRIGHTNESS, _cfg(), order=[0, 1, 2])


def test_order_none_is_byte_identical():
    a = render_svg(PARTICLES, BRIGHTNESS, _cfg())
    b = render_svg(PARTICLES, BRIGHTNESS, _cfg(), order=None)
    assert a == b


# ── stroke (plotter) mode ───────────────────────────────────────────────


def test_stroke_mode_emits_outline_circles():
    text = render_svg(PARTICLES, BRIGHTNESS, _cfg(), stroke_width=0.5)
    assert text.count('fill="none"') == 2
    assert 'stroke-width="0.5000"' in text
    assert 'stroke="black"' in text


def test_stroke_mode_preserves_outer_diameter():
    """draw_r + W/2 == nominal r: outer ink edge stays on the measurement."""
    cfg = _cfg()  # max_dot=4, dot_scale=2; dark dot → nominal r = 2.0 px
    W = 0.5
    text = render_svg(PARTICLES, BRIGHTNESS, cfg, stroke_width=W)
    m = re.search(r'r="([\d.]+)" fill="none"', text)
    assert m is not None
    draw_r = float(m.group(1))
    nominal_r = 2.0  # dark pixel: (max_dot - 0 * dot_scale) / 2 = 4/2
    assert draw_r + W / 2.0 == pytest.approx(nominal_r, abs=1e-4)
    # dot wider than the pen → comment reports the nominal ink diameter
    assert "<!-- d=4.0000mm -->" in text
    assert "pen-limited" not in text


def test_stroke_mode_clamps_tiny_dots():
    """A dot smaller than the pen still emits (clamped radius, not skipped)
    and its comment reports the pen-limited diameter plus the nominal one."""
    cfg = _cfg(min_dot_size=0.1, dot_size_factor=0.0)  # all dots r = 0.05 px
    text = render_svg(PARTICLES, BRIGHTNESS, cfg, stroke_width=1.0)
    radii = [float(m) for m in re.findall(r'r="([\d.]+)" fill="none"', text)]
    assert all(r >= 0.001 for r in radii)
    # 1 px stroke at 25.4 dpi = 1.0000 mm; nominal d = 0.1 px = 0.1000 mm
    assert "<!-- d=1.0000mm (pen-limited; nominal 0.1000mm) -->" in text


def test_stroke_invert_uses_white_stroke():
    bright_particles = np.array([[1.5, 0.5], [0.5, 1.5]])
    text = render_svg(bright_particles, BRIGHTNESS, _cfg(invert=True), stroke_width=0.3)
    assert 'stroke="white"' in text
    assert 'fill="none"' in text
