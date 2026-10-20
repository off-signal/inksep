"""Separation quality report: mm math, overlap detection (unique dots vs pairs)."""

import numpy as np
import pytest

from inksep.core import SeparationReport, compute_separation_report, format_separation_report


def _report(particles, brightness, **kwargs) -> SeparationReport:
    defaults = dict(max_dot=8.0, dot_scale=0.0, sep_factor=0.85, dpi=25.4, invert=False)
    defaults.update(kwargs)
    return compute_separation_report(particles, brightness, **defaults)


def test_known_distances_in_mm():
    """dpi=25.4 → 1 px/mm; two black dots 10 px apart."""
    black = np.zeros((30, 30), dtype=np.float32)
    pts = np.array([[10.0, 10.0], [20.0, 10.0]])
    r = _report(pts, black)
    assert r["global_min_mm"] == pytest.approx(10.0)
    assert r["max_dot_mm"] == pytest.approx(8.0)
    assert r["ratio"] == pytest.approx(1.25)
    assert r["n_dots"] == 2
    assert r["n_overlap"] == 0
    assert r["n_overlap_pairs"] == 0
    assert r["pct_overlap"] == 0.0


def test_overlap_detected_with_depth():
    """Two dots 5 px apart, each 4 px radius → 3 px overlap depth.

    Both dots touch each other, so the unique-dot count is 2 while the pair
    count is 1.
    """
    black = np.zeros((30, 30), dtype=np.float32)
    pts = np.array([[10.0, 10.0], [15.0, 10.0]])
    r = _report(pts, black)
    assert r["n_overlap"] == 2
    assert r["n_overlap_pairs"] == 1
    assert r["pct_overlap"] == pytest.approx(100.0)
    assert r["max_depth_mm"] == pytest.approx(3.0)
    assert r["ratio"] == pytest.approx(5.0 / 8.0)


def test_unique_dots_and_pairs_are_distinct_counts():
    """A-B-C chain: 3 dots touch, but only 2 overlapping pairs; the report
    must not conflate 'dots touching a neighbour' with 'overlapping pairs'
    (dividing pairs by dots was dimensionally nonsense)."""
    black = np.zeros((30, 30), dtype=np.float32)
    pts = np.array([[2.0, 10.0], [7.0, 10.0], [12.0, 10.0]])
    r = _report(pts, black)  # r=4 each; gaps 5 < 8 overlap, ends 10 > 8 don't
    assert r["n_overlap_pairs"] == 2
    assert r["n_overlap"] == 3
    assert r["pct_overlap"] == pytest.approx(100.0)


def test_report_keys_and_types():
    black = np.zeros((20, 20), dtype=np.float32)
    pts = np.array([[5.0, 5.0], [12.0, 8.0], [15.0, 15.0]])
    r = _report(pts, black)
    assert set(r) == {
        "global_min_mm",
        "global_mean_mm",
        "max_dot_mm",
        "ratio",
        "n_overlap",
        "n_overlap_pairs",
        "pct_overlap",
        "max_depth_mm",
        "dark_mean_mm",
        "dark_min_mm",
        "n_dots",
    }
    assert isinstance(r["n_overlap"], int)
    assert isinstance(r["n_overlap_pairs"], int)
    assert isinstance(r["n_dots"], int)
    for key, value in r.items():
        assert isinstance(value, float | int), key


def test_single_particle_does_not_crash():
    black = np.zeros((20, 20), dtype=np.float32)
    pts = np.array([[10.0, 10.0]])
    r = _report(pts, black)
    assert r["n_dots"] == 1
    assert r["n_overlap"] == 0
    assert r["n_overlap_pairs"] == 0


def test_invert_affects_radii_and_overlap():
    """Inverted: v=0 becomes bright → smaller radii → far-apart dots don't overlap."""
    black = np.zeros((30, 30), dtype=np.float32)
    pts = np.array([[10.0, 10.0], [17.0, 10.0]])  # 7 px apart
    # dot_scale=8 → invert shrinks radius from 4 px to 0 px
    normal = _report(pts, black, invert=False, dot_scale=8.0)
    inverted = _report(pts, black, invert=True, dot_scale=8.0)
    assert normal["n_overlap"] == 2  # r = 4 + 4 > 7, both dots touch
    assert normal["n_overlap_pairs"] == 1
    assert inverted["n_overlap"] == 0  # r = 0 + 0 < 7
    assert inverted["n_overlap_pairs"] == 0


def test_overlap_in_separation_band_is_counted():
    """Regression: pairs closer than max_dot but farther than sep_factor*max_dot
    still physically overlap and must be reported; a smaller search radius
    would miss exactly this band."""
    black = np.zeros((30, 30), dtype=np.float32)
    # 7.8 px apart with 8 px diameters → overlap, but > 0.85 * 8 = 6.8 search
    pts = np.array([[10.0, 10.0], [17.8, 10.0]])
    r = _report(pts, black, sep_factor=0.85)
    assert r["n_overlap_pairs"] == 1
    assert r["n_overlap"] == 2
    assert r["max_depth_mm"] == pytest.approx(0.2)


def test_format_report_is_printable():
    black = np.zeros((30, 30), dtype=np.float32)
    pts = np.array([[10.0, 10.0], [20.0, 10.0]])
    text = format_separation_report(_report(pts, black))
    assert "SEPARATION REPORT" in text
    assert "CLEAN" in text
    assert "mm" in text


def test_format_report_states_touching_dots_and_pairs():
    black = np.zeros((30, 30), dtype=np.float32)
    pts = np.array([[10.0, 10.0], [15.0, 10.0]])
    text = format_separation_report(_report(pts, black))
    assert "Overlapping dots  : 2 / 2 (100.0%) touch ≥1 neighbor" in text
    assert "from 1 pairs, max merge depth 3.0000 mm" in text
    assert "Max overlap depth" not in text
