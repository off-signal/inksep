"""End-to-end CLI behaviour."""

import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from inksep import InkSep
from inksep.cli import main


def _base_args(inp: Path, out: Path, extra: list[str] | None = None) -> list[str]:
    args = [
        str(inp),
        str(out),
        "--no-preview",
        "--stipples",
        "15",
        "--generations",
        "2",
        "--seed",
        "1",
    ]
    return args + (extra or [])


def test_end_to_end_produces_valid_svg(tmp_path, gradient_png):
    out = tmp_path / "out.svg"
    rc = main(_base_args(gradient_png, out))
    assert rc == 0
    assert out.exists()
    root = ET.parse(out).getroot()
    assert root.tag.endswith("svg")


def test_seeded_runs_are_byte_identical(tmp_path, gradient_png):
    out1 = tmp_path / "a.svg"
    out2 = tmp_path / "b.svg"
    assert main(_base_args(gradient_png, out1, ["--seed", "42"])) == 0
    assert main(_base_args(gradient_png, out2, ["--seed", "42"])) == 0
    assert out1.read_bytes() == out2.read_bytes()


def test_generations_zero_is_a_clean_error(tmp_path, gradient_png):
    with pytest.raises(SystemExit) as exc:
        main(_base_args(gradient_png, tmp_path / "out.svg", ["--generations", "0"]))
    assert exc.value.code == 2


def test_stipples_zero_is_a_clean_error(tmp_path, gradient_png):
    with pytest.raises(SystemExit) as exc:
        main(_base_args(gradient_png, tmp_path / "out.svg", ["--stipples", "0"]))
    assert exc.value.code == 2


def test_version_flag():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_uniform_white_image_fails_gracefully(tmp_path, uniform_white_png, capsys):
    rc = main(_base_args(uniform_white_png, tmp_path / "out.svg"))
    assert rc == 1
    assert "uniform white" in capsys.readouterr().err


def test_missing_input_fails_gracefully(tmp_path, capsys):
    rc = main(_base_args(tmp_path / "nope.png", tmp_path / "out.svg"))
    assert rc == 1


def test_python_dash_m_entry_point(tmp_path, gradient_png):
    out = tmp_path / "out.svg"
    proc = subprocess.run(
        [sys.executable, "-m", "inksep", *_base_args(gradient_png, out)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()


def test_library_api_end_to_end(tmp_path, gradient_png):
    gen = InkSep(
        str(gradient_png),
        n_stipples=20,
        border_width=2,
        show_preview=False,
        seed=7,
    )
    gen.run(generations=3, verbose=False)
    out = tmp_path / "lib.svg"
    gen.save_svg(out)
    assert out.exists()
    report = gen.separation_report()
    assert report["n_dots"] == 20
    assert report["global_min_mm"] > 0


def test_dpi_zero_is_a_clean_error(tmp_path, gradient_png):
    with pytest.raises(SystemExit) as exc:
        main(_base_args(gradient_png, tmp_path / "out.svg", ["--dpi", "0"]))
    assert exc.value.code == 2


def test_package_version_fallback(monkeypatch):
    from importlib import metadata

    from inksep.cli import _package_version

    def _raise(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", _raise)
    assert _package_version() == "unknown"


def test_cli_preview_snapshot_flow_under_agg(tmp_path, gradient_png):
    """Full run without --no-preview: Agg backend → headless PNG snapshots."""
    out = tmp_path / "out.svg"
    rc = main(
        [
            str(gradient_png),
            str(out),
            "--stipples",
            "5",
            "--generations",
            "1",
            "--seed",
            "3",
        ]
    )
    assert rc == 0
    assert out.exists()


def test_run_rejects_zero_generations(gradient_png):
    gen = InkSep(str(gradient_png), n_stipples=5, border_width=0, show_preview=False, seed=1)
    with pytest.raises(ValueError, match="generations"):
        gen.run(generations=0)


def test_inksep_preview_integration(tmp_path, gradient_png):
    """InkSep with show_preview=True runs the snapshot preview end to end."""
    gen = InkSep(
        str(gradient_png),
        n_stipples=8,
        border_width=0,
        show_preview=True,
        seed=1,
    )
    assert gen.preview is not None
    gen.run(generations=2, verbose=False)
    out = tmp_path / "with_preview.svg"
    gen.save_svg(out)
    assert out.exists()
    assert (Path(gen.preview.snapshot_dir) / "final.png").exists()


def test_preview_without_matplotlib_errors_cleanly(tmp_path, gradient_png, monkeypatch):
    """A broken install without matplotlib: --no-preview must still work and
    requesting a preview must fail with a helpful message."""
    import types

    broken = types.ModuleType("inksep.preview")  # lacks _INTERACTIVE → ImportError
    monkeypatch.setitem(sys.modules, "inksep.preview", broken)
    out = tmp_path / "out.svg"
    with pytest.raises(SystemExit) as exc:
        main(
            [
                str(gradient_png),
                str(out),
                "--stipples",
                "5",
                "--generations",
                "1",
                "--seed",
                "1",
            ]
        )
    assert exc.value.code == 2


# ── travel optimization + stroke mode (CLI) ─────────────────────────────


def test_optimize_travel_cli_produces_valid_svg(tmp_path, gradient_png):
    out = tmp_path / "out.svg"
    rc = main(_base_args(gradient_png, out, ["--optimize-travel"]))
    assert rc == 0
    assert out.exists()
    root = ET.parse(out).getroot()
    assert root.tag.endswith("svg")


def test_optimize_travel_geometry_matches_plain(tmp_path, gradient_png):
    """--optimize-travel reorders dots but doesn't change any dot's geometry."""
    out_plain = tmp_path / "plain.svg"
    out_tsp = tmp_path / "tsp.svg"
    assert main(_base_args(gradient_png, out_plain)) == 0
    assert main(_base_args(gradient_png, out_tsp, ["--optimize-travel"])) == 0

    radii = lambda p: sorted(re.findall(r'r="([\d.]+)"', p.read_text()))  # noqa: E731
    assert radii(out_tsp) == radii(out_plain)
    assert out_tsp.read_text() != out_plain.read_text()  # order differs


def test_travel_budget_zero_is_clean_error(tmp_path, gradient_png):
    with pytest.raises(SystemExit) as exc:
        main(
            _base_args(gradient_png, tmp_path / "out.svg", ["--travel-budget", "0"])
        )
    assert exc.value.code == 2


def test_stroke_width_cli_emits_stroke_dots(tmp_path, gradient_png):
    out = tmp_path / "out.svg"
    rc = main(_base_args(gradient_png, out, ["--stroke-width", "0.2"]))
    assert rc == 0
    assert out.exists()
    text = out.read_text()
    assert 'fill="none"' in text
    assert 'stroke="black"' in text
    # --stroke-width is mm; SVG stroke-width is px: 0.2mm * 96dpi/25.4 = 0.7559px
    assert 'stroke-width="0.7559"' in text


def test_stroke_width_zero_is_clean_error(tmp_path, gradient_png):
    with pytest.raises(SystemExit) as exc:
        main(_base_args(gradient_png, tmp_path / "out.svg", ["--stroke-width", "0"]))
    assert exc.value.code == 2


def test_library_optimize_travel_and_stroke(tmp_path, gradient_png, caplog):
    """Library API: optimize_travel + stroke_width_mm work end to end."""
    import logging

    gen = InkSep(
        str(gradient_png),
        n_stipples=20,
        border_width=2,
        show_preview=False,
        seed=5,
        optimize_travel=True,
        travel_budget=1.0,
        stroke_width_mm=0.2,
    )
    gen.run(generations=2, verbose=False)
    out = tmp_path / "combo.svg"
    with caplog.at_level(logging.INFO, logger="inksep"):
        gen.save_svg(out)
    assert out.exists()
    text = out.read_text()
    assert 'fill="none"' in text
    assert "pen-up" in caplog.text


def test_library_travel_budget_validation(gradient_png):
    with pytest.raises(ValueError, match="travel_budget"):
        InkSep(str(gradient_png), show_preview=False, travel_budget=0)


def test_library_stroke_width_validation(gradient_png):
    with pytest.raises(ValueError, match="stroke_width_mm"):
        InkSep(str(gradient_png), show_preview=False, stroke_width_mm=0)


# ── units (mm), preflight, dry-run, background ───────────────────────────


def test_mm_length_parser():
    import argparse

    from inksep.cli import _mm_length

    assert _mm_length("0.5") == pytest.approx(0.5)
    assert _mm_length("0.5mm") == pytest.approx(0.5)
    assert _mm_length(" 1.25 MM ") == pytest.approx(1.25)
    with pytest.raises(argparse.ArgumentTypeError, match="px is not supported"):
        _mm_length("0.46px")
    with pytest.raises(argparse.ArgumentTypeError, match="not a number"):
        _mm_length("wide")


@pytest.mark.parametrize("flag,value", [("--min-dot", "0.46px"), ("--border", "6px")])
def test_px_values_are_clean_argparse_errors(tmp_path, gradient_png, capsys, flag, value):
    with pytest.raises(SystemExit) as exc:
        main(_base_args(gradient_png, tmp_path / "out.svg", [flag, value]))
    assert exc.value.code == 2
    assert "px is not supported" in capsys.readouterr().err


def test_min_dot_is_millimetre_diameter(tmp_path, gradient_png, caplog):
    """--min-dot 0.5 (mm) @96dpi → 1.8898 px diameter → r=0.9449 px."""
    import logging

    out = tmp_path / "out.svg"
    with caplog.at_level(logging.INFO, logger="inksep"):
        rc = main(_base_args(gradient_png, out, ["--min-dot", "0.5", "--dot-range", "0"]))
    assert rc == 0
    px = 0.5 * 96.0 / 25.4
    assert f'r="{px / 2:.4f}"' in out.read_text()
    assert "Dot range   : 0.5000mm – 0.5000mm" in caplog.text
    assert "px –" not in caplog.text  # header no longer prints px parentheses


def test_mm_suffix_accepted(tmp_path, gradient_png, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="inksep"):
        rc = main(
            _base_args(
                gradient_png,
                tmp_path / "out.svg",
                ["--dry-run", "--min-dot", "0.5mm", "--dot-range", "0"],
            )
        )
    assert rc == 0
    assert "Dot range   : 0.5000mm – 0.5000mm" in caplog.text


def test_dry_run_prints_preflight_and_writes_no_file(tmp_path, gradient_png, caplog):
    import logging

    out = tmp_path / "out.svg"
    with caplog.at_level(logging.INFO, logger="inksep"):
        rc = main(_base_args(gradient_png, out, ["--dry-run"]))
    assert rc == 0
    assert not out.exists()
    text = caplog.text
    assert "Document    : " in text and "96dpi" in text
    assert "dots/cm²" in text
    assert "Dot range   : 0.4630mm – 2.3151mm" in text
    assert "Dry run" in text
    assert "move=" not in text  # no generations ran
    assert "SVG saved" not in text


def test_preflight_header_reports_document_and_density(tmp_path, gradient_png, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="inksep"):
        assert main(_base_args(gradient_png, tmp_path / "out.svg", ["--dry-run"])) == 0
    assert "Document    : 21.17mm × 15.87mm  (80×60px @ 96dpi)" in caplog.text
    assert "Stipples    : 15  (" in caplog.text


def test_border_mm_keeps_dots_inside_inset_bounds(tmp_path, gradient_png):
    """--border 1.5 mm → round(3.559) = 4 px inset at 96 dpi."""
    out = tmp_path / "out.svg"
    assert main(_base_args(gradient_png, out, ["--border", "1.5"])) == 0
    xs = [float(m) for m in re.findall(r'cx="([\d.]+)"', out.read_text())]
    assert min(xs) >= 4.0 - 1e-6
    assert max(xs) <= 76.0 + 1e-6


def test_no_background_omits_rect_default_keeps_it(tmp_path, gradient_png):
    ns = {"svg": "http://www.w3.org/2000/svg"}
    out = tmp_path / "out.svg"
    assert main(_base_args(gradient_png, out, ["--no-background"])) == 0
    root = ET.parse(out).getroot()
    assert root.find("svg:rect", ns) is None
    assert root.findall("svg:g/svg:circle", ns)

    out2 = tmp_path / "with_bg.svg"
    assert main(_base_args(gradient_png, out2)) == 0
    assert ET.parse(out2).getroot().find("svg:rect", ns) is not None


def test_preflight_pen_margin_ok(tmp_path, gradient_png, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="inksep"):
        rc = main(
            _base_args(gradient_png, tmp_path / "out.svg", ["--dry-run", "--stroke-width", "0.30"])
        )
    assert rc == 0
    assert "Dot style   : stroke 0.30mm" in caplog.text
    assert "Pen margin  : pen 0.30mm → min dot 0.46mm: OK" in caplog.text


def test_preflight_pen_margin_flags_clamp(tmp_path, gradient_png, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="inksep"):
        rc = main(
            _base_args(gradient_png, tmp_path / "out.svg", ["--dry-run", "--stroke-width", "0.6"])
        )
    assert rc == 0
    assert "Pen margin  : pen 0.60mm → min dot 0.46mm" in caplog.text
    assert "clamping expected" in caplog.text
    assert "exact count printed after the run" in caplog.text


def test_travel_budget_help_documents_between_passes():
    from inksep.cli import _build_parser

    flat = " ".join(_build_parser().format_help().split())
    assert "between passes" in flat
    assert "overshoot" in flat


def test_help_documents_mm_and_source_pixel_units():
    from inksep.cli import _build_parser

    help_text = _build_parser().format_help()
    assert "--min-dot MM" in help_text
    assert "--border MM" in help_text
    assert "--step PX" in help_text
    assert "source pixels" in help_text
    assert "px is rejected" in help_text


def test_library_pen_clamp_stats_variants(gradient_png):
    plain = InkSep(str(gradient_png), n_stipples=10, show_preview=False, seed=1)
    assert plain.pen_clamp_stats() is None  # no stroke → nothing to report

    stroked = InkSep(
        str(gradient_png), n_stipples=10, show_preview=False, seed=1, stroke_width_mm=0.2
    )
    stats = stroked.pen_clamp_stats()
    assert stats is not None
    n, min_d = stats
    assert n >= 0
    assert min_d > 0

    # cutoff removes every dot → empty-layout branch
    empty = InkSep(
        str(gradient_png),
        n_stipples=10,
        show_preview=False,
        seed=1,
        stroke_width_mm=0.2,
        white_cutoff=1.0,
    )
    assert empty.pen_clamp_stats() == (0, pytest.approx(1.75 / (96.0 / 25.4)))


def test_library_background_flag(tmp_path, gradient_png):
    gen = InkSep(
        str(gradient_png),
        n_stipples=10,
        border_width=2,
        show_preview=False,
        seed=1,
        background=False,
    )
    gen.run(generations=1, verbose=False)
    out = tmp_path / "no_bg.svg"
    gen.save_svg(out)
    assert "<rect" not in out.read_text()


def test_pen_clamp_warning_when_dots_fall_below_pen(tmp_path, gradient_png, caplog):
    """Dots thinner than the pen: warning fires with exact counts, and the
    SVG comments switch to pen-limited diameters."""
    import logging

    gen = InkSep(
        str(gradient_png),
        n_stipples=10,
        border_width=2,
        show_preview=False,
        seed=1,
        min_dot_size=1.0,  # every dot = 1 px = 0.26 mm …
        dot_size_factor=0.0,
        stroke_width_mm=0.6,  # … below the 0.6 mm pen
    )
    gen.run(generations=1, verbose=False)
    out = tmp_path / "clamped.svg"
    with caplog.at_level(logging.WARNING, logger="inksep"):
        gen.save_svg(out)
    assert "Pen clamp" in caplog.text
    assert "clamped dots will render at pen width" in caplog.text
    text = out.read_text()
    assert "pen-limited" in text
    assert 'stroke-width="2.2677"' in text  # 0.6 mm @ 96 dpi in px


def test_damping_help_explains_what_it_does():
    from inksep.cli import _build_parser

    help_text = _build_parser().format_help()
    assert "--damping 0-1" in help_text
    for word in ("inertia", "centroid", "jitter", "separation", "generations"):
        assert word in help_text.lower()


def test_tol_negative_is_a_clean_error(tmp_path, gradient_png):
    with pytest.raises(SystemExit) as exc:
        main(_base_args(gradient_png, tmp_path / "out.svg", ["--tol", "-0.1"]))
    assert exc.value.code == 2


def test_library_run_tol_zero_runs_all_generations(gradient_png, caplog):
    import logging

    gen = InkSep(
        str(gradient_png), n_stipples=20, border_width=2, show_preview=False, seed=7
    )
    with caplog.at_level(logging.INFO, logger="inksep"):
        completed = gen.run(generations=4, verbose=False)
    assert completed == 4
    assert "Early stop" not in caplog.text


def test_library_run_huge_tol_stops_after_three_streak(tmp_path, gradient_png, caplog):
    """Early stop skips the tail; the separation report still runs."""
    import logging

    gen = InkSep(
        str(gradient_png),
        n_stipples=20,
        border_width=2,
        show_preview=False,
        seed=7,
        sep_factor=0.5,
    )
    with caplog.at_level(logging.INFO, logger="inksep"):
        completed = gen.run(generations=50, tol=1e6)
    assert completed == 3
    assert "Early stop : gen 3" in caplog.text
    assert "SEPARATION REPORT" in caplog.text
    out = tmp_path / "early.svg"
    gen.save_svg(out)
    assert out.exists()


def test_library_run_negative_tol_raises(gradient_png):
    gen = InkSep(str(gradient_png), n_stipples=20, show_preview=False, seed=7)
    with pytest.raises(ValueError, match="tol"):
        gen.run(generations=2, tol=-0.1)


def test_tol_cli_preflight_stop_and_total(tmp_path, gradient_png, caplog):
    """--tol shows in preflight, stops at the streak, and Total time
    averages over generations actually run."""
    import logging

    out = tmp_path / "out.svg"
    extra = ["--generations", "50", "--tol", "1e6"]
    with caplog.at_level(logging.INFO, logger="inksep"):
        rc = main(_base_args(gradient_png, out, extra))
    assert rc == 0
    assert out.exists()
    text = caplog.text
    assert "Early stop  : when mean move < 1000000.0000px" in text
    assert "Early stop : gen 3" in text
    assert "Total time" in text


def test_tol_help_explains_early_stop():
    from inksep.cli import _build_parser

    help_text = _build_parser().format_help()
    assert "--tol PX" in help_text
    normalized = " ".join(help_text.split())
    assert (
        "mean particle movement stays below TOL px for 3 consecutive generations"
        in normalized
    )
