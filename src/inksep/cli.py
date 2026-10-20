"""Command-line interface for inksep."""

from __future__ import annotations

import argparse
import configparser
import difflib
import logging
import sys
import time
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

from .generator import InkSep

logger = logging.getLogger("inksep")

_DEFAULT_DPI = 96.0
_DEFAULT_MIN_DOT_MM = 1.75 * 25.4 / _DEFAULT_DPI  # 1.75 px at 96 dpi
_DEFAULT_BORDER_MM = 6 * 25.4 / _DEFAULT_DPI  # 6 px at 96 dpi

_PRESET_FILENAME = "inksep.ini"
_FORBIDDEN_PRESET_KEYS = frozenset({"preset", "version", "dry-run", "help"})
_PRESET_TRUE = frozenset({"1", "yes", "true", "on"})
_PRESET_FALSE = frozenset({"0", "no", "false", "off"})


def _mm_length(text: str) -> float:
    """Parse a physical length in millimetres: a bare number or an ``mm`` suffix.
    Pixel values are rejected so physical sizes never depend on rendering resolution.
    """
    t = text.strip().lower()
    if t.endswith("px"):
        raise argparse.ArgumentTypeError(
            f"{text!r}: px is not supported; give millimetres "
            "(e.g. 0.46 or 0.46mm)"
        )
    if t.endswith("mm"):
        t = t[: -len("mm")].strip()
    try:
        return float(t)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a number (millimetres expected)"
        ) from None


class _PresetError(Exception):
    """A problem with inksep.ini; the message is user-facing."""


def _preset_option_names(ap: argparse.ArgumentParser) -> list[str]:
    return [opt[2:] for opt in ap._option_string_actions if opt.startswith("--")]


def _coerce_preset_keys(
    raw_items: dict[str, str], section: str, ap: argparse.ArgumentParser
) -> dict[str, object]:
    """Convert one section's raw strings into typed parser values, keyed by dest."""
    out: dict[str, object] = {}
    for key, raw in raw_items.items():
        norm = key.replace("_", "-")
        if norm in _FORBIDDEN_PRESET_KEYS:
            raise _PresetError(
                f"{_PRESET_FILENAME} [{section}]: '{key}' is not allowed in a preset"
            )
        action = ap._option_string_actions.get(f"--{norm}")
        if action is None or action.dest == argparse.SUPPRESS:
            near = difflib.get_close_matches(norm, _preset_option_names(ap), n=1)
            hint = f" (did you mean '{near[0]}'?)" if near else ""
            raise _PresetError(
                f"{_PRESET_FILENAME} [{section}]: unknown option '{key}'{hint}"
            )
        value = raw.strip()
        typ = action.type
        if typ is None:
            token = value.lower()
            if token in _PRESET_TRUE:
                out[action.dest] = True
            elif token in _PRESET_FALSE:
                out[action.dest] = False
            else:
                raise _PresetError(
                    f"{_PRESET_FILENAME} [{section}]: {key} = {value!r}: "
                    "expected true or false"
                )
        elif callable(typ):
            try:
                out[action.dest] = typ(value)
            except (argparse.ArgumentTypeError, ValueError) as exc:
                raise _PresetError(
                    f"{_PRESET_FILENAME} [{section}]: {key} = {value!r}: {exc}"
                ) from None
        else:
            raise _PresetError(
                f"{_PRESET_FILENAME} [{section}]: {key}: "
                "this option cannot be set from a preset"
            )
    return out


def _load_preset(
    names: Sequence[str], ap: argparse.ArgumentParser
) -> dict[str, object]:
    """Merge the named sections of ./inksep.ini into one dest-to-value mapping.

    Merge order, lowest to highest: [DEFAULT], each named section left to
    right (later wins), then explicit command-line flags in main. A section
    contributes only the keys it sets itself.
    """
    for name in names:
        if "/" in name or "\\" in name or name.endswith(".ini"):
            raise _PresetError(
                f"--preset {name!r}: expects a section name from "
                f"./{_PRESET_FILENAME}, not a path"
            )
    path = Path(_PRESET_FILENAME)
    if not path.is_file():
        raise _PresetError(
            f"{_PRESET_FILENAME} not found in the working directory; "
            "create one ([DEFAULT] + preset sections) or run without --preset"
        )
    cp = configparser.ConfigParser(
        interpolation=None, inline_comment_prefixes=("#", ";")
    )
    try:
        with path.open(encoding="utf-8") as fh:
            cp.read_file(fh)
    except (configparser.Error, OSError) as exc:
        raise _PresetError(f"{_PRESET_FILENAME}: {exc}") from None

    available = cp.sections()
    resolved: list[str] = []
    for name in names:
        if name in available:
            resolved.append(name)
            continue
        ci = [sec for sec in available if sec.casefold() == name.casefold()]
        if len(ci) == 1:
            resolved.append(ci[0])
        elif len(ci) > 1:
            raise _PresetError(
                f"{_PRESET_FILENAME}: sections {', '.join(ci)} differ only by "
                "case; use the exact spelling"
            )
        else:
            listing = ", ".join(available) if available else "(none)"
            raise _PresetError(
                f"{_PRESET_FILENAME} has no section [{name}]; available: {listing}"
            )

    defaults = dict(cp.defaults())
    values = _coerce_preset_keys(defaults, "DEFAULT", ap)
    for section in resolved:
        section_items = dict(cp.items(section, raw=True))
        explicit = {
            k: v
            for k, v in section_items.items()
            if k not in defaults or v != defaults[k]
        }
        values.update(_coerce_preset_keys(explicit, section, ap))
    return values


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="inksep",
        description="Stipple an image into a physically sized SVG "
        "(pen plotter, laser, print)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", help="Input image")
    ap.add_argument("output", help="Output SVG path")
    ap.add_argument(
        "--preset",
        action="append",
        metavar="NAME",
        default=argparse.SUPPRESS,
        help=f"Preset section name from ./{_PRESET_FILENAME} in the working "
        "directory. Repeatable: sections merge left to right over [DEFAULT]; "
        "command-line flags always win.",
    )
    ap.add_argument("--stipples", type=int, default=2000, metavar="N")
    ap.add_argument(
        "--min-dot",
        type=_mm_length,
        default=_DEFAULT_MIN_DOT_MM,
        metavar="MM",
        help="Smallest dot diameter, in millimetres (px is rejected: "
        "physical sizes must not depend on rendering resolution) "
        "(default: %(default).2f mm).",
    )
    ap.add_argument(
        "--dot-range",
        type=float,
        default=4.0,
        metavar="F",
        help="How much bigger dark dots get: largest diameter = smallest × (1 + F).",
    )
    ap.add_argument("--cutoff", type=float, default=0.0, metavar="0-1")
    ap.add_argument("--generations", type=int, default=100, metavar="N")
    ap.add_argument(
        "--tol",
        type=float,
        default=0.0,
        metavar="PX",
        help="Stop early once the dots settle: mean particle movement stays "
        "below TOL px for 3 consecutive generations, and the separation "
        "report still runs. 0 = never; --generations is then the maximum.",
    )
    ap.add_argument("--invert", action="store_true")
    ap.add_argument(
        "--damping",
        type=float,
        default=0.0,
        metavar="0-1",
        help="Inertia: each generation moves only part of the way to the new "
        "centroid. 0 = move fully (default, fastest). Raise it (e.g. "
        "0.2-0.5) to calm jitter when the separation push fights the move; "
        "convergence then needs more generations.",
    )
    ap.add_argument(
        "--step",
        type=float,
        default=1.0,
        metavar="PX",
        help="Sampling grid size, in source pixels. Larger = faster, coarser "
        "relaxation. Algorithm knob; rarely needs changing.",
    )
    ap.add_argument(
        "--dpi",
        type=float,
        default=96.0,
        metavar="DPI",
        help="Assumed resolution of the input image; it sets the physical "
        "size of the output (document width = pixels / dpi, and your mm dot "
        "sizes are measured against that). Think of it as a size dial: "
        "lower dpi = larger print, higher dpi = smaller.",
    )
    ap.add_argument(
        "--border",
        type=_mm_length,
        default=_DEFAULT_BORDER_MM,
        metavar="MM",
        help="Clear margin at the image edge, in millimetres, rounded to "
        "whole source pixels (default: %(default).2f mm).",
    )
    ap.add_argument(
        "--sep-factor",
        type=float,
        default=0.85,
        metavar="F",
        help="Minimum distance between dark dots, as a fraction of their "
        "radii added together. 0 turns separation off.",
    )
    ap.add_argument(
        "--sep-dark-threshold",
        type=float,
        default=0.50,
        metavar="0-1",
        help="Only dots darker than this are held apart; lighter dots "
        "(shading, highlights) pack freely.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the settings summary and exit without stippling or "
        "writing a file.",
    )
    ap.add_argument(
        "--no-background",
        action="store_true",
        help="Leave out the white background rectangle, for plotters that "
        "must not trace a page border.",
    )
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--preview-every", type=int, default=1, metavar="N")
    ap.add_argument(
        "--optimize-travel",
        action="store_true",
        help="Reorder the dots so the pen lifts and travels less. Geometry "
        "is unchanged; only the drawing order.",
    )
    ap.add_argument(
        "--travel-budget",
        type=float,
        default=5.0,
        metavar="SEC",
        help="Seconds allowed for that reordering, checked between passes; "
        "one pass on a very large run (15k+ dots) can overshoot it.",
    )
    ap.add_argument(
        "--stroke-width",
        type=float,
        default=None,
        metavar="MM",
        help="Pen tip diameter in millimetres, for pen plotters. Dots are "
        "drawn as outline circles sized so the finished ink still measures "
        "what you asked for. Leave unset for filled dots (print, laser).",
    )
    ap.add_argument("--seed", type=int, default=None, metavar="N")
    ap.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )
    return ap


def _package_version() -> str:
    try:
        return metadata.version("inksep")
    except metadata.PackageNotFoundError:
        return "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    ap = _build_parser()
    pre, _ = ap.parse_known_args(argv)
    preset_names: list[str] | None = getattr(pre, "preset", None)
    if preset_names:
        preset_values: dict[str, object] = {}
        try:
            preset_values = _load_preset(preset_names, ap)
        except _PresetError as exc:
            ap.error(str(exc))
        ap.set_defaults(**preset_values)
    args = ap.parse_args(argv)

    if args.stipples < 1:
        ap.error("--stipples must be >= 1")
    if args.generations < 1:
        ap.error("--generations must be >= 1")
    if args.tol < 0:
        ap.error("--tol must be >= 0")
    if args.dpi <= 0:
        ap.error("--dpi must be > 0")
    if args.travel_budget <= 0:
        ap.error("--travel-budget must be > 0")
    if args.stroke_width is not None and args.stroke_width <= 0:
        ap.error("--stroke-width must be > 0 when set")

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if args.dry_run:
        preview_mode = "disabled (dry run)"
        show_preview = False
    elif args.no_preview:
        preview_mode = "disabled"
        show_preview = False
    else:
        try:
            from .preview import _INTERACTIVE
        except ImportError:
            ap.error("preview requires matplotlib; reinstall inksep to restore it")
        else:
            preview_mode = "live window" if _INTERACTIVE else "PNG snapshots"
            show_preview = True

    # Sizes are mm on the command line; internals are source pixels.
    # Convert once, here.
    px_per_mm = args.dpi / 25.4
    min_dot_px = args.min_dot * px_per_mm
    min_dot_mm = min_dot_px / px_per_mm
    max_dot_px = min_dot_px * (1.0 + args.dot_range)
    max_dot_mm = max_dot_px / px_per_mm
    border_px = round(args.border * px_per_mm)
    sep_mm = (args.sep_factor * max_dot_px) / px_per_mm if args.sep_factor > 0 else 0.0

    try:
        gen = InkSep(
            image_path=args.input,
            n_stipples=args.stipples,
            min_dot_size=min_dot_px,
            dot_size_factor=args.dot_range,
            white_cutoff=args.cutoff,
            invert=args.invert,
            border_width=border_px,
            dpi=args.dpi,
            step=args.step,
            damping=args.damping,
            sep_factor=args.sep_factor,
            sep_dark_threshold=args.sep_dark_threshold,
            show_preview=show_preview,
            preview_every=args.preview_every,
            seed=args.seed,
            optimize_travel=args.optimize_travel,
            travel_budget=args.travel_budget,
            stroke_width_mm=args.stroke_width,
            background=not args.no_background,
        )
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # preflight block: everything known before a single generation runs
    area_cm2 = (gen.img_w / px_per_mm / 10.0) * (gen.img_h / px_per_mm / 10.0)
    density = args.stipples / area_cm2
    bar = "-" * 60
    logger.info(bar)
    logger.info("  inksep: Separation-Constrained Stippling")
    logger.info("  Preview : %s", preview_mode)
    logger.info(bar)
    logger.info("  Input       : %s", args.input)
    logger.info("  Output      : %s", args.output)
    if preset_names:
        logger.info(
            "  Preset      : %s (%s)", " + ".join(preset_names), _PRESET_FILENAME
        )
    logger.info(
        "  Document    : %.2fmm × %.2fmm  (%d×%dpx @ %gdpi)",
        gen.img_w / px_per_mm,
        gen.img_h / px_per_mm,
        gen.img_w,
        gen.img_h,
        args.dpi,
    )
    logger.info("  Stipples    : %d  (%.1f dots/cm²)", args.stipples, density)
    logger.info("  Dot range   : %.4fmm – %.4fmm", min_dot_mm, max_dot_mm)
    logger.info(
        "  Generations : %d   Damping: %g   Step: %gpx",
        args.generations,
        args.damping,
        args.step,
    )
    if args.tol > 0:
        logger.info("  Early stop  : when mean move < %.4fpx for 3 gens", args.tol)
    logger.info(
        "  Sep factor  : %g  →  min gap between dark dots = %.4fmm  (threshold=%g)",
        args.sep_factor,
        sep_mm,
        args.sep_dark_threshold,
    )
    logger.info("  Seed        : %s", args.seed if args.seed is not None else "random")
    if args.optimize_travel:
        logger.info(
            "  Travel opt  : on  (budget %.1fs, checked between passes)",
            args.travel_budget,
        )
    if args.stroke_width is not None:
        logger.info("  Dot style   : stroke %.2fmm  (plotter)", args.stroke_width)
        if args.stroke_width > min_dot_mm:
            logger.info(
                "  Pen margin  : pen %.2fmm → min dot %.2fmm: clamping expected; "
                "exact count printed after the run",
                args.stroke_width,
                min_dot_mm,
            )
        else:
            logger.info(
                "  Pen margin  : pen %.2fmm → min dot %.2fmm: OK",
                args.stroke_width,
                min_dot_mm,
            )
    logger.info(bar)

    if args.dry_run:
        logger.info("  Dry run     : preflight only, no stippling, no output file.")
        logger.info(bar)
        return 0

    t0 = time.perf_counter()
    try:
        completed = gen.run(generations=args.generations, tol=args.tol)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    total = time.perf_counter() - t0
    logger.info(
        "  Total time : %.1fs  (%.0fms/generation)",
        total,
        total / completed * 1000,
    )

    gen.save_svg(args.output)
    return 0
