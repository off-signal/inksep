"""--preset / inksep.ini loading: typing, merge semantics, precedence, errors."""

import logging
from pathlib import Path

import pytest

from inksep.cli import _build_parser, _load_preset, _PresetError, main


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _ini(text: str) -> None:
    Path("inksep.ini").write_text(text, encoding="utf-8")


def _load(*names: str) -> dict[str, object]:
    return _load_preset(list(names), _build_parser())


def _dry_args(image: Path, out: Path, *extra: str) -> list[str]:
    return [str(image), str(out), "--dry-run", *extra]


# ── loader: typing, defaults, merging ────────────────────────────────────


def test_load_coerces_types_through_flag_actions(cwd):
    _ini(
        "[core]\n"
        "dpi = 25.4\n"
        "stipples = 1234\n"
        "min-dot = 0.5\n"
        "seed = 7\n"
        "invert = true\n"
    )
    vals = _load("core")
    assert vals["dpi"] == pytest.approx(25.4)
    assert vals["stipples"] == 1234 and isinstance(vals["stipples"], int)
    assert vals["min_dot"] == pytest.approx(0.5)
    assert vals["seed"] == 7
    assert vals["invert"] is True


def test_boolean_values_accept_common_spellings(cwd):
    _ini("[s]\ninvert = false\nno-preview = off\noptimize-travel = YES\n")
    vals = _load("s")
    assert vals["invert"] is False
    assert vals["no_preview"] is False
    assert vals["optimize_travel"] is True


def test_default_section_is_the_base(cwd):
    _ini("[DEFAULT]\ndpi = 100\n[a]\nstipples = 5\n")
    vals = _load("a")
    assert vals["dpi"] == pytest.approx(100.0)
    assert vals["stipples"] == 5


def test_later_section_wins_on_conflict(cwd):
    _ini("[a]\nstipples = 1\n[b]\nstipples = 2\n")
    assert _load("a", "b")["stipples"] == 2


def test_later_section_does_not_reinject_default_over_earlier_override(cwd):
    """The merge care case: [b] never mentions stipples, so [a]'s override
    must survive, not silently revert to the [DEFAULT] value."""
    _ini("[DEFAULT]\nstipples = 4000\ndpi = 100\n[a]\nstipples = 5555\n[b]\ngenerations = 3\n")
    vals = _load("a", "b")
    assert vals["stipples"] == 5555
    assert vals["generations"] == 3
    assert vals["dpi"] == pytest.approx(100.0)


def test_underscore_and_uppercase_keys_normalize(cwd):
    _ini("[s]\nMIN_DOT = 0.7\ndot_range = 1.5\n")
    vals = _load("s")
    assert vals["min_dot"] == pytest.approx(0.7)
    assert vals["dot_range"] == pytest.approx(1.5)


def test_comments_full_line_and_inline(cwd):
    _ini("# full line comment\n; another\n[a] # trailing on header\ndpi = 100 # after value\n")
    vals = _load("a")
    assert vals["dpi"] == pytest.approx(100.0)


# ── loader: errors ───────────────────────────────────────────────────────


def test_unknown_key_suggests_close_match(cwd):
    _ini("[s]\nstiple = 5\n")
    with pytest.raises(_PresetError, match="unknown option 'stiple'") as exc:
        _load("s")
    assert "stipples" in str(exc.value)


def test_unknown_key_without_close_match_has_no_hint(cwd):
    _ini("[s]\nzzzzq = 5\n")
    with pytest.raises(_PresetError) as exc:
        _load("s")
    assert "unknown option 'zzzzq'" in str(exc.value)
    assert "did you mean" not in str(exc.value)


@pytest.mark.parametrize("key", ["preset", "version", "dry-run", "help"])
def test_forbidden_keys_rejected(cwd, key):
    _ini(f"[s]\n{key} = true\n")
    with pytest.raises(_PresetError, match="not allowed in a preset"):
        _load("s")


def test_px_value_in_preset_is_rejected_with_section_context(cwd):
    _ini("[plot]\nmin-dot = 0.5px\n")
    with pytest.raises(_PresetError) as exc:
        _load("plot")
    msg = str(exc.value)
    assert "[plot]" in msg
    assert "px is not supported" in msg


def test_bad_number_reports_section_and_key(cwd):
    _ini("[core]\ndpi = fast\n")
    with pytest.raises(_PresetError) as exc:
        _load("core")
    msg = str(exc.value)
    assert "[core]" in msg and "dpi" in msg and "fast" in msg


def test_bad_boolean_reports_expected_true_or_false(cwd):
    _ini("[s]\ninvert = maybe\n")
    with pytest.raises(_PresetError, match="expected true or false"):
        _load("s")


def test_empty_value_is_a_clean_error(cwd):
    _ini("[s]\nstipples =\n")
    with pytest.raises(_PresetError) as exc:
        _load("s")
    assert "stipples" in str(exc.value)


def test_missing_section_lists_available(cwd):
    _ini("[low]\ndot-range = 0.3\n[high]\ndot-range = 0.86\n")
    with pytest.raises(_PresetError) as exc:
        _load("mid")
    msg = str(exc.value)
    assert "no section [mid]" in msg
    assert "low" in msg and "high" in msg


def test_file_without_sections_lists_none(cwd):
    _ini("# nothing here\n")
    with pytest.raises(_PresetError, match="\\(none\\)"):
        _load("x")


def test_case_insensitive_section_lookup(cwd):
    _ini("[PLOTTER]\nstipples = 9\n")
    assert _load("plotter")["stipples"] == 9


def test_sections_differing_only_by_case_are_ambiguous(cwd):
    _ini("[Mid]\nstipples = 1\n[mid]\nstipples = 2\n")
    with pytest.raises(_PresetError, match="differ only by case"):
        _load("MID")


@pytest.mark.parametrize("name", ["../other.ini", "sub/dir", "plan.ini"])
def test_path_like_preset_name_rejected(cwd, name):
    _ini("[a]\n")
    with pytest.raises(_PresetError, match="not a path"):
        _load(name)


def test_missing_file_points_at_template(cwd):
    with pytest.raises(_PresetError) as exc:
        _load("a")
    msg = str(exc.value)
    assert "inksep.ini not found" in msg
    assert "[DEFAULT] + preset sections" in msg


def test_malformed_ini_is_wrapped(cwd):
    _ini("no section header here\nfoo = 1\n")
    with pytest.raises(_PresetError) as exc:
        _load("a")
    assert str(exc.value).startswith("inksep.ini:")


def test_read_oserror_is_wrapped(cwd, monkeypatch):
    ap = _build_parser()  # before patching: building the parser reads metadata
    _ini("[a]\n")

    def _boom(self, *args, **kwargs):
        raise OSError("disk on fire")

    monkeypatch.setattr(Path, "open", _boom)
    with pytest.raises(_PresetError, match="disk on fire"):
        _load_preset(["a"], ap)


def test_option_with_unparseable_type_is_clean_error(cwd):
    ap = _build_parser()
    ap._option_string_actions["--seed"].type = "int"  # never happens in prod
    _ini("[s]\nseed = 5\n")
    with pytest.raises(_PresetError, match="cannot be set from a preset"):
        _load_preset(["s"], ap)


# ── end to end through main() ────────────────────────────────────────────


def test_composed_presets_appear_in_preflight(cwd, gradient_png, caplog):
    _ini(
        "[DEFAULT]\n"
        "dpi = 300\n"
        "min-dot = 0.3\n"
        "no-preview = true\n"
        "[sparse]\n"
        "stipples = 8000\n"
        "[low]\n"
        "dot-range = 0.30\n"
    )
    out = Path("out.svg")
    with caplog.at_level(logging.INFO, logger="inksep"):
        rc = main(_dry_args(gradient_png, out, "--preset", "sparse", "--preset", "low"))
    assert rc == 0
    text = caplog.text
    assert "Preset      : sparse + low (inksep.ini)" in text
    assert "Stipples    : 8000  (" in text
    assert "Dot range   : 0.3000mm – 0.3900mm" in text
    assert "@ 300dpi)" in text
    assert "Preview : disabled" in text


def test_cli_flag_overrides_preset_value(cwd, gradient_png, caplog):
    _ini("[one]\nstipples = 111\n")
    out = Path("out.svg")
    with caplog.at_level(logging.INFO, logger="inksep"):
        rc = main(_dry_args(gradient_png, out, "--preset", "one", "--stipples", "222"))
    assert rc == 0
    assert "Stipples    : 222  (" in caplog.text


def test_inksep_ini_ignored_without_preset_flag(cwd, gradient_png, caplog):
    _ini("[one]\nstipples = 7777\n")
    out = Path("out.svg")
    with caplog.at_level(logging.INFO, logger="inksep"):
        rc = main(_dry_args(gradient_png, out))
    assert rc == 0
    assert "Preset      :" not in caplog.text
    assert "Stipples    : 2000  (" in caplog.text


def test_missing_file_via_main_is_exit_2(cwd, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["nope.png", "out.svg", "--preset", "x"])
    assert exc.value.code == 2
    assert "inksep.ini not found" in capsys.readouterr().err


def test_missing_section_via_main_lists_sections(cwd, capsys):
    _ini("[low]\ndot-range = 0.3\n")
    with pytest.raises(SystemExit) as exc:
        main(["nope.png", "out.svg", "--preset", "nope"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "no section [nope]" in err
    assert "available: low" in err


def test_help_documents_preset_flag():
    flat = " ".join(_build_parser().format_help().split())
    assert "--preset NAME" in flat
    assert "inksep.ini" in flat
    assert "command-line flags always win" in flat


def test_real_run_with_preset_writes_svg(cwd, gradient_png, caplog):
    _ini("[tiny]\nstipples = 12\ngenerations = 2\nseed = 1\nno-preview = true\n")
    out = Path("out.svg")
    with caplog.at_level(logging.INFO, logger="inksep"):
        rc = main([str(gradient_png), str(out), "--preset", "tiny"])
    assert rc == 0
    assert out.exists()
    assert "Preset      : tiny (inksep.ini)" in caplog.text
