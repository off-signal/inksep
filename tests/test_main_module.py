"""``python -m inksep`` entry point."""

import runpy
import sys

import pytest


def test_run_module_version(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["inksep", "--version"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("inksep", run_name="__main__")
    assert exc.value.code == 0
