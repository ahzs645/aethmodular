"""Tests for the dependency-light repository command interface."""

from __future__ import annotations

import json

from aethmodular_cli import cli


def test_parser_routes_doctor():
    args = cli.build_parser().parse_args(["doctor"])
    assert args.func is cli.command_doctor
    assert args.json is False


def test_parser_accepts_repeated_resample_sites():
    args = cli.build_parser().parse_args(
        ["data", "resample", "--site", "ETAD", "--site", "Delhi"]
    )
    assert args.func is cli.command_data_resample
    assert args.site == ["ETAD", "Delhi"]
    assert args.list_sites is False


def test_notebook_check_reads_sources_but_not_outputs(tmp_path):
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["from pathlib import Path\n", "root = Path('.')"],
                "outputs": [{"output_type": "stream", "text": "/Users/example/ignored"}],
            }
        ]
    }
    path = tmp_path / "portable.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    assert cli._notebook_issues(path) == []


def test_notebook_check_flags_machine_and_legacy_paths(tmp_path):
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": "root = '/Users/example/aethmodular-clean/FTIR_HIPS_Chem'",
            }
        ]
    }
    path = tmp_path / "nonportable.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    issues = cli._notebook_issues(path)
    assert "machine-specific path (/Users/)" in issues
    assert "legacy path (FTIR_HIPS_Chem)" in issues
    assert "legacy path (aethmodular-clean)" in issues


def test_notebook_changed_flag_routes_to_incremental_check():
    args = cli.build_parser().parse_args(["notebook", "check", "--changed"])
    assert args.changed is True
    assert args.notebooks == []


def test_build_registry_discovers_numbered_targets():
    targets = cli._build_targets("ftir-calibration")
    assert "01_adama_spectra" in targets
    assert "15_downstream_recommended_calibration" in targets


def test_resample_command_forwards_site_selection(monkeypatch):
    called = {}

    def fake_run_script(relative, script_args):
        called["relative"] = relative
        called["args"] = script_args
        return 0

    monkeypatch.setattr(cli, "_run_script", fake_run_script)
    args = cli.build_parser().parse_args(
        ["data", "resample", "--site", "ETAD", "--site", "Delhi"]
    )
    assert args.func(args) == 0
    assert called == {
        "relative": "scripts/pipelines/create_9am_resampled_datasets.py",
        "args": ["--site", "ETAD", "--site", "Delhi"],
    }
