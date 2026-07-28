"""Tests for notebook bootstrap and calendar plumbing helpers."""

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "ftir_hips_chem"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import nbsetup  # noqa: E402
from config import ETHIOPIA_SEASONS  # noqa: E402
from prep import add_calendar_columns, output_dirs  # noqa: E402


def test_output_dirs_creates_absolute_canonical_paths(tmp_path):
    directories = output_dirs("example_analysis", data_root=tmp_path)

    assert directories == {
        "plots": (tmp_path / "output" / "plots" / "example_analysis").resolve(),
        "tables": (tmp_path / "output" / "tables" / "example_analysis").resolve(),
    }
    assert all(path.is_absolute() and path.is_dir() for path in directories.values())


def test_output_dirs_honors_custom_subdirs(tmp_path):
    directories = output_dirs("daily", subdirs=("figures", "data"), data_root=tmp_path)

    assert set(directories) == {"figures", "data"}
    assert directories["figures"] == (tmp_path / "output" / "figures" / "daily").resolve()
    assert directories["data"].is_dir()


def test_bootstrap_returns_absolute_paths_and_creates_outputs(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_root = tmp_path / "external-data"
    monkeypatch.setattr(nbsetup, "find_repo_root", lambda: repo_root)
    monkeypatch.setattr(nbsetup, "DATA_ROOT", data_root)

    paths = nbsetup.bootstrap("calendar_check", style=False)

    assert paths.repo_root == repo_root.resolve()
    assert paths.data_root == data_root.resolve()
    # scripts_dir points at the real package directory, not a path rebuilt from
    # repo_root, so it stays correct even with find_repo_root patched out.
    assert paths.scripts_dir == Path(nbsetup.__file__).resolve().parent
    assert (paths.scripts_dir / "nbsetup.py").is_file()
    assert paths.output_root == (data_root / "output").resolve()
    assert paths.plots_dir == (data_root / "output" / "plots" / "calendar_check").resolve()
    assert paths.tables_dir == (data_root / "output" / "tables" / "calendar_check").resolve()
    assert paths.plots_dir.is_dir()
    assert paths.tables_dir.is_dir()


def test_bootstrap_applies_style_only_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(nbsetup, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(nbsetup, "DATA_ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(nbsetup, "apply_default_style", lambda: calls.append(True))

    nbsetup.bootstrap("styled", style=True)
    nbsetup.bootstrap("unstyled", style=False)

    assert calls == [True]


def test_bootstrap_propagates_repo_root_failure(monkeypatch):
    def fail():
        raise FileNotFoundError("missing root")

    monkeypatch.setattr(nbsetup, "find_repo_root", fail)

    with pytest.raises(FileNotFoundError, match="missing root"):
        nbsetup.bootstrap("never-created")


def test_add_calendar_columns_from_datetime_index_without_mutating_input():
    index = pd.to_datetime(["2024-02-29 05:00", "2024-07-01 18:30"])
    original = pd.DataFrame({"value": [1, 2]}, index=index)

    result = add_calendar_columns(original)

    assert list(result["Month"]) == [2, 7]
    assert list(result["Hour"]) == [5, 18]
    assert list(result["DayOfWeek"]) == [3, 0]
    assert list(result["DayOfYear"]) == [60, 183]
    assert list(result["season"]) == ["Dry (Oct-Feb)", "Kiremt (Jun-Sep)"]
    assert list(original.columns) == ["value"]


def test_add_calendar_columns_from_named_date_column():
    original = pd.DataFrame(
        {"sample_date": ["2023-03-15 09:30", "2023-10-02 22:00"]}
    )

    result = add_calendar_columns(original, date_col="sample_date")

    assert list(result["Month"]) == [3, 10]
    assert list(result["Hour"]) == [9, 22]
    assert list(result["season"]) == ["Belg (Mar-May)", "Dry (Oct-Feb)"]


def test_add_calendar_columns_inplace_returns_same_frame():
    frame = pd.DataFrame(index=pd.to_datetime(["2024-06-01"]))

    result = add_calendar_columns(frame, inplace=True)

    assert result is frame
    assert frame.loc[frame.index[0], "season"] == "Kiremt (Jun-Sep)"


def test_add_calendar_columns_accepts_explicit_season_mapping():
    custom_seasons = {
        "First half": {"months": [1, 2, 3, 4, 5, 6]},
        "Second half": {"months": [7, 8, 9, 10, 11, 12]},
    }
    frame = pd.DataFrame(index=pd.to_datetime(["2024-01-01", "2024-12-01"]))

    result = add_calendar_columns(frame, seasons=custom_seasons)

    assert list(result["season"]) == ["First half", "Second half"]
    assert ETHIOPIA_SEASONS["Dry (Oct-Feb)"]["months"] == [10, 11, 12, 1, 2]


def test_add_calendar_columns_requires_datetime_source():
    frame = pd.DataFrame({"value": [1, 2]})

    with pytest.raises(TypeError, match="date_col is required"):
        add_calendar_columns(frame)
