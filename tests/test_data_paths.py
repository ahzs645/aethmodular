"""Tests for external dataset path resolution.

These pin the contract the notebook migration depends on: every helper is
overridable by environment variable, none hardcodes an account name, and the
resolved locations are the same ones the notebooks previously spelled out by
hand.
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "ftir_hips_chem"
    / "scripts"
)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import data_paths  # noqa: E402

ENV_VARS = [
    ("AETHMODULAR_MAIA_DATA_ROOT", "maia_data_root"),
    ("AETHMODULAR_AETHALOMETRY_DIR", "aethalometry_dir"),
    ("AETHMODULAR_WEATHER_DIR", "weather_dir"),
    ("AETHMODULAR_FTIR_LOCAL_DB", "ftir_local_db"),
]


@pytest.mark.parametrize("env_var,func_name", ENV_VARS)
def test_environment_variable_wins(env_var, func_name, tmp_path, monkeypatch):
    monkeypatch.setenv(env_var, str(tmp_path))
    assert getattr(data_paths, func_name)() == tmp_path


@pytest.mark.parametrize("env_var,func_name", ENV_VARS)
def test_environment_variable_expands_user(env_var, func_name, monkeypatch):
    monkeypatch.setenv(env_var, "~/some-data-dir")
    assert getattr(data_paths, func_name)() == Path.home() / "some-data-dir"


def test_dataset_dirs_hang_off_the_maia_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHMODULAR_MAIA_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("AETHMODULAR_AETHALOMETRY_DIR", raising=False)
    assert data_paths.aethalometry_dir() == tmp_path / "Aethelometry Data"


def test_aeronet_and_improve_share_the_one_maia_root(tmp_path, monkeypatch):
    """The MAIA prefix used to be spelled out separately in each dataset module.
    Overriding the single root must move all of them together."""
    import aeronet
    import improve_io

    monkeypatch.setenv("AETHMODULAR_MAIA_DATA_ROOT", str(tmp_path))
    for var in ("AETHMODULAR_AERONET_DIR", "AETHMODULAR_IMPROVE_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(aeronet, "AERONET_DATA_DIR", tmp_path / "nonexistent-aeronet")
    monkeypatch.setattr(improve_io, "IMPROVE_DATA_DIR", tmp_path / "nonexistent-improve")

    assert aeronet.aeronet_dir() == tmp_path / "AERONET"
    assert improve_io.improve_dir() == tmp_path / "Improve"


def test_weather_prefers_the_in_repo_copy_over_drive(tmp_path, monkeypatch):
    local = tmp_path / "Weather Data"
    (local / "Meteostat").mkdir(parents=True)
    (local / "Meteostat" / "x.csv").write_text("a,b\n1,2\n")
    monkeypatch.delenv("AETHMODULAR_WEATHER_DIR", raising=False)
    monkeypatch.setattr(data_paths, "WEATHER_DATA_DIR", local)
    assert data_paths.weather_dir() == local


def test_weather_falls_back_to_drive_when_the_repo_copy_is_empty(tmp_path, monkeypatch):
    empty = tmp_path / "empty-weather"
    empty.mkdir()
    drive = tmp_path / "drive-data"
    monkeypatch.delenv("AETHMODULAR_WEATHER_DIR", raising=False)
    monkeypatch.setenv("AETHMODULAR_MAIA_DATA_ROOT", str(drive))
    monkeypatch.setattr(data_paths, "WEATHER_DATA_DIR", empty)
    assert data_paths.weather_dir() == drive / "Weather Data"


def test_source_contains_no_account_name_or_home_directory():
    source = (SCRIPTS_DIR / "data_paths.py").read_text()
    assert "ahzs645" not in source
    assert "/Users/" not in source


def test_drive_subdirectory_spelling_is_preserved():
    """The directory on Drive is spelled 'Aethelometry', not 'Aethalometry'.
    Correcting the spelling here would silently resolve to nothing."""
    assert data_paths.AETHALOMETRY_SUBDIR == "Aethelometry Data"


def test_describe_reports_every_entry_point_with_existence():
    described = data_paths.describe()
    assert {
        "repo_data_root",
        "drive_root",
        "maia_data_root",
        "aethalometry_dir",
        "weather_dir",
        "ftir_local_db",
    } <= set(described)
    for path, exists in described.values():
        assert isinstance(path, Path)
        assert isinstance(exists, bool)
        assert path.is_absolute()


def test_ftir_local_db_is_reported_not_raised_when_absent(tmp_path, monkeypatch):
    """It is legitimately absent on this machine; callers guard on is_dir()."""
    monkeypatch.setenv("AETHMODULAR_FTIR_LOCAL_DB", str(tmp_path / "nope"))
    assert data_paths.ftir_local_db().is_dir() is False


def test_ftir_candidates_prefer_the_current_davis_data_layout(tmp_path, monkeypatch):
    """The FTIR tree moved under 'Grad/Data/Davis Data' (seen 2026-08-10). When
    more than one layout is present the current one must win, or a stale copy
    left behind by an earlier move silently becomes the source of truth."""
    drive = tmp_path / "My Drive"
    current = drive / "University/Research/Grad/Data/Davis Data/FTIR/local_db"
    previous = drive / "University/Research/Grad/Data/FTIR/local_db"
    current.mkdir(parents=True)
    previous.mkdir(parents=True)
    monkeypatch.delenv("AETHMODULAR_FTIR_LOCAL_DB", raising=False)
    monkeypatch.setattr(data_paths, "drive_root", lambda: drive)
    assert data_paths.ftir_local_db() == current


def test_maia_root_prefers_the_current_davis_data_layout(tmp_path, monkeypatch):
    """Same move, for the raw-data root the other datasets hang off."""
    drive = tmp_path / "My Drive"
    current = drive / "University/Research/Grad/Data/Davis Data"
    previous = drive / "University/Research/Grad/UC Davis Ann/NASA MAIA/Data"
    current.mkdir(parents=True)
    previous.mkdir(parents=True)
    monkeypatch.delenv("AETHMODULAR_MAIA_DATA_ROOT", raising=False)
    monkeypatch.setattr(data_paths, "drive_root", lambda: drive)
    assert data_paths.maia_data_root() == current


def test_maia_root_still_finds_the_previous_layout(tmp_path, monkeypatch):
    """A machine that has not been reorganised must keep working."""
    drive = tmp_path / "My Drive"
    previous = drive / "University/Research/Grad/UC Davis Ann/NASA MAIA/Data"
    previous.mkdir(parents=True)
    monkeypatch.delenv("AETHMODULAR_MAIA_DATA_ROOT", raising=False)
    monkeypatch.setattr(data_paths, "drive_root", lambda: drive)
    assert data_paths.maia_data_root() == previous


def test_etad_dir_hangs_off_the_maia_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHMODULAR_MAIA_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("AETHMODULAR_ETAD_DIR", raising=False)
    assert data_paths.etad_dir() == tmp_path / "DAVIS/ETAD FTIR"


def test_ftir_candidates_prefer_the_layout_that_exists(tmp_path, monkeypatch):
    """Notebooks hardcoded 'My Drive/FTIR/local_db', which does not exist here,
    and degraded to a 'BLOCKED' message while the tables sat under
    'University/Research/Grad/Data/FTIR'. The resolver must find the real one."""
    drive = tmp_path / "My Drive"
    real = drive / "University/Research/Grad/Data/FTIR/local_db"
    real.mkdir(parents=True)
    monkeypatch.delenv("AETHMODULAR_FTIR_LOCAL_DB", raising=False)
    monkeypatch.setattr(data_paths, "drive_root", lambda: drive)
    assert data_paths.ftir_local_db() == real


def test_ftir_candidates_fall_back_to_the_short_layout(tmp_path, monkeypatch):
    drive = tmp_path / "My Drive"
    short = drive / "FTIR" / "local_db"
    short.mkdir(parents=True)
    monkeypatch.delenv("AETHMODULAR_FTIR_LOCAL_DB", raising=False)
    monkeypatch.setattr(data_paths, "drive_root", lambda: drive)
    assert data_paths.ftir_local_db() == short


def test_ftir_returns_a_concrete_path_when_nothing_exists(tmp_path, monkeypatch):
    """Callers guard with is_dir() and degrade; a concrete path makes the
    failure message useful, so this must not raise. The path it names is the
    *current* layout, so the message points at where the data should be."""
    drive = tmp_path / "My Drive"
    monkeypatch.delenv("AETHMODULAR_FTIR_LOCAL_DB", raising=False)
    monkeypatch.setattr(data_paths, "drive_root", lambda: drive)
    resolved = data_paths.ftir_local_db()
    assert resolved.is_dir() is False
    assert resolved == drive / "University/Research/Grad/Data/Davis Data/FTIR/local_db"


def test_ftir_spectra_dir_uses_the_same_candidate_order(tmp_path, monkeypatch):
    drive = tmp_path / "My Drive"
    (drive / "University/Research/Grad/Data/FTIR").mkdir(parents=True)
    (drive / "FTIR").mkdir(parents=True)
    monkeypatch.delenv("AETHMODULAR_FTIR_SPECTRA_DIR", raising=False)
    monkeypatch.setattr(data_paths, "drive_root", lambda: drive)
    assert data_paths.ftir_spectra_dir() == drive / "University/Research/Grad/Data/FTIR"


def test_maia_root_and_drive_root_are_distinct_env_vars():
    """AETHMODULAR_MAIA_DATA_ROOT means the '.../NASA MAIA/Data' directory, NOT
    the Drive root. One notebook used it for the Drive root, which would have
    made the two meanings collide for anyone who set it."""
    source = (SCRIPTS_DIR / "data_paths.py").read_text()
    assert "AETHMODULAR_MAIA_DATA_ROOT" in source
    assert "AETHMODULAR_DRIVE_ROOT" not in source  # that one belongs to drive_root()


def test_weather_file_searches_both_copies(tmp_path, monkeypatch):
    """The repo copy and the Drive copy hold DIFFERENT files, so preferring one
    directory wholesale silently misses files that live only in the other. This
    is the bug weather_file() exists to prevent."""
    repo = tmp_path / "repo-weather" / "Meteostat"
    drive = tmp_path / "drive-data" / "Weather Data" / "Meteostat"
    repo.mkdir(parents=True)
    drive.mkdir(parents=True)
    (repo / "master_meteostat.csv").write_text("a\n1\n")
    (drive / "addis_ababa_weather_data_cleaned.csv").write_text("a\n1\n")

    monkeypatch.delenv("AETHMODULAR_WEATHER_DIR", raising=False)
    monkeypatch.setattr(data_paths, "WEATHER_DATA_DIR", repo.parent)
    monkeypatch.setenv("AETHMODULAR_MAIA_DATA_ROOT", str(tmp_path / "drive-data"))

    assert data_paths.weather_file("master_meteostat.csv") == repo / "master_meteostat.csv"
    assert (
        data_paths.weather_file("addis_ababa_weather_data_cleaned.csv")
        == drive / "addis_ababa_weather_data_cleaned.csv"
    )


def test_weather_file_accepts_multiple_candidate_names(tmp_path, monkeypatch):
    repo = tmp_path / "repo-weather" / "Meteostat"
    repo.mkdir(parents=True)
    (repo / "second_choice.csv").write_text("a\n1\n")
    monkeypatch.delenv("AETHMODULAR_WEATHER_DIR", raising=False)
    monkeypatch.setattr(data_paths, "WEATHER_DATA_DIR", repo.parent)
    monkeypatch.setenv("AETHMODULAR_MAIA_DATA_ROOT", str(tmp_path / "nothing"))
    found = data_paths.weather_file("first_choice.csv", "second_choice.csv")
    assert found == repo / "second_choice.csv"


def test_weather_file_raises_listing_every_location_tried(tmp_path, monkeypatch):
    monkeypatch.delenv("AETHMODULAR_WEATHER_DIR", raising=False)
    monkeypatch.setattr(data_paths, "WEATHER_DATA_DIR", tmp_path / "a")
    monkeypatch.setenv("AETHMODULAR_MAIA_DATA_ROOT", str(tmp_path / "b"))
    with pytest.raises(FileNotFoundError) as excinfo:
        data_paths.weather_file("absent.csv")
    message = str(excinfo.value)
    assert str(tmp_path / "a") in message
    assert str(tmp_path / "b") in message
    assert "AETHMODULAR_WEATHER_DIR" in message


def test_weather_file_requires_a_name():
    with pytest.raises(ValueError):
        data_paths.weather_file()
