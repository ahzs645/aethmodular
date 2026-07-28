from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.ftir_hips_chem.scripts import improve_io


CHEMISTRY_COLUMNS = [
    "Dataset",
    "SiteCode",
    "POC",
    "Date",
    "AuxID",
    "ECf_Val",
    "OCf_Val",
    "fAbs_Val",
    "FlowRate_Val",
    "FEf_Val",
    "MF_Val",
    "SampDur_Val",
    "SOILf_Val",
]


def chemistry_row(
    poc,
    date,
    *,
    ec=2.0,
    oc=4.0,
    fabs=10.0,
    flow=10.0,
    duration=100.0,
):
    return {
        "Dataset": "IMPROVE",
        "SiteCode": "TEST1",
        "POC": poc,
        "Date": date,
        "AuxID": poc + 100,
        "ECf_Val": ec,
        "OCf_Val": oc,
        "fAbs_Val": fabs,
        "FlowRate_Val": flow,
        "FEf_Val": 1.0,
        "MF_Val": 0.5,
        "SampDur_Val": duration,
        "SOILf_Val": 2.0,
    }


def write_fed_sources(directory):
    excel_rows = [
        chemistry_row(1, "2020-01-01", oc=-999),
        chemistry_row(2, "2020-01-02", flow=-999),
        chemistry_row(3, "2020-01-03", ec=0),
        chemistry_row(4, "2020-01-04", fabs=-1),
        chemistry_row(5, "2020-01-05", duration=0),
    ]
    pd.DataFrame(excel_rows, columns=CHEMISTRY_COLUMNS).to_excel(
        directory / "query.xlsx",
        sheet_name="Data",
        index=False,
    )

    duplicate = chemistry_row(1, "2020-01-01", ec=99, fabs=199)
    text = "\n".join(
        [
            "FED Query Wizard export",
            "Dataset Notes",
            "Data",
            "Units and flags follow",
            "|".join(CHEMISTRY_COLUMNS),
            "|".join(str(duplicate[column]) for column in CHEMISTRY_COLUMNS),
        ]
    )
    (directory / "public_export.txt").write_text(text + "\n", encoding="utf-8")


def test_read_and_clean_improve_exports(tmp_path):
    write_fed_sources(tmp_path)

    chemistry, _, manifest = improve_io.read_improve_exports(tmp_path)

    text_manifest = manifest.loc[
        manifest["source"].str.endswith("public_export.txt")
    ].iloc[0]
    assert text_manifest["format"] == "fed_multisection"
    assert text_manifest["header_line"] == 5
    assert pd.isna(
        chemistry.loc[
            chemistry["source_file"].str.endswith("query.xlsx")
            & chemistry["POC"].eq(1),
            "OCf_Val",
        ].iloc[0]
    )

    rt = pd.DataFrame(
        [
            {
                "Dataset": "IMPROVE",
                "SiteCode": "TEST1",
                "POC": 1,
                "Date": "2020-01-01",
                "AuxID": 101,
                "RefF_635_Val": 12.0,
                "TransF_635_Val": 8.0,
                "source_file": str(tmp_path / "rt.xlsx"),
            }
        ]
    )
    metadata = {
        "Sites": pd.DataFrame(
            [{"Code": "TEST1", "Site": "Synthetic Site", "Country": "Testland"}]
        )
    }

    clean = improve_io.clean_improve(chemistry, rt=rt, metadata=metadata)
    keyed = clean.set_index("POC")

    assert set(keyed.index) == {1, 2, 5}
    assert keyed.loc[1, "ECf_Val"] == pytest.approx(2.0)
    assert keyed.loc[1, "volume_m3"] == pytest.approx(1.0)
    assert keyed.loc[1, "EC_loading_ug"] == pytest.approx(2.0)
    assert keyed.loc[1, "EC_loading_ug_cm2_area_3p5"] == pytest.approx(2.0 / 3.5)
    assert pd.isna(keyed.loc[2, "volume_m3"])
    assert pd.isna(keyed.loc[5, "volume_m3"])
    assert keyed.loc[1, "RefF_635_Val"] == pytest.approx(12.0)
    assert bool(keyed.loc[1, "rt_available"])
    assert keyed.loc[1, "SiteName"] == "Synthetic Site"

    area_columns = [
        column
        for column in clean.columns
        if column.startswith("EC_loading_ug_cm2_area_")
    ]
    assert area_columns == [
        "EC_loading_ug_cm2_area_2p2",
        "EC_loading_ug_cm2_area_3p5",
        "EC_loading_ug_cm2_area_4p0",
        "EC_loading_ug_cm2_area_3p53",
    ]


def test_load_improve_clean_forwards_usecols(tmp_path):
    cache = tmp_path / "improve_valid_cleaned.csv"
    pd.DataFrame(
        {
            "SiteCode": ["TEST1"],
            "ECf_Val": [2.0],
            "unused": [np.nan],
        }
    ).to_csv(cache, index=False)

    loaded = improve_io.load_improve_clean(
        cache,
        usecols=lambda column: column in {"SiteCode", "ECf_Val"},
    )

    assert loaded.columns.tolist() == ["SiteCode", "ECf_Val"]


def test_load_improve_clean_raises_when_cache_and_source_are_missing(
    tmp_path,
    monkeypatch,
):
    cache = tmp_path / "missing" / "improve_valid_cleaned.csv"
    source = tmp_path / "missing_improve_source"
    monkeypatch.setenv("AETHMODULAR_IMPROVE_DIR", str(source))

    with pytest.raises(FileNotFoundError) as exc_info:
        improve_io.load_improve_clean(cache)

    message = str(exc_info.value)
    assert str(cache) in message
    assert "AETHMODULAR_IMPROVE_DIR" in message


def test_improve_io_source_contains_no_account_name():
    source = Path(improve_io.__file__).read_text(encoding="utf-8")

    assert "ahzs645" not in source


def test_module_imports_on_the_flat_script_path_too():
    """Notebooks put scripts/ on sys.path and import flat; the package path is
    only used by the test suite. Both must work."""
    import importlib
    import sys

    scripts_dir = (
        Path(__file__).resolve().parents[1]
        / "research"
        / "ftir_hips_chem"
        / "scripts"
    )
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    flat = importlib.import_module("improve_io")
    assert flat.safe_area(3.5) == "3p5"
    assert callable(flat.load_improve_clean)


def test_cached_file_is_used_without_touching_the_source_directory(tmp_path, monkeypatch):
    cache = tmp_path / "improve_valid_cleaned.csv"
    pd.DataFrame({"SiteCode": ["TEST1"], "ECf_Val": [1.5]}).to_csv(cache, index=False)

    def explode():
        raise AssertionError("improve_dir() must not be consulted on a cache hit")

    monkeypatch.setattr(improve_io, "improve_dir", explode)
    loaded = improve_io.load_improve_clean(path=cache)
    assert loaded["ECf_Val"].tolist() == [1.5]


def test_rebuild_ignores_an_existing_cache_and_rewrites_it(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    write_fed_sources(source)
    cache = tmp_path / "improve_valid_cleaned.csv"
    pd.DataFrame({"SiteCode": ["STALE"], "ECf_Val": [0.0]}).to_csv(cache, index=False)

    import os

    os.environ["AETHMODULAR_IMPROVE_DIR"] = str(source)
    try:
        rebuilt = improve_io.load_improve_clean(path=cache, rebuild=True)
    finally:
        del os.environ["AETHMODULAR_IMPROVE_DIR"]

    assert "STALE" not in rebuilt["SiteCode"].tolist()
    assert rebuilt["SiteCode"].tolist() == ["TEST1"] * len(rebuilt)
    # the stale cache was actually overwritten on disk, not just in memory
    assert "STALE" not in cache.read_text()


def test_high_fabs_sweep_is_not_the_white_style_sweep():
    """These two IMPROVE sweeps differ only in their middle value (3.5 vs 3.53).
    Deriving one from the other reads as a typo and breaks silently, so they are
    separate constants and must stay that way."""
    from research.ftir_hips_chem.scripts.config import (
        IMPROVE_AREA_SENSITIVITY_CM2,
        IMPROVE_DEPOSIT_AREA_CM2,
        IMPROVE_HIGH_FABS_AREAS_CM2,
        SPARTAN_DEPOSIT_AREA_CM2,
    )

    assert tuple(IMPROVE_HIGH_FABS_AREAS_CM2) == (2.2, 3.5, 4.0)
    assert tuple(IMPROVE_AREA_SENSITIVITY_CM2) == (2.2, 3.53, 4.0)
    assert IMPROVE_HIGH_FABS_AREAS_CM2 != IMPROVE_AREA_SENSITIVITY_CM2
    # the primary area must be one the sweep actually produces a column for
    assert IMPROVE_DEPOSIT_AREA_CM2 in IMPROVE_HIGH_FABS_AREAS_CM2
    # and SPARTAN's area must not already be in the sweep, or it would be
    # appended twice and emit a duplicate column
    assert SPARTAN_DEPOSIT_AREA_CM2 not in IMPROVE_HIGH_FABS_AREAS_CM2


def test_area_columns_have_no_duplicates(tmp_path):
    write_fed_sources(tmp_path)
    chemistry, _, _ = improve_io.read_improve_exports(tmp_path)
    clean = improve_io.clean_improve(chemistry)
    area_columns = [c for c in clean.columns if c.startswith("EC_loading_ug_cm2_area_")]
    assert len(area_columns) == len(set(area_columns))
    assert clean.columns.duplicated().sum() == 0
