from pathlib import Path

import pandas as pd
import pytest

from research.ftir_hips_chem.scripts import aeronet


def write_aeronet_csv(
    path: Path,
    *,
    header_offset: int,
    date_column: str,
    value_column: str,
) -> None:
    metadata = [f"AERONET metadata line {i}" for i in range(header_offset)]
    rows = [
        f"{date_column},{value_column},Site",
        "02:01:2024,-999.0,Test Site",
        "01:01:2024,0.25,Test Site",
    ]
    path.write_text("\n".join(metadata + rows) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("header_offset", "date_column", "kind", "value_column"),
    [
        (5, "Date(dd:mm:yyyy)", "aod", aeronet.COLS["aod500"]),
        (6, "Date_(dd:mm:yyyy)", "sda", aeronet.COLS["fmf"]),
    ],
)
def test_load_aeronet_sniffs_header_dates_and_missing_values(
    tmp_path,
    header_offset,
    date_column,
    kind,
    value_column,
):
    path = tmp_path / f"{kind}.csv"
    write_aeronet_csv(
        path,
        header_offset=header_offset,
        date_column=date_column,
        value_column=value_column,
    )

    result = aeronet.load_aeronet(path, kind=kind)

    assert aeronet.find_header_line(path) == header_offset
    assert result.index.name == "Date"
    assert result.index.tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]
    assert result.loc["2024-01-01", value_column] == pytest.approx(0.25)
    assert pd.isna(result.loc["2024-01-02", value_column])


def test_load_aeronet_localizes_index(tmp_path):
    path = tmp_path / "aod.csv"
    write_aeronet_csv(
        path,
        header_offset=5,
        date_column="Date(dd:mm:yyyy)",
        value_column=aeronet.COLS["aod500"],
    )

    result = aeronet.load_aeronet(path, tz="UTC")

    assert str(result.index.tz) == "UTC"


def test_load_aeronet_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError, match="kind must be"):
        aeronet.load_aeronet(tmp_path / "unused.csv", kind="inversion")


def test_resolve_column_accepts_alias_and_real_name():
    df = pd.DataFrame({aeronet.COLS["aod500"]: [0.2]})

    assert aeronet.resolve_column(df, "aod500") == "AOD_500nm"
    assert aeronet.resolve_column(df, "AOD_500nm") == "AOD_500nm"


def test_resolve_column_error_lists_available_columns():
    df = pd.DataFrame({"AOD_440nm": [0.2]})

    with pytest.raises(KeyError, match="AOD_440nm"):
        aeronet.resolve_column(df, "aod500")


def test_merge_aeronet_outer_joins_and_prefers_aod_values():
    aod = pd.DataFrame(
        {"shared": ["aod"], "AOD_500nm": [0.2]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    sda = pd.DataFrame(
        {"shared": ["sda"], "FineModeFraction_500nm[eta]": [0.7]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    ssa = pd.DataFrame(
        {"SSA_440": [0.9]},
        index=pd.to_datetime(["2024-01-02"]),
    )

    result = aeronet.merge_aeronet(aod=aod, sda=sda, ssa=ssa)

    assert result.index.tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]
    assert result.loc["2024-01-01", "shared"] == "aod"
    assert result.loc["2024-01-01", "FineModeFraction_500nm[eta]"] == pytest.approx(0.7)
    assert result.loc["2024-01-02", "SSA_440"] == pytest.approx(0.9)


def test_merge_aeronet_with_no_inputs_is_empty():
    assert aeronet.merge_aeronet().empty


def test_aeronet_dir_prefers_environment(monkeypatch, tmp_path):
    configured = tmp_path / "configured"
    configured.mkdir()
    monkeypatch.setattr(aeronet, "AERONET_DATA_DIR", configured)
    monkeypatch.setenv("AETHMODULAR_AERONET_DIR", str(tmp_path / "override"))

    assert aeronet.aeronet_dir() == tmp_path / "override"


def test_aeronet_dir_uses_nonempty_configured_directory(monkeypatch, tmp_path):
    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "sample.csv").touch()
    monkeypatch.delenv("AETHMODULAR_AERONET_DIR", raising=False)
    monkeypatch.setattr(aeronet, "AERONET_DATA_DIR", configured)
    monkeypatch.setattr(aeronet, "maia_data_root", lambda: tmp_path / "drive")

    assert aeronet.aeronet_dir() == configured


def test_aeronet_dir_falls_back_to_drive(monkeypatch, tmp_path):
    configured = tmp_path / "configured"
    configured.mkdir()
    drive = tmp_path / "drive"
    monkeypatch.delenv("AETHMODULAR_AERONET_DIR", raising=False)
    monkeypatch.setattr(aeronet, "AERONET_DATA_DIR", configured)
    # The MAIA prefix now lives once in data_paths.maia_data_root() instead of
    # being spelled out here, so patch that seam.
    monkeypatch.setattr(aeronet, "maia_data_root", lambda: drive)

    assert aeronet.aeronet_dir() == drive / "AERONET"
