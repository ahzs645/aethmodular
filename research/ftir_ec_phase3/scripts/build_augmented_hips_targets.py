"""Build locked confirmation targets from provisional reconstructed HIPS Fabs.

The reconstruction in :mod:`reconstruct_hips_fabs` recovers Fabs for filters
that were absent from the shipped SPARTAN HIPS batch but have raw T1/R1, a
known lot blank line, a positive sample volume, and an FTIR spectrum.  This
script turns those rows into *separate holdout targets* and into combined
shipped+reconstructed sensitivity targets for the calibration explorer.

Nothing in the shipped target folders is modified.  Keeping the recovered
rows in distinct folders is deliberate: configurations selected before these
values were reconstructed can be evaluated on them exactly once without
silently re-selecting a winner.

Usage::

    uv run python research/ftir_ec_phase3/scripts/build_augmented_hips_targets.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PHASE3 = HERE.parent
TARGETS = REPO / "calibration_explorer/targets"
DEFAULT_RECON = PHASE3 / "output/tables/hips/reconstructed_fabs.csv"

sys.path.insert(0, str(REPO / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(HERE))

from build_spartan_target import (  # noqa: E402
    SEASON,
    app_grid,
    corrected,
    decode_scans,
)
from phase3_common import PATHS  # noqa: E402


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = [column.strip('\ufeff"') for column in frame.columns]
    return frame


def _validate_reconstruction(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Site", "FilterId", "Fabs_reconstructed", "Volume_m3",
        "AnalysisTimestamp", "Lot",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"reconstruction is missing columns: {sorted(missing)}")
    usable = frame[
        frame["Fabs_reconstructed"].notna()
        & frame["Volume_m3"].gt(0)
        & frame["Site"].isin(["ETAD", "ETBI", "INDH"])
    ].copy()
    if usable["FilterId"].duplicated().any():
        duplicate = usable.loc[usable["FilterId"].duplicated(False), "FilterId"].tolist()
        raise ValueError(f"reconstruction is not one row per filter: {duplicate[:5]}")
    if len(usable) != 54:
        raise ValueError(f"expected 54 usable recovered filters, found {len(usable)}")
    expected = {"ETAD": 14, "ETBI": 14, "INDH": 26}
    counts = usable["Site"].value_counts().to_dict()
    if counts != expected:
        raise ValueError(f"unexpected recovered counts: {counts}, expected {expected}")
    return usable


def _shipped_reference(site: str) -> pd.DataFrame:
    columns = ["Site", "FilterId", "LotId", "Fabs", "Volume"]
    hips = pd.read_csv(PATHS.spartan_hips_primary, encoding="cp1252",
                       usecols=columns, low_memory=False)
    part = hips[hips["Site"].eq(site)].drop_duplicates("FilterId").copy()
    return part.rename(columns={"FilterId": "ExternalFilterId"})


def _combined_reference(site: str, filters: pd.DataFrame,
                        recovered: pd.DataFrame) -> pd.DataFrame:
    """One reference row per filter, shipped values taking precedence."""
    source = filters[[
        "MediaId", "ExternalFilterId", "SampleVolume_m3", "SamplingStartDate"
    ]].drop_duplicates("ExternalFilterId").copy()
    if source["ExternalFilterId"].duplicated().any():
        raise ValueError(f"{site}: filter metadata is not one row per ExternalFilterId")

    shipped = _shipped_reference(site)[
        ["ExternalFilterId", "Fabs", "LotId"]
    ]
    provisional = recovered[recovered["Site"].eq(site)][
        ["FilterId", "Fabs_reconstructed", "Lot"]
    ].rename(columns={
        "FilterId": "ExternalFilterId",
        "Fabs_reconstructed": "Fabs_recovered",
        "Lot": "Lot_recovered",
    })
    merged = source.merge(shipped, on="ExternalFilterId", how="left",
                          validate="one_to_one")
    merged = merged.merge(provisional, on="ExternalFilterId", how="left",
                          validate="one_to_one")
    overlap = merged["Fabs"].notna() & merged["Fabs_recovered"].notna()
    if overlap.any():
        ids = merged.loc[overlap, "ExternalFilterId"].tolist()
        raise ValueError(f"{site}: recovered rows overlap shipped rows: {ids[:5]}")
    merged["ReferenceSource"] = np.where(
        merged["Fabs"].notna(), "shipped",
        np.where(merged["Fabs_recovered"].notna(), "reconstructed", None),
    )
    merged["Fabs"] = merged["Fabs"].fillna(merged["Fabs_recovered"])
    merged["LotId"] = merged["LotId"].fillna(merged["Lot_recovered"])
    merged["Volume_m3"] = merged["SampleVolume_m3"]
    sample_date = pd.to_datetime(
        merged["SamplingStartDate"], format="mixed", errors="coerce"
    )
    merged["Date"] = sample_date.dt.date.astype(str).where(sample_date.notna(), "")
    month = sample_date.dt.month
    merged["Group"] = month.map(
        lambda value: SEASON.get(int(value), "unknown")
        if pd.notna(value) else "unknown"
    )
    return merged[
        merged["Fabs"].notna() & merged["Volume_m3"].gt(0)
    ].copy()


def _write_target(name: str, spectra: pd.DataFrame, reference: pd.DataFrame) -> None:
    if reference["MediaId"].duplicated().any():
        raise ValueError(f"{name}: reference MediaId is not unique")
    if not set(reference["MediaId"]).issubset(set(spectra.index)):
        missing = sorted(set(reference["MediaId"]) - set(spectra.index))
        raise ValueError(f"{name}: {len(missing)} reference rows lack spectra")
    selected = spectra.loc[reference["MediaId"].astype(int)]
    if selected.isna().any(axis=None):
        raise ValueError(f"{name}: selected spectra contain missing values")
    corr, _ = corrected(selected)
    folder = TARGETS / name
    folder.mkdir(parents=True, exist_ok=True)
    selected.reset_index().to_csv(folder / "spectra.csv", index=False)
    reference[[
        "MediaId", "ExternalFilterId", "Fabs", "Volume_m3", "Date", "Group",
        "ReferenceSource", "LotId",
    ]].to_csv(folder / "reference.csv", index=False)
    corr.reset_index().to_csv(folder / "spectra_corrected.csv", index=False)


def _staged_site(site: str, recovered: pd.DataFrame, staged_root: Path) -> dict:
    folder = staged_root / site
    filters = _read_csv(folder / f"{site}_filters.csv")
    analyses = _read_csv(folder / f"{site}_ftir_analysis.csv")
    scans = _read_csv(folder / f"{site}_scans_base64.csv")
    wcols, wn = app_grid()
    spectra = decode_scans(scans, analyses, wcols, wn)
    reference = _combined_reference(site, filters, recovered)
    holdout = reference[reference["ReferenceSource"].eq("reconstructed")].copy()
    base = site.lower()
    _write_target(f"{base}_reconstructed_holdout", spectra, holdout)
    _write_target(f"{base}_augmented", spectra, reference)
    return {
        "site": site,
        "decoded_spectra": len(spectra),
        "shipped": int(reference["ReferenceSource"].eq("shipped").sum()),
        "reconstructed": len(holdout),
        "combined": len(reference),
    }


def _addis(recovered: pd.DataFrame) -> dict:
    raw = pd.read_csv(PATHS.etad_dir / "ETAD_FTIR_spectra.csv")
    meta = pd.read_csv(PATHS.etad_dir / "ETAD_metadata.csv")
    wcols, _ = app_grid()
    available = [column for column in wcols if column in raw.columns]
    if available != wcols:
        raise ValueError(f"ETAD spectra cover {len(available)}/{len(wcols)} app columns")
    spectra = raw.groupby("MediaId")[wcols].mean()
    reference = _combined_reference("ETAD", meta, recovered)
    holdout = reference[reference["ReferenceSource"].eq("reconstructed")].copy()
    _write_target("addis_reconstructed_holdout", spectra, holdout)
    _write_target("addis_augmented", spectra, reference)
    return {
        "site": "ETAD",
        "decoded_spectra": len(spectra),
        "shipped": int(reference["ReferenceSource"].eq("shipped").sum()),
        "reconstructed": len(holdout),
        "combined": len(reference),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstructed", type=Path, default=DEFAULT_RECON)
    parser.add_argument("--staged-root", type=Path, default=(
        PATHS.ftir_dir.parent / "DAVIS/SPARTAN FTIR pulls"
    ))
    args = parser.parse_args()

    recovered = _validate_reconstruction(_read_csv(args.reconstructed))
    summaries = [_addis(recovered)]
    for site in ("ETBI", "INDH"):
        summaries.append(_staged_site(site, recovered, args.staged_root))
    summary = pd.DataFrame(summaries)
    print(summary.to_string(index=False))
    print(f"\nwrote six non-destructive targets under {TARGETS.relative_to(REPO)}")


if __name__ == "__main__":
    main()
