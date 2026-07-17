"""Shared loaders for the phase-3 FTIR EC notebooks.

Everything here replicates the phase-2 conventions in
``research/ftir_hips_chem`` (ftir_08–ftir_10): the same Addis evaluation
table, the same TOR loading construction, and the same locked site-held-out
test protocol. Import this module first — it also puts the phase-2 ``scripts``
directory on ``sys.path`` so ``pls_transfer``, ``config`` etc. resolve.
"""

from __future__ import annotations

import sys
from pathlib import Path

PHASE3_DIR = Path(__file__).resolve().parent.parent
PHASE2_SCRIPTS = PHASE3_DIR.parent / "ftir_hips_chem" / "scripts"
if str(PHASE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PHASE2_SCRIPTS))

import numpy as np
import pandas as pd

from pls_transfer import FTIRTransferPaths  # noqa: E402  (needs sys.path above)

PATHS = FTIRTransferPaths.defaults()
PHASE2_TABLES = PHASE3_DIR.parent / "ftir_hips_chem" / "output" / "tables"


def load_addis_evaluation(season_for_month=None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Addis spectra + HIPS + deployed-EC table, exactly as in ftir_10 cell 3.

    Returns ``(etad_eval, X_etad, wn)`` where rows are one physical filter
    (replicate FTIR scans averaged), restricted to complete spectra with a
    HIPS Fabs and a positive sample volume.
    """
    raw_etad = pd.read_csv(PATHS.etad_dir / "ETAD_FTIR_spectra.csv")
    etad_meta = pd.read_csv(PATHS.etad_dir / "ETAD_metadata.csv")
    wcols = sorted(
        [c for c in raw_etad.columns if c not in ("SampleAnalysisId", "MediaId")],
        key=lambda value: -float(value),
    )
    wn = np.array([float(c) for c in wcols])
    etad_spectra = raw_etad.groupby("MediaId", as_index=False)[wcols].mean()

    hips = pd.read_csv(PATHS.spartan_hips_primary, encoding="cp1252")
    hips_etad = (
        hips[hips["Site"].eq("ETAD")][["FilterId", "Fabs", "tau", "DepositArea", "Volume"]]
        .drop_duplicates("FilterId")
    )

    sys.path.insert(0, str(PHASE2_SCRIPTS))
    from data_matching import load_filter_data  # noqa: E402

    filter_data = load_filter_data()
    deployed = (
        filter_data[(filter_data["Site"].eq("ETAD")) & (filter_data["Parameter"].eq("EC_ftir"))]
        [["FilterId", "Concentration", "CalibrationSetId"]]
        .drop_duplicates("FilterId")
        .rename(columns={"Concentration": "EC_deployed_ugm3"})
    )

    etad = (
        etad_spectra.merge(etad_meta, on="MediaId", how="left", validate="one_to_one")
        .merge(
            hips_etad.rename(columns={"FilterId": "ExternalFilterId"}),
            on="ExternalFilterId", how="left", validate="one_to_one",
        )
        .merge(deployed, left_on="ExternalFilterId", right_on="FilterId",
               how="left", validate="one_to_one")
    )
    etad["SamplingStartDate"] = pd.to_datetime(etad["SamplingStartDate"], errors="coerce")
    if season_for_month is not None:
        etad["season"] = etad["SamplingStartDate"].dt.month.map(season_for_month)
    etad["has_complete_spectrum"] = etad[wcols].notna().all(axis=1)
    etad_eval = etad[
        etad["has_complete_spectrum"] & etad["Fabs"].notna() & etad["SampleVolume_m3"].gt(0)
    ].copy()
    X_etad = etad_eval[wcols].to_numpy(float)
    etad_eval.attrs["wcols"] = wcols
    return etad_eval, X_etad, wn


def load_tor_loadings() -> pd.DataFrame:
    """One row per (Site, date) with TOR EC and OC filter loadings (µg/filter).

    Loadings use the phase-2 construction:
    ``Value (µg/m³) × AverageFlowRate/1000 (m³/min) × ElapsedTime (min) / 1000``.
    The OC/EC ratio is concentration ratio == loading ratio (same volume).
    """
    tor = pd.read_csv(
        PATHS.ftir_dir / "local_db/tables/results_tor.csv",
        usecols=["Site", "SampleDate", "Parameter", "Value", "AverageFlowRate", "ElapsedTime"],
    )
    frames = {}
    for parameter in ("EC", "OC"):
        part = tor[tor["Parameter"].eq(parameter)].copy()
        part["date"] = pd.to_datetime(part["SampleDate"], format="mixed", errors="coerce").dt.normalize()
        part = part.drop_duplicates(["Site", "date"])
        part[f"TOR_{parameter}_loading_ug"] = (
            part["Value"] * (part["AverageFlowRate"] / 1000 * part["ElapsedTime"]) / 1000
        )
        part[f"TOR_{parameter}_ugm3"] = part["Value"]
        frames[parameter] = part[
            ["Site", "date", f"TOR_{parameter}_loading_ug", f"TOR_{parameter}_ugm3"]
        ]
    merged = frames["EC"].merge(frames["OC"], on=["Site", "date"], how="outer", validate="one_to_one")
    merged["OC_EC_ratio"] = merged["TOR_OC_ugm3"] / merged["TOR_EC_ugm3"]
    return merged


def load_pool_metadata() -> pd.DataFrame:
    """The ftir_09 full-pool similarity table (one row per lot-248/251 spectrum)."""
    similarity_path = PHASE2_TABLES / "pls_transfer" / "improve_full_pool_addis_similarity.csv"
    if not similarity_path.exists():
        raise FileNotFoundError(
            "Run ftir_09_improve_analog_selection.ipynb first: " + str(similarity_path)
        )
    similarity = pd.read_csv(similarity_path)
    similarity["date"] = pd.to_datetime(
        similarity["SampleDate"], format="mixed", errors="coerce"
    ).dt.normalize()
    return similarity


def band_center(X, wavenumbers, peak_band=(1550, 1680),
                left=(1750, 1800), right=(1500, 1530)):
    """Peak center/height of the ~1600 cm⁻¹ band above a local continuum.

    Shared by ftir_12 (raw offset-corrected spectra) and ftir_13 (AIRSpec-corrected
    spectra). An argmax on the window boundary is a leaking neighbor band (e.g. the
    carbonyl shoulder at the high edge), not a resolved peak — flagged as ``edge_hit``.
    """
    X = np.asarray(X, float)
    wavenumbers = np.asarray(wavenumbers, float)
    peak_mask = (wavenumbers >= peak_band[0]) & (wavenumbers <= peak_band[1])
    left_mask = (wavenumbers >= left[0]) & (wavenumbers <= left[1])
    right_mask = (wavenumbers >= right[0]) & (wavenumbers <= right[1])
    left_x, right_x = wavenumbers[left_mask].mean(), wavenumbers[right_mask].mean()
    left_y, right_y = X[:, left_mask].mean(axis=1), X[:, right_mask].mean(axis=1)
    slope = (right_y - left_y) / (right_x - left_x)
    continuum = left_y[:, None] + slope[:, None] * (wavenumbers[peak_mask][None, :] - left_x)
    relative = X[:, peak_mask] - continuum
    argmax = np.argmax(relative, axis=1)
    center = wavenumbers[peak_mask][argmax]
    height = relative.max(axis=1)
    edge_hit = (argmax <= 1) | (argmax >= peak_mask.sum() - 2)
    return center, height, edge_hit


def load_pool_spectra(analysis_ids, wcols) -> pd.DataFrame:
    """Fetch spectra rows for ``analysis_ids`` from the chunked 13k-pool CSV."""
    needed = set(int(v) for v in analysis_ids)
    parts = []
    pool_path = PATHS.ftir_dir / "local_db/spectra_248_251.csv"
    for chunk in pd.read_csv(pool_path, chunksize=750):
        keep = chunk["AnalysisId"].astype(int).isin(needed)
        if keep.any():
            parts.append(chunk.loc[keep, ["AnalysisId"] + list(wcols)])
    return pd.concat(parts, ignore_index=True).drop_duplicates("AnalysisId")
