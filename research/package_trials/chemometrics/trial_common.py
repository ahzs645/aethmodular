"""Shared data loading for the chemometrics package trials.

Loads the lowest-OC/EC 800 cohort (spectra X, TOR EC loadings y, IMPROVE site
labels) exactly the way ``run_ftir_21.py`` constructs it, and caches the result
as an ``.npz`` so the 760 MB pool CSV is only chunk-scanned once.

Everything here is read-only with respect to the repo and the Drive tree.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
PHASE3_SCRIPTS = REPO / "research" / "ftir_ec_phase3" / "scripts"
if str(PHASE3_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PHASE3_SCRIPTS))

# phase3_common also puts the phase-2 scripts dir (pls_transfer etc.) on sys.path.
from phase3_common import PATHS, load_pool_metadata, load_pool_spectra, load_tor_loadings  # noqa: E402

COHORT_CSV = (REPO / "research" / "ftir_ec_phase3" / "output" / "tables"
              / "ftir11" / "lowest_ocec_800_cohort.csv")
CACHE = HERE / "cache" / "lowest_ocec_800.npz"


def load_cohort_800(verbose: bool = True):
    """Return ``(X, y, sites, analysis_ids, wavenumbers)`` for the 800 cohort.

    X : (n, p) float64 raw spectra, columns in the pool CSV's native
        (descending-wavenumber) order.
    y : (n,) TOR EC loadings in µg/filter, phase-2 construction
        (Value × AverageFlowRate/1000 × ElapsedTime / 1000).
    sites : (n,) IMPROVE site codes (str).
    """
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=False) as z:
            return (z["X"], z["y"], z["sites"].astype(str), z["analysis_ids"],
                    z["wavenumbers"])

    cohort = pd.read_csv(COHORT_CSV)
    ids = cohort["AnalysisId"].astype(int).to_numpy()

    # Wavenumber columns straight from the pool CSV header.
    header = pd.read_csv(PATHS.ftir_dir / "local_db/spectra_248_251.csv", nrows=0)
    wcols = [c for c in header.columns
             if c not in ("AnalysisId", "FilterId", "SampleDate", "Site")]
    wavenumbers = np.array([float(c) for c in wcols])

    t0 = time.perf_counter()
    spectra = (load_pool_spectra(ids, wcols)
               .set_index("AnalysisId").loc[ids])
    if verbose:
        print(f"chunk-scanned pool CSV for {len(ids)} rows "
              f"in {time.perf_counter() - t0:.1f} s")

    # y + Site via the canonical ftir_21 construction (metadata ⋈ TOR on Site+date),
    # then cross-checked against the columns the cohort table itself carries.
    pool = (load_pool_metadata()
            .merge(load_tor_loadings(), on=["Site", "date"], how="left",
                   validate="many_to_one")
            .query("TOR_EC_loading_ug > 0")
            .drop_duplicates("FilterId"))
    pool["AnalysisId"] = pool["AnalysisId"].astype(int)
    pool = (pool.drop_duplicates("AnalysisId").set_index("AnalysisId")
            [["Site", "TOR_EC_loading_ug"]].loc[ids])

    assert (pool["Site"].to_numpy() == cohort["Site"].to_numpy()).all()
    assert np.allclose(pool["TOR_EC_loading_ug"].to_numpy(),
                       cohort["TOR_EC_loading_ug"].to_numpy())

    X = spectra[wcols].to_numpy(float)
    y = pool["TOR_EC_loading_ug"].to_numpy(float)
    sites = pool["Site"].to_numpy(str)
    assert X.shape[0] == len(y) == len(ids)
    assert np.isfinite(X).all() and np.isfinite(y).all()

    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y, sites=sites, analysis_ids=ids,
                        wavenumbers=wavenumbers)
    return X, y, sites, ids, wavenumbers


if __name__ == "__main__":
    X, y, sites, ids, wn = load_cohort_800()
    print(f"X {X.shape}, y mean {y.mean():.3f} µg/filter, "
          f"{pd.Series(sites).nunique()} sites, "
          f"wn {wn.max():.0f}..{wn.min():.0f} cm-1")
