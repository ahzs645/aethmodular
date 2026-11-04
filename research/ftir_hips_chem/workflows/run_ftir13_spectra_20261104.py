"""ftir_13's two calibrations, refitted on raw and VIBES spectra for the Protocols crossplot.

ftir_13 [17] draws the Lowest-OC/EC (800) and Biomass-smoke (906) calibrations on AIRSpec spectra only. The same
setup applies to any spectra: the same cohorts, TOR target, seed and component rule; OC/EC fitted on the training
side of its site-disjoint split, smoke fitted on every filter, as ftir_13 does. Only the spectra change:

    raw         as measured, 4000-500 cm-1 (the full500 bundle)
    VIBES       the gallery's completed rank-70 run, 4000-500 cm-1
    VIBES cut   the same spectra with the channels below 1425 cm-1 dropped, AIRSpec's window (sensitivity only)

The VIBES library holds 762 of the 800 OC/EC filters and 887 of the 906 smoke filters, so raw and AIRSpec are also
fitted on those same filters: that measures what the missing ones cost. VIBES at Addis covers the 233 PM2.5 filters
of the frozen run; the six PM10 filters come from the 19 Oct network run, same background model, and the two runs
are compared on the 233.

The refit is gated: AIRSpec must reproduce the tracked ftir_13 predictions and raw OC/EC must reproduce ftir_11's.
k_sweep.csv holds each calibration at fixed k, because the component rule, not the spectra or the filters, can be
what moves a slope: on raw smoke, losing 19 filters takes the rule from k = 7 to 3, but at any one k they change little.

Run from the repo root:
    uv run --no-sync python research/ftir_hips_chem/workflows/run_ftir13_spectra_20261104.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupShuffleSplit

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / "scripts"))
import vibes_current  # noqa: E402
from pls_transfer import component_cv_curve, regression_metrics, select_first_major_minimum  # noqa: E402

TABLES = AREA / "output/tables"
PHASE3 = ROOT / "research/ftir_ec_phase3/output"
BUNDLE = TABLES / "vibes_colab_bundle_full500/stage/data"
NETWORK = TABLES / "network_baselines_20261019"
TRACKED = ROOT / "docs/openresearch-execution/airspec/expected/addis_predictions_corrected.csv"
OUT = TABLES / "ftir13_spectra_20261104"
SPLIT_SEED = 20260717  # ftir_13's
COHORTS = {"ocec": "lowest-OCEC 800", "smoke": "Smoke 906"}
MAC = 10.0
FIXED_K = (3, 5, 7, 9, 12, 15, 20)


def fit(X, y, sites, test_sites):
    """ftir_13's locked fit; ``test_sites`` is None for the smoke cohort, which it fits on every filter."""
    train = np.ones(len(y), bool) if test_sites is None else ~np.isin(sites, test_sites)
    curve = component_cv_curve(X[train], y[train], range(1, 41), groups=sites[train], n_splits=5, random_state=42)
    k, _ = select_first_major_minimum(curve)
    model = PLSRegression(n_components=k, scale=False).fit(X[train], y[train])
    held = {}
    if test_sites is not None:
        held = regression_metrics(y[~train], model.predict(X[~train]).ravel())
    return model, int(k), int(train.sum()), held


def load_cohorts():
    ocec = pd.read_csv(PHASE3 / "tables/ftir11/lowest_ocec_800_cohort.csv")
    smoke = pd.read_csv(TABLES / "pls_calibration_phase2/smoke_cohort_spectral_selection.csv")
    sites = ocec.Site.to_numpy()
    _, test = next(GroupShuffleSplit(n_splits=1, test_size=.20, random_state=SPLIT_SEED).split(ocec, groups=sites))
    return {"ocec": (ocec, np.unique(sites[test])), "smoke": (smoke, None)}


def spectra(addis):
    """Each spectra type as {name: (library row by id, library X, Addis X, id column)}."""
    pool = pd.read_csv(BUNDLE / "pool_metadata.csv", usecols=["AnalysisId"])
    raw = np.load(BUNDLE / "pool_raw.npy", mmap_mode="r")
    etad = pd.read_csv(BUNDLE / "etad_metadata.csv", usecols=["MediaId"])
    etad_raw = np.load(BUNDLE / "etad_raw.npy", mmap_mode="r")
    etad_row = pd.Series(etad.index, index=etad.MediaId)
    out = {"raw": (dict(zip(pool.AnalysisId, range(len(pool)))), raw, np.asarray(etad_raw[etad_row[addis.MediaId]], float), "AnalysisId")}

    air = np.load(PHASE3 / "corrected/improve_pool_corrected_df6.npz", allow_pickle=True)
    air_etad = np.load(PHASE3 / "corrected/etad_corrected_df6.npz", allow_pickle=True)
    media = pd.DataFrame(air_etad["corrected"].astype(float)).assign(MediaId=air_etad["media_id"].astype(int)).groupby("MediaId").mean()
    out["AIRSpec df1=6"] = (dict(zip(air["analysis_id"].astype(int), range(len(air["analysis_id"])))), air["corrected"],
                            media.loc[addis.MediaId].to_numpy(float), "AnalysisId")

    data = vibes_current.load_completed()
    cases = data["cases"]
    lib = cases.loc[cases.kind.eq("calibration")]
    rows = dict(zip(pd.to_numeric(lib.filter_id).astype(int), lib.index))
    target = cases.loc[cases.kind.eq("target")]
    target_row = pd.Series(target.index, index=target.sample_id.str.split(":").str[1].astype(int))
    full, wn = data["arrays"]["VIBES-full-r70"], data["grids"]["VIBES-full-r70"]

    # the frozen 233 from the gallery's run; every Addis filter again from the 19 Oct network run, to compare and to fill
    net_cases = pd.read_csv(NETWORK / "vibes_bundle/cases.csv")
    net_x = np.load(NETWORK / "vibes_bundle/X_model_grid.npy", mmap_mode="r")
    net_base = np.load(NETWORK / "vibes_baselines_model_grid.npy", mmap_mode="r")
    net = net_cases.loc[net_cases.media_id.isin(addis.MediaId)]
    corrected = pd.DataFrame(np.asarray(net_x[net.index] - net_base[net.index], float), index=net.media_id.to_numpy()).groupby(level=0).mean()
    from_network = corrected.loc[addis.MediaId].to_numpy(float)
    inside = addis.MediaId.isin(target_row.index).to_numpy()
    filled = from_network.copy()
    filled[inside] = np.asarray(full[target_row[addis.MediaId[inside]].to_numpy()], float)
    cut = wn >= 1425
    out["VIBES"] = (rows, full, filled, "FilterId")
    out["VIBES cut"] = (rows, full[:, cut], filled[:, cut], "FilterId")
    return out, from_network, inside


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tracked = pd.read_csv(TRACKED)
    addis = tracked[["MediaId", "ExternalFilterId", "Fabs"]].copy()
    meta = pd.read_csv(BUNDLE / "etad_metadata.csv").set_index("MediaId")
    volume = meta.SampleVolume_m3.loc[addis.MediaId].to_numpy(float)
    ftir11 = pd.read_csv(PHASE3 / "tables/ftir11/addis_predictions.csv").set_index("MediaId").loc[addis.MediaId]
    fixed = ftir11.in_fixed_phase2_cohort.to_numpy(bool)
    x = addis.Fabs.to_numpy(float) / MAC

    cohorts = load_cohorts()
    sources, from_network, inside = spectra(addis)
    table, fits, sweep = addis.copy(), [], []
    for cohort, (frame, test_sites) in cohorts.items():
        y, sites = frame.TOR_EC_loading_ug.to_numpy(float), frame.Site.to_numpy()
        covered = frame.FilterId.isin(sources["VIBES"][0]).to_numpy()
        for name, (rows, X, XA, key) in sources.items():
            sets = {"cohort": np.ones(len(frame), bool), "VIBES-covered": covered} if "VIBES" not in name else {"VIBES-covered": covered}
            for which, keep in sets.items():
                pos = [rows[int(v)] for v in frame.loc[keep, key]]
                model, k, n_train, held = fit(np.asarray(X[pos], float), y[keep], sites[keep], test_sites)
                pred = model.predict(XA).ravel() / volume
                table[f"{COHORTS[cohort]}, {name} [{which}]"] = pred
                for label, mask in (("fixed 190", fixed), ("all 239", np.ones(len(x), bool))):
                    fits.append({"cohort": COHORTS[cohort], "spectra": name, "filters": which, "n_cohort": int(keep.sum()), "n_train": n_train, "k": k,
                                 "evaluation": label, "n": int(mask.sum()), **regression_metrics(x[mask], pred[mask]),
                                 **{f"heldout_TOR_{m}": v for m, v in held.items() if m in ("R2", "RMSE", "slope")}})
                train = np.ones(keep.sum(), bool) if test_sites is None else ~np.isin(sites[keep], test_sites)
                for fixed_k in FIXED_K:
                    at_k = PLSRegression(n_components=fixed_k, scale=False).fit(np.asarray(X[pos], float)[train], y[keep][train]).predict(XA).ravel() / volume
                    sweep.append({"cohort": COHORTS[cohort], "spectra": name, "filters": which, "k": fixed_k, "rule_k": k,
                                  **regression_metrics(x[fixed], at_k[fixed])})
                print(f"{cohort} {name} [{which}]: n {keep.sum()}, k {k}, {fits[-2]['slope']:.3f}x {fits[-2]['intercept']:+.3f}", flush=True)
                if name == "VIBES" and cohort == "ocec":
                    net_pred = model.predict(from_network).ravel() / volume
                    print(f"  network vs frozen VIBES prediction on the {inside.sum()}: median |diff| "
                          f"{np.median(np.abs(net_pred[inside] - pred[inside])):.3f}, worst {np.abs(net_pred[inside] - pred[inside]).max():.3f} µg/m³", flush=True)

    # the gates: a refit that misses what the notebooks published is a different setup
    for cohort, label in COHORTS.items():
        worst = np.abs(table[f"{label}, AIRSpec df1=6 [cohort]"] - tracked[f"{label}, AIRSpec df1=6"]).max()
        print(f"gate {label}: AIRSpec differs from tracked by at most {worst:.2e} µg/m³", flush=True)
        if not worst < 1e-3:
            raise ValueError(f"{label}: AIRSpec refit does not reproduce ftir_13")
    worst = np.nanmax(np.abs(table["lowest-OCEC 800, raw [cohort]"].to_numpy() - ftir11.EC_lowest_ocec_800_ugm3.to_numpy()))
    print(f"gate raw OC/EC: differs from ftir_11 by at most {worst:.3f} µg/m³", flush=True)
    table["vibes_source"] = np.where(inside, "frozen rank-70", "19 Oct network")
    table.to_csv(OUT / "addis_predictions.csv", index=False)
    pd.DataFrame(fits).to_csv(OUT / "fits.csv", index=False)
    pd.DataFrame(sweep).to_csv(OUT / "k_sweep.csv", index=False)
    print(pd.DataFrame(fits).query("evaluation == 'fixed 190'")[["cohort", "spectra", "filters", "n_cohort", "n_train", "k", "slope", "intercept", "R2", "RMSE", "heldout_TOR_R2"]].round(3).to_string())


if __name__ == "__main__":
    main()
