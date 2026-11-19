"""Is there an IMPROVE calibration that brings Addis FTIR EC onto linearised HIPS? (19 Nov 2026)

Since HIPS was linearised (docs/hips-loading-protocol.md) no calibration choice has been re-read against it. This
refits the calibration over the choices that have moved Addis EC before, fits each result against HIPS as measured
and linearised (the central IMPROVE ceiling and both ends of its interval), and scores the same model against
thermal EC on IMPROVE sites it was not trained on, so a model that reaches a slope of 1 at Addis can be read beside
what it does where EC is known.

* spectra     raw (no baseline), spline baseline, VIBES (rank 70, tau 0.1, converged)
* training    eleven sets: every QC-passed IMPROVE filter; thermal EC below 150, 50 or 25 µg; the six urban sites;
              the locked lowest-OC/EC cohort; the smokiest tenth by non-soil potassium; filters of 15 µg EC or more;
              lot 251 only; the 500 spectra nearest the Addis median spectrum; every filter, weighted toward heavy ones
* components  1 to 30

No model is fitted to an Addis reference value. "Nearest the Addis median" uses Addis spectra, not Addis HIPS or EC.

Inputs: the 22 Oct decision bundle (``--bundle``; only on Drive) and the 16 Nov run's cached VIBES spectra and row
table (``output/tables/vibes_tau_calibration_20261116``). Writes, under
``output/tables/linearised_calibration_sweep_20261119/`` (git-ignored):

* ``sweep.csv``        one row per model and HIPS comparator: Deming slope and intercept (µg/m³ at MAC 10, lambda as
                       the closeout), R², median FTIR ÷ HIPS, the MAC at which the slope is 1, and the IMPROVE scores
* ``predictions.npz``  each model's Addis predictions (µg/m³) at every k, with the filter ids and the comparators

    uv run python research/ftir_hips_chem/workflows/run_linearised_calibration_sweep_20261119.py --bundle BUNDLE
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(AREA / "scripts")]
import calibration_gain as cg  # noqa: E402
import continuum_anchor as ca  # noqa: E402
from hips_loading import DEPOSIT_AREA_CM2, IMPROVE_CEILING  # noqa: E402
from pls_transfer import deming_regression  # noqa: E402

LAMBDA_ADDIS = 2.96          # the closeout's error ratio at MAC 10 (gallery calibration.json deming_lambda_mac10)
MAC = 10.0
KMAX = 30
NEAREST = 500
URBAN = ("ATLA1", "BIRM1", "FRES1", "PHOE1", "PITT1", "PUSO1")
CACHE = AREA / "output/tables/vibes_tau_calibration_20261116"
OUT = AREA / "output/tables/linearised_calibration_sweep_20261119"
SPECTRA = ("raw", "spline", "VIBES")
ALL, WEIGHTED, NEAR = "All IMPROVE filters", "All filters, weighted toward heavy ones", f"{NEAREST} nearest the Addis median spectrum"


def comparators(fabs, volume):
    """HIPS BC (µg/m³ at MAC 10) as measured and linearised at each ceiling; NaN at or past a ceiling."""
    tau = fabs * volume / (100.0 * DEPOSIT_AREA_CM2)
    out = {"measured": fabs / MAC}
    for name, ceiling in IMPROVE_CEILING.values().items():
        out[name] = np.where(tau < ceiling, fabs / (1.0 - tau / ceiling), np.nan) / MAC
    return out


def training_sets(meta, keep, X, addis):
    """Row masks over the IMPROVE spectra. Only the nearest-to-Addis set depends on the spectra it is drawn from."""
    y = meta.ec_ug.to_numpy(float)
    nsk = meta.nsk_ugm3.to_numpy(float)
    distance = np.linalg.norm(X - np.median(addis, axis=0), axis=1)
    nearest = np.zeros(len(y), bool)
    nearest[np.flatnonzero(keep)[np.argsort(distance[keep])[:NEAREST]]] = True
    return {
        ALL: keep,
        "Thermal EC below 150 µg": keep & (y < 150),
        "Thermal EC below 50 µg": keep & (y < 50),
        "Thermal EC below 25 µg": keep & (y < 25),
        "Six urban sites": keep & meta.urban.to_numpy(bool),
        "Lowest OC/EC (locked cohort)": keep & meta.locked800.to_numpy(bool),
        "Smokiest tenth (non-soil potassium)": keep & (nsk >= np.nanquantile(nsk[keep], 0.9)),
        "Thermal EC 15 µg or more": keep & (y >= 15),
        "Lot 251 only": keep & meta.lot.astype(str).str.startswith("251").to_numpy(),
        NEAR: nearest,
        WEIGHTED: keep,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True, help="decision_workbook_20261022/bundle")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    bundle = args.bundle

    improve, addis = pd.read_csv(bundle / "improve_meta.csv"), pd.read_csv(bundle / "addis_meta.csv")
    rows = pd.read_csv(CACHE / "improve_rows.csv")
    if not np.array_equal(rows.filter_id.astype(str), improve.filter_id.astype(str)):
        raise ValueError("the 16 Nov row table and the bundle's IMPROVE rows are not the same filters in the same order")
    y = improve.ec_ug.to_numpy(float)
    sites, dates = improve.site.to_numpy(), improve.date.to_numpy()
    keep = np.isfinite(y) & (y > 0) & rows.raw_qc_ok.to_numpy(bool)
    scored = np.flatnonzero(keep)
    folds = [(np.flatnonzero(keep)[tr], np.flatnonzero(keep)[te]) for tr, te in cg.site_folds(sites[keep])]
    weights = cg.loading_weights(y)
    paths = {"raw": (bundle / "improve_raw.npy", bundle / "addis_raw.npy"),
             "spline": (bundle / "improve_spline.npy", bundle / "addis_spline.npy"),
             "VIBES": (CACHE / "corrected_tau0.1_improve.npy", CACHE / "corrected_tau0.1_addis.npy")}

    counted = ~addis.excluded.to_numpy(bool)
    volume = addis.volume_m3.to_numpy(float)[counted]
    x = comparators(addis.fabs_Mm1.to_numpy(float)[counted], volume)

    records, predictions = [], {}
    for name in SPECTRA:
        X, XA = np.asarray(np.load(paths[name][0], mmap_mode="r"), float), np.asarray(np.load(paths[name][1]), float)[counted]
        for label, members in training_sets(improve, keep, X, XA).items():
            w = weights if label == WEIGHTED else None
            fit = lambda rows_: ca.pls_fit(X[rows_], y[rows_], KMAX, weights=None if w is None else w[rows_])  # noqa: E731
            ec = ca.pls_predict(fit(np.flatnonzero(members)), XA) / volume[:, None]                              # µg/m³
            predictions[f"{name}|{label}"] = ec.astype(np.float32)
            # thermal EC on IMPROVE sites the model was not trained on: the set's filters at the other sites train it
            held = np.full((len(y), KMAX), np.nan)
            for train, test in folds:
                own = train[members[train]]
                if len(own) > KMAX + 5:
                    held[test] = ca.pls_predict(fit(own), X[test])
            for k in range(1, KMAX + 1):
                score = cg.fair_scores(held[scored, k - 1], y[scored], sites[scored], dates[scored], URBAN)
                for comparator, hips in x.items():
                    ok = np.isfinite(hips) & np.isfinite(ec[:, k - 1])
                    line = deming_regression(pd.Series(hips[ok]), pd.Series(ec[ok, k - 1]), LAMBDA_ADDIS)
                    records.append({"spectra": name, "training": label, "n_training": int(members.sum()), "k": k, "hips": comparator, "n": int(ok.sum()),
                                    "slope": line["slope"], "intercept": line["intercept"], "r2": float(np.corrcoef(hips[ok], ec[ok, k - 1])[0, 1] ** 2),
                                    "median_ratio": float(np.median(ec[ok, k - 1] / hips[ok])), "mac_for_slope_1": MAC / line["slope"],
                                    "median_ec_ugm3": float(np.median(ec[ok, k - 1])), "mean_hips_ugm3": float(hips[ok].mean()),
                                    "improve_urban_level": score["urban_level"], "improve_heavy_level": score["heavy_level"], "improve_light_level": score["light_level"],
                                    "improve_median_abs_error_ug": score["overall_median_abs_error"], "improve_rmse_ug": score["overall_rmse"]})
            print(name, "|", label, int(members.sum()), "done", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(args.out / "sweep.csv", index=False)
    np.savez_compressed(args.out / "predictions.npz", filter_id=addis.filter_id.to_numpy(str)[counted], season=addis.season.to_numpy(str)[counted],
                        **{f"hips_{name}": values for name, values in x.items()}, **{f"ec|{key}": value for key, value in predictions.items()})
    central = table[table.hips.eq("central") & table.training.eq(ALL) & table.k.isin([4, 8, 12, 18, 25, 30])]
    print(central.pivot(index="spectra", columns="k", values="slope").round(2).to_string())
    print(central.pivot(index="spectra", columns="k", values="intercept").round(2).to_string())
    print("largest slope against linearised HIPS (central):", round(float(table[table.hips.eq("central")].slope.max()), 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
