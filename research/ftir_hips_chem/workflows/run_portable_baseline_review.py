"""Reproduce a bounded site/filter/spectrum investigation from committed inputs.

This is an exploratory diagnostic of FTIR–optical disagreement, not a new EC
calibration. It never replaces source measurements or labels spectra as sources.
Run from any directory with the repository's scientific Python environment:

    python research/ftir_hips_chem/workflows/run_portable_baseline_review.py

The packed raw spectra are coarsened; the clipped/normalized AIRSpec display
trace is deliberately not used. Native Addis checks have a separate workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "research/ftir_hips_chem/scripts"
sys.path.insert(0, str(SCRIPTS))
from config import MAC_VALUE, SITES, season_for_month  # noqa: E402
from portable_baseline_screen import (  # noqa: E402
    asymmetric_baseline, load_packed_spectra, local_band_contrast, run_synthetic_controls,
)
from portable_cohort_screen import (  # noqa: E402
    load_public_chemistry, optical_pair_ledger, site_optical_summary,
)


BANDS = {
    "contrast_1580_1650": ((1580, 1650), ((1500, 1540), (1760, 1800))),
    "contrast_1600_1635": ((1600, 1635), ((1540, 1570), (1660, 1690))),
    "contrast_1700_1750": ((1700, 1750), ((1640, 1670), (1780, 1810))),
    "contrast_2850_2950": ((2850, 2950), ((2750, 2800), (3000, 3040))),
    "contrast_3200_3400": ((3200, 3400), ((3080, 3130), (3500, 3550))),
}
SPECTRAL_FEATURES = [
    *BANDS, "baseline_sensitivity_rms", "protected_1617_change",
    "negative_fraction_soft", "removed_mean_soft", "escape_recomputed",
]
BASELINE_FEATURES = ["baseline_sensitivity_rms", "protected_1617_change", "negative_fraction_soft", "removed_mean_soft"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nuisance_matrix(frame):
    """Descriptive controls, not an errors-in-variables calibration.

    Published optical BC and PM mass control concentration; volume controls
    some concentration-to-loading differences but may be inferred upstream.
    Month harmonics, elapsed time, lot and optical source control measured
    metadata. Missing acquisition batch/calibration identity cannot be fixed.
    """
    dates = pd.to_datetime(frame["date"])
    month = dates.dt.month.to_numpy()
    numeric = np.column_stack([
        frame["optical_bc_ugm3"], frame["pm25_ugm3"], frame["volume_m3"],
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
        (dates - pd.Timestamp("2020-01-01")).dt.days.to_numpy() / 365.25,
    ]).astype(float)
    cats = pd.get_dummies(frame[["lot_group", "optical_source"]].astype(str), dtype=float)
    return np.column_stack([numeric, cats.to_numpy()])


def residualize_rank(values, controls):
    """Linear residuals of ranks; categorical controls stay dummy-coded."""
    c = np.column_stack([np.ones(len(values)), controls])
    ranked = rankdata(values)
    return ranked - c @ np.linalg.lstsq(c, ranked, rcond=None)[0]


def association_table(frame):
    """Within-site screens, avoiding a pooled site-separation association.

    No uncorrected wavelength-by-wavelength significance search is made. These
    ten prelisted indicators remain exploratory because prior results informed
    their selection. Report directions and squared correlation, not p-value
    discovery claims. Predictive utility is checked separately in time blocks.
    """
    rows = []
    for code, group in frame.groupby("code", sort=True):
        eligible = group["spectral_pair_eligible"].to_numpy(bool)
        for feature in SPECTRAL_FEATURES:
            subset = group.loc[eligible].copy()
            cols = [feature, "optical_disagreement_ugm3", "optical_bc_ugm3", "pm25_ugm3", "volume_m3"]
            subset = subset.loc[np.isfinite(subset[cols].to_numpy(float)).all(axis=1)]
            if len(subset) < 20:
                continue
            x = subset[feature].to_numpy(float)
            y = subset["optical_disagreement_ugm3"].to_numpy(float)
            c = nuisance_matrix(subset)
            # Rank continuous controls; keep the one-hot categories intact.
            c[:, :6] = np.column_stack([rankdata(c[:, k]) for k in range(6)])
            xr, yr = residualize_rank(x, c), residualize_rank(y, c)
            raw = float(spearmanr(x, y).statistic) if np.std(x) else np.nan
            partial = float(np.corrcoef(xr, yr)[0, 1]) if np.std(xr) > 1e-10 and np.std(yr) > 1e-10 else np.nan
            rows.append({"code": code, "feature": feature, "n": len(subset),
                         "spearman_r": raw, "spearman_r_squared": raw**2,
                         "partial_rank_r": partial, "partial_rank_r_squared": partial**2,
                         "control_rank": int(np.linalg.matrix_rank(np.column_stack([np.ones(len(c)), c]))),
                         "interpretation": "exploratory FTIR-optical disagreement association; not EC error"})
    return pd.DataFrame(rows)


def select_ridge(x, y, groups):
    """Tune only inside the outer training set, grouped by collection quarter."""
    unique = np.unique(groups)
    alphas = (0.1, 1.0, 10.0, 100.0, 1000.0)
    if len(unique) < 3:
        return 100.0
    splits = list(GroupKFold(min(4, len(unique))).split(x, y, groups))
    losses = []
    for alpha in alphas:
        errors = []
        for train, test in splits:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x[train], y[train])
            errors.extend((model.predict(x[test]) - y[test])**2)
        losses.append(np.mean(errors))
    return alphas[int(np.argmin(losses))]


def chronological_screen(frame):
    """Paired forward-quarter evaluation of residual prediction at Addis.

    Both models see identical training/test filters. Held-out data never enter
    scaling or penalty selection. This predicts disagreement with an optical
    quantity and cannot identify the correct EC measurement.
    """
    if frame["id"].duplicated().any():
        raise ValueError("chronological screen requires one row per physical filter")
    f = frame.loc[(frame["code"] == "ETAD") & frame["spectral_pair_eligible"]].copy()
    need = ["optical_disagreement_ugm3", "optical_bc_ugm3", "pm25_ugm3", "volume_m3", *SPECTRAL_FEATURES]
    f = f.loc[np.isfinite(f[need].to_numpy(float)).all(axis=1)].sort_values("date").reset_index(drop=True)
    if len(f) < 60:
        return pd.DataFrame(), pd.DataFrame()
    groups = pd.to_datetime(f["date"]).dt.to_period("Q").astype(str).to_numpy()
    quarters = np.unique(groups)
    c = nuisance_matrix(f)
    s = f[SPECTRAL_FEATURES].to_numpy(float)
    y = f["optical_disagreement_ugm3"].to_numpy(float)
    rows = []
    for q in quarters:
        train, test = np.where(groups < q)[0], np.where(groups == q)[0]
        if len(train) < 60 or len(test) < 4 or len(np.unique(groups[train])) < 3:
            continue
        predictions, penalties = {}, {}
        designs = {
            "metadata": c,
            "baseline": np.column_stack([c, f[BASELINE_FEATURES].to_numpy(float)]),
            "bands": np.column_stack([c, f[list(BANDS)].to_numpy(float)]),
            "metadata_spectra": np.column_stack([c, s]),
        }
        for label, x in designs.items():
            alpha = select_ridge(x[train], y[train], groups[train])
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x[train], y[train])
            predictions[label] = model.predict(x[test])
            penalties[label] = alpha
        for j, index in enumerate(test):
            rows.append({"id": f.at[index, "id"], "date": f.at[index, "date"], "quarter": q,
                         "train_n": len(train), "train_end": f.iloc[train[-1]]["date"],
                         "actual_disagreement": y[index],
                         "metadata_prediction": predictions["metadata"][j],
                         "baseline_prediction": predictions["baseline"][j],
                         "bands_prediction": predictions["bands"][j],
                         "spectra_prediction": predictions["metadata_spectra"][j],
                         "metadata_alpha": penalties["metadata"], "baseline_alpha": penalties["baseline"],
                         "bands_alpha": penalties["bands"],
                         "spectra_alpha": penalties["metadata_spectra"]})
    predictions = pd.DataFrame(rows)
    summary = []
    if len(predictions):
        for label, g in [("all_test_quarters", predictions), *list(predictions.groupby("quarter"))]:
            actual = g["actual_disagreement"].to_numpy()
            em, es = g["metadata_prediction"].to_numpy()-actual, g["spectra_prediction"].to_numpy()-actual
            eb, ep = g["baseline_prediction"].to_numpy()-actual, g["bands_prediction"].to_numpy()-actual
            summary.append({"quarter": label, "n": len(g), "metadata_rmse": np.sqrt(np.mean(em**2)),
                            "baseline_indicators_rmse": np.sqrt(np.mean(eb**2)),
                            "band_contrasts_rmse": np.sqrt(np.mean(ep**2)),
                            "spectra_rmse": np.sqrt(np.mean(es**2)),
                            "delta_rmse_spectra_minus_metadata": np.sqrt(np.mean(es**2))-np.sqrt(np.mean(em**2)),
                            "metadata_mae": np.mean(np.abs(em)), "spectra_mae": np.mean(np.abs(es)),
                            "units": "ug/m3 optical-disagreement prediction, not independent EC error"})
    return predictions, pd.DataFrame(summary)


def known_case_comparisons(frame):
    """Carry forward named historical cases alongside comparable controls.

    This is a selected investigation panel, not a random or confirmatory sample.
    Concentrations are comparable here only after showing the reported volume.
    """
    ids = ["ETAD-0037", "ETAD-0036", "ETAD-0040", "ETAD-0139", "ETAD-0158", "ETAD-0221", "ETAD-0202"]
    columns = ["id", "filter_id", "date", "lot_group", "type", "ec_ftir_ugm3", "optical_bc_ugm3",
               "optical_disagreement_ugm3", "pm25_ugm3", "volume_m3", "optical_source",
               "contrast_1580_1650", "contrast_1600_1635", "contrast_1700_1750",
               "contrast_2850_2950", "contrast_3200_3400", "baseline_sensitivity_rms",
               "negative_fraction_soft", "escape_recomputed", "spectral_pair_eligible"]
    result = frame.set_index("id", drop=False).reindex(ids).reindex(columns=columns).reset_index(drop=True)
    result["investigation_note"] = result["id"].map({
        "ETAD-0037": "Previously documented near-flat scan; retained for lab adjudication.",
        "ETAD-0036": "Nearby same-lot control for 0037 with comparable PM and optical BC.",
        "ETAD-0040": "Nearby same-lot control for 0037 with comparable PM and optical BC.",
        "ETAD-0139": "Previously documented discordant repeat scans; compare each before averaging.",
        "ETAD-0158": "Previously documented physical hole; retains current source eligibility pending review.",
        "ETAD-0221": "Similar optical BC to 0202, but different PM mass and sampling season.",
        "ETAD-0202": "Similar optical BC to 0221, but different PM mass and sampling season.",
    })
    return result


def shortlist(frame):
    """Balanced investigation panel; selection is not an exclusion rule."""
    rows = []
    for code in ("ETAD", "ETBI", "INDH", "CHTS", "USPA"):
        g = frame.loc[(frame["code"] == code) & frame["spectral_pair_eligible"]].copy()
        if not len(g):
            continue
        g["abs_disagreement"] = g["optical_disagreement_ugm3"].abs()
        g["sensitivity_percentile"] = g["baseline_sensitivity_rms"].rank(pct=True)
        g["disagreement_percentile"] = g["abs_disagreement"].rank(pct=True)
        # Four corners are deliberately retained, including apparently ordinary controls.
        for b in (False, True):
            for d in (False, True):
                candidates = g.loc[(g["sensitivity_percentile"] >= .5) == b]
                candidates = candidates.loc[(candidates["disagreement_percentile"] >= .5) == d].copy()
                candidates["selection_distance"] = ((candidates["sensitivity_percentile"] - float(b))**2
                                                      + (candidates["disagreement_percentile"] - float(d))**2)
                for _, r in candidates.sort_values(["selection_distance", "id"]).head(2).iterrows():
                    row = r.to_dict()
                    row["panel_reason"] = f"{'high' if b else 'low'} baseline sensitivity / {'high' if d else 'low'} optical disagreement"
                    rows.append(row)
    cols = ["code", "id", "filter_id", "date", "lot_group", "optical_source", "ec_ftir_ugm3",
            "optical_bc_ugm3", "optical_disagreement_ugm3", "baseline_sensitivity_rms", "contrast_1580_1650",
            "escape_recomputed", "panel_reason", "qc", "analyses"]
    return pd.DataFrame(rows).reindex(columns=cols)


def make_figure(frame, shortlist_frame, raw, wn, baselines, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plotting import PlotConfig
    PlotConfig.set(sites="all", show_stats=False)
    # No shared plotting routine draws spectra and their removed components.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    etad = shortlist_frame.loc[shortlist_frame["code"] == "ETAD"]
    for ax, (_, row) in zip(axes.flat, etad.groupby("panel_reason", sort=True).first().reset_index().iterrows()):
        i = int(frame.index[frame["id"].eq(row["id"])][0])
        ax.plot(wn, raw[i], color="#303841", lw=1, label="Raw (binned)")
        ax.plot(wn, baselines["soft"][i], color=SITES["Addis_Ababa"]["color"], lw=1.2, label="Exploratory baseline A")
        ax.plot(wn, baselines["stiff"][i], color="#6554a4", lw=1.2, label="Exploratory baseline B")
        ax.axvspan(1580, 1650, color="#777777", alpha=.14)
        ax.set_xlim(3998, 1425)
        ax.set_title(f"{row['id']} · {row['date']}\n{row['panel_reason']}", fontsize=10)
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax.set_ylabel("Absorbance")
        ax.text(.02, .95, f"FTIR − optical BC = {row['optical_disagreement_ugm3']:+.2f} µg/m³", transform=ax.transAxes, va="top", fontsize=9)
    axes.flat[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Addis investigation panel: baseline sensitivity and optical disagreement are separate", fontsize=13)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "research/ftir_hips_chem/output/tables/baseline_review_20261102")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    folder = ROOT / "gallery/app/public/data/research/lot_spectra"
    packed = load_packed_spectra(folder)
    meta = pd.DataFrame(packed["filters"])
    raw_all, wn_all = packed["views"]["raw"]["values"], packed["views"]["raw"]["wavenumbers"]
    shared = wn_all >= 1425
    raw, wn = raw_all[:, shared], wn_all[shared]
    baseline_runs = {
        "soft": asymmetric_baseline(wn, raw, lam=1e4),
        "stiff": asymmetric_baseline(wn, raw, lam=1e6),
        "protected": asymmetric_baseline(wn, raw, lam=1e4, protected_windows=((1580, 1660),)),
    }
    soft, stiff, protected = (baseline_runs[k] for k in ("soft", "stiff", "protected"))
    meta["baseline_sensitivity_rms"] = np.sqrt(np.mean((soft["baseline"]-stiff["baseline"])**2, axis=1))
    band = (wn >= 1580) & (wn <= 1650)
    meta["protected_1617_change"] = np.mean((protected["corrected"]-soft["corrected"])[:, band], axis=1)
    meta["negative_fraction_soft"] = np.mean(soft["corrected"] < 0, axis=1)
    meta["removed_mean_soft"] = np.mean(soft["baseline"], axis=1)
    meta["baseline_converged"] = soft["converged"] & stiff["converged"] & protected["converged"]
    for name, (window, flanks) in BANDS.items():
        contrast = local_band_contrast(wn, raw, band=window, flanks=flanks)
        meta[name] = contrast["height"]
        meta[name+"_peak_wn"] = contrast["peak_wn"]
    descriptor = json.loads((folder / "lot_spectra.json").read_text())
    deriv = packed["views"]["deriv"]["values"]
    low, high = (np.asarray(descriptor["improve"][k], dtype=float) for k in ("deriv_p05", "deriv_p95"))
    if low.shape != (deriv.shape[1],) or high.shape != low.shape:
        raise ValueError("derivative reference envelope does not match the packed grid")
    meta["escape_recomputed"] = ((deriv < low) | (deriv > high)).mean(axis=1)
    meta["lot_group"] = meta["lot"].map(lambda v: "unknown" if pd.isna(v) else str(int(v)))
    meta["ethiopia_season_dry_feb"] = pd.to_datetime(meta["date"]).dt.month.map(season_for_month)
    chemistry = load_public_chemistry(ROOT)
    ledger = optical_pair_ledger(chemistry)
    chemistry_cols = ["id", "code", "date", "is_excluded", "exclusion_reason", "ec_ftir_ugm3", "pm25_ugm3",
                      "volume_m3", "optical_bc_ugm3", "optical_disagreement_ugm3", "optical_source", "pair_eligible", "pair_reason"]
    # Join on explicit filter identity. Dates are audited, not used as fuzzy keys.
    frame = meta.merge(ledger[chemistry_cols].rename(columns={"code": "chemistry_code", "date": "chemistry_date"}),
                       on="id", how="left", validate="one_to_one")
    frame["date_matches"] = frame["date"].eq(frame["chemistry_date"])
    frame["qc_recorded"] = frame["qc"].notna()
    # An explicit non-1 scan code is held for review; missing QC remains unknown,
    # never relabelled as pass. This is recorded eligibility, not value trimming.
    frame["qc_review_required"] = frame["qc"].map(lambda v: v is not None and str(v) not in ("nan", "", "1"))
    frame["spectral_pair_eligible"] = (frame["type"].eq("PM2.5") & frame["pair_eligible"].eq(True)
                                        & frame["date_matches"] & frame["code"].eq(frame["chemistry_code"])
                                        & frame["baseline_converged"] & ~frame["qc_review_required"])
    frame["spectral_pair_reason"] = np.select(
        [~frame["type"].eq("PM2.5"), frame["chemistry_code"].isna(), ~frame["date_matches"],
         ~frame["code"].eq(frame["chemistry_code"]), ~frame["pair_eligible"].eq(True),
         ~frame["baseline_converged"], frame["qc_review_required"]],
        ["non-PM2.5 spectrum", "no public chemistry identity", "collection-date discrepancy", "site-code discrepancy",
         "no eligible optical pair", "baseline did not converge", "explicit scan QC code requires review"],
        default="eligible exploratory pair")
    summary = site_optical_summary(chemistry)
    summary.to_csv(out / "network_optical_summary.csv", index=False)
    ledger[["id", "code", "date", "is_excluded", "exclusion_reason", "ec_ftir_ugm3", "hips_fabs_Mm1",
            "optical_bc_ugm3", "optical_source", "optical_disagreement_ugm3", "pair_eligible", "pair_reason",
            "nonpositive_pair", "pm25_ugm3", "volume_m3"]].to_csv(out / "network_pair_ledger.csv", index=False)
    frame.to_csv(out / "filter_diagnostic_ledger.csv", index=False)
    associations = association_table(frame)
    associations.to_csv(out / "within_site_associations.csv", index=False)
    predictions, validation = chronological_screen(frame)
    predictions.to_csv(out / "addis_forward_predictions.csv", index=False)
    validation.to_csv(out / "addis_forward_summary.csv", index=False)
    sensitivity = []
    for population, eligible in (
        ("direct_hips_only", frame["optical_source"].eq("direct_hips_fabs")),
        ("known_lots_248_251", frame["lot_group"].isin(["248", "251"])),
    ):
        reduced = frame.copy()
        reduced["spectral_pair_eligible"] &= eligible
        _, table = chronological_screen(reduced)
        if len(table):
            table.insert(0, "population", population)
            sensitivity.append(table)
    pd.concat(sensitivity, ignore_index=True).to_csv(out / "addis_forward_population_sensitivity.csv", index=False)
    known_case_comparisons(frame).to_csv(out / "addis_named_case_comparisons.csv", index=False)
    panel = shortlist(frame)
    panel.to_csv(out / "remeasurement_panel.csv", index=False)
    pd.DataFrame(run_synthetic_controls(wn)).to_csv(out / "synthetic_smoother_controls.csv", index=False)
    inventory = frame.groupby(["code", "type"], as_index=False).agg(
        n=("id", "size"), eligible_pairs=("spectral_pair_eligible", "sum"),
        missing_lot=("lot", lambda x: x.isna().sum()),
        median_escape=("escape_recomputed", "median"),
        median_sensitivity=("baseline_sensitivity_rms", "median"),
        date_start=("date", "min"), date_end=("date", "max"))
    inventory.to_csv(out / "spectral_inventory.csv", index=False)
    make_figure(frame, panel, raw, wn, {k:v["baseline"] for k,v in baseline_runs.items()}, out / "addis_investigation_panel.svg")
    # Record exact numerical/data inputs; raw native data are never needed for this portable run.
    sources = [Path(__file__), SCRIPTS / "portable_baseline_screen.py", SCRIPTS / "portable_cohort_screen.py",
               SCRIPTS / "config.py", SCRIPTS / "outliers.py", ROOT / "gallery/app/public/data/spartan_reference.json",
               *(folder / p for p in ("lot_spectra.json", "raw.bin", "deriv.bin"))]
    receipt = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "mac_m2_g": MAC_VALUE, "spectral_provenance": packed["provenance"],
        "baseline_settings": {k:v["settings"] for k,v in baseline_runs.items()},
        "convergence": {k:int(v["converged"].sum()) for k,v in baseline_runs.items()},
        "counts": {"public_rows":len(chemistry), "public_sites":int(chemistry["code"].nunique()),
                   "spectral_rows":len(frame), "spectral_eligible_pairs":int(frame["spectral_pair_eligible"].sum()),
                   "forward_test_predictions":len(predictions)},
        "band_definitions": BANDS,
        "limits": ["Exploratory; spectral features were motivated by historical investigations.",
                   "The response is optical disagreement, not independently measured EC error.",
                   "Public and in-house products are not interchangeable model runs.",
                   "No native raw spectra, actual blanks, or IMPROVE calibration weights enter this portable run.",
                   "Missing lot, scan-batch and calibration-version metadata leave confounding unresolved.",
                   "Volume may be inferred upstream; mass per filter is not treated as an independent observation.",
                   "Chronological tests may also cross an undocumented upstream product/calibration change.",
                   "Panel thresholds are relative within site and are not validated QC cutoffs."],
    }
    receipt["outputs_sha256"] = {p.name:sha256(p) for p in sorted(out.iterdir()) if p.suffix in (".csv", ".svg")}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps(receipt["counts"], indent=2))
    print(inventory.to_string(index=False))
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
