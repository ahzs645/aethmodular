"""Model form, every combination tried, and feature importance (ftir_54, ftir_55, ftir_51 05; ftir_07, ftir_08).

- ftir_54: five model forms (PLS, HGB, RF, MLP, PLS→HGB) on the same site-disjoint split, the labelled
  extrapolation tests inside IMPROVE, the Addis domain diagnostics and the HGB tuning grid. The notebook
  never wrote its per-filter predictions, so the crossplots by model form are listed as missing.
- ftir_55 / ftir_51 05: the cohort × spectra summary and the protocol comparison the notebooks displayed;
  the page redraws the search itself from calibration.json (which the browser already has).
- ftir_07 / ftir_08 (research/ftir_hips_chem): the VIP overlap and band summaries they displayed; the VIP
  profiles, the band edges, and ftir_08's HIPS-target tables (the IMPROVE site-held-out and Addis CV curves,
  each Addis filter's transfer and nested-CV prediction, the nested folds' k) are local CSVs, added when
  present and otherwise listed as missing. The page redraws the transfer crossplots and their metrics live.
"""
from __future__ import annotations

import pandas as pd

from ml_pages._common import HIPS_CHEM, PHASE3, long, number, page, rel

KEY = "model-form"

NB54 = "ftir_54_model_form_robustness"
NB55 = "ftir_55_every_combination_tried"
NB51 = "ftir_51_priority_figure_export"
NB07 = "ftir_07_current_pls_vip_diagnostics"
NB08 = "ftir_08_hips_transfer_and_vip"
T54 = "output/tables/ftir54/"
VIP = "output/tables/pls_transfer/"

OVERLAP = {
    "spearman_r": "Spearman r of the VIP profiles",
    "important_a_n": "wavenumbers with VIP ≥ 1, first model",
    "important_b_n": "wavenumbers with VIP ≥ 1, second model",
    "important_jaccard": "Jaccard overlap of VIP ≥ 1 sets",
    "top_n_overlap": "top-200 wavenumbers shared",
    "top_n_overlap_fraction": "share of top-200 shared",
    "a_vip_mass_on_b_important": "first model's VIP² mass where the second has VIP ≥ 1",
}


def _bands(frame: pd.DataFrame, comparison: str) -> pd.DataFrame:
    """A VIP² band table (the notebook's pivot band × model, or the long CSV) as comparison, model, band, share."""
    if "model" in frame.columns and "vip2_mass_fraction" in frame.columns:
        frame = frame.pivot(index="band", columns="model", values="vip2_mass_fraction").reset_index()
    out = long(frame, ["band"], "model", "vip2_share")
    return out.assign(comparison=comparison)


TRANSFER, ADDIS = "IMPROVE HIPS → Addis external test", "Addis-only HIPS"

SPECTRA = {"airspec": "baseline-corrected", "deriv2": "2nd derivative", "raw": "raw"}


def _best(cells: pd.DataFrame, spectra: str) -> str:
    """ftir_55's best passing lowest-OC/EC row on one spectra treatment, in words."""
    r = cells[(cells.cohort == "ocec") & (cells.spectra == spectra)].iloc[0]
    return (f"{SPECTRA[spectra]}: {int(r.passing)} pass, best score {r.best_score:.2f} at cutoff {int(r.best_cutoff)}, "
            f"k = {int(r.best_k)}, {r.best_slope:.2f}x {_signed(r.best_intercept)}")


def _reads(*sources: dict) -> str:
    """How a two-notebook table was read: one word when both parts agree, else each part's, in path order."""
    reads = [s["read"] for s in sources]
    return reads[0] if len(set(reads)) == 1 else "; ".join(reads)


def _minus(v: float, digits: int) -> str:
    return f"{'−' if v < 0 else ''}{abs(v):.{digits}f}"


def _signed(v: float) -> str:
    return f"{'−' if v < 0 else '+'} {abs(v):.2f}"


def _vip_wide(columns: dict[str, str]):
    """A VIP profile CSV as wn plus one column per model (wide: half the size of the long form; the page melts it)."""
    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.rename(columns={"wavenumber_cm-1": "wn"})[["wn", *columns]].rename(columns=columns)
        return out.assign(wn=out.wn.round(2))
    return transform


# scripts/pls_transfer.py FUNCTIONAL_BANDS: the broad bands summarize_vip_bands sums VIP² over (and the page shades)
FUNCTIONAL_BANDS = ((3000.0, 3600.0, "O-H / N-H region"), (2800.0, 3000.0, "aliphatic C-H stretch"),
                    (1650.0, 1850.0, "carbonyl region"), (1500.0, 1650.0, "amide / aromatic / N-H region"),
                    (500.0, 1500.0, "fingerprint region"))


def _band_edges(*read: tuple[pd.DataFrame, dict]) -> tuple[pd.DataFrame, dict]:
    """The band edges from the first band CSV that carries them, else the module's constant."""
    for frame, source in read:
        if source["read"] == "csv" and {"low_cm-1", "high_cm-1"} <= set(frame.columns):
            edges = frame.drop_duplicates("band")[["band", "low_cm-1", "high_cm-1"]].rename(columns={"low_cm-1": "low", "high_cm-1": "high"})
            return edges, {"notebook": source["notebook"], "read": "csv", "path": source["path"]}
    edges = pd.DataFrame(FUNCTIONAL_BANDS, columns=["low", "high", "band"])[["band", "low", "high"]]
    return edges, {"notebook": NB08, "read": "tracked source", "path": rel(HIPS_CHEM / "scripts/pls_transfer.py"), "cell": "FUNCTIONAL_BANDS"}


# ftir_08 [9]'s per-filter predictions, under short names (Fabs, observed; the three models' predicted Fabs)
PREDICTIONS = {"ExternalFilterId": "id", "Fabs": "Fabs", "IMPROVE_HIPS_transfer_prediction": "improve",
               "Addis_HIPS_nested_CV_prediction": "addis", "naive_direct_Fabs_prediction": "naive"}


def build():
    p = page(KEY, "Model form, every combination tried, and feature importance")

    # ------------------------------------------------------------ ftir_54 model-form robustness
    forms = p.shown("model_forms", PHASE3, NB54, "summary.to_csv", csv=T54 + "model_form_comparison.csv")
    extrap = p.shown("extrapolation", PHASE3, NB54, "extrap.to_csv", csv=T54 + "labelled_extrapolation_tests.csv")
    diag, source = p.read(PHASE3, NB54, "diag_corr.to_csv", csv=T54 + "addis_domain_diagnostics_spearman.csv", index="diagnostic")
    p.add("domain_spearman", long(diag, ["diagnostic"], "against", "rho"), **source)
    p.shown("hgb_tuning", PHASE3, NB54, "tune.to_csv", csv=T54 + "hgb_grouped_cv_tuning_800.csv")
    beyond = p.stream(PHASE3, NB54, "diag_corr.to_csv")
    rebuild54 = f"cd {rel(PHASE3)} && uv run python scripts/build_notebooks.py 54"
    for table, what, cell, needs in (
        ("addis_by_model_form", "each of the 190 fixed Addis filters' prediction under each model form, cohort and representation "
         "(ftir_54 [9] crossplot by model form; never written — the notebook keeps them in `predictions` only)",
         "[8]", "addis_predictions_by_model_form.csv"),
        ("test_a_by_model_form", "the 800 lowest-OC/EC filters' Test A predictions per model form (ftir_54 [13b]; kept in `ext_preds` only)",
         "[12]", "labelled_extrapolation_predictions.csv"),
        ("addis_d2_by_model_form", "each Addis filter's PCA D², PLS D² and random-forest spread with its prediction per model form (ftir_54 [15]; "
         "only the Spearman summary was written)", "[15]", "addis_domain_diagnostics_per_filter.csv"),
    ):
        p.missing.append({"table": table, "what": what, "needs": rel(PHASE3 / (T54 + needs)),
                          "run": f"add a to_csv of it to ftir_54 cell {cell}, then {rebuild54}",
                          "after": "ftir_11's lowest_ocec_800_cohort.csv and ftir_13's corrected pool (output/corrected/*_df6.npz)"})

    # ------------------------------------------------------------ ftir_55 / ftir_51 05: every combination tried
    cells = p.shown("ftir55_cells", PHASE3, NB55, "cell.to_csv", csv="output/tables/ftir55/cohort_by_spectra_summary.csv")
    cache = p.stream(PHASE3, NB55, 'addis.to_csv(TABLES / "addis_all_configurations.csv"')
    p.shown("cv_protocols", PHASE3, NB51, "cv.to_csv", csv="output/tables/ftir51/cv_protocol_comparison.csv")

    # ------------------------------------------------------------ ftir_07 / ftir_08: feature importance
    # the cell first displays the Drive source inventory (local paths: not exported), then the reconstruction check
    p.shown("vip_reconstruction", HIPS_CHEM, NB07, "reconstruction.to_csv", nth=1, csv=VIP + "current_model_reconstruction_check.csv")
    ec_oc, s07 = p.read(HIPS_CHEM, NB07, "overlap_table.to_csv", csv=VIP + "ec_oc_vip_overlap.csv")
    hips, s08 = p.read(HIPS_CHEM, NB08, "vip_compare_table.to_csv", nth=0, csv=VIP + "improve_addis_hips_vip_overlap.csv")
    overlap = pd.concat([ec_oc.assign(comparison="ftir_07 · IMPROVE EC vs OC (July 7 models)", first="EC", second="OC"),
                         hips.assign(comparison="ftir_08 · IMPROVE HIPS vs Addis HIPS", first="IMPROVE HIPS", second="Addis HIPS")],
                        ignore_index=True)
    p.add("vip_overlap", overlap, notebook=f"{NB07}; {NB08}", read=_reads(s07, s08), path=f"{s07['path']}; {s08['path']}",
          cell="overlap_table.to_csv; vip_compare_table.to_csv")
    b07, sb07 = p.read(HIPS_CHEM, NB07, "band_summary.to_csv", nth=0, csv=VIP + "current_model_vip_band_summary.csv")
    b08, sb08 = p.read(HIPS_CHEM, NB08, "vip_compare_table.to_csv", nth=1, csv=VIP + "improve_addis_hips_vip_bands.csv")
    p.add("vip_bands", pd.concat([_bands(b07, "ftir_07 · EC vs OC"), _bands(b08, "ftir_08 · IMPROVE vs Addis HIPS")], ignore_index=True),
          notebook=f"{NB07}; {NB08}", read=_reads(sb07, sb08), path=f"{sb07['path']}; {sb08['path']}", cell="band_summary.to_csv; vip_compare_table.to_csv")
    p.shown("shared_peaks", HIPS_CHEM, NB07, "band_summary.to_csv", nth=1, csv=VIP + "ec_oc_shared_vip_peaks.csv")
    metrics = p.shown("hips_transfer", HIPS_CHEM, NB08, "metric_table.to_csv", csv=VIP + "hips_transfer_model_metrics.csv").set_index("model")
    p.optional("vip_ec_oc", HIPS_CHEM, NB07, VIP + "current_ec_oc_vip_profiles.csv",
               "ftir_07's EC and OC VIP profiles on the 2,722-point grid (written locally; needs the Drive's July 7 model exports)",
               transform=_vip_wide({"EC_VIP": "EC", "OC_VIP": "OC"}))
    p.optional("vip_hips", HIPS_CHEM, NB08, VIP + "improve_addis_hips_vip_profiles.csv",
               "ftir_08's IMPROVE-HIPS and Addis-HIPS VIP profiles (written locally; needs the Drive's SPARTAN HIPS and ETAD spectra)",
               transform=_vip_wide({"IMPROVE_HIPS_VIP": "IMPROVE HIPS", "Addis_HIPS_VIP": "Addis HIPS"}))
    edges, se = _band_edges((b08, sb08), (b07, sb07))
    p.add("vip_band_edges", edges, **se)

    # ------------------------------------------------------------ ftir_08: HIPS as the PLS target
    p.shown("hips_audit", HIPS_CHEM, NB08, "data_audit.to_csv", nth=0, csv=VIP + "hips_transfer_data_audit.csv")
    drive = "written locally; needs the Drive's SPARTAN HIPS and ETAD spectra"
    p.optional("hips_cv_improve", HIPS_CHEM, NB08, VIP + "improve_hips_group_cv_curve.csv",
               f"ftir_08 [7]'s IMPROVE HIPS component curve, site-held-out 5-fold RMSE of optical depth τ per k ({drive})")
    p.optional("hips_cv_addis", HIPS_CHEM, NB08, VIP + "addis_hips_cv_curve.csv",
               f"ftir_08 [9]'s Addis-only HIPS component curve, random 5-fold RMSE of τ per k ({drive})")
    p.optional("hips_nested_k", HIPS_CHEM, NB08, VIP + "addis_nested_cv_component_choices.csv",
               f"ftir_08 [9]'s nested-CV outer folds and the k each inner search chose ({drive})")
    preds = p.optional("hips_addis_filters", HIPS_CHEM, NB08, VIP + "addis_hips_transfer_predictions.csv",
                       f"each of the 239 Addis filters' observed HIPS Fabs and its IMPROVE-transfer, Addis nested-CV and naive direct-Fabs predictions "
                       f"(ftir_08 [10] transfer crossplot; {drive})",
                       transform=lambda f: f[list(PREDICTIONS)].rename(columns=PREDICTIONS))

    # ------------------------------------------------------------ headline, read off the same tables
    locked = forms[(forms.cohort == "lowest-OC/EC 800") & (forms.representation == "corrected")].set_index("model")
    b = extrap[extrap.test.str.startswith("B")].set_index("model")
    p.text["finding"] = (
        f"**Model form (ftir_54).** On the locked lowest-OC/EC 800 cohort with corrected spectra, PLS holds held-out IMPROVE "
        f"R² {locked.loc['PLS', 'heldout_R2']:.3f} against {locked.drop('PLS').heldout_R2.min():.2f}–{locked.drop('PLS').heldout_R2.max():.2f} for the other four forms, "
        f"and all {len(forms)} fits put the Addis intercept (against Fabs/10) below zero (−{-forms.addis_intercept.min():.1f} to −{-forms.addis_intercept.max():.1f} µg/m³). "
        f"Where the truth is known (inside IMPROVE), PLS extrapolates best: at the spectral edge it keeps slope {b.loc['PLS', 'slope']:.2f} "
        f"while the trees fall to {b.loc[['HGB', 'RF'], 'slope'].min():.2f}–{b.loc[['HGB', 'RF'], 'slope'].max():.2f}. "
        f"Addis sits inside the 800-cohort envelope ({number(r'PCA D² (\d+)%', beyond, int)} % beyond the PCA D² p95, "
        f"{number(r'PLS D² (\d+)%', beyond, int)} % beyond the PLS D² p95).\n\n"
        f"**Every combination tried (ftir_55).** The explorer's cache held {number(r'unique configurations: ([\d,]+)', cache, lambda s: int(s.replace(',', ''))):,} "
        f"Addis configurations; {number(r'slope 0.7 to 1.3\): (\d+)', cache, int)} passed the core guardrails (held-out R² ≥ 0.85, slope 0.7–1.3) and "
        f"{number(r'all five diagnostics present and passing: (\d+)', cache, int)} all five, every one of them in the lowest-OC/EC family "
        f"(Deming against HIPS Fabs/10; {_best(cells, 'deriv2')}; {_best(cells, 'airspec')}). The 2nd-derivative row scores best but is a manual-k row that does not transfer "
        "to Bishoftu; the baseline-corrected one is the vetted winner.\n\n"
        f"**Feature importance (ftir_07, ftir_08).** The IMPROVE EC and OC models share {int(ec_oc.top_n_overlap.iloc[0])} of their top-200 VIP "
        f"wavenumbers (Spearman r {ec_oc.spearman_r.iloc[0]:.3f}); the IMPROVE and Addis HIPS models share {int(hips.top_n_overlap.iloc[0])} "
        f"(r {hips.spearman_r.iloc[0]:.3f}).\n\n"
        f"**HIPS as the target (ftir_08).** Trained on IMPROVE HIPS optical depth (k {int(metrics.loc[TRANSFER, 'components'])}, chosen on held-out sites), "
        f"the model predicts Addis Fabs with slope {metrics.loc[TRANSFER, 'slope']:.2f} and bias {_minus(metrics.loc[TRANSFER, 'bias'], 1)} "
        + (f"(median {preds.improve.median():.1f} against {preds.Fabs.median():.1f} observed) " if preds is not None else "")
        + f"on {int(metrics.loc[TRANSFER, 'n'])} filters; an Addis-only model reaches nested-CV R² {metrics.loc[ADDIS, 'R2']:.3f}, RMSE {metrics.loc[ADDIS, 'RMSE']:.2f}. "
        "Fabs against Fabs: no MAC enters.")
    return p
