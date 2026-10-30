"""The Machine learning section's map: every model-fitting result in the notebooks, and where the gallery shows it.

Curated from the 30 Oct 2026 notebook audit (docs/notebook-gallery-audit-2026-10-30.md): one row per group of
figures that needs a model fitted on spectra. `where` is the gallery page that draws it (a hash link), `status`
says whether that page draws it from this checkout's data, and `needs` what would complete a partial or missing row.
"""
from __future__ import annotations

import pandas as pd

from ._common import page

KEY = "overview"
AUDIT = "docs/notebook-gallery-audit-2026-10-30.md"

# status: here (this section), elsewhere (another gallery page), partial, missing
# (group, result, notebooks, where label, link, status, needs)
ROWS = [
    ("Component selection", "PLS factors under site-grouped and interleaved cross-validation; the decision and verdict tables",
     "ftir_11, ftir_20, ftir_23, ftir_32, ftir_31 §1.5", "Machine learning → Components and CV", "#tab=ml&aep=components", "here", ""),
    ("Component selection", "RMSECV by k and the rule k for every configuration and preset",
     "calibration explorer", "Calibration → k sweep, CV curve", "#tab=calibration", "elsewhere", ""),
    ("Component selection", "Interleaved 10-fold and grouping × fold-count (2×2) curves",
     "ftir_20, ftir_23, ftir_32", "", "", "missing", "a re-run writes ftir20/rmsecv_curves.csv, ftir23/selection_curves.csv, ftir32/cv_scheme_2x2_summary.csv"),
    ("Component selection", "Out-of-site RMSE vs k for three preprocessings",
     "decision_workbook_20261022 §2/§7, experiments_20261022 §4", "Machine learning → Decision workbook", "#tab=ml&aep=decision", "partial",
     "pooled RMSE over k is drawn; the workbook's §2 mean-over-folds curve was never saved (PNG only): add it to cell 39's save list"),
    ("Component selection", "Smoke-like IMPROVE filters drifting with k as Addis does; IMPROVE validation blind to k",
     "ksweep_20261014", "Machine learning → k sweep vs IMPROVE", "#tab=ml&aep=ksweep", "here", ""),
    ("Protocols and transfer", "Interleaved (app) vs site-held-out end to end, transfer roundup, bootstrap intervals",
     "ftir_13, ftir_15, ftir_17, ftir_18, ftir_21, ftir_22, ftir_31", "Machine learning → Protocols and transfer", "#tab=ml&aep=protocols", "here", ""),
    ("Protocols and transfer", "One calibration scored on early/late and odd/even halves",
     "calibration explorer", "Calibration → Blind-half check", "#tab=calibration", "elsewhere", ""),
    ("Protocols and transfer", "MAC 6 vs 10 across six setups",
     "ftir_16, ftir_19", "Research → MAC 6 vs 10", "#tab=research&aep=mac-choice", "elsewhere", ""),
    ("Calibration variants", "CAL-0…CAL-7 on 906 IMPROVE filters, chasing the origin, calibration stability",
     "ftir_ec_calibration_2026_06_25 05–14, spartan_ec_2026_06_16 05/07/08", "Machine learning → June 2026 variants", "#tab=ml&aep=june", "here", ""),
    ("Calibration variants", "Ethiopia EC by sample choice, components and lots",
     "ftir_ec_calibration_2026_06_25 07/09/10/12/15, spartan_ec 08", "Research → Ethiopia calibrations (Jun 2026)", "#tab=research&aep=ethiopia-june", "elsewhere", ""),
    ("Residuals and targets", "Residual learners against shuffled-label nulls; the paired increment and mass",
     "ftir_43, ftir_45, ftir_46", "Machine learning → Residual learners", "#tab=ml&aep=residual", "here", ""),
    ("Residuals and targets", "Calibrations on other EC definitions; Adama TOR through the locked calibrations",
     "ftir_41, ftir_42, ftir_44, ftir_48", "Machine learning → What counts as EC", "#tab=ml&aep=targets", "here", ""),
    ("Residuals and targets", "The Adama TOR pilot against IMPROVE, the network and Addis",
     "experiments_20261022 §8", "Experimental → Adama TOR study", "#tab=experimental&xp=tor", "elsewhere", "its export (experimental/) is git-ignored"),
    ("Model form and importance", "PLS against other model forms; every combination tried; VIP tables",
     "ftir_54, ftir_55, ftir_51, ftir_07, ftir_08", "Machine learning → Model form and VIP", "#tab=ml&aep=model-form", "here", ""),
    ("Model form and importance", "The specification curve over every scored configuration",
     "ftir_55, calibration explorer", "Calibration → Specification curve", "#tab=calibration", "elsewhere", "spec_curve.json is git-ignored"),
    ("Model form and importance", "VIP profiles of the deployed and corrected models",
     "28 Oct meeting §8, Satoshi follow-up", "Meeting → 28 Oct; Satoshi", "#tab=meeting&meeting=meeting-20261028", "elsewhere", ""),
    ("Model form and importance", "VIP spectra of the HIPS-target models; HIPS as the PLS target (CV curves, Addis transfer)",
     "ftir_08", "Machine learning → Model form and VIP", "#tab=ml&aep=model-form", "here", ""),
    ("Model form and importance", "ftir_07's EC vs OC VIP profile",
     "ftir_07", "Machine learning → Model form and VIP", "#tab=ml&aep=model-form", "partial", "its overlap tables are drawn; current_ec_oc_vip_profiles.csv is not among the shared outputs"),
    ("Domain and applicability", "T² and Q, di-PLS, hubness, per-filter domain vs error (follows the MAC)",
     "ftir_36, ftir_39, ftir_50, char_12", "Research → Applicability domain", "#tab=research&aep=applicability", "elsewhere", ""),
    ("Domain and applicability", "Ward classes over the IMPROVE network, site similarity, match threshold",
     "ftir_52", "Research → Network spectral map", "#tab=research&aep=network-map", "elsewhere", ""),
    ("Domain and applicability", "Residual vs D² per filter; ftir_50's per-filter T²/Q ratios; hubness Lorenz",
     "ftir_15, ftir_22, ftir_30, ftir_50, ftir_54", "", "", "missing", "per-filter D² in export_calibration_runs.py; ftir_50 re-run"),
    ("Baseline correction", "AIRSpec spline vs VIBES: paired spectra, held-out blanks, injections, rank",
     "ftir_vibes_vs_airspec, VIBES Colab 01–08, vibes_followup_experiments", "Baselines", "#tab=baseline", "elsewhere", "its exports (baseline_*) are git-ignored"),
    ("Analogs and cohorts", "Cohort rankings and compositions, the analog lab, look-alike checks",
     "ftir_09, ftir_33, meeting_followup_20260917, ann_weekly_20260910", "Calibration → cohorts, Analog lab; Meeting 17 Sep", "#tab=calibration", "elsewhere", ""),
    ("Analogs and cohorts", "Local calibrations on shape and loading neighbours",
     "experiments_20261022 §3", "Experimental → Analogs", "#tab=experimental&xp=analogs", "elsewhere", "its export (experimental/) is git-ignored"),
    ("Analogs and cohorts", "Analog membership and bootstrap selection frequencies; PCA/T²/Q scores",
     "ann_weekly_20260910_figures, ann_weekly_20260910_diagnostics, ftir_09", "", "", "missing", "the tables are in local output/ only: export them"),
    ("Error predictors and flags", "Error-predicting selector, abstention, winner classifier, oracle",
     "experiments_20261022 §5", "Experimental → Selector", "#tab=experimental&xp=selector", "elsewhere", "its export (experimental/) is git-ignored"),
    ("Error predictors and flags", "What the FTIR EC flag sees and where it is blind",
     "decision_workbook_20261022 §6b", "Experimental → Flag", "#tab=experimental&xp=flag", "elsewhere", ""),
    ("Error predictors and flags", "Where the reference model fails, specialist calibrations, the flag on SPARTAN sites (live, follows the MAC)",
     "decision_workbook_20261022 §3, §4, §6, §7, §8", "Machine learning → Decision workbook", "#tab=ml&aep=decision", "here", ""),
    ("Error predictors and flags", "Later-year, new-site and random splits by k",
     "experiments_20261022 §6", "Experimental → Forward", "#tab=experimental&xp=forward", "elsewhere", "its export (experimental/) is git-ignored"),
    ("Charcoal archive", "Silhouette, KMeans type mixes, Mahalanobis summaries, transform benchmark; analog matching; feedstock classifiers",
     "char_02, char_03, char_04, char_06, char_11", "Machine learning → Charcoal archive", "#tab=ml&aep=charcoal", "partial",
     "the PCA scores, per-filter Mahalanobis, Ward linkage and AIRSpec fit traces were never saved: a save in run_char_02/04/11.py"),
    ("Charcoal archive", "Charcoal similarity per Addis filter by season, lot and loading (live)",
     "char_02, char_03, char_05–char_09", "Research → Charcoal similarity", "#tab=research&aep=charcoal", "elsewhere", ""),
    ("Charcoal archive", "Calibration-set spectra (906, 952) against Addis",
     "July07, char_12", "Calibration → cohort spectra", "#tab=calibration", "partial", "add the 906 and 952 sets to calibration_runs.json cohort_spectra"),
]


def build():
    p = page(KEY, "Machine learning: every model-fitting result and where it lives")
    frame = pd.DataFrame(ROWS, columns=["group", "result", "notebooks", "where", "link", "status", "needs"])
    p.add("map", frame, notebook="the 30 Oct 2026 notebook audit", read="curated", path=AUDIT)
    counts = frame.status.value_counts()
    p.text["finding"] = (
        f"{len(frame)} groups of model-fitting results: {counts.get('here', 0)} drawn in this section, {counts.get('elsewhere', 0)} "
        f"on other gallery pages, {counts.get('partial', 0)} partly and {counts.get('missing', 0)} not yet (each says what it needs).")
    return p
