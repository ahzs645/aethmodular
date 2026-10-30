"""Component selection and cross-validation schemes: ftir_11 §8, ftir_20, ftir_23, ftir_32 (ftir_31 §1.5 redraws ftir_23).

How many PLS factors each calibration keeps depends on two choices made together: the cross-validation
scheme (site-grouped 5-fold vs the Calibration app's interleaved 10-fold) and the stopping rule (first
major minimum vs first k within 5 % of the minimum). The executed notebooks display their decision and
verdict tables; the curves themselves are written only to the gitignored output/tables, so they are
listed as missing. The site-grouped curves the gallery already has (calibration.json sweeps,
calibration_runs.json curves) and the rule k under each protocol (calibration.json grid) are drawn by
the page directly and not duplicated here.
"""
from __future__ import annotations

import re

import pandas as pd

from ml_pages._common import PHASE3, long, page, rel

KEY = "components"

NB11 = "ftir_11_ocec_ratio_cohort"
NB20 = "ftir_20_component_selection_across_setups"
NB23 = "ftir_23_component_selection_by_protocol"
NB32 = "ftir_32_cv_scheme_ablation"

RULES = {
    "app protocol k (interleaved + within-5%)": "interleaved 10-fold + within 5 % (app)",
    "phase-3 protocol k (grouped + first-major)": "site-grouped 5-fold + first major minimum (phase 3)",
    "grouped CV + within-5%": "site-grouped 5-fold + within 5 %",
    "interleaved CV + first-major": "interleaved 10-fold + first major minimum",
    "locked k in use": "locked k in use",
}


def build():
    p = page(KEY, "Component selection and cross-validation schemes")
    out = "output/tables/"

    # ftir_20 §3: both schemes × both rules on six setups (the notebook displays the raw-spectra rows)
    summary = p.shown("scheme_summary", PHASE3, NB20, "component_selection_summary.csv", csv=out + "ftir20/component_selection_summary.csv")
    if "preprocessing" not in summary.columns:
        p.tables["scheme_summary"] = summary.assign(preprocessing="raw")
    optimism = p.shown("optimism", PHASE3, NB20, "interleaved_cv_optimism.csv", csv=out + "ftir20/interleaved_cv_optimism.csv")

    # ftir_20 §4: k under each rule next to the locked k, one row per (setup, rule)
    k_rule, source = p.read(PHASE3, NB20, "k_by_rule_vs_locked.csv", csv=out + "ftir20/k_by_rule_vs_locked.csv")
    k_rule = k_rule.drop(columns=[c for c in k_rule.columns if c == "locked k source"])
    k_long = long(k_rule, ["cohort"], "rule", "k")
    k_long["rule"] = k_long["rule"].map(lambda r: RULES.get(r, r))
    p.add("k_by_rule", k_long, **source)

    # ftir_23 §3: what each protocol selected (on the training side of the site split) and what it does at Addis
    selection = p.shown("selection", PHASE3, NB23, "display(selection[", csv=out + "ftir23/selection_decisions.csv")
    p.tables["selection"] = selection.drop(columns=[c for c in ("local_minima",) if c in selection.columns])

    # ftir_32: the CV-scheme 2 × 2 (fold structure × fold count). The summary is printed, the verdict displayed.
    scheme_csv = PHASE3 / out / "ftir32/cv_scheme_2x2_summary.csv"
    if scheme_csv.exists():
        p.add("scheme_2x2", pd.read_csv(scheme_csv), notebook=NB32, read="csv", path=rel(scheme_csv))
    else:
        text = p.stream(PHASE3, NB32, 'summary.to_csv(TABLE_DIR / "cv_scheme_2x2_summary.csv"')
        rows = [{"cohort": c.strip(), "scheme": s.strip(), "floor_pct_rmsecv": float(f), "k_within_5pct": int(a), "k_global_min": int(b)}
                for c, s, f, a, b in re.findall(r"^(.+?\(\d+\))\s+(\S+ · \d+-fold)\s+floor\s+([\d.]+)%\s+k\(5%\)\s+(\d+)\s+k\(min\)\s+(\d+)", text, re.M)]
        if len(rows) != 12:
            raise ValueError(f"ftir_32's printed 2 × 2 summary changed format ({len(rows)} rows parsed, 12 expected)")
        p.add("scheme_2x2", pd.DataFrame(rows), notebook=NB32, read="executed notebook output (printed)",
              path=rel(PHASE3 / f"{NB32}.ipynb"), cell='summary.to_csv(TABLE_DIR / "cv_scheme_2x2_summary.csv"')
    verdict, source = p.read(PHASE3, NB32, "display(verdict)", csv=out + "ftir32/cv_scheme_2x2_verdict.csv", index="cohort")
    p.add("verdict_2x2", long(verdict, ["cohort"], "ratio", "value"), **source)

    # ftir_11 §6: the cohort-size cut (400 / 800 / 1600) and its random controls, site-held-out k and TOR test
    p.shown("ocec_sizes", PHASE3, NB11, "heldout_metrics.to_csv", nth=0, csv=out + "ftir11/site_held_out_tor_metrics.csv")

    # the curves behind every table above: written only by a re-run
    p.optional("rmsecv_curves", PHASE3, NB20, out + "ftir20/rmsecv_curves.csv",
               "ftir_20's RMSECV curves, k = 1–30, for six setups under interleaved 10-fold and site-grouped 5-fold CV on raw and "
               "2nd-derivative spectra (the interleaved half is in no gallery export)")
    p.optional("selection_curves", PHASE3, NB23, out + "ftir23/selection_curves.csv",
               "ftir_23's curve under each protocol with its rule threshold (the within-5 % band and the ±1 SE ribbon), six cohorts")
    p.missing.append({
        "table": "cv_2x2_curves",
        "what": "ftir_32's four curves per cohort (interleaved 10/5-fold, site-grouped 5/10-fold); the notebook saves only the summary and verdict, "
                "so its cell 7 needs `pd.concat([c.assign(cohort=n, scheme=s) for (n, s), c in curves.items()]).to_csv(TABLE_DIR / 'cv_scheme_2x2_curves.csv')` first",
        "needs": rel(PHASE3 / out / "ftir32/cv_scheme_2x2_curves.csv"),
        "run": f"cd {rel(PHASE3)} && uv run python scripts/build_notebooks.py 32",
        "after": "ftir_11's output/tables/ftir11 (lowest_ocec_800_cohort.csv) and output/corrected/improve_pool_corrected_df6.npz",
    })

    # findings, read off the same tables the charts draw
    sel = p.tables["selection"].set_index(["cohort", "mode"])
    ocec = sel.loc[("Lowest-OC/EC (800)", "app")], sel.loc[("Lowest-OC/EC (800)", "site_heldout")]
    raw = optimism[optimism.get("preprocessing", pd.Series("raw", index=optimism.index)) == "raw"].set_index("cohort")["optimism_ratio"]
    v = verdict.set_index("cohort")
    p.text["finding"] = (
        f"The same 800 lowest-OC/EC filters give k = {int(ocec[0].k)} under the Calibration app's protocol and k = {int(ocec[1].k)} "
        f"under phase 3's site-held-out protocol, and the Addis intercept moves from {ocec[0].addis_intercept:.2f} to "
        f"{ocec[1].addis_intercept:.2f} µg/m³ with it. Interleaved folds put filters from the same IMPROVE site on both sides of a split: "
        f"they flatter the smoke cohorts' error floor by ×{raw.max():.2f} and the OC/EC cohorts' by about ×{raw.loc['Lowest-OC/EC (800)']:.2f}. "
        f"ftir_32 rules out the fold count: interleaved 5-fold ≈ 10-fold (ratio {v['count effect (interleaved 5 / 10)'].min():.2f}–"
        f"{v['count effect (interleaved 5 / 10)'].max():.2f}), so the optimism comes from interleaving.")
    return p
