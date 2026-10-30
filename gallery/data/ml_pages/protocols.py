"""Protocols end to end, transfer and bootstrap uncertainty: ftir_21, ftir_22, ftir_15, ftir_11, ftir_13, ftir_17, ftir_18
(ftir_31 §1.1, 1.2, 1.6, 1.7, 2.2 and 2.6 redraw ftir_21/22 from their committed tables).

Every calibration run twice, under the Calibration app's protocol (interleaved 10-fold CV, first k within 5 %
of the minimum, fit on every cohort filter) and phase 3's site-held-out protocol (site-grouped 5-fold, first
major minimum, fit on the training side of a site-disjoint split); the site-cluster bootstrap of the Addis
fit under each; the residual drivers; and every training set tried, on the same Addis filters.

The executed notebooks display these tables at MAC 10 on the fixed 190-filter Addis cohort (ftir_18: on the
239 available pairs). Two of them are pandas pivots whose two-row header the shared reader cannot split, so
they are parsed here with their column spans. The per-filter predictions behind the crossplots, the bootstrap
draws and the per-filter D² are written only to the gitignored output/tables, so they are listed as missing;
the one tracked per-filter file (ftir_13's AIRSpec predictions, docs/openresearch-execution/airspec/expected)
is exported as it is. The rule k, intercept and slope under all three explorer protocols (calibration.json
grid) and the per-filter readouts and HIPS of calibration_runs.json are read by the page directly.
"""
from __future__ import annotations

from html.parser import HTMLParser

import pandas as pd

from ml_pages._common import PHASE3, ROOT, page, rel

KEY = "protocols"

NB11 = "ftir_11_ocec_ratio_cohort"
NB13 = "ftir_13_airspec_corrected_calibrations"
NB15 = "ftir_15_uncertainty_and_hybrid"
NB17 = "ftir_17_deck_figures_and_seasonal_spectra"
NB18 = "ftir_18_transfer_roundup"
NB21 = "ftir_21_both_protocols_end_to_end"
NB22 = "ftir_22_figures_under_both_protocols"
AIRSPEC_TRACKED = ROOT / "docs/openresearch-execution/airspec/expected/addis_predictions_corrected.csv"
OUT = "output/tables/"


class _Spans(HTMLParser):
    """Every <table> as rows of (tag, text, colspan): enough to rebuild a pandas pivot's two-row header."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[dict[str, list]] = []
        self._section, self._row, self._cell = "body", None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append({"head": [], "body": []})
        elif tag in ("thead", "tbody"):
            self._section = "head" if tag == "thead" else "body"
        elif tag == "tr":
            self._row = []
        elif tag in ("th", "td") and self._row is not None:
            self._cell = (tag, [], int(dict(attrs).get("colspan") or 1))

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self._cell is not None:
            self._row.append((self._cell[0], "".join(self._cell[1]).strip(), self._cell[2]))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.tables[-1][self._section].append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell[1].append(data)


def _number(text: str):
    try:
        return float(text)
    except ValueError:
        return None if text in ("", "NaN", "nan") else text


def pivot_shown(p, notebook: str, marker: str, nth: int = 0) -> tuple[pd.DataFrame, dict]:
    """A displayed `df.pivot(index=…, columns=…, values=[…])`: one row per (index, column value), one column per value.

    pandas writes the value names on the first header row (each spanning the pivoted columns), the pivoted
    column values on the second (its first cell is the columns' name) and the index name on the third.
    """
    cell = p.notebook(PHASE3, notebook)._cell(marker)
    html = ["".join(o["data"]["text/html"]) for o in cell.get("outputs", []) if "text/html" in (o.get("data") or {})]
    parser = _Spans()
    for fragment in html:
        parser.feed(fragment)
    table = parser.tables[nth]
    top, middle, bottom = table["head"]
    values = [text for _, text, span in top[1:] for _ in range(span)]
    pivoted = [text for _, text, _ in middle[1:]]
    column_name, index_name = middle[0][1], bottom[0][1]
    if len(values) != len(pivoted) or "..." in values:
        raise ValueError(f"{notebook}: the pivot under {marker!r} does not have a two-row value × column header")
    records = []
    for row in table["body"]:
        label, cells = row[0][1], [text for tag, text, _ in row[1:]]
        for level in dict.fromkeys(pivoted):
            records.append({index_name: label, column_name: level,
                            **{v: _number(c) for v, c, lv in zip(values, cells, pivoted) if lv == level}})
    return pd.DataFrame(records), {"notebook": notebook, "read": "executed notebook output",
                                    "path": rel(PHASE3 / f"{notebook}.ipynb"), "cell": marker}


def shown_or_csv(p, name: str, notebook: str, marker: str, csv: str, keep, nth: int = 0, columns: list[str] | None = None) -> pd.DataFrame:
    """A table the notebook displayed as a subset of the CSV it wrote: the CSV, cut to the same subset, when it exists."""
    frame, source = p.read(PHASE3, notebook, marker, nth=nth, csv=csv)
    if source["read"] == "csv":
        frame = frame[keep(frame)]
        if columns:
            frame = frame[[c for c in columns if c in frame.columns]]
    return p.add(name, frame, **source)


def build():
    p = page(KEY, "Protocols end to end, transfer and bootstrap uncertainty")

    # ---- ftir_21: every setup under both protocols (fixed 190-filter cohort, MAC 10)
    metrics_csv = PHASE3 / OUT / "ftir21/addis_metrics_by_mode.csv"
    if metrics_csv.exists():
        m = pd.read_csv(metrics_csv)
        m = m[(m["MAC_m2_g"] == 10) & (m["evaluation_set"] == "fixed phase-2 cohort")]
        p.add("headline", m[["cohort", "mode", "k", "slope", "intercept", "R2", "RMSE"]], notebook=NB21, read="csv", path=rel(metrics_csv))
    else:
        headline, source = pivot_shown(p, NB21, "headline_by_mode.csv")
        p.add("headline", headline, **source)
    shown_or_csv(p, "tor_test", NB21, "display(summary[['cohort', 'mode', 'k', 'n_train', 'pct_rmsecv_floor'", OUT + "ftir21/calibration_summary_by_mode.csv",
                 lambda f: f.index == f.index, nth=1, columns=["cohort", "mode", "k", "n_train", "pct_rmsecv_floor", "heldout_TOR_R2", "heldout_TOR_slope"])

    # ---- ftir_22: the ftir_11–19 figures re-derived under both protocols
    order_csv = PHASE3 / OUT / "ftir22/row_order_sensitivity.csv"
    if order_csv.exists():
        p.add("row_order", pd.read_csv(order_csv), notebook=NB22, read="csv", path=rel(order_csv))
    else:
        order, source = pivot_shown(p, NB22, "row_order_sensitivity.csv")
        p.add("row_order", order, **source)
    shown_or_csv(p, "boot_modes", NB22, "bootstrap_ci_by_mode.csv", OUT + "ftir22/bootstrap_ci_by_mode.csv", lambda f: f["metric"] == "intercept")
    p.shown("drivers_modes", PHASE3, NB22, "residual_drivers_by_mode.csv", csv=OUT + "ftir22/residual_drivers_by_mode.csv")
    p.shown("size_sweep", PHASE3, NB22, "cohort_size_sweep_by_mode.csv", csv=OUT + "ftir22/cohort_size_sweep_by_mode.csv")
    p.shown("figure_index", PHASE3, NB22, "figure_update_index.csv", csv=OUT + "ftir22/figure_update_index.csv")

    # ---- ftir_15: the locked models' bootstrap, residual drivers by season, and the hybrid cohort
    p.shown("boot_locked", PHASE3, NB15, "addis_bootstrap_ci.csv", csv=OUT + "ftir15/addis_bootstrap_ci.csv")
    p.shown("drivers_locked", PHASE3, NB15, "addis_residual_drivers.csv", nth=0, csv=OUT + "ftir15/addis_residual_drivers.csv")
    p.shown("season_residuals", PHASE3, NB15, "addis_residual_drivers.csv", nth=1, csv=OUT + "ftir15/addis_residuals_by_season.csv")
    p.shown("hybrid", PHASE3, NB15, "hybrid_vs_ocec_metrics.csv", csv=OUT + "ftir15/hybrid_vs_ocec_metrics.csv")

    # ---- training-set choices, each on the fixed 190 filters at MAC 10 (ftir_11, ftir_13, ftir_17) or the 239 pairs (ftir_18)
    fixed10 = lambda f: (f["MAC_m2_g"] == 10) & (f["cohort"] == "fixed phase-2 cohort")  # noqa: E731
    shown_or_csv(p, "sizes_addis", NB11, "heldout_metrics.to_csv", OUT + "ftir11/addis_metrics.csv", fixed10, nth=1,
                 columns=["model", "n", "slope", "intercept", "R2", "RMSE", "bias"])
    shown_or_csv(p, "airspec_addis", NB13, "addis_metrics_corrected.csv", OUT + "ftir13/addis_metrics_corrected.csv",
                 lambda f: f["MAC_m2_g"] == 10, columns=["model", "n", "slope", "intercept", "R2", "RMSE"])
    p.shown("hips_transfer", PHASE3, NB13, "hips_transfer_corrected.csv", csv=OUT + "ftir13/hips_transfer_corrected.csv")
    p.shown("fullpool", PHASE3, NB17, "no_cleaning_fullpool_metrics.csv", csv=OUT + "ftir17/no_cleaning_fullpool_metrics.csv")
    shown_or_csv(p, "roundup", NB18, "transfer_roundup_metrics.csv", OUT + "ftir18/transfer_roundup_metrics.csv",
                 lambda f: (f["MAC_m2_g"] == 10) & (f["cohort"] == "available pairs"),
                 columns=["model", "kind", "n", "slope", "intercept", "R2", "RMSE", "MAE", "bias", "median_bias"])

    # ftir_13 [17]: the tracked per-filter AIRSpec predictions (239 Addis filters, HIPS Fabs alongside)
    airspec = pd.read_csv(AIRSPEC_TRACKED).drop(columns=["MediaId"])
    p.add("airspec_filters", airspec, notebook=NB13, read="tracked csv", path=rel(AIRSPEC_TRACKED))

    # ---- per-filter tables only a re-run writes
    p.optional("predictions_by_mode", PHASE3, NB21, OUT + "ftir21/addis_predictions_by_mode.csv",
               "ftir_21's per-filter Addis predictions for the six setups under both protocols (the app-protocol clouds of its crossplots, ftir_31 §1.2)")
    p.optional("residuals_by_mode", PHASE3, NB22, OUT + "ftir22/addis_residuals_by_mode.csv",
               "ftir_22's per-filter Addis residual, D² and season under both protocols (residual vs D², ftir_22 [13] and ftir_31 §1.3)")
    p.optional("boot_draws_modes", PHASE3, NB22, OUT + "ftir22/bootstrap_draws_by_mode.csv",
               "ftir_22's 200 site-cluster bootstrap draws per protocol (the distributions behind the intervals)")
    p.optional("boot_draws_locked", PHASE3, NB15, OUT + "ftir15/addis_bootstrap_draws.csv",
               "ftir_15's 200 bootstrap draws of intercept, slope and RMSE for the two locked models")
    p.optional("roundup_filters", PHASE3, NB18, OUT + "ftir18/addis_roundup_predictions.csv",
               "ftir_18's per-filter predictions for the seven training choices (the HIPS-τ transfer and Addis-only nested CV are in no gallery export)")
    p.optional("sizes_filters", PHASE3, NB11, OUT + "ftir11/addis_predictions.csv",
               "ftir_11's per-filter Addis predictions for the OC/EC 400 and 1600 cohorts (800 is the gallery's ocec800_raw preset)")
    p.missing.append({
        "table": "hybrid_filters",
        "what": "ftir_15's hybrid-cohort (OC/EC ∩ spectral similarity) per-filter predictions: never saved, so cell 9 needs a to_csv of them before a re-run",
        "needs": rel(PHASE3 / OUT / "ftir15/hybrid_addis_predictions.csv"),
        "run": f"cd {rel(PHASE3)} && uv run python scripts/build_notebooks.py 15",
        "after": "ftir_11 and ftir_13 outputs (the OC/EC cohort and the AIRSpec-corrected caches)",
    })

    # ---- findings, from the same tables
    h = p.tables["headline"].set_index(["cohort", "mode"])
    swing = (h.xs("app", level="mode")["intercept"] - h.xs("site_heldout", level="mode")["intercept"]).abs()
    boot = p.tables["boot_modes"].set_index(["spectra", "mode"])
    air = boot.loc[("AIRSpec df1=6", "site_heldout")], boot.loc[("AIRSpec df1=6", "app")]
    drivers = p.tables["drivers_modes"]
    raw_r = drivers[drivers["spectra"] == "raw"]["pearson_r_residual_vs_D2"]
    air_r = drivers[drivers["spectra"] != "raw"]["pearson_r_residual_vs_D2"]
    p.text["finding"] = (
        f"Run end to end under both protocols, the six setups' Addis intercepts move by up to {swing.max():.1f} µg/m³ "
        f"({swing.idxmax()}) and by {swing.loc['Lowest-OC/EC + AIRSpec (800)']:.2f} for the AIRSpec-corrected OC/EC cohort: the protocol is a "
        f"lever on raw spectra, not on corrected ones. The corrected intercept's site-cluster bootstrap interval is "
        f"[{air[0].ci_low:.2f}, {air[0].ci_high:.2f}] site-held-out and [{air[1].ci_low:.2f}, {air[1].ci_high:.2f}] under the app protocol; "
        f"raw-model residuals track the score-space distance D² (r {raw_r.min():.2f}–{raw_r.max():.2f}) under either protocol, "
        f"corrected-model residuals do not (r ≤ {air_r.max():.2f}).")
    return p
