"""Residual learners and null controls: ftir_43, ftir_45, ftir_46 (and ftir_53 §5, which re-draws ftir_46's mass term).

What a learner on the FTIR − HIPS/10 residual finds at Addis, against a proportional-disagreement null
(simulated HIPS with no spectral information), against a metadata baseline on the same time blocks (the
paired increment), and what gravimetric mass adds. Every number is the executed notebook's (MAC 10): the
residual learners were fitted on residuals at Fabs/10, so their R² and RMSE do not move with the reader's
MAC. The page re-expresses only what is exact (mass slopes against Fabs/MAC scale by 10/MAC).

The null draws, the residual learner's coefficient spectrum and the per-filter char_06 classes are written
only by a re-run (listed under `missing`).
"""
from __future__ import annotations

from html.parser import HTMLParser

import pandas as pd

from ._common import PHASE3, CHAR, number, page, rel

KEY = "residual"
NB43 = "ftir_43_residual_learner_null_control"
NB45 = "ftir_45_residual_increment_attribution"
NB46 = "ftir_46_paired_increment_and_mass"
NB53 = "ftir_53_committee_backup_figures"

# ladder rungs in the notebook's order (its figure draws them bottom-up in this order)
RUNGS = ["0  no correction", "a  constant", "b  f(X)", "c0 f(X)+season+volume (ftir_43 M2)", "c  c0+lot",
         "c+C c+mass µg/m³", "c+L c+mass µg/filter", "e  spectra only (ftir_43 M3)", "d  c+spectra (stacked)",
         "d+C c+C+spectra (stacked)", "d+L c+L+spectra (stacked)"]


class _Grid(HTMLParser):
    """Every <table> as rows of (tag, text, colspan): enough to read a pandas pivot's multi-row header,
    which notebook_outputs flattens without colspans (its two 'blocked' columns collide)."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[tuple[str, str, int]]]] = []
        self._row = self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append([])
        elif tag == "tr":
            self._row = []
        elif tag in ("th", "td") and self._row is not None:
            self._cell = [tag, "", int(dict(attrs).get("colspan", 1) or 1)]

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self._cell is not None:
            self._row.append((self._cell[0], self._cell[1].strip(), self._cell[2]))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.tables[-1].append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell[1] += data


def pivot_long(html: str, index: str, inner: str) -> pd.DataFrame:
    """A displayed `frame.pivot(index=…, columns=inner, values=[…])` back to long form: one row per
    (index, inner) with one column per value."""
    grid = _Grid()
    grid.feed(html)
    rows = grid.tables[0]
    outer: list[str] = []
    for tag, text, span in rows[0][1:]:
        outer += [text] * span
    inners = [text for _, text, _ in rows[1][1:]]
    body = [r for r in rows if r and r[0][0] == "th" and any(t == "td" for t, _, _ in r)]
    records: dict[tuple[str, str], dict] = {}
    for r in body:
        label = r[0][1]
        for (value, sub), (_, text, _) in zip(zip(outer, inners), r[1:]):
            rec = records.setdefault((label, sub), {index: label, inner: sub})
            rec[value] = None if text in ("NaN", "") else float(text)
    return pd.DataFrame(records.values())


def ladder(p) -> None:
    """ftir_46's ladder: the CSV is long already; the display is a pivot by scheme."""
    csv = PHASE3 / "output/tables/ftir46/ladder_summary.csv"
    if csv.exists():
        frame = pd.read_csv(csv)
        p.add("ladder_summary", frame.drop(columns=[c for c in ("k_chosen",) if c in frame]),
              notebook=NB46, read="csv", path=rel(csv))
        return
    marker = "ladder.to_csv(OUT / 'ladder_summary.csv'"
    cell = p.notebook(PHASE3, NB46)._cell(marker)
    html = next("".join(o["data"]["text/html"]) for o in cell["outputs"] if "text/html" in (o.get("data") or {}))
    frame = pivot_long(html, "model", "scheme")
    if set(frame["model"]) != set(RUNGS) or set(frame["scheme"]) != {"blocked", "forward"}:
        raise ValueError("ftir_46's displayed ladder has changed shape")
    p.add("ladder_summary", frame, notebook=NB46, read="executed notebook output", path=rel(PHASE3 / f"{NB46}.ipynb"), cell=marker)


def build():
    p = page(KEY, "Residual learners and null controls")
    out43, out45, out46 = (f"output/tables/ftir{n}/" for n in (43, 45, 46))

    # ftir_43: the five learners on time-blocked folds, and the proportional-disagreement null
    models = p.shown("model_comparison", PHASE3, NB43, "results.to_csv(OUT / 'model_comparison.csv'", csv=out43 + "model_comparison.csv")
    null43 = p.shown("null_distribution_summary", PHASE3, NB43, "null_summary.to_csv", csv=out43 + "null_distribution_summary.csv")
    rel43 = p.stream(PHASE3, NB43, "etad_eval, _X_raw, _wn = load_addis_evaluation()")
    a, c, sigma = (number(r"b = ([\d.]+)·f\(X\)", rel43), number(r"f\(X\) \+ ([\d.]+) \+ ε", rel43), number(r"σ_ε = ([\d.]+)", rel43))
    p.add("null_relation", pd.DataFrame([{"a": a, "c": c, "sigma": sigma, "n": number(r"time-blockable\): (\d+)", rel43, int)}]),
          notebook=NB43, read="executed notebook output (printed)", path=rel(PHASE3 / f"{NB43}.ipynb"), cell="load_addis_evaluation()")
    p.missing.append({"table": "null_draws_43", "what": "ftir_43's 200 null refits per pipeline (its violin; the notebook keeps them in memory and writes only the median and q95)",
                      "needs": rel(PHASE3 / out43 / "null_distribution_draws.csv") + " (add null_r2_m1/null_r2_m3 .to_csv to cell [7] first)",
                      "run": f"cd {rel(PHASE3)} && uv run python scripts/build_notebooks.py 43", "after": None})

    # ftir_45: stepwise metadata models, residual by char_06 class, the learner's coefficient spectrum
    p.shown("stepwise_attribution", PHASE3, NB45, "results.to_csv(OUT / 'stepwise_attribution.csv'", csv=out45 + "stepwise_attribution.csv")
    p.shown("residual_by_char06_class", PHASE3, NB45, "class_summary.to_csv", csv=out45 + "residual_by_char06_class.csv", index="class")
    p.optional("residual_learner_coefficients", PHASE3, NB45, out45 + "residual_learner_coefficients.csv",
               "the residual learner's PLS coefficient spectrum (ftir_45 [9c])")
    p.optional("char06_classes", CHAR, "char_06_dry_season_anomaly", "output/tables/char06/per_filter_classification.csv",
               "each Addis filter's char_06 class, for ftir_45's residual histograms by class [9b]")

    # ftir_46: the paired ladder, its differences, the null, and mass
    ladder(p)
    paired = p.shown("paired_differences", PHASE3, NB46, "paired_diffs.to_csv", csv=out46 + "paired_differences.csv")
    null46 = p.shown("null_increment_summary", PHASE3, NB46, "null_summary.to_csv(OUT / 'null_increment_summary.csv'", csv=out46 + "null_increment_summary.csv")
    p.optional("null_increment_draws", PHASE3, NB46, out46 + "null_increment_draws.csv",
               "ftir_46's 200 proportional-disagreement null draws of every rung and increment (a distribution behind each null interval)")
    partial = p.shown("addis_residual_vs_mass_partial_slopes", PHASE3, NB46, "partial.to_csv", csv=out46 + "addis_residual_vs_mass_partial_slopes.csv")
    p.optional("explorer_predictions", PHASE3, NB46, out46 + "explorer_predictions.csv",
               "ftir_46's per-filter f(X), Fabs, volume and gravimetric mass at the five sites (its residual-vs-mass scatter and grid [13], [19b]); "
               "the gallery's filters.json has PM2.5 mass for only 188 of the 239 Addis filters and none at Bishoftu, so the scatter is not rebuilt live")
    p.shown("cross_site_coverage", PHASE3, NB46, "coverage.to_csv", csv=out46 + "cross_site_coverage.csv")
    framings = []
    for nth, framing in ((0, "per volume"), (1, "per deposit")):
        frame, source = p.read(PHASE3, NB46, "site_fits.to_csv", nth=nth)
        framings.append(frame.drop(columns=[c for c in ("index",) if c in frame]).assign(framing=framing))
    csv = PHASE3 / out46 / "cross_site_mass_slopes.csv"
    if csv.exists():
        p.add("cross_site_mass_slopes", pd.read_csv(csv), notebook=NB46, read="csv", path=rel(csv))
    else:
        p.add("cross_site_mass_slopes", pd.concat(framings), **source)
    p.shown("mass_term_ftir53", PHASE3, NB53, "mass_term_by_site.csv", csv="output/tables/ftir53/mass_term_by_site.csv")

    m = models.set_index("model")
    n43 = null43.set_index("pipeline")
    pdiff = paired.set_index(["scheme", "contrast"])
    base = pdiff.loc[("blocked", "(d)−(c): spectra beyond metadata")]
    plus = pdiff.loc[("blocked", "(d+)−(c+): spectra beyond metadata+mass")]
    nd = null46.set_index("stat").loc["dRMSE_d_c"]
    vol = partial.set_index(["framing", "conditioning"]).loc[("per volume (µg/m³)", "f(X) + season")]
    p.text["finding"] = (
        f"On the 239 Addis filters (time-blocked folds) the HIPS/10 − f(X) relation is b = {a:.3f}·f(X) + {c:.3f}, so the residual "
        f"carries {1 - a:.3f}·f(X) by construction. A learner given only f(X) reaches residual R² "
        f"{m.loc['M1 f(X) only', 'residual_R2']:.3f}, inside the no-information null (median {n43.loc['M1 f(X) only', 'null_R2_median']:.3f}, "
        f"p = {n43.loc['M1 f(X) only', 'p_null_ge_observed']:.2f}); the spectra learner reaches {m.loc['M3 spectra (residual learner)', 'residual_R2']:.3f}, "
        f"above the null's q95 {n43.loc['M3 spectra', 'null_R2_q95']:.3f} (ftir_43). Paired against f(X) + season + volume + lot on the same blocks, "
        f"the spectra gain ΔRMSE {base.pooled_dRMSE:+.3f} µg/m³ [{base.block_boot_ci_lo:+.3f}, {base.block_boot_ci_hi:+.3f}] is not significant "
        f"(it does beat the null: q95 {nd.null_q95:+.3f}); with gravimetric mass in the base it becomes {plus.pooled_dRMSE:+.3f} "
        f"[{plus.block_boot_ci_lo:+.3f}, {plus.block_boot_ci_hi:+.3f}] (ftir_46). No named feature carries the increment (ftir_45). "
        f"At fixed f(X) and season the residual falls {vol.slope:+.3f} µg/m³ per µg/m³ of mass [{vol.ci_lo:+.3f}, {vol.ci_hi:+.3f}]: "
        "the lead is loading, and the mass term is two to three times larger at the Ethiopian sites than elsewhere.")
    return p
