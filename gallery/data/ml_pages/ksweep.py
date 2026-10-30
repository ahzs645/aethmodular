"""The k sweep against IMPROVE: research/ftir_hips_chem/ksweep_20261014 (Test 1 of the 14 Oct intercept plan).

Does the Addis intercept follow the number of PLS components k, and can IMPROVE validation see it?
workflows/run_ksweep_20261014.py fits five baselines (spline; VIBES 87 blanks 4000–500 and cut to 4000–1425;
the two minus-LABE1 banks) on the standard all-IMPROVE lot-251 cohort and on four 500-analog seasonal
cohorts, 100 site-disjoint 80/20 splits each, and reads every k = 1–25 from one k = 25 fit per split.
Readouts per split and k: the Addis Deming line vs HIPS Fabs/10 (and its OLS line), and the held-out IMPROVE
OLS lines vs TOR EC and vs IMPROVE HIPS Fabs/10 in five IMPROVE-only subsets (all, high non-soil K, low
non-soil K, high OC/EC, the phase-2 smoke cohort).

Reads, in order of preference, the workflow's CSVs (output/tables/ksweep_20261014, gitignored, local) or the
tables the executed notebook displayed (k = 2, 4, 8, 11, 15, 20, 25 only, and only the baselines it displayed).
Shipped: split medians and IQRs per (baseline, cohort, set, k), the Addis OLS intercept ÷ mean reference
(medians over the 100 splits, from parts/<baseline>.csv), the pooled out-of-fold lines, the TOR-vs-HIPS
reference gap and the protocol k. Not shipped: the per-split dumps (parts/*.csv, 73 MB) and the IMPROVE
K/Fe cache. When the CSVs are present, every number is checked against the notebook's displays.

Everything is at MAC 10 (Fabs/10). Intercepts in µg/m³ do not move with the MAC; Deming and OLS slopes against
Fabs/MAC scale by MAC/10, and so does an intercept divided by the mean HIPS reference (its denominator is
mean Fabs/MAC). The page re-expresses those; R² and everything against TOR is MAC-free.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from ._common import HIPS_CHEM, page, rel
from .residual import _Grid

KEY = "ksweep"
NB = "ksweep_20261014"
T = HIPS_CHEM / "output/tables/ksweep_20261014"
RUN = ("uv run --no-sync python research/ftir_hips_chem/workflows/run_ksweep_20261014.py "
       "(about 4 min; needs the rank-70 VIBES context and the minus-LABE1 Colab runs on the Drive, vibes_current.load_context)")

# the notebook's short names (its BASE), in its legend order
BASE = {"AIRSpec": "Spline", "VIBES-full-r70": "VIBES 87 (4000–500)", "VIBES-full-cut-r70": "VIBES 87 cut 1425",
        "VIBES-lot251-x1": "VIBES 287 lot-251 −LABE1", "VIBES-allblanks-x1": "VIBES 365 −LABE1"}
LONG = {"AIRSpec": "Spline baseline", "VIBES-full-r70": "VIBES, 87 blanks (4000–500)", "VIBES-full-cut-r70": "VIBES, 87 blanks, cut to 4000–1425",
        "VIBES-lot251-x1": "VIBES, 287 lot-251 blanks minus LABE1", "VIBES-allblanks-x1": "VIBES, 365 labeled blanks minus LABE1"}
GROUPS = ["All Addis", "Dry (Oct-Feb)", "Belg (Mar-May)", "Kiremt (Jun-Sep)"]
# run_ksweep_20261014.SUBSET_LABELS, in its order
SUBSETS = {"all": "all held-out IMPROVE", "nsK_top10": "non-soil K (K − 0.6 Fe) ≥ library p90", "nsK_bottom50": "non-soil K ≤ library median",
           "ocec_top10": "TOR OC/EC ≥ library p90", "smoke_cohort": "phase-2 smoke cohort"}
KS = [2, 4, 8, 11, 15, 20, 25]
CODE = {v: k for k, v in BASE.items()}

ADDIS_COLS = ["intercept", "intercept_q25", "intercept_q75", "slope", "slope_q25", "slope_q75", "R2", "n_median"]
IMPROVE_COLS = ["R2", "R2_q25", "R2_q75", "slope", "intercept", "intercept_rel", "intercept_rel_q25", "intercept_rel_q75",
                "hips_slope", "hips_intercept", "hips_intercept_rel", "hips_intercept_rel_q25", "hips_intercept_rel_q75", "x_mean", "hips_x_mean", "n_median"]

M3, M5, M7, M9, M11 = ("tbl.rename(index=BASE, level=1)", "im[im.set.eq('all') & im.k.isin(KS)]", "display(gap.round(3))",
                       "pp = pooled[pooled.cohort.eq('standard')", "display(proto.pivot(")


def wide_to_long(frame: pd.DataFrame, ids: list[str], value: str) -> pd.DataFrame:
    """A displayed `pivot(..., columns='k')` back to one row per k."""
    out = frame.melt(id_vars=ids, var_name="k", value_name=value)
    out["k"] = out["k"].astype(int)
    if "baseline" in out:
        out["baseline"] = out["baseline"].map(CODE)
    return out


def pooled_display(nb) -> pd.DataFrame:
    """[9]'s pivot with two column levels (rel_hips, rel_tor) × k: notebook_outputs flattens it without colspans,
    so the repeated k labels collide; read the cells in order instead (residual._Grid keeps the colspans)."""
    cell = nb._cell(M9)
    html = next("".join(o["data"]["text/html"]) for o in cell["outputs"] if "text/html" in (o.get("data") or {}))
    grid = _Grid()
    grid.feed(html)
    head, *body = grid.tables[0]
    outer = [text for _, text, span in head[2:] for _ in range(span)]
    ks = [int(text) for _, text, _ in body[0][2:]]
    if outer != ["rel_hips"] * len(KS) + ["rel_tor"] * len(KS) or ks != KS * 2:
        raise ValueError(f"{NB}: the pooled table's columns changed")
    rows, outer_label = [], None
    for r in body:
        index = [text for tag, text, _ in r if tag == "th"]
        values = [float(text) for tag, text, _ in r if tag == "td"]
        if len(values) != 2 * len(KS):
            continue  # the header rows
        if len(index) == 2:
            outer_label = index[0]
        for j, k in enumerate(KS):
            rows.append({"set": outer_label, "baseline": CODE[index[-1]], "k": k, "rel_hips": values[j], "rel_tor": values[len(KS) + j]})
    return pd.DataFrame(rows)


class Shown:
    """The tables the executed notebook displayed, in long form with the workflow's column names."""

    def __init__(self, p) -> None:
        nb = p.notebook(HIPS_CHEM, NB)
        self.src = lambda cell: {"notebook": NB, "read": "executed notebook output", "path": rel(HIPS_CHEM / f"{NB}.ipynb"), "cell": cell}
        std = wide_to_long(nb.table(M3), ["set", "baseline"], "intercept").assign(cohort="standard")
        an = wide_to_long(nb.table(M11, 0), ["cohort", "baseline"], "intercept")
        an = an.assign(set=an.cohort.str.split("|").str[1], cohort="analog")
        self.addis = pd.concat([std, an], ignore_index=True)
        self.r2 = wide_to_long(nb.table(M5), ["baseline"], "R2").assign(set="all")
        self.hips_rel = wide_to_long(nb.table(M7, 0), ["set", "baseline"], "hips_intercept_rel")
        self.addis_rel = wide_to_long(nb.table(M7, 1), ["baseline"], "rel").assign(set="All Addis")
        self.gap = nb.table(M7, 2)
        self.pooled = pooled_display(nb)
        proto = nb.table(M11, 1).melt(id_vars=["baseline"], var_name="cohort", value_name="protocol_k")
        proto["baseline"] = proto["baseline"].map(CODE)
        proto["set"] = proto.cohort.str.split("|").str[1].fillna("all")
        proto["cohort"] = np.where(proto.cohort.eq("standard"), "standard", "analog")
        self.proto = proto
        self.sizes = {k: int(v) for k, v in re.findall(r"'(\w+)': (\d+)", nb.stream("T = Path('output/tables/ksweep_20261014')"))}


def from_csv(p, shown: Shown) -> list[dict]:
    """Every table from the workflow's CSVs; returns the agreement with the notebook's displays."""
    curves = pd.read_csv(T / "curves.csv")
    curves["cohort_kind"] = np.where(curves.cohort.eq("standard"), "standard", "analog")
    src = lambda f: {"notebook": NB, "read": "csv", "path": rel(T / f)}  # noqa: E731

    ad = curves[curves.readout.eq("addis")].copy()
    # an analog cohort is judged on its own season only; its 'set' is that season
    ad = ad[ad.cohort.eq("standard") | ad.cohort.eq("analog|" + ad.set)]
    addis = ad[["baseline", "cohort_kind", "set", "k", *ADDIS_COLS]].rename(columns={"cohort_kind": "cohort"})
    p.add("addis_k", addis, **src("curves.csv"))

    im = curves[curves.readout.eq("improve") & curves.cohort.eq("standard")]
    improve = im[["baseline", "set", "k", *IMPROVE_COLS]]
    p.add("improve_k", improve, **src("curves.csv"))

    # Addis OLS intercept ÷ mean Fabs/10, median and IQR over the splits (the notebook's [7] left column)
    parts = []
    for b in BASE:
        f = pd.read_csv(T / "parts" / f"{b}.csv", usecols=["baseline", "cohort", "readout", "set", "k", "repeat", "ols_intercept", "ols_slope", "x_mean"])
        f = f[f.cohort.eq("standard") & f.readout.eq("addis")]
        parts.append(f.assign(rel=f.ols_intercept / f.x_mean))
    a = pd.concat(parts, ignore_index=True)
    g = a.groupby(["baseline", "set", "k"])
    addis_rel = g.rel.median().rename("rel").to_frame().join(g.rel.quantile(0.25).rename("rel_q25")).join(g.rel.quantile(0.75).rename("rel_q75")) \
        .join(g.ols_intercept.median().rename("ols_intercept")).join(g.ols_slope.median().rename("ols_slope")).join(g.x_mean.median().rename("x_mean")).reset_index()
    p.add("addis_rel", addis_rel, notebook=NB, read="csv", path=rel(T / "parts") + "/<baseline>.csv")

    pooled = pd.read_csv(T / "pooled.csv")
    pooled = pooled[pooled.cohort.eq("standard")].assign(rel_tor=lambda d: d.intercept / d.x_mean, rel_hips=lambda d: d.hips_intercept / d.hips_x_mean)
    p.add("pooled", pooled[["baseline", "set", "k", "n", "R2", "slope", "intercept", "hips_slope", "hips_intercept", "x_mean", "hips_x_mean", "rel_tor", "rel_hips"]],
          **src("pooled.csv"))
    gap = pd.read_csv(T / "reference_gap.csv")
    p.add("reference_gap", gap, **src("reference_gap.csv"))
    proto = pd.read_csv(T / "protocol_k.csv")
    proto["set"] = proto.cohort.str.split("|").str[1].fillna("all")
    proto["cohort"] = np.where(proto.cohort.eq("standard"), "standard", "analog")
    p.add("protocol_k", proto, **src("protocol_k.csv"))
    summary = json.loads((T / "summary.json").read_text())
    p.add("subsets", pd.DataFrame([{"set": s, "label": lab, "n": summary["subset_sizes"][s]} for s, lab in SUBSETS.items()]), **src("summary.json"))
    p.text["run"] = (f"Workflow run: {summary['n_repeats']} site-disjoint splits per cohort, k = 1–{summary['k_max']}, {summary['addis_n']} Addis filters, "
                     f"{summary['improve_n']:,} IMPROVE lot-251 filters, Addis Deming λ = {summary['lambda_addis']:.2f} at MAC {summary['mac']}, "
                     f"non-soil K known for {100 * summary['nsK_coverage']:.1f} % of the library.")

    # the same numbers as the notebook displayed them (rounded to 2 dp; the gap to 3)
    def agree(name, mine, theirs, keys, col, dp=2):
        m = mine.merge(theirs[[*keys, col]], on=keys, suffixes=("", "_nb"))
        d = (m[col].round(dp) - m[f"{col}_nb"]).abs()
        return {"table": name, "compared": int(len(m)), "displayed": int(theirs[col].notna().sum()), "max_abs_diff": float(d.max()), "decimals": dp}

    checks = [
        agree("addis_k · intercept (standard, [3])", addis[addis.cohort.eq("standard")], shown.addis[shown.addis.cohort.eq("standard")], ["baseline", "set", "k"], "intercept"),
        agree("addis_k · intercept (analogs, [11])", addis[addis.cohort.eq("analog")], shown.addis[shown.addis.cohort.eq("analog")], ["baseline", "set", "k"], "intercept"),
        agree("improve_k · R² vs TOR ([5])", improve, shown.r2, ["baseline", "set", "k"], "R2"),
        agree("improve_k · intercept ÷ mean HIPS/10 ([7])", improve, shown.hips_rel, ["baseline", "set", "k"], "hips_intercept_rel"),
        agree("addis_rel · OLS intercept ÷ mean Fabs/10 ([7])", addis_rel, shown.addis_rel, ["baseline", "set", "k"], "rel"),
        agree("pooled · ÷ mean HIPS/10 ([9])", p.tables["pooled"], shown.pooled, ["baseline", "set", "k"], "rel_hips"),
        agree("pooled · ÷ mean TOR ([9])", p.tables["pooled"], shown.pooled, ["baseline", "set", "k"], "rel_tor"),
        agree("reference_gap · intercept ([7])", gap, shown.gap, ["set"], "intercept", dp=3),
        agree("protocol_k ([11])", proto, shown.proto, ["baseline", "cohort", "set"], "protocol_k", dp=0),
    ]
    return checks


def from_notebook(p, shown: Shown) -> None:
    """The displayed tables only: k = 2…25 at seven values, intercepts and R² without IQRs."""
    p.add("addis_k", shown.addis[["baseline", "cohort", "set", "k", "intercept"]], **shown.src(f"{M3}; {M11}"))
    improve = shown.r2.merge(shown.hips_rel, on=["baseline", "set", "k"], how="outer")
    p.add("improve_k", improve, **shown.src(f"{M5}; {M7}"))
    p.add("addis_rel", shown.addis_rel, **shown.src(M7))
    p.add("pooled", shown.pooled, **shown.src(M9))
    p.add("reference_gap", shown.gap, **shown.src(M7))
    p.add("protocol_k", shown.proto, **shown.src(M11))
    p.add("subsets", pd.DataFrame([{"set": s, "label": lab, "n": shown.sizes.get(s)} for s, lab in SUBSETS.items()]),
          **shown.src("T = Path('output/tables/ksweep_20261014')"))
    needs = rel(T)
    p.missing += [
        {"table": "addis_k", "what": "the Addis intercept and Deming slope at every k = 1–25 with the IQR over the 100 splits (the notebook displayed seven k and no IQR)",
         "needs": f"{needs}/curves.csv", "run": RUN, "after": None},
        {"table": "improve_k", "what": "the held-out IMPROVE R², slopes and relative intercepts (vs TOR and vs HIPS) for every subset and baseline at every k",
         "needs": f"{needs}/curves.csv", "run": RUN, "after": None},
        {"table": "addis_rel", "what": "the Addis OLS intercept ÷ mean reference for each season and its IQR over the splits",
         "needs": f"{needs}/parts/<baseline>.csv", "run": RUN, "after": None},
        {"table": "pooled", "what": "the pooled out-of-fold lines for every baseline at every k", "needs": f"{needs}/pooled.csv", "run": RUN, "after": None},
    ]


def build():
    p = page(KEY, "k sweep vs IMPROVE")
    p.add("baselines", pd.DataFrame([{"baseline": b, "label": BASE[b], "long": LONG[b], "order": i} for i, b in enumerate(BASE)]),
          notebook=NB, read="workflow labels", path=rel(HIPS_CHEM / "workflows/run_ksweep_20261014.py"))
    shown = Shown(p)
    if (T / "curves.csv").exists() and all((T / "parts" / f"{b}.csv").exists() for b in BASE):
        checks = from_csv(p, shown)
        p.add("checks", pd.DataFrame(checks), notebook=NB, read="csv vs executed notebook output", path=rel(HIPS_CHEM / f"{NB}.ipynb"))
        worst = max(c["max_abs_diff"] for c in checks)
        p.text["check"] = (f"Checked against the executed notebook's displays: {sum(c['compared'] for c in checks)} displayed numbers in "
                           f"{len(checks)} tables, " + ("every one equal at the notebook's decimals." if worst == 0 else
                                                        f"largest difference after rounding to the notebook's decimals {worst:.3g}."))
    else:
        from_notebook(p, shown)

    a = p.tables["addis_k"].set_index(["cohort", "set", "baseline", "k"]).intercept
    r2 = p.tables["improve_k"].set_index(["set", "baseline", "k"]).R2
    gap = p.tables["reference_gap"].set_index("set")
    p.text["finding"] = (
        f"On the standard all-IMPROVE model the All-Addis intercept falls with k for every baseline: from k = 2 to k = 20 spline goes "
        f"{a['standard', 'All Addis', 'AIRSpec', 2]:+.2f} → {a['standard', 'All Addis', 'AIRSpec', 20]:+.2f} µg/m³ and VIBES 87 "
        f"{a['standard', 'All Addis', 'VIBES-full-r70', 2]:+.2f} → {a['standard', 'All Addis', 'VIBES-full-r70', 20]:+.2f}, the minus-LABE1 banks like VIBES 87. "
        f"Held-out IMPROVE R² vs TOR plateaus by k ≈ 8–11 (spline {r2['all', 'AIRSpec', 8]:.2f} at k = 8, {r2['all', 'AIRSpec', 25]:.2f} at k = 25), "
        f"so the standard validation cannot tell k = 11 from k = 25. Smoke-like IMPROVE filters go to about −0.3 of their mean HIPS/10 and level off by "
        f"k ≈ 7, for spline and VIBES alike. Their own TOR-vs-HIPS line (no FTIR) crosses lower still, at {gap.loc['ocec_top10', 'intercept'] / gap.loc['ocec_top10', 'x_mean']:+.2f} "
        f"(high OC/EC) and {gap.loc['smoke_cohort', 'intercept'] / gap.loc['smoke_cohort', 'x_mean']:+.2f} (smoke cohort) of the same mean, because TOR reads low against "
        f"HIPS in smoke (median Fabs/TOR EC {gap.loc['smoke_cohort', 'median_mac']:.1f} against {gap.loc['all', 'median_mac']:.1f} network-wide): the IMPROVE "
        f"smoke offset is a reference gap the models partly inherit. The further VIBES drift at Addis has no IMPROVE counterpart.")
    p.text = {k: p.text[k] for k in ("finding", "run", "check") if k in p.text}  # the headline first
    return p
