"""Addis against the IMPROVE network in loading space: the summaries behind the IMPROVE HIPS-offset notebooks.

Run where the IMPROVE FED export is available (it is not in the repo):
    uv run python gallery/data/export_research_improve_envelope.py
The cleaned IMPROVE cache is research/ftir_hips_chem/output/improve_high_fabs_comparison/improve_valid_cleaned.csv;
improve_io.load_improve_clean() builds it from the FED Query Wizard exports when it is missing (set
AETHMODULAR_IMPROVE_DIR, or the Drive mount data_paths.IMPROVE_SUBDIR resolves). The SPARTAN side is read from
the repo's research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl.

~380,000 IMPROVE filters cannot be shipped to a browser, so this writes what the notebooks plot, already summarised:

  bounds       the Addis (ETAD) p05/p95/min/max of fAbs, FTIR EC mass on the filter and EC surface loading,
               after the exclusion registry (improve_first_order_loading_range_analysis, warren_cena_improve_prep_analysis)
  screens      how many post-2003 IMPROVE filters (and sites) survive each bound, alone and in sequence
               (improve_first_order_loading_range_analysis, improve_spartan_may_full_analysis §4)
  density      2-D counts of post-2003 IMPROVE filters in (EC mass µg, fAbs) and (EC µg/cm², fAbs), linear axes from 0
               (anne_spartan_improve_ec_mass_fabs, next_tests_2026_04/00_fed_addis_loading_envelope)
  matched      ETAD against IMPROVE rows matched on EC mass alone and on EC concentration alone: quantiles of fAbs,
               fAbs / EC, and OLS fAbs ~ EC (etad_vs_improve_ec_matched_comparisons)
  sweep        rows with fAbs >= a threshold: n, sites, OLS slope and intercept (improve_hips_offset_narrative §5)
  intercepts   the fAbs-at-EC-0 intercept over several fit ranges, a FED analogue of ftir_26 (which used the Davis
               HIPS x TOR bridge, not in the repo)
  smoke        OC/EC >= 5 among the Addis-comparable and the fAbs >= 70 rows (improve_smoke_event_qc)
  rt           FED 635 nm reflectance/transmittance pairs binned, with median fAbs, the initial-reading line, and
               the minimum readings' deficit from it by fAbs (improve_fed_rt_proxy_figure2,
               improve_white_style_rt_calibration_space)
  mass_space   network quantiles of EC, OC, iron and PM2.5 mass on the filter and of fAbs x V / A
               (spartan_vs_improve_mass_comparison)

Writes gallery/app/public/data/research/improve_envelope/envelope.json (private: the research section's data).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = ROOT / "gallery/app/public/data/research/improve_envelope"
STABLE_YEAR = 2003            # calibrated HIPS fAbs "since 2003" (improve_first_order_loading_range_analysis)
OCEC_ADDIS_LIKE = 2.27        # research/ftir_ec_phase3 (OCEC_ADDIS_LIKE in run_ftir_26/30)
OCEC_SMOKE = 5.0              # improve_smoke_event_qc (OC_EC_SMOKE_MIN)
HIGH_FABS = 70.0              # research/improve_hips_offset (HIGH_TAIL_THRESHOLD)
SWEEP = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
RT_PAIRS = {"initial": ("RefI_635_Val", "TransI_635_Val"), "final": ("RefF_635_Val", "TransF_635_Val"),
            "minimum": ("RefM_635_Val", "TransM_635_Val")}
IMPROVE_COLS = ["SiteCode", "Date", "ECf_Val", "OCf_Val", "fAbs_Val", "FEf_Val", "MF_Val", "volume_m3", "MF_volume_m3",
                "EC_loading_ug", "EC_loading_ug_cm2_primary", "MF_loading_ug", *[c for p in RT_PAIRS.values() for c in p]]


def r(v, digits=4):
    """A JSON number: finite floats rounded to `digits` significant figures, else None."""
    if v is None:
        return None
    v = float(v)
    return float(f"{v:.{digits}g}") if np.isfinite(v) else None


def ols(x: pd.Series, y: pd.Series) -> dict:
    """OLS of y on x: n, slope, intercept, R² (None when fewer than 3 points or no spread)."""
    d = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3 or d.iloc[:, 0].nunique() < 2:
        return {"n": len(d), "slope": None, "intercept": None, "r2": None}
    slope, intercept = np.polyfit(d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float), 1)
    corr = np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1]
    return {"n": len(d), "slope": r(slope), "intercept": r(intercept), "r2": r(corr ** 2)}


def quantiles(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {"n": len(s), **{f"p{int(q * 100):02d}": (r(s.quantile(q)) if len(s) else None) for q in QUANTILES}}


# ---------------------------------------------------------------- SPARTAN / Addis side
def spartan_reference() -> pd.DataFrame:
    """Focus-site PM2.5 filters with fAbs, FTIR EC (µg/m³), EC mass (µg) and EC surface loading (µg/cm²), after
    the exclusion registry; the same rows export_research_hips_optics.py writes, joined to FTIR EC concentration."""
    from config import FILTER_DATA_PATH, SITES
    from export_research_hips_optics import base_id, build
    from outliers import apply_exclusion_flags

    long = pd.read_pickle(FILTER_DATA_PATH)
    optics = build(long)
    optics = optics[optics["type"].eq("PM2.5")].copy()
    ec = long[long["Parameter"].eq("EC_ftir")].assign(id=lambda d: d["FilterId"].map(base_id))
    optics = optics.merge(ec.groupby("id")["Concentration"].first().rename("ec_conc"), left_on="id", right_index=True, how="left")
    optics = optics.merge(long[long["Parameter"].eq("OC_ftir")].assign(id=lambda d: d["FilterId"].map(base_id))
                          .groupby("id")["Concentration"].first().rename("oc_conc"), left_on="id", right_index=True, how="left")
    name_of = {cfg["code"]: name for name, cfg in SITES.items()}
    parts = []
    for code, g in optics.groupby("code"):
        g = g.assign(date=pd.to_datetime(g["date"]), filter_id=g["filter_id"])
        parts.append(apply_exclusion_flags(g, name_of.get(code, code)))
    out = pd.concat(parts, ignore_index=True)
    out = out[~out["is_excluded"] & (out["fabs"] > 0) & (out["ec_conc"] > 0)].copy()
    out["ec_ug_cm2"] = out["ec_ug"] / out["area_cm2"]
    return out


def etad_bounds(spartan: pd.DataFrame) -> dict:
    """p05/p95 (the screen) and min/max (the audit) of the three Addis axes."""
    etad = spartan[spartan["code"].eq("ETAD")]
    out = {}
    for key, col in {"fabs": "fabs", "ec_ug": "ec_ug", "ec_ug_cm2": "ec_ug_cm2", "ec_conc": "ec_conc"}.items():
        s = etad[col].dropna()
        out[key] = {"n": len(s), "p05": r(s.quantile(0.05)), "p95": r(s.quantile(0.95)), "min": r(s.min()), "max": r(s.max())}
    return out


# ---------------------------------------------------------------- IMPROVE side
def load_improve() -> pd.DataFrame:
    from improve_io import load_improve_clean

    imp = load_improve_clean(usecols=lambda c: c in set(IMPROVE_COLS))
    imp["Date"] = pd.to_datetime(imp["Date"], errors="coerce")
    return prepare_improve(imp)


def prepare_improve(imp: pd.DataFrame) -> pd.DataFrame:
    imp = imp.copy()
    imp["year"] = imp["Date"].dt.year
    imp["post"] = imp["year"] >= STABLE_YEAR
    imp["oc_ec"] = imp["OCf_Val"] / imp["ECf_Val"]
    imp["fabs_ec"] = imp["fAbs_Val"] / imp["ECf_Val"]
    imp["valid_loading"] = imp["EC_loading_ug"].notna() & imp["EC_loading_ug_cm2_primary"].notna()
    return imp


def inside(s: pd.Series, b: dict) -> pd.Series:
    return s.between(b["p05"], b["p95"], inclusive="both")


def screens(imp: pd.DataFrame, bounds: dict) -> list[dict]:
    """Each Addis bound alone and in sequence, on the post-2003 rows with a loading."""
    post = imp[imp["post"]]
    base = post[post["valid_loading"]]
    f = inside(base["fAbs_Val"], bounds["fabs"])
    m = inside(base["EC_loading_ug"], bounds["ec_ug"])
    s = inside(base["EC_loading_ug_cm2_primary"], bounds["ec_ug_cm2"])
    steps = [("All cleaned IMPROVE rows", imp, "all"), (f"Post-{STABLE_YEAR}", post, "all"),
             ("… with a volume (EC mass and surface loading)", base, "all"),
             ("fAbs in the Addis p05–p95", base[f], "alone"), ("EC mass in the Addis p05–p95", base[m], "alone"),
             ("EC surface loading in the Addis p05–p95", base[s], "alone"),
             ("fAbs and EC mass", base[f & m], "sequence"), ("fAbs, EC mass and EC surface loading", base[f & m & s], "sequence")]
    return [{"step": name, "kind": kind, "n": int(len(d)), "sites": int(d["SiteCode"].nunique())} for name, d, kind in steps]


def density(x: pd.Series, y: pd.Series, x_max: float, y_max: float, bins: int = 60) -> dict:
    """Counts on a bins x bins grid from 0 to (x_max, y_max); only non-empty cells, plus how many lie beyond."""
    d = pd.concat([x, y], axis=1).dropna()
    d = d[(d.iloc[:, 0] >= 0) & (d.iloc[:, 1] >= 0)]
    within = d[(d.iloc[:, 0] < x_max) & (d.iloc[:, 1] < y_max)]
    counts, _, _ = np.histogram2d(within.iloc[:, 0], within.iloc[:, 1], bins=bins, range=[[0, x_max], [0, y_max]])
    cells = [[int(i), int(j), int(c)] for (i, j), c in np.ndenumerate(counts) if c > 0]
    return {"x_max": r(x_max), "y_max": r(y_max), "bins": bins, "cells": cells, "n": int(len(d)), "beyond": int(len(d) - len(within))}


def matched(imp: pd.DataFrame, spartan: pd.DataFrame, bounds: dict) -> list[dict]:
    post = imp[imp["post"]]
    etad = spartan[spartan["code"].eq("ETAD")]
    groups = [
        ("ETAD (Addis)", etad["fabs"], etad["fabs"] / etad["ec_conc"], etad["ec_conc"], etad["ec_ug"]),
    ]
    for label, sel in [("IMPROVE, EC mass matched", inside(post["EC_loading_ug"], bounds["ec_ug"])),
                       ("IMPROVE, EC concentration matched", inside(post["ECf_Val"], bounds["ec_conc"]))]:
        g = post[sel]
        groups.append((label, g["fAbs_Val"], g["fabs_ec"], g["ECf_Val"], g["EC_loading_ug"]))
    out = []
    for label, fabs, fabs_ec, ec, ec_ug in groups:
        out.append({"group": label, "fabs": quantiles(fabs), "fabs_ec": quantiles(fabs_ec), "ec_conc": quantiles(ec),
                    "ec_ug": quantiles(ec_ug), "fit": ols(ec, fabs)})
    return out


def sweep(imp: pd.DataFrame) -> list[dict]:
    post = imp[imp["post"]]
    out = []
    for t in SWEEP:
        g = post[post["fAbs_Val"] >= t]
        out.append({"threshold": t, "sites": int(g["SiteCode"].nunique()), "median_fabs_ec": r(g["fabs_ec"].median()) if len(g) else None,
                    **ols(g["ECf_Val"], g["fAbs_Val"])})
    return out


def intercepts(imp: pd.DataFrame, min_site_n: int = 100) -> list[dict]:
    """fAbs at EC = 0 (OLS of fAbs on FED ECf) over the fit ranges ftir_26 used."""
    post = imp[imp["post"]]
    p95 = post["ECf_Val"].quantile(0.95)
    like = post[post["oc_ec"] <= OCEC_ADDIS_LIKE]
    ranges = [("All post-2003", post), (f"EC ≤ p95 ({p95:.2f} µg/m³)", post[post["ECf_Val"] <= p95]),
              (f"OC/EC ≤ {OCEC_ADDIS_LIKE} (Addis-like)", like),
              ("Addis-like, EC ≤ its p95", like[like["ECf_Val"] <= like["ECf_Val"].quantile(0.95)])]
    out = [{"range": name, **ols(g["ECf_Val"], g["fAbs_Val"])} for name, g in ranges]
    per_site = [ols(g["ECf_Val"], g["fAbs_Val"])["intercept"] for _, g in post.groupby("SiteCode") if len(g) >= min_site_n]
    per_site = [v for v in per_site if v is not None]
    out.append({"range": f"Per-site median (sites with ≥ {min_site_n} rows)", "n": len(per_site), "slope": None,
                "intercept": r(np.median(per_site)) if per_site else None, "r2": None})
    return out


def smoke(imp: pd.DataFrame, bounds: dict) -> list[dict]:
    post = imp[imp["post"] & imp["valid_loading"]]
    comparable = inside(post["fAbs_Val"], bounds["fabs"]) & inside(post["EC_loading_ug"], bounds["ec_ug"]) & inside(post["EC_loading_ug_cm2_primary"], bounds["ec_ug_cm2"])
    out = []
    for label, g in [("All post-2003 with a loading", post), ("Addis-comparable (all three bounds)", post[comparable]),
                     (f"fAbs ≥ {HIGH_FABS:g} Mm⁻¹", post[post["fAbs_Val"] >= HIGH_FABS])]:
        smoky = g["oc_ec"] >= OCEC_SMOKE
        out.append({"group": label, "n": int(len(g)), "sites": int(g["SiteCode"].nunique()), "smoke_like": int(smoky.sum()),
                    "smoke_share": r(smoky.mean()) if len(g) else None, "median_oc_ec": r(g["oc_ec"].median()) if len(g) else None})
    return out


def rt(imp: pd.DataFrame, bounds: dict, bins: int = 50, min_cell: int = 3, max_points: int = 2000) -> dict | None:
    """FED 635 nm R/T proxies: binned (count, median fAbs), the Addis-comparable rows, the initial-reading line."""
    post = imp[imp["post"]]
    if not any(c in post.columns for p in RT_PAIRS.values() for c in p):
        return None
    comparable = post["valid_loading"] & inside(post["fAbs_Val"], bounds["fabs"]) & inside(post["EC_loading_ug"], bounds["ec_ug"])
    out: dict = {"pairs": {}}
    for label, (rc, tc) in RT_PAIRS.items():
        if rc not in post.columns or tc not in post.columns:
            continue
        d = post[[rc, tc, "fAbs_Val"]].dropna()
        if d.empty:
            continue
        x0, x1 = d[rc].quantile([0.001, 0.999]).to_numpy(float)
        y0, y1 = d[tc].quantile([0.001, 0.999]).to_numpy(float)
        xi = np.clip(((d[rc] - x0) / (x1 - x0 or 1) * bins).astype(int), 0, bins - 1)
        yi = np.clip(((d[tc] - y0) / (y1 - y0 or 1) * bins).astype(int), 0, bins - 1)
        g = d.assign(i=xi, j=yi).groupby(["i", "j"])["fAbs_Val"].agg(["size", "median"]).reset_index()
        g = g[g["size"] >= min_cell]
        pts = post.loc[comparable, [rc, tc, "fAbs_Val"]].dropna()
        if len(pts) > max_points:
            pts = pts.sample(max_points, random_state=7)
        out["pairs"][label] = {"ref": rc, "trans": tc, "n": int(len(d)), "x": [r(x0), r(x1)], "y": [r(y0), r(y1)], "bins": bins,
                               "cells": [[int(a), int(b), int(n), r(m, 3)] for a, b, n, m in g.itertuples(index=False)],
                               "comparable": [[r(a, 4), r(b, 4), r(c, 3)] for a, b, c in pts.itertuples(index=False)]}
    ri, ti = RT_PAIRS["initial"]
    rm, tm = RT_PAIRS["minimum"]
    if {ri, ti, rm, tm}.issubset(post.columns):
        line = ols(post[ri], post[ti])
        out["initial_line"] = line
        if line["slope"] is not None:
            d = post[[rm, tm, "fAbs_Val"]].dropna()
            d = d.assign(deficit=d[tm] - (line["intercept"] + line["slope"] * d[rm]))
            edges = [0, 2, 5, 10, 20, 40, 70, 100, np.inf]
            d["bin"] = pd.cut(d["fAbs_Val"], edges, right=False)
            out["deficit"] = [{"fabs_lo": r(iv.left), "fabs_hi": None if not np.isfinite(iv.right) else r(iv.right), **quantiles(g["deficit"])}
                              for iv, g in d.groupby("bin", observed=True) if len(g)]
    return out


def mass_space(imp: pd.DataFrame, spartan: pd.DataFrame, area_improve: float, area_spartan: float) -> list[dict]:
    """Quantiles in mass on the filter (µg) and of fAbs x V / A (dimensionless optical depth), per network/site."""
    post = imp[imp["post"]]
    rows = []
    imp_q = {
        "EC mass (µg)": post["EC_loading_ug"], "OC mass (µg)": post["OCf_Val"] * post["volume_m3"],
        "Iron mass (µg)": post["FEf_Val"] * post["volume_m3"],
        "PM2.5 mass (µg)": post["MF_loading_ug"] if "MF_loading_ug" in post.columns else post["MF_Val"] * post["volume_m3"],
        "fAbs × V / A": post["fAbs_Val"] * 1e-6 * post["volume_m3"] / (area_improve * 1e-4),
    }
    for q, s in imp_q.items():
        rows.append({"group": "IMPROVE", "quantity": q, **quantiles(s)})
    for code, g in spartan.groupby("code"):
        sp_q = {"EC mass (µg)": g["ec_ug"], "OC mass (µg)": g["oc_ug"], "fAbs × V / A": g["fabs"] * 1e-6 * g["volume_m3"] / (g["area_cm2"].fillna(area_spartan) * 1e-4)}
        for q, s in sp_q.items():
            rows.append({"group": code, "quantity": q, **quantiles(s)})
    return rows


def bundle(imp: pd.DataFrame, spartan: pd.DataFrame, area_improve: float = 3.5, area_spartan: float = 3.53) -> dict:
    bounds = etad_bounds(spartan)
    post = imp[imp["post"]]
    etad = spartan[spartan["code"].eq("ETAD")]
    x_mass = max(float(etad["ec_ug"].max()), float(post["EC_loading_ug"].quantile(0.999))) * 1.1
    x_surf = max(float(etad["ec_ug_cm2"].max()), float(post["EC_loading_ug_cm2_primary"].quantile(0.999))) * 1.1
    y_fabs = max(float(etad["fabs"].max()), float(post["fAbs_Val"].quantile(0.999))) * 1.1
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"improve": "research/ftir_hips_chem/output/improve_high_fabs_comparison/improve_valid_cleaned.csv (improve_io.load_improve_clean)",
                   "spartan": "research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl"},
        "improve": {"n": int(len(imp)), "n_post": int(len(post)), "sites": int(imp["SiteCode"].nunique()),
                    "date_min": str(imp["Date"].min().date()), "date_max": str(imp["Date"].max().date()), "area_cm2": area_improve},
        "bounds": bounds,
        "screens": screens(imp, bounds),
        "density": {"mass": density(post["EC_loading_ug"], post["fAbs_Val"], x_mass, y_fabs),
                    "surface": density(post["EC_loading_ug_cm2_primary"], post["fAbs_Val"], x_surf, y_fabs)},
        "matched": matched(imp, spartan, bounds),
        "sweep": sweep(imp),
        "intercepts": intercepts(imp),
        "smoke": smoke(imp, bounds),
        "rt": rt(imp, bounds),
        "mass_space": mass_space(imp, spartan, area_improve, area_spartan),
    }


def main() -> None:
    from config import IMPROVE_DEPOSIT_AREA_CM2, SPARTAN_DEPOSIT_AREA_CM2

    spartan = spartan_reference()
    imp = load_improve()
    payload = bundle(imp, spartan, IMPROVE_DEPOSIT_AREA_CM2, SPARTAN_DEPOSIT_AREA_CM2)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "envelope.json").write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {OUT / 'envelope.json'}: {payload['improve']['n_post']:,} post-{STABLE_YEAR} IMPROVE rows")
    for s in payload["screens"]:
        print(f"  {s['step']}: {s['n']:,} rows, {s['sites']} sites")


if __name__ == "__main__":
    main()
