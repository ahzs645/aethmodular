"""Export the 2026-09-17 meeting follow-up for the gallery's click-through tab.

Run with:  uv run --no-sync python gallery/data/export_meeting_followup.py

Reads only the saved outputs of
  research/ftir_hips_chem/workflows/run_meeting_followup_20260917.py
  research/ftir_hips_chem/workflows/run_meeting_grid_20260917.py
and writes gallery/app/public/data/meeting/followup_20260917.json. Nothing is
refitted here. Headline sentences are built from the same tables the charts
draw, so the prose cannot drift from the numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research/ftir_hips_chem/output/tables/meeting_followup_20260917"
OUT = ROOT / "gallery/app/public/data/meeting"
OUT.mkdir(parents=True, exist_ok=True)
SEASONS = ["Dry (Oct-Feb)", "Belg (Mar-May)", "Kiremt (Jun-Sep)"]
GROUPS = ["All Addis"] + SEASONS
SEASON_COLORS = {"Dry (Oct-Feb)": "#E67E22", "Belg (Mar-May)": "#27AE60",
                 "Kiremt (Jun-Sep)": "#3498DB", "All Addis": "#5b6470"}


def r(v, d=4):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if not np.isfinite(f) else round(f, d)


def records(frame, digits=4):
    out = []
    for row in frame.to_dict(orient="records"):
        out.append({k: r(v, digits) if isinstance(v, (int, float, np.floating, np.integer)) else
                    (None if (isinstance(v, float) and not np.isfinite(v)) else v)
                    for k, v in row.items()})
    return out


def columns(frame, digits=4):
    out = {}
    for c in frame.columns:
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s):
            out[c] = [r(v, digits) for v in s.to_numpy()]
        else:
            out[c] = [None if pd.isna(v) else str(v) for v in s]
    return out


def short_fit(fit_id):
    fam, method, rest = fit_id.split("|", 2)
    return fam, method, rest


METHOD_NAMES = {"AIRSpec": "Spline baseline", "VIBES": "VIBES baseline (4000–1425)",
                "VIBES-full": "VIBES baseline (4000–500)", "VIBES-full-cut": "VIBES baseline (fit 4000–500, used 4000–1425)"}
SPEC_COHORT_AND_SETTINGS = [
    ("Lowest OC/EC", lambda r: r["category"] == "ocec"),
    ("Spectral analogs (correlation), all Addis", lambda r: r["category"] == "corr_all"),
    ("Spectral analogs (correlation), Dry season", lambda r: r["category"] == "corr_dry"),
    ("Spectral analogs (correlation), Belg season", lambda r: r["category"] == "corr_belg"),
    ("Spectral analogs (correlation), Kiremt season", lambda r: r["category"] == "corr_kiremt"),
    ("Ethiopia-shaped smoke", lambda r: r["category"] == "eth_shaped"),
    ("Spectral analogs (VIP-weighted distance)", lambda r: r["category"] == "vip_analogs"),
    ("All IMPROVE filters, or smoke (906)", lambda r: r["category"] in ("pool", "smoke")),
    ("Site-grouped cross-validation", lambda r: r["mode"] == "site_heldout"),
    ("Interleaved cross-validation", lambda r: r["mode"] == "app"),
    ("9 or fewer PLS factors", lambda r: r["k"] <= 9),
    ("15 or more PLS factors", lambda r: r["k"] >= 15),
    ("Cohort under 600 filters", lambda r: r["cutoff"] < 600),
]
N_COHORT = 8   # rows 0-7 of SPEC_COHORT_AND_SETTINGS are cohorts, the rest calibration settings


def spec_choices(methods):
    """Preprocessing rows (one per baseline present in the grid), then cohort and settings rows."""
    pre = [(METHOD_NAMES.get(m, m), (lambda r, m=m: r["method"] == m)) for m in methods]
    choices = pre + SPEC_COHORT_AND_SETTINGS
    n = len(pre)
    groups = [("Spectral preprocessing", 0, n), ("Calibration cohort", n, n + N_COHORT),
              ("Calibration settings", n + N_COHORT, len(choices))]
    return choices, groups


def export_spec_curve(grid_path, out_path):
    """The meeting grid as a specification curve, in spec_curve.json's schema.

    Guardrails are the same function and thresholds as the Calibration tab's
    curve (research/ftir_ec_phase3/scripts/spec_curve.py). Protocol B has no
    held-out split, so like the original it cannot pass the held-out floor.
    """
    import sys
    sys.path.insert(0, str(ROOT / "research/ftir_ec_phase3/scripts"))
    from spec_curve import GUARDRAIL_TEXT, HELDOUT_FLOOR, SLOPE_RANGE, passes_guardrails

    rows = [json.loads(l) for l in grid_path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("all_intercept") is not None and "extrap_pct" in r]
    rows.sort(key=lambda r: r["all_intercept"])
    methods = [m for m in METHOD_NAMES if any(x["method"] == m for x in rows)]
    SPEC_CHOICES, SPEC_GROUPS = spec_choices(methods)
    std = [{**r, "deming_slope": r["all_slope"], "deming_intercept": r["all_intercept"]} for r in rows]
    ok = [passes_guardrails(r) for r in std]
    bits = lambda flags: "".join("1" if f else "0" for f in flags)  # noqa: E731
    codes = [m.lower() for m in methods]
    cats = {"co": sorted({r["category"] for r in rows}), "sp": codes,
            "sel": codes, "mo": ["app", "site_heldout"], "lot": ["all"]}
    idx = {k: {v: i for i, v in enumerate(vs)} for k, vs in cats.items()}
    payload = {
        "source": str(grid_path.relative_to(ROOT)),
        "n": len(rows), "n_pass": sum(ok), "guardrails": GUARDRAIL_TEXT,
        "heldout_floor": HELDOUT_FLOOR, "slope_range": list(SLOPE_RANGE),
        "choices": [c for c, _ in SPEC_CHOICES],
        "groups": [{"title": t, "start": a, "end": b} for t, a, b in SPEC_GROUPS],
        "masks": [bits(t(r) for r in rows) for _, t in SPEC_CHOICES],
        "ok": bits(ok),
        "intercept": [r(x["all_intercept"]) for x in rows],
        "slope": [r(x["all_slope"]) for x in rows],
        "heldout": [r(x.get("heldout_R2"), 3) for x in rows],
        "cut": [x["cutoff"] for x in rows], "k": [x["k"] for x in rows],
        "categories": cats,
        "co": [idx["co"][x["category"]] for x in rows],
        "sp": [idx["sp"][x["method"].lower()] for x in rows],
        "sel": [idx["sel"][x["method"].lower()] for x in rows],
        "mo": [idx["mo"][x["mode"]] for x in rows],
        "lot": [0] * len(rows),
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    print(f"spec curve: {len(rows)} specifications, {sum(ok)} pass every guardrail -> {out_path.name}")


def main():
    summary = json.loads((SRC / "summary.json").read_text())
    fits = pd.read_csv(SRC / "fits.csv")
    stats = pd.read_csv(SRC / "panel_stats.csv")
    ap = pd.read_csv(SRC / "analog_predictions.csv")
    adp = pd.read_csv(SRC / "addis_predictions.csv")
    rep = pd.read_csv(SRC / "repeated_splits.csv")
    stitched = pd.read_csv(SRC / "stitched_repeats.csv")
    op = pd.read_csv(SRC / "op_summary.csv")
    opt = pd.read_csv(SRC / "op_terciles.csv")
    repro = pd.read_csv(SRC / "meeting_reproduction.csv")
    overlap = pd.read_csv(SRC / "analog_overlap.csv")
    refs = pd.read_csv(SRC / "improve_references.csv")
    spectra = json.loads((SRC / "spectra_quantiles.json").read_text())

    # ---- per-filter analog predictions (locked mask only; the meeting mask is reproduction)
    ap = ap.loc[~ap.fit_id.str.contains(r"\|meeting\|")].copy()
    parts = ap.fit_id.str.split("|")
    ap["family"] = parts.str[0]
    ap["method"] = parts.str[1]
    ap["model"] = np.where(ap.family.eq("seasonal"), "seasonal calibration", parts.str[2])
    ap["group"] = np.where(ap.family.eq("seasonal"), parts.str[3], parts.str[3])
    ap["fabs10"] = ap.fabs_Mm1 / 10.0
    analog_cols = ["family", "method", "model", "group", "filter_id", "Site", "date", "lot", "role",
                   "tor_ec_ugm3", "fabs10", "ftir_ec_ugm3", "op_tor_frac", "tor_oc_ec", "y"]
    analogs = columns(ap[analog_cols], 4)

    adp = adp.loc[~adp.fit_id.str.contains(r"\|meeting\|")].copy()
    adp["fabs10"] = adp.Fabs / 10.0
    addis = columns(adp[["fit_id", "MediaId", "ExternalFilterId", "date", "season", "fabs10", "ftir_ec_ugm3"]], 4)

    # ---- pool MAC histogram and analog-set MAC distributions (log bins)
    ok = (refs.tor_ec_ugm3 > 0.02) & (refs.fabs_Mm1 > 0)
    mac = (refs.fabs_Mm1 / refs.tor_ec_ugm3)[ok]
    edges = np.geomspace(2, 60, 41)
    hist = {"edges": [r(e, 3) for e in edges],
            "pool": np.histogram(mac.clip(edges[0], edges[-1]), edges)[0].tolist(),
            "pool_median": r(mac.median(), 3), "pool_n": int(ok.sum())}
    members = pd.read_csv(SRC / "analog_membership.csv")
    members = members.loc[members["mask"].eq("locked")].merge(
        refs[["filter_id", "tor_ec_ugm3", "fabs_Mm1"]], on="filter_id")
    mm = members.loc[(members.tor_ec_ugm3 > 0.02) & (members.fabs_Mm1 > 0)]
    for (method, g), d in mm.groupby(["method", "group"]):
        v = (d.fabs_Mm1 / d.tor_ec_ugm3).clip(edges[0], edges[-1])
        hist[f"{method}|{g}"] = np.histogram(v, edges)[0].tolist()
        hist[f"{method}|{g}|median"] = r(v.median(), 3)

    # ---- spectra quantiles, every 3rd channel to keep the file small
    # each baseline keeps its own grid (the full-range VIBES runs to 500 cm-1)
    grids = {k: np.asarray(v) for k, v in spectra.items() if k == "wn" or k.startswith("wn|")}
    keeps = {k: np.arange(0, len(v), 3) for k, v in grids.items()}
    spec = {k: [r(v, 1) for v in grids[k][keeps[k]]] for k in grids}
    for k, q in spectra.items():
        if k in grids:
            continue
        gk = f"wn|{k.split('|')[0]}"
        keep = keeps.get(gk, keeps["wn"])
        spec[k] = [[r(v, 5) for v in np.asarray(row)[keep]] for row in q]

    # ---- category grid
    grid_path = SRC / "category_grid.jsonl"
    grid = pd.DataFrame([json.loads(l) for l in grid_path.read_text().splitlines() if l.strip()]) \
        if grid_path.exists() else pd.DataFrame()
    grid_cols = [c for c in ["method", "category", "label", "cutoff", "mode", "k", "train_n",
                             "heldout_R2", "all_slope", "all_intercept", "all_R2",
                             "dry_slope", "dry_intercept", "belg_slope", "belg_intercept",
                             "kiremt_slope", "kiremt_intercept", "negative_pct"] if c in grid]
    grid = grid.sort_values(["method", "category", "mode", "cutoff"]) if len(grid) else grid

    grid_summary = []
    if len(grid):
        sh = grid.loc[grid["mode"].eq("site_heldout") & (grid.heldout_R2 >= 0.85)].copy()
        sh["score"] = (1 - sh.all_slope).abs() + sh.all_intercept.abs()
        for method, d in sh.groupby("method"):
            best = d.sort_values("score").iloc[0]
            grid_summary.append({
                "method": method, "passing": int(len(d)),
                "balanced": int(((d.all_intercept.abs() < 1) & d.all_slope.between(0.8, 1.2)).sum()),
                "slope_min": r(d.all_slope.min(), 3), "slope_max": r(d.all_slope.max(), 3),
                "intercept_min": r(d.all_intercept.min(), 3), "intercept_max": r(d.all_intercept.max(), 3),
                "best_label": best.label, "best_cutoff": int(best.cutoff), "best_k": int(best.k),
                "best_slope": r(best.all_slope, 3), "best_intercept": r(best.all_intercept, 3),
                "best_heldout_R2": r(best.heldout_R2, 3)})

    # ---- headline numbers pulled from the tables
    def rep_med(method, g, col):
        s = rep.loc[rep.method.eq(method) & rep.group.eq(g), col]
        return float(s.median()), float(s.quantile(.1)), float(s.quantile(.9))

    heads = {}
    for method in summary.get("methods", ["AIRSpec", "VIBES"]):
        for g in GROUPS:
            heads[f"{method}|{g}"] = {
                "addis_slope": rep_med(method, g, "addis_slope"),
                "addis_intercept": rep_med(method, g, "addis_intercept"),
                "analog_ftir_fabs": rep_med(method, g, "ftir_vs_fabs_slope"),
                "analog_tor_fabs": rep_med(method, g, "tor_vs_fabs_slope"),
                "analog_ftir_tor": rep_med(method, g, "ftir_vs_tor_slope"),
                "k": rep_med(method, g, "k"),
            }
    pool_op = op.loc[op.method.eq("pool")].iloc[0]

    payload = {
        "schema_version": 1,
        "generated_from": str(SRC.relative_to(ROOT)),
        "summary": summary,
        "season_colors": SEASON_COLORS,
        "groups": GROUPS,
        "seasons": SEASONS,
        "fits": records(fits),
        "stats": records(stats),
        "analogs": analogs,
        "addis": addis,
        "repeats": columns(rep, 4),
        "stitched_repeats": columns(stitched, 4),
        "op_summary": records(op),
        "op_terciles": records(opt),
        "op_pool_rho": r(pool_op.spearman_op_vs_log_mac, 3),
        "reproduction": records(repro),
        "overlap": records(overlap),
        "mac_hist": hist,
        "spectra": spec,
        "grid": columns(grid[grid_cols], 4) if len(grid) else None,
        "grid_n": int(len(grid)),
        "grid_summary": grid_summary,
        "headlines": heads,
    }
    if grid_path.exists():
        export_spec_curve(grid_path, OUT / "spec_curve_20260917.json")
    path = OUT / "followup_20260917.json"
    path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB), grid rows {len(grid)}")


if __name__ == "__main__":
    main()
