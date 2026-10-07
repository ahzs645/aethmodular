"""Export Satoshi's 2 Oct 2026 follow-up for the gallery's "Satoshi 2 Oct" tab.

Run with:  uv run --no-sync python gallery/data/export_satoshi_followup.py

Reads only the saved outputs of
  research/ftir_hips_chem/workflows/run_satoshi_followup_20261002.py
and writes gallery/app/public/data/satoshi/followup_20261002.json. Nothing is
refitted here. The short answers are built from the same tables the charts draw.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research/ftir_hips_chem/output/tables/satoshi_followup_20261002"
OUT = ROOT / "gallery/app/public/data/satoshi"
SEASONS = ["Dry (Oct-Feb)", "Belg (Mar-May)", "Kiremt (Jun-Sep)"]
GROUPS = ["All Addis"] + SEASONS
SEASON_COLORS = {"Dry (Oct-Feb)": "#E67E22", "Belg (Mar-May)": "#27AE60",
                 "Kiremt (Jun-Sep)": "#3498DB", "All Addis": "#5b6470"}
SPLINE, VIBES = "AIRSpec", "VIBES-full-r70"


def r(v, d=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if not np.isfinite(f) else round(f, d)


def records(frame, digits=4):
    return [{k: (r(v, digits) if isinstance(v, (int, float, np.floating, np.integer)) else
                 (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v))
             for k, v in row.items()} for row in frame.to_dict(orient="records")]


def columns(frame, digits=4):
    out = {}
    for c in frame.columns:
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s):
            out[c] = [r(v, digits) for v in s.to_numpy()]
        else:
            out[c] = [None if pd.isna(v) else str(v) for v in s]
    return out


def q(s, p):
    s = pd.Series(s).dropna()
    return float(np.percentile(s, p)) if len(s) else None


METRICS = ("addis_slope", "addis_intercept", "addis_R2", "k", "improve_test_R2")


def swap_tables(fits):
    """Protocol fit + repeat spread per arm, and the two effects."""
    arms = []
    rep = fits.loc[fits.repeat.ge(0)]
    for key, d in fits.groupby(["selected_on", "mask", "calibrated_on", "group"], sort=False):
        protocol = d.loc[d.repeat.eq(-1)].iloc[0]
        rr = rep.loc[(rep.selected_on.eq(key[0]) & rep["mask"].eq(key[1]) & rep.calibrated_on.eq(key[2])
                      & rep.group.eq(key[3]))]
        row = dict(zip(["selected_on", "mask", "calibrated_on", "group"], key), n_repeats=len(rr))
        for m in METRICS:
            row[f"{m}_protocol"] = protocol[m]
            row[f"{m}_median"], row[f"{m}_p10"], row[f"{m}_p90"] = rr[m].median(), q(rr[m], 10), q(rr[m], 90)
        arms.append(row)
    arms = pd.DataFrame(arms)
    effects = []
    for mask in ("presented", "oct1"):
        vsel = (VIBES, mask)
        for g in GROUPS:
            def arm(sel, cal):
                return rep.loc[rep.selected_on.eq(sel[0]) & rep["mask"].eq(sel[1]) & rep.calibrated_on.eq(cal)
                               & rep.group.eq(g)].set_index("repeat")
            for metric in ("addis_slope", "addis_intercept"):
                # baseline effect: same cohort and split, calibrated on VIBES instead of spline (paired)
                for sel in ((SPLINE, "presented"), vsel):
                    diff = (arm(sel, VIBES)[metric] - arm(sel, SPLINE)[metric]).dropna()
                    effects.append({"mask": mask, "group": g, "metric": metric, "effect": "baseline",
                                    "held_fixed": f"selected on {'spline' if sel[0] == SPLINE else 'VIBES'}",
                                    "median": diff.median(), "p10": q(diff, 10), "p90": q(diff, 90), "paired": True})
                # training-set effect: different cohorts, same calibration baseline (independent draws)
                for cal in (SPLINE, VIBES):
                    a, b = arm((SPLINE, "presented"), cal)[metric].dropna(), arm(vsel, cal)[metric].dropna()
                    allpairs = (b.to_numpy()[:, None] - a.to_numpy()[None, :]).ravel()
                    effects.append({"mask": mask, "group": g, "metric": metric, "effect": "training set",
                                    "held_fixed": f"calibrated on {'spline' if cal == SPLINE else 'VIBES'}",
                                    "median": b.median() - a.median(), "p10": q(allpairs, 10),
                                    "p90": q(allpairs, 90), "paired": False})
    return arms, pd.DataFrame(effects)


def headlines(overlap, effects, ladder, vipsim, envelope, bands):
    """Numbers quoted in the short answers, keyed by what they answer."""
    h = {}
    o = overlap.loc[overlap["mask"].eq("presented")].set_index("group")
    h["overlap"] = {g: {"shared": int(o.loc[g, "shared"]), "spline_only": int(o.loc[g, "spline_only"]),
                        "vibes_only": int(o.loc[g, "vibes_only"])} for g in GROUPS}
    e = effects.loc[effects["mask"].eq("presented") & effects.metric.eq("addis_intercept")]
    h["intercept_effects"] = {g: {"training": e.loc[e.group.eq(g) & e.effect.eq("training set"), "median"].abs().mean(),
                                  "baseline": e.loc[e.group.eq(g) & e.effect.eq("baseline"), "median"].abs().mean()}
                              for g in GROUPS}
    s = vipsim.loc[vipsim["mask"].eq("presented")]
    h["vip_spearman"] = {b: [s.loc[s.baseline.eq(b), "spearman_vs_standard"].min(),
                             s.loc[s.baseline.eq(b), "spearman_vs_standard"].max()] for b in (SPLINE, VIBES)}
    vb = bands.loc[bands.model_id.str.startswith(f"analog|{VIBES}|presented")
                   & bands.band.isin(["PTFE C–F (1050–1250)", "1250–1425"])]
    h["vibes_analog_vip2_1050_1425"] = vb.groupby("model_id").vip2_share.sum().agg(["min", "max"]).tolist()
    st = bands.loc[bands.model_id.eq(f"standard|{VIBES}")].set_index("band").vip2_share
    h["vibes_standard_vip2"] = {"1050–1425": float(st.get("PTFE C–F (1050–1250)", 0) + st.get("1250–1425", 0)),
                                "CO₂ / 1800–2500": float(st.get("CO₂ / 1800–2500", 0))}
    own = ladder.loc[ladder.family.eq("analog") & ladder["mask"].eq("presented")]
    cv = own.loc[own.stage.eq("cross-validation")]
    h["analog_cv_r2"] = {b: [cv.loc[cv.baseline.eq(b), "R2"].min(), cv.loc[cv.baseline.eq(b), "R2"].max()]
                         for b in (SPLINE, VIBES)}
    h["analog_cv_bias_max_abs"] = float(cv.bias.abs().max())
    ad = own.loc[own.stage.eq("Addis test set") & own.eval_group.eq(own.group)]
    h["analog_addis_intercept"] = {b: [ad.loc[ad.baseline.eq(b), "deming_intercept"].min(),
                                       ad.loc[ad.baseline.eq(b), "deming_intercept"].max()] for b in (SPLINE, VIBES)}
    h["loading_beyond_p95"] = {b: ad.loc[ad.baseline.eq(b)].set_index("group").loading_beyond_train_p95.to_dict()
                               for b in (SPLINE, VIBES)}
    std_ad = ladder.loc[ladder.family.eq("standard") & ladder.population.eq("addis") & ladder.eval_group.eq("All Addis")]
    h["standard_loading_beyond_p95"] = std_ad.set_index("baseline").loading_beyond_train_p95.to_dict()
    h["standard_addis"] = std_ad.set_index("baseline")[["deming_slope", "deming_intercept", "addis_loading_median_ug",
                                                        "train_loading_median_ug", "train_loading_p95_ug"]].to_dict("index")
    env = envelope.loc[envelope["mask"].eq("presented")]
    h["envelope_pct_median"] = {b: [env.loc[env.baseline.eq(b), "pct_median"].min(),
                                    env.loc[env.baseline.eq(b), "pct_median"].max()] for b in (SPLINE, VIBES)}
    h["envelope_below_p10"] = {b: [env.loc[env.baseline.eq(b), "below_p10"].min(),
                                   env.loc[env.baseline.eq(b), "below_p10"].max()] for b in (SPLINE, VIBES)}
    return json.loads(json.dumps(h, default=float), parse_float=lambda v: round(float(v), 4))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((SRC / "summary.json").read_text())
    overlap = pd.read_csv(SRC / "overlap_summary.csv")
    props = pd.read_csv(SRC / "overlap_properties.csv")
    members = pd.read_csv(SRC / "overlap_membership.csv")
    fits = pd.read_csv(SRC / "swap_fits.csv")
    repro = pd.read_csv(SRC / "swap_reproduction.csv")
    ladder = pd.read_csv(SRC / "fit_ladder.csv")
    lpts = pd.read_csv(SRC / "ladder_points.csv")
    bands = pd.read_csv(SRC / "vip_bands.csv")
    vipsim = pd.read_csv(SRC / "vip_similarity.csv")
    vip = json.loads((SRC / "vip_profiles.json").read_text())
    envelope = pd.read_csv(SRC / "envelope_summary.csv")
    epts = pd.read_csv(SRC / "envelope_points.csv")
    years = pd.read_csv(SRC / "analog_years.csv")
    arms, effects = swap_tables(fits)
    for m in vip["models"].values():
        m["vip"] = [round(v, 3) for v in m["vip"]]
        m.pop("coef", None)
    payload = {
        "schema_version": 1,
        "generated_from": str(SRC.relative_to(ROOT)),
        "summary": summary,
        "season_colors": SEASON_COLORS, "groups": GROUPS, "seasons": SEASONS,
        "overlap": records(overlap), "overlap_props": records(props),
        "members": columns(members.drop(columns=["date"]), 4),
        "swap_arms": records(arms), "swap_effects": records(effects), "swap_reproduction": records(repro),
        "swap_repeats": columns(fits.loc[fits.repeat.ge(0), ["selected_on", "mask", "calibrated_on", "group",
                                                              "repeat", "addis_slope", "addis_intercept", "k"]], 3),
        "ladder": records(ladder),
        "ladder_points": columns(lpts, 3),
        "vip": vip, "vip_bands": records(bands), "vip_similarity": records(vipsim),
        "envelope": records(envelope),
        "envelope_points": columns(epts[["baseline", "filter_id", "Site", "role", "tor_ec_ugm3",
                                         "ftir_oos_ugm3", "residual_pct"]], 3),
        "years": records(years),
        "headlines": headlines(overlap, effects, ladder, vipsim, envelope, bands),
    }
    path = OUT / "followup_20261002.json"
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False, ensure_ascii=False))
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
