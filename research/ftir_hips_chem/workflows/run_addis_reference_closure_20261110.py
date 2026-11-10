"""Which measurement carries the Addis offset, and what correction the existing data support (10 Nov 2026).

The 9 Nov status page left this open: "Which measurement carries the offset stays open". FTIR EC against HIPS
has a slope near 2 and an intercept near -6 µg/m³; against the aethalometer FTIR is proportional and HIPS has
the offset. Two instruments cannot settle which one bends. This uses every independent measurement already on
the same Addis filters, including two that are neither optical nor FTIR (gravimetric PM2.5 and XRF potassium).

It reads only committed files, so it runs from a plain checkout:

* ``gallery/review/closeout_20261107/continuum_baseline.json``  per-filter Addis predictions of every saved
  candidate, HIPS BC, the aethalometer match, and held-out-site IMPROVE predictions with TOR (7 Nov closeout)
* ``gallery/app/public/data/filters.json``                       PM2.5 mass, elements, ions, FTIR OC and OM
* ``Filter Data/unified_filter_dataset.pkl``                      sampled volume (optical depth = Fabs V / 100 A)
* ``gallery/app/public/data/spartan_reference.json``             every SPARTAN site, for the ``network`` stage

Stages (default: all)

* ``linearity``  response exponent of each instrument against each other one, with a third as the instrument
* ``triple``     triple collocation: each instrument's random error with no reference
* ``response``   the shape of the HIPS response at Addis against each linear measurement
* ``season``     FTIR / aethalometer by season and filter lot
* ``level``      what bounds the absolute FTIR level: held-out IMPROVE filters by OC/EC, mass closure, anchors
* ``product``    per-filter table: FTIR EC with a level range, and HIPS before and after linearising
* ``network``    the same exponent at every SPARTAN site, which needs no aethalometer (PM2.5 mass is the instrument)

    uv run python research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py

Writes ``output/tables/addis_reference_closure_20261110/`` (git-ignored) and, with ``--summary``, the numbers
quoted in docs/addis-reference-closure-2026-11-10.md as one JSON file. Nothing here is fitted to make FTIR agree
with an optical instrument, and nothing here measures thermal EC at Addis: the level stays a range.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path[:0] = [str(AREA / "scripts")]
import reference_closure as rc  # noqa: E402
from config import BASE_FILTER_ID_PATTERN, BASE_FILTER_ID_REPL, MAC_VALUE  # noqa: E402
from pls_transfer import deming_regression  # noqa: E402

CLOSEOUT = ROOT / "gallery/review/closeout_20261107/continuum_baseline.json"
FILTERS = ROOT / "gallery/app/public/data/filters.json"
NETWORK = ROOT / "gallery/app/public/data/spartan_reference.json"
UNIFIED = AREA / "Filter Data/unified_filter_dataset.pkl"
OUT = AREA / "output/tables/addis_reference_closure_20261110"
SUMMARY = ROOT / "docs/addis-reference-closure-2026-11-10.json"

DEPOSIT_AREA_CM2 = 3.53          # SPARTAN DepositArea on every ETAD HIPS row of the unified dataset
PUBLISHED, ANCHORED = "published", "vibes_anchorhp k18"
# reader-facing names (docs/naming-conventions.md); internal codes stay in the column keys
LABEL = {
    "fabs": "HIPS Fabs", PUBLISHED: "FTIR EC, SPARTAN published", ANCHORED: "FTIR EC, PTFE-anchored VIBES",
    "aeth_ir": "Aethalometer IR BCc", "pm25": "PM2.5 mass (gravimetric)", "potassium": "Potassium (XRF)",
    "hips_lin": "HIPS, saved linearisation", "tor_equiv": "Inferred TOR comparator",
}
STAGES = ("linearity", "triple", "response", "season", "level", "product", "network")
NETWORK_MIN_FILTERS = 40
# IMPROVE soil formula (Malm et al. 1994), as on the 28 Oct speciation table
SOIL = {"Aluminum": 2.20, "Silicon": 2.49, "Calcium": 1.63, "Iron": 2.42, "Titanium": 1.94}


# --------------------------------------------------------------------------- data
def load() -> tuple[pd.DataFrame, dict]:
    """One row per counted Addis filter: every saved FTIR candidate, the optical and the independent measures."""
    closeout = json.loads(CLOSEOUT.read_text())
    table = pd.DataFrame(closeout["addis_filters"]).set_index("filter_id")
    table = table[~table["excluded"].astype(bool)].copy()
    table["date"] = pd.to_datetime(table["date"])

    public = json.loads(FILTERS.read_text())
    chem = pd.DataFrame(public["rows"])
    chem = chem[chem["code"] == "ETAD"].set_index("id")
    wanted = ["PM2.5 mass", "Potassium", "Sulfate", "Nitrate", "Ammonium", "Sodium", "Magnesium", "Zinc", "Lead",
              "Copper", "OC (FTIR)", "OM (FTIR)", *SOIL]
    table = table.join(chem[wanted])

    unified = pd.read_pickle(UNIFIED)
    hips = unified[(unified["Site"] == "ETAD") & (unified["Parameter"] == "HIPS_Fabs")]
    volume = hips.groupby(hips["FilterId"].str.replace(BASE_FILTER_ID_PATTERN, BASE_FILTER_ID_REPL, regex=True))["Volume_m3"].first()
    table["volume_m3"] = volume.reindex(table.index)
    table["volume_known"] = table["volume_m3"].notna()
    # Addis volumes barely vary (SD under 2 %), so the site median stands in where the HIPS row is not in the
    # unified dataset; those filters are marked and can be dropped with --known-volume-only.
    table["volume_m3"] = table["volume_m3"].fillna(table["volume_m3"].median())

    table["fabs"] = table["hips_bc"] * MAC_VALUE                     # Mm-1; hips_bc is Fabs / MAC_VALUE
    table["tau"] = table["fabs"] * table["volume_m3"] / (100.0 * DEPOSIT_AREA_CM2)
    table["pm25"], table["potassium"] = table["PM2.5 mass"], table["Potassium"]
    return table, closeout


def per_filter(table: pd.DataFrame, column: str) -> pd.Series:
    """µg per filter of a concentration column: the loading the filter actually carried."""
    return table[column] * table["volume_m3"]


# --------------------------------------------------------------------------- stages
def stage_linearity(table, closeout, n_boot):
    """Who is proportional to whom. b = 1 is proportional; the instrument Z removes the flattening by noise in X."""
    plan = [
        ("fabs", "aeth_ir", ("pm25", PUBLISHED, "potassium")),
        ("fabs", PUBLISHED, ("aeth_ir", "pm25", "potassium")),
        ("fabs", "pm25", ("aeth_ir", PUBLISHED, "potassium")),
        (PUBLISHED, "aeth_ir", ("pm25", "fabs", "potassium")),
        (ANCHORED, "aeth_ir", ("pm25", "fabs", "potassium")),
        (PUBLISHED, "pm25", ("aeth_ir", "fabs", "potassium")),
        ("aeth_ir", "pm25", (PUBLISHED, "fabs", "potassium")),
        ("hips_lin", "aeth_ir", ("pm25", PUBLISHED, "potassium")),
        ("tor_equiv", "aeth_ir", ("pm25", PUBLISHED, "potassium")),
    ]
    rows = []
    for y, x, instruments in plan:
        for z in instruments:
            r = rc.response_exponent(table[y], table[x], table[z], n_boot=n_boot)
            rows.append({"y": LABEL[y], "x": LABEL[x], "instrument": LABEL[z], "group": "all", **r})
    for season, part in table.groupby("season"):
        for y in ("fabs", PUBLISHED):
            r = rc.response_exponent(part[y], part["aeth_ir"], part["pm25"], n_boot=n_boot)
            rows.append({"y": LABEL[y], "x": LABEL["aeth_ir"], "instrument": LABEL["pm25"], "group": season, **r})
    return {"linearity": pd.DataFrame(rows)}


def stage_triple(table, closeout, n_boot):
    """Random error of each instrument with no reference, and each one's exponent relative to the first."""
    triplets = [
        (PUBLISHED, "aeth_ir", "fabs"), (ANCHORED, "aeth_ir", "fabs"),
        (PUBLISHED, "aeth_ir", "pm25"), ("aeth_ir", PUBLISHED, "hips_lin"),
    ]
    rows = []
    for triplet in triplets:
        result = rc.triple_collocation(*(table[c] for c in triplet), n_boot=n_boot)
        for column, m in zip(triplet, result["measurements"]):
            rows.append({"triplet": " · ".join(LABEL[c] for c in triplet), "measurement": LABEL[column],
                         "n": result["n"], "signal_sd": result["signal_sd"], **m})
    return {"triple_collocation": pd.DataFrame(rows)}


def stage_response(table, closeout, n_boot):
    """Shape of HIPS optical depth against each linear loading, and what linearising does to the FTIR intercept."""
    references = {"aeth_ir": "aethalometer IR BCc", PUBLISHED: "FTIR EC, SPARTAN published",
                  ANCHORED: "FTIR EC, PTFE-anchored VIBES", "pm25": "PM2.5 mass"}
    fits = []
    for column, name in references.items():
        loading = per_filter(table, column)
        for model in rc.RESPONSE_MODELS:
            f = rc.fit_response(loading, table["tau"], model, n_boot=min(n_boot, 300))
            row = {"loading": name, "model": model, "n": f["n"], "rms_log": f["rms_log"], "aic": f["aic"]}
            for i, (p, (lo, hi)) in enumerate(zip(f["params"], f["intervals"]), start=1):
                row |= {f"p{i}": p, f"p{i}_lo": lo, f"p{i}_hi": hi}
            fits.append(row)
    fits = pd.DataFrame(fits)
    fits["delta_aic_vs_proportional"] = fits["aic"] - fits.groupby("loading")["aic"].transform(
        lambda s: s[fits.loc[s.index, "model"] == "proportional"].iloc[0])

    # tau_inf fitted on Beijing + Pasadena + Delhi with Addis held out (run_intercept_fixes_20261103 stage stress)
    stress = closeout["settings"]["stress"]
    held_out = {"central": stress["curve"]["tau_inf"], "lo": stress["tau_inf_boot"][0], "hi": stress["tau_inf_boot"][2]}
    lam = float(closeout["settings"]["LAMBDA_ADDIS"])
    rows = []
    for label, tau_inf in held_out.items():
        linear_fabs = rc.linearise(table["tau"], tau_inf) * 100.0 * DEPOSIT_AREA_CM2 / table["volume_m3"]
        usable = np.isfinite(linear_fabs)
        for model in (PUBLISHED, ANCHORED):
            d = deming_regression(linear_fabs[usable] / MAC_VALUE, table.loc[usable, model], lam)
            e = rc.response_exponent(linear_fabs, table["aeth_ir"], table["pm25"], n_boot=min(n_boot, 500))
            rows.append({"tau_inf": tau_inf, "which": label, "model": LABEL[model], "n": d["n"],
                         "dropped_at_or_above_tau_inf": int((~usable & table["tau"].notna()).sum()),
                         "slope_vs_linearised_hips": d["slope"], "intercept_vs_linearised_hips": d["intercept"],
                         "linearised_exponent_vs_aethalometer": e["exponent"],
                         "median_correction_factor": float(np.nanmedian(linear_fabs / table["fabs"])),
                         "mac_for_agreement": float(np.nanmedian(linear_fabs / table[model]))})
    raw = []
    for model in (PUBLISHED, ANCHORED):
        d = deming_regression(table["hips_bc"], table[model], lam)
        raw.append({"model": LABEL[model], "n": d["n"], "slope_vs_hips": d["slope"], "intercept_vs_hips": d["intercept"]})
    return {"hips_response_fits": fits, "linearised_hips": pd.DataFrame(rows), "raw_hips": pd.DataFrame(raw)}


def stage_season(table, closeout, n_boot):
    """Is there a season effect in FTIR once the comparator is not HIPS? Median FTIR / aethalometer by season and lot."""
    rows = []
    for keys, part in [(("all", "all"), table), *[((s, "all"), p) for s, p in table.groupby("season")],
                       *[((s, str(int(lot))), p) for (s, lot), p in table.groupby(["season", "lot"])]]:
        row = {"season": keys[0], "lot": keys[1], "n": int(part["aeth_ir"].notna().sum())}
        if row["n"] < 5:
            continue
        for column, name in ((PUBLISHED, "published"), (ANCHORED, "anchored"), ("hips_bc", "hips_mac10")):
            ratio = (part[column] / part["aeth_ir"]).dropna()
            row[f"{name}_over_aethalometer"] = float(ratio.median()) if len(ratio) else np.nan
        rows.append(row)
    return {"season_lot": pd.DataFrame(rows)}


def stage_level(table, closeout, n_boot):
    """Evidence on the absolute FTIR level. None of it is thermal EC measured at Addis."""
    improve = pd.DataFrame(closeout["improve_filters"])
    models = {"raw k18": "Raw spectra", ANCHORED: "PTFE-anchored VIBES", "spline k18": "Spline baseline",
              "anchored_balanced k12": "Anchored, OC/EC-balanced weights"}
    improve["ocec_bin"] = pd.cut(improve["ocec"], [0, 1.5, 2.5, 4.0, 100.0], labels=["< 1.5", "1.5-2.5", "2.5-4", "> 4"])
    by_ocec = []
    for name, part in improve[improve["tor_ug"] >= 15].groupby("ocec_bin", observed=True):
        row = {"tor_oc_over_ec": name, "n": len(part), "sites": part["site"].nunique()}
        for key, label in models.items():
            row[label] = float(np.median(part[key] / part["tor_ug"]))
        by_ocec.append(row)

    addis_ocec = (table["OC (FTIR)"] / table[PUBLISHED]).dropna()
    soil = sum(table[element] * factor for element, factor in SOIL.items())
    ions = table["Sulfate"] + table["Nitrate"] + table["Ammonium"].fillna(0)
    trace = table["Potassium"] + table["Sodium"].fillna(0) + table["Magnesium"].fillna(0) + table["Zinc"] + table["Lead"] + table["Copper"]
    rest = table["OM (FTIR)"] + soil + ions + trace
    closure = []
    for multiplier in (1.0, 1.15, 1.3, 1.5, 1.7, 1.9):
        ratio = ((rest + multiplier * table[PUBLISHED]) / table["pm25"]).dropna()
        closure.append({"ec_multiplier": multiplier, "n": len(ratio), "reconstructed_over_gravimetric": float(ratio.median()),
                        "ec_share_of_pm25": float((multiplier * table[PUBLISHED] / table["pm25"]).median())})

    bridge = pd.DataFrame(closeout["aeth_bridge"]).set_index("code")
    adama = closeout["settings"]["tor"]
    anchors = [
        {"anchor": "Held-out IMPROVE sites, TOR ≥ 15 µg and OC/EC < 2.5 (FTIR ÷ TOR, raw spectra, 18 components)",
         "published_over_true": np.nan, "value": float(np.median(improve.query("tor_ug >= 15 and ocec < 2.5").eval("`raw k18` / tor_ug"))),
         "note": "calibration set, not Addis; Addis FTIR OC/EC is lower still"},
        {"anchor": "Same, OC/EC < 1.8", "published_over_true": np.nan,
         "value": float(np.median(improve.query("tor_ug >= 15 and ocec < 1.8").eval("`raw k18` / tor_ug"))),
         "note": f"n {len(improve.query('tor_ug >= 15 and ocec < 1.8'))}"},
        {"anchor": "Aethalometer, scaled by Pasadena's published FTIR ÷ aethalometer",
         "published_over_true": float(bridge.loc["ETAD", "published_over_aeth"] / bridge.loc["USPA", "published_over_aeth"]),
         "value": np.nan, "note": "assumes the two MA350 units share a calibration; Beijing shows they need not"},
        {"anchor": "Aethalometer, scaled by Delhi's published FTIR ÷ aethalometer",
         "published_over_true": float(bridge.loc["ETAD", "published_over_aeth"] / bridge.loc["INDH", "published_over_aeth"]),
         "value": np.nan, "note": "24 Delhi filters"},
        {"anchor": "Adama pilot: Fabs ÷ TOR EC (m²/g), five filters", "published_over_true": np.nan,
         "value": float(adama["fabs_over_tor_median"]), "note": "another site and sampler; optical depth 0.15-0.44"},
    ]
    # The one closure that needs no MAC convention: Adama's own Fabs / TOR, linearised with the same constant as
    # Addis, against the MAC that Addis FTIR EC would need. Both sides move with tau_inf, so each is paired.
    pilot = pd.DataFrame(closeout["adama"])
    stress = closeout["settings"]["stress"]
    for which, tau_inf in (("central", stress["curve"]["tau_inf"]), ("lo", stress["tau_inf_boot"][0]), ("hi", stress["tau_inf_boot"][2])):
        adama_mac = float(np.median(pilot["fabs_over_tor"] * rc.linearise(pilot["tau"], tau_inf) / pilot["tau"]))
        addis_fabs = rc.linearise(table["tau"], tau_inf) * 100.0 * DEPOSIT_AREA_CM2 / table["volume_m3"]
        needed = float(np.nanmedian(addis_fabs / table[PUBLISHED]))
        anchors.append({"anchor": f"Linearised HIPS at the Adama MAC (m²/g), ceiling {tau_inf:.2f}",
                        "published_over_true": adama_mac / needed, "value": adama_mac,
                        "note": f"Addis needs {needed:.1f} m²/g for agreement; five Adama filters set the MAC"})
    return {"improve_ratio_by_ocec": pd.DataFrame(by_ocec), "mass_closure": pd.DataFrame(closure),
            "level_anchors": pd.DataFrame(anchors),
            "addis_ocec": pd.DataFrame([{"n": len(addis_ocec), "median": addis_ocec.median(),
                                         "p05": addis_ocec.quantile(0.05), "p95": addis_ocec.quantile(0.95)}])}


def stage_product(table, closeout, n_boot, *, level=(1.0, 1.2, 1.5)):
    """Per-filter values to carry forward: FTIR EC unchanged with a level range, and HIPS before and after linearising.

    The level range is the span the stage ``level`` anchors allow, not a fitted correction: ``ftir_ec_low`` is
    the published value, ``ftir_ec_high`` the top of the range. Nothing here uses an optical instrument to set
    the FTIR level. The HIPS columns use the saturation constant fitted with Addis held out.
    """
    stress = closeout["settings"]["stress"]
    out = table[["date", "season", "lot", "volume_m3", "volume_known", "aeth_ir", "fabs", "tau"]].copy()
    out = out.rename(columns={"aeth_ir": "aethalometer_ir_bcc_ugm3", "fabs": "hips_fabs_Mm1", "tau": "hips_optical_depth"})
    out["ftir_ec_published_ugm3"] = table[PUBLISHED]
    out["ftir_ec_anchored_ugm3"] = table[ANCHORED]
    low, central, high = level
    for name, factor in (("low", low), ("central", central), ("high", high)):
        out[f"ftir_ec_{name}_ugm3"] = table[PUBLISHED] * factor
    back = 100.0 * DEPOSIT_AREA_CM2 / table["volume_m3"]
    for name, tau_inf in (("central", stress["curve"]["tau_inf"]), ("lo", stress["tau_inf_boot"][0]), ("hi", stress["tau_inf_boot"][2])):
        out[f"hips_fabs_linearised_{name}_Mm1"] = rc.linearise(table["tau"], tau_inf) * back
    return {"addis_per_filter": out.reset_index()}


def stage_network(table, closeout, n_boot):
    """HIPS against FTIR EC at every SPARTAN site, with gravimetric PM2.5 as the instrument.

    No aethalometer is needed, so this covers the network. It is weaker than the Addis test: PM2.5 is a good
    instrument only where EC is a steady share of it, and the intervals say where it is not (dust and sulfate
    sites). An exponent below 1 says HIPS and FTIR EC are not proportional there; with two EC-like measures it
    cannot by itself say which one bends. Optical depth uses the 3.53 cm² deposit area throughout.
    """
    rows = pd.DataFrame(json.loads(NETWORK.read_text())["rows"])
    rows = rows[~rows["excluded"].astype(bool)].copy()
    rows["fabs"] = rows["HIPS Fabs"].fillna(rows["BC (public HIPS)"] * MAC_VALUE)
    rows["tau"] = rows["fabs"] * rows["Sampled volume"] / (100.0 * DEPOSIT_AREA_CM2)
    out = []
    for code, part in rows.groupby("code"):
        part = part[["fabs", "EC (FTIR)", "PM2.5 mass", "tau"]].dropna()
        part = part[(part[["fabs", "EC (FTIR)", "PM2.5 mass"]] > 0).all(axis=1)]
        if len(part) < NETWORK_MIN_FILTERS:
            continue
        r = rc.response_exponent(part["fabs"], part["EC (FTIR)"], part["PM2.5 mass"], n_boot=min(n_boot, 500))
        out.append({"code": code, "n": r["n"], "median_optical_depth": float(part["tau"].median()),
                    "p90_optical_depth": float(part["tau"].quantile(0.9)),
                    "hips_exponent_vs_ftir_ec": r["exponent"], "lo": r["lo"], "hi": r["hi"]})
    return {"network_exponents": pd.DataFrame(out).sort_values("median_optical_depth")}


RUN = {"network": stage_network, "linearity": stage_linearity, "triple": stage_triple, "response": stage_response,
       "season": stage_season, "level": stage_level, "product": stage_product}


def summarise(tables: dict[str, pd.DataFrame], table: pd.DataFrame) -> dict:
    """The numbers the write-up quotes, so the doc can be checked against one file."""
    def records(name):
        return json.loads(tables[name].to_json(orient="records", double_precision=4)) if name in tables else None
    return {
        "generated_from": "research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py",
        "inputs": {"closeout": str(CLOSEOUT.relative_to(ROOT)), "filters": str(FILTERS.relative_to(ROOT)),
                   "unified": str(UNIFIED.relative_to(ROOT))},
        "counts": {"filters": int(len(table)), "with_aethalometer": int(table["aeth_ir"].notna().sum()),
                   "with_pm25": int(table["pm25"].notna().sum()), "with_published": int(table[PUBLISHED].notna().sum()),
                   "volume_known": int(table["volume_known"].sum())},
        **{name: records(name) for name in ("linearity", "triple_collocation", "hips_response_fits", "linearised_hips",
                                            "raw_hips", "season_lot", "improve_ratio_by_ocec", "mass_closure",
                                            "level_anchors", "addis_ocec", "network_exponents")},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--known-volume-only", action="store_true", help="drop filters whose sampled volume is the site median")
    parser.add_argument("--summary", nargs="?", const=SUMMARY, type=Path, help="also write the quoted numbers as JSON")
    args = parser.parse_args(argv)

    table, closeout = load()
    if args.known_volume_only:
        table = table[table["volume_known"]]
    args.output.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    for stage in args.stages:
        tables |= RUN[stage](table, closeout, args.bootstrap)
    for name, frame in tables.items():
        frame.to_csv(args.output / f"{name}.csv", index=False)
        print(f"{name}: {len(frame)} rows")
    if args.summary:
        args.summary.write_text(json.dumps(summarise(tables, table), indent=1) + "\n")
        print(f"summary: {args.summary.relative_to(ROOT) if args.summary.is_relative_to(ROOT) else args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
