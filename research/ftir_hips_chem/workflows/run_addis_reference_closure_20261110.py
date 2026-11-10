"""Which measurement carries the Addis offset, and what correction the existing data support (10 Nov 2026).

The 9 Nov status page left this open: "Which measurement carries the offset stays open". FTIR EC against HIPS
has a slope near 2 and an intercept near -6 µg/m³. Two instruments cannot settle which one bends.

**The aethalometer is not used.** The Addis MA350 never had its flow split fixed and is not a reference there,
so nothing in the default run reads an aethalometer value, at Addis or anywhere else (the 3 Nov saturation
ceiling was fitted on aethalometers at three other sites, so it is not used either). The argument rests on:

* thermal EC: the IMPROVE record of TOR EC and HIPS on the same sample (stage ``improve``), which says how HIPS
  responds to EC as the filter gets darker, with no FTIR and no aethalometer;
* two measurements on the Addis filters that are neither optical nor FTIR: gravimetric PM2.5 and XRF potassium.

Inputs. Everything but the IMPROVE tables is committed, so the other stages run from a plain checkout:

* ``gallery/review/closeout_20261107/continuum_baseline.json``  per-filter Addis predictions of every saved
  candidate, HIPS BC, and held-out-site IMPROVE predictions with TOR (7 Nov closeout)
* ``gallery/app/public/data/filters.json``                       PM2.5 mass, elements, ions, FTIR OC and OM
* ``Filter Data/unified_filter_dataset.pkl``                      sampled volume (optical depth = Fabs V / 100 A)
* ``gallery/app/public/data/spartan_reference.json``             every SPARTAN site, for the ``network`` stage
* ``<FTIR local_db>/tables/results_{tor,hips}.csv``              IMPROVE TOR and HIPS (Drive only; data_paths.ftir_local_db).
  Where they are absent, ``improve`` reuses the stratum table saved in the summary JSON.

Stages (default: all)

* ``improve``    HIPS optical depth against thermal EC loading across IMPROVE, and the saturation ceiling
* ``linearity``  response exponent of each Addis measurement against each other one, with a third as the instrument
* ``response``   the shape of the HIPS response at Addis, and FTIR against HIPS linearised with the IMPROVE ceiling
* ``season``     FTIR / linearised HIPS and FTIR / PM2.5 by season and filter lot
* ``level``      what bounds the absolute FTIR level: held-out IMPROVE filters by OC/EC, mass closure, MAC anchors
* ``product``    per-filter table: FTIR EC with a level range, and HIPS before and after linearising
* ``network``    the same exponent at every SPARTAN site (PM2.5 mass is the instrument)

    uv run python research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py --summary

``--with-aethalometer`` adds the aethalometer rows and a triple collocation as a side check, labelled as such.
Writes ``output/tables/addis_reference_closure_20261110/`` (git-ignored) and, with ``--summary``, the numbers
quoted in docs/addis-reference-closure-2026-11-10.md. Nothing here is fitted to make FTIR agree with an optical
instrument, and nothing here measures thermal EC at Addis: the level stays a range.
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
import data_paths  # noqa: E402
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
    "aeth_ir": "Aethalometer IR BCc (side check)", "pm25": "PM2.5 mass (gravimetric)", "potassium": "Potassium (XRF)",
}
STAGES = ("improve", "linearity", "response", "season", "level", "product", "network")
NETWORK_MIN_FILTERS = 40
# IMPROVE strata: the previous sample's TOR EC loading at the same site (µg per filter)
IMPROVE_EDGES = (0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 11, 15, 20, 27, 36, 50, 70, 100, 1e4)
IMPROVE_MIN_STRATUM = 150
IMPROVE_POPULATIONS = ("all sites", "sites with median OC/EC < 4", "six urban sites")
IMPROVE_PRIMARY = "all sites"
URBAN = ("ATLA1", "BIRM1", "FRES1", "PHOE1", "PITT1", "PUSO1")       # Ann's urban IMPROVE sites (28 Oct)
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


# --------------------------------------------------------------------------- IMPROVE thermal EC
def improve_samples() -> pd.DataFrame | None:
    """One row per IMPROVE sample with TOR EC and HIPS: EC loading (µg per filter) and HIPS optical depth."""
    tables = data_paths.ftir_local_db() / "tables"
    if not (tables / "results_tor.csv").is_file() or not (tables / "results_hips.csv").is_file():
        return None

    def read(name, parameters):
        frame = pd.read_csv(tables / name, encoding="utf-8-sig", low_memory=False,
                            usecols=["Site", "SampleDate", "Parameter", "Value", "AverageFlowRate", "ElapsedTime"])
        frame = frame[frame["Parameter"].isin(parameters)].copy()
        frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
        wide = frame.pivot_table(index=["Site", "SampleDate"], columns="Parameter", values="Value", aggfunc="first")
        volume = frame.groupby(["Site", "SampleDate"])[["AverageFlowRate", "ElapsedTime"]].first()
        wide["volume_m3"] = volume["AverageFlowRate"] * volume["ElapsedTime"] / 1000.0
        return wide

    tor, hips = read("results_tor.csv", ["EC", "OC"]), read("results_hips.csv", ["fAbs"])
    both = tor.join(hips, how="inner", lsuffix="_quartz", rsuffix="_ptfe").reset_index()
    both["date"] = pd.to_datetime(both["SampleDate"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    ok = (both["EC"] > 0) & (both["fAbs"] > 0) & both["date"].notna()
    ok &= both["volume_m3_quartz"].between(20, 45) & both["volume_m3_ptfe"].between(20, 45)
    both = both[ok].sort_values(["Site", "date"]).reset_index(drop=True)
    both["loading_ug"] = both["EC"] / 1000.0 * both["volume_m3_quartz"]           # TOR values are ng/m3
    both["tau"] = both["fAbs"] * both["volume_m3_ptfe"] / (100.0 * DEPOSIT_AREA_CM2)
    both["ocec"] = both["OC"] / both["EC"]
    previous = both.groupby("Site")["loading_ug"].shift(1)
    gap = (both["date"] - both.groupby("Site")["date"].shift(1)).dt.days
    both["previous_loading_ug"] = previous.where(gap.between(1, 7))
    return both.dropna(subset=["previous_loading_ug"])


def _strata(samples: pd.DataFrame) -> pd.DataFrame:
    groups = samples.groupby(np.digitize(samples["previous_loading_ug"], IMPROVE_EDGES))
    table = pd.DataFrame({"n": groups.size(), "loading_ug": groups["loading_ug"].median(), "tau": groups["tau"].median()})
    return table[table["n"] >= IMPROVE_MIN_STRATUM]


def _ceiling(strata: pd.DataFrame) -> tuple[float, float]:
    from scipy import optimize
    residual = lambda p: np.log(rc.hyperbolic(strata["loading_ug"], *p)) - np.log(strata["tau"])  # noqa: E731
    a, tau_inf = optimize.least_squares(residual, [0.04, 2.0], bounds=([0.0, 0.2], [1.0, 100.0])).x
    return float(a), float(tau_inf)


def stage_improve(table, closeout, n_boot, tables, aethalometer):
    """How HIPS responds to thermal EC as the filter darkens, from IMPROVE alone.

    Each sample is placed in a stratum by the *previous* sample's TOR EC loading at the same site. That
    ordering is unrelated to this sample's measurement errors in either instrument, so the stratum medians of
    HIPS optical depth and TOR EC loading are free of the bias that binning on TOR or on HIPS would cause. A
    saturating curve through the medians gives the ceiling; the interval resamples whole sites.
    """
    samples = improve_samples()
    if samples is None:
        if not SUMMARY.is_file():
            raise FileNotFoundError("IMPROVE tables not found and no saved summary: set AETHMODULAR_FTIR_LOCAL_DB (uv run aeth drive mirror)")
        saved = json.loads(SUMMARY.read_text())
        print("improve: IMPROVE tables not found; reusing the stratum table and ceilings saved in the summary")
        return {"improve_strata": pd.DataFrame(saved["improve_strata"]), "improve_ceiling": pd.DataFrame(saved["improve_ceiling"])}
    site_ocec = samples.groupby("Site")["ocec"].median()
    populations = {"all sites": samples, "sites with median OC/EC < 4": samples[samples["Site"].map(site_ocec) < 4],
                   "six urban sites": samples[samples["Site"].isin(URBAN)]}
    rng = np.random.default_rng(20261110)
    strata_rows, ceilings = [], []
    for name, part in populations.items():
        strata = _strata(part)
        a, tau_inf = _ceiling(strata)
        by_site = {site: frame for site, frame in part.groupby("Site")}
        sites = np.array(list(by_site))
        boot = np.array([_ceiling(_strata(pd.concat([by_site[s] for s in rng.choice(sites, sites.size)])))
                         for _ in range(min(n_boot, 200))])
        (a_lo, a_hi), (t_lo, t_hi) = (np.percentile(boot[:, i], [5, 95]) for i in (0, 1))
        to_mac = 100.0 * DEPOSIT_AREA_CM2
        ceilings.append({"population": name, "samples": len(part), "sites": int(sites.size), "strata": len(strata),
                         "years": f"{part['date'].dt.year.min()}-{part['date'].dt.year.max()}",
                         "tau_inf": tau_inf, "tau_inf_lo": t_lo, "tau_inf_hi": t_hi,
                         "unsaturated_fabs_over_ec": a * to_mac, "unsaturated_lo": a_lo * to_mac, "unsaturated_hi": a_hi * to_mac,
                         "max_stratum_tau": float(strata["tau"].max())})
        strata_rows.append(strata.assign(population=name, fabs_over_ec=lambda d: to_mac * d["tau"] / d["loading_ug"],
                                         fitted_tau=lambda d: rc.hyperbolic(d["loading_ug"], a, tau_inf)).reset_index(drop=True))
    return {"improve_strata": pd.concat(strata_rows, ignore_index=True), "improve_ceiling": pd.DataFrame(ceilings)}


def thermal_ceiling(tables) -> dict[str, float]:
    """The IMPROVE ceiling used to linearise HIPS everywhere below: central value and 90 % interval."""
    row = tables["improve_ceiling"].set_index("population").loc[IMPROVE_PRIMARY]
    return {"central": float(row["tau_inf"]), "lo": float(row["tau_inf_lo"]), "hi": float(row["tau_inf_hi"])}


# --------------------------------------------------------------------------- stages
def stage_linearity(table, closeout, n_boot, tables, aethalometer):
    """Who is proportional to whom at Addis. b = 1 is proportional; the instrument Z removes the flattening by noise in X.

    Without a BC-specific reference this cannot by itself say which of HIPS and FTIR EC bends: it says they are
    not proportional to each other, and how each scales with gravimetric mass. The IMPROVE stage supplies the rest.
    The Frisch bounds need no instrument at all: with independent errors the true exponent lies between the
    log-log slope of Y on X and the reciprocal of the slope of X on Y.
    """
    plan = [
        ("fabs", PUBLISHED, ("pm25", "potassium")), ("fabs", ANCHORED, ("pm25", "potassium")),
        ("fabs", "pm25", (PUBLISHED, "potassium")), (PUBLISHED, "pm25", ("fabs", "potassium")),
    ]
    if aethalometer:
        plan += [("fabs", "aeth_ir", ("pm25", "potassium")), (PUBLISHED, "aeth_ir", ("pm25", "potassium"))]
    rows = []
    for y, x, instruments in plan:
        for z in instruments:
            r = rc.response_exponent(table[y], table[x], table[z], n_boot=n_boot)
            rows.append({"y": LABEL[y], "x": LABEL[x], "instrument": LABEL[z], "group": "all", **r})
    for season, part in table.groupby("season"):
        r = rc.response_exponent(part["fabs"], part[PUBLISHED], part["pm25"], n_boot=n_boot)
        rows.append({"y": LABEL["fabs"], "x": LABEL[PUBLISHED], "instrument": LABEL["pm25"], "group": season, **r})
    bounds = []
    for model in (PUBLISHED, ANCHORED):
        pair = np.log(table[["fabs", model]].where(table[["fabs", model]] > 0)).dropna()
        forward = np.polyfit(pair[model], pair["fabs"], 1)[0]
        reverse = 1.0 / np.polyfit(pair["fabs"], pair[model], 1)[0]
        bounds.append({"y": LABEL["fabs"], "x": LABEL[model], "n": len(pair), "lower": forward, "upper": reverse})
    return {"linearity": pd.DataFrame(rows), "frisch_bounds": pd.DataFrame(bounds)}


def stage_response(table, closeout, n_boot, tables, aethalometer):
    """HIPS optical depth against each linear loading at Addis, and FTIR against HIPS linearised with the IMPROVE ceiling."""
    references = {PUBLISHED: "FTIR EC, SPARTAN published", ANCHORED: "FTIR EC, PTFE-anchored VIBES", "pm25": "PM2.5 mass"}
    fits = []
    for column, name in references.items():
        for model in rc.RESPONSE_MODELS:
            f = rc.fit_response(per_filter(table, column), table["tau"], model, n_boot=min(n_boot, 300))
            row = {"loading": name, "model": model, "n": f["n"], "rms_log": f["rms_log"], "aic": f["aic"]}
            for i, (value, (lo, hi)) in enumerate(zip(f["params"], f["intervals"]), start=1):
                row |= {f"p{i}": value, f"p{i}_lo": lo, f"p{i}_hi": hi}
            fits.append(row)
    fits = pd.DataFrame(fits)
    fits["delta_aic_vs_proportional"] = fits["aic"] - fits.groupby("loading")["aic"].transform(
        lambda s: s[fits.loc[s.index, "model"] == "proportional"].iloc[0])

    ceiling = thermal_ceiling(tables)
    lam = float(closeout["settings"]["LAMBDA_ADDIS"])
    rows = []
    for which, tau_inf in ceiling.items():
        linear_fabs = rc.linearise(table["tau"], tau_inf) * 100.0 * DEPOSIT_AREA_CM2 / table["volume_m3"]
        usable = np.isfinite(linear_fabs)
        # what the IMPROVE curve predicts for the exponent of HIPS against a proportional EC measure, on these filters
        predicted = float(np.nanmean(1.0 - table["tau"] / tau_inf))
        for model in (PUBLISHED, ANCHORED):
            d = deming_regression(linear_fabs[usable] / MAC_VALUE, table.loc[usable, model], lam)
            e = rc.response_exponent(linear_fabs, table[model], table["pm25"], n_boot=min(n_boot, 500))
            rows.append({"tau_inf": tau_inf, "which": which, "model": LABEL[model], "n": d["n"],
                         "slope_vs_linearised_hips": d["slope"], "intercept_vs_linearised_hips": d["intercept"],
                         "predicted_hips_exponent": predicted,
                         "linearised_exponent_vs_ftir": e["exponent"], "linearised_exponent_lo": e["lo"], "linearised_exponent_hi": e["hi"],
                         "median_correction_factor": float(np.nanmedian(linear_fabs / table["fabs"])),
                         "mac_for_agreement": float(np.nanmedian(linear_fabs / table[model]))})
    raw = []
    for model in (PUBLISHED, ANCHORED):
        d = deming_regression(table["hips_bc"], table[model], lam)
        raw.append({"model": LABEL[model], "n": d["n"], "slope_vs_hips": d["slope"], "intercept_vs_hips": d["intercept"]})
    return {"hips_response_fits": fits, "linearised_hips": pd.DataFrame(rows), "raw_hips": pd.DataFrame(raw)}


def stage_season(table, closeout, n_boot, tables, aethalometer):
    """Is there a season effect in FTIR once HIPS is linearised? Median ratios by season and filter lot.

    FTIR / PM2.5 is shown beside it, but EC's share of PM2.5 can really change with season, so it is context
    and not a test.
    """
    tau_inf = thermal_ceiling(tables)["central"]
    linear_bc = rc.linearise(table["tau"], tau_inf) * 100.0 * DEPOSIT_AREA_CM2 / table["volume_m3"] / MAC_VALUE
    rows = []
    for keys, part in [(("all", "all"), table), *[((s, "all"), q) for s, q in table.groupby("season")],
                       *[((s, str(int(lot))), q) for (s, lot), q in table.groupby(["season", "lot"])]]:
        n = int(part[PUBLISHED].notna().sum())
        if n < 5:
            continue
        row = {"season": keys[0], "lot": keys[1], "n": n,
               "published_over_linearised_hips": float((part[PUBLISHED] / linear_bc[part.index]).median()),
               "published_over_hips": float((part[PUBLISHED] / part["hips_bc"]).median()),
               "published_over_pm25": float((part[PUBLISHED] / part["pm25"]).median()),
               "median_optical_depth": float(part.loc[part[PUBLISHED].notna(), "tau"].median())}
        if aethalometer:
            row["published_over_aethalometer"] = float((part[PUBLISHED] / part["aeth_ir"]).median())
        rows.append(row)
    return {"season_lot": pd.DataFrame(rows)}


def stage_level(table, closeout, n_boot, tables, aethalometer):
    """Evidence on the absolute FTIR level. None of it is thermal EC measured at Addis, and none uses an aethalometer."""
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

    anchors = [
        {"anchor": "Held-out IMPROVE sites, TOR ≥ 15 µg and OC/EC < 2.5 (FTIR ÷ TOR, raw spectra, 18 components)",
         "published_over_true": np.nan, "value": float(np.median(improve.query("tor_ug >= 15 and ocec < 2.5").eval("`raw k18` / tor_ug"))),
         "note": "calibration set, not Addis; Addis FTIR OC/EC is lower still"},
        {"anchor": "Same, OC/EC < 1.8", "published_over_true": np.nan,
         "value": float(np.median(improve.query("tor_ug >= 15 and ocec < 1.8").eval("`raw k18` / tor_ug"))),
         "note": f"n {len(improve.query('tor_ug >= 15 and ocec < 1.8'))}"},
    ]
    # MAC closures: the MAC that Addis FTIR EC needs once HIPS is linearised, against a MAC measured with thermal EC.
    # Both sides use the same ceiling, so each row pairs them.
    ceiling = thermal_ceiling(tables)
    pilot = pd.DataFrame(closeout["adama"])
    unsaturated = tables["improve_ceiling"].set_index("population")
    for which, tau_inf in ceiling.items():
        addis_fabs = rc.linearise(table["tau"], tau_inf) * 100.0 * DEPOSIT_AREA_CM2 / table["volume_m3"]
        needed = float(np.nanmedian(addis_fabs / table[PUBLISHED]))
        adama_mac = float(np.median(pilot["fabs_over_tor"] * rc.linearise(pilot["tau"], tau_inf) / pilot["tau"]))
        anchors.append({"anchor": f"Linearised HIPS at the Adama MAC (m²/g), ceiling {tau_inf:.2f}",
                        "published_over_true": adama_mac / needed, "value": adama_mac,
                        "note": f"Addis needs {needed:.1f} m²/g for agreement; five Adama filters, another site and sampler"})
        if which == "central":
            for population in IMPROVE_POPULATIONS:
                mac = float(unsaturated.loc[population, "unsaturated_fabs_over_ec"])
                anchors.append({"anchor": f"Linearised HIPS at the IMPROVE unsaturated MAC (m²/g), {population}",
                                "published_over_true": mac / needed, "value": mac,
                                "note": f"Addis needs {needed:.1f} m²/g; assumes Addis soot absorbs like IMPROVE's per unit EC"})
    return {"improve_ratio_by_ocec": pd.DataFrame(by_ocec), "mass_closure": pd.DataFrame(closure),
            "level_anchors": pd.DataFrame(anchors),
            "addis_ocec": pd.DataFrame([{"n": len(addis_ocec), "median": addis_ocec.median(),
                                         "p05": addis_ocec.quantile(0.05), "p95": addis_ocec.quantile(0.95)}])}


def stage_product(table, closeout, n_boot, tables, aethalometer, *, level=(1.0, 1.2, 1.5)):
    """Per-filter values to carry forward: FTIR EC unchanged with a level range, and HIPS before and after linearising.

    The level range is the span the stage ``level`` anchors allow, not a fitted correction: ``ftir_ec_low`` is
    the published value, ``ftir_ec_high`` the top of the range. Nothing here uses an optical instrument to set
    the FTIR level. The HIPS columns use the ceiling fitted on IMPROVE thermal EC; no aethalometer value is written.
    """
    out = table[["date", "season", "lot", "volume_m3", "volume_known", "fabs", "tau"]].copy()
    out = out.rename(columns={"fabs": "hips_fabs_Mm1", "tau": "hips_optical_depth"})
    out["ftir_ec_published_ugm3"] = table[PUBLISHED]
    out["ftir_ec_anchored_ugm3"] = table[ANCHORED]
    for name, factor in zip(("low", "central", "high"), level):
        out[f"ftir_ec_{name}_ugm3"] = table[PUBLISHED] * factor
    back = 100.0 * DEPOSIT_AREA_CM2 / table["volume_m3"]
    for name, tau_inf in thermal_ceiling(tables).items():
        out[f"hips_fabs_linearised_{name}_Mm1"] = rc.linearise(table["tau"], tau_inf) * back
    return {"addis_per_filter": out.reset_index()}


def stage_network(table, closeout, n_boot, tables, aethalometer):
    """HIPS against FTIR EC at every SPARTAN site, with gravimetric PM2.5 as the instrument.

    PM2.5 is a good instrument only where EC is a steady share of it, and the intervals say where it is not
    (dust and sulfate sites). An exponent below 1 says HIPS and FTIR EC are not proportional there; beside it
    is what the IMPROVE ceiling predicts for HIPS alone at that site's optical depths. Deposit area 3.53 cm².
    """
    tau_inf = thermal_ceiling(tables)["central"]
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
                    "hips_exponent_vs_ftir_ec": r["exponent"], "lo": r["lo"], "hi": r["hi"],
                    "predicted_from_improve": float(np.mean(np.clip(1.0 - part["tau"] / tau_inf, 0.0, 1.0)))})
    return {"network_exponents": pd.DataFrame(out).sort_values("median_optical_depth")}


def stage_aethalometer(table, closeout, n_boot, tables, aethalometer):
    """Side check only (``--with-aethalometer``): three-way random error. The Addis MA350 is not a reference."""
    rows = []
    result = rc.triple_collocation(table[PUBLISHED], table["aeth_ir"], table["fabs"], n_boot=n_boot)
    for column, m in zip((PUBLISHED, "aeth_ir", "fabs"), result["measurements"]):
        rows.append({"measurement": LABEL[column], "n": result["n"], "signal_sd": result["signal_sd"], **m})
    return {"aethalometer_triple_collocation": pd.DataFrame(rows)}


RUN = {"improve": stage_improve, "linearity": stage_linearity, "response": stage_response, "season": stage_season,
       "level": stage_level, "product": stage_product, "network": stage_network}
SUMMARY_TABLES = ("improve_ceiling", "improve_strata", "linearity", "frisch_bounds", "hips_response_fits", "linearised_hips",
                  "raw_hips", "season_lot", "improve_ratio_by_ocec", "mass_closure", "level_anchors", "addis_ocec",
                  "network_exponents")


def run(stages=STAGES, *, n_boot=2000, aethalometer=False, known_volume_only=False):
    """Run the stages in order and return (tables, Addis table). ``improve`` always runs first: the rest need its ceiling."""
    table, closeout = load()
    if known_volume_only:
        table = table[table["volume_known"]]
    tables: dict[str, pd.DataFrame] = {}
    for stage in dict.fromkeys(("improve", *stages)):
        tables |= RUN[stage](table, closeout, n_boot, tables, aethalometer)
    if aethalometer:
        tables |= stage_aethalometer(table, closeout, n_boot, tables, aethalometer)
    return tables, table


def summarise(tables: dict[str, pd.DataFrame], table: pd.DataFrame, aethalometer: bool) -> dict:
    """The numbers the write-up quotes, so the doc can be checked against one file."""
    return {
        "generated_from": "research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py",
        "aethalometer_used": bool(aethalometer),
        "inputs": {"closeout": str(CLOSEOUT.relative_to(ROOT)), "filters": str(FILTERS.relative_to(ROOT)),
                   "unified": str(UNIFIED.relative_to(ROOT)), "network": str(NETWORK.relative_to(ROOT)),
                   "improve": "<FTIR local_db>/tables/results_tor.csv, results_hips.csv (Drive)"},
        "counts": {"filters": int(len(table)), "with_pm25": int(table["pm25"].notna().sum()),
                   "with_published": int(table[PUBLISHED].notna().sum()), "volume_known": int(table["volume_known"].sum())},
        **{name: json.loads(tables[name].to_json(orient="records", double_precision=4)) for name in SUMMARY_TABLES if name in tables},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--known-volume-only", action="store_true", help="drop filters whose sampled volume is the site median")
    parser.add_argument("--with-aethalometer", action="store_true", help="add the aethalometer rows as a side check (not a reference at Addis)")
    parser.add_argument("--summary", nargs="?", const=SUMMARY, type=Path, help="also write the quoted numbers as JSON")
    args = parser.parse_args(argv)

    tables, table = run(args.stages, n_boot=args.bootstrap, aethalometer=args.with_aethalometer, known_volume_only=args.known_volume_only)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(args.output / f"{name}.csv", index=False)
        print(f"{name}: {len(frame)} rows")
    if args.summary:
        if args.with_aethalometer and args.summary == SUMMARY:
            raise SystemExit("the committed summary is the aethalometer-free run; pass another --summary path with --with-aethalometer")
        args.summary.write_text(json.dumps(summarise(tables, table, args.with_aethalometer), indent=1) + "\n")
        print(f"summary: {args.summary.relative_to(ROOT) if args.summary.is_relative_to(ROOT) else args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
