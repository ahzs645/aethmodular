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
* ``ftir``       the FTIR side against thermal EC: a raw-spectra EC and OC calibration refitted with whole sites held
                 out, the level bias under groupings that do not select on a filter's own error, and how far
                 Addis sits outside IMPROVE's composition (needs the Drive spectra; reuses the summary otherwise)
* ``level``      what bounds the absolute FTIR level: the fair IMPROVE groupings, mass closure, MAC anchors
* ``product``    per-filter table: FTIR EC with a level range, and HIPS before and after linearising
* ``network``    every SPARTAN site: the exponent, and FTIR against HIPS before and after linearising, per filter
* ``candidates`` every saved Addis calibration by season, against HIPS as measured and linearised
* ``source``     is FTIR proportional to thermal EC on held-out IMPROVE sites, and does filter lot matter there

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
STAGES = ("improve", "linearity", "response", "season", "ftir", "level", "product", "network", "candidates", "source")
PLS_COMPONENTS = 18              # as the closeout's fixed-k comparison
FTIR_MIN_EC_UG = 5.0             # FTIR OC/EC is only read where predicted EC is clear of zero
NETWORK_MIN_FILTERS = 40
# IMPROVE strata: the previous sample's TOR EC loading at the same site (µg per filter)
IMPROVE_EDGES = (0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 11, 15, 20, 27, 36, 50, 70, 100, 1e4)
IMPROVE_MIN_STRATUM = 150
IMPROVE_POPULATIONS = ("all sites", "sites with median OC/EC < 4", "six urban IMPROVE sites")
IMPROVE_CHECKS = ("2005–2014", "2015–2025", "optical pyrolysis > 0")   # robustness, all sites
SITE_MAP = ROOT / "gallery/app/public/data/spartan_map.json"
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

    tor, hips = read("results_tor.csv", ["EC", "OC", "OPTR"]), read("results_hips.csv", ["fAbs"])
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
    year = samples["date"].dt.year
    populations = {"all sites": samples, "sites with median OC/EC < 4": samples[samples["Site"].map(site_ocec) < 4],
                   "six urban IMPROVE sites": samples[samples["Site"].isin(URBAN)],
                   "2005–2014": samples[year <= 2014], "2015–2025": samples[year >= 2015],
                   "optical pyrolysis > 0": samples[samples["OPTR"] > 0]}
    rng = np.random.default_rng(20261110)
    strata_rows, ceilings = [], []
    for name, part in populations.items():
        strata = _strata(part)
        a, tau_inf = _ceiling(strata)
        by_site = {site: frame for site, frame in part.groupby("Site")}
        sites = np.array(list(by_site))
        boot = np.array([_ceiling(_strata(pd.concat([by_site[s] for s in rng.choice(sites, sites.size)])))
                         for _ in range(min(n_boot, 200 if name in IMPROVE_POPULATIONS else 60))])
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


def _ftir_refit() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Raw-spectra PLS for TOR EC and TOR OC (µg per filter), whole IMPROVE sites held out, and the same models on Addis.

    Fitting OC beside EC gives an FTIR OC/EC that means the same thing on IMPROVE and on Addis. One row per
    physical filter. TOR EC below 150 µg per filter, as in the closeout; no other exclusion is applied here.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold

    local = data_paths.ftir_local_db()
    etad = data_paths.etad_dir()
    spectra_path = local / "spectra_248_251.csv"
    if not spectra_path.is_file() or not (etad / "ETAD_FTIR_spectra.csv").is_file():
        return None
    spectra = pd.read_csv(spectra_path, dtype={"Site": str, "SampleDate": str})
    channels = [c for c in spectra.columns if c not in ("AnalysisId", "FilterId", "SampleDate", "Site")]
    tor = pd.read_csv(local / "tables/results_tor.csv", encoding="utf-8-sig", low_memory=False,
                      usecols=["Site", "SampleDate", "Parameter", "Value", "AverageFlowRate", "ElapsedTime"])
    tor = tor[tor["Parameter"].isin(["EC", "OC"])].copy()
    tor["Value"] = pd.to_numeric(tor["Value"], errors="coerce")
    tor["day"] = pd.to_datetime(tor["SampleDate"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce").dt.strftime("%Y-%m-%d")
    wide = tor.pivot_table(index=["Site", "day"], columns="Parameter", values="Value", aggfunc="first")
    flow = tor.groupby(["Site", "day"])[["AverageFlowRate", "ElapsedTime"]].first()
    volume = flow["AverageFlowRate"] * flow["ElapsedTime"] / 1000.0
    wide["ec_ug"], wide["oc_ug"] = wide["EC"] / 1000.0 * volume, wide["OC"] / 1000.0 * volume    # TOR values are ng/m3
    spectra["day"] = pd.to_datetime(spectra["SampleDate"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce").dt.strftime("%Y-%m-%d")
    pool = spectra[["FilterId", "Site", "day"]].join(wide[["ec_ug", "oc_ug"]], on=["Site", "day"])
    keep = pool["ec_ug"].between(0, 150, inclusive="neither") & (pool["oc_ug"] > 0) & np.isfinite(spectra[channels]).all(axis=1)
    pool = pool[keep].drop_duplicates("FilterId")
    x = spectra.loc[pool.index, channels].to_numpy(float)
    pool = pool.reset_index(drop=True)

    raw = pd.read_csv(etad / "ETAD_FTIR_spectra.csv")
    etad_channels = [c for c in raw.columns if c not in ("SampleAnalysisId", "MediaId")]
    if not np.allclose(np.array(etad_channels, float), np.array(channels, float)):
        raise ValueError("Addis and IMPROVE spectra are on different wavenumber grids")
    per_filter = raw.groupby("MediaId")[etad_channels].mean()                 # repeat scans of one filter averaged
    meta = pd.read_csv(etad / "ETAD_metadata.csv").set_index("MediaId").loc[per_filter.index]
    addis = pd.DataFrame({"filter_id": meta["ExternalFilterId"].str.replace(BASE_FILTER_ID_PATTERN, BASE_FILTER_ID_REPL, regex=True).to_numpy(),
                          "type": meta["ExternalFilterType"].to_numpy(), "volume_m3": meta["SampleVolume_m3"].to_numpy()})
    folds = list(GroupKFold(5).split(x, groups=pool["Site"]))
    for target in ("ec_ug", "oc_ug"):
        y = pool[target].to_numpy()
        held_out = np.full(y.size, np.nan)
        for train, test in folds:
            held_out[test] = PLSRegression(PLS_COMPONENTS, scale=False).fit(x[train], y[train]).predict(x[test]).ravel()
        pool[f"ftir_{target}"] = held_out
        addis[f"ftir_{target}"] = PLSRegression(PLS_COMPONENTS, scale=False).fit(x, y).predict(per_filter.to_numpy(float)).ravel()
    return pool, addis[addis["type"].eq("PM2.5")].drop(columns="type")


def stage_ftir(table, closeout, n_boot, tables, aethalometer):
    """The FTIR side, against thermal EC, and how much of it IMPROVE can say anything about.

    The trap this stage exists for: grouping IMPROVE filters by their own TOR EC or TOR OC/EC selects on TOR's
    measurement error, so "FTIR ÷ TOR on heavy, low-OC/EC filters" reads lower than the truth (and grouping by
    FTIR's own OC/EC reads higher). Two groupings cannot do that: by site, and by the previous sample at the
    site. All three are reported so the size of the artefact is visible.
    """
    names = ("ftir_level", "ftir_candidates", "ftir_composition", "ftir_ocec_values", "addis_refit", "ftir_by_site", "ftir_selection")
    refit = _ftir_refit()
    if refit is None:
        if not SUMMARY.is_file():
            raise FileNotFoundError("IMPROVE spectra not found and no saved summary: set AETHMODULAR_FTIR_LOCAL_DB and AETHMODULAR_ETAD_DIR")
        saved = json.loads(SUMMARY.read_text())
        print("ftir: spectra not found; reusing the tables saved in the summary")
        return {name: pd.DataFrame(saved[name]) for name in names}
    pool, addis = refit
    pool["date"] = pd.to_datetime(pool["day"])
    pool = pool.sort_values(["Site", "date"]).reset_index(drop=True)
    pool["ocec"], pool["ratio"] = pool["oc_ug"] / pool["ec_ug"], pool["ftir_ec_ug"] / pool["ec_ug"]
    gap = (pool["date"] - pool.groupby("Site")["date"].shift(1)).dt.days
    for column in ("ocec", "ec_ug"):
        pool[f"previous_{column}"] = pool.groupby("Site")[column].shift(1).where(gap.between(1, 7))
    pool["site_ocec"] = pool["Site"].map(pool.groupby("Site")["ocec"].median())
    rng = np.random.default_rng(20261110)

    def level(part, grouping, label, fair):
        by_site = {site: frame["ratio"].to_numpy() for site, frame in part.groupby("Site")}
        sites = np.array(list(by_site))
        boot = [np.median(np.concatenate([by_site[s] for s in rng.choice(sites, sites.size)])) for _ in range(min(n_boot, 400))]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"grouping": grouping, "filters": label, "fair": fair, "n": len(part), "sites": int(sites.size),
                "ftir_over_tor": float(part["ratio"].median()), "lo": lo, "hi": hi,
                "sum_ratio": float(part["ftir_ec_ug"].sum() / part["ec_ug"].sum()),
                "median_tor_ocec": float(part["ocec"].median()), "median_tor_ec_ug": float(part["ec_ug"].median())}

    own, previous, site = "this filter's own TOR (selects on its error)", "the previous sample at the site", "the site"
    heavy = pool["ec_ug"] >= 15
    rows = [
        level(pool[heavy & (pool["ocec"] < 2.5)], own, "TOR EC ≥ 15 µg and OC/EC < 2.5", False),
        level(pool[heavy & pool["ocec"].between(2.5, 4)], own, "TOR EC ≥ 15 µg and OC/EC 2.5–4", False),
        level(pool[(pool["previous_ec_ug"] >= 15) & (pool["previous_ocec"] < 2.5)], previous, "previous EC ≥ 15 µg and OC/EC < 2.5", True),
        level(pool[(pool["previous_ec_ug"] >= 15) & pool["previous_ocec"].between(2.5, 4)], previous, "previous EC ≥ 15 µg and OC/EC 2.5–4", True),
        level(pool[pool["Site"].isin(URBAN)], site, "six urban IMPROVE sites", True),
        level(pool[pool["site_ocec"].between(2.5, 3.5)], site, "sites with median OC/EC 2.5–3.5", True),
        level(pool[pool["site_ocec"].between(3.5, 5)], site, "sites with median OC/EC 3.5–5", True),
        level(pool[pool["site_ocec"] > 5], site, "sites with median OC/EC above 5", True),
    ]
    # One point per IMPROVE site: the level against the site's own composition. IMPROVE is mostly rural and remote;
    # its few urban sites are the low-OC/EC end, and still far from Addis.
    by_site = pool.groupby("Site").agg(n=("ratio", "size"), median_tor_ocec=("ocec", "median"), median_tor_ec_ug=("ec_ug", "median"),
                                       ftir_over_tor=("ratio", "median")).reset_index().rename(columns={"Site": "site"})
    by_site = by_site[by_site["n"] >= 30]
    by_site["kind"] = np.where(by_site["site"].isin(URBAN), "urban IMPROVE site", "other IMPROVE site")

    # The artefact, drawn: the same filters binned by three different OC/EC ratios.
    pool["ftir_ocec"] = pool["ftir_oc_ug"] / pool["ftir_ec_ug"]
    edges = [0, 1.5, 2, 2.5, 3, 4, 6, 100]

    def curve(frame, column, label):
        groups = frame.groupby(pd.cut(frame[column], edges), observed=True)
        out = pd.DataFrame({"chosen_by": label, "ocec": groups[column].median(), "ftir_over_tor": groups["ratio"].median(), "n": groups.size()})
        return out[out["n"] >= 30].reset_index(drop=True)

    selection = pd.concat([
        curve(pool[pool["ec_ug"] >= FTIR_MIN_EC_UG], "ocec", "TOR's own OC/EC on this filter"),
        curve(pool[pool["ftir_ec_ug"] >= FTIR_MIN_EC_UG], "ftir_ocec", "FTIR's own OC/EC on this filter"),
        curve(pool[pool["previous_ec_ug"] >= FTIR_MIN_EC_UG], "previous_ocec", "TOR OC/EC of the previous sample at the site"),
    ], ignore_index=True)

    improve = pd.DataFrame(closeout["improve_filters"])
    urban, selected = improve[improve["site"].isin(URBAN)], improve.query("tor_ug >= 15 and ocec < 2.5")
    labels = {"raw k18": "Raw spectra", "spline k18": "Spline baseline", "vibes k18": "Production VIBES", "vibes_deflate k18": "Deflated VIBES",
              ANCHORED: "PTFE-anchored VIBES", "anchored_noCH k18": "Anchored, C–H masked", "anchored_balanced k12": "Anchored, OC/EC-balanced weights"}
    candidates = pd.DataFrame([{"calibration": label, "urban_n": len(urban), "urban_sites": float(np.median(urban[key] / urban["tor_ug"])),
                                "urban_sum_ratio": float(urban[key].sum() / urban["tor_ug"].sum()), "selected_n": len(selected),
                                "selected_on_own_tor": float(np.median(selected[key] / selected["tor_ug"]))} for key, label in labels.items()])

    clear = pool[pool["ftir_ec_ug"] >= FTIR_MIN_EC_UG]
    improve_ocec = clear["ftir_oc_ug"] / clear["ftir_ec_ug"]
    addis = addis.assign(ftir_ec_ugm3=addis["ftir_ec_ug"] / addis["volume_m3"], ftir_oc_ugm3=addis["ftir_oc_ug"] / addis["volume_m3"],
                         ftir_ocec=addis["ftir_oc_ug"] / addis["ftir_ec_ug"])
    addis = addis[addis["filter_id"].isin(table.index)].merge(table[[PUBLISHED, "season", "lot"]], left_on="filter_id", right_index=True)
    fifth = float(improve_ocec.quantile(0.05))
    both = addis.dropna(subset=[PUBLISHED])
    composition = pd.DataFrame([{
        "improve_filters": len(clear), "improve_p05": fifth, "improve_median": float(improve_ocec.median()),
        "improve_urban_median": float(improve_ocec[clear["Site"].isin(URBAN)].median()),
        "addis_filters": len(addis), "addis_median": float(addis["ftir_ocec"].median()), "addis_p95": float(addis["ftir_ocec"].quantile(0.95)),
        "addis_share_below_improve_p05": float((addis["ftir_ocec"] < fifth).mean()),
        "refit_vs_published_r": float(np.corrcoef(both["ftir_ec_ugm3"], both[PUBLISHED])[0, 1]),
        "published_over_refit": float((both[PUBLISHED] / both["ftir_ec_ugm3"]).median())}])
    values = pd.concat([pd.DataFrame({"population": "IMPROVE (FTIR EC ≥ 5 µg per filter)", "ftir_ocec": improve_ocec.clip(-1, 15).round(2)}),
                        pd.DataFrame({"population": "Addis Ababa", "ftir_ocec": addis["ftir_ocec"].round(2)})], ignore_index=True)
    return {"ftir_level": pd.DataFrame(rows), "ftir_candidates": candidates, "ftir_composition": composition, "ftir_ocec_values": values,
            "addis_refit": addis[["filter_id", "season", "lot", "ftir_ec_ugm3", "ftir_oc_ugm3", "ftir_ocec"]].round(4),
            "ftir_by_site": by_site.round(4), "ftir_selection": selection.round(4)}


def stage_level(table, closeout, n_boot, tables, aethalometer):
    """Evidence on the absolute FTIR level. None of it is thermal EC measured at Addis, and none uses an aethalometer."""
    improve = pd.DataFrame(closeout["improve_filters"])
    models = {"raw k18": "Raw spectra", ANCHORED: "PTFE-anchored VIBES", "spline k18": "Spline baseline",
              "anchored_balanced k12": "Anchored, OC/EC-balanced weights"}
    # Kept only for comparison with the closeout: binning on a filter's own TOR OC/EC selects on TOR's error and reads
    # low. The fair groupings are in stage ``ftir``; this table is not used for the level.
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

    # FTIR ÷ TOR on the IMPROVE groupings that do not select on a filter's own error (stage ``ftir``). The refit reads
    # Addis lower than the published product by a fixed factor, so the same bias is carried to the published scale.
    fair = tables["ftir_level"][tables["ftir_level"]["fair"].astype(bool)].set_index("filters")
    over_refit = float(tables["ftir_composition"]["published_over_refit"].iloc[0])
    anchors = []
    for label in ("six urban IMPROVE sites", "previous EC ≥ 15 µg and OC/EC < 2.5"):
        anchors.append({"anchor": f"Held-out IMPROVE, {label} (FTIR ÷ TOR, raw spectra, 18 components)",
                        "published_over_true": float(fair.loc[label, "ftir_over_tor"]) * over_refit, "value": float(fair.loc[label, "ftir_over_tor"]),
                        "note": f"published reads {over_refit:.2f}× this refit at Addis; TOR OC/EC {fair.loc[label, 'median_tor_ocec']:.1f} there, far above Addis"})
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


def stage_product(table, closeout, n_boot, tables, aethalometer, *, level=(1.0, 1.1, 1.4)):
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
    """Every SPARTAN site: is HIPS proportional to FTIR EC, and what linearising HIPS does to the comparison.

    The exponent uses gravimetric PM2.5 (and, separately, XRF potassium) as the instrument. PM2.5 is a good
    instrument only where EC is a steady share of it, and the intervals say where it is not (dust and sulfate
    sites). Beside it is what the IMPROVE ceiling predicts for HIPS alone at that site's optical depths. The
    Deming fits use the Addis error ratio at every site, so they are comparable with each other, not tuned per
    site. Deposit area 3.53 cm². ``network_points`` is one row per filter for the gallery.
    """
    tau_inf = thermal_ceiling(tables)["central"]
    lam = float(closeout["settings"]["LAMBDA_ADDIS"])
    names = {site["code"]: site.get("name") or site["code"] for site in json.loads(SITE_MAP.read_text()).get("sites", [])}
    rows = pd.DataFrame(json.loads(NETWORK.read_text())["rows"])
    rows = rows[~rows["excluded"].astype(bool)].copy()
    rows["fabs"] = rows["HIPS Fabs"].fillna(rows["BC (public HIPS)"] * MAC_VALUE)
    rows["tau"] = rows["fabs"] * rows["Sampled volume"] / (100.0 * DEPOSIT_AREA_CM2)
    rows["hips_bc"] = rows["fabs"] / MAC_VALUE
    rows["hips_linearised"] = rc.linearise(rows["tau"], tau_inf) * 100.0 * DEPOSIT_AREA_CM2 / rows["Sampled volume"] / MAC_VALUE
    sites, points = [], []
    for code, part in rows.groupby("code"):
        part = part[(part[["fabs", "EC (FTIR)", "PM2.5 mass"]] > 0).all(axis=1) & part["tau"].notna()]
        if len(part) < NETWORK_MIN_FILTERS:
            continue
        r = rc.response_exponent(part["fabs"], part["EC (FTIR)"], part["PM2.5 mass"], n_boot=min(n_boot, 500))
        row = {"code": code, "site": names.get(code, code), "n": r["n"], "median_optical_depth": float(part["tau"].median()),
               "p90_optical_depth": float(part["tau"].quantile(0.9)), "share_above_half": float((part["tau"] > 0.5).mean()),
               "hips_exponent_vs_ftir_ec": r["exponent"], "lo": r["lo"], "hi": r["hi"],
               "predicted_from_improve": float(np.mean(np.clip(1.0 - part["tau"] / tau_inf, 0.0, 1.0)))}
        potassium = part[part["Potassium"] > 0]
        if len(potassium) >= NETWORK_MIN_FILTERS:
            k = rc.response_exponent(potassium["fabs"], potassium["EC (FTIR)"], potassium["Potassium"], n_boot=min(n_boot, 300))
            row |= {"exponent_potassium": k["exponent"], "potassium_lo": k["lo"], "potassium_hi": k["hi"]}
        usable = part[np.isfinite(part["hips_linearised"])]
        before = deming_regression(part["hips_bc"], part["EC (FTIR)"], lam)
        after = deming_regression(usable["hips_linearised"], usable["EC (FTIR)"], lam)
        row |= {"slope_measured": before["slope"], "intercept_measured": before["intercept"],
                "slope_linearised": after["slope"], "intercept_linearised": after["intercept"],
                "median_ftir_ec": float(part["EC (FTIR)"].median()), "dropped_above_ceiling": int(len(part) - len(usable)),
                "median_ftir_over_hips": float((part["EC (FTIR)"] / part["hips_bc"]).median()),
                "median_ftir_over_linearised": float((usable["EC (FTIR)"] / usable["hips_linearised"]).median())}
        sites.append(row)
        points.append(part.assign(site=names.get(code, code))[["id", "code", "site", "date", "tau", "hips_bc", "hips_linearised", "EC (FTIR)", "PM2.5 mass"]])
    return {"network_exponents": pd.DataFrame(sites).sort_values("median_optical_depth"),
            "network_points": pd.concat(points, ignore_index=True).rename(columns={"EC (FTIR)": "ftir_ec", "PM2.5 mass": "pm25"})}


def stage_candidates(table, closeout, n_boot, tables, aethalometer):
    """Every saved Addis calibration, by season, against HIPS as measured and against linearised HIPS.

    The season-specific and cohort conclusions of September and October read slopes and intercepts against
    HIPS / 10. This repeats the reading for each calibration the closeout saved per filter.
    """
    tau_inf = thermal_ceiling(tables)["central"]
    lam = float(closeout["settings"]["LAMBDA_ADDIS"])
    comparators = {"HIPS as measured": table["hips_bc"],
                   "HIPS linearised": rc.linearise(table["tau"], tau_inf) * 100.0 * DEPOSIT_AREA_CM2 / table["volume_m3"] / MAC_VALUE}
    names = {"published": "SPARTAN published", "raw k18": "Raw spectra", "spline k18": "Spline baseline", "vibes k18": "Production VIBES",
             "vibes_deflate k18": "Deflated VIBES", "vibes_anchorhp k18": "PTFE-anchored VIBES", "anchored_balanced k12": "Anchored, OC/EC-balanced weights",
             "epo k18": "Anchored + EPO (historical)", "mask_org k18": "Anchored, organic bands masked", "cont_only k10": "Continuum windows only",
             "dipls_g10 k18": "Domain-invariant PLS", "iw k18": "Importance-weighted PLS"}
    rows = []
    for key, label in names.items():
        for comparator, x in comparators.items():
            for season, index in [("All", table.index), *[(name, part.index) for name, part in table.groupby("season")]]:
                pair = pd.DataFrame({"x": x[index], "y": table.loc[index, key]}).replace([np.inf, -np.inf], np.nan).dropna()
                if len(pair) < 20:
                    continue
                d = deming_regression(pair["x"], pair["y"], lam)
                rows.append({"calibration": label, "comparator": comparator, "season": season, "n": d["n"], "slope": d["slope"],
                             "intercept": d["intercept"], "intercept_share_of_mean": d["intercept"] / pair["y"].mean(),
                             "median_ratio": float((pair["y"] / pair["x"]).median())})
    return {"candidates_by_season": pd.DataFrame(rows)}


def stage_source(table, closeout, n_boot, tables, aethalometer):
    """Two checks inside the calibration set, where thermal EC exists.

    1. Is FTIR EC proportional to TOR EC on held-out sites? Exponent of the held-out prediction against TOR EC,
       with HIPS as the instrument. If it is near 1, the small leftover at Addis points at the HIPS ceiling
       there rather than at FTIR rising faster than EC.
    2. Does filter lot matter? Addis lot 248 reads low. The same lots exist in IMPROVE (catalogue on the Drive;
       reused from the saved summary where it is absent).
    """
    improve = pd.DataFrame(closeout["improve_filters"])
    models = {"raw k18": "Raw spectra", ANCHORED: "PTFE-anchored VIBES", "spline k18": "Spline baseline"}
    groups = {"all held-out filters": improve, "TOR OC/EC < 2.5": improve[improve["ocec"] < 2.5], "TOR OC/EC ≥ 2.5": improve[improve["ocec"] >= 2.5],
              "TOR EC ≥ 15 µg": improve[improve["tor_ug"] >= 15]}
    rows = []
    for key, label in models.items():
        for name, part in groups.items():
            r = rc.response_exponent(part[key], part["tor_ug"], part["hips_ug"], n_boot=min(n_boot, 500))
            rows.append({"calibration": label, "filters": name, **r})
    catalogue = data_paths.ftir_local_db() / "tables/ftir_catalog.csv"
    if catalogue.is_file():
        lots = pd.read_csv(catalogue, encoding="utf-8-sig", usecols=["FilterId", "LotNumber"], dtype=str).drop_duplicates("FilterId")
        improve["lot"] = improve["filter_id"].astype(str).map(lots.set_index("FilterId")["LotNumber"])
        lot_rows = []
        for name, part in {"all held-out filters": improve, "TOR OC/EC < 2.5": improve[improve["ocec"] < 2.5]}.items():
            for lot, sub in part.groupby("lot"):
                lot_rows.append({"filters": name, "lot": lot, "n": len(sub), **{label: float(np.median(sub[key] / sub["tor_ug"])) for key, label in models.items()}})
        lot_table = pd.DataFrame(lot_rows)
    elif SUMMARY.is_file() and "improve_lot" in (saved := json.loads(SUMMARY.read_text())):
        lot_table = pd.DataFrame(saved["improve_lot"])
    else:
        lot_table = pd.DataFrame(columns=["filters", "lot", "n", *models.values()])
    return {"improve_ftir_linearity": pd.DataFrame(rows), "improve_lot": lot_table}


def stage_aethalometer(table, closeout, n_boot, tables, aethalometer):
    """Side check only (``--with-aethalometer``): three-way random error. The Addis MA350 is not a reference."""
    rows = []
    result = rc.triple_collocation(table[PUBLISHED], table["aeth_ir"], table["fabs"], n_boot=n_boot)
    for column, m in zip((PUBLISHED, "aeth_ir", "fabs"), result["measurements"]):
        rows.append({"measurement": LABEL[column], "n": result["n"], "signal_sd": result["signal_sd"], **m})
    return {"aethalometer_triple_collocation": pd.DataFrame(rows)}


RUN = {"improve": stage_improve, "linearity": stage_linearity, "response": stage_response, "season": stage_season,
       "ftir": stage_ftir, "level": stage_level, "product": stage_product, "network": stage_network, "candidates": stage_candidates, "source": stage_source}
SUMMARY_TABLES = ("improve_ceiling", "improve_strata", "linearity", "frisch_bounds", "hips_response_fits", "linearised_hips",
                  "raw_hips", "season_lot", "improve_ratio_by_ocec", "mass_closure", "level_anchors", "addis_ocec",
                  "network_exponents", "candidates_by_season", "improve_ftir_linearity", "improve_lot",
                  "ftir_level", "ftir_candidates", "ftir_composition", "ftir_ocec_values", "addis_refit", "ftir_by_site", "ftir_selection")


def run(stages=STAGES, *, n_boot=2000, aethalometer=False, known_volume_only=False):
    """Run the stages in order and return (tables, Addis table). ``improve`` always runs first: the rest need its ceiling."""
    table, closeout = load()
    if known_volume_only:
        table = table[table["volume_known"]]
    tables: dict[str, pd.DataFrame] = {}
    needed = ("improve", "ftir") if "level" in stages else ("improve",)
    for stage in dict.fromkeys((*needed, *[s for s in STAGES if s in stages])):
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
