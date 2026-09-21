#!/usr/bin/env python3
"""Export the repo's filter chemistry into JSON the React gallery can render.

Reads `unified_filter_dataset.pkl` through the repo's own loaders so the units,
MAC value, filter-id replicate stripping, and Ethiopian season calendar all come
from `research/ftir_hips_chem/scripts/config.py` rather than being restated here.

Writes into gallery/app/public/data/:
  meta.json          sites, seasons, wavelengths, MAC, provenance
  filters.json       one record per filter -- the workhorse table
  correlation.json   per-site chemistry correlation matrices
  composition.json   mean PM2.5 composition per site
  census.json        rollup of the notebook chart census

Run:  python gallery/data/export_data.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "research" / "ftir_hips_chem" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import config  # noqa: E402
from data_matching import (  # noqa: E402
    _param_to_column_name, load_filter_data, pivot_filter_by_id,
)
from etad_factors import (  # noqa: E402
    add_dominant_source, load_etad_factors_with_filter_ids, normalize_gf_fractions,
)
from outliers import apply_exclusion_flags  # noqa: E402

OUT = REPO / "gallery" / "app" / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# Column names below are the ones `pivot_filter_by_id` emits (it applies its own
# rename map, e.g. EC_ftir -> ftir_ec). Addressing them directly rather than
# guessing keeps EC/OC — the whole point of the estate — from silently dropping.
COMPOSITION = [
    ("chemspec_filter_pm2.5_mass", "PM2.5 mass"),
    ("chemspec_oc", "OC (TOR)"),
    ("chemspec_ec", "EC (TOR)"),
    ("chemspec_sulfate_ion_pm2.5", "Sulfate"),
    ("chemspec_nitrate_ion_pm2.5", "Nitrate"),
    ("chemspec_ammonium_ion_pm2.5", "Ammonium"),
    ("chemspec_potassium_ion_pm2.5", "Potassium"),
    ("chemspec_sodium_ion_pm2.5", "Sodium"),
    ("chemspec_calcium_ion_pm2.5", "Calcium"),
    ("chemspec_magnesium_ion_pm2.5", "Magnesium"),
    ("iron", "Iron"),
    ("chemspec_aluminum_pm2.5", "Aluminum"),
    ("chemspec_titanium_pm2.5", "Titanium"),
    ("chemspec_silicon_pm2.5", "Silicon"),
    ("chemspec_zinc_pm2.5", "Zinc"),
    ("chemspec_copper_pm2.5", "Copper"),
    ("chemspec_lead_pm2.5", "Lead"),
]
# FTIR functional groups, the optical columns, and the HIPS error terms.
# HIPS_Uncertainty is carried as its own parameter row, not as a column on the
# Fabs row (see AGENTS.md) — it is what pins Deming lambda, so it ships too.
CORE = [
    ("ftir_ec", "EC (FTIR)"),
    ("ftir_oc", "OC (FTIR)"),
    ("om", "OM (FTIR)"),
    ("alkanech", "Alkane CH"),
    ("alcoholcoh", "Alcohol COH"),
    ("carboxyliccooh", "Carboxylic COOH"),
    ("naco", "Non-acid CO"),
    ("hips_fabs", "HIPS Fabs"),
    ("hips_uncertainty", "HIPS Fabs uncertainty"),
    ("hips_mdl", "HIPS MDL"),
]

# Measurement families. A flat 28-item dropdown hides the fact that these are
# four different instruments answering different questions; grouping keeps
# "EC (FTIR) vs EC (TOR)" (a method comparison) visibly distinct from
# "EC vs Silicon" (a source question).
FIELD_GROUPS = [
    ("Carbon", ["EC (FTIR)", "EC (TOR)", "OC (FTIR)", "OC (TOR)", "OM (FTIR)"]),
    ("Optical (HIPS)", ["HIPS Fabs", "HIPS BC", "HIPS Fabs uncertainty", "HIPS MDL"]),
    ("FTIR functional groups", ["Alkane CH", "Alcohol COH", "Carboxylic COOH", "Non-acid CO"]),
    ("Mass", ["PM2.5 mass"]),
    ("Ions", ["Sulfate", "Nitrate", "Ammonium", "Potassium", "Sodium", "Calcium", "Magnesium"]),
    ("Metals & crustal", ["Silicon", "Aluminum", "Iron", "Titanium", "Zinc", "Copper", "Lead"]),
]

# PMF source factors (ETAD only), in GF1-GF5 order so the stack matches the
# notebooks. Colours are NOT invented here — they are the palette defined in
# research/ftir_hips_chem/ETAD_Factor_Analysis.ipynb, the notebook that owns
# this factor solution. config.py has no source palette; if one is ever added,
# read it from there instead.
PMF_SOURCES = [
    ("sea_salt_frac", "K_F1 Sea Salt Mixed (ug/m3)", "Sea Salt Mixed", "#1f77b4"),
    ("wood_frac", "K_F2 Wood Burning (ug/m3)", "Wood Burning", "#ff7f0e"),
    ("charcoal_frac", "K_F3 Charcoal (ug/m3)", "Charcoal", "#2ca02c"),
    ("polluted_marine_frac", "K_F4 Polluted Marine (ug/m3)", "Polluted Marine", "#9467bd"),
    ("fossil_fuel_frac", "K_F5 Fossil Fuel Combustion (ug/m3)", "Fossil Fuel Combustion", "#d62728"),
]

SITE_BY_CODE = {v["code"]: (k, v) for k, v in config.SITES.items()}


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if (pd.isna(v) or np.isinf(v)) else round(float(v), 6)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if pd.isna(v):
        return None
    return v


def season_of(month: int) -> str:
    """Ethiopian calendar, default (dry-owns-February) convention from config."""
    return config.season_for_month(month)


def unit_map(raw: pd.DataFrame) -> dict[str, str]:
    """Declared unit per pivoted column name.

    The trace metals (Silicon, Iron, Aluminum, Zinc, Titanium, Lead, Copper,
    Magnesium) are stored in ng/m3 while the ions and carbon fractions are in
    ug/m3. Reading `Concentration_Units` is exact; prep.to_ugm3's median>100
    heuristic is not, and a treemap that mixes the two is off by 1000x.
    """
    out: dict[str, str] = {}
    for param, grp in raw.groupby("Parameter")["Concentration_Units"]:
        vals = grp.dropna().unique()
        if len(vals):
            out[_param_to_column_name(param)] = str(vals[0])
    return out


def normalise_units(df: pd.DataFrame, units: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Convert every ng/m3 column to ug/m3 so one figure can mix species."""
    final: dict[str, str] = {}
    for col, unit in units.items():
        if col not in df.columns:
            continue
        if unit.lower() in ("ng/m3", "ng/m^3", "ng m-3"):
            df[col] = pd.to_numeric(df[col], errors="coerce") / 1000.0
            final[col] = "ug/m3"
        else:
            final[col] = unit
    return df, final


def build_filters(raw: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for code, (site_name, site_cfg) in SITE_BY_CODE.items():
        # hips_units='both' -> hips_fabs stays Mm^-1, hips_bc_ugm3 is the ug/m3 form
        piv = pivot_filter_by_id(raw, code, hips_units="both")
        if piv.empty:
            continue
        # Flag rather than drop — the exclusion system is deliberately
        # non-destructive (AGENTS.md), so the app can show what was removed
        # and why instead of silently losing the audit trail.
        piv = apply_exclusion_flags(piv, site_name)
        piv["site_code"] = code
        piv["site"] = site_name.replace("_", " ")
        piv["site_color"] = site_cfg["color"]
        frames.append(piv)
    df = pd.concat(frames, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["season"] = df["month"].map(season_of)
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    # lat/lon per site, straight off the source table
    coords = (raw.groupby("Site")[["Latitude", "Longitude"]].median()
                 .rename(columns={"Latitude": "lat", "Longitude": "lon"}))
    df = df.merge(coords, left_on="site_code", right_index=True, how="left")
    return df


def main():
    print("loading unified_filter_dataset.pkl ...")
    raw = load_filter_data()
    print(f"  {len(raw):,} rows, {raw['FilterId'].nunique():,} filter ids")

    df = build_filters(raw)
    print(f"pivoted -> {len(df):,} filters across {df['site_code'].nunique()} sites")

    df, units = normalise_units(df, unit_map(raw))
    n_ng = sum(1 for c, u in unit_map(raw).items() if u.lower().startswith('ng') and c in df.columns)
    print(f"units normalised: {n_ng} ng/m3 columns converted to ug/m3")

    colmap = {}
    missing = []
    for col, label in COMPOSITION + CORE:
        if col in df.columns:
            colmap[label] = col
        else:
            missing.append(label)
    if missing:
        print("  ! not present in this dataset:", ", ".join(missing))
    field_units = {label: units.get(col, "") for label, col in colmap.items()}
    field_units["HIPS BC"] = "ug/m3"
    print(f"resolved {len(colmap)} of {len(COMPOSITION)+len(CORE)} named columns")

    # ---------------- filters.json ----------------
    keep_meta = ["base_filter_id", "site", "site_code", "site_color", "date_str",
                 "year", "month", "season", "lat", "lon"]
    records = []
    for _, r in df.iterrows():
        rec = {
            "id": r["base_filter_id"],
            "site": r["site"],
            "code": r["site_code"],
            "color": r["site_color"],
            "date": r["date_str"],
            "year": _clean(r["year"]),
            "month": _clean(r["month"]),
            "season": r["season"],
            "lat": _clean(r["lat"]),
            "lon": _clean(r["lon"]),
            "excluded": bool(r.get("is_excluded", False)),
            "exclusion_reason": (r.get("exclusion_reason") or "") or None,
        }
        for label, col in colmap.items():
            rec[label] = _clean(r.get(col))
        if "hips_bc_ugm3" in df.columns:
            rec["HIPS BC"] = _clean(r.get("hips_bc_ugm3"))
        records.append(rec)

    numeric_fields = sorted({k for r in records for k, v in r.items()
                             if isinstance(v, (int, float)) and not isinstance(v, bool)
                             and k not in ("year", "month", "lat", "lon")})
    coverage = {f: sum(1 for r in records if r.get(f) is not None) for f in numeric_fields}

    (OUT / "filters.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl",
        "n": len(records),
        "n_excluded": sum(1 for r in records if r["excluded"]),
        "fields": numeric_fields,
        "coverage": coverage,
        "rows": records,
    }, indent=0))
    print(f"wrote filters.json  ({len(records):,} rows, {len(numeric_fields)} numeric fields)")

    # ---------------- meta.json ----------------
    sites = []
    for code, (name, cfg) in SITE_BY_CODE.items():
        sub = df[df["site_code"] == code]
        if sub.empty:
            continue
        sites.append({
            "code": code,
            "name": name.replace("_", " "),
            "location": cfg["location"],
            "color": cfg["color"],
            "timezone": cfg["timezone"],
            "lat": _clean(sub["lat"].median()),
            "lon": _clean(sub["lon"].median()),
            "n_filters": int(len(sub)),
            "date_min": sub["date_str"].min(),
            "date_max": sub["date_str"].max(),
        })

    (OUT / "meta.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mac_value": config.MAC_VALUE,
        "season_convention": config.DEFAULT_SEASON_CONVENTION,
        "seasons": [{"name": k, "months": v["months"], "color": v["color"]}
                    for k, v in config.ETHIOPIA_SEASONS.items()],
        # Both published Ethiopian calendars. February is a high-BC month and
        # the two disagree about which season owns it, so seasonal means are
        # NOT interchangeable between them (AGENTS.md). Shipping both lets the
        # app switch live and label which one produced a number.
        "season_conventions": {
            key: [
                {"name": name, "months": spec["months"], "color": spec["color"]}
                for name, spec in config.resolve_seasons(key).items()
            ]
            for key in config.SEASON_CONVENTIONS
        },
        "wavelengths_nm": config.WAVELENGTHS_NM,
        "field_units": field_units,
        "field_groups": [
            {"label": label, "fields": [f for f in fields if f in numeric_fields]}
            for label, fields in FIELD_GROUPS
            if any(f in numeric_fields for f in fields)
        ],
        "aae_regions": getattr(config, "AAE_REGIONS", None),
        "sites": sites,
        "fields": numeric_fields,
        "coverage": coverage,
    }, indent=1, default=str))
    print(f"wrote meta.json     ({len(sites)} sites)")

    # correlation and composition used to be precomputed here. The app now
    # derives both from the live row subset, so season/site filters actually
    # move those numbers; a precomputed copy would silently ignore them.

    # ---------------- pmf.json ----------------
    # Normalisation is mandatory: raw GF1-GF5 are PM2.5 mass fractions summing
    # to 0.03-0.46 per row, not relative source contributions. Using the repo
    # helpers rather than hand-rolling is an explicit AGENTS.md rule.
    try:
        factors = add_dominant_source(normalize_gf_fractions(load_etad_factors_with_filter_ids()))
    except Exception as exc:  # pragma: no cover - data may be absent
        print(f"! PMF factors unavailable ({exc}); skipping pmf.json")
        factors = None

    if factors is not None and not factors.empty:
        factors = factors.copy()
        factors["date"] = pd.to_datetime(factors["date"], errors="coerce")
        factors = factors.dropna(subset=["date"])
        pmf_rows = []
        for _, r in factors.iterrows():
            frac = {label: _clean(r.get(fcol)) for fcol, _a, label, _c in PMF_SOURCES}
            absolute = {label: _clean(r.get(acol)) for _f, acol, label, _c in PMF_SOURCES}
            total = sum(v for v in absolute.values() if v is not None)
            month = int(r["date"].month)
            pmf_rows.append({
                "id": r["base_filter_id"],
                "filter_id": r.get("FilterId"),
                "date": r["date"].strftime("%Y-%m-%d"),
                "year": int(r["date"].year),
                "month": month,
                "season": season_of(month),
                "fraction": frac,
                "ugm3": absolute,
                "total_ugm3": _clean(total),
                "dominant_source": r.get("dominant_source"),
                "dominant_fraction": _clean(r.get("dominant_fraction")),
            })
        pmf_rows.sort(key=lambda d: d["date"])
        (OUT / "pmf.json").write_text(json.dumps({
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "site": "ETAD",
            "site_name": "Addis Ababa",
            "note": ("GF1-GF5 normalised to relative source contributions via "
                     "normalize_gf_fractions; raw values are PM2.5 mass fractions."),
            # `key` is the normalized fraction column minus the _frac suffix,
            # which is exactly what add_dominant_source writes into
            # dominant_source — so the app matches on it rather than on a
            # prefix of the display label.
            "sources": [
                {"key": fcol.replace("_frac", ""), "label": label, "color": color}
                for fcol, _a, label, color in PMF_SOURCES
            ],
            "n": len(pmf_rows),
            "rows": pmf_rows,
        }, indent=0, default=str))
        mean_dom = factors["dominant_fraction"].mean()
        print(f"wrote pmf.json      ({len(pmf_rows)} filters, mean dominant fraction {mean_dom:.3f})")

    # ---------------- census.json ----------------
    census_path = REPO / "gallery" / "census" / "chart_census.json"
    if census_path.exists():
        recs = json.loads(census_path.read_text())
        from collections import Counter
        by_chart = Counter(r["gallery_chart"] for r in recs)
        by_cat = Counter(r["gallery_category"] for r in recs)
        by_nb = Counter(r["notebook"] for r in recs)
        examples = {}
        for r in recs:
            examples.setdefault(r["gallery_chart"], [])
            if len(examples[r["gallery_chart"]]) < 12 and r["title"]:
                examples[r["gallery_chart"]].append(
                    {"notebook": r["notebook"], "title": r["title"],
                     "x": r["xlabel"], "y": r["ylabel"]})
        (OUT / "census.json").write_text(json.dumps({
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_notebooks": len(by_nb), "n_figures": len(recs),
            "by_category": by_cat.most_common(),
            "by_chart": by_chart.most_common(),
            "top_notebooks": by_nb.most_common(25),
            "examples": examples,
        }, indent=0))
        print(f"wrote census.json   ({len(recs)} figures)")
    else:
        print("! census not built yet — run gallery/census/build_census.py first")

    print("\ndone ->", OUT)


if __name__ == "__main__":
    main()
