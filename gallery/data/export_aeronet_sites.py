"""Per-site (and per-season) AERONET summaries over the SPARTAN filters, for the gallery Map.

Run with:  uv run python gallery/data/export_aeronet_sites.py      (after export_aeronet.py)

Inputs (all committed gallery exports, so this needs no local cache):
  gallery/app/public/data/aeronet/filters.json     per filter: the AERONET means over its sampling window
  gallery/app/public/data/aeronet/stations.json    per site: the station used and its distance
  gallery/app/public/data/filters.json             the four focus sites' filter chemistry (HIPS Fabs)
  gallery/app/public/data/spartan_reference.json   public SPARTAN chemistry for every site
  gallery/app/src/siteSeasons.network.json         each site's season calendar (month bins)
  research/maia/inventory/spartan_sites.csv        the MAIA target area each SPARTAN site sits in

For each site with an AERONET station, over its non-excluded filters with a value in the window, the
median, mean, quartiles and n of: AOD 500, AAOD 675, AAE 440-870, SSA 440, the AOD-to-surface-PM
conversion eta = PM2.5 / AOD 500 (ug/m3 per unit AOD, what satellite PM2.5 retrievals such as MAIA's
depend on) and the absorption scale height H = AAOD 675 / (Fabs x 1e-6) (m), all year and by season.
Also the Spearman correlation of PM2.5 with AOD 500, which says whether eta is worth reading at all.

Writes gallery/app/public/data/aeronet/sites_summary.json.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "gallery/app/public/data"
AERONET = DATA / "aeronet"
OUT = AERONET / "sites_summary.json"
CALENDARS = ROOT / "gallery/app/src/siteSeasons.network.json"
MAIA = ROOT / "research/maia/inventory/spartan_sites.csv"

AOD500, AAOD675 = "AOD 500 nm (AERONET)", "AAOD 675 nm (AERONET)"
# key -> label (as the Map's picker shows it), unit, decimals, what it is
QUANTITIES = {
    "aod500": ("AOD 500 nm (AERONET)", "", 3, "Direct-sun AOD at 500 nm"),
    "aaod675": ("AAOD 675 nm (AERONET)", "", 4, "Absorption AOD at 675 nm (inversions)"),
    "aae": ("AAE 440-870 (AERONET)", "", 2, "Absorption Angstrom exponent 440-870 nm (inversions)"),
    "ssa440": ("SSA 440 nm (AERONET)", "", 3, "Single-scattering albedo at 440 nm (inversions)"),
    "eta": ("η = PM2.5 / AOD 500 (AERONET)", "µg/m³ per unit AOD", 1,
            "Filter PM2.5 mass over the window's AOD 500: the AOD-to-surface-PM conversion"),
    "h": ("Absorption scale height (AERONET)", "m", 0,
          "AAOD 675 / (surface Fabs × 10⁻⁶ m⁻¹); Fabs is HIPS where measured, else public BC × 10"),
}
SOURCE_FIELD = {"aod500": AOD500, "aaod675": AAOD675, "aae": "AAE 440-870 (AERONET)", "ssa440": "SSA 440 nm (AERONET)"}
CHEMISTRY = ("PM2.5 mass", "HIPS Fabs", "BC (public HIPS)")


def r(value, digits):
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


def filter_table() -> pd.DataFrame:
    """One row per physical filter: code, date, exclusion and the chemistry eta and H need.

    The focus sites' filters.json comes first and the public reference fills what it lacks, the
    way the gallery merges them (App.tsx supplementFilter).
    """
    rows: dict[str, dict] = {}
    for name in ("filters.json", "spartan_reference.json"):
        for row in json.loads((DATA / name).read_text())["rows"]:
            if not row.get("code") or not row.get("date"):
                continue
            have = rows.setdefault(row["id"], {"id": row["id"], "code": row["code"], "date": row["date"],
                                                "excluded": bool(row.get("excluded"))})
            have["excluded"] = have["excluded"] or bool(row.get("excluded"))
            for f in CHEMISTRY:
                if have.get(f) is None and isinstance(row.get(f), (int, float)):
                    have[f] = float(row[f])
    frame = pd.DataFrame(list(rows.values()))
    for f in CHEMISTRY:
        if f not in frame:
            frame[f] = np.nan
    return frame


def aeronet_table() -> pd.DataFrame:
    payload = json.loads((AERONET / "filters.json").read_text())
    frame = pd.DataFrame.from_dict(payload["rows"], orient="index", columns=payload["fields"]).astype(float)
    frame.index.name = "id"
    return frame.reset_index()


def season_lookup(calendars: dict) -> dict[str, dict[int, str]]:
    """code -> month -> season name, from the gallery's network calendar table."""
    return {code: {m: s["name"] for s in cal["seasons"] for m in s["months"]} for code, cal in calendars.items()}


def derive(frame: pd.DataFrame) -> pd.DataFrame:
    """The six quantities per filter. eta needs PM2.5 > 0 and AOD 500 > 0; H needs Fabs > 0."""
    out = pd.DataFrame(index=frame.index)
    for key, field in SOURCE_FIELD.items():
        out[key] = frame[field]
    pm, aod = frame["PM2.5 mass"], frame[AOD500]
    out["eta"] = (pm / aod).where((pm > 0) & (aod > 0))
    fabs = frame["HIPS Fabs"].where(frame["HIPS Fabs"].notna(), frame["BC (public HIPS)"] * 10)
    out["h"] = (frame[AAOD675] / (fabs * 1e-6)).where(fabs > 0)
    return out


def summarise(values: pd.Series, digits: int, full: bool = True) -> dict | None:
    v = values.dropna().to_numpy(dtype=float)
    if not len(v):
        return None
    out = {"n": int(len(v)), "median": r(float(np.median(v)), digits)}
    if full:
        out.update(mean=r(float(v.mean()), digits), p25=r(float(np.percentile(v, 25)), digits),
                   p75=r(float(np.percentile(v, 75)), digits))
    return out


def spearman(x: pd.Series, y: pd.Series, min_n: int = 8) -> float | None:
    ok = x.notna() & y.notna()
    if ok.sum() < min_n:
        return None
    rho = x[ok].rank().corr(y[ok].rank())
    return r(rho, 2)


def site_summary(part: pd.DataFrame, values: pd.DataFrame, months: dict[int, str] | None, order: list[str]) -> dict:
    quantities = {k: s for k in QUANTITIES if (s := summarise(values[k], QUANTITIES[k][2])) is not None}
    out = {"n_filters": int(len(part)), "rho_pm_aod": spearman(part["PM2.5 mass"].where(part["PM2.5 mass"] > 0), part[AOD500]),
           "all": quantities, "seasons": {}}
    if months:
        season = pd.to_datetime(part["date"]).dt.month.map(months)
        for name in order:
            sel = season == name
            if not sel.any():
                continue
            entry = {k: s for k in QUANTITIES if (s := summarise(values.loc[sel, k], QUANTITIES[k][2], full=False)) is not None}
            if entry:
                out["seasons"][name] = {"n_filters": int(sel.sum()), **entry,
                                        "rho_pm_aod": spearman(part.loc[sel, "PM2.5 mass"].where(part.loc[sel, "PM2.5 mass"] > 0), part.loc[sel, AOD500])}
    return out


def maia_targets() -> dict[str, dict]:
    with MAIA.open(newline="", encoding="utf-8") as handle:
        return {row["code"]: {"target": row["target"], "type": row["target_type"]}
                for row in csv.DictReader(handle) if row.get("target")}


def main() -> None:
    calendars = json.loads(CALENDARS.read_text())
    seasons = season_lookup(calendars)
    stations = json.loads((AERONET / "stations.json").read_text())
    merged = aeronet_table().merge(filter_table(), on="id", how="left")
    missing = merged["code"].isna()
    if missing.any():
        # an AERONET filter whose chemistry row is gone: its code is the id's prefix, and it has no date
        merged.loc[missing, "code"] = merged.loc[missing, "id"].str.split("-").str[0]
    merged = merged[~merged["excluded"].fillna(False).astype(bool)]
    values = derive(merged)
    sites = {}
    for code, part in merged.groupby("code"):
        if code not in stations["sites"]:
            continue
        st = stations["sites"][code]
        dated = part["date"].notna()
        order = [s["name"] for s in calendars.get(code, {}).get("seasons", [])]
        sites[code] = {"station": st["station"], "km": st["km"],
                       **site_summary(part[dated], values.loc[part.index[dated]], seasons.get(code), order)}
    maia = maia_targets()
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(), "level": stations["level"], "source": stations["source"],
        "note": ("Per SPARTAN site with an AERONET station: medians over its non-excluded filters of the AERONET means inside "
                 "each filter's sampling window (export_aeronet.py), all year and by the site's season calendar "
                 "(siteSeasons.network.json). eta = filter PM2.5 / AOD 500, for filters with both positive; "
                 "H = AAOD 675 / surface Fabs. AOD is a daylight, clear-sky mean; PM2.5 covers the whole window."),
        "quantities": [{"key": k, "label": q[0], "unit": q[1], "digits": q[2], "about": q[3]} for k, q in QUANTITIES.items()],
        "seasons": {code: [s["name"] for s in calendars[code]["seasons"]] for code in sites if code in calendars},
        "maia": maia,
        "sites": sites,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    eta = {c: s["all"].get("eta", {}).get("median") for c, s in sites.items()}
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(sites)} sites, {sum(1 for c in sites if c in maia)} in a MAIA target area")
    print("median eta (ug/m3 per unit AOD 500): " + ", ".join(f"{c} {v}" for c, v in sorted(eta.items(), key=lambda kv: kv[1] or 0)))


if __name__ == "__main__":
    main()
