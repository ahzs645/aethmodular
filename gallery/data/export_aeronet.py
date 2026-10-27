"""AERONET over each filter's sampling window, for the gallery's AERONET tab and AERONET fields.

Run with:  uv run --with timezonefinder python gallery/data/export_aeronet.py

Inputs (all local; research/ftir_hips_chem/workflows/fetch_aeronet_spartan.py makes them):
  research/ftir_hips_chem/output/tables/aeronet_spartan/{aod,sda,inv}_{points,daily}_lev<L>.parquet
  research/ftir_hips_chem/output/tables/aeronet_spartan/stations_lev<L>.csv
  gallery/app/public/data/weather/filters.json   each filter's local sampling window (export_weather.py)

Both levels are exported when their parquet exists: Level 1.5 (the fields every tab merges) and
Level 2.0 (`--level 20` in the fetch; the AERONET tab's Level toggle). `--level 15` or `--level 20`
exports one.

Each filter's value is the mean of the individual retrievals inside its sampling window (local start
to end, converted to UTC with the site's standard offset), as the weather export does with ERA5 hours.
Sun photometers retrieve only in daylight and clear sky, so a window's mean covers its clear daytime
hours; a filter with no retrieval in its window gets no value. Older intermittent SPARTAN filters
spread 24 h of sampling over about 8 days, and the window is then the whole elapsed period.

Derived per retrieval, then averaged over the window like every other field (so a window's mean is of
coincident pairs, not a ratio of means):
  AAE 440-675 and 675-870   two-band absorption Ångström exponents, ln(AAOD1/AAOD2) / ln(λ2/λ1), only
                            where both AAODs are positive (research/ftir_ec_phase3/scripts/
                            run_aeronet_three_wavelength.py)
  SAE 440-675               scattering Ångström exponent from SAOD = AOD x SSA (Cazorla et al. 2013)
  dAOD, dSSA                AOD 440 - AOD 870 and SSA 440 - SSA 870 (Eom et al. 2025, as
                            notebooks/analysis/surface_columnar_linkage.ipynb uses them)
and per window, from the window means: an aerosol type after Giles et al. (2012) (aerosol_type()).

Writes to gallery/app/public/data/aeronet/ (Level 1.5) and aeronet/lev20/ (Level 2.0: the same files
except the whole-set filters.json):
  filters.json        per filter: the fields below, plus retrieval counts (the whole set, for scripts)
  index.json + filters/<CODE>.json   the same split by site: the gallery loads only the sites in view
  daily/<CODE>.json   per site: AERONET's own daily means, for the timeline and the climatology
  stations.json       per site: the station pulled, its distance and how much of the filter record it covers

Data: NASA AERONET Version 3 (aeronet.gsfc.nasa.gov).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_weather import r, std_offset_hours  # noqa: E402

CACHE = ROOT / "research/ftir_hips_chem/output/tables/aeronet_spartan"
LEVEL = "15"
OUT = ROOT / "gallery/app/public/data/aeronet"
# where each level's files go: Level 1.5 at the top (what every tab reads), Level 2.0 beside it
LEVEL_DIRS = {"15": OUT, "20": OUT / "lev20"}

# gallery field -> (product, parquet column, unit, decimals)
FIELDS = {
    "AOD 440 nm (AERONET)": ("aod", "aod440", "", 3),
    "AOD 500 nm (AERONET)": ("aod", "aod500", "", 3),
    "AOD 675 nm (AERONET)": ("aod", "aod675", "", 3),
    "AOD 870 nm (AERONET)": ("aod", "aod870", "", 3),
    "Angstrom exponent 440-870 (AERONET)": ("aod", "ae440_870", "", 2),
    "Precipitable water (AERONET)": ("aod", "pw_cm", "cm", 2),
    "Fine-mode AOD 500 nm (AERONET)": ("sda", "fine_aod500", "", 3),
    "Coarse-mode AOD 500 nm (AERONET)": ("sda", "coarse_aod500", "", 3),
    "Fine-mode fraction 500 nm (AERONET)": ("sda", "fmf500", "", 2),
    "AAOD 440 nm (AERONET)": ("inv", "aaod440", "", 4),
    "AAOD 675 nm (AERONET)": ("inv", "aaod675", "", 4),
    "AAOD 870 nm (AERONET)": ("inv", "aaod870", "", 4),
    "AAE 440-870 (AERONET)": ("inv", "aae440_870", "", 2),
    "SSA 440 nm (AERONET)": ("inv", "ssa440", "", 3),
    "SSA 675 nm (AERONET)": ("inv", "ssa675", "", 3),
    "Inversion AOD 440 nm (AERONET)": ("inv", "inv_aod440", "", 3),
    # added 27 Oct 2026: the rest of the inversion's spectrum and the typing quantities
    "SSA 870 nm (AERONET)": ("inv", "ssa870", "", 3),
    "SSA 1020 nm (AERONET)": ("inv", "ssa1020", "", 3),
    "AAE 440-675 (AERONET)": ("inv", "aae440_675", "", 2),
    "AAE 675-870 (AERONET)": ("inv", "aae675_870", "", 2),
    "SAE 440-675 (AERONET)": ("inv", "sae440_675", "", 2),
    "Inversion extinction AE 440-870 (AERONET)": ("inv", "eae440_870", "", 2),
    "dSSA 440-870 (AERONET)": ("inv", "dssa", "", 3),
    "Sphericity (AERONET)": ("inv", "sphericity", "%", 0),
    "dAOD 440-870 (AERONET)": ("aod", "daod", "", 3),
}
# the window's aerosol type (aerosol_type): a class number, which the gallery's grouping names
TYPE_FIELD = "Aerosol type class (AERONET)"
FMF_FIELD, SSA_FIELD = "Fine-mode fraction 500 nm (AERONET)", "SSA 440 nm (AERONET)"
AEROSOL_TYPES = {
    1: "Coarse-dominated (dust-like)",
    2: "Mixed size",
    3: "Fine, weakly absorbing (U/I-like)",
    4: "Fine, moderately absorbing",
    5: "Fine, strongly absorbing (smoke/BC-like)",
    6: "Fine, no inversion",
}
# daily series key -> (product, parquet column, decimals)
DAILY = {
    "aod440": ("aod", "aod440", 3), "aod500": ("aod", "aod500", 3), "aod675": ("aod", "aod675", 3),
    "ae": ("aod", "ae440_870", 2), "pw": ("aod", "pw_cm", 2), "fmf": ("sda", "fmf500", 2),
    "aaod440": ("inv", "aaod440", 4), "aaod675": ("inv", "aaod675", 4), "aae": ("inv", "aae440_870", 2),
    "ssa440": ("inv", "ssa440", 3), "ssa675": ("inv", "ssa675", 3),
}


def two_band_exponent(a1, a2, l1: float, l2: float) -> np.ndarray:
    """Ångström exponent between two wavelengths, ln(a1/a2) / ln(l2/l1); missing unless both are positive."""
    a1, a2 = np.asarray(a1, dtype=float), np.asarray(a2, dtype=float)
    ok = (a1 > 0) & (a2 > 0)
    return np.where(ok, np.log(np.where(ok, a1, 1.0) / np.where(ok, a2, 1.0)) / np.log(l2 / l1), np.nan)


def derive(product: str, frame: pd.DataFrame) -> pd.DataFrame:
    """The per-retrieval quantities the window means need (see the module docstring)."""
    frame = frame.copy()
    if product == "inv":
        frame["aae440_675"] = two_band_exponent(frame["aaod440"], frame["aaod675"], 440, 675)
        frame["aae675_870"] = two_band_exponent(frame["aaod675"], frame["aaod870"], 675, 870)
        frame["sae440_675"] = two_band_exponent(frame["inv_aod440"] * frame["ssa440"],
                                                frame["inv_aod675"] * frame["ssa675"], 440, 675)
        frame["dssa"] = frame["ssa440"] - frame["ssa870"]
    elif product == "aod":
        frame["daod"] = frame["aod440"] - frame["aod870"]
    return frame


def aerosol_type(fmf, ssa440) -> np.ndarray:
    """Each window's class (AEROSOL_TYPES) from its mean fine-mode fraction and SSA 440; 0 without an FMF.

    After Giles et al. (2012), JGR 117, D17203, sect. 3.4 and Fig. 11: coarse-dominated at η ≤ 0.33, mixed
    to 0.66, fine-dominated above; the fine windows by SSA 440: > 0.95 weakly absorbing (their
    urban/industrial cluster sits above 0.95), 0.90 to 0.95, and ≤ 0.90 strongly absorbing (their
    biomass-burning cluster, ~0.89, which urban BC can also reach). Giles found these clusters rather than
    setting a scheme: the cuts are the boundaries their Fig. 11 averages use. They used the inversion's
    η at 550 nm; this uses the SDA fine fraction at 500 nm.
    """
    fmf, ssa = np.asarray(fmf, dtype=float), np.asarray(ssa440, dtype=float)
    fine = np.where(~np.isfinite(ssa), 6, np.where(ssa > 0.95, 3, np.where(ssa > 0.90, 4, 5)))
    out = np.where(fmf <= 0.33, 1, np.where(fmf <= 0.66, 2, fine))
    return np.where(np.isfinite(fmf), out, 0)


def load(product: str, kind: str, level: str = LEVEL) -> pd.DataFrame:
    frame = pd.read_parquet(CACHE / f"{product}_{kind}_lev{level}.parquet")
    frame["time_utc"] = pd.to_datetime(frame["time_utc"])
    return derive(product, frame) if kind == "points" else frame


def window_means(points: pd.DataFrame, columns: list[str], starts: np.ndarray, ends: np.ndarray):
    """Mean of each column over the retrievals in [start, end) per window, and each window's slice of points."""
    t = points["time_utc"].to_numpy()
    lo, hi = np.searchsorted(t, starts, "left"), np.searchsorted(t, ends, "left")
    values = points[columns].to_numpy(dtype=float)
    ok = np.isfinite(values)
    # running sums turn every window's mean into two lookups
    total = np.vstack([np.zeros(len(columns)), np.cumsum(np.where(ok, values, 0.0), axis=0)])
    count = np.vstack([np.zeros(len(columns)), np.cumsum(ok, axis=0)])
    n = count[hi] - count[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(n > 0, (total[hi] - total[lo]) / n, np.nan)
    return means, lo, hi


def write_split(payload: dict, out: Path = OUT) -> None:
    """filters.json split by site (a filter id starts with its site code), as export_weather does."""
    rows, counts = {}, {}
    for fid, v in payload["rows"].items():
        rows.setdefault(fid.split("-")[0], {})[fid] = v
    for fid, v in payload["counts"].items():
        counts.setdefault(fid.split("-")[0], {})[fid] = v
    folder = out / "filters"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.json"):
        old.unlink()
    for code in sorted(rows):
        (folder / f"{code}.json").write_text(json.dumps(dict(rows=rows[code], counts=counts.get(code, {})),
                                                        separators=(",", ":"), allow_nan=False))
    index = {k: v for k, v in payload.items() if k not in ("rows", "counts")}
    index["sites"] = {code: len(v) for code, v in sorted(rows.items())}
    (out / "index.json").write_text(json.dumps(index, separators=(",", ":"), allow_nan=False))


def filter_windows() -> pd.DataFrame:
    """Every filter's local window: the weather export's, else 24 h from its gallery date."""
    windows = json.loads((ROOT / "gallery/app/public/data/weather/filters.json").read_text())["windows"]
    w = pd.DataFrame([(fid, fid.split("-")[0], pd.Timestamp(v[0]), pd.Timestamp(v[1])) for fid, v in windows.items()],
                     columns=["id", "code", "start", "end"])
    have, extra = set(windows), []
    for name in ("spartan_reference.json", "filters.json"):
        for row in json.loads((ROOT / "gallery/app/public/data" / name).read_text())["rows"]:
            if row.get("date") and row.get("code") and row["id"] not in have:
                have.add(row["id"])
                start = pd.Timestamp(row["date"])
                extra.append((row["id"], row["code"], start, start + pd.Timedelta(hours=24)))
    return pd.concat([w, pd.DataFrame(extra, columns=w.columns)], ignore_index=True)


def export_level(level: str, w: pd.DataFrame, smap: dict) -> None:
    out = LEVEL_DIRS[level]
    out.mkdir(parents=True, exist_ok=True)
    (out / "daily").mkdir(exist_ok=True)
    label = "1.5" if level == "15" else "2.0"
    stations = pd.read_csv(CACHE / f"stations_lev{level}.csv")
    chosen = stations[stations.chosen].set_index("code")

    points = {p: load(p, "points", level) for p in ("aod", "sda", "inv")}
    names = list(FIELDS)
    i_fmf, i_ssa = names.index(FMF_FIELD), names.index(SSA_FIELD)
    rows, counts, site_info = {}, {}, {}
    for code, part in w.groupby("code"):
        if code not in chosen.index or code not in smap:
            continue
        off = pd.Timedelta(hours=std_offset_hours(smap[code]["lat"], smap[code]["lon"]))
        starts = (part["start"] - off).to_numpy(dtype="datetime64[ns]")
        ends = (part["end"] - off).to_numpy(dtype="datetime64[ns]")
        values = np.full((len(part), len(names)), np.nan)
        n, aod_days = {}, None
        for product, frame in points.items():
            site_points = frame[frame.site == code].sort_values("time_utc")
            cols = [(i, FIELDS[f][1]) for i, f in enumerate(names) if FIELDS[f][0] == product]
            means, lo, hi = window_means(site_points, [c for _, c in cols], starts, ends)
            n[product] = hi - lo
            for j, (i, _) in enumerate(cols):
                values[:, i] = means[:, j]
            if product == "aod":
                days = site_points["time_utc"].dt.strftime("%Y-%m-%d").to_numpy()
                aod_days = [len(set(days[a:b])) for a, b in zip(lo, hi)]
        types = aerosol_type(values[:, i_fmf], values[:, i_ssa])
        for k, fid in enumerate(part["id"]):
            if not np.isfinite(values[k]).any():
                continue
            rows[fid] = [r(v, FIELDS[f][3]) for f, v in zip(names, values[k])] + [int(types[k]) or None]
            counts[fid] = [int(n["aod"][k]), int(aod_days[k]), int(n["inv"][k])]
        s = chosen.loc[code]
        site_info[code] = dict(station=s.station, km=float(s.km), filters=int(len(part)),
                               with_aod=int(sum(1 for fid in part["id"] if fid in counts and counts[fid][0] > 0)),
                               with_inversion=int(sum(1 for fid in part["id"] if fid in counts and counts[fid][2] > 0)),
                               candidates=stations[stations.code == code][["station", "km", "window_days_with_aod"]].to_dict("records"))
        print(f"  {code} ({s.station}): {site_info[code]['with_aod']} filters with AOD, {site_info[code]['with_inversion']} with an inversion")

    absorption = ("Level 2.0 keeps an inversion's absorption (AAOD, SSA, AAE) only at AOD 440 ≥ 0.4, so clean sites have few or none."
                  if level == "20" else "Level 1.5 is not final-calibrated, and inversions at AOD 440 < 0.4 carry large absorption uncertainty.")
    meta = dict(generated=datetime.now(timezone.utc).isoformat(), level=label,
                source="NASA AERONET Version 3 (aeronet.gsfc.nasa.gov): direct-sun AOD, SDA fine/coarse split, almucantar inversions",
                note=f"Means of the individual retrievals inside each filter's sampling window (local start to end). Daylight, cloud-free retrievals only; {absorption}")
    fields = [*names, TYPE_FIELD]
    payload = dict(**meta, fields=fields, field_units={**{f: FIELDS[f][2] for f in names}, TYPE_FIELD: "class"},
                   aerosol_types={str(k): v for k, v in AEROSOL_TYPES.items()},
                   count_fields=["AOD retrievals", "Days with AOD", "Inversions"], rows=rows, counts=counts)
    if level == "15":
        (out / "filters.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    write_split(payload, out)
    (out / "stations.json").write_text(json.dumps(dict(**meta, sites=site_info), separators=(",", ":"), allow_nan=False))

    daily = {p: load(p, "daily", level) for p in ("aod", "sda", "inv")}
    for old in (out / "daily").glob("*.json"):
        old.unlink()
    for code in site_info:
        parts = []
        for key, (product, col, _) in DAILY.items():
            d = daily[product][daily[product].site == code]
            parts.append(d.set_index(d["time_utc"].dt.strftime("%Y-%m-%d"))[col].rename(key))
        table = pd.concat(parts, axis=1).sort_index()
        # Level 2.0 has many inversion-free days: keep only days with something on them
        table = table.dropna(how="all")
        body = {"date": table.index.tolist(), **{k: [r(v, DAILY[k][2]) for v in table[k]] for k in DAILY}}
        (out / "daily" / f"{code}.json").write_text(json.dumps(body, separators=(",", ":"), allow_nan=False))

    typed = [v[-1] for v in rows.values()]
    print(f"Level {label}: {len(rows)} filters with AERONET at {len(site_info)} sites, "
          f"{sum(1 for c in counts.values() if c[2] > 0)} with an inversion; types "
          + ", ".join(f"{AEROSOL_TYPES[k]} {typed.count(k)}" for k in AEROSOL_TYPES))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", choices=["15", "20"], help="export one level (default: each level whose parquet exists)")
    ap.add_argument("--split-only", action="store_true", help="re-split the Level 1.5 filters.json; nothing else")
    args = ap.parse_args()
    if args.split_only:
        write_split(json.loads((OUT / "filters.json").read_text()))
        return
    levels = [args.level] if args.level else [lv for lv in LEVEL_DIRS if (CACHE / f"inv_points_lev{lv}.parquet").exists()]
    smap = {s["code"]: s for s in json.loads((ROOT / "gallery/app/public/data/spartan_map.json").read_text())["sites"]}
    w = filter_windows()
    for level in levels:
        export_level(level, w, smap)
    size = sum(p.stat().st_size for p in OUT.rglob("*.json"))
    print(f"{size / 1e6:.1f} MB in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
