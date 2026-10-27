"""AERONET over each filter's sampling window, for the gallery's AERONET tab and AERONET fields.

Run with:  uv run --with timezonefinder python gallery/data/export_aeronet.py

Inputs (all local; research/ftir_hips_chem/workflows/fetch_aeronet_spartan.py makes them):
  research/ftir_hips_chem/output/tables/aeronet_spartan/{aod,sda,inv}_{points,daily}_lev15.parquet
  research/ftir_hips_chem/output/tables/aeronet_spartan/stations_lev15.csv
  gallery/app/public/data/weather/filters.json   each filter's local sampling window (export_weather.py)

Each filter's value is the mean of the individual retrievals inside its sampling window (local start
to end, converted to UTC with the site's standard offset), as the weather export does with ERA5 hours.
Sun photometers retrieve only in daylight and clear sky, so a window's mean covers its clear daytime
hours; a filter with no retrieval in its window gets no value. Older intermittent SPARTAN filters
spread 24 h of sampling over about 8 days, and the window is then the whole elapsed period.

Writes to gallery/app/public/data/aeronet/:
  filters.json        per filter: the fields below, plus retrieval counts (the whole set, for scripts)
  index.json + filters/<CODE>.json   the same split by site: the gallery loads only the sites in view
  daily/<CODE>.json   per site: AERONET's own daily means, for the timeline and the climatology
  stations.json       per site: the station pulled, its distance and how much of the filter record it covers

Data: NASA AERONET Version 3 (aeronet.gsfc.nasa.gov), Level 1.5 unless the cache says otherwise.
"""

from __future__ import annotations

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
}
# daily series key -> (product, parquet column, decimals)
DAILY = {
    "aod440": ("aod", "aod440", 3), "aod500": ("aod", "aod500", 3), "aod675": ("aod", "aod675", 3),
    "ae": ("aod", "ae440_870", 2), "pw": ("aod", "pw_cm", 2), "fmf": ("sda", "fmf500", 2),
    "aaod440": ("inv", "aaod440", 4), "aaod675": ("inv", "aaod675", 4), "aae": ("inv", "aae440_870", 2),
    "ssa440": ("inv", "ssa440", 3), "ssa675": ("inv", "ssa675", 3),
}


def load(product: str, kind: str) -> pd.DataFrame:
    frame = pd.read_parquet(CACHE / f"{product}_{kind}_lev{LEVEL}.parquet")
    frame["time_utc"] = pd.to_datetime(frame["time_utc"])
    return frame


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


def write_split(payload: dict) -> None:
    """filters.json split by site (a filter id starts with its site code), as export_weather does."""
    rows, counts = {}, {}
    for fid, v in payload["rows"].items():
        rows.setdefault(fid.split("-")[0], {})[fid] = v
    for fid, v in payload["counts"].items():
        counts.setdefault(fid.split("-")[0], {})[fid] = v
    folder = OUT / "filters"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.json"):
        old.unlink()
    for code in sorted(rows):
        (folder / f"{code}.json").write_text(json.dumps(dict(rows=rows[code], counts=counts.get(code, {})),
                                                        separators=(",", ":"), allow_nan=False))
    index = {k: v for k, v in payload.items() if k not in ("rows", "counts")}
    index["sites"] = {code: len(v) for code, v in sorted(rows.items())}
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":"), allow_nan=False))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "daily").mkdir(exist_ok=True)
    stations = pd.read_csv(CACHE / f"stations_lev{LEVEL}.csv")
    chosen = stations[stations.chosen].set_index("code")
    smap = {s["code"]: s for s in json.loads((ROOT / "gallery/app/public/data/spartan_map.json").read_text())["sites"]}
    windows = json.loads((ROOT / "gallery/app/public/data/weather/filters.json").read_text())["windows"]
    w = pd.DataFrame([(fid, fid.split("-")[0], pd.Timestamp(v[0]), pd.Timestamp(v[1])) for fid, v in windows.items()],
                     columns=["id", "code", "start", "end"])
    # filters the weather export has no window for get 24 h from their gallery date
    have, extra = set(windows), []
    for name in ("spartan_reference.json", "filters.json"):
        for row in json.loads((ROOT / "gallery/app/public/data" / name).read_text())["rows"]:
            if row.get("date") and row.get("code") and row["id"] not in have:
                have.add(row["id"])
                start = pd.Timestamp(row["date"])
                extra.append((row["id"], row["code"], start, start + pd.Timedelta(hours=24)))
    w = pd.concat([w, pd.DataFrame(extra, columns=w.columns)], ignore_index=True)

    points = {p: load(p, "points") for p in ("aod", "sda", "inv")}
    names = list(FIELDS)
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
        for k, fid in enumerate(part["id"]):
            if not np.isfinite(values[k]).any():
                continue
            rows[fid] = [r(v, FIELDS[f][3]) for f, v in zip(names, values[k])]
            counts[fid] = [int(n["aod"][k]), int(aod_days[k]), int(n["inv"][k])]
        s = chosen.loc[code]
        site_info[code] = dict(station=s.station, km=float(s.km), filters=int(len(part)),
                               with_aod=int(sum(1 for fid in part["id"] if fid in counts and counts[fid][0] > 0)),
                               with_inversion=int(sum(1 for fid in part["id"] if fid in counts and counts[fid][2] > 0)),
                               candidates=stations[stations.code == code][["station", "km", "window_days_with_aod"]].to_dict("records"))
        print(f"{code} ({s.station}): {site_info[code]['with_aod']} filters with AOD, {site_info[code]['with_inversion']} with an inversion")

    meta = dict(generated=datetime.now(timezone.utc).isoformat(), level="1.5" if LEVEL == "15" else "2.0",
                source="NASA AERONET Version 3 (aeronet.gsfc.nasa.gov): direct-sun AOD, SDA fine/coarse split, almucantar inversions",
                note="Means of the individual retrievals inside each filter's sampling window (local start to end). Daylight, cloud-free retrievals only; Level 1.5 is not final-calibrated, and inversions at AOD 440 < 0.4 carry large absorption uncertainty.")
    payload = dict(**meta, fields=names, field_units={f: FIELDS[f][2] for f in names},
                   count_fields=["AOD retrievals", "Days with AOD", "Inversions"], rows=rows, counts=counts)
    (OUT / "filters.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    write_split(payload)
    (OUT / "stations.json").write_text(json.dumps(dict(**meta, sites=site_info), separators=(",", ":"), allow_nan=False))

    daily = {p: load(p, "daily") for p in ("aod", "sda", "inv")}
    for code in site_info:
        parts = []
        for key, (product, col, _) in DAILY.items():
            d = daily[product][daily[product].site == code]
            parts.append(d.set_index(d["time_utc"].dt.strftime("%Y-%m-%d"))[col].rename(key))
        table = pd.concat(parts, axis=1).sort_index()
        out = {"date": table.index.tolist(), **{k: [r(v, DAILY[k][2]) for v in table[k]] for k in DAILY}}
        (OUT / "daily" / f"{code}.json").write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))

    size = sum(p.stat().st_size for p in OUT.rglob("*.json"))
    print(f"{len(rows)} filters with AERONET at {len(site_info)} sites; {size / 1e6:.1f} MB written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    if "--split-only" in sys.argv:
        write_split(json.loads((OUT / "filters.json").read_text()))
    else:
        main()
