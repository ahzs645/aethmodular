"""Catalogue what NASA MAIA publishes, and pull its files.

Two public sources, both read without a login:

- The MAIA MMGIS portal (https://maia-mmgis.jpl.nasa.gov/?mission=MAIA): target areas,
  the static context grids drawn for each target, and the surface-monitor lists.
- NASA's Common Metadata Repository (CMR): every MAIA collection archived at the
  ASDC, and every granule's target area, day, version and size.

Downloading granules needs an Earthdata Login token (EARTHDATA_TOKEN, see .env.example).

Output layout:
    research/maia/inventory/CATALOG.md            human-readable rollup
    research/maia/inventory/catalog.json          counts, sources and when it was built
    research/maia/inventory/targets.csv           one row per target area
    research/maia/inventory/layers.csv            one row per portal map layer
    research/maia/inventory/monitors.csv          one row per surface monitor
    research/maia/inventory/collections.csv       one row per CMR collection
    research/maia/inventory/granule_coverage.csv  collection x target: first/last day, days, gaps, size
    research/maia/inventory/granule_monthly.csv   collection x target x month: days with a granule
    research/maia/inventory/spartan_sites.csv     each SPARTAN site: its target, monitors, overlap with filter days
    data/maia/meta/                               raw API responses (gitignored; reused by --offline)
    data/maia/raw/<collection>/<target>/          downloaded granules (gitignored)

Usage:
    uv run aeth maia catalog                       # refresh everything from the portal and CMR
    uv run aeth maia catalog --offline             # rebuild the tables from data/maia/meta
    uv run aeth maia pull --collection MAIA_ANC_SURFACEMONITOR_PM_TOTAL \\
        --target ETH-AddisAbaba --start 2023-01-01 --end 2023-03-31 --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# scripts/ is not an installed package and the CLI runs this file by path, so
# put scripts/ on sys.path to make `common` importable. See scripts/common/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import REPO_ROOT  # noqa: E402

MMGIS = "https://maia-mmgis.jpl.nasa.gov"
MMGIS_MISSION = f"{MMGIS}/Missions/MAIA"
CMR = "https://cmr.earthdata.nasa.gov/search"

OUT_DIR = REPO_ROOT / "research" / "maia" / "inventory"
META_DIR = REPO_ROOT / "data" / "maia" / "meta"
RAW_DIR = REPO_ROOT / "data" / "maia" / "raw"
GALLERY_DATA = REPO_ROOT / "gallery" / "app" / "public" / "data"

TARGET_TYPES = ("pta", "sta", "cvta")
TARGET_TYPE_LABELS = {"PTA": "primary", "STA": "secondary", "CVTA": "calibration/validation"}
# Portal layer file stems -> the monitor list each one is.
MONITOR_LISTS = {
    "combined": "Data/Monitors/monitors_combined_cron.json",
    "speciated": "Data/Monitors/monitors_speciated.json",
    "pm25": "Data/Monitors/monitors_pm25.json",
    "pm10": "Data/Monitors/monitors_pm10.json",
}
TARGET_RE = re.compile(r"(?<![A-Za-z])([A-Z]{3}-[A-Za-z0-9]+)(?![A-Za-z0-9])")
DAY_RE = re.compile(r"_(\d{8})T\d{6}Z")
VERSION_RE = re.compile(r"_(V[0-9A-Za-z]+(?:p[0-9A-Za-z]+)*)\.\w+$")
LAYER_RES_RE = re.compile(r"(\d+\s?k?m)\b")
LAYER_DATE_RE = re.compile(r"_(\d{8})_")

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "aethmodular-maia-catalog"


# ----------------------------------------------------------------------------- fetching

def fetch_json(url: str, cache_name: str, offline: bool, params: dict | None = None) -> object:
    """GET a JSON document, caching it under data/maia/meta/ so --offline can rebuild."""
    path = META_DIR / cache_name
    if offline:
        if not path.exists():
            raise SystemExit(f"--offline: {path.relative_to(REPO_ROOT)} is missing; run without --offline first")
        return json.loads(path.read_text())
    for attempt in range(4):
        try:
            response = SESSION.get(url, params=params, timeout=180)
            response.raise_for_status()
            data = response.json()
            break
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def cmr_collections(offline: bool) -> list[dict]:
    """Every collection whose short name starts with MAIA, plus any MAIA keyword hit at the ASDC."""
    by_id: dict[str, dict] = {}
    queries = [
        ("collections_shortname.json", {"short_name": "MAIA*", "options[short_name][pattern]": "true"}),
        ("collections_keyword.json", {"keyword": "MAIA", "provider": "LARC_CLOUD"}),
    ]
    for name, query in queries:
        data = fetch_json(f"{CMR}/collections.json", name, offline,
                          {**query, "page_size": 200, "include_granule_counts": "true"})
        for entry in data["feed"]["entry"]:
            if str(entry.get("short_name", "")).upper().startswith("MAIA"):
                by_id[entry["id"]] = entry
    return sorted(by_id.values(), key=lambda e: e["short_name"])


def cmr_granules(concept_id: str, offline: bool) -> list[dict]:
    """Every granule of one collection (title, start, end, size, data URL), paged with CMR-Search-After."""
    path = META_DIR / f"granules_{concept_id}.json"
    if offline:
        if not path.exists():
            raise SystemExit(f"--offline: {path.relative_to(REPO_ROOT)} is missing; run without --offline first")
        return json.loads(path.read_text())
    granules: list[dict] = []
    search_after = None
    while True:
        headers = {"CMR-Search-After": search_after} if search_after else {}
        response = SESSION.get(f"{CMR}/granules.json", headers=headers, timeout=300,
                               params={"collection_concept_id": concept_id, "page_size": 2000,
                                       "sort_key": "start_date"})
        response.raise_for_status()
        entries = response.json()["feed"]["entry"]
        for entry in entries:
            granules.append({
                "title": entry.get("title", ""),
                "start": entry.get("time_start"),
                "end": entry.get("time_end"),
                "size_mb": float(entry.get("granule_size") or 0.0),
                "url": next(iter(data_links(entry)), ""),
            })
        search_after = response.headers.get("CMR-Search-After")
        if not entries or not search_after or len(entries) < 2000:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(granules))
    return granules


# ----------------------------------------------------------------------------- parsing

def data_links(entry: dict) -> list[str]:
    """The granule's own file links (not the collection links CMR copies onto every granule)."""
    return [link["href"] for link in entry.get("links", [])
            if link.get("rel", "").endswith("/data#") and not link.get("inherited")
            and link.get("href", "").startswith("https")]


def newest_per_day(titles: list[str]) -> list[str]:
    """Keep the newest version of each target-day; the ASDC keeps every reprocessing."""
    best: dict[tuple[str, str], tuple[str, str]] = {}
    for title in titles:
        parsed = parse_granule_title(title.rsplit("/", 1)[-1])
        key = (parsed["target"], parsed["day"])
        if key not in best or parsed["version"] > best[key][0]:
            best[key] = (parsed["version"], title)
    return [title for _, title in sorted(best.values(), key=lambda v: v[1])]


def parse_granule_title(title: str, known_targets: set[str] | None = None) -> dict:
    """Target area, day and version from a MAIA granule file name.

    >>> parse_granule_title("MAIA_ANC_SURFACEMONITOR_PM_TOTAL_IND-Delhi_20261018T000000Z_F01_V2026p08p26.nc")
    {'target': 'IND-Delhi', 'day': '2026-10-18', 'version': 'V2026p08p26'}
    """
    targets = TARGET_RE.findall(title)
    if known_targets:
        targets = [t for t in targets if t in known_targets] or targets
    day = DAY_RE.search(title)
    version = VERSION_RE.search(title)
    return {
        "target": targets[0] if targets else "",
        "day": f"{day.group(1)[:4]}-{day.group(1)[4:6]}-{day.group(1)[6:]}" if day else "",
        "version": version.group(1) if version else "",
    }


def rings(geometry: dict) -> list[list[list[float]]]:
    """Outer rings of a GeoJSON geometry. The portal draws target outlines as LineStrings."""
    kind, coords = geometry.get("type"), geometry.get("coordinates", [])
    if kind == "LineString":
        return [coords]
    if kind in ("MultiLineString",):
        return coords
    if kind == "Polygon":
        return coords[:1]
    if kind == "MultiPolygon":
        return [polygon[0] for polygon in coords]
    return []


def point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    for i in range(len(ring)):
        x1, y1 = ring[i][:2]
        x2, y2 = ring[i - 1][:2]
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def coverage(days: list[str]) -> dict:
    """First and last day, distinct days, the span, and the days inside the span with no granule."""
    distinct = sorted(set(d for d in days if d))
    if not distinct:
        return {"first": "", "last": "", "n_days": 0, "span_days": 0, "missing_days": 0, "longest_gap_days": 0}
    parsed = [date.fromisoformat(d) for d in distinct]
    span = (parsed[-1] - parsed[0]).days + 1
    gaps = [(b - a).days - 1 for a, b in zip(parsed, parsed[1:])]
    return {
        "first": distinct[0],
        "last": distinct[-1],
        "n_days": len(distinct),
        "span_days": span,
        "missing_days": span - len(distinct),
        "longest_gap_days": max(gaps, default=0),
    }


def layer_row(target: str, layer: dict) -> dict:
    name = layer.get("name", "")
    # Layer names start with the target or its country code (sometimes mistyped, e.g. "VNN" for VNM-Hanoi).
    head, _, rest = name.partition(" ")
    label = rest if rest and re.fullmatch(r"[A-Z]{3}(-[A-Za-z0-9]+)?", head) else name
    url = layer.get("url", "") or ""
    res = LAYER_RES_RE.search(label)
    stamp = LAYER_DATE_RE.search(url)
    return {
        "target": target,
        "layer": re.sub(r"(\s+\d+\s?k?m)+$", "", label.replace("_", " ")).strip(),
        "resolution": res.group(1).replace(" ", "") if res else "",
        "date": f"{stamp.group(1)[:4]}-{stamp.group(1)[4:6]}-{stamp.group(1)[6:]}" if stamp else "",
        "kind": layer.get("type", ""),
        "url": url if url.startswith("http") else f"{MMGIS_MISSION}/{url}",
    }


# ----------------------------------------------------------------------------- catalogue

def load_targets(offline: bool) -> tuple[pd.DataFrame, dict[str, list]]:
    centres = fetch_json(f"{MMGIS_MISSION}/Layers/targets.json", "targets.json", offline)
    outlines: dict[str, list] = {}
    kinds: dict[str, str] = {}
    for kind in TARGET_TYPES:
        data = fetch_json(f"{MMGIS_MISSION}/Layers/target_polygons_{kind}.json", f"target_polygons_{kind}.json", offline)
        for feature in data["features"]:
            name = feature["properties"]["name"].strip()
            outlines[name] = rings(feature["geometry"])
            kinds[name] = kind.upper()
    rows = []
    for feature in centres["features"]:
        p = feature["properties"]
        name = str(p.get("target_short_name", "")).strip()
        if not name:
            continue
        lons = [pt[0] for ring in outlines.get(name, []) for pt in ring]
        lats = [pt[1] for ring in outlines.get(name, []) for pt in ring]
        rows.append({
            "target": name,
            "long_name": p.get("target_long_name", ""),
            "type": p.get("target_type") or kinds.get(name, ""),
            "status": p.get("target_status", ""),
            "activated": p.get("date_last_activated", ""),
            "deactivated": p.get("date_last_deactivated", ""),
            "center_lat": p.get("target_center_lat_deg"),
            "center_lon": p.get("target_center_lon_deg"),
            "outline_west": min(lons) if lons else None,
            "outline_east": max(lons) if lons else None,
            "outline_south": min(lats) if lats else None,
            "outline_north": max(lats) if lats else None,
            "level4_box_km": "x".join(str(p.get(k) or "") for k in ("level4_bb_ew_dist_km", "level4_bb_ns_dist_km")).strip("x"),
        })
    return pd.DataFrame(rows).sort_values(["type", "target"]).reset_index(drop=True), outlines


def target_of(lat: float, lon: float, outlines: dict[str, list], types: dict[str, str]) -> str:
    """The target whose outline contains the point; a primary target wins over a secondary one."""
    order = {"PTA": 0, "STA": 1, "CVTA": 2}
    hits = [name for name, rs in outlines.items() if any(point_in_ring(lon, lat, r) for r in rs)]
    return min(hits, key=lambda n: (order.get(types.get(n, ""), 9), n)) if hits else ""


def load_layers(config: dict) -> pd.DataFrame:
    rows = []
    group = next((layer for layer in config["layers"] if layer.get("name") == "Targets"), {"sublayers": []})
    for target in group.get("sublayers", []):
        name = target.get("name", "").strip()
        for layer in target.get("sublayers", []) or []:
            rows.append(layer_row(name, layer))
    return pd.DataFrame(rows)


def load_monitors(offline: bool, outlines: dict, types: dict) -> pd.DataFrame:
    by_id: dict[str, dict] = {}
    for list_name, rel in MONITOR_LISTS.items():
        data = fetch_json(f"{MMGIS_MISSION}/{rel}", f"monitors_{list_name}.json", offline)
        for feature in data["features"]:
            p = feature["properties"]
            lat, lon = p.get("latitude"), p.get("longitude")
            key = p.get("maiaMonitorId") or f"{lat}_{lon}_{p.get('Monitor Name')}"
            row = by_id.setdefault(key, {
                "monitor_id": key, "name": str(p.get("Monitor Name", "")).strip(),
                "lat": lat, "lon": lon, "source": p.get("Source", ""), "source_code": p.get("Source Name", ""),
                "lists": [],
            })
            row["lists"].append(list_name)
    frame = pd.DataFrame(by_id.values())
    frame["lists"] = frame["lists"].map(lambda v: ";".join(sorted(set(v))))
    frame["target"] = [target_of(lat, lon, outlines, types) if pd.notna(lat) else ""
                       for lat, lon in zip(frame["lat"], frame["lon"])]
    return frame.sort_values(["target", "source", "name"]).reset_index(drop=True)


def load_spartan_sites() -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """SPARTAN sites and their filter days, from the committed gallery exports."""
    sites = json.loads((GALLERY_DATA / "spartan_map.json").read_text())["sites"]
    days: dict[str, set[str]] = defaultdict(set)
    for name in ("spartan_reference.json", "filters.json"):
        path = GALLERY_DATA / name
        if path.exists():
            for row in json.loads(path.read_text())["rows"]:
                if row.get("date"):
                    days[row["code"]].add(row["date"])
    frame = pd.DataFrame([{"code": s["code"], "name": s["name"], "lat": s["lat"], "lon": s["lon"]} for s in sites])
    return frame, days


def build_catalog(offline: bool) -> dict:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    config = fetch_json(f"{MMGIS}/api/configure/get", "mmgis_config.json", offline, {"mission": "MAIA"})
    config = config.get("config", config)

    targets, outlines = load_targets(offline)
    types = dict(zip(targets["target"], targets["type"]))
    known = set(targets["target"]) | set(outlines)
    layers = load_layers(config)
    monitors = load_monitors(offline, outlines, types)

    collections = cmr_collections(offline)
    coll_rows, cov_rows, month_rows = [], [], []
    granule_days: dict[tuple[str, str], set[str]] = defaultdict(set)
    for entry in collections:
        granules = cmr_granules(entry["id"], offline)
        newest = set(newest_per_day([g["title"] for g in granules]))
        coll_rows.append({
            "short_name": entry["short_name"],
            "version": entry.get("version_id", ""),
            "concept_id": entry["id"],
            "title": entry.get("title", ""),
            "processing_level": entry.get("processing_level_id", ""),
            "platforms": ";".join(entry.get("platforms", []) or []),
            "time_start": (entry.get("time_start") or "")[:10],
            "time_end": (entry.get("time_end") or "")[:10] or "ongoing",
            "granules": len(granules),
            "size_gb": round(sum(g["size_mb"] for g in granules) / 1024, 2),
            "size_gb_newest": round(sum(g["size_mb"] for g in granules
                                        if g["title"] in newest) / 1024, 2),
            "doi": next((link.get("href", "") for link in entry.get("links", []) if "doi.org" in link.get("href", "")), ""),
            "earthdata_search": f"https://search.earthdata.nasa.gov/search/granules?p={entry['id']}",
        })
        per_target: dict[str, list[dict]] = defaultdict(list)
        for granule in granules:
            parsed = parse_granule_title(granule["title"], known)
            per_target[parsed["target"] or "(none)"].append({**granule, **parsed})
        for target, items in sorted(per_target.items()):
            days = [g["day"] or (g["start"] or "")[:10] for g in items]
            granule_days[(entry["short_name"], target)].update(d for d in days if d)
            cov_rows.append({
                "collection": entry["short_name"], "target": target, "type": types.get(target, ""),
                **coverage(days), "granules": len(items),
                "size_mb": round(sum(g["size_mb"] for g in items), 1),
                "versions": ";".join(sorted({g["version"] for g in items if g["version"]})),
            })
            months: dict[str, set[str]] = defaultdict(set)
            for d in days:
                if d:
                    months[d[:7]].add(d)
            for month, ds in sorted(months.items()):
                month_rows.append({"collection": entry["short_name"], "target": target, "month": month, "days": len(ds)})

    sites, filter_days = load_spartan_sites()
    spartan_monitors = monitors[monitors["source"].str.contains("SPARTAN", case=False, na=False)]
    site_rows = []
    for site in sites.itertuples():
        target = target_of(site.lat, site.lon, outlines, types)
        fdays = filter_days.get(site.code, set())
        listed = [haversine_km(site.lat, site.lon, m.lat, m.lon) for m in spartan_monitors.itertuples()]
        others = monitors[~monitors.index.isin(spartan_monitors.index)]
        nearest = min(((haversine_km(site.lat, site.lon, m.lat, m.lon), m.source) for m in others.itertuples()),
                      default=(float("nan"), ""))
        row = {
            "code": site.code, "name": site.name, "lat": site.lat, "lon": site.lon,
            "target": target, "target_type": types.get(target, ""),
            "filter_days": len(fdays),
            "filter_first": min(fdays, default=""), "filter_last": max(fdays, default=""),
            "listed_as_maia_monitor": bool(listed) and min(listed) <= 2.0,
            "nearest_other_monitor_km": round(nearest[0], 1) if not math.isnan(nearest[0]) else None,
            "nearest_other_monitor_source": nearest[1],
            "monitors_in_target": int((monitors["target"] == target).sum()) if target else 0,
        }
        for coll in coll_rows:
            days = granule_days.get((coll["short_name"], target), set()) if target else set()
            row[f"{coll['short_name']}_filter_days"] = len(fdays & days)
        site_rows.append(row)

    return {
        "generated": generated,
        "targets": targets, "layers": layers, "monitors": monitors,
        "collections": pd.DataFrame(coll_rows), "coverage": pd.DataFrame(cov_rows),
        "monthly": pd.DataFrame(month_rows), "spartan": pd.DataFrame(site_rows),
    }


# ----------------------------------------------------------------------------- report

def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_none_\n"
    cells = frame.astype(object).where(frame.notna(), "")
    lines = ["| " + " | ".join(map(str, cells.columns)) + " |", "|" + "---|" * len(cells.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in cells.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def write_outputs(cat: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, name in [("targets", "targets.csv"), ("layers", "layers.csv"), ("monitors", "monitors.csv"),
                      ("collections", "collections.csv"), ("coverage", "granule_coverage.csv"),
                      ("monthly", "granule_monthly.csv"), ("spartan", "spartan_sites.csv")]:
        cat[key].to_csv(OUT_DIR / name, index=False)

    targets, layers, monitors = cat["targets"], cat["layers"], cat["monitors"]
    collections, cov, spartan = cat["collections"], cat["coverage"], cat["spartan"]
    summary = {
        "generated": cat["generated"],
        "sources": {"mmgis": f"{MMGIS}/?mission=MAIA", "cmr": CMR},
        "targets": targets["type"].value_counts().to_dict(),
        "layers": int(len(layers)),
        "monitors": int(len(monitors)),
        "monitors_by_source": monitors["source"].value_counts().to_dict(),
        "collections": collections[["short_name", "version", "granules", "time_start", "time_end"]].to_dict("records"),
        "spartan_sites_in_targets": int((spartan["target"] != "").sum()),
    }
    (OUT_DIR / "catalog.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")

    layer_kinds = (layers.groupby(["layer", "resolution"])
                   .agg(targets=("target", "nunique"),
                        image_dates=("date", lambda s: " → ".join(sorted({min(d for d in s if d), max(d for d in s if d)})) if any(s) else ""))
                   .reset_index().sort_values("targets", ascending=False))
    per_target = targets[["target", "type", "status", "activated"]].copy()
    per_target["monitors"] = per_target["target"].map(monitors["target"].value_counts()).fillna(0).astype(int)
    per_target["layers"] = per_target["target"].map(layers["target"].value_counts()).fillna(0).astype(int)
    per_target["spartan_sites"] = per_target["target"].map(
        spartan[spartan["target"] != ""].groupby("target")["code"].agg(", ".join)).fillna("")
    for coll in collections["short_name"]:
        sub = cov[cov["collection"] == coll].set_index("target")
        per_target[coll.replace("MAIA_", "")] = per_target["target"].map(
            lambda t: f"{sub.at[t, 'first']} → {sub.at[t, 'last']} ({sub.at[t, 'n_days']} d)" if t in sub.index else "")
    source_counts = monitors.groupby("source").agg(monitors=("monitor_id", "size"),
                                                   in_a_target=("target", lambda s: int((s != "").sum()))).reset_index()
    site_cols = ["code", "name", "target", "target_type", "filter_days", "filter_first", "filter_last",
                 "listed_as_maia_monitor", "nearest_other_monitor_km", "nearest_other_monitor_source"]
    site_cols += [c for c in spartan.columns if c.endswith("_filter_days") and c != "filter_days"]

    text = f"""# MAIA catalogue

Generated by `scripts/pipelines/maia_catalog.py` (`uv run aeth maia catalog`) on {cat['generated']}.
Sources: the MAIA MMGIS portal ({MMGIS}/?mission=MAIA) and NASA CMR ({CMR}).
Both are public; downloading granules needs an Earthdata Login token (`aeth maia pull`).

## Archived products (CMR, ASDC)

{md_table(collections[['short_name', 'version', 'processing_level', 'time_start', 'time_end', 'granules', 'size_gb', 'size_gb_newest', 'doi']])}
Granules are one file per target area per day, and the ASDC keeps every
reprocessed version: `size_gb` counts them all, `size_gb_newest` only the newest
version of each target-day (what `aeth maia pull` downloads by default). `granule_coverage.csv` gives each
collection × target its first and last day, days present, missing days and
longest gap; `granule_monthly.csv` gives days per month.

## Target areas

{md_table(per_target)}
## Portal map layers

Static context grids, drawn as map tiles for each target (`layers.csv` has every tile URL).

{md_table(layer_kinds)}
## Surface monitors

{len(monitors):,} monitors across the portal's combined, speciated, PM2.5 and PM10 lists
(`monitors.csv`; `target` is the target outline the monitor falls in).

{md_table(source_counts.sort_values('monitors', ascending=False))}
## SPARTAN sites

`target` is the MAIA outline the site falls in (a primary target wins over a
secondary one). `*_filter_days` counts the site's filter days on which that
collection has a granule for the target.

{md_table(spartan[site_cols])}"""
    (OUT_DIR / "CATALOG.md").write_text(text)


# ----------------------------------------------------------------------------- pull

def pull(args: argparse.Namespace) -> int:
    concept = None
    for entry in cmr_collections(offline=False):
        if entry["short_name"] == args.collection:
            concept = entry["id"]
    if not concept:
        raise SystemExit(f"no CMR collection named {args.collection}; see research/maia/inventory/collections.csv")
    params = {"collection_concept_id": concept, "page_size": 2000, "sort_key": "start_date",
              "temporal": f"{args.start}T00:00:00Z,{args.end}T23:59:59Z"}
    if args.target:
        params["readable_granule_name"] = f"*{args.target}*"
        params["options[readable_granule_name][pattern]"] = "true"
    response = SESSION.get(f"{CMR}/granules.json", params=params, timeout=300)
    response.raise_for_status()
    urls = [url for entry in response.json()["feed"]["entry"] for url in data_links(entry)]
    # CMR's temporal filter matches any overlap, so a granule ending at 00:00 on --start comes back too.
    urls = [u for u in urls if args.start <= (parse_granule_title(u.rsplit("/", 1)[-1])["day"] or args.start) <= args.end]
    if not args.all_versions:
        urls = newest_per_day(urls)
    print(f"{len(urls)} granules for {args.collection} {args.target or '(all targets)'} {args.start}..{args.end}")
    if args.dry_run or not urls:
        for url in urls:
            print(url)
        return 0
    token = os.environ.get("EARTHDATA_TOKEN")
    if not token:
        raise SystemExit("set EARTHDATA_TOKEN (an Earthdata Login user token; see .env.example) to download")
    headers = {"Authorization": f"Bearer {token}"}
    for url in urls:
        target = parse_granule_title(url.rsplit("/", 1)[-1])["target"] or "all"
        dest = RAW_DIR / args.collection / target / url.rsplit("/", 1)[-1]
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with SESSION.get(url, headers=headers, stream=True, timeout=600) as r:
            if r.status_code in (401, 403) or "text/html" in r.headers.get("Content-Type", ""):
                raise SystemExit(f"Earthdata refused {url} ({r.status_code}); check EARTHDATA_TOKEN")
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
        print(f"  {dest.relative_to(REPO_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    cat = sub.add_parser("catalog", help="Rebuild research/maia/inventory from the portal and CMR.")
    cat.add_argument("--offline", action="store_true", help="Rebuild from the cached responses in data/maia/meta.")
    get = sub.add_parser("pull", help="Download granules for one collection (needs EARTHDATA_TOKEN).")
    get.add_argument("--collection", required=True, help="CMR short name, e.g. MAIA_ANC_SURFACEMONITOR_PM_TOTAL")
    get.add_argument("--target", default="", help="Target area, e.g. ETH-AddisAbaba (default: all)")
    get.add_argument("--start", default="2021-01-01")
    get.add_argument("--end", default=date.today().isoformat())
    get.add_argument("--dry-run", action="store_true", help="List the granule URLs without downloading.")
    get.add_argument("--all-versions", action="store_true", help="Every reprocessed version, not only the newest per day.")
    args = parser.parse_args(argv)
    if args.command == "pull":
        return pull(args)
    write_outputs(build_catalog(offline=getattr(args, "offline", False)))
    print(f"wrote {OUT_DIR.relative_to(REPO_ROOT)}/CATALOG.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
