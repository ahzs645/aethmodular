"""Package existing notebook coverage tables and recompute study sample-day overlap.

Run from the repository: uv run python gallery/data/export_availability.py
No notebooks are rewritten and no remote data are downloaded. Saved network
tables retain their own population/version; the four-site study is recomputed
from the canonical filter and processed aethalometer loaders.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research/ftir_hips_chem/scripts"))
from config import FILTER_DATA_PATH, PROCESSED_SITES_DIR, SITES
from data_matching import add_base_filter_id, load_aethalometer_data, load_filter_data
from site_labels import display_site_name

OUT = REPO / "gallery/app/public/data/availability.json"
INVENTORY = REPO / "research/spartan/inventory"
OVERVIEW = REPO / "research/spartan/all_sites_overview/tables"
SOURCES: dict[str, dict] = {}


def source(path: Path, role: str) -> None:
    try:
        name = path.relative_to(REPO).as_posix()
    except ValueError:
        # Never leak an external machine-specific data-root path into an export.
        name = f"AETHMODULAR_DATA_ROOT/{path.name}"
    SOURCES[name] = {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "role": role}


def table(path: Path) -> list[dict]:
    source(path, "Saved notebook/pipeline table")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def dates(values) -> list[str]:
    """Local calendar labels; never shift a local day through UTC."""
    result = pd.to_datetime(values, errors="coerce").dropna()
    return sorted({v.strftime("%Y-%m-%d") for v in result})


def build_network():
    coverage = table(INVENTORY / "coverage.csv")
    monthly = table(INVENTORY / "monthly_samples.csv")
    registry = table(OVERVIEW / "all_site_inventory_summary.csv")
    map_path = OUT.with_name("spartan_map.json")
    map_names = {}
    if map_path.exists():
        source(map_path, "Public-header site names, including sites absent from the older lookup")
        map_names = {s["code"]: s["name"] for s in json.loads(map_path.read_text())["sites"]}
    hips = table(INVENTORY / "hips_coverage_by_site.csv")
    hips_monthly = table(OVERVIEW / "hips_fabs_monthly_counts.csv")
    bridge = table(INVENTORY / "hips_vs_public_link.csv")
    assert len({(r["site"], r["FilterId_base"]) for r in bridge}) == len(bridge), "Duplicate physical IDs in HIPS bridge"
    latest = table(INVENTORY / "overview.csv")
    release_date = max(r["remote_last_modified"][:10] for r in latest if r["remote_last_modified"])
    future_endpoints = []
    products = [
        ("ChemSpecPM25", "PM2.5 chemistry", "filters"),
        ("ChemSpecPM10", "PM10 chemistry", "filters"),
        ("ReconstrPM25", "PM2.5 reconstruction", "filters"),
        ("DailyScaPM25", "Daily PM2.5 scattering", "daily timestamps"),
        ("DailyScaPM10", "Daily PM10 scattering", "daily timestamps"),
        ("HourlyScaPM25", "Hourly PM2.5 scattering", "hourly timestamps"),
        ("HourlyScaPM10", "Hourly PM10 scattering", "hourly timestamps"),
        ("DailyEstPM25", "Daily PM2.5 estimate", "daily timestamps"),
        ("HourlyEstPM25", "Hourly PM2.5 estimate", "hourly timestamps"),
        ("HIPS", "HIPS Fabs samples", "filters"),
    ]
    sites = {r["site"]: {"code": r["site"], "name": display_site_name(r["site"], map_names.get(r["site"]), r["lookup_name"]),
                             "blanks": int(float(r["hips_blank_filters"] or 0)), "products": {}}
             for r in registry}
    newer = {(r["site"], r["subproduct"]): r for r in latest}
    for row in coverage:
        site = sites.setdefault(row["site"], {"code": row["site"], "name": row["site"], "blanks": 0, "products": {}})
        info = {k: number(row[k]) for k in ["n_samples", "completeness_pct", "median_interval_h", "p95_interval_h", "max_gap_days"]}
        info.update({"date_min": row["date_min"][:10] or None, "date_max": row["date_max"][:10] or None})
        latest_end = newer.get((row["site"], row["subproduct"]), {}).get("date_max", "")[:10]
        if latest_end > release_date:
            future_endpoints.append({"code": row["site"], "product": row["subproduct"], "advertised_date": latest_end})
            latest_end = ""
        # The pull inventory and cadence analysis are separate snapshots.
        info["newer_date_max"] = latest_end if latest_end and info["date_max"] and latest_end > info["date_max"] and latest_end < "2030" else None
        site["products"][row["subproduct"]] = info
    for row in hips:
        sites[row["Site"]]["products"]["HIPS"] = {
            "n_samples": 0, "date_min": None, "date_max": None,
            "inventory_n_samples": int(float(row["n_filters"])), "inventory_date_max": row["date_max"][:10],
            "completeness_pct": None, "median_interval_h": None, "p95_interval_h": None, "max_gap_days": None,
            "newer_date_max": None,
        }
    month_rows = [{"code": r["site"], "product": r["subproduct"], "month": r["month"][:7], "count": int(float(r["samples"]))} for r in monthly]
    bridge_months, positive_months = {}, {}
    for row in bridge:
        value = number(row["fabs_mean"])
        if value is None or not row["sample_date"]:
            continue
        day = row["sample_date"][:10]
        p = sites[row["site"]]["products"]["HIPS"]
        p["n_samples"] += 1
        p["date_min"] = min(p["date_min"] or day, day)
        p["date_max"] = max(p["date_max"] or day, day)
        key = (row["site"], day[:7])
        bridge_months[key] = bridge_months.get(key, 0) + 1
        if value > 0:
            positive_months[key] = positive_months.get(key, 0) + 1
    # Reconcile the notebook's positive-only heatmap, then broaden availability
    # to all finite values. Zero/negative measurements are not missing data.
    for row in hips_monthly:
        for month, value in row.items():
            if month != "site":
                assert positive_months.get((row["site"], month), 0) == int(float(value))
    month_rows += [{"code": code, "product": "HIPS", "month": month, "count": count}
                   for (code, month), count in sorted(bridge_months.items())]
    # Catch stale/mixed tables rather than silently publishing inconsistent totals.
    totals = {}
    for r in month_rows:
        key = (r["code"], r["product"])
        totals[key] = totals.get(key, 0) + r["count"]
    for site in sites.values():
        for product, info in site["products"].items():
            assert totals.get((site["code"], product), 0) == (info["n_samples"] or 0), (site["code"], product, "monthly total differs from inventory")
    return {"products": [{"key": k, "label": label, "unit": unit} for k, label, unit in products],
            "sites": sorted(sites.values(), key=lambda r: r["code"]), "monthly": month_rows,
            "public_release": release_date, "future_inventory_endpoints": future_endpoints,
            "hips_inventory_total": sum(int(float(r["n_filters"])) for r in hips),
            "hips_nonpositive": sum(1 for r in bridge if number(r["fabs_mean"]) is not None and float(r["fabs_mean"]) <= 0)}


def build_study():
    source(FILTER_DATA_PATH, "Canonical unified filter input")
    filters = add_base_filter_id(load_filter_data())
    aeth = load_aethalometer_data()
    sites = []
    for name, cfg in SITES.items():
        code = cfg["code"]
        streams, ids, finite_rows = {}, {}, {}
        for key, src, param in [("ftir", "FTIR", "EC_ftir"), ("hips", "HIPS", "HIPS_Fabs")]:
            rows = filters[(filters.Site == code) & (filters.DataSource == src) & (filters.Parameter == param)]
            finite = rows[np.isfinite(pd.to_numeric(rows.Concentration, errors="coerce"))]
            finite_rows[key] = finite
            ids[key] = set(finite.base_filter_id.dropna())
            streams[key] = {"days": dates(finite.SampleDate), "reported_days": len(dates(rows.SampleDate)),
                            "filters": len(ids[key]), "analysis_last": max(dates(rows.AnalysisDate), default=None)}
        shared_ids = ids["ftir"] & ids["hips"]
        paired = []
        conflicts = 0
        for fid in sorted(shared_ids):
            ftir_days = set(dates(finite_rows["ftir"].loc[finite_rows["ftir"].base_filter_id == fid, "SampleDate"]))
            hips_days = set(dates(finite_rows["hips"].loc[finite_rows["hips"].base_filter_id == fid, "SampleDate"]))
            common = sorted(ftir_days & hips_days)
            conflicts += not bool(common)
            paired.append({"id": fid, "days": common})
        df = aeth.get(name)
        aeth_info = None
        if df is not None:
            source(PROCESSED_SITES_DIR / cfg["file"], "Canonical processed 9am aethalometer input")
            finite_ir = np.isfinite(pd.to_numeric(df["IR BCc"], errors="coerce"))
            completeness = pd.to_numeric(df["data_completeness_pct"], errors="coerce") if "data_completeness_pct" in df else None
            aeth_info = {"record_days": dates(df.day_9am), "days": dates(df.loc[finite_ir, "day_9am"]),
                         "days50": dates(df.loc[finite_ir & (completeness >= 50), "day_9am"]) if completeness is not None else None,
                         "days80": dates(df.loc[finite_ir & (completeness >= 80), "day_9am"]) if completeness is not None else None,
                         "mean_completeness": number(completeness.mean()) if completeness is not None else None,
                         "over100_days": int((completeness > 100).sum()) if completeness is not None else None,
                         "file": cfg["file"]}
        sites.append({"code": code, "name": name.replace("_", " "), "color": cfg["color"],
                      **streams, "aeth": aeth_info, "paired_filters": paired, "date_conflicts": int(conflicts)})
    return sites


def main():
    source(Path(__file__), "Coverage exporter")
    source(Path(__file__).with_name("site_labels.py"), "Readable site-name formatting")
    for helper in ["config.py", "data_matching.py"]:
        source(REPO / "research/ftir_hips_chem/scripts" / helper, "Shared analysis configuration/loader")
    notebooks = [
        "notebooks/analysis/data_availability/filter_data_availability_strip_chart.ipynb",
        "research/ftir_hips_chem/Multi_Site_Analysis.ipynb",
        "research/spartan/spartan_all_sites_hips_overview.ipynb",
        "research/spartan/spartan_improve_orientation.ipynb",
    ]
    for path in notebooks:
        source(REPO / path, "Notebook informing the view and counting rules")
    network = build_network()
    study = build_study()
    orientation = table(REPO / "research/spartan/orientation/tables/shared_coverage_summary.csv")
    for r in orientation:
        r["n"] = int(r["n"])
        r["n_sites"] = int(r["n_sites"])
    result = {"schema_version": 1, "generated": datetime.now(timezone.utc).isoformat(),
              "network": network, "study": study, "orientation": orientation, "sources": list(SOURCES.values())}
    OUT.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Wrote {OUT.relative_to(REPO)}: {len(network['sites'])} inventory sites, {len(study)} study sites")


if __name__ == "__main__":
    main()
