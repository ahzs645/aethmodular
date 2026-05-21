"""Analyze SPARTAN coverage: per site x subproduct, compute date span,
unique sample timestamps, completeness vs the expected cadence, and
average + median interval between samples. Also list, for each site,
which filter-based parameters are present and how many filters carry
each parameter.

Inputs come from data/spartan/raw/ (already downloaded by
spartan_pull_and_summarize.py).

Outputs land in research/spartan/inventory/:
    coverage.csv              one row per (site, subproduct)
    site_parameter_matrix.csv site x parameter matrix of filter counts (FilterBased only)
    COVERAGE.md               human-readable report
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "spartan" / "raw"
OUT_DIR = REPO_ROOT / "research" / "spartan" / "inventory"

# Expected sample step in hours for each subproduct (None = filter-based, irregular).
EXPECTED_STEP_H: dict[str, float | None] = {
    "ChemSpecPM10": None,
    "ChemSpecPM25": None,
    "ReconstrPM25": None,
    "DailyScaPM10": 24.0,
    "DailyScaPM25": 24.0,
    "HourlyScaPM10": 1.0,
    "HourlyScaPM25": 1.0,
    "DailyEstPM25": 24.0,
    "HourlyEstPM25": 1.0,
}


def _find_header_line(path: Path, max_scan: int = 5) -> int:
    with open(path, "r", errors="replace") as f:
        for i in range(max_scan):
            line = f.readline()
            if not line:
                return 0
            stripped = line.strip()
            if not stripped:
                continue
            if "," in stripped and not stripped.lstrip().startswith("#"):
                low = stripped.lower()
                if any(tok in low for tok in ("site_code", "year", "year_local")):
                    return i
    return 0


def _load(path: Path) -> pd.DataFrame:
    header = _find_header_line(path)
    return pd.read_csv(path, skiprows=header, low_memory=False)


def _build_dt(df: pd.DataFrame) -> pd.Series:
    cols = {c.lower(): c for c in df.columns}

    def col(*names: str) -> str | None:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    y = col("year_local", "start_year_local", "year")
    m = col("month_local", "start_month_local", "month")
    d = col("day_local", "start_day_local", "day")
    h = col("hour_local", "start_hour_local", "hour")
    if not (y and m and d):
        return pd.Series([], dtype="datetime64[ns]")

    parts = {
        "year": pd.to_numeric(df[y], errors="coerce"),
        "month": pd.to_numeric(df[m], errors="coerce"),
        "day": pd.to_numeric(df[d], errors="coerce"),
    }
    if h:
        parts["hour"] = pd.to_numeric(df[h], errors="coerce").fillna(0).astype(int)
    frame = pd.DataFrame(parts).dropna(subset=["year", "month", "day"])
    # Drop the bad-year ILNZ rows (year 21xx) so they don't distort stats
    frame = frame[(frame["year"] >= 2010) & (frame["year"] <= 2030)]
    return pd.to_datetime(frame, errors="coerce").dropna()


def analyze_file(product: str, subproduct: str, site: str, path: Path) -> dict:
    """Compute coverage and interval stats for a single CSV."""
    df = _load(path)
    dt = _build_dt(df)
    if dt.empty:
        return {
            "product": product, "subproduct": subproduct, "site": site,
            "rows": len(df), "notes": "no_datetime",
        }

    # For filter-based, "samples" are unique filter IDs (each filter has many parameter rows).
    # For Nephel/Estimate products, "samples" are unique timestamps.
    if product == "FilterBased":
        fid_col = next((c for c in df.columns if c.lower() == "filter_id"), None)
        if fid_col is not None:
            samples = df.dropna(subset=[fid_col]).drop_duplicates(subset=[fid_col])
            sample_times = _build_dt(samples).sort_values().reset_index(drop=True)
        else:
            sample_times = dt.drop_duplicates().sort_values().reset_index(drop=True)
        n_samples = len(sample_times)
    else:
        sample_times = dt.drop_duplicates().sort_values().reset_index(drop=True)
        n_samples = len(sample_times)

    date_min = sample_times.min()
    date_max = sample_times.max()
    span_days = max((date_max - date_min).total_seconds() / 86400.0, 0.0)

    # Inter-sample intervals (hours)
    if len(sample_times) >= 2:
        diffs_h = sample_times.diff().dt.total_seconds().dropna() / 3600.0
        mean_h = float(diffs_h.mean())
        median_h = float(diffs_h.median())
        p95_h = float(diffs_h.quantile(0.95))
        max_gap_h = float(diffs_h.max())
    else:
        mean_h = median_h = p95_h = max_gap_h = float("nan")

    # Completeness vs expected cadence
    expected_step_h = EXPECTED_STEP_H.get(subproduct)
    if expected_step_h and span_days > 0:
        expected_samples = (span_days * 24.0) / expected_step_h + 1
        completeness = n_samples / expected_samples
    elif product == "FilterBased" and span_days > 0:
        # SPARTAN nominal cadence: 1 filter every 9 days
        expected_samples = span_days / 9.0 + 1
        completeness = n_samples / expected_samples
    else:
        expected_samples = float("nan")
        completeness = float("nan")

    return {
        "product": product,
        "subproduct": subproduct,
        "site": site,
        "rows": len(df),
        "n_samples": n_samples,
        "date_min": date_min.strftime("%Y-%m-%d"),
        "date_max": date_max.strftime("%Y-%m-%d"),
        "span_days": round(span_days, 1),
        "expected_samples": round(expected_samples, 0) if not np.isnan(expected_samples) else "",
        "completeness_pct": round(completeness * 100, 1) if not np.isnan(completeness) else "",
        "mean_interval_h": round(mean_h, 2),
        "median_interval_h": round(median_h, 2),
        "p95_interval_h": round(p95_h, 2),
        "max_gap_days": round(max_gap_h / 24.0, 1),
        "notes": "",
    }


def collect_param_matrix() -> pd.DataFrame:
    """Build site x parameter counts (filter-based ChemSpec PM2.5/PM10 only).

    Counts are number of unique filters carrying a value for that parameter.
    """
    rows = []
    for sub in ("ChemSpecPM25", "ChemSpecPM10"):
        for path in sorted((RAW_DIR / "FilterBased" / sub).glob("FilterBased_*.csv")):
            site = path.stem.rsplit("_", 1)[-1]
            df = _load(path)
            pname = next((c for c in df.columns if c.lower() == "parameter_name"), None)
            fid = next((c for c in df.columns if c.lower() == "filter_id"), None)
            val = next((c for c in df.columns if c.lower() == "value"), None)
            if not (pname and fid):
                continue
            valid = df.dropna(subset=[fid, pname])
            if val is not None:
                # Count filters where value is non-null and not NaN
                valid = valid[pd.to_numeric(valid[val], errors="coerce").notna()]
            counts = valid.groupby(pname)[fid].nunique()
            for param, c in counts.items():
                rows.append({"subproduct": sub, "site": site, "parameter": param, "n_filters": int(c)})
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    wide = long.pivot_table(
        index=["subproduct", "parameter"], columns="site",
        values="n_filters", fill_value=0,
    ).sort_index()
    wide["n_sites"] = (wide > 0).sum(axis=1)
    wide["total_filters"] = wide.drop(columns="n_sites").sum(axis=1)
    return wide


def main() -> int:
    if not RAW_DIR.exists():
        print(f"No raw data at {RAW_DIR}; run spartan_pull_and_summarize.py first.")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning files for coverage stats...")
    results = []
    files = sorted(RAW_DIR.glob("*/*/*.csv"))
    for i, path in enumerate(files, 1):
        product = path.parents[1].name
        subproduct = path.parent.name
        site = path.stem.rsplit("_", 1)[-1]
        try:
            stats = analyze_file(product, subproduct, site, path)
        except Exception as e:  # noqa: BLE001
            stats = {"product": product, "subproduct": subproduct, "site": site, "notes": f"err: {e}"}
        results.append(stats)
        print(f"  [{i:>3}/{len(files)}] {product}/{subproduct}/{site}: "
              f"{stats.get('n_samples', '?')} samples, "
              f"completeness={stats.get('completeness_pct', '?')}%, "
              f"median Δ={stats.get('median_interval_h', '?')}h")

    cov = pd.DataFrame(results).sort_values(["product", "subproduct", "site"])
    cov.to_csv(OUT_DIR / "coverage.csv", index=False)
    print(f"\nWrote {OUT_DIR/'coverage.csv'}")

    print("\nBuilding site x parameter matrix...")
    pm = collect_param_matrix()
    if not pm.empty:
        pm.to_csv(OUT_DIR / "site_parameter_matrix.csv")
        print(f"Wrote {OUT_DIR/'site_parameter_matrix.csv'}")

    # ---- markdown report ----
    lines: list[str] = []
    add = lines.append

    add("# SPARTAN coverage analysis\n")
    add("Computed from cached files under `data/spartan/raw/`.\n")
    add("- *Samples*: unique filter IDs (FilterBased) or unique timestamps (NephelProcd / TimeResPM25).")
    add("- *Completeness*: actual samples / expected samples over the active span.")
    add("  - Hourly products → 1 sample/hr; Daily → 1 sample/day; FilterBased → SPARTAN nominal 1-in-9-days.")
    add("- *Median Δ*: median time between consecutive samples (hours).")
    add("- ILNZ rows with year 2166/2167 in the source files were filtered out as bad data.\n")

    add("## Network-level coverage by subproduct\n")
    g = (
        cov.dropna(subset=["completeness_pct"])
        .groupby(["product", "subproduct"])
        .agg(
            sites=("site", "nunique"),
            samples=("n_samples", "sum"),
            mean_completeness_pct=("completeness_pct", lambda s: round(pd.to_numeric(s, errors="coerce").mean(), 1)),
            median_completeness_pct=("completeness_pct", lambda s: round(pd.to_numeric(s, errors="coerce").median(), 1)),
            median_interval_h=("median_interval_h", lambda s: round(pd.to_numeric(s, errors="coerce").median(), 2)),
            median_span_days=("span_days", lambda s: round(pd.to_numeric(s, errors="coerce").median(), 0)),
        )
        .reset_index()
    )
    add(g.to_markdown(index=False))

    add("\n\n## Per-site coverage detail\n")
    show = cov[[
        "product", "subproduct", "site", "n_samples", "date_min", "date_max",
        "span_days", "completeness_pct", "median_interval_h", "max_gap_days",
    ]]
    for (prod, sub), g2 in show.groupby(["product", "subproduct"]):
        add(f"\n### {prod} / {sub}\n")
        add(g2.drop(columns=["product", "subproduct"])
              .sort_values("completeness_pct", ascending=False, na_position="last")
              .to_markdown(index=False))

    add("\n\n## Site x filter-based parameter matrix\n")
    if not pm.empty:
        add("(Number of unique filters carrying a valid (non-null) value for each parameter.)\n")
        # show summary: parameters present at >= 30 sites vs sparse
        common = pm[pm["n_sites"] >= 30].index.get_level_values("parameter").unique().tolist()
        sparse = pm[pm["n_sites"] < 30].index.get_level_values("parameter").unique().tolist()
        add(f"- Parameters at >=30 sites: **{len(common)}** "
            f"(e.g. {', '.join(common[:10])}{'...' if len(common) > 10 else ''})")
        add(f"- Parameters at <30 sites: **{len(sparse)}** "
            f"(e.g. {', '.join(sparse[:10])}{'...' if len(sparse) > 10 else ''})")
        add("\nFull matrix is in `site_parameter_matrix.csv`.\n")

    (OUT_DIR / "COVERAGE.md").write_text("\n".join(lines))
    print(f"Wrote {OUT_DIR/'COVERAGE.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
