"""Pull every SPARTAN public CSV and summarize what's available.

Source: http://data.spartan-network.org/GroupedByProduct/
Mirror of the same content (by site): http://data.spartan-network.org/GroupedBySite/

Output layout:
    data/spartan/raw/<Product>/<SubProduct>/<file>.csv   downloaded files (gitignored)
    research/spartan/inventory/overview.csv              one row per file
    research/spartan/inventory/parameter_counts.csv      parameter x site, long-format products
    research/spartan/inventory/parameter_counts_long.csv same in long format
    research/spartan/inventory/site_coverage.csv         site x subproduct presence + row counts
    research/spartan/inventory/REPORT.md                 human-readable rollup

Usage:
    python scripts/pipelines/spartan_pull_and_summarize.py             # download + summarize
    python scripts/pipelines/spartan_pull_and_summarize.py --skip-download
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

BASE = "http://data.spartan-network.org/GroupedByProduct/"
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "spartan" / "raw"
SUMMARY_DIR = REPO_ROOT / "research" / "spartan" / "inventory"

HREF_RE = re.compile(r'href="([^"?/][^"]*)"')


def list_dir(url: str) -> list[str]:
    """Return the entries inside an Apache autoindex page, excluding dotfiles."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    entries = HREF_RE.findall(r.text)
    return [e for e in entries if not e.startswith(".") and not e.startswith("_")]


@dataclass
class FileSpec:
    product: str          # FilterBased | NephelProcd | TimeResPM25
    subproduct: str       # ChemSpecPM25, HourlyEstPM25, ...
    site: str             # 4-letter SPARTAN code
    filename: str
    url: str
    local_path: Path


def crawl() -> list[FileSpec]:
    """Walk Product/SubProduct/ and collect every .csv file."""
    specs: list[FileSpec] = []
    for product in list_dir(BASE):
        product = product.rstrip("/")
        prod_url = urljoin(BASE, product + "/")
        for sub in list_dir(prod_url):
            sub = sub.rstrip("/")
            sub_url = urljoin(prod_url, sub + "/")
            for fname in list_dir(sub_url):
                if not fname.lower().endswith(".csv"):
                    continue
                # SPARTAN filenames are <Product>_<SubProduct>_<SITE>.csv
                site = fname.rsplit("_", 1)[-1].split(".")[0]
                specs.append(
                    FileSpec(
                        product=product,
                        subproduct=sub,
                        site=site,
                        filename=fname,
                        url=urljoin(sub_url, fname),
                        local_path=RAW_DIR / product / sub / fname,
                    )
                )
    return specs


def download_one(spec: FileSpec) -> tuple[FileSpec, int, str]:
    """Stream one file to disk. Returns (spec, bytes_written, status)."""
    spec.local_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.local_path.exists() and spec.local_path.stat().st_size > 0:
        return spec, spec.local_path.stat().st_size, "cached"
    try:
        with requests.get(spec.url, stream=True, timeout=300) as r:
            r.raise_for_status()
            tmp = spec.local_path.with_suffix(spec.local_path.suffix + ".part")
            n = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
                        n += len(chunk)
            tmp.replace(spec.local_path)
        return spec, n, "downloaded"
    except Exception as e:  # noqa: BLE001
        return spec, 0, f"error: {e}"


def download_all(specs: list[FileSpec], workers: int = 8) -> None:
    print(f"Downloading {len(specs)} files with {workers} workers...")
    total_bytes = 0
    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for spec, n, status in ex.map(download_one, specs):
            done += 1
            total_bytes += n
            print(
                f"  [{done:>3}/{len(specs)}] {status:>10}  "
                f"{n/1024:7.1f} KB  {spec.product}/{spec.subproduct}/{spec.filename}"
            )
    print(f"Done. Wrote/verified {total_bytes/1024/1024:.1f} MB total.")


# ---------------------------------------------------------------------------
# Per-file inspection
# ---------------------------------------------------------------------------

@dataclass
class FileSummary:
    spec: FileSpec
    bytes: int
    n_rows: int = 0
    n_cols: int = 0
    columns: list[str] = field(default_factory=list)
    date_min: str = ""
    date_max: str = ""
    parameter_counts: dict[str, int] = field(default_factory=dict)
    notes: str = ""


def _find_header_line(path: Path, max_scan: int = 5) -> int:
    """Some SPARTAN CSVs start with 1-2 comment lines before the header row."""
    with open(path, "r", errors="replace") as f:
        for i in range(max_scan):
            line = f.readline()
            if not line:
                return 0
            stripped = line.strip()
            if not stripped:
                continue
            # header rows include a comma and start with a typical column token
            if "," in stripped and not stripped.lstrip().startswith("#"):
                # Heuristic: real CSV header rather than a free-text first line
                low = stripped.lower()
                if any(tok in low for tok in ("site_code", "year", "year_local")):
                    return i
    return 0


def _build_date(df: pd.DataFrame, product: str, subproduct: str) -> pd.Series:
    """Construct a datetime column from whatever year/month/day fields exist."""
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
    return pd.to_datetime(frame, errors="coerce").dropna()


def summarize_file(spec: FileSpec) -> FileSummary:
    path = spec.local_path
    s = FileSummary(spec=spec, bytes=path.stat().st_size if path.exists() else 0)
    if not path.exists():
        s.notes = "missing"
        return s

    header = _find_header_line(path)
    try:
        df = pd.read_csv(path, skiprows=header, low_memory=False)
    except Exception as e:  # noqa: BLE001
        s.notes = f"read_error: {e}"
        return s

    s.n_rows = len(df)
    s.n_cols = df.shape[1]
    s.columns = [str(c) for c in df.columns]

    dt = _build_date(df, spec.product, spec.subproduct)
    if len(dt):
        s.date_min = dt.min().strftime("%Y-%m-%d")
        s.date_max = dt.max().strftime("%Y-%m-%d")

    # Long-format products carry a Parameter_Name column we want to inventory
    pname_col = next((c for c in df.columns if c.lower() == "parameter_name"), None)
    if pname_col is not None:
        s.parameter_counts = (
            df[pname_col].astype(str).value_counts().to_dict()
        )

    return s


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_overview(summaries: list[FileSummary]) -> pd.DataFrame:
    rows = []
    for s in summaries:
        rows.append(
            {
                "product": s.spec.product,
                "subproduct": s.spec.subproduct,
                "site": s.spec.site,
                "filename": s.spec.filename,
                "size_kb": round(s.bytes / 1024, 1),
                "rows": s.n_rows,
                "cols": s.n_cols,
                "date_min": s.date_min,
                "date_max": s.date_max,
                "n_parameters": len(s.parameter_counts),
                "notes": s.notes,
            }
        )
    df = pd.DataFrame(rows).sort_values(["product", "subproduct", "site"])
    df.to_csv(SUMMARY_DIR / "overview.csv", index=False)
    return df


def write_parameter_counts(summaries: list[FileSummary]) -> pd.DataFrame:
    """For products with Parameter_Name, build a (product/subproduct/parameter) x site matrix."""
    records = []
    for s in summaries:
        for param, cnt in s.parameter_counts.items():
            records.append(
                {
                    "product": s.spec.product,
                    "subproduct": s.spec.subproduct,
                    "parameter": param,
                    "site": s.spec.site,
                    "rows": cnt,
                }
            )
    if not records:
        return pd.DataFrame()
    long = pd.DataFrame(records)
    wide = long.pivot_table(
        index=["product", "subproduct", "parameter"],
        columns="site",
        values="rows",
        fill_value=0,
    ).sort_index()
    wide["TOTAL_rows"] = wide.sum(axis=1)
    wide["n_sites"] = (wide.drop(columns="TOTAL_rows") > 0).sum(axis=1)
    wide.to_csv(SUMMARY_DIR / "parameter_counts.csv")
    long.to_csv(SUMMARY_DIR / "parameter_counts_long.csv", index=False)
    return wide


def write_site_coverage(summaries: list[FileSummary]) -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            {
                "site": s.spec.site,
                "subproduct": f"{s.spec.product}/{s.spec.subproduct}",
                "rows": s.n_rows,
            }
            for s in summaries
        ]
    )
    coverage = rows.pivot_table(
        index="site", columns="subproduct", values="rows", fill_value=0
    ).sort_index()
    coverage["TOTAL_rows"] = coverage.sum(axis=1)
    coverage["n_subproducts"] = (coverage.drop(columns="TOTAL_rows") > 0).sum(axis=1)
    coverage.to_csv(SUMMARY_DIR / "site_coverage.csv")
    return coverage


def write_report(
    overview: pd.DataFrame,
    coverage: pd.DataFrame,
    params: pd.DataFrame,
) -> None:
    lines: list[str] = []
    add = lines.append

    add("# SPARTAN public data inventory\n")
    add(f"- Files discovered: **{len(overview)}**")
    add(f"- Sites: **{overview['site'].nunique()}**")
    add(f"- Subproducts: **{overview['subproduct'].nunique()}**")
    add(f"- Total rows across all files: **{int(overview['rows'].sum()):,}**")
    add(f"- Total bytes on disk: **{overview['size_kb'].sum()/1024:.1f} MB**\n")

    add("## Files per subproduct (with row totals and date range)\n")
    g = (
        overview.groupby(["product", "subproduct"])
        .agg(
            files=("site", "count"),
            sites=("site", "nunique"),
            rows=("rows", "sum"),
            size_mb=("size_kb", lambda s: round(s.sum() / 1024, 1)),
            date_min=("date_min", lambda s: min([x for x in s if x] or [""])),
            date_max=("date_max", lambda s: max([x for x in s if x] or [""])),
        )
        .reset_index()
    )
    add(g.to_markdown(index=False))
    add("")

    add("\n## Sites x subproducts (rows; blank = no file)\n")
    add(coverage.drop(columns=["TOTAL_rows", "n_subproducts"]).to_markdown())
    add("")
    add("\n### Site totals\n")
    add(
        coverage[["TOTAL_rows", "n_subproducts"]]
        .sort_values("TOTAL_rows", ascending=False)
        .to_markdown()
    )
    add("")

    if not params.empty:
        add("\n## Filter-based & time-resolved: rows per parameter, summed across sites\n")
        top = (
            params[["TOTAL_rows", "n_sites"]]
            .reset_index()
            .sort_values(["product", "subproduct", "TOTAL_rows"], ascending=[True, True, False])
        )
        add(top.to_markdown(index=False))
        add("")

    (SUMMARY_DIR / "REPORT.md").write_text("\n".join(lines))


def print_console_summary(overview: pd.DataFrame, coverage: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("SPARTAN INVENTORY")
    print("=" * 78)
    print(f"files      : {len(overview)}")
    print(f"sites      : {overview['site'].nunique()}")
    print(f"rows total : {int(overview['rows'].sum()):,}")
    print(f"bytes total: {overview['size_kb'].sum()/1024:.1f} MB")

    print("\nBy subproduct:")
    g = overview.groupby(["product", "subproduct"]).agg(
        files=("site", "count"),
        rows=("rows", "sum"),
        date_min=("date_min", lambda s: min([x for x in s if x] or [""])),
        date_max=("date_max", lambda s: max([x for x in s if x] or [""])),
    )
    print(g.to_string())

    print("\nTop 10 sites by total rows:")
    print(coverage["TOTAL_rows"].sort_values(ascending=False).head(10).to_string())
    print("=" * 78)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip download step (use already-cached files).")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    print("Crawling SPARTAN index...")
    specs = crawl()
    print(f"  found {len(specs)} CSV files across "
          f"{len({s.product for s in specs})} products, "
          f"{len({s.subproduct for s in specs})} subproducts, "
          f"{len({s.site for s in specs})} sites.")

    if not args.skip_download:
        download_all(specs, workers=args.workers)

    print("\nSummarizing each file...")
    summaries: list[FileSummary] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for s in ex.map(summarize_file, specs):
            summaries.append(s)
            tag = s.notes or f"{s.n_rows} rows"
            print(f"  {s.spec.product}/{s.spec.subproduct}/{s.spec.site}: {tag}")

    print("\nWriting summaries...")
    overview = write_overview(summaries)
    params = write_parameter_counts(summaries)
    coverage = write_site_coverage(summaries)
    write_report(overview, coverage, params)
    print(f"  wrote {SUMMARY_DIR}/overview.csv")
    print(f"  wrote {SUMMARY_DIR}/parameter_counts.csv")
    print(f"  wrote {SUMMARY_DIR}/site_coverage.csv")
    print(f"  wrote {SUMMARY_DIR}/REPORT.md")

    print_console_summary(overview, coverage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
