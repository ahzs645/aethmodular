"""Pull every SPARTAN public CSV and summarize what's available.

Source: https://spartan-cloud.s3.us-west-2.amazonaws.com/GroupByProduct/
Official directory: https://www.spartan-network.org/data

Output layout:
    data/spartan/raw/<Product>/<SubProduct>/<file>.csv   downloaded files (gitignored)
    research/spartan/inventory/overview.csv              one row per file
    research/spartan/inventory/parameter_counts.csv      parameter x site, long-format products
    research/spartan/inventory/parameter_counts_long.csv same in long format
    research/spartan/inventory/site_coverage.csv         site x subproduct presence + row counts
    research/spartan/inventory/REPORT.md                 human-readable rollup

Usage:
    uv run aeth spartan pull                  # synchronize + summarize
    uv run aeth spartan pull --check-updates  # read-only remote comparison
    uv run aeth spartan pull --skip-download  # offline re-summarize
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urljoin

import pandas as pd
import requests

# scripts/ is not an installed package and the CLI runs this file by path, so
# put scripts/ on sys.path to make `common` importable. See scripts/common/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.spartan_io import (  # noqa: E402
    OUT_DIR as SUMMARY_DIR,
    RAW_DIR,
    build_datetime,
    find_col,
    find_header_line,
    site_from_path,
)

BUCKET = "https://spartan-cloud.s3.us-west-2.amazonaws.com/"
PREFIX = "GroupByProduct/"
BASE = urljoin(BUCKET, PREFIX)
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


@dataclass
class FileSpec:
    product: str          # FilterBased | NephelProcd | TimeResPM25
    subproduct: str       # ChemSpecPM25, HourlyEstPM25, ...
    site: str             # 4-letter SPARTAN code
    filename: str
    url: str
    local_path: Path
    remote_etag: str = ""
    remote_last_modified: str = ""
    remote_size_bytes: int | None = None


def crawl() -> list[FileSpec]:
    """List the public S3 bucket with pagination and collect product CSVs."""
    specs: list[FileSpec] = []
    continuation: str | None = None
    while True:
        params = {"list-type": "2", "prefix": PREFIX, "max-keys": "1000"}
        if continuation:
            params["continuation-token"] = continuation
        r = requests.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for item in root.findall("s3:Contents", S3_NS):
            key = item.findtext("s3:Key", default="", namespaces=S3_NS)
            parts = key.removeprefix(PREFIX).split("/")
            if not key.startswith(PREFIX) or len(parts) != 3 or not parts[-1].lower().endswith(".csv"):
                continue
            product, sub, fname = parts
            site = fname.rsplit("_", 1)[-1].split(".")[0]
            size = item.findtext("s3:Size", namespaces=S3_NS)
            specs.append(
                FileSpec(
                    product=product,
                    subproduct=sub,
                    site=site,
                    filename=fname,
                    url=urljoin(BUCKET, quote(key, safe="/")),
                    local_path=RAW_DIR / product / sub / fname,
                    remote_etag=(item.findtext("s3:ETag", default="", namespaces=S3_NS).strip('"')),
                    remote_last_modified=item.findtext("s3:LastModified", default="", namespaces=S3_NS),
                    remote_size_bytes=int(size) if size is not None else None,
                )
            )
        if root.findtext("s3:IsTruncated", default="false", namespaces=S3_NS) != "true":
            break
        continuation = root.findtext("s3:NextContinuationToken", namespaces=S3_NS)
        if not continuation:
            raise ValueError("S3 listing is truncated but has no continuation token")
    return sorted(specs, key=spec_key)


def scan_local() -> list[FileSpec]:
    """Build FileSpecs from already-downloaded files, without touching the network.

    Mirrors crawl()'s directory grammar:
        RAW_DIR/<Product>/<SubProduct>/<Product>_<SubProduct>_<SITE>.csv

    Used for --skip-download so the offline path stays offline. `url` is filled
    in for provenance but is never fetched on this path.
    """
    specs: list[FileSpec] = []
    previous = inventory_metadata()
    if not RAW_DIR.is_dir():
        return specs
    for product_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        for sub_dir in sorted(p for p in product_dir.iterdir() if p.is_dir()):
            for path in sorted(sub_dir.glob("*.csv")):
                prior = previous.get((product_dir.name, sub_dir.name, path.name), {})
                specs.append(
                    FileSpec(
                        product=product_dir.name,
                        subproduct=sub_dir.name,
                        site=site_from_path(path),
                        filename=path.name,
                        url=urljoin(BASE, f"{product_dir.name}/{sub_dir.name}/{path.name}"),
                        local_path=path,
                        remote_etag=saved_etag(prior),
                        remote_last_modified=(str(prior["remote_last_modified"])
                                              if pd.notna(prior.get("remote_last_modified")) else ""),
                        remote_size_bytes=(int(prior["remote_size_bytes"])
                                           if pd.notna(prior.get("remote_size_bytes")) else None),
                    )
                )
    return specs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(spec: FileSpec, *, check_only: bool = False) -> tuple[FileSpec, int, str]:
    """Compare remote bytes with the cache; atomically install new/changed files.

    Hashing the response is deliberate: the public index does not give us a
    version manifest, and size/Last-Modified alone cannot prove equality.
    """
    local_exists = spec.local_path.is_file() and spec.local_path.stat().st_size > 0
    if check_only and not local_exists:
        # The S3 listing proves the key exists, but provides no local content
        # baseline; downloading it would not make a revision check possible.
        return spec, 0, "new"
    tmp = spec.local_path.with_suffix(spec.local_path.suffix + ".part")
    if not check_only:
        spec.local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(spec.url, stream=True, timeout=300) as r:
            r.raise_for_status()
            n = 0
            remote_hash = hashlib.sha256()
            if check_only:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        remote_hash.update(chunk)
                        n += len(chunk)
            else:
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if not chunk:
                            continue
                        f.write(chunk)
                        remote_hash.update(chunk)
                        n += len(chunk)
        if local_exists and remote_hash.hexdigest() == sha256_file(spec.local_path):
            if not check_only:
                tmp.unlink(missing_ok=True)
            return spec, n, "unchanged"
        if check_only:
            return spec, n, "changed" if local_exists else "new"
        tmp.replace(spec.local_path)
        return spec, n, "updated" if local_exists else "downloaded"
    except Exception as e:  # noqa: BLE001
        return spec, 0, f"error: {e}"
    finally:
        if not check_only:
            tmp.unlink(missing_ok=True)


def download_all(specs: list[FileSpec], workers: int = 8,
                 *, check_only: bool = False) -> list[tuple[FileSpec, int, str]]:
    print(f"{'Checking' if check_only else 'Synchronizing'} {len(specs)} files with {workers} workers...")
    total_bytes = 0
    done = 0
    results: list[tuple[FileSpec, int, str]] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for spec, n, status in ex.map(lambda s: download_one(s, check_only=check_only), specs):
            results.append((spec, n, status))
            done += 1
            total_bytes += n
            if not check_only or status == "changed" or status.startswith("error:"):
                print(
                    f"  [{done:>3}/{len(specs)}] {status:>10}  "
                    f"{n/1024:7.1f} KB  {spec.product}/{spec.subproduct}/{spec.filename}"
                )
    print(f"Done. Streamed {total_bytes/1024/1024:.1f} MB from SPARTAN.")
    return results


def inventory_metadata() -> dict[tuple[str, str, str], dict]:
    """Saved object metadata; supports checks when local CSV bytes are absent."""
    path = SUMMARY_DIR / "overview.csv"
    if not path.is_file():
        return {}
    old = pd.read_csv(path)
    return {
        (str(r["product"]), str(r["subproduct"]), str(r["filename"])): r
        for r in old.to_dict("records")
    }


def saved_etag(row: dict | None) -> str:
    value = (row or {}).get("remote_etag")
    return str(value) if pd.notna(value) else ""


def spec_key(spec: FileSpec) -> tuple[str, str, str]:
    return spec.product, spec.subproduct, spec.filename


def print_update_check(specs: list[FileSpec],
                       results: list[tuple[FileSpec, int, str]]) -> int:
    """Print a read-only comparison, including what cannot be verified."""
    previous = inventory_metadata()
    remote = {spec_key(s) for s in specs}
    additions = remote - previous.keys() if previous else set()
    removals = previous.keys() - remote
    changed = [s for s, _, status in results if status == "changed"]
    errors = [(s, status) for s, _, status in results if status.startswith("error:")]
    metadata_changed = [s for s, _, status in results if status == "new"
                        and (old_etag := saved_etag(previous.get(spec_key(s))))
                        and s.remote_etag != old_etag]
    metadata_unchanged = [s for s, _, status in results if status == "new"
                          and (old_etag := saved_etag(previous.get(spec_key(s))))
                          and s.remote_etag == old_etag]
    unverified = [s for s, _, status in results if status == "new" and spec_key(s) in previous
                  and not saved_etag(previous[spec_key(s)])]
    print("\nSPARTAN update check (no local files or reports changed)")
    print(f"New filenames since saved inventory: {len(additions)}")
    print(f"Filenames absent from current index: {len(removals)}")
    print(f"Revised files proven by content hash: {len(changed)}")
    print(f"Objects with changed S3 ETags and no local bytes: {len(metadata_changed)}")
    print(f"Objects with unchanged S3 ETags and no local bytes: {len(metadata_unchanged)}")
    print(f"Existing filenames with no local bytes or saved ETag: {len(unverified)}")
    for title, keys in (("New", additions), ("Absent", removals)):
        for product, subproduct, filename in sorted(keys):
            print(f"  {title}: {product}/{subproduct}/{filename}")
    for spec in changed:
        print(f"  Revised: {spec.product}/{spec.subproduct}/{spec.filename}")
    for spec in metadata_changed:
        print(f"  S3 object changed: {spec.product}/{spec.subproduct}/{spec.filename}")
    for spec, status in errors:
        print(f"  Failed: {spec.url}: {status}", file=sys.stderr)
    if not previous:
        print("No saved overview.csv baseline; filename changes cannot be classified.")
    if unverified:
        print("Revision status is unknown for files without a local raw copy or saved ETag; run pull to establish a baseline.")
    if metadata_unchanged:
        print("An unchanged ETag is a bucket metadata comparison, not a local content-hash check.")
    return 1 if errors else 0


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


def summarize_file(spec: FileSpec) -> FileSummary:
    path = spec.local_path
    s = FileSummary(spec=spec, bytes=path.stat().st_size if path.exists() else 0)
    if not path.exists():
        s.notes = "missing"
        return s

    header = find_header_line(path)
    try:
        df = pd.read_csv(path, skiprows=header, low_memory=False)
    except Exception as e:  # noqa: BLE001
        s.notes = f"read_error: {e}"
        return s

    s.n_rows = len(df)
    s.n_cols = df.shape[1]
    s.columns = [str(c) for c in df.columns]

    dt = build_datetime(df)
    if len(dt):
        s.date_min = dt.min().strftime("%Y-%m-%d")
        s.date_max = dt.max().strftime("%Y-%m-%d")

    # Long-format products carry a Parameter_Name column we want to inventory
    pname_col = find_col(df, "parameter_name")
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
                "remote_etag": s.spec.remote_etag,
                "remote_last_modified": s.spec.remote_last_modified,
                "remote_size_bytes": s.spec.remote_size_bytes,
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
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--skip-download", action="store_true",
                    help="Skip download step (use already-cached files).")
    mode.add_argument("--check-updates", action="store_true",
                      help="Read-only index and content comparison; do not change local data or reports.")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be at least 1")

    if not args.check_updates:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    # crawl() hits the network, so it must stay behind the --skip-download
    # guard: previously it ran unconditionally and the documented offline mode
    # still required connectivity.
    if args.skip_download:
        print(f"Scanning local files under {RAW_DIR}...")
        specs = scan_local()
        if not specs:
            print(f"ERROR: --skip-download given but no CSVs found under {RAW_DIR}.",
                  file=sys.stderr)
            print("       Run without --skip-download at least once first.", file=sys.stderr)
            return 1
    else:
        print("Crawling SPARTAN index...")
        try:
            specs = crawl()
        except (requests.RequestException, ET.ParseError, ValueError) as e:
            print(f"ERROR: SPARTAN index unavailable: {e}", file=sys.stderr)
            return 1
        if not specs:
            print("ERROR: SPARTAN index returned no CSV files; preserving existing inventory.", file=sys.stderr)
            return 1

    print(f"  found {len(specs)} CSV files across "
          f"{len({s.product for s in specs})} products, "
          f"{len({s.subproduct for s in specs})} subproducts, "
          f"{len({s.site for s in specs})} sites.")

    if not args.skip_download:
        results = download_all(specs, workers=args.workers, check_only=args.check_updates)
        if args.check_updates:
            return print_update_check(specs, results)
        failures = [(s, status) for s, _, status in results if status.startswith("error:")]
        if failures:
            for spec, status in failures:
                print(f"ERROR: {spec.url}: {status}", file=sys.stderr)
            print("Inventory was not rewritten because downloads failed.", file=sys.stderr)
            return 1

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
