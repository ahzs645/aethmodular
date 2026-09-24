"""Compare the local unified-filter Parquet with a hosted Zoer dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from zoer_env import zoer_url


DATA_ROOT = Path(os.environ.get("AETHMODULAR_DATA_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_PARQUET = DATA_ROOT / "output/tables/unified_filter_dataset.parquet"


def api_json(base_url: str, path: str, body: dict | None = None) -> dict:
    url = f"{base_url.rstrip('/')}/api{path}"
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    request = Request(
        url,
        data=payload,
        headers={"Accept": "application/json", **({"Content-Type": "application/json"} if payload else {})},
        method="POST" if payload else "GET",
    )
    try:
        with urlopen(request, timeout=20) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(f"Zoer returned HTTP {error.code} for {path}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach Zoer at {base_url}: {error.reason}") from error


def local_counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Local Parquet missing: {path}. Run unified_filter_parquet.py first.")
    site = pd.read_parquet(path, columns=["Site"])["Site"]
    return {"row_count": len(site), "etad_count": int(site.eq("ETAD").sum()), "site_count": int(site.nunique())}


def hosted_table(dataset: dict, source_name: str) -> str:
    if dataset.get("status") != "ready":
        raise RuntimeError(f"Dataset is {dataset.get('status')!r}; rebuild it in Zoer before testing.")
    summary = dataset.get("lastBuildSummary") or {}
    tables = summary.get("tables") or []
    matches = [table for table in tables if table.get("sourceName") == source_name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one ingested table from {source_name}; found {len(matches)}. Check the dataset build summary.")
    return matches[0]["name"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=None, help="Zoer server root; defaults to ZOER_URL from .env")
    parser.add_argument("--local-parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--dataset-id", help="Zoer dataset ID after the Parquet has been uploaded and rebuilt")
    parser.add_argument("--list", action="store_true", help="List hosted datasets without querying rows")
    args = parser.parse_args()
    args.base_url = args.base_url or zoer_url()

    baseline = local_counts(args.local_parquet.expanduser().resolve())
    print(json.dumps({"local": baseline, "parquet": str(args.local_parquet)}))

    if args.list:
        datasets = api_json(args.base_url, "/datasets").get("datasets", [])
        print(json.dumps({"hostedDatasets": [{"id": item["id"], "name": item["name"], "status": item["status"]} for item in datasets]}))
    if not args.dataset_id:
        return

    dataset = api_json(args.base_url, f"/datasets/{args.dataset_id}")["dataset"]
    table = hosted_table(dataset, args.local_parquet.name)
    quoted_table = '"' + table.replace('"', '""') + '"'
    query = f"SELECT COUNT(*) AS row_count, COUNT(*) FILTER (WHERE Site = 'ETAD') AS etad_count, COUNT(DISTINCT Site) AS site_count FROM {quoted_table}"
    rows = api_json(args.base_url, f"/datasets/{args.dataset_id}/query", {"query": query})["result"]["rows"]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one aggregate row from Zoer, received {len(rows)}")
    hosted = {key: int(rows[0][key]) for key in baseline}
    if hosted != baseline:
        raise RuntimeError(f"Hosted counts differ: local={baseline}, hosted={hosted}")
    print(json.dumps({"datasetId": args.dataset_id, "table": table, "hosted": hosted, "verified": True}))


if __name__ == "__main__":
    main()
