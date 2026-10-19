"""Compare a native Davis bundle with the local public PM2.5 chemistry mirror.

Read-only inputs. Keep raw HIPS availability separate from public derived BC.
Run with uv run; see docs/spartan-network-data-pulls.md for acquisition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from common.spartan_io import RAW_DIR, normalize_fid, read_spartan_csv  # noqa: E402
from aethmodular_cli.env import display_path  # noqa: E402


def audit(bundle: Path, public_dir: Path, out: Path) -> dict:
    paths = sorted(public_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No public chemistry CSVs in {display_path(public_dir)}")
    inventory_path = bundle / "filter_inventory.csv"
    raw_path = bundle / "spartan_hips_raw_all.csv"
    public = pd.concat([read_spartan_csv(p) for p in paths], ignore_index=True)
    public["base_id"] = normalize_fid(public["Filter_ID"])
    # Retain the whole public chemistry population, including filters without carbon.
    result = public[["Site_Code", "base_id"]].drop_duplicates().rename(
        columns={"Site_Code": "site", "base_id": "filter_id"}
    ).set_index("filter_id")
    if not result.index.is_unique:
        raise ValueError("Public physical filter key maps to more than one site")
    valid = pd.to_numeric(public["Value"], errors="coerce").notna()
    methods = pd.to_numeric(public["Method_Code"], errors="coerce")
    for name, mask in {
        "public_ec": public.Parameter_Name.eq("EC PM2.5"),
        "public_oc": public.Parameter_Name.eq("OC PM2.5"),
        "public_bc_hips": public.Parameter_Name.eq("BC PM2.5") & methods.isin([219, 221]),
        "public_bc_ssr_estimate": public.Parameter_Name.eq("BC PM2.5") & methods.eq(220),
    }.items():
        result[name] = result.index.isin(public.loc[mask & valid, "base_id"])

    inventory = pd.read_csv(inventory_path)
    inventory = inventory[inventory.ExternalFilterType.eq("PM2.5")].copy()
    inventory["filter_id"] = normalize_fid(inventory.ExternalFilterId)
    # A non-unique physical metadata key needs investigation, not implicit summation.
    if inventory.filter_id.duplicated().any():
        raise ValueError("PM2.5 database metadata has repeated base filter IDs")
    inventory = inventory.set_index("filter_id")
    result["db_metadata"] = result.index.isin(inventory.index)
    for source, dest in [("FtirAnalyses", "db_ftir_analyses"),
                         ("HipsAnalyses", "db_hips_analyses")]:
        result[dest] = result.index.map(inventory[source]).fillna(0).astype(int)

    raw = pd.read_csv(raw_path)
    raw["filter_id"] = normalize_fid(raw.ExternalFilterId)
    samples = raw[pd.to_numeric(raw.ResultTypeId, errors="coerce").eq(0)]
    counts = samples.groupby("filter_id").SampleAnalysisId.nunique()
    result["db_hips_sample_analyses"] = result.index.map(counts).fillna(0).astype(int)
    result["missing_db_ftir"] = result.db_ftir_analyses.eq(0)
    result["missing_db_hips_sample"] = result.db_hips_sample_analyses.eq(0)
    result = result.reset_index().sort_values(["site", "filter_id"])
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "public_filters": len(result),
        "public_ec_filters": int(result.public_ec.sum()),
        "public_ec_with_db_ftir": int((result.public_ec & ~result.missing_db_ftir).sum()),
        "public_ec_with_db_hips_sample": int((result.public_ec & ~result.missing_db_hips_sample).sum()),
        "public_ec_with_direct_public_hips_bc": int((result.public_ec & result.public_bc_hips).sum()),
        "public_ec_with_public_ssr_estimate": int((result.public_ec & result.public_bc_ssr_estimate).sum()),
        "note": "Raw sample availability is not final QC-passed Fabs availability; FTIR analysis presence does not verify paired scan completeness.",
        "sources": [{"path": display_path(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in [*paths, inventory_path, raw_path]],
    }
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "public_filter_parity.csv", index=False)
    result[result.missing_db_ftir | result.missing_db_hips_sample].to_csv(
        out / "public_filter_gaps.csv", index=False)
    result.groupby("site").agg(
        public_filters=("filter_id", "size"), public_ec=("public_ec", "sum"),
        missing_db_ftir=("missing_db_ftir", "sum"),
        missing_db_hips_sample=("missing_db_hips_sample", "sum"),
    ).to_csv(out / "site_parity.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {k: v for k, v in summary.items() if k != "sources"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Unpacked Davis HIPS bundle")
    parser.add_argument("--public-dir", type=Path, default=RAW_DIR / "FilterBased/ChemSpecPM25")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.bundle, args.public_dir, args.out), indent=2))
