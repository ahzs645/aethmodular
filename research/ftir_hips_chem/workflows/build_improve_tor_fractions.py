"""Extract public IMPROVE thermal fractions for the local FTIR pool.

Input ZIPs are the IMPAER yearly files from
https://views.cira.colostate.edu/fed/DataFiles/ (IMPROVE Aerosol raw data).
Run from the repository root with ``uv run python`` after downloading 2020-2023
to output/tables/improve_tor_fractions/raw/.

The public data can contain two samples for a site/date. Match the local TOR
mirror's EC, OC, EC1, OPTR, and OPTT to select the same public sample. A
site/date without a local mirror is used only when the public sample is unique.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
OUT = HERE / "output/tables/improve_tor_fractions"
POOL = HERE / "output/tables/pls_transfer/improve_full_pool_addis_similarity.csv"
sys.path.insert(0, str(REPO / "research/ftir_ec_phase3/scripts"))
from phase3_common import PATHS  # noqa: E402

PARAMETERS = {
    "ECf": "EC_TOR",
    "OCf": "OC_TOR",
    "EC1f": "EC1",
    "EC2f": "EC2",
    "EC3f": "EC3",
    "OC1f": "OC1",
    "OC2f": "OC2",
    "OC3f": "OC3",
    "OC4f": "OC4",
    "OPf": "OPTR",
    "OPTf": "OPTT",
}
MIRROR_PAIRS = {
    "EC": "EC_TOR",
    "OC": "OC_TOR",
    "EC1": "EC1",
    "OPTR": "OPTR",
    "OPTT": "OPTT",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(POOL, usecols=["AnalysisId", "FilterId", "Site", "SampleDate"])
    pool["SampleDate"] = pd.to_datetime(pool["SampleDate"], format="mixed").dt.strftime("%Y-%m-%d")
    keys = pool[["Site", "SampleDate"]].drop_duplicates()
    pool_count = pool.groupby(["Site", "SampleDate"]).size().rename("pool_spectra_n")

    frames = []
    sources = []
    for year in range(2020, 2024):
        path = OUT / "raw" / f"IMPAER_{year}.txt.zip"
        if not path.exists():
            raise FileNotFoundError(path)
        sources.append({
            "url": f"https://vibe.cira.colostate.edu/data/export/IMPAER/IMPAER_{year}.txt.zip",
            "file": str(path.relative_to(REPO)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        frame = pd.read_csv(
            path, sep="|", compression="zip",
            usecols=["SiteCode", "POC", "FactDate", "ParamCode", "MethodID",
                     "Units", "FactValue", "Status", "ProviderStatus"],
        )
        frame = frame[frame["ParamCode"].isin(PARAMETERS)].copy()
        frame = frame.merge(keys, left_on=["SiteCode", "FactDate"],
                            right_on=["Site", "SampleDate"], how="inner")
        frames.append(frame)
    fed = pd.concat(frames, ignore_index=True)
    if fed.duplicated(["Site", "SampleDate", "POC", "ParamCode"]).any():
        raise ValueError("Duplicate public site/date/POC/parameter rows")
    if set(fed["Units"].dropna()) != {"ug/m^3"}:
        raise ValueError("Unexpected IMPROVE fraction units")
    fed["FactValue"] = fed["FactValue"].replace(-999, np.nan)
    fed["field"] = fed["ParamCode"].map(PARAMETERS)
    index = ["Site", "SampleDate", "POC"]
    values = fed.pivot(index=index, columns="field", values="FactValue")
    statuses = fed.pivot(index=index, columns="field", values="Status").add_prefix("status_")
    provider_statuses = fed.pivot(
        index=index, columns="field", values="ProviderStatus"
    ).add_prefix("provider_status_")
    public = values.join(statuses).join(provider_statuses).reset_index()

    mirror = pd.read_csv(
        PATHS.ftir_dir / "local_db/tables/results_tor.csv",
        usecols=["Site", "SampleDate", "Parameter", "Value"],
    )
    mirror = mirror[mirror["Parameter"].isin(MIRROR_PAIRS)].copy()
    mirror["SampleDate"] = pd.to_datetime(
        mirror["SampleDate"], format="mixed", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    mirror = mirror.merge(keys, on=["Site", "SampleDate"], how="inner")
    if mirror.duplicated(["Site", "SampleDate", "Parameter"]).any():
        raise ValueError("Duplicate local TOR site/date/parameter rows")
    mirror = mirror.pivot(index=["Site", "SampleDate"],
                          columns="Parameter", values="Value").reset_index()
    mirror = mirror.rename(columns={field: f"mirror_{field}" for field in MIRROR_PAIRS})
    mirror["mirror_present"] = True

    candidates = keys.merge(public, on=["Site", "SampleDate"], how="left")
    candidates = candidates.merge(mirror, on=["Site", "SampleDate"], how="left")
    match_columns = []
    for local, source in MIRROR_PAIRS.items():
        name = f"match_{local}"
        candidates[name] = np.isclose(
            candidates[f"mirror_{local}"] / 1000, candidates[source], rtol=0, atol=0.000011,
        )
        match_columns.append(name)
    candidates["mirror_match_n"] = candidates[match_columns].sum(axis=1)
    public_n = candidates.groupby(["Site", "SampleDate"])["POC"].transform("count")
    candidates["public_samples_n"] = public_n
    exact = candidates["mirror_present"].eq(True) & candidates["mirror_match_n"].eq(5)
    unique_without_mirror = candidates["mirror_present"].isna() & public_n.eq(1)
    candidates["source_match"] = np.select(
        [exact, unique_without_mirror, candidates["POC"].isna(),
         candidates["mirror_present"].eq(True), public_n.gt(1)],
        ["exact_mirror", "single_public_no_mirror", "no_public", "mirror_unmatched",
         "ambiguous_public"],
        default="unmatched",
    )
    if candidates[exact].duplicated(["Site", "SampleDate"]).any():
        raise ValueError("More than one exact public match for a local TOR result")
    selected = candidates[exact | unique_without_mirror].copy()
    unresolved = candidates[~candidates[["Site", "SampleDate"]].apply(tuple, axis=1).isin(
        selected[["Site", "SampleDate"]].apply(tuple, axis=1)
    )].drop_duplicates(["Site", "SampleDate"]).copy()
    unresolved[list(PARAMETERS.values())] = np.nan
    ambiguous = unresolved["public_samples_n"].gt(1)
    unresolved.loc[ambiguous, [
        "POC", *[f"status_{v}" for v in PARAMETERS.values()],
        *[f"provider_status_{v}" for v in PARAMETERS.values()],
    ]] = np.nan
    result = pd.concat([selected, unresolved], ignore_index=True)
    result = result.merge(pool_count.reset_index(), on=["Site", "SampleDate"], validate="one_to_one")
    result["EC_closure_error_ugm3"] = (
        result["EC_TOR"] - (result["EC1"] + result["EC2"] + result["EC3"] - result["OPTR"])
    )
    result["OC_closure_error_ugm3"] = (
        result["OC_TOR"] - (result["OC1"] + result["OC2"] + result["OC3"]
                              + result["OC4"] + result["OPTR"])
    )
    field_cols = list(PARAMETERS.values())
    result["fractions_complete"] = result[field_cols].notna().all(axis=1)
    result = result[["Site", "SampleDate", "POC", "source_match", "mirror_match_n",
                     "public_samples_n", "pool_spectra_n", "fractions_complete",
                     *field_cols, *[f"status_{v}" for v in field_cols],
                     *[f"provider_status_{v}" for v in field_cols],
                     "EC_closure_error_ugm3", "OC_closure_error_ugm3"]]
    result = result.rename(columns={v: f"{v}_ugm3" for v in field_cols})
    result = result.sort_values(["Site", "SampleDate"])
    output = OUT / "improve_tor_fractions_pool_site_dates.csv"
    result.to_csv(output, index=False)

    ledger = {
        "pool_spectra_rows": len(pool),
        "pool_site_dates": len(keys),
        "source_match": result["source_match"].value_counts().to_dict(),
        "complete_fraction_site_dates": int(result["fractions_complete"].sum()),
        "complete_exact_mirror_site_dates": int((
            result["fractions_complete"] & result["source_match"].eq("exact_mirror")
        ).sum()),
        "max_abs_ec_closure_error_ugm3": float(result["EC_closure_error_ugm3"].abs().max()),
        "max_abs_oc_closure_error_ugm3": float(result["OC_closure_error_ugm3"].abs().max()),
        "source_files": sources,
        "notes": [
            "Public concentrations are ug/m^3; local mirror values were ng/m^3 and divided by 1000 for matching.",
            "A site/date may have multiple FTIR spectra. This table has one TOR record per site/date and is not a physical-filter ID match.",
            "-999 public sentinel values were converted to missing. Status fields retain source QA codes.",
        ],
    }
    (OUT / "provenance.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps({k: v for k, v in ledger.items() if k != "source_files"}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
