"""Portable, provenance-separated FTIR / HIPS screening of the committed gallery export.

This module measures optical disagreement, never independent thermal-EC error.
It reads one row per physical filter, retains every original field and flag,
and does not infer a missing HIPS measurement from the SSR estimate. The public
EC and ChemSpec EC fields are duplicate FTIR products, not two measurements.

The export already carries the central exclusion registry's decisions. We use
those frozen flags and ``outliers.get_clean_data`` rather than applying the
registry's placeholder date-only aethalometer exclusions to filter-only data.
Additional exclusions made by a caller must update ``is_excluded`` and retain
an explicit reason; no observation is dropped or winsorized by this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import MAC_VALUE
from outliers import get_clean_data
from plotting.utils import deming


PUBLIC_CHEMISTRY_PATH = Path("gallery/app/public/data/spartan_reference.json")
DIRECT_HIPS = "direct_hips_fabs"
PUBLIC_EXTENSION = "public_hips_extension"
COMBINED_HIPS = "combined_hips_sources"
DEMING_LAMBDAS = (0.25, 1.0, 4.0)
# Source-product convention, distinct from the analysis MAC in config.py.
PUBLIC_HIPS_MAC_M2_G = 10.0
FIELD_ALIASES = {
    "EC (FTIR)": "ec_ftir_ugm3",
    "HIPS Fabs": "hips_fabs_Mm1",
    "HIPS Fabs uncertainty": "hips_fabs_uncertainty_Mm1",
    "HIPS MDL": "hips_mdl_Mm1",
    "BC (public HIPS)": "bc_public_hips_ugm3",
    "Sampled volume": "volume_m3",
    "PM2.5 mass": "pm25_ugm3",
}
VOLUME_CAVEAT = (
    "Reported volume mixes HIPS measurement, functional-group delivery and "
    "SS5i duration x nominal flow; the export has no per-row provenance. "
    "Coverage is not coverage of independently measured flow."
)


def _assert_physical_rows(frame: pd.DataFrame) -> None:
    for key in ("id", "code"):
        if key not in frame or frame[key].isna().any() or frame[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"physical-filter ledger requires nonempty {key}")
    if frame["id"].duplicated().any():
        raise ValueError("duplicate physical-filter IDs; adjudicate identities before screening")


def load_public_chemistry(root: str | Path) -> pd.DataFrame:
    """Read the committed PM2.5 ledger, keeping original names plus explicit unit aliases.

    ``root`` is the repository root. Missing numerical measurements stay NaN;
    invalid numeric strings fail instead of silently disappearing. The returned
    ``date`` is the source string and ``parsed_date`` is a pandas timestamp.
    Source hash, units, source-field descriptions and caveats are in ``attrs``.
    """
    path = Path(root) / PUBLIC_CHEMISTRY_PATH
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("rows"), list):
        raise ValueError("unsupported public chemistry schema")
    frame = pd.DataFrame(payload["rows"])
    _assert_physical_rows(frame)
    if payload.get("n", len(frame)) != len(frame):
        raise ValueError("public chemistry row count disagrees with manifest")
    if "date" not in frame or "excluded" not in frame or "exclusion_reason" not in frame:
        raise ValueError("public chemistry requires dates and explicit exclusion flags/reasons")
    if not frame["excluded"].map(lambda v: isinstance(v, bool)).all():
        raise ValueError("excluded must be explicit booleans")
    frame["parsed_date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["published_excluded"] = frame["excluded"].astype(bool)
    frame["is_excluded"] = frame["published_excluded"]
    frame["exclusion_reason"] = frame["exclusion_reason"].fillna("").astype(str)
    if (frame["is_excluded"] & frame["exclusion_reason"].str.strip().eq("")).any():
        raise ValueError("excluded physical filters require a reason")
    for original, alias in FIELD_ALIASES.items():
        values = frame[original] if original in frame else pd.Series(np.nan, index=frame.index)
        frame[alias] = pd.to_numeric(values, errors="raise").astype(float)
    frame["hips_bc_ugm3"] = frame["hips_fabs_Mm1"] / MAC_VALUE
    frame["hips_bc_uncertainty_ugm3"] = frame["hips_fabs_uncertainty_Mm1"] / MAC_VALUE
    frame["volume_source"] = "mixed_export_without_per_row_provenance"
    frame.attrs.update({
        "source": PUBLIC_CHEMISTRY_PATH.as_posix(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "generated": payload.get("generated"),
        "field_units": payload.get("field_units", {}),
        "field_sources": payload.get("field_sources", {}),
        "source_note": payload.get("note", ""),
        "mac_m2_g": MAC_VALUE,
        "exclusion_policy": "preserve exported central-registry flags; no new value-based exclusions",
        "volume_caveat": VOLUME_CAVEAT,
        "claim_limit": "FTIR / HIPS optical disagreement, not independent thermal EC accuracy",
    })
    return frame


def optical_pair_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain all filters with an explicit optical source and eligibility reason.

    A finite directly exported Fabs takes priority. Only when it is absent may
    the separately identified public *HIPS* BC extend coverage. The public HIPS
    product uses MAC 10 in this export and is rescaled if the analysis MAC
    differs. SSR and ChemSpec carbon are never used as replacements.
    Nonpositive measurements stay in the ledger and in the
    primary calculation, with a separate flag for sensitivity analyses.
    """
    _assert_physical_rows(frame)
    required = ("ec_ftir_ugm3", "hips_fabs_Mm1", "bc_public_hips_ugm3", "is_excluded")
    if any(c not in frame for c in required):
        raise ValueError("use load_public_chemistry before constructing the pair ledger")
    out = frame.copy()
    if out["is_excluded"].isna().any():
        raise ValueError("is_excluded may not be unknown")
    direct = np.isfinite(out["hips_fabs_Mm1"])
    extension = ~direct & np.isfinite(out["bc_public_hips_ugm3"])
    out["optical_source"] = np.select([direct, extension], [DIRECT_HIPS, PUBLIC_EXTENSION], default="missing")
    out["optical_bc_ugm3"] = np.where(
        direct, out["hips_fabs_Mm1"] / MAC_VALUE,
        np.where(extension, out["bc_public_hips_ugm3"] * PUBLIC_HIPS_MAC_M2_G / MAC_VALUE, np.nan),
    )
    # get_clean_data also respects a caller's separately documented is_outlier flag.
    clean_ids = set(get_clean_data(out)["id"])
    clean = out["id"].isin(clean_ids)
    have_ec = np.isfinite(out["ec_ftir_ugm3"])
    have_optical = direct | extension
    out["pair_eligible"] = clean & have_ec & have_optical
    reason = np.where(~clean, "excluded_with_recorded_reason", "")
    reason = np.where(clean & ~have_ec & ~have_optical, "missing_ftir_ec_and_hips", reason)
    reason = np.where(clean & ~have_ec & have_optical, "missing_ftir_ec", reason)
    reason = np.where(clean & have_ec & ~have_optical, "missing_hips", reason)
    reason = np.where(out["pair_eligible"], "eligible_optical_diagnostic", reason)
    out["pair_reason"] = reason
    out["nonpositive_pair"] = out["pair_eligible"] & (
        out["ec_ftir_ugm3"].le(0) | out["optical_bc_ugm3"].le(0)
    )
    out["optical_disagreement_ugm3"] = (
        out["ec_ftir_ugm3"] - out["optical_bc_ugm3"]
    ).where(out["pair_eligible"])
    return out


def site_optical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe each site's direct, public-extension and combined optical pairs.

    Lambda=Var(FTIR-error)/Var(optical-error) is varied at 0.25, 1 and 4 only as
    sensitivity analysis. No lambda is called correct: the reported HIPS
    uncertainty's semantics and the target-domain FTIR error remain unresolved.
    R² is squared Pearson correlation, not predictive R². No improvement claim
    can be made from these in-sample summaries.
    """
    ledger = optical_pair_ledger(frame)
    rows = []
    for code, full in ledger.groupby("code", sort=True):
        clean = get_clean_data(full)
        pairs = full.loc[full["pair_eligible"]]
        for source in (DIRECT_HIPS, PUBLIC_EXTENSION, COMBINED_HIPS):
            part = pairs if source == COMBINED_HIPS else pairs.loc[pairs["optical_source"].eq(source)]
            x, y = part["optical_bc_ugm3"].to_numpy(float), part["ec_ftir_ugm3"].to_numpy(float)
            diff = y - x
            dates = pd.to_datetime(part["date"], errors="raise").dropna()
            row = {
                "code": code, "optical_source": source,
                "n_physical_filters": len(full), "n_excluded": len(full) - len(clean),
                "n_retained_filters": len(clean),
                "n_retained_with_ec": int(np.isfinite(clean["ec_ftir_ugm3"]).sum()),
                "n_pairs": len(part),
                "n_nonpositive_pairs": int(part["nonpositive_pair"].sum()),
                "n_with_hips_uncertainty": int(np.isfinite(part["hips_bc_uncertainty_ugm3"]).sum()),
                "n_with_reported_positive_volume": int((np.isfinite(part["volume_m3"]) & part["volume_m3"].gt(0)).sum()),
                "first_date": dates.min().date().isoformat() if len(dates) else None,
                "last_date": dates.max().date().isoformat() if len(dates) else None,
                "median_ec_ftir_ugm3": float(np.median(y)) if len(y) else np.nan,
                "median_optical_bc_ugm3": float(np.median(x)) if len(x) else np.nan,
                "median_ec_minus_optical_ugm3": float(np.median(diff)) if len(diff) else np.nan,
                "median_absolute_disagreement_ugm3": float(np.median(np.abs(diff))) if len(diff) else np.nan,
                "optical_disagreement_rmse_ugm3": float(np.sqrt(np.mean(diff**2))) if len(diff) else np.nan,
                "pearson_r": np.nan, "pearson_r_squared": np.nan,
                "mac_m2_g": MAC_VALUE,
                "fit_scope": "optical_agreement_only_no_independent_EC_truth",
                "volume_source": "mixed_export_without_per_row_provenance",
            }
            if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
                row["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
                row["pearson_r_squared"] = row["pearson_r"] ** 2
            for lam in DEMING_LAMBDAS:
                label = str(lam).replace(".", "p")
                slope, intercept = deming(x, y, lam=lam)
                row[f"deming_lambda_{label}_slope"] = float(slope)
                row[f"deming_lambda_{label}_intercept_ugm3"] = float(intercept)
            rows.append(row)
    result = pd.DataFrame(rows)
    result.attrs.update(frame.attrs)
    result.attrs["deming_note"] = "lambda sensitivity only; no validated target-domain variance ratio"
    return result
