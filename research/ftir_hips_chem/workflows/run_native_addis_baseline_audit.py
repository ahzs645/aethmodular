"""Audit native Addis spectra, saved signed spline baselines, and replicates.

This workflow is a diagnostic of baseline handling and optical disagreement.
It does not refit EC, replace source measurements, or use HIPS as EC truth.
Raw data stay outside version control. All filters, including undated samples,
field blanks and PM10, remain in the output ledger with explicit eligibility.

Run with the repository environment and an explicitly supplied source folder::

    uv run python research/ftir_hips_chem/workflows/run_native_addis_baseline_audit.py \
      --native-dir "$AETHMODULAR_ETAD_NATIVE_DIR"

The folder must contain ETAD_FTIR_spectra.csv, ETAD_metadata.csv,
etad_spectra_lotmap.csv, and spectra_baselined_AIRSPEC.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(ROOT / "research/ftir_ec_phase3/scripts"))

from airspec_baseline import SEG1, SEG2, airspec_baseline_matrix, make_mask  # noqa: E402
from config import MAC_VALUE  # noqa: E402
from data_matching import base_filter_id, pivot_filter_by_id  # noqa: E402
from outliers import apply_exclusion_flags  # noqa: E402
from portable_baseline_screen import local_band_contrast  # noqa: E402
from portable_cohort_screen import load_public_chemistry, optical_pair_ledger  # noqa: E402
from spectra.bands import band_mean  # noqa: E402
from spectra.preprocess import second_derivative  # noqa: E402


SOURCE_NAMES = (
    "ETAD_FTIR_spectra.csv", "ETAD_metadata.csv", "etad_spectra_lotmap.csv",
    "spectra_baselined_AIRSPEC.csv",
)
BANDS = {
    "near_1617": ((1600, 1635), ((1540, 1570), (1660, 1690))),
    "broad_1600": ((1580, 1650), ((1500, 1540), (1760, 1800))),
    "carbonyl": ((1700, 1750), ((1640, 1670), (1780, 1810))),
    "ch": ((2850, 2950), ((2750, 2800), (3000, 3040))),
    "oh_nh": ((3200, 3400), ((3080, 3130), (3500, 3550))),
}
VARIANTS = {"df6_4": (6, 4), "df8_4": (8, 4), "df6_3": (6, 3), "df6_6": (6, 6)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def load_native(folder):
    raw = pd.read_csv(folder / SOURCE_NAMES[0])
    metadata = pd.read_csv(folder / SOURCE_NAMES[1])
    lots = pd.read_csv(folder / SOURCE_NAMES[2])
    saved = pd.read_csv(folder / SOURCE_NAMES[3])
    if raw.SampleAnalysisId.duplicated().any() or saved.id.duplicated().any():
        raise ValueError("analysis IDs must be unique")
    if metadata.MediaId.duplicated().any() or lots.SampleAnalysisId.duplicated().any():
        raise ValueError("metadata keys must be unique")
    numeric = pd.to_numeric(raw.columns, errors="coerce").to_numpy(float)
    spectral = np.isfinite(numeric) & (numeric >= 100) & (numeric <= 5000)
    wn = numeric[spectral]
    if np.any(np.diff(wn) >= 0) or not np.allclose(np.diff(wn), np.diff(wn)[0]):
        raise ValueError("native grid must be uniformly spaced and descending")
    if raw.columns[spectral].tolist() != saved.columns[1:].tolist():
        raise ValueError("raw and saved corrected wavenumber columns differ")
    if set(raw.SampleAnalysisId) != set(saved.id):
        raise ValueError("raw and saved corrected analysis IDs differ")
    saved = saved.set_index("id").loc[raw.SampleAnalysisId]
    y = raw.loc[:, spectral].to_numpy(float)
    corrected = saved.to_numpy(float)
    region = make_mask(wn, SEG1) | make_mask(wn, SEG2)
    if not np.isfinite(y).all() or not np.isfinite(corrected[:, region]).all():
        raise ValueError("native inputs contain incomplete analyzed spectra")
    if np.isfinite(corrected[:, ~region]).any():
        raise ValueError("saved correction unexpectedly extends outside AIRSpec region")
    scans = raw[["SampleAnalysisId", "MediaId"]].merge(
        metadata, on="MediaId", how="left", validate="many_to_one", indicator=True,
    )
    if not scans._merge.eq("both").all():
        raise ValueError("some scans lack metadata")
    scans = scans.drop(columns="_merge")
    scans = scans.merge(lots, on=["SampleAnalysisId", "MediaId", "ExternalFilterId"],
                        how="left", validate="one_to_one", indicator=True)
    if not scans._merge.eq("both").all():
        raise ValueError("some scans lack a lot-map record")
    scans = scans.drop(columns="_merge")
    if scans.groupby("MediaId").LotId.nunique(dropna=False).max() > 1:
        raise ValueError("a physical filter maps to conflicting lots")
    scans["base_filter_id"] = scans.ExternalFilterId.map(base_filter_id)
    scans["native_date"] = pd.to_datetime(scans.SamplingStartDate, errors="coerce")
    scans["lot_group"] = scans.LotId.astype("Int64").astype("string").fillna("unknown")
    return scans, wn, y, corrected, region


def features(wn, raw, corrected, region):
    idx = int(np.argmin(np.abs(wn - 1617)))
    c = corrected[:, region]
    removed = raw[:, region] - c
    abs_area = np.abs(c).sum(axis=1)
    feat = {
        "raw_full_range": np.ptp(raw, axis=1),
        "raw_full_min": raw.min(axis=1), "raw_full_max": raw.max(axis=1),
        "raw_region_rms": np.sqrt(np.mean(raw[:, region]**2, axis=1)),
        "raw_1617": raw[:, idx], "corrected_1617": corrected[:, idx],
        "removed_1617": raw[:, idx] - corrected[:, idx],
        "corrected_rms": np.sqrt(np.mean(c**2, axis=1)),
        "removed_rms": np.sqrt(np.mean(removed**2, axis=1)),
        "negative_point_fraction": np.mean(c < 0, axis=1),
        "negative_absorbance_fraction": np.divide(
            np.maximum(-c, 0).sum(axis=1), abs_area,
            out=np.full(len(c), np.nan), where=abs_area > 0),
        "zero_point_fraction": np.mean(c == 0, axis=1),
        "raw_continuum_3500_3900": band_mean(raw, wn, (3500, 3900)),
        "removed_continuum_3500_3900": band_mean(raw - corrected, wn, (3500, 3900)),
    }
    # Derivative values are expressed per (cm^-1)^2, not per array-index^2.
    derivative = second_derivative(raw, window_length=15, polyorder=3) / np.median(np.diff(wn))**2
    feat["derivative_1617"] = derivative[:, idx]
    feat["derivative_near_1617"] = band_mean(derivative, wn, (1600, 1635))
    for name, (band, flanks) in BANDS.items():
        for label, values in (("raw", raw[:, region]), ("corrected", c)):
            contrast = local_band_contrast(wn[region], values, band=band, flanks=flanks)
            feat[f"{label}_{name}_contrast"] = contrast["height"]
    return pd.DataFrame(feat), derivative


def summarize_groups(frame):
    metrics = ["raw_1617", "corrected_1617", "corrected_rms", "negative_point_fraction",
               "negative_absorbance_fraction", "raw_near_1617_contrast",
               "corrected_near_1617_contrast", "raw_continuum_3500_3900"]
    rows = []
    for (kind, lot), group in frame.groupby(["ExternalFilterType", "lot_group"], dropna=False):
        row = {"type": kind, "lot": lot, "physical_filters": len(group),
               "scans": int(group.n_scans.sum()),
               "negative_corrected_1617_filters": int(group.corrected_1617.lt(0).sum())}
        for metric in metrics:
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def repeatability(scans, wn, raw, corrected, derivative, region):
    idx = int(np.argmin(np.abs(wn - 1617)))
    rows = []
    for media, indices in scans.groupby("MediaId", sort=True).indices.items():
        for a, b in itertools.combinations(indices, 2):
            item = {"MediaId": media, "base_filter_id": scans.iloc[a].base_filter_id,
                    "type": scans.iloc[a].ExternalFilterType, "lot": scans.iloc[a].lot_group,
                    "analysis_a": scans.iloc[a].SampleAnalysisId,
                    "analysis_b": scans.iloc[b].SampleAnalysisId}
            for label, matrix in (("raw", raw), ("corrected", corrected), ("derivative", derivative)):
                v1, v2 = matrix[a, region], matrix[b, region]
                rms_diff = np.sqrt(np.mean((v1-v2)**2))
                signal = (np.sqrt(np.mean(v1**2)) + np.sqrt(np.mean(v2**2))) / 2
                item[f"{label}_difference_rms"] = float(rms_diff)
                item[f"{label}_relative_difference_rms"] = float(rms_diff/signal) if signal else np.nan
                item[f"{label}_1617_abs_difference"] = float(abs(matrix[a, idx]-matrix[b, idx]))
                r = np.corrcoef(v1, v2)[0, 1] if np.std(v1) > 0 and np.std(v2) > 0 else np.nan
                item[f"{label}_spectral_pearson_r"] = float(r)
                item[f"{label}_spectral_pearson_r_squared"] = float(r*r)
                item[f"{label}_max_abs_difference"] = float(np.max(abs(v1-v2)))
            rfull = np.corrcoef(raw[a], raw[b])[0, 1]
            item["raw_full_grid_pearson_r"] = float(rfull)
            item["raw_full_grid_pearson_r_squared"] = float(rfull*rfull)
            rows.append(item)
    return pd.DataFrame(rows)


def associations(frame, *, target_column="hips_minus_ftir_ec_ugm3",
                 target_label="HIPS/MAC minus existing FTIR EC; not independent EC error"):
    """Prespecified descriptive associations with the measured optical gap.

    Rank adjustment is not causal identification. The FTIR term shares spectra
    with these predictors; no association here proves an EC measurement error.
    """
    rows = []
    feature_cols = [c for c in frame if c.startswith(("raw_", "corrected_", "removed_", "derivative_"))]
    feature_cols += ["negative_point_fraction", "negative_absorbance_fraction",
                     "segment1_sensitivity_rms", "segment2_sensitivity_rms"]
    eligible = frame.loc[frame.optical_pair_eligible].copy()
    for feature in feature_cols:
        need = [feature, target_column, "hips_bc_ugm3",
                "MassCollectedOnFilter_ug", "SampleVolume_m3"]
        sub = eligible.loc[np.isfinite(eligible[need].to_numpy(float)).all(axis=1)].copy()
        if len(sub) < 20:
            continue
        x, target = sub[feature].to_numpy(float), sub[target_column].to_numpy(float)
        dates = pd.to_datetime(sub.native_date)
        month = dates.dt.month.to_numpy()
        continuous = np.column_stack([
            sub.hips_bc_ugm3, sub.MassCollectedOnFilter_ug, sub.SampleVolume_m3,
            np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12),
            (dates-pd.Timestamp("2020-01-01")).dt.days,
        ])
        ranked = np.column_stack([rankdata(continuous[:, j]) for j in range(continuous.shape[1])])
        lots = pd.get_dummies(sub.lot_group, dtype=float).to_numpy()
        design = np.column_stack([np.ones(len(sub)), ranked, lots])
        if "current_optical_source" in sub:
            source = pd.get_dummies(sub.current_optical_source, dtype=float).to_numpy()
            design = np.column_stack([design, source])
        xr = rankdata(x) - design @ np.linalg.lstsq(design, rankdata(x), rcond=None)[0]
        yr = rankdata(target) - design @ np.linalg.lstsq(design, rankdata(target), rcond=None)[0]
        pearson = np.corrcoef(x, target)[0, 1] if np.std(x) > 0 else np.nan
        spearman = spearmanr(x, target).statistic if np.std(x) > 0 else np.nan
        partial = np.corrcoef(xr, yr)[0, 1] if np.std(xr) > 1e-10 and np.std(yr) > 1e-10 else np.nan
        rows.append({"feature": feature, "n_physical_filters": len(sub),
                     "pearson_r": pearson, "pearson_r_squared": pearson**2,
                     "spearman_r": spearman, "spearman_r_squared": spearman**2,
                     "partial_rank_r": partial, "partial_rank_r_squared": partial**2,
                     "adjustment_rank": int(np.linalg.matrix_rank(design)),
                     "target": target_label})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/ftir_hips_chem/output/tables/native_baseline_audit_20261102")
    parser.add_argument("--filter-dataset", type=Path,
                        default=ROOT / "research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scans, wn, raw, corrected, region = load_native(args.native_dir)
    sf, derivative = features(wn, raw, corrected, region)
    scan_ledger = pd.concat([scans, sf], axis=1)
    scan_ledger["prior_documented_review_flag"] = np.select(
        [scan_ledger.SampleAnalysisId.eq(406), scan_ledger.SampleAnalysisId.eq(1626),
         scan_ledger.base_filter_id.eq("ETAD-0158")],
        ["ETAD-0037 nearly flat spectrum; laboratory review already requested in docs/open-items.md",
         "ETAD-0139 first scan nearly flat; discordant repeat already documented in docs/open-items.md",
         "Filter hole documented in docs/open-items.md; no new canonical exclusion here"],
        default="",
    )
    scan_ledger.to_csv(args.output / "scan_ledger.csv", index=False)

    groups = scans.groupby("MediaId", sort=True).indices
    owner_indices = [indices for indices in groups.values()]
    physical = scans.iloc[[indices[0] for indices in owner_indices]].drop(columns="SampleAnalysisId").reset_index(drop=True)
    physical["n_scans"] = [len(indices) for indices in owner_indices]
    physical["analysis_ids"] = [";".join(scans.iloc[indices].SampleAnalysisId.astype(str)) for indices in owner_indices]
    mean_raw = np.vstack([raw[indices].mean(axis=0) for indices in owner_indices])
    # Baseline fitting is nonlinear: fit per scan first, then average.
    mean_corrected = np.vstack([corrected[indices].mean(axis=0) for indices in owner_indices])
    pf, _ = features(wn, mean_raw, mean_corrected, region)
    physical = pd.concat([physical, pf], axis=1)
    paired = pivot_filter_by_id(pd.read_pickle(args.filter_dataset), "ETAD",
                               params=["EC_ftir", "OC_ftir", "HIPS_Fabs"], hips_units="both")
    if paired.base_filter_id.duplicated().any():
        raise ValueError("source measurement pivot has duplicate physical filters")
    paired["date"] = pd.to_datetime(paired["date"], errors="raise")
    physical = physical.merge(paired.rename(columns={"date": "product_date"}),
                              on="base_filter_id", how="left", validate="one_to_one")
    physical["date"] = physical.native_date.fillna(physical.product_date)
    physical = apply_exclusion_flags(physical, "Addis_Ababa", date_tolerance_days=0)
    physical["native_product_date_difference_days"] = (
        physical.native_date - physical.product_date).dt.total_seconds() / 86400
    physical["hips_minus_ftir_ec_ugm3"] = physical.hips_bc_ugm3 - physical.ftir_ec
    physical["optical_pair_eligible"] = (
        physical.ExternalFilterType.eq("PM2.5") & ~physical.is_excluded &
        physical.native_date.notna() & physical.SampleVolume_m3.gt(0) &
        physical.native_product_date_difference_days.eq(0) &
        np.isfinite(physical[["ftir_ec", "hips_bc_ugm3"]]).all(axis=1)
    )
    physical["date_or_volume_missing"] = physical.native_date.isna() | ~physical.SampleVolume_m3.gt(0)
    physical["gallery_presence_expected"] = physical.ExternalFilterType.isin(["PM2.5", "PM10"]) & physical.native_date.notna()

    # Four fixed parameter combinations diagnose sensitivity; none is selected
    # using optical agreement and none changes the EC products above.
    variant_arrays = {}
    parameter_rows = []
    validation_rows = []
    for name, (df1, df2) in VARIANTS.items():
        print(f"Computing native spline {name}: {len(scans)} scans", flush=True)
        _, trial = airspec_baseline_matrix(wn, raw, df1=df1, df2=df2,
                                          n_jobs=args.jobs, chunksize=2)
        averaged = np.vstack([trial[indices].mean(axis=0) for indices in owner_indices])
        variant_arrays[name] = averaged
        tf, _ = features(wn, mean_raw, averaged, region)
        for i, row in physical.iterrows():
            delta = averaged[i, region] - mean_corrected[i, region]
            parameter_rows.append({"base_filter_id": row.base_filter_id,
                                   "type": row.ExternalFilterType, "lot": row.lot_group,
                                   "df1": df1, "df2": df2, "variant": name,
                                   "rms_change_from_saved_R": np.sqrt(np.mean(delta**2)),
                                   "corrected_1617": tf.at[i, "corrected_1617"],
                                   "corrected_near_1617_contrast": tf.at[i, "corrected_near_1617_contrast"],
                                   "negative_point_fraction": tf.at[i, "negative_point_fraction"],
                                   "corrected_rms": tf.at[i, "corrected_rms"]})
        if name == "df6_4":
            errors = trial[:, region] - corrected[:, region]
            p95 = np.quantile(abs(corrected[:, region]), 0.95, axis=1)
            maximum = abs(errors).max(axis=1)
            validation_rows = [{"SampleAnalysisId": scans.iloc[i].SampleAnalysisId,
                                "max_abs_error": maximum[i],
                                "rmse": np.sqrt(np.mean(errors[i]**2)),
                                "max_error_over_p95_signal": maximum[i]/p95[i]}
                               for i in range(len(scans))]
    baseline = variant_arrays["df6_4"][:, region]
    physical["segment1_sensitivity_rms"] = np.sqrt(np.mean((variant_arrays["df8_4"][:, region]-baseline)**2, axis=1))
    physical["segment2_sensitivity_rms"] = np.maximum.reduce([
        np.sqrt(np.mean((variant_arrays[name][:, region]-baseline)**2, axis=1))
        for name in ("df6_3", "df6_6")])
    parameter_table = pd.DataFrame(parameter_rows)
    validation = pd.DataFrame(validation_rows)
    reps = repeatability(scans, wn, raw, corrected, derivative, region)

    descriptor_path = ROOT / "gallery/app/public/data/research/lot_spectra/lot_spectra.json"
    descriptor = json.loads(descriptor_path.read_text())
    packed = pd.DataFrame(descriptor["filters"])
    packed = packed.loc[packed.code.eq("ETAD")]
    physical["in_packed_export"] = physical.base_filter_id.isin(packed.id)
    physical["packed_absence_evidence"] = np.select(
        [physical.in_packed_export, physical.ExternalFilterType.eq("FB"), physical.native_date.isna()],
        ["present", "field blank: excluded by exporter TYPES", "missing sampling date: excluded by exporter dated mask"],
        default="unexplained; investigate",
    )
    if physical.packed_absence_evidence.eq("unexplained; investigate").any():
        raise ValueError("unexplained native-versus-packed cohort difference")
    physical.to_csv(args.output / "physical_filter_ledger.csv", index=False)
    summarize_groups(physical).to_csv(args.output / "type_lot_summary.csv", index=False)
    parameter_table.to_csv(args.output / "parameter_sensitivity.csv", index=False)
    validation.to_csv(args.output / "r_python_validation.csv", index=False)
    reps.to_csv(args.output / "same_filter_replicate_pairs.csv", index=False)
    associations(physical).to_csv(args.output / "native_feature_associations.csv", index=False)
    physical.loc[~physical.in_packed_export].to_csv(args.output / "native_filters_absent_from_packed.csv", index=False)
    # A separate, current-chemistry view bridges the frozen unified dataset
    # and the portable screen. Keep the primary native ledger/products intact.
    current = optical_pair_ledger(load_public_chemistry(ROOT))
    current = current.loc[current.code.eq("ETAD"), [
        "id", "date", "pair_eligible", "ec_ftir_ugm3", "optical_bc_ugm3", "optical_source",
    ]].rename(columns={"id": "base_filter_id", "date": "current_date",
                       "pair_eligible": "current_measurement_pair_eligible",
                       "ec_ftir_ugm3": "current_ec_ftir_ugm3",
                       "optical_bc_ugm3": "current_optical_bc_ugm3",
                       "optical_source": "current_optical_source"})
    current["current_date"] = pd.to_datetime(current.current_date)
    current_native = physical.merge(current, on="base_filter_id", how="left", validate="one_to_one")
    current_native["frozen_native_pair_eligible"] = current_native.optical_pair_eligible
    current_native["current_packed_pair_eligible"] = (
        current_native.current_measurement_pair_eligible.eq(True) &
        current_native.ExternalFilterType.eq("PM2.5") & current_native.in_packed_export &
        current_native.native_date.eq(current_native.current_date) &
        current_native.SampleVolume_m3.gt(0)
    )
    current_native["cohort_membership"] = np.select([
        current_native.frozen_native_pair_eligible & current_native.current_packed_pair_eligible,
        current_native.frozen_native_pair_eligible, current_native.current_packed_pair_eligible,
    ], ["both", "frozen_native_only", "current_packed_only"], default="neither")
    current_native[["base_filter_id", "ExternalFilterType", "native_date", "current_date",
                    "frozen_native_pair_eligible", "current_packed_pair_eligible", "cohort_membership",
                    "ftir_ec", "hips_bc_ugm3", "current_ec_ftir_ugm3", "current_optical_bc_ugm3",
                    "current_optical_source"]].to_csv(args.output / "chemistry_cohort_reconciliation.csv", index=False)
    current_native["ftir_ec"] = current_native.current_ec_ftir_ugm3
    current_native["hips_bc_ugm3"] = current_native.current_optical_bc_ugm3
    current_native["optical_pair_eligible"] = current_native.current_packed_pair_eligible
    current_native["ftir_minus_optical_ugm3"] = current_native.ftir_ec-current_native.hips_bc_ugm3
    # Explicitly remove the old-sign target from this secondary analysis view.
    current_native = current_native.drop(columns="hips_minus_ftir_ec_ugm3")
    current_target = "current FTIR EC minus selected HIPS BC; not independent EC error"
    current_associations = []
    for cohort in ("current_combined_hips", "current_direct_hips"):
        view = current_native.copy()
        if cohort == "current_direct_hips":
            view["optical_pair_eligible"] &= view.current_optical_source.eq("direct_hips_fabs")
        table = associations(view, target_column="ftir_minus_optical_ugm3", target_label=current_target)
        table.insert(0, "cohort", cohort)
        current_associations.append(table)
    pd.concat(current_associations, ignore_index=True).to_csv(
        args.output / "current_chemistry_native_feature_associations.csv", index=False)
    scan_ledger.loc[scan_ledger.prior_documented_review_flag.ne("")].to_csv(
        args.output / "prior_documented_problem_scans.csv", index=False)
    focused_ids = ["ETAD-0036", "ETAD-0037", "ETAD-0040", "ETAD-0139",
                   "ETAD-0158", "ETAD-0202", "ETAD-0221"]
    focused = pd.concat([
        scan_ledger.loc[scan_ledger.base_filter_id.isin(focused_ids)].assign(row_kind="scan"),
        physical.loc[physical.base_filter_id.isin(focused_ids)].assign(row_kind="physical_filter_average"),
    ], ignore_index=True)
    focused.to_csv(args.output / "focused_filter_comparisons.csv", index=False)
    # Retain both acquired scans and the published average. The 1668-only row
    # is a labelled sensitivity result, not an exclusion or source replacement.
    p139 = physical.index[physical.base_filter_id.eq("ETAD-0139")].tolist()
    good139 = scans.index[scans.SampleAnalysisId.eq(1668)].tolist()
    prior_flag_sensitivity = []
    if len(p139) == len(good139) == 1:
        a, b = p139[0], good139[0]
        alt_features, _ = features(wn, raw[b:b+1], corrected[b:b+1], region)
        for label, values in (("all_acquired_scans_mean", pf.iloc[a]),
                              ("1668_only_review_sensitivity", alt_features.iloc[0])):
            prior_flag_sensitivity.append({"base_filter_id": "ETAD-0139", "variant": label,
                                           "source_ec_product_unchanged": physical.iloc[a].ftir_ec,
                                           "source_hips_bc_product_unchanged": physical.iloc[a].hips_bc_ugm3,
                                           **values.to_dict()})
    pd.DataFrame(prior_flag_sensitivity).to_csv(
        args.output / "etad0139_flagged_scan_sensitivity.csv", index=False)

    # Median curves retain sign, absolute scale, and blank traces for plotting.
    curve_rows = []
    for kind in physical.ExternalFilterType.unique():
        mask = physical.ExternalFilterType.eq(kind).to_numpy()
        for label, matrix in (("raw", mean_raw), ("saved_spline", mean_corrected)):
            curve_rows.extend({"type": kind, "view": label, "wavenumber": wn[j],
                               "median_absorbance": np.median(matrix[mask, j]),
                               "q25_absorbance": np.quantile(matrix[mask, j], 0.25),
                               "q75_absorbance": np.quantile(matrix[mask, j], 0.75),
                               "physical_filters": int(mask.sum())}
                              for j in np.flatnonzero(region))
    pd.DataFrame(curve_rows).to_csv(args.output / "signed_median_spectra.csv", index=False)

    blank = physical.loc[physical.ExternalFilterType.eq("FB")]
    pm25 = physical.loc[physical.ExternalFilterType.eq("PM2.5")]
    idx1617 = int(np.argmin(abs(wn-1617)))
    summary = {
        "scope": "native Addis baseline and optical-disagreement diagnostics, not EC validation",
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "inputs": [{"name": name, "bytes": (args.native_dir/name).stat().st_size,
                    "sha256": sha256(args.native_dir/name)} for name in SOURCE_NAMES],
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), ROOT / "research/ftir_ec_phase3/scripts/airspec_baseline.py",
            ROOT / "research/ftir_hips_chem/scripts/config.py",
            ROOT / "research/ftir_hips_chem/scripts/data_matching.py",
            ROOT / "research/ftir_hips_chem/scripts/outliers.py",
            ROOT / "research/ftir_hips_chem/scripts/portable_baseline_screen.py",
            ROOT / "research/ftir_hips_chem/scripts/portable_cohort_screen.py",
            ROOT / "research/ftir_hips_chem/scripts/spectra/bands.py",
            ROOT / "research/ftir_hips_chem/scripts/spectra/preprocess.py",
        )},
        "source_measurement_sha256": sha256(args.filter_dataset),
        "packed_descriptor_sha256": sha256(descriptor_path),
        "current_chemistry_sha256": sha256(ROOT / "gallery/app/public/data/spartan_reference.json"),
        "wavenumbers": len(wn), "corrected_wavenumbers": int(region.sum()),
        "wavenumber_step_cm_minus_1": abs(np.median(np.diff(wn))),
        "native_1617_wavenumber": wn[idx1617],
        "scans": len(scans), "physical_filters": len(physical),
        "physical_types": physical.ExternalFilterType.value_counts().to_dict(),
        "scan_types": scans.ExternalFilterType.value_counts().to_dict(),
        "pair_eligible_physical_filters": int(physical.optical_pair_eligible.sum()),
        "current_chemistry_paired_physical_filters": int(current_native.current_packed_pair_eligible.sum()),
        "current_chemistry_pair_sources": current_native.loc[
            current_native.current_packed_pair_eligible, "current_optical_source"].value_counts().to_dict(),
        "chemistry_cohort_membership": current_native.cohort_membership.value_counts().to_dict(),
        "native_product_date_mismatches": int(physical.native_product_date_difference_days.dropna().ne(0).sum()),
        "extra_scans": len(scans)-len(physical), "replicated_filters": reps.MediaId.nunique(),
        "replicate_pairs": len(reps),
        "retained_prior_documented_problem_scans": scan_ledger.loc[
            scan_ledger.prior_documented_review_flag.ne(""), "SampleAnalysisId"].tolist(),
        "blank_negative_1617_count": int(blank.corrected_1617.lt(0).sum()),
        "blank_1617_median": blank.corrected_1617.median(),
        "blank_rms_median": blank.corrected_rms.median(),
        "blank_negative_point_fraction_median": blank.negative_point_fraction.median(),
        "pm25_1617_median": pm25.corrected_1617.median(),
        "pm25_negative_point_fraction_median": pm25.negative_point_fraction.median(),
        "saved_R_port_all_cell_rmse": np.sqrt(np.mean(validation.rmse**2)),
        "saved_R_port_max_abs_error": validation.max_abs_error.max(),
        "saved_R_port_max_error_over_p95_signal": validation.max_error_over_p95_signal.max(),
        "segment1_change_at_1617_max_abs": float(np.max(abs(variant_arrays["df8_4"][:, idx1617]-variant_arrays["df6_4"][:, idx1617]))),
        "MAC_m2_per_g": MAC_VALUE,
        "diagnostic_parameter_variants": VARIANTS,
        "limitations": [
            "No independent EC reference for these native Addis PTFE filters.",
            "No native instrument IDs, acquisition timestamps, or background-scan identities in these four input files.",
            "Spectral correlations with HIPS/MAC minus FTIR share an FTIR-derived target term; they are not causal attribution.",
            "Parameter sensitivity cannot choose an accurate baseline or validate a new EC calibration.",
            "Negative corrected blank residuals can reflect correction shape, background mismatch, noise or field-blank material.",
            "Replicate pairs from one physical filter are not independent filters; repeatability is not external accuracy.",
            "The native derivative uses a 15-point, cubic Savitzky-Golay operation divided by grid spacing squared.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(safe_json(summary), indent=2) + "\n")
    print(json.dumps(safe_json(summary), indent=2))


if __name__ == "__main__":
    main()
