"""Export Research > Baseline lab from reproducible, existing-data investigations.

    uv run python research/ftir_hips_chem/workflows/run_portable_baseline_review.py
    AETHMODULAR_ETAD_NATIVE_DIR=<native inputs> uv run python gallery/data/export_baseline_lab.py

Required: portable review tables and the committed packed raw spectra.
Optional: the native audit, native control experiments, source-calibration
experiments and original native spectra. Optional datasets are reported as
absent rather than replaced with synthetic observations. The Research section
owns these private assets; they are excluded from public gallery builds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / "research/ftir_hips_chem"
TABLES = AREA / "output/tables"
for path in (AREA / "scripts", AREA / "workflows"):
    sys.path.insert(0, str(path))
from config import MAC_VALUE, SITES  # noqa: E402
from existing_data_cohorts import cohort_summaries  # noqa: E402
from portable_baseline_screen import asymmetric_baseline, load_packed_spectra  # noqa: E402

OUT = ROOT / "gallery/app/public/data/research/baseline_lab"
FIELDS = [
    "id", "code", "date", "type", "lot_group", "ethiopia_season_dry_feb",
    "is_excluded", "exclusion_reason", "spectral_pair_eligible", "spectral_pair_reason",
    "ec_ftir_ugm3", "optical_bc_ugm3", "optical_disagreement_ugm3", "optical_source",
    "pm25_ugm3", "volume_m3", "baseline_sensitivity_rms", "escape_recomputed",
    "protected_1617_change", "negative_fraction_soft", "removed_mean_soft",
    "contrast_1580_1650", "contrast_1600_1635", "contrast_1700_1750",
    "contrast_2850_2950", "contrast_3200_3400", "analyses", "qc",
]
REVIEW_NOTES = {
    "ETAD-0037": "Previously documented near-flat acquisition; review sample and background identity.",
    "ETAD-0139": "Discordant original scans 1626 and 1668; their mean dilutes the informative spectrum.",
    "ETAD-0158": "Previously documented physical hole; retained pending recorded laboratory adjudication.",
}
METHODS = [
    {"id": "raw", "label": "Raw absorbance"},
    {"id": "spline", "label": "Spline baseline", "description": "Signed native saved R result; df1=6, df2=4."},
    {"id": "exploratory_soft", "label": "Exploratory smoother A",
     "description": "Packed common grid, asymmetric smoother λ=10⁴; diagnostic only."},
    {"id": "exploratory_stiff", "label": "Exploratory smoother B",
     "description": "Packed common grid, asymmetric smoother λ=10⁶; diagnostic only."},
    {"id": "protected_soft", "label": "Protected-window smoother A",
     "description": "Constructed control only; excludes 1500–1760 from the baseline fit; λ=10⁴."},
    {"id": "protected_stiff", "label": "Protected-window smoother B",
     "description": "Constructed control only; excludes 1500–1760 from the baseline fit; λ=10⁶."},
]
METHOD_ALIASES = {
    "spline_default": "spline", "neutral_arpls": "neutral",
    "vibes_etad_r3": "vibes_local3", "vibes_etad_r8": "vibes_local8",
    "plain_lam1e4": "exploratory_soft", "plain_lam1e6": "exploratory_stiff",
    "protected_1500_1760_lam1e4": "protected_soft",
    "protected_1500_1760_lam1e6": "protected_stiff",
}
CALIBRATION_LABELS = {
    "raw_full": "Raw · 500–4000", "raw_shared": "Raw · 1425–4000",
    "spline6_4": "Spline baseline · df 6/4", "spline6_3": "Spline baseline · df 6/3",
    "spline6_6": "Spline baseline · df 6/6", "neutral_arpls": "Independent arPLS · default iterations",
    "removed_continuum": "Removed spline component",
    "raw_without_1600": "Raw · omit 1580–1650",
    "spline_without_1600": "Spline · omit 1580–1650",
    "spline_1600_only": "Spline · only 1580–1650",
    "second_derivative": "Second derivative",
}


def method_id(name):
    return METHOD_ALIASES.get(str(name), str(name))


def clean(value):
    """Keep absent observations null and retain signed finite measurements."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(f"{value:.9g}") if np.isfinite(value) else None
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), allow_nan=False, separators=(",", ":")) + "\n")


def records(frame):
    return clean(frame.to_dict("records"))


def read_csv(path):
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def curve(identifier, label, method, kind, wn, y, **extra):
    wn, y = np.asarray(wn, float), np.asarray(y, float)
    if wn.shape != y.shape:
        raise ValueError("spectral grids and values must align")
    # Preserve unsupported channels as null. Dropping them would join lines
    # across internal gaps (e.g. the neutral method's PTFE mask).
    keep = np.isfinite(wn)
    for key in ("lo", "hi"):
        if key in extra:
            values = np.asarray(extra[key], float)
            if values.shape != wn.shape:
                raise ValueError("spectral uncertainty and value grids must align")
            extra[key] = values[keep]
    return clean({"id": identifier, "label": label, "method": method, "kind": kind,
                  "wn": wn[keep], "y": y[keep], **extra})


def append_native_records(filters, native_ledger):
    """Append blank/undated native-only identities without creating optical pairs."""
    if native_ledger.empty:
        return filters
    seen = {r["id"] for r in filters}
    for _, r in native_ledger.iterrows():
        identifier = r["base_filter_id"]
        if identifier in seen:
            continue
        filters.append(clean({
            "id": identifier, "code": "ETAD", "date": r.get("native_date"),
            "type": r["ExternalFilterType"], "lot_group": str(r["lot_group"]),
            "is_excluded": bool(r.get("is_excluded", False)),
            "exclusion_reason": r.get("exclusion_reason"),
            "spectral_pair_eligible": False,
            "spectral_pair_reason": "Native blank or undated spectrum; no current eligible concentration pair.",
            "ec_ftir_ugm3": None, "optical_bc_ugm3": None, "optical_disagreement_ugm3": None,
            "optical_source": "none", "pm25_ugm3": None, "volume_m3": r.get("SampleVolume_m3"),
            "baseline_sensitivity_rms": None, "escape_recomputed": None,
            "review_note": r.get("packed_absence_evidence"),
        }))
        seen.add(identifier)
    return filters


def export_packed_spectra(out, folder):
    packed = load_packed_spectra(folder)
    raw = packed["views"]["raw"]["values"]
    wn = packed["views"]["raw"]["wavenumbers"]
    shared = wn >= 1425
    methods = {
        "exploratory_soft": asymmetric_baseline(wn[shared], raw[:, shared], lam=1e4),
        "exploratory_stiff": asymmetric_baseline(wn[shared], raw[:, shared], lam=1e6),
    }
    ids = []
    for i, identifier in enumerate(packed["filters"]["id"]):
        curves = [curve(identifier + "-raw", "Raw (binned)", "raw", "raw", wn, raw[i])]
        for key, run in methods.items():
            label = next(m["label"] for m in METHODS if m["id"] == key)
            for kind, values in (("corrected", run["corrected"]), ("removed", run["baseline"])):
                curves.append(curve(identifier + "-" + key + "-" + kind,
                                    label, key, kind, wn[shared], values[i]))
        write_json(out / "spectra" / (identifier + ".json"), {
            "schema_version": 1, "id": identifier, "source": "packed", "curves": curves,
            "notes": ["Raw spectra are binned at approximately 7.7 cm⁻¹.",
                      "Exploratory smoother curves are signed and are not a deployed EC calibration.",
                      "Clipped, area-normalized gallery AIRSpec display traces are not used."],
        })
        ids.append(identifier)
    return ids


def export_native_spectra(out, native_dir, controls_dir):
    if not native_dir or not (native_dir / "ETAD_FTIR_spectra.csv").is_file():
        return []
    from run_native_addis_baseline_audit import load_native
    from airspec_baseline import SEG1, SEG2, find_min_pos, make_mask

    scans, wn, raw, corrected, _ = load_native(native_dir)
    labels = {r["id"]: r["label"] for r in METHODS}
    labels.update({method_id(r["method"]): r["label"]
                   for r in records(read_csv(controls_dir / "methods.csv"))})
    shared = make_mask(wn, SEG1) | make_mask(wn, SEG2)
    segment2 = make_mask(wn, SEG2)
    segment_wn = wn[segment2]
    # The original spline forces corrected=0 below its extrapolation end.
    # Those are algorithmic padding, not measurements of zero absorbance.
    original_ends = np.array([segment_wn[min(len(segment_wn)-1,
                                           find_min_pos(segment_wn, row[segment2])+20)] for row in raw])
    supports = {"spline": shared[None, :] & (wn[None, :] >= original_ends[:, None]-1e-8)}
    boundaries = read_csv(controls_dir / "boundary_sensitivity.csv")
    for name, group in boundaries.groupby("method") if not boundaries.empty else []:
        ends = group.set_index("SampleAnalysisId").extension_last_wavenumber.reindex(scans.SampleAnalysisId)
        if ends.isna().any():
            raise ValueError("spline support lacks an original scan boundary")
        supports[method_id(name)] = shared[None, :] & (wn[None, :] >= ends.to_numpy()[:, None]-1e-8)
    additional = {}
    arrays_path = controls_dir / "native_method_arrays.npz"
    if arrays_path.is_file():
        with np.load(arrays_path, allow_pickle=False) as arrays:
            # Named arrays avoid any positional assumption about method/scan order.
            keys = set(arrays.files)
            grid_key = "wn" if "wn" in keys else "wavenumbers"
            id_key = "SampleAnalysisId" if "SampleAnalysisId" in keys else "analysis_ids"
            if grid_key in keys and id_key in keys:
                native_wn = arrays[grid_key]
                if not np.allclose(native_wn, wn):
                    raise ValueError("control experiment and original native grids differ")
                order = pd.Index(arrays[id_key].astype(str)).get_indexer(scans.SampleAnalysisId.astype(str))
                if (order < 0).any():
                    raise ValueError("control experiment lacks an original native scan")
                for key in sorted(keys):
                    if key.startswith("corrected__"):
                        method = method_id(key.removeprefix("corrected__"))
                        # The recorded R spline remains the original displayed result.
                        if method != "spline":
                            additional[method] = arrays[key][order]
    ids = []
    for identifier, group in scans.groupby("base_filter_id", sort=True):
        ix = group.index.to_numpy()
        curves = [curve(identifier + "-raw", "Raw · physical-filter mean", "raw", "raw", wn, raw[ix].mean(axis=0))]
        methods = {"spline": corrected, **additional}
        for method, matrix in methods.items():
            mean = matrix[ix].mean(axis=0)
            if method in supports:
                mean = np.where(supports[method][ix].all(axis=0), mean, np.nan)
            label = labels.get(method, method.replace("_", " "))
            curves.append(curve(identifier + "-" + method, label, method, "corrected", wn, mean))
            curves.append(curve(identifier + "-" + method + "-removed", label, method, "removed", wn, raw[ix].mean(axis=0)-mean))
        if len(ix) > 1:
            for i in ix:
                scan = str(scans.at[i, "SampleAnalysisId"])
                curves.append(curve(identifier + "-scan-" + scan, "Raw · scan " + scan,
                                    "raw", "raw", wn, raw[i], scan_id=scan))
                curves.append(curve(identifier + "-spline-scan-" + scan, "Spline · scan " + scan,
                                    "spline", "corrected", wn,
                                    np.where(supports["spline"][i], corrected[i], np.nan), scan_id=scan))
        write_json(out / "spectra" / (identifier + ".json"), {
            "schema_version": 1, "id": identifier, "source": "native", "curves": curves,
            "notes": ["Original signed absorbance; no area normalization or nonnegative clipping.",
                      "Repeated scans are averaged within each physical filter; original repeats remain inspectable.",
                      "Unsupported channels appear as gaps. Spline zeros below each scan's extension end are algorithmic padding and are hidden; numerical experiment scores retain the stated original pipeline convention.",
                      REVIEW_NOTES.get(identifier, "Baseline comparisons are diagnostic, not independently validated EC.")],
        })
        ids.append(identifier)
    return ids


def aggregate_curves(native):
    d = read_csv(native / "signed_median_spectra.csv")
    curves = []
    if not d.empty:
        for (kind, view), g in d.groupby(["type", "view"], sort=True):
            # The old aggregate file lacks each parent's support boundary.
            # Keep raw aggregates only until the native controls supply those
            # boundaries; a padded corrected median is not an observed trace.
            if view == "saved_spline":
                continue
            g = g.sort_values("wavenumber", ascending=False)
            curves.append(curve(
                str(kind) + "-" + str(view), str(kind) + " · " + str(view),
                "spline" if view == "saved_spline" else "raw",
                "corrected" if view == "saved_spline" else "raw",
                g.wavenumber, g.median_absorbance, group=kind,
                lo=g.q25_absorbance.to_numpy(), hi=g.q75_absorbance.to_numpy(),
                n=int(g.physical_filters.iloc[0]),
            ))
    return curves


def control_tables(folder):
    """Keep parent-held-out, lot-transfer and descriptive controls distinguishable."""
    methods, blanks, injections, repeats, curves = [], [], [], [], []
    for r in records(read_csv(folder / "methods.csv")):
        methods.append({"id": method_id(r["method"]), "label": r["label"],
                        "description": f"{r['family']}; {r['setting']}. Fit grid: {r['fit_grid']}; comparison: {r['comparison_grid']}."})
    blank_frame = read_csv(folder / "heldout_blank_metrics.csv")
    if not blank_frame.empty:
        if blank_frame.duplicated(["scheme", "method", "base_filter_id"]).any():
            raise ValueError("a physical blank appears twice in a validation scheme")
        for r in records(blank_frame):
            split = ("Held-out physical blanks · five folds" if r["scheme"] == "grouped_5fold"
                     else "Leave one lot out · " + str(r["fold"]).replace("train", "train ").replace("_test", " → test "))
            blanks.append({"id": r["base_filter_id"], "method": method_id(r["method"]),
                           "lot": r["lot_group"], "corrected_1617": r["corrected_1617"],
                           "corrected_rms": r["corrected_rms"], "negative_point_fraction": r["negative_fraction"],
                           "split": split, "converged": r.get("fit_converged", r.get("converged")),
                           "convergence_note": r.get("convergence_note") or r.get("status")})
    injected = pd.concat([read_csv(folder / name) for name in
                          ("injection_recovery.csv", "ambient_injection_recovery.csv")], ignore_index=True)
    if not injected.empty:
        keys = ["scheme", "method", "control", "amplitude"]
        if injected.duplicated([*keys, "base_filter_id"]).any():
            raise ValueError("injection children must have one observation per physical parent and setting")
        for (scheme, method, control, amplitude), group in injected.groupby(keys, sort=True):
            values = group.area_recovery_fraction.replace([np.inf, -np.inf], np.nan)
            prefix = {"grouped_5fold": "Held-out blank parents",
                      "ambient_input_stratified": "Ambient parents · input-stratified"}.get(scheme, str(scheme))
            converged = group.fit_converged.fillna(False).astype(bool) if "fit_converged" in group else None
            injections.append(clean({
                "method": method_id(method), "control": prefix + " · " + control.replace("_", " "),
                "amplitude": amplitude, "recovery_fraction": values.median(),
                "recovery_q25": values.quantile(.25), "recovery_q75": values.quantile(.75),
                "n_parents": group.base_filter_id.nunique(), "n_finite": values.notna().sum(),
                "n_converged": int(converged.sum()) if converged is not None else None,
                "recovery_converged": values[converged].median() if converged is not None else None,
                "units": "fraction of added area",
            }))
    repeated = read_csv(folder / "replicate_filter_metrics.csv")
    if not repeated.empty:
        if repeated.duplicated(["method", "base_filter_id"]).any():
            raise ValueError("repeat metrics must be aggregated to physical filters")
        for r in records(repeated):
            repeats.append({"id": r["base_filter_id"], "method": method_id(r["method"]),
                            "filter_type": r["filter_type"], "n_pairs": r["n_pairs"],
                            "raw_relative_difference_rms": r.get("raw_relative_pair_rms"),
                            "corrected_relative_difference_rms": r["relative_pair_rms"],
                            "raw_difference_rms": r["raw_pair_rms"],
                            "corrected_difference_rms": r["corrected_pair_rms"]})
    aggregate = read_csv(folder / "spectral_curves.csv")
    boundaries = read_csv(folder / "boundary_sensitivity.csv")
    if not aggregate.empty:
        for (method, kind), group in aggregate.groupby(["method", "filter_type"], sort=True):
            group = group.sort_values("wavenumber", ascending=False)
            if method.startswith("spline") and not boundaries.empty:
                boundary_rows = boundaries.loc[boundaries.method.eq(method) & boundaries.ExternalFilterType.eq(kind)]
                if len(boundary_rows):
                    supported = group.wavenumber.ge(boundary_rows.extension_last_wavenumber.max()-1e-8)
                    group.loc[~supported, ["median", "q25", "q75"]] = np.nan
            name = next((r["label"] for r in methods if r["id"] == method_id(method)), "Raw" if method == "raw" else method)
            curves.append(curve(
                "control-" + str(kind) + "-" + method, kind + " · " + name,
                method_id(method), "raw" if method == "raw" else "corrected", group.wavenumber, group["median"],
                group=kind, lo=group.q25, hi=group.q75, n=int(group.n_physical_filters.iloc[0]),
            ))
    return methods, blanks, injections, repeats, curves


def calibration_tables(folder):
    """Use matched source-reference refits; never label target optical agreement EC accuracy."""
    methods, rows = {}, []
    paired = read_csv(folder / "paired_site_bootstrap.csv")
    paired = {} if paired.empty else paired.set_index("arm").to_dict("index")
    populations = [
        (folder, "Source calibration · 905 filters · documented QC flag"),
        (folder / "historical906_all_scans", "Source sensitivity · 906 filters · flagged scan retained"),
    ]
    for source, cohort in populations:
        frame = read_csv(source / "metrics.csv")
        if frame.empty:
            continue
        for r in records(frame):
            evaluation = r["evaluation"]
            if evaluation == "site_grouped_pooled":
                split = "Site-held-out cross-validation · pooled"
            elif evaluation == "time_block_2022_11_01":
                split = "Later samples · after 2022-11-01"
            elif evaluation == "site_grouped_outer_fold":
                split = "Site-held-out fold " + str(r["fold"])
            else:
                continue
            identifier = "cal_" + r["arm"]
            methods[identifier] = {"id": identifier, "label": CALIBRATION_LABELS.get(r["arm"], r["arm"]),
                                   "description": "Separate source-trained PLS recalibration; existing collocated TOR reference."}
            uncertainty = paired.get(r["arm"]) if source == folder and evaluation == "site_grouped_pooled" else None
            uncertainty_note = (f" Paired site-bootstrap RMSE difference from spline 6/4: {uncertainty['delta_rmse']:.3f} "
                                f"µg/filter; 95% interval {uncertainty['delta_rmse_ci025']:.3f} to {uncertainty['delta_rmse_ci975']:.3f}."
                                if uncertainty else "")
            rows.append({"cohort": cohort, "method": identifier, "split": split, "n": r["n"],
                         "rmse": r["rmse"], "mae": r["mae"], "r_squared": r["predictive_r2"],
                         "units": "µg/filter", "reference": "Collocated IMPROVE thermal-optical reflectance EC",
                         "calibration_matched": True,
                         "note": "Source-domain validation; not independent Ethiopian EC. "
                                 "PLS components chosen inside the training fold. "
                                 + ("The previously documented TONT1 offset scan is flagged equally across all methods."
                                    if source == folder else "Retains the previously documented TONT1 offset scan to show its influence.") + uncertainty_note})
    # Put the pooled cross-validation first, followed by chronology and individual folds.
    rows.sort(key=lambda r: (r["cohort"], 0 if "pooled" in r["split"] else 1 if "Later" in r["split"] else 2, r["method"], r["split"]))
    note = ("Historical smoke-selected source calibration uses 905 physical filters from 140 sites, after the already documented "
            "TONT1 offset flag. The complete 906-filter sensitivity is kept separately. All candidates refit PLS "
            "inside matched folds. The source set is dominated by known lot 251 and does not establish performance "
            "on Ethiopian lot 248 or on the current global VIBES pipeline. The default neutral arPLS misses "
            "tolerance for 126/905 source spectra and 51/319 target scans; its fixed-preset refit is retained "
            "as such, separate from the 500-iteration native control. Candidate design is exploratory.") if rows else (
            "Existing source-reference recalibration is not present in this export.")
    return list(methods.values()), rows, note


def calibration_blank_rows(folder):
    return [{"method": "cal_" + r["arm"], "n": r["n_physical_blanks"],
             "median": r["median"], "mean": r["mean"], "q25": r["q25"], "q75": r["q75"],
             "rms_from_nominal_zero": r["rmse_from_nominal_zero"], "units": "µg/filter"}
            for r in records(read_csv(folder / "target_blank_predictions.csv"))]


def validate_seasonal_addon_inputs(main, addon):
    """Require identical shared source/data hashes for a separately run add-on."""
    if main.get("schema_version") != 1 or addon.get("schema_version") != 1:
        raise ValueError("seasonal main and add-on must use schema_version 1")
    source = addon.get("provenance", {})
    start, end = source.get("hashes_start", {}), source.get("hashes_end", {})
    if not start or start != end or source.get("source_and_inputs_unchanged") is not True:
        raise ValueError("seasonal add-on source/input hashes changed during analysis")
    main_inputs = main.get("provenance", {}).get("inputs", {})
    required = ("main_workflow", "core", "calendar_config", "ledger", "arrays", "boundaries", "weather")
    matched = {}
    for role in required:
        record = start.get(role, {})
        candidates = [path for path in main_inputs if Path(path).name == record.get("filename")]
        if len(candidates) != 1:
            raise ValueError(f"seasonal add-on has no unambiguous common input for {role}")
        path = candidates[0]
        digest = record.get("sha256", "")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or digest != main_inputs[path]):
            raise ValueError(f"seasonal add-on SHA-256 mismatch for common input {role}")
        matched[path] = digest
    return matched


def merge_seasonal_region_analysis(main, addon):
    """Validate and add held-out seasonal diagnostics without refitting a model."""
    matched = validate_seasonal_addon_inputs(main, addon)
    source = addon["provenance"]

    model_ids = {row["id"] for row in main.get("models", [])}
    fold_ids = {row["id"] for row in main.get("folds", [])}
    identity = ("task", "calendar", "cohort", "model_id", "fold_id")

    def score_key(row):
        return tuple(row.get(k) for k in identity)

    score_index = {score_key(row): row for row in main.get("scores", [])}

    def parent_score(row):
        if row.get("model_id") not in model_ids or row.get("fold_id") not in fold_ids:
            raise ValueError("seasonal add-on references an unknown model or fold")
        parent = score_index.get(score_key(row))
        if parent is None or not row.get("season"):
            raise ValueError("seasonal add-on lacks a matching pooled score or explicit season")
        return parent

    performance = addon.get("performance", [])
    if not performance or not addon.get("importance"):
        raise ValueError("seasonal add-on needs performance and interval importance rows")
    seasonal_scores = list(main.get("seasonal_region_scores", []))
    season_index = {(*score_key(row), row.get("season")): row for row in seasonal_scores}
    for row in performance:
        parent = parent_score(row)
        key = (*score_key(row), row["season"])
        if key in season_index:
            raise ValueError("duplicate seasonal performance key")
        if (row.get("n_train") != parent.get("n_train")
                or not 0 < row.get("n_test", 0) <= parent.get("n_test", 0)):
            raise ValueError("seasonal add-on sample counts disagree with the pooled model")
        labelled = {**row, "training_scope": "pooled", "evaluation_scope": "selected_heldout_season"}
        seasonal_scores.append(labelled)
        season_index[key] = labelled

    importance = list(main.get("importance", []))
    importance_fields = ("task", "model_id", "fold_id", "kind", "scheme", "region_id", "season")
    seen = {tuple(row.get(k) for k in importance_fields) for row in importance}
    if len(seen) != len(importance):
        raise ValueError("duplicate seasonal importance key in main analysis")
    for row in addon["importance"]:
        parent_score(row)
        key = tuple(row.get(k) for k in importance_fields)
        if key in seen:
            raise ValueError("duplicate seasonal importance key")
        seen.add(key)
        performance_row = season_index.get((*score_key(row), row["season"]))
        if (performance_row is None or row.get("n_test") != performance_row.get("n_test")
                or row.get("kind") != "permutation" or row.get("scheme") != "within_season"
                or not row.get("region_id")):
            raise ValueError("seasonal importance must match its within-season performance population")
        n_train, n_test = performance_row["n_train"], performance_row["n_test"]
        importance.append({
            **row, "n_train": n_train, "training_scope": "pooled",
            "evaluation_scope": "selected_heldout_season",
            "evaluation_note": f"Model trained on {n_train} pooled filters; joint interval shuffles among {n_test} held-out filters in {row['season']}.",
        })
    return {
        **main, "importance": importance, "seasonal_region_scores": seasonal_scores,
        "provenance": {**main.get("provenance", {}), "seasonal_region_analysis": {
            **source, "matched_main_inputs": matched,
            "limitations": addon.get("limitations", []),
        }},
    }


def merge_spline_transition_analysis(main, addon):
    """Append new sensitivity models only when their original populations match."""
    matched = validate_seasonal_addon_inputs(main, addon)
    for field in ("models", "representations"):
        existing_ids = {row["id"] for row in main.get(field, [])}
        new_ids = [row.get("id") for row in addon.get(field, [])]
        if not new_ids or not all(new_ids) or len(set(new_ids)) != len(new_ids) or existing_ids.intersection(new_ids):
            raise ValueError(f"transition {field} must have new, unique IDs without collisions")
    model_ids = {row["id"] for row in addon["models"]}
    representations = {row["id"] for row in [*main.get("representations", []), *addon["representations"]]}
    if any(row.get("representation") not in representations for row in addon["models"]):
        raise ValueError("transition model references an unknown representation")
    fold_ids = {row["id"] for row in main.get("folds", [])}
    context_fields = ("task", "calendar", "cohort", "fold_id")

    def context(row):
        return tuple(row.get(field) for field in context_fields)

    population_counts = {}
    for row in main.get("scores", []):
        population_counts.setdefault(context(row), set()).add((row.get("n_train"), row.get("n_test")))
    heldout_ids = {}
    for row in main.get("predictions", []):
        heldout_ids.setdefault(context(row), set()).add(row.get("id"))
    scores, predictions = addon.get("scores", []), addon.get("predictions", [])
    if not scores or not predictions:
        raise ValueError("transition add-on needs scores and held-out predictions")
    score_index = {}
    for row in scores:
        if row.get("model_id") not in model_ids or row.get("fold_id") not in fold_ids:
            raise ValueError("transition score references an unknown model or fold")
        population = context(row)
        if (row.get("n_train"), row.get("n_test")) not in population_counts.get(population, set()):
            raise ValueError("transition sample counts or evaluation population disagree with main analysis")
        key = (*population, row["model_id"])
        if key in score_index:
            raise ValueError("duplicate transition score key")
        score_index[key] = row
    predicted_ids = {}
    for row in predictions:
        population = context(row)
        key = (*population, row.get("model_id"))
        identifier = row.get("id")
        if key not in score_index or identifier not in heldout_ids.get(population, set()):
            raise ValueError("transition prediction references an unknown score or held-out physical filter")
        ids = predicted_ids.setdefault(key, set())
        if identifier in ids:
            raise ValueError("duplicate transition physical-filter prediction")
        ids.add(identifier)
    for key, row in score_index.items():
        ids = predicted_ids.get(key, set())
        if len(ids) != row["n_test"] or ids != heldout_ids.get(context(row), set()):
            raise ValueError("transition predictions must cover the unchanged held-out physical population")
    merged = {**main, **{field: [*main.get(field, []), *addon[field]]
                        for field in ("models", "representations", "scores", "predictions")}}
    merged["provenance"] = {
        **main.get("provenance", {}), "spline_transition_analysis": {
            **addon["provenance"], "matched_main_inputs": matched,
            **{key: addon.get(key) for key in ("transition_definition", "masked_channels", "adjacent_channel_summary", "limitations")},
        },
    }
    return merged


def clear_generated_ml_tables(out, names, input_dirs):
    """Delete only explicitly owned download directories, never their inputs."""
    owned_dirs = [out / "tables" / name for name in names]
    for target in owned_dirs:
        if any(source.resolve().is_relative_to(target.resolve()) for source in input_dirs):
            raise ValueError("seasonal input directory must not be inside its generated export directory")
    for target in owned_dirs:
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)


def export_seasonal_ml(out, main_dir, by_season_dir, transition_dir=None):
    """Publish optional ML views; clear only this exporter's generated ML files."""
    if transition_dir is None:
        transition_dir = TABLES / "spline_transition_sensitivity_20261103"
    main_source = main_dir / "seasonal_ml.json"
    folders = []
    merged = None
    if main_source.is_file():
        folders.append(("seasonal_ml", main_dir))
        for prefix, folder, filename, merge in (
            ("seasonal_ml_by_season", by_season_dir, "by_season.json", merge_seasonal_region_analysis),
            ("spline_transition", transition_dir, "transition_sensitivity.json", merge_spline_transition_analysis),
        ):
            addon_source = folder / filename
            if addon_source.is_file():
                merged = merge(merged if merged is not None else json.loads(main_source.read_text()),
                               json.loads(addon_source.read_text()))
                folders.append((prefix, folder))
    clear_generated_ml_tables(out, ("seasonal_ml", "seasonal_ml_by_season", "spline_transition"),
                              (main_dir, by_season_dir, transition_dir))
    served = out / "seasonal_ml.json"
    if not folders:
        served.unlink(missing_ok=True)
    elif merged is None:
        shutil.copyfile(main_source, served)
    else:
        # Keep main numerical values exact; the general gallery cleaner rounds floats.
        served.write_text(json.dumps(merged, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    return folders


SEASONAL_CHECKS_KEYS = ("lots", "ec_line", "residual_scores", "residual_importance", "lot_control", "lot_separability",
                         "limitations", "provenance")


def export_seasonal_checks(out, checks_dir):
    """Serve the lot / FTIR-EC controls (run_seasonal_ml_checks.py) beside the seasonal ML view.

    The checks are a separate saved analysis: they never alter the main ML scores. The served
    JSON is copied byte for byte; only this exporter's own download folder is replaced.
    """
    source = checks_dir / "checks.json"
    clear_generated_ml_tables(out, ("seasonal_checks",), (checks_dir,))
    served = out / "seasonal_checks.json"
    if not source.is_file():
        served.unlink(missing_ok=True)
        return []
    payload = json.loads(source.read_text())
    missing = [key for key in SEASONAL_CHECKS_KEYS if key not in payload]
    if payload.get("schema_version") != 1 or missing:
        raise ValueError(f"seasonal checks must be schema_version 1 with {missing or 'all keys'}")
    if payload["provenance"].get("inputs_unchanged_during_run") is not True:
        raise ValueError("seasonal checks inputs changed during the run")
    if not payload["ec_line"] or not payload["residual_scores"]:
        raise ValueError("seasonal checks need EC-line and residual scores")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, served)
    return [("seasonal_checks", checks_dir)]


def copy_download_folders(out, download_folders):
    downloads = []
    for prefix, folder in download_folders:
        if not folder.is_dir():
            continue
        for source in sorted(folder.rglob("*")):
            if not source.is_file() or source.suffix not in (".csv", ".json"):
                continue
            relative = source.relative_to(folder)
            target = out / "tables" / prefix / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            downloads.append({"label": prefix + " · " + str(relative), "path": "research/baseline_lab/tables/" + prefix + "/" + relative.as_posix()})
    return downloads


def update_seasonal_checks_only(out, checks_dir):
    """Refresh only the checks in an existing export (the full export needs the native audit inputs)."""
    main_path = out / "baseline_lab.json"
    if not main_path.is_file():
        raise ValueError("no existing baseline_lab.json to update; run the full export")
    data = json.loads(main_path.read_text())
    folders = export_seasonal_checks(out, checks_dir)
    downloads = [d for d in data.get("downloads", []) if not d["label"].startswith("seasonal_checks · ")]
    data["downloads"] = [*downloads, *copy_download_folders(out, folders)]
    write_json(main_path, data)
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        inputs = {k: v for k, v in manifest.get("inputs", {}).items() if not Path(k).is_relative_to(checks_dir.relative_to(ROOT))} \
            if checks_dir.is_relative_to(ROOT) else manifest.get("inputs", {})
        for _, folder in folders:
            for p in sorted(folder.rglob("*")):
                if p.is_file() and p.suffix in (".csv", ".json") and p.is_relative_to(ROOT):
                    inputs[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest["inputs"] = inputs
        write_json(manifest_path, manifest)
    return folders


def build_export(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(args.portable_dir / "filter_diagnostic_ledger.csv")
    if f.id.duplicated().any():
        raise ValueError("duplicate physical filter in portable ledger")
    f["review_note"] = f.id.map(REVIEW_NOTES)
    filters = records(f.reindex(columns=[*FIELDS, "review_note"]))
    for row in filters:
        row["is_excluded"] = row["is_excluded"] is True
        row["spectral_pair_eligible"] = row["spectral_pair_eligible"] is True
    native = read_csv(args.native_audit_dir / "physical_filter_ledger.csv")
    filters = append_native_records(filters, native)
    associations = []
    d = read_csv(args.portable_dir / "within_site_associations.csv")
    for r in records(d):
        associations.append({**r, "cohort": "Portable current products", "controls": "Optical BC, PM2.5, volume, month, time, lot and source."})
    d = read_csv(args.native_audit_dir / "current_chemistry_native_feature_associations.csv")
    for r in records(d):
        associations.append({**r, "code": "ETAD", "n": r["n_physical_filters"],
                             "cohort": "Native · " + r["cohort"],
                             "controls": "Current FTIR minus HIPS; optical BC, collected mass, volume, month, time, lot and source."})
    chronology = []
    for name, population in (("addis_forward_summary.csv", "All eligible"), ("addis_forward_population_sensitivity.csv", None)):
        for r in records(read_csv(args.portable_dir / name)):
            chronology.append({**r, "population": population or r["population"]})
    cohorts, matches, selection = cohort_summaries(f)
    analysis_out = TABLES / "baseline_lab_20261103"
    analysis_out.mkdir(parents=True, exist_ok=True)
    cohorts.to_csv(analysis_out / "cohort_summaries.csv", index=False)
    matches.to_csv(analysis_out / "ethiopia_matched_pairs.csv", index=False)
    write_json(analysis_out / "cohort_selection.json", selection)
    blanks = []
    if not native.empty:
        for r in records(native.loc[native.ExternalFilterType.eq("FB")]):
            blanks.append({"id": r["base_filter_id"], "method": "spline", "lot": r["lot_group"],
                           "corrected_1617": r["corrected_1617"], "corrected_rms": r["corrected_rms"],
                           "negative_point_fraction": r["negative_point_fraction"], "split": "Saved spline · descriptive"})
    repeats = []
    for r in records(read_csv(args.native_audit_dir / "same_filter_replicate_pairs.csv")):
        repeats.append({**r, "id": r["base_filter_id"], "method": "spline"})
    injections = []
    # Synthetic controls stay explicitly separate from the new actual-blank experiments.
    for r in records(read_csv(args.portable_dir / "synthetic_smoother_controls.csv")):
        injections.append({"method": method_id(r["candidate"]), "control": "Constructed · " + r["control"],
                           "amplitude": r["amplitude_absorbance"], "recovery_fraction": r["area_recovery_fraction"],
                           "n_parents": r["independent_parent_count"], "units": "fraction of added area"})
    ctrl_methods, ctrl_blanks, ctrl_injections, ctrl_repeats, ctrl_curves = control_tables(args.controls_dir)
    cal_methods, calibration, calibration_note = calibration_tables(args.calibration_dir)
    methods = list({r["id"]: r for r in [*METHODS, *ctrl_methods, *cal_methods]}.values())
    blanks.extend(ctrl_blanks)
    injections = [*ctrl_injections, *injections]
    # New repeat experiments use one point per physical filter, not one per pair.
    # The older pair table remains downloadable for provenance.
    if ctrl_repeats:
        repeats = ctrl_repeats
    spectra_ids = export_packed_spectra(out, ROOT / "gallery/app/public/data/research/lot_spectra")
    native_ids = export_native_spectra(out, args.native_dir, args.controls_dir)
    spectra_ids = sorted(set(spectra_ids) | set(native_ids))
    by_code = {v["code"]: {"code": v["code"], "name": v.get("label", k.replace("_", " ")), "color": v["color"]}
               for k, v in SITES.items()}
    # Bishoftu is a newer site; use its recorded colour from the current site export.
    meta = json.loads((ROOT / "gallery/app/public/data/spartan_map.json").read_text())
    for site in meta.get("sites", []):
        code = site.get("code")
        if code and code not in by_code:
            by_code[code] = {"code": code, "name": site.get("name", code), "color": site.get("color", "#607d8b")}
    downloads = []
    seasonal_ml_dir = getattr(args, "seasonal_ml_dir", TABLES / "seasonal_spectral_ml_20261103")
    seasonal_by_season_dir = getattr(args, "seasonal_ml_by_season_dir", TABLES / "seasonal_spectral_ml_by_season_20261103")
    spline_transition_dir = getattr(args, "spline_transition_dir", TABLES / "spline_transition_sensitivity_20261103")
    seasonal_pmf_dir = getattr(args, "seasonal_pmf_dir", TABLES / "seasonal_pmf_triangulation_20261103")
    seasonal_folders = export_seasonal_ml(out, seasonal_ml_dir, seasonal_by_season_dir, spline_transition_dir)
    seasonal_folders += export_seasonal_checks(out, getattr(args, "seasonal_checks_dir", TABLES / "seasonal_ml_checks_20261103"))
    clear_generated_ml_tables(out, ("seasonal_pmf",), (seasonal_pmf_dir,))
    # Globally fitted PMF results are downloads only, never ML inputs or model scores.
    pmf_folders = [("seasonal_pmf", seasonal_pmf_dir)] if seasonal_pmf_dir.is_dir() else []
    download_folders = [("portable", args.portable_dir), ("native", args.native_audit_dir),
                        ("controls", args.controls_dir), ("calibration", args.calibration_dir),
                        ("cohorts", analysis_out), *seasonal_folders, *pmf_folders]
    downloads.extend(copy_download_folders(out, download_folders))
    data = {
        "schema_version": 1, "generated": datetime.now(timezone.utc).isoformat(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "mac_m2_g": MAC_VALUE, "sites": [by_code[c] for c in sorted(f.code.unique())],
        "methods": methods, "filters": filters, "associations": associations, "chronology": chronology,
        "cohorts": records(cohorts), "blanks": blanks, "injections": injections, "repeats": repeats,
        "curves": ctrl_curves or aggregate_curves(args.native_audit_dir), "calibration": calibration,
        "calibration_blanks": calibration_blank_rows(args.calibration_dir),
        "calibration_blanks_cohort": "Primary 905 source models · 37 Addis field blanks",
        "calibration_note": calibration_note, "spectra_base": "research/baseline_lab/spectra/",
        "spectra_ids": spectra_ids, "native_spectra_ids": native_ids, "downloads": downloads,
        "notes": [
            "Existing measurements only. FTIR minus optical BC is disagreement, not independently measured EC error.",
            "FTIR-derived EC shares information with the spectral predictors; associations do not establish baseline causality.",
            "Native blank and repeat controls assess reproducibility and signal preservation.",
            "Spectral display traces remain signed. No source EC product or canonical exclusion is changed.",
            "Unsupported spectral channels are displayed as gaps. Spline RMS comparisons retain the original algorithm's forced-zero tail inside the stated 1425–4000 cm⁻¹ scoring window.",
            "Current Bishoftu spectral coverage does not include Kiremt.",
        ],
    }
    write_json(out / "baseline_lab.json", data)
    write_json(out / "manifest.json", {
        "schema_version": 1, "source_revision": data["source_revision"],
        "main": "baseline_lab.json", "spectra_count": len(spectra_ids), "native_spectra_count": len(native_ids),
        "cohort_selection": selection,
        "inputs": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for prefix, folder in download_folders if prefix != "cohorts" and folder.is_dir()
                   for p in sorted(folder.rglob("*"))
                   if p.is_file() and p.suffix in (".csv", ".json") and p.is_relative_to(ROOT)},
    })
    print(json.dumps({"filters": len(filters), "eligible_pairs": int(f.spectral_pair_eligible.sum()),
                      "spectra": len(spectra_ids), "native_spectra": len(native_ids),
                      "matched_ethiopia_pairs": len(matches),
                      "output": str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out)}))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--portable-dir", type=Path, default=TABLES / "baseline_review_20261102")
    parser.add_argument("--native-audit-dir", type=Path, default=TABLES / "native_baseline_audit_20261102")
    parser.add_argument("--controls-dir", type=Path, default=TABLES / "baseline_existing_data_20261103")
    parser.add_argument("--calibration-dir", type=Path, default=TABLES / "calibration_existing_data_20261103")
    parser.add_argument("--seasonal-ml-dir", type=Path, default=TABLES / "seasonal_spectral_ml_20261103")
    parser.add_argument("--seasonal-ml-by-season-dir", type=Path, default=TABLES / "seasonal_spectral_ml_by_season_20261103")
    parser.add_argument("--spline-transition-dir", type=Path, default=TABLES / "spline_transition_sensitivity_20261103")
    parser.add_argument("--seasonal-pmf-dir", type=Path, default=TABLES / "seasonal_pmf_triangulation_20261103")
    parser.add_argument("--seasonal-checks-dir", type=Path, default=TABLES / "seasonal_ml_checks_20261103")
    parser.add_argument("--seasonal-checks-only", action="store_true",
                        help="refresh only seasonal_checks.json and its downloads in an existing export")
    parser.add_argument("--native-dir", type=Path,
                        default=Path(os.environ["AETHMODULAR_ETAD_NATIVE_DIR"]) if os.environ.get("AETHMODULAR_ETAD_NATIVE_DIR") else None)
    args = parser.parse_args()
    if args.seasonal_checks_only:
        folders = update_seasonal_checks_only(args.output, args.seasonal_checks_dir)
        print(json.dumps({"seasonal_checks": bool(folders), "output": str(args.output)}))
        return
    build_export(args)


if __name__ == "__main__":
    main()
