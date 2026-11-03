"""Evaluate native baseline reliability with the existing ETAD scans and blanks.

Run with the repository's explorer/vibes dependencies installed::

    uv run --extra explorer --extra vibes python \
      research/ftir_hips_chem/workflows/run_existing_data_baseline_controls.py \
      --native-dir "$AETHMODULAR_ETAD_NATIVE_DIR"

No EC calibration, optical response, canonical exclusion or input file is
changed. Candidate methods are compared on the same native 1425–4000 cm^-1
grid. Local fixed-rank VIBES banks are distinct from the global rank-70 product.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_baseline_controls import (  # noqa: E402
    METHODS, INJECTIONS, AMPLITUDES, blank_fold_assignments,
    correct_vibes, fixed_vibes_background, neutral_correction,
    paired_injection_metrics, segmented_spline_variants,
)
from portable_baseline_screen import local_band_contrast  # noqa: E402
from run_native_addis_baseline_audit import load_native, SOURCE_NAMES  # noqa: E402

BANDS = {
    "near_1617": ((1600, 1635), ((1540, 1570), (1660, 1690))),
    "broad_1600": ((1580, 1650), ((1500, 1540), (1760, 1800))),
    "carbonyl": ((1700, 1750), ((1640, 1670), (1780, 1810))),
    "ch": ((2850, 2950), ((2750, 2800), (3000, 3040))),
    "oh_nh": ((3200, 3400), ((3080, 3130), (3500, 3550))),
}
SEED = 20261103


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe_json(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def metrics(wn, raw, corrected, shared):
    """All comparable metrics use the identical native shared window."""
    values = corrected[:, shared]
    valid = np.isfinite(values).all(axis=1)
    out = pd.DataFrame({"success": valid,
                        "raw_rms": np.sqrt(np.mean(raw[:, shared]**2, axis=1))})
    out["corrected_rms"] = np.sqrt(np.mean(values**2, axis=1))
    out["corrected_mean"] = np.mean(values, axis=1)
    out["negative_fraction"] = np.where(valid, np.mean(values < 0, axis=1), np.nan)
    out["removed_rms"] = np.sqrt(np.mean((raw[:, shared]-values)**2, axis=1))
    index = int(np.argmin(abs(wn-1617)))
    out["corrected_1617"] = corrected[:, index]
    out["removed_1617"] = raw[:, index]-corrected[:, index]
    for name, (band, flanks) in BANDS.items():
        heights = np.full(len(values), np.nan)
        if valid.any():
            heights[valid] = local_band_contrast(
                wn[shared], values[valid], band=band, flanks=flanks,
            )["height"]
        out[f"{name}_contrast"] = heights
    return out


def with_convergence(frame, converged):
    """Keep finite-but-unconverged values, without labelling them successful."""
    frame = frame.copy()
    frame["finite_output"] = frame.success
    frame["fit_converged"] = np.asarray(converged, bool)
    frame["success"] = frame.finite_output & frame.fit_converged
    return frame


def spectral_rows(wn, matrices, metadata, shared):
    # Retain native channels around the questioned band; thin only the displayed
    # long-range summaries. Numerical metrics always use the full native grid.
    pick = np.unique(np.concatenate((np.flatnonzero(shared)[::8],
                                    np.flatnonzero((wn >= 1500) & (wn <= 1800)))))
    rows = []
    for method, matrix in matrices.items():
        for kind in ("PM2.5", "FB", "PM10"):
            mask = metadata.ExternalFilterType.eq(kind).to_numpy()
            valid = mask & np.isfinite(matrix[:, shared]).all(axis=1)
            if not valid.any():
                continue
            q25, med, q75 = np.quantile(matrix[valid][:, pick], [.25, .5, .75], axis=0)
            rows.extend({"method": method, "filter_type": kind, "wavenumber": wn[j],
                         "median": med[k], "q25": q25[k], "q75": q75[k],
                         "n_physical_filters": int(valid.sum())}
                        for k, j in enumerate(pick))
    return pd.DataFrame(rows), pick


def run(args):
    start = time.perf_counter()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    def log(text):
        print(f"[{time.perf_counter()-start:7.1f}s] {text}", flush=True)
        (output / "execution_status.json").write_text(json.dumps({
            "state": "running", "stage": text, "elapsed_seconds": time.perf_counter()-start,
        }, indent=2)+"\n")

    scans, wn, raw, saved, shared = load_native(args.native_dir)
    owner_indices = [np.asarray(v) for v in scans.groupby("MediaId", sort=True).indices.values()]
    physical = scans.iloc[[indices[0] for indices in owner_indices]].reset_index(drop=True).copy()
    physical["physical_row"] = np.arange(len(physical))
    physical["n_scans"] = [len(indices) for indices in owner_indices]
    physical["analysis_ids"] = [";".join(scans.iloc[ix].SampleAnalysisId.astype(str)) for ix in owner_indices]
    physical_raw = np.stack([raw[ix].mean(axis=0) for ix in owner_indices])
    blank = physical.loc[physical.ExternalFilterType.eq("FB")].copy().reset_index(drop=True)
    blank_raw = physical_raw[blank.physical_row.to_numpy()]
    assignments = blank_fold_assignments(blank, seed=SEED)
    fold_by_media = assignments.set_index("MediaId").fold.to_dict()
    blank["fold"] = blank.MediaId.map(fold_by_media).astype(int)
    blank["blank_row"] = np.arange(len(blank))
    scans["row"] = np.arange(len(scans))
    scans["blank_fold"] = scans.MediaId.map(fold_by_media)
    scans.to_csv(output / "scan_metadata.csv", index=False)
    physical.to_csv(output / "physical_filter_metadata.csv", index=False)
    blank.to_csv(output / "blank_folds.csv", index=False)
    pd.DataFrame(METHODS).assign(comparison_grid="native 1425–4000 cm^-1").to_csv(
        output / "methods.csv", index=False)

    log(f"Native spline controls: {len(scans)} scans, {len(physical)} physical filters")
    bases, corrections, boundary = segmented_spline_variants(wn, raw)
    boundary = boundary.merge(scans[["row", "SampleAnalysisId", "MediaId", "base_filter_id",
                                    "ExternalFilterType", "lot_group"]], on="row", validate="many_to_one")
    difference = corrections["spline_default"][:, shared]-saved[:, shared]
    validation = pd.DataFrame({"SampleAnalysisId": scans.SampleAnalysisId,
                               "rms_error": np.sqrt(np.mean(difference**2, axis=1)),
                               "max_abs_error": abs(difference).max(axis=1)})
    if validation.max_abs_error.max() > 2e-6:
        raise ValueError("native reference does not reproduce the saved R baseline")
    validation.to_csv(output / "spline_reference_validation.csv", index=False)
    default_boundary = boundary.loc[boundary.method.eq("spline_default")].set_index("row")
    boundary["boundary_change_cm_1"] = boundary.segment2_lower_boundary - boundary.row.map(
        default_boundary.segment2_lower_boundary)
    for method in [m["method"] for m in METHODS if m["family"] == "spline"]:
        delta = corrections[method]-corrections["spline_default"]
        rows = boundary.method.eq(method)
        boundary.loc[rows, "corrected_rms_change"] = np.sqrt(np.mean(delta[:, shared]**2, axis=1))
        boundary.loc[rows, "corrected_1617_change"] = delta[:, np.argmin(abs(wn-1617))]
    boundary.to_csv(output / "boundary_sensitivity.csv", index=False)
    logs = [boundary.assign(scope="native_scan")]

    neutral_logs = {}
    native_convergence = {method: np.ones(len(raw), bool) for method in corrections}
    for max_iter in (50, 500):
        method = "neutral_arpls" if max_iter == 50 else "neutral_arpls_iter500"
        log(f"{method} on all native scans")
        bases[method], corrections[method], neutral_logs[method] = neutral_correction(wn, raw, max_iter=max_iter)
        native_convergence[method] = neutral_logs[method].converged.to_numpy(bool)
        logs.append(neutral_logs[method].assign(scope="native_scan"))

    # Each injected child's original physical parent remains outside its VIBES
    # background bank. Mean raw blank is corrected before paired subtraction.
    case_records, case_spectra = [], []
    parent_case = {}
    additions = {}
    for row in blank.itertuples():
        parent_case[int(row.MediaId)] = len(case_records)
        common = {"parent_media_id": int(row.MediaId), "base_filter_id": row.base_filter_id,
                  "lot_group": row.lot_group, "fold": int(row.fold), "blank_row": int(row.blank_row)}
        case_records.append({**common, "case_id": f"blank:{row.MediaId}",
                             "control": "unspiked", "amplitude": 0.})
        case_spectra.append(blank_raw[row.blank_row])
        for control in INJECTIONS:
            shape = np.exp(-.5*((wn-control["center"])/control["sigma"])**2)
            for amplitude in AMPLITUDES:
                key = (control["control"], amplitude)
                additions[key] = amplitude*shape
                case_records.append({**common, "case_id": f"blank:{row.MediaId}:{key[0]}:{amplitude}",
                                     "control": key[0], "amplitude": amplitude})
                case_spectra.append(blank_raw[row.blank_row]+additions[key])
    cases = pd.DataFrame(case_records)
    cases["case_row"] = np.arange(len(cases))
    case_raw = np.stack(case_spectra)
    cases.to_csv(output / "injection_cases.csv", index=False)
    log(f"Spline/neutral controls on {len(cases)} cases from {len(blank)} actual blank parents")
    _, case_corrected, case_boundary = segmented_spline_variants(wn, case_raw)
    case_convergence = {method: np.ones(len(cases), bool) for method in case_corrected}
    logs.append(case_boundary.assign(scope="blank_case"))
    for max_iter in (50, 500):
        method = "neutral_arpls" if max_iter == 50 else "neutral_arpls_iter500"
        _, case_corrected[method], nlog = neutral_correction(wn, case_raw, max_iter=max_iter)
        case_convergence[method] = nlog.converged.to_numpy(bool)
        logs.append(nlog.assign(scope="blank_case"))

    fold_models = []
    for method, rank in (("vibes_etad_r3", 3), ("vibes_etad_r8", 8)):
        corrections[method] = np.full_like(raw, np.nan)
        bases[method] = np.full_like(raw, np.nan)
        case_corrected[method] = np.full_like(case_raw, np.nan)
        for fold in range(5):
            train = blank.fold.ne(fold).to_numpy()
            take = cases.index[cases.fold.eq(fold)].to_numpy()
            scan_rows = scans.index[scans.ExternalFilterType.eq("FB") & scans.blank_fold.eq(fold)].to_numpy()
            bank = fixed_vibes_background(wn, blank_raw[train],
                                          blank_ids=blank.loc[train, "MediaId"].astype(str), rank=rank)
            values = np.vstack((case_raw[take], raw[scan_rows]))
            ids = cases.loc[take, "case_id"].tolist() + [f"scan:{s}" for s in scans.loc[scan_rows, "SampleAnalysisId"]]
            parents = cases.loc[take, "parent_media_id"].astype(str).tolist() + scans.loc[scan_rows, "MediaId"].astype(str).tolist()
            log(f"{method}, held-out fold {fold+1}/5: {len(values)} evaluations, {train.sum()} training blanks")
            base, corr, fitlog = correct_vibes(wn, values, bank, sample_ids=ids,
                                             parent_ids=parents, workers=args.workers)
            case_corrected[method][take] = corr[:len(take)]
            bases[method][scan_rows] = base[len(take):]
            corrections[method][scan_rows] = corr[len(take):]
            logs.append(fitlog.assign(method=method, scope="vibes_grouped_fold", fold=fold))
            fold_models.append({"scheme": "grouped_5fold", "fold": fold, "method": method,
                                "rank": rank, "training_blanks": int(train.sum()),
                                "training_ids": ";".join(bank.blank_ids),
                                "test_parent_ids": ";".join(blank.loc[~train, "MediaId"].astype(str)),
                                "overlap": 0, "status": "fitted"})
        ambient = scans.index[~scans.ExternalFilterType.eq("FB")].to_numpy()
        full = fixed_vibes_background(wn, blank_raw, blank_ids=blank.MediaId.astype(str), rank=rank)
        log(f"{method}, all {len(blank)} physical blanks → {len(ambient)} ambient scans")
        base, corr, fitlog = correct_vibes(
            wn, raw[ambient], full,
            sample_ids=[f"scan:{s}" for s in scans.loc[ambient, "SampleAnalysisId"]],
            parent_ids=scans.loc[ambient, "MediaId"].astype(str).tolist(), workers=args.workers,
        )
        bases[method][ambient], corrections[method][ambient] = base, corr
        logs.append(fitlog.assign(method=method, scope="vibes_ambient_full_local_bank"))
        native_convergence[method] = np.isfinite(corrections[method][:, shared]).all(axis=1)
        case_convergence[method] = np.isfinite(case_corrected[method][:, shared]).all(axis=1)

    log("Scoring physical held-out blanks and known-lot transfer")
    blank_rows = []
    unspiked = cases.index[cases.control.eq("unspiked")].to_numpy()
    for method, matrix in case_corrected.items():
        results = with_convergence(metrics(wn, case_raw[unspiked], matrix[unspiked], shared),
                                   case_convergence[method][unspiked])
        for i, case_index in enumerate(unspiked):
            case = cases.loc[case_index]
            blank_rows.append({"scheme": "grouped_5fold", "fold": int(case.fold),
                               "method": method, "MediaId": int(case.parent_media_id),
                               "base_filter_id": case.base_filter_id, "lot_group": case.lot_group,
                               "training_blanks": int(blank.fold.ne(case.fold).sum()) if method.startswith("vibes") else 0,
                               "status": "evaluated" if results.iloc[i].success else "finite_not_converged",
                               **results.iloc[i].to_dict()})
    for test_lot, train_lot in (("248", "251"), ("251", "248")):
        train = blank.lot_group.eq(train_lot).to_numpy()
        test = blank.lot_group.eq(test_lot).to_numpy()
        test_rows = np.flatnonzero(test)
        parent_rows = np.array([parent_case[int(mid)] for mid in blank.loc[test, "MediaId"]])
        for method in [m["method"] for m in METHODS]:
            status = "evaluated"
            if method.startswith("vibes"):
                rank = 3 if method.endswith("r3") else 8
                if train.sum() < rank+2:
                    corr = np.full_like(blank_raw[test], np.nan)
                    status = "insufficient_training_blanks"
                else:
                    bank = fixed_vibes_background(wn, blank_raw[train],
                                                  blank_ids=blank.loc[train, "MediaId"].astype(str), rank=rank)
                    log(f"{method}: train known lot {train_lot} ({train.sum()}); test {test_lot} ({test.sum()})")
                    _, corr, fitlog = correct_vibes(
                        wn, blank_raw[test], bank,
                        sample_ids=[f"lot_test:{mid}" for mid in blank.loc[test, "MediaId"]],
                        parent_ids=blank.loc[test, "MediaId"].astype(str).tolist(), workers=args.workers,
                    )
                    logs.append(fitlog.assign(method=method, scope="known_lot_transfer", test_lot=test_lot))
                fold_models.append({"scheme": "known_lot_transfer", "fold": f"train{train_lot}_test{test_lot}",
                                    "method": method, "rank": rank, "training_blanks": int(train.sum()),
                                    "training_ids": ";".join(blank.loc[train, "MediaId"].astype(str)),
                                    "test_parent_ids": ";".join(blank.loc[test, "MediaId"].astype(str)),
                                    "overlap": 0, "status": status})
            else:
                corr = case_corrected[method][parent_rows]
            converged = (np.isfinite(corr[:, shared]).all(axis=1) if method.startswith("vibes")
                         else case_convergence[method][parent_rows])
            results = with_convergence(metrics(wn, blank_raw[test], corr, shared), converged)
            for i, b_row in enumerate(test_rows):
                meta = blank.iloc[b_row]
                blank_rows.append({"scheme": "known_lot_transfer", "fold": f"train{train_lot}_test{test_lot}",
                                   "method": method, "MediaId": int(meta.MediaId),
                                   "base_filter_id": meta.base_filter_id, "lot_group": meta.lot_group,
                                   "training_blanks": int(train.sum()) if method.startswith("vibes") else 0,
                                   "status": ("finite_not_converged" if status == "evaluated" and
                                              not results.iloc[i].success else status),
                                   **results.iloc[i].to_dict()})
    heldout = pd.DataFrame(blank_rows)
    heldout.to_csv(output / "heldout_blank_metrics.csv", index=False)
    pd.DataFrame(fold_models).to_csv(output / "background_banks.csv", index=False)

    log("Scoring paired injection recovery and repeated scans")
    injection_rows, recovered_curves = [], {}
    boundary_index = case_boundary.set_index(["row", "method"])
    bounds = {c["control"]: c["bounds"] for c in INJECTIONS}
    for method, matrix in case_corrected.items():
        for case in cases.loc[~cases.control.eq("unspiked")].itertuples():
            parent_index = parent_case[int(case.parent_media_id)]
            added = additions[(case.control, case.amplitude)]
            result = paired_injection_metrics(wn, matrix[parent_index], matrix[case.case_row],
                                              added, bounds=bounds[case.control])
            result["finite_output"] = result["success"]
            result["fit_converged"] = bool(case_convergence[method][parent_index] and
                                           case_convergence[method][case.case_row])
            result["success"] = bool(result["finite_output"] and result["fit_converged"])
            shift = np.nan
            if (case.case_row, method) in boundary_index.index:
                shift = (boundary_index.loc[(case.case_row, method), "segment2_lower_boundary"] -
                         boundary_index.loc[(parent_index, method), "segment2_lower_boundary"])
            injection_rows.append({"scheme": "grouped_5fold", "method": method,
                                   "MediaId": int(case.parent_media_id), "base_filter_id": case.base_filter_id,
                                   "lot_group": case.lot_group, "fold": int(case.fold),
                                   "control": case.control, "amplitude": case.amplitude,
                                   "anchor_shift_cm_1": shift, **result})
            recovered_curves.setdefault((method, case.control, case.amplitude), []).append(
                matrix[case.case_row]-matrix[parent_index])
    injection_table = pd.DataFrame(injection_rows)
    injection_table.to_csv(output / "injection_recovery.csv", index=False)

    scan_metric_frames, physical_metric_frames = [], []
    physical_corrected = {}
    for method, matrix in corrections.items():
        one_metrics = with_convergence(metrics(wn, raw, matrix, shared), native_convergence[method])
        scan_metric_frames.append(pd.concat([scans[["SampleAnalysisId", "MediaId", "base_filter_id",
                                                   "ExternalFilterType", "lot_group"]].reset_index(drop=True),
                                             one_metrics], axis=1).assign(method=method))
        # Preserve every scan; a failed fit is not silently removed by nanmean.
        physical_corrected[method] = np.stack([matrix[ix].mean(axis=0) for ix in owner_indices])
        physical_metric_frames.append(pd.concat([
            physical[["MediaId", "base_filter_id", "ExternalFilterType", "lot_group", "n_scans"]],
            with_convergence(metrics(wn, physical_raw, physical_corrected[method], shared),
                             [native_convergence[method][ix].all() for ix in owner_indices])], axis=1).assign(method=method))
    scan_metrics = pd.concat(scan_metric_frames, ignore_index=True)
    physical_metrics = pd.concat(physical_metric_frames, ignore_index=True)
    scan_metrics.to_csv(output / "scan_metrics.csv", index=False)
    physical_metrics.to_csv(output / "physical_filter_metrics.csv", index=False)

    repeat_rows = []
    for indices in owner_indices:
        for a, b in itertools.combinations(sorted(indices, key=lambda i: scans.iloc[i].SampleAnalysisId), 2):
            meta = scans.iloc[a]
            for method, matrix in corrections.items():
                one, two = matrix[a, shared], matrix[b, shared]
                finite = bool(np.isfinite(one).all() and np.isfinite(two).all())
                rms = float(np.sqrt(np.mean((one-two)**2))) if finite else np.nan
                mean_strength = float(np.sqrt(np.mean(((one+two)/2)**2))) if finite else np.nan
                raw_pair_rms = float(np.sqrt(np.mean((raw[a, shared]-raw[b, shared])**2)))
                raw_mean_strength = float(np.sqrt(np.mean(((raw[a, shared]+raw[b, shared])/2)**2)))
                converged = bool(native_convergence[method][a] and native_convergence[method][b])
                repeat_rows.append({"MediaId": int(meta.MediaId), "base_filter_id": meta.base_filter_id,
                                    "filter_type": meta.ExternalFilterType, "lot_group": meta.lot_group,
                                    "analysis_a": int(scans.iloc[a].SampleAnalysisId),
                                    "analysis_b": int(scans.iloc[b].SampleAnalysisId), "method": method,
                                    "finite_output": finite, "fit_converged": converged,
                                    "success": finite and converged, "corrected_pair_rms": rms,
                                    "average_signal_rms": mean_strength,
                                    "relative_pair_rms": rms/mean_strength if mean_strength > 0 else np.nan,
                                    "raw_pair_rms": raw_pair_rms,
                                    "raw_average_signal_rms": raw_mean_strength,
                                    "raw_relative_pair_rms": raw_pair_rms/raw_mean_strength if raw_mean_strength > 0 else np.nan,
                                    "delta_1617": float(matrix[a, np.argmin(abs(wn-1617))]-matrix[b, np.argmin(abs(wn-1617))])})
    repeats = pd.DataFrame(repeat_rows)
    repeats.to_csv(output / "replicate_metrics.csv", index=False)
    repeat_physical = repeats.groupby(["MediaId", "base_filter_id", "filter_type", "lot_group", "method"],
                                      dropna=False).agg(
        n_pairs=("analysis_a", "size"), successful_pairs=("success", "sum"),
        corrected_pair_rms=("corrected_pair_rms", "median"),
        raw_pair_rms=("raw_pair_rms", "median"), relative_pair_rms=("relative_pair_rms", "median"),
        raw_relative_pair_rms=("raw_relative_pair_rms", "median"),
        raw_average_signal_rms=("raw_average_signal_rms", "median"),
    ).reset_index()
    repeat_physical.to_csv(output / "replicate_filter_metrics.csv", index=False)

    curve_table, picks = spectral_rows(wn, {"raw": physical_raw, **physical_corrected}, physical, shared)
    curve_table.to_csv(output / "spectral_curves.csv", index=False)
    curve_rows = []
    for (method, control, amplitude), rows in recovered_curves.items():
        array = np.stack(rows)
        valid = np.isfinite(array[:, shared]).all(axis=1)
        if not valid.any():
            continue
        q25, median, q75 = np.quantile(array[valid][:, picks], [.25, .5, .75], axis=0)
        expected = additions[(control, amplitude)]
        curve_rows.extend({"method": method, "control": control, "amplitude": amplitude,
                           "wavenumber": wn[j], "known_addition": expected[j],
                           "median_recovered": median[k], "q25_recovered": q25[k],
                           "q75_recovered": q75[k], "n_physical_parents": int(valid.sum())}
                          for k, j in enumerate(picks))
    pd.DataFrame(curve_rows).to_csv(output / "injection_spectral_curves.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(output / "fit_diagnostics.csv", index=False)

    log("Saving signed native arrays and compact comparison summaries")
    arrays = {"wavenumbers": wn, "analysis_ids": scans.SampleAnalysisId.to_numpy(),
              "media_ids": scans.MediaId.to_numpy(), "filter_ids": scans.base_filter_id.to_numpy(str),
              "raw": raw, "saved_corrected": saved,
              "physical_media_ids": physical.MediaId.to_numpy(),
              "physical_filter_ids": physical.base_filter_id.to_numpy(str),
              "physical_raw": physical_raw, "blank_bank_mean": blank_raw.mean(axis=0),
              "blank_bank_std": blank_raw.std(axis=0, ddof=1), "shared_mask": shared}
    arrays.update({f"baseline__{key}": value for key, value in bases.items()})
    arrays.update({f"corrected__{key}": value for key, value in corrections.items()})
    arrays.update({f"physical_corrected__{key}": value for key, value in physical_corrected.items()})
    np.savez_compressed(output / "native_method_arrays.npz", **arrays)
    blank_summary = heldout.groupby(["scheme", "method", "lot_group"], dropna=False).agg(
        n=("MediaId", "size"), successful=("success", "sum"), median_rms=("corrected_rms", "median"),
        finite_outputs=("finite_output", "sum"),
        q25_rms=("corrected_rms", lambda s: s.quantile(.25)),
        q75_rms=("corrected_rms", lambda s: s.quantile(.75)),
        median_1617=("corrected_1617", "median"), median_negative_fraction=("negative_fraction", "median"),
    ).reset_index()
    blank_summary.to_csv(output / "blank_summary.csv", index=False)
    injection_summary = injection_table.groupby(["method", "control", "amplitude"], dropna=False).agg(
        n_parents=("MediaId", "size"), successful=("success", "sum"),
        finite_outputs=("finite_output", "sum"),
        median_recovery=("area_recovery_fraction", "median"),
        q25_recovery=("area_recovery_fraction", lambda s: s.quantile(.25)),
        q75_recovery=("area_recovery_fraction", lambda s: s.quantile(.75)),
        min_recovery=("area_recovery_fraction", "min"), max_recovery=("area_recovery_fraction", "max"),
        median_relative_rms_error=("relative_rms_recovery_error", "median"),
    ).reset_index()
    converged_recovery = injection_table.loc[injection_table.success].groupby(
        ["method", "control", "amplitude"]).area_recovery_fraction.median().rename("median_recovery_converged")
    injection_summary = injection_summary.merge(converged_recovery, on=["method", "control", "amplitude"], how="left")
    injection_summary.to_csv(output / "injection_summary.csv", index=False)
    summary = {
        "scope": "Existing-data baseline reliability controls; no new sampling or EC product changes",
        "seed": SEED, "methods": list(METHODS), "native_scans": len(scans),
        "physical_filters": len(physical), "physical_blanks": len(blank),
        "blank_lots": blank.lot_group.value_counts().to_dict(),
        "repeated_physical_filters": int((physical.n_scans > 1).sum()),
        "native_grid_points": len(wn), "comparison_grid_points": int(shared.sum()),
        "native_spacing_cm_1": float(abs(np.median(np.diff(wn)))),
        "heldout_blank_folds": 5, "injection_parents": len(blank),
        "injections": list(INJECTIONS), "amplitudes": list(AMPLITUDES),
        "numerical_spline_reference_max_abs_error": validation.max_abs_error.max(),
        "numerical_spline_reference_rms_error": np.sqrt(np.mean(difference**2)),
        "method_scan_success": scan_metrics.groupby("method").success.agg(["count", "sum"]).to_dict("index"),
        "neutral_native_nonconverged": {method: int((~log.converged).sum()) for method, log in neutral_logs.items()},
        "blank_summary": blank_summary.to_dict("records"),
        "injection_summary": injection_summary.to_dict("records"),
        "elapsed_seconds": time.perf_counter()-start,
        "limitations": [
            "No independent ambient EC truth or optical outcome enters these controls.",
            "All actual injection parents are ETAD field blanks; added Gaussian shapes are defined controls, not identified ambient species.",
            "The local VIBES banks contain at most 37 physical blanks, with fixed ranks 3 and 8; they are not the completed global rank-70 product.",
            "Five-fold blank evaluation holds out every scan and injected child of a physical parent; unknown-lot blanks are omitted from known-lot transfer.",
            "There are only five known lot-248 blanks. Rank 8 cannot train on that lot alone and is recorded as unavailable for that transfer.",
            "Lower blank residue and higher repeatability can coexist with removal of real smooth signal; assess them jointly with injection recovery.",
            "Native methods have different fit domains; numerical comparison is restricted to shared 1425–4000 cm^-1.",
            "Neutral is the repository's anchor-free robustness check, not a physically established true baseline.",
            "Ambient repeats are never silently dropped. Failure in one scan propagates to its physical-filter average.",
            "Neutral finite-but-unconverged fits remain visible with separate flags; success requires convergence. All-finite summary statistics are accompanied by convergence counts and a converged-only injection median.",
            "Held-out blank controls correct the mean raw physical blank, as do its injected children; ambient physical-filter traces average corrected individual scans.",
            "Aggregate spectral curves include all finite estimates; inspect per-filter convergence flags before interpreting neutral curves.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(safe_json(summary), indent=2, allow_nan=False)+"\n")
    receipt = {
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ["numpy", "scipy", "pandas", "scikit-learn", "pybaselines", "cvxpy", "clarabel"]},
        "inputs": [{"name": name, "bytes": (args.native_dir/name).stat().st_size,
                    "sha256": sha(args.native_dir/name)} for name in SOURCE_NAMES],
        "code": {str(path.relative_to(ROOT)): sha(path) for path in (
            Path(__file__), ROOT / "research/ftir_hips_chem/scripts/native_baseline_controls.py",
            ROOT / "research/ftir_ec_phase3/scripts/airspec_baseline.py",
            ROOT / "research/ftir_ec_phase3/scripts/neutral_baseline.py",
            ROOT / "research/ftir_hips_chem/scripts/vibes_baseline.py",
            ROOT / "research/ftir_hips_chem/scripts/portable_baseline_screen.py",
            ROOT / "research/ftir_hips_chem/workflows/run_native_addis_baseline_audit.py",
            ROOT / "research/ftir_hips_chem/vendor/pyvibes/SOURCE.json",
            *sorted((ROOT / "research/ftir_hips_chem/vendor/pyvibes/src/vibes").rglob("*.py")),
        )},
        "seed": SEED, "workers": args.workers,
        "output_sha256": {path.name: sha(path) for path in sorted(output.iterdir())
                          if path.is_file() and path.name not in ("receipt.json", "execution_status.json")},
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    (output / "execution_status.json").write_text(json.dumps({
        "state": "completed", "elapsed_seconds": time.perf_counter()-start,
    }, indent=2)+"\n")
    print(json.dumps({"state": "completed", "scans": len(scans), "blanks": len(blank),
                      "elapsed_seconds": time.perf_counter()-start}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/ftir_hips_chem/output/tables/baseline_existing_data_20261103")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    with threadpool_limits(limits=1):
        run(args)
