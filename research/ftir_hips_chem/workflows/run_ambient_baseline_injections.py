"""Bounded input-selected ambient injection check after native blank controls.

The actual blank parents all place the default spline boundary near 1521 cm^-1.
This separate stress test selects at most five existing PM2.5 physical filters
in each of three boundary strata, spread over raw shared-window RMS. A fixed
0.01 AU Gaussian is added at 1617 (sigma 25) or 2800 (sigma 600). No HIPS/EC
outcome or prior disagreement determines selection. This is a selected stress
test, not a representative prevalence estimate or chemical/EC validation.

Run after run_existing_data_baseline_controls.py completes::

    uv run --extra explorer --extra vibes python \
      research/ftir_hips_chem/workflows/run_ambient_baseline_injections.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_baseline_controls import (  # noqa: E402
    METHODS, INJECTIONS, correct_vibes, fixed_vibes_background,
    neutral_correction, paired_injection_metrics, segmented_spline_variants,
)
from run_existing_data_baseline_controls import safe_json, sha  # noqa: E402

AMPLITUDE = .01
SCHEME = "ambient_input_stratified"


def run(args):
    start = time.perf_counter()
    folder = args.folder
    receipt_path = folder / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    needed = ("native_method_arrays.npz", "physical_filter_metadata.csv")
    for name in needed:
        if sha(folder/name) != receipt["output_sha256"][name]:
            raise ValueError(f"completed native input changed: {name}")
    status = json.loads((folder/"execution_status.json").read_text())
    if status["state"] != "completed":
        raise ValueError("finish the primary native workflow before this supplement")
    meta = pd.read_csv(folder / "physical_filter_metadata.csv", dtype={"lot_group": str})
    with np.load(folder / "native_method_arrays.npz", allow_pickle=False) as z:
        wn, physical_raw, shared = z["wavenumbers"], z["physical_raw"], z["shared_mask"]
        if not np.array_equal(z["physical_media_ids"], meta.MediaId):
            raise ValueError("native physical metadata order differs from arrays")
    pm = meta.loc[meta.ExternalFilterType.eq("PM2.5")].copy()
    _, _, boundaries = segmented_spline_variants(wn, physical_raw[pm.physical_row])
    default = boundaries.loc[boundaries.method.eq("spline_default")].sort_values("row")
    pm["default_boundary_cm_1"] = default.segment2_lower_boundary.to_numpy()
    pm["raw_shared_rms"] = np.sqrt(np.mean(physical_raw[pm.physical_row][:, shared]**2, axis=1))
    pm["boundary_stratum"] = np.select(
        [pm.default_boundary_cm_1 <= 1521., pm.default_boundary_cm_1 <= 1560.],
        ["low_search_edge", "middle_1521_1560"], default="high_above_1560")
    selected_rows = []
    for _, group in pm.groupby("boundary_stratum", sort=True):
        ordered = group.sort_values(["raw_shared_rms", "MediaId"])
        # Endpoint and interior order statistics, with deterministic tie breaks.
        positions = np.rint(np.linspace(0, len(ordered)-1, min(5, len(ordered)))).astype(int)
        selected_rows.extend(ordered.iloc[np.unique(positions)].physical_row.tolist())
    pm["selected"] = pm.physical_row.isin(selected_rows)
    selected = pm.loc[pm.selected].sort_values(["boundary_stratum", "raw_shared_rms", "MediaId"])
    pm.to_csv(folder / "ambient_injection_selection.csv", index=False)
    records, values, additions, parent_case = [], [], {}, {}
    for item in selected.itertuples():
        parent_case[int(item.MediaId)] = len(records)
        common = {"MediaId": int(item.MediaId), "base_filter_id": item.base_filter_id,
                  "lot_group": item.lot_group, "boundary_stratum": item.boundary_stratum,
                  "default_boundary_cm_1": item.default_boundary_cm_1,
                  "raw_shared_rms": item.raw_shared_rms}
        records.append({**common, "case_id": f"ambient:{item.MediaId}", "control": "unspiked", "amplitude": 0.})
        values.append(physical_raw[item.physical_row])
        for spec in INJECTIONS:
            addition = AMPLITUDE*np.exp(-.5*((wn-spec["center"])/spec["sigma"])**2)
            additions[spec["control"]] = addition
            records.append({**common, "case_id": f"ambient:{item.MediaId}:{spec['control']}",
                            "control": spec["control"], "amplitude": AMPLITUDE})
            values.append(physical_raw[item.physical_row]+addition)
    cases = pd.DataFrame(records)
    cases["case_row"] = np.arange(len(cases))
    raw = np.stack(values)
    print(f"Ambient stress control: {len(selected)} physical parents; {len(cases)} cases", flush=True)
    _, corrected, boundaries = segmented_spline_variants(wn, raw)
    logs = [boundaries]
    converged = {key: np.ones(len(raw), bool) for key in corrected}
    for iterations in (50, 500):
        method = "neutral_arpls" if iterations == 50 else "neutral_arpls_iter500"
        _, corrected[method], fit_log = neutral_correction(wn, raw, max_iter=iterations)
        converged[method] = fit_log.converged.to_numpy(bool)
        logs.append(fit_log)
    blank_rows = meta.ExternalFilterType.eq("FB")
    for method, rank in (("vibes_etad_r3", 3), ("vibes_etad_r8", 8)):
        bank = fixed_vibes_background(wn, physical_raw[blank_rows],
                                      blank_ids=meta.loc[blank_rows, "MediaId"].astype(str), rank=rank)
        print(f"{method}: {len(raw)} ambient-parent cases", flush=True)
        _, corrected[method], fit_log = correct_vibes(
            wn, raw, bank, sample_ids=cases.case_id.tolist(),
            parent_ids=cases.MediaId.astype(str).tolist(), workers=args.workers,
        )
        converged[method] = fit_log.success.to_numpy(bool)
        logs.append(fit_log.assign(method=method))
    boundary_index = boundaries.set_index(["row", "method"])
    bounds = {item["control"]: item["bounds"] for item in INJECTIONS}
    metrics, curve_groups = [], {}
    for method, matrix in corrected.items():
        for case in cases.loc[~cases.control.eq("unspiked")].itertuples():
            parent = parent_case[int(case.MediaId)]
            result = paired_injection_metrics(wn, matrix[parent], matrix[case.case_row],
                                              additions[case.control], bounds=bounds[case.control])
            result["finite_output"] = result["success"]
            result["fit_converged"] = bool(converged[method][parent] and converged[method][case.case_row])
            result["success"] = bool(result["finite_output"] and result["fit_converged"])
            before, after = np.nan, np.nan
            if (case.case_row, method) in boundary_index.index:
                before = float(boundary_index.loc[(parent, method), "segment2_lower_boundary"])
                after = float(boundary_index.loc[(case.case_row, method), "segment2_lower_boundary"])
            metrics.append({"scheme": SCHEME, "method": method,
                            **{key: getattr(case, key) for key in (
                                "MediaId", "base_filter_id", "lot_group", "boundary_stratum",
                                "default_boundary_cm_1", "raw_shared_rms", "control", "amplitude")},
                            "anchor_before_cm_1": before, "anchor_after_cm_1": after,
                            "anchor_shift_cm_1": after-before, **result})
            curve_groups.setdefault((method, case.control, case.boundary_stratum), []).append(
                matrix[case.case_row]-matrix[parent])
    results = pd.DataFrame(metrics)
    results.to_csv(folder / "ambient_injection_recovery.csv", index=False)
    summary = results.groupby(["method", "control", "boundary_stratum"], dropna=False).agg(
        n_parents=("MediaId", "size"), successful=("success", "sum"), finite_outputs=("finite_output", "sum"),
        median_recovery=("area_recovery_fraction", "median"),
        q25_recovery=("area_recovery_fraction", lambda s: s.quantile(.25)),
        q75_recovery=("area_recovery_fraction", lambda s: s.quantile(.75)),
        min_recovery=("area_recovery_fraction", "min"), max_recovery=("area_recovery_fraction", "max"),
        median_anchor_shift=("anchor_shift_cm_1", "median"),
    ).reset_index().assign(scheme=SCHEME, amplitude=AMPLITUDE)
    valid_median = results.loc[results.success].groupby(
        ["method", "control", "boundary_stratum"]).area_recovery_fraction.median().rename("median_recovery_converged")
    summary = summary.merge(valid_median, on=["method", "control", "boundary_stratum"], how="left")
    summary.to_csv(folder / "ambient_injection_summary.csv", index=False)
    picks = np.unique(np.concatenate((np.flatnonzero(shared)[::8],
                                      np.flatnonzero((wn >= 1500) & (wn <= 1800)))))
    curve_rows = []
    for (method, control, stratum), rows in curve_groups.items():
        array = np.stack(rows)
        finite = np.isfinite(array[:, shared]).all(axis=1)
        if not finite.any():
            continue
        q25, med, q75 = np.quantile(array[finite][:, picks], [.25, .5, .75], axis=0)
        curve_rows.extend({"scheme": SCHEME, "method": method, "control": control,
                           "amplitude": AMPLITUDE, "boundary_stratum": stratum,
                           "wavenumber": wn[j], "known_addition": additions[control][j],
                           "median_recovered": med[k], "q25_recovered": q25[k],
                           "q75_recovered": q75[k], "n_physical_parents": int(finite.sum())}
                          for k, j in enumerate(picks))
    pd.DataFrame(curve_rows).to_csv(folder / "ambient_injection_spectral_curves.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(folder / "ambient_injection_fit_diagnostics.csv", index=False)
    supplement = {
        "scope": "Input-selected ambient-shape stress test; distinct from held-out blank controls",
        "primary_receipt_sha256": sha(receipt_path),
        "input_sha256": {name: sha(folder/name) for name in needed},
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            Path(__file__), ROOT / "research/ftir_hips_chem/scripts/native_baseline_controls.py",
            ROOT / "research/ftir_hips_chem/workflows/run_existing_data_baseline_controls.py")},
        "scheme": SCHEME, "n_physical_parents": len(selected),
        "selection": "Within default physical-mean spline-boundary <=1521 / (1521,1560] / >1560 strata, select up to five equally spaced raw-RMS order statistics, tie-break by MediaId.",
        "selected_filter_ids": selected.base_filter_id.tolist(), "amplitude": AMPLITUDE,
        "controls": list(INJECTIONS), "elapsed_seconds": time.perf_counter()-start,
        "limitations": ["Selected input extremes are not a representative ambient cohort.",
                        "Known Gaussian additions test preservation; their shapes do not identify ambient compounds or establish EC accuracy.",
                        "No EC/HIPS outcome enters selection or fitting.",
                        "All-finite results retain failed-convergence estimates with explicit flags and converged-only summary medians."],
        "output_sha256": {path.name: sha(path) for path in sorted(folder.glob("ambient_injection_*.csv"))},
    }
    (folder / "ambient_injection_receipt.json").write_text(json.dumps(safe_json(supplement), indent=2, allow_nan=False)+"\n")
    print(json.dumps({"state": "completed", "parents": len(selected),
                      "cases": len(cases), "elapsed_seconds": time.perf_counter()-start}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path,
                        default=ROOT / "research/ftir_hips_chem/output/tables/baseline_existing_data_20261103")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)
