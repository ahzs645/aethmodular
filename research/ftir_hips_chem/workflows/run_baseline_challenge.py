"""Challenge baseline assumptions using existing, signed native Addis spectra.

No optical target or EC product enters this experiment. The data folder is the
same four-file native export used by run_native_addis_baseline_audit.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(ROOT / "research/ftir_ec_phase3/scripts"))

from airspec_baseline import airspec_baseline_matrix
from baseline_challenge import (
    PROTECTED_WINDOWS, fit_additive_blank_model, independent_blank_bank,
    injected_shapes, recovery_metrics, whittaker_arpls,
)
from run_native_addis_baseline_audit import SOURCE_NAMES, load_native, safe_json


METHODS = (
    "spline_df6_4", "arpls_1e4", "arpls_1e6", "arpls_1e8",
    "blank_mean", "blank_pc3_full", "blank_pc3_protected",
)
BACKGROUND_METHODS = METHODS[-3:]
AMPLITUDES = (0.002, 0.01)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate_blanks(scans, raw):
    rows, values = [], []
    for fid, indices in scans.groupby("base_filter_id", sort=True).indices.items():
        group = scans.iloc[indices]
        if not group.ExternalFilterType.eq("FB").all():
            continue
        if group.lot_group.nunique(dropna=False) != 1:
            raise ValueError("physical blank has conflicting lots")
        rows.append({"id": fid, "lot": group.lot_group.iloc[0], "n_scans": len(indices)})
        values.append(raw[indices].mean(axis=0))
    return pd.DataFrame(rows), np.asarray(values)


def background_correction(method, wn, values, bank, bank_ids, sample_ids):
    if set(map(str, bank_ids)) & set(map(str, sample_ids)):
        raise ValueError("background fitting and evaluation identities overlap")
    if method == "blank_mean":
        return values - bank.mean(axis=0), {"rank": 0, "condition_number": None, "fit_channels": 0}
    model = fit_additive_blank_model(
        wn, bank, blank_ids=bank_ids, rank=3, protected=method.endswith("protected"),
    )
    return model.correct(values, sample_ids=sample_ids), {
        "rank": model.rank, "condition_number": model.condition_number,
        "fit_channels": int(model.fit_mask.sum()),
    }


def challenge_trial_metrics(original, injected, signal, *, original_converged, injected_converged):
    """Retain a failed solver trial without treating its values as successful.

    A nonfinite solve has no valid recovery metrics. Finite iteration-limit
    outputs may still be inspected, but are excluded from converged summaries.
    """
    finite = np.isfinite(original).all() and np.isfinite(injected).all()
    converged = bool(finite and original_converged and injected_converged)
    metrics = (recovery_metrics(original, injected, signal) if finite else {
        "signal_gain": np.nan, "relative_recovery_error": np.nan,
        "relative_shape_distortion": np.nan,
    })
    return {
        "converged": converged,
        "metric_status": "ok" if converged else ("solver_not_converged" if finite else "nonfinite_correction"),
        **metrics,
    }


def mark_common_converged(rows, *, group_keys=None):
    """Identify complete comparison groups successful for every eligible method."""
    keys = group_keys or ["holdout", "id", "shape", "amplitude"]
    result = rows.copy()
    result["common_converged"] = False
    for _, group in result.groupby(keys, sort=False, dropna=False):
        holdout = group.holdout.iloc[0]
        if holdout not in ("filter", "known_lot"):
            raise ValueError(f"unknown comparison holdout: {holdout}")
        expected = METHODS if holdout == "filter" else BACKGROUND_METHODS
        complete = len(group) == len(expected) and set(group.method) == set(expected)
        result.loc[group.index, "common_converged"] = bool(complete and group.converged.all())
    return result


def challenge_statistics(rows):
    rows = rows.copy()
    keys = ["holdout", "method", "shape", "amplitude"]
    summaries = []
    for key, frame in rows.groupby(keys, sort=True):
        item = dict(zip(keys, key))
        ok = frame.loc[frame.converged]
        item.update(trials=len(frame), converged_trials=len(ok), failed_trials=int((~frame.converged).sum()))
        for metric in ("signal_gain", "relative_recovery_error", "relative_shape_distortion"):
            item[metric + "_median"] = float(ok[metric].median()) if len(ok) else np.nan
            item[metric + "_q10"] = float(ok[metric].quantile(.1)) if len(ok) else np.nan
            item[metric + "_q90"] = float(ok[metric].quantile(.9)) if len(ok) else np.nan
        summaries.append(item)
    return pd.DataFrame(summaries)


def make_figure(challenges, blank_metrics, output):
    from plotting import apply_default_style
    import matplotlib.pyplot as plt

    apply_default_style()
    selected = challenges.loc[challenges.holdout.eq("filter") & challenges.common_converged]
    shape_order = list(injected_shapes(np.linspace(4000, 1425, 20)))
    gains = selected.pivot_table(index="method", columns="shape", values="signal_gain", aggfunc="median")
    gains = gains.reindex(index=METHODS, columns=shape_order)
    labels = ["Spline baseline", "arPLS: 10⁴", "arPLS: 10⁶", "arPLS: 10⁸",
              "Blank mean", "Blank PCs + trend", "Blank PCs + trend; protected bands"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [1.8, 1]})
    im = ax.imshow(gains.to_numpy(), aspect="auto", cmap="RdYlBu", vmin=0, vmax=1.2)
    ax.set_yticks(range(len(METHODS)), labels)
    ax.set_xticks(range(len(shape_order)), ["1617\nσ=8", "1617\nσ=40", "1617\nσ=120",
                                           "C–H\nσ=18", "O–H\nσ=150", "Broad\nlinear"])
    for i in range(len(METHODS)):
        for j in range(len(shape_order)):
            value = gains.iloc[i, j]
            label = f"{int(round(100*value))}%" if np.isfinite(value) else "—"
            ax.text(j, i, label, ha="center", va="center", fontsize=10)
    shared_groups = selected[["id", "shape", "amplitude"]].drop_duplicates()
    ax.set_title("Known signal recovered: common converged trials\n"
                 f"{len(shared_groups)} challenges per method; {shared_groups.id.nunique()} physical blanks")
    fig.colorbar(im, ax=ax, shrink=.75, label="Signal gain (1 = preserved)")
    residual = blank_metrics.loc[blank_metrics.holdout.eq("filter") & blank_metrics.common_converged]
    grouped = [residual.loc[residual.method.eq(method), "corrected_rms"].to_numpy() for method in METHODS]
    bx.boxplot(grouped, vert=False, tick_labels=labels, showfliers=False)
    bx.set_xscale("log")
    bx.set_xlabel("Held-out blank RMS absorbance")
    bx.set_title(f"Blank removal: {residual.id.nunique()} shared blanks\nSeparate common-convergence cohort")
    bx.invert_yaxis()
    fig.suptitle("Existing-data baseline challenge — diagnostic, not EC validation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "baseline_challenge.png", bbox_inches="tight")
    fig.savefig(output / "baseline_challenge.svg", bbox_inches="tight")
    plt.close(fig)


def run(native_dir, output, jobs):
    output.mkdir(parents=True, exist_ok=True)
    # Snapshot before fitting: edits made during a long run must not be
    # retroactively attributed to the results of already-loaded code.
    input_receipts = {
        name: {"sha256": digest(native_dir/name), "bytes": (native_dir/name).stat().st_size}
        for name in SOURCE_NAMES
    }
    code_receipts = {
        str(p.relative_to(ROOT)): digest(p)
        for p in [Path(__file__),
                  ROOT / "research/ftir_hips_chem/scripts/baseline_challenge.py",
                  ROOT / "research/ftir_ec_phase3/scripts/airspec_baseline.py",
                  ROOT / "research/ftir_hips_chem/workflows/run_native_addis_baseline_audit.py"]
    }
    scans, full_wn, raw_full, saved_full, region = load_native(native_dir)
    wn, raw, saved = full_wn[region], raw_full[:, region], saved_full[:, region]
    if not np.allclose(np.diff(wn), np.diff(wn)[0]):
        raise ValueError("Whittaker challenge requires a uniform native grid")
    blank_meta, blanks = aggregate_blanks(scans, raw)
    ids, lots = blank_meta.id.to_numpy(), blank_meta.lot.to_numpy()
    shapes = injected_shapes(wn)
    conditions = [{"shape": "none", "amplitude": 0.0, "signal": np.zeros(len(wn))}]
    conditions.extend({"shape": name, "amplitude": amplitude, "signal": amplitude * shape}
                      for name, shape in shapes.items() for amplitude in AMPLITUDES)
    inputs = np.asarray([blank + spec["signal"] for blank in blanks for spec in conditions])
    k = len(conditions)
    print(f"Loaded {len(scans)} scans; {len(blanks)} independent blanks; {len(inputs)} constructed inputs", flush=True)
    corrections, convergence = {}, {}
    _, corrections[METHODS[0]] = airspec_baseline_matrix(wn, inputs, df1=6, df2=4, n_jobs=jobs, chunksize=13)
    convergence[METHODS[0]] = np.isfinite(corrections[METHODS[0]]).all(axis=1)
    solver_rows = []
    print("Completed native spline challenges", flush=True)
    for method, lam in zip(METHODS[1:4], [1e4, 1e6, 1e8]):
        values, flags = [], []
        for i, spectrum in enumerate(inputs):
            corrected, status = whittaker_arpls(spectrum, lam=lam)
            values.append(corrected)
            flags.append(status["converged"])
            spec = conditions[i % k]
            solver_rows.append({"method": method, "id": ids[i // k], "shape": spec["shape"],
                                "amplitude": spec["amplitude"], **status})
        corrections[method], convergence[method] = np.asarray(values), np.asarray(flags)
        print(f"Completed {method}: {sum(flags)}/{len(flags)} converged", flush=True)
    trial_rows, residual_rows, bank_rows, model_rows = [], [], [], []
    channel = int(np.argmin(np.abs(wn - 1617)))
    for holdout in ("filter", "known_lot"):
        for b, fid in enumerate(ids):
            if holdout == "known_lot" and lots[b] == "unknown":
                continue
            bank, bank_ids, bank_lots = independent_blank_bank(blanks, ids, lots, fid, holdout=holdout)
            bank_rows.append({"holdout": holdout, "id": fid, "lot": lots[b],
                              "training_blanks": len(bank_ids), "training_ids": "|".join(bank_ids),
                              "training_lots": "|".join(sorted(set(bank_lots)))})
            trial_inputs = inputs[b*k:(b+1)*k]
            methods = METHODS if holdout == "filter" else BACKGROUND_METHODS
            for method in methods:
                if method in BACKGROUND_METHODS:
                    corrected, fit_info = background_correction(method, wn, trial_inputs, bank, bank_ids, [fid] * k)
                    flags = np.isfinite(corrected).all(axis=1)
                    model_rows.append({
                        "context": "heldout_blank_injections", "holdout": holdout,
                        "method": method, "id": fid, "scan_id": None, "lot": lots[b],
                        "training_blanks": len(bank_ids), "training_ids": "|".join(bank_ids),
                        "training_lots": "|".join(sorted(set(bank_lots))), **fit_info,
                    })
                else:
                    corrected = corrections[method][b*k:(b+1)*k]
                    flags = convergence[method][b*k:(b+1)*k]
                residual_rows.append({"holdout": holdout, "method": method, "id": fid, "lot": lots[b],
                                      "converged": flags[0], "raw_rms": float(np.sqrt(np.mean(blanks[b]**2))),
                                      "corrected_rms": float(np.sqrt(np.mean(corrected[0]**2))),
                                      "corrected_1617": float(corrected[0, channel])})
                for j, spec in enumerate(conditions[1:], 1):
                    trial_rows.append({"holdout": holdout, "method": method, "id": fid, "lot": lots[b],
                                       "shape": spec["shape"], "amplitude": spec["amplitude"],
                                       **challenge_trial_metrics(
                                           corrected[0], corrected[j], spec["signal"],
                                           original_converged=flags[0], injected_converged=flags[j])})
        print(f"Completed {holdout} background challenges", flush=True)
    trials = mark_common_converged(pd.DataFrame(trial_rows))
    residuals = mark_common_converged(pd.DataFrame(residual_rows), group_keys=["holdout", "id"])
    summary = challenge_statistics(trials)
    common_summary = challenge_statistics(trials.loc[trials.common_converged])
    # Native-repeat amplitudes are a separate outcome from blank/injection tests.
    repeated = scans.base_filter_id.map(scans.base_filter_id.value_counts()).gt(1).to_numpy()
    native_rows = []
    repeat_indices = np.flatnonzero(repeated)
    native_corrections = {METHODS[0]: saved[repeat_indices]}
    native_flags = {METHODS[0]: np.ones(len(repeat_indices), dtype=bool)}
    for method, lam in zip(METHODS[1:4], [1e4, 1e6, 1e8]):
        fits = [whittaker_arpls(raw[i], lam=lam) for i in repeat_indices]
        native_corrections[method] = np.asarray([x[0] for x in fits])
        native_flags[method] = np.asarray([x[1]["converged"] for x in fits])
    for method in BACKGROUND_METHODS:
        values = []
        for i in repeat_indices:
            row = scans.iloc[i]
            if row.ExternalFilterType == "FB":
                bank, bank_ids, bank_lots = independent_blank_bank(blanks, ids, lots, row.base_filter_id)
                bank_rule = "filter"
            else:
                same = lots == row.lot_group
                use_same = row.lot_group != "unknown" and same.sum() >= 4
                bank, bank_ids, bank_lots = (blanks[same], ids[same], lots[same]) if use_same else (blanks, ids, lots)
                bank_rule = "matched_lot" if use_same else "all_available"
            result, fit_info = background_correction(method, wn, raw[[i]], bank, bank_ids, [row.base_filter_id])
            model_rows.append({
                "context": "native_repeat", "holdout": bank_rule, "method": method,
                "id": row.base_filter_id, "scan_id": int(row.SampleAnalysisId), "lot": row.lot_group,
                "training_blanks": len(bank_ids), "training_ids": "|".join(bank_ids),
                "training_lots": "|".join(sorted(set(bank_lots))), **fit_info,
            })
            values.append(result[0])
        native_corrections[method] = np.asarray(values)
        native_flags[method] = np.isfinite(native_corrections[method]).all(axis=1)
    for position, i in enumerate(repeat_indices):
        row = scans.iloc[i]
        for method in METHODS:
            spectrum = native_corrections[method][position]
            native_rows.append({"id": row.base_filter_id, "scan_id": int(row.SampleAnalysisId),
                               "type": row.ExternalFilterType, "lot": row.lot_group, "method": method,
                               "converged": native_flags[method][position],
                               "corrected_rms": float(np.sqrt(np.mean(spectrum**2))),
                               "corrected_1617": float(spectrum[channel])})
    native = pd.DataFrame(native_rows)
    pair_rows = []
    import itertools
    for fid, indices in scans.iloc[repeat_indices].groupby("base_filter_id", sort=True).indices.items():
        for a, b in itertools.combinations(indices, 2):
            first, second = scans.iloc[repeat_indices[a]], scans.iloc[repeat_indices[b]]
            for method in METHODS:
                ca, cb = native_corrections[method][a], native_corrections[method][b]
                scale = np.sqrt(np.mean(((ca+cb)/2)**2))
                pair_rows.append({"id": fid, "type": first.ExternalFilterType,
                                  "scan_a": int(first.SampleAnalysisId), "scan_b": int(second.SampleAnalysisId),
                                  "method": method,
                                  "converged": bool(native_flags[method][a] and native_flags[method][b]),
                                  "absolute_difference_rms": float(np.sqrt(np.mean((ca-cb)**2))),
                                  "relative_difference_rms": float(np.sqrt(np.mean((ca-cb)**2))/scale) if scale > 0 else np.nan})
    pairs = pd.DataFrame(pair_rows)
    tables = {"injection_trials.csv": trials, "injection_summary.csv": summary,
              "common_converged_injection_summary.csv": common_summary,
              "heldout_blank_metrics.csv": residuals, "background_training_ledger.csv": pd.DataFrame(bank_rows),
              "background_model_ledger.csv": pd.DataFrame(model_rows),
              "arpls_solver_status.csv": pd.DataFrame(solver_rows), "repeat_scan_metrics.csv": native,
              "repeat_pair_metrics.csv": pairs, "physical_blanks.csv": blank_meta}
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    manifest = {
        "claim": "Signal preservation and blank/repeat diagnostics only; no EC refit or accuracy claim",
        "native_scans": len(scans), "physical_blanks": len(blanks), "blank_scans": int(blank_meta.n_scans.sum()),
        "known_lot_blanks": int(blank_meta.lot.ne("unknown").sum()),
        "grid_channels": len(wn), "grid_spacing_cm1": float(abs(np.diff(wn)[0])),
        "methods": METHODS, "amplitudes_absorbance": AMPLITUDES, "gaussian_widths": "sigma, not FWHM",
        "protected_windows_cm1": PROTECTED_WINDOWS,
        "injection_trials": len(trials), "nonconverged_injection_trials": int((~trials.converged).sum()),
        "common_converged_injection_trials": int(trials.common_converged.sum()),
        "common_converged_filter_challenges_per_method": int(
            trials.loc[trials.holdout.eq("filter") & trials.common_converged,
                       ["id", "shape", "amplitude"]].drop_duplicates().shape[0]),
        "common_convergence_rule": "All seven methods for filter holdout; all three background methods for known-lot holdout. Blank-only comparisons use a separate intersection.",
        "aggregation": "Physical blank means before fitting; filter and known-lot holdouts; solver failures retained",
        "arpls_implementation": {
            "penalty": "second channel differences on fixed uniform native grid",
            "negative_residual_sd_ddof": 0,
            "note": "Population SD is an explicit implementation choice; numerical equivalence to published MATLAB or other arPLS code is not established",
        },
        "background_model_ledger_rows": len(model_rows),
        "repeat_filters": int(scans.loc[repeated, "base_filter_id"].nunique()), "repeat_scans": int(repeated.sum()),
        "repeat_comparisons_per_method": int(len(pairs) / len(METHODS)),
        "receipt_snapshot": "start_of_run",
        "inputs": input_receipts,
        "code": code_receipts,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    (output / "manifest.json").write_text(json.dumps(safe_json(manifest), indent=2) + "\n")
    make_figure(trials, residuals, output)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("inputs", "code")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "research/ftir_hips_chem/output/tables/baseline_challenge_20261102")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    run(args.native_dir, args.output, args.jobs)
