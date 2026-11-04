"""Profile unchanged VIBES stages and check an experimental batched MAP solver.

Synthetic inputs are a numerical smoke test, never a forecast for real spectra.
An input NPZ must contain spectra, mean, components and wavenumbers; it must
contain one already frozen blank model. Do not combine different lots/models.
No raw spectra or fitted arrays are written by this benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cvxpy as cp
import vibes_baseline as adapter


def synthetic_inputs(samples: int, channels: int, rank: int, seed: int):
    """Create a deterministic PCA-shaped basis with weak and broad injections."""
    if min(samples, rank) < 1 or channels <= rank:
        raise ValueError("Require samples >= 1 and channels > rank >= 1")
    rng = np.random.default_rng(seed)
    wn = np.linspace(3998.0, 500.0, channels)
    t = np.linspace(0, 1, channels)
    basis = np.cos(np.pi * (t[:, None] + 0.5 / channels) * np.arange(rank)[None, :])
    q, _ = np.linalg.qr(basis)
    components = q * np.geomspace(0.8, 0.005, rank)
    mean = 0.15 + 0.3 * np.exp(-0.5 * ((wn - 1200) / 180) ** 2)
    blank_part = mean + rng.normal(0, 0.5, (samples, rank)) @ components.T
    levels = np.resize(np.array([0.002, 0.02, 0.15]), samples)
    signal = levels[:, None] * np.exp(-0.5 * ((wn - 2400) / 650) ** 2)
    signal += 0.03 * np.exp(-0.5 * ((wn - 1710) / 32) ** 2)
    spectra = blank_part + signal + rng.normal(0, 2e-5, blank_part.shape)
    return wn, spectra, mean, components


def source_hashes():
    files = [
        "research/ftir_hips_chem/workflows/benchmark_vibes_gpu.py",
        "research/ftir_hips_chem/scripts/vibes_baseline.py",
        "research/ftir_hips_chem/scripts/vibes_map_batch.py",
        "research/ftir_hips_chem/vendor/pyvibes/src/vibes/absorbance_estimators/map.py",
        "research/ftir_hips_chem/vendor/pyvibes/src/vibes/absorbance_estimators/vibes.py",
    ]
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in files if (ROOT / p).exists()}


def profile_reference(wn, spectra, mean, components, tau):
    """Observe existing code without changing its objective or solve settings."""
    bg = adapter.VibesBackground(wn, mean, components, pd.DataFrame(), (), 0.0)
    records, baselines = [], []
    real_minimize, real_solve = adapter.minimize, cp.Problem.solve
    start_all = time.perf_counter()
    for i, row in enumerate(spectra):
        timings = {"elbo_wall_seconds": 0.0, "map_wall_seconds": 0.0,
                   "elbo_calls": 0, "map_status": None,
                   "map_solver_seconds": None, "map_compile_seconds": None}

        def timed_minimize(*args, **kwargs):
            start = time.perf_counter()
            try:
                return real_minimize(*args, **kwargs)
            finally:
                timings["elbo_wall_seconds"] += time.perf_counter() - start
                timings["elbo_calls"] += 1

        def timed_solve(problem, *args, **kwargs):
            start = time.perf_counter()
            try:
                result = real_solve(problem, *args, **kwargs)
                timings["map_status"] = problem.status
                timings["map_compile_seconds"] = problem.compilation_time
                if problem.solver_stats is not None:
                    timings["map_solver_seconds"] = problem.solver_stats.solve_time
                return result
            finally:
                timings["map_wall_seconds"] += time.perf_counter() - start

        with patch.object(adapter, "minimize", timed_minimize), patch.object(cp.Problem, "solve", timed_solve):
            baseline, _, diag = adapter.vibes_baseline_matrix(
                wn, row[None, :], bg, sample_ids=[f"case-{i}"], tau=tau,
            )
        rec = diag.iloc[0].to_dict()
        rec.update(timings)
        # Finite arrays alone are not a successful reference solve.
        rec["reference_accepted"] = bool(rec["success"] and rec["map_status"] == cp.OPTIMAL)
        baselines.append(baseline[0])
        records.append(rec)
    return np.asarray(baselines), records, time.perf_counter() - start_all


def cached_cpu_map(spectra, mean, components, taus, sigmas):
    """Compile once per blank model as a fairer CPU MAP comparison (DPP QP)."""
    complete_start = time.perf_counter()
    n, c = components.shape
    x = cp.Variable(c)
    positive = cp.Variable(n, nonneg=True)
    negative = cp.Variable(n, nonneg=True)
    value = cp.Parameter(n)
    tau = cp.Parameter(nonneg=True)
    sigma = cp.Parameter(nonneg=True)
    objective = tau * cp.sum(positive) + (1 - tau) * cp.sum(negative)
    objective += 0.5 * sigma * cp.sum_squares(x)
    problem = cp.Problem(cp.Minimize(objective), [components @ x + positive - negative == value])
    if not problem.is_dpp():
        raise RuntimeError("Cached CPU comparator must satisfy DPP")
    outputs, statuses, seconds, certificates = [], [], [], []
    for y, t, s in zip(spectra, taus, sigmas):
        value.value, tau.value, sigma.value = y - mean, t, s
        start = time.perf_counter()
        problem.solve(solver=cp.CLARABEL, warm_start=True)
        seconds.append(time.perf_counter() - start)
        statuses.append(problem.status)
        outputs.append(mean + components @ x.value if x.value is not None else np.full(n, np.nan))
        cert = {"verified": False}
        if x.value is not None and problem.constraints[0].dual_value is not None:
            # Equality is Wx + positive - negative = y-mu; its multiplier has
            # the opposite sign to the bounded dual of the pinball objective.
            dual = -problem.constraints[0].dual_value
            feasible_dual = np.clip(dual, t - 1, t)
            residual = y - mean - components @ x.value
            primal_obj = np.maximum(t * residual, (t - 1) * residual).sum()
            primal_obj += 0.5 * s * np.sum(x.value**2)
            dual_obj = feasible_dual @ (y - mean) - np.sum((components.T @ feasible_dual)**2) / (2 * s)
            gap = float(primal_obj - dual_obj)
            gap_limit = 2e-7 + 2e-7 * max(1.0, abs(primal_obj), abs(dual_obj))
            cert = {"primal_objective": float(primal_obj), "dual_objective": float(dual_obj),
                    "duality_gap": gap, "gap_limit": gap_limit,
                    "dual_box_violation": float(np.max(np.abs(dual - feasible_dual))),
                    "verified": bool(problem.status == cp.OPTIMAL and np.isfinite(gap)
                                     and -gap_limit <= gap <= gap_limit)}
        certificates.append(cert)
    return np.asarray(outputs), statuses, seconds, certificates, time.perf_counter() - complete_start


def machine_info():
    versions = {}
    for name in ("numpy", "scipy", "pandas", "cvxpy", "clarabel", "cupy-cuda12x", "cupy-cuda13x"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    gpu = {"nvidia_smi_available": bool(shutil.which("nvidia-smi"))}
    if gpu["nvidia_smi_available"]:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                     "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            gpu["devices"] = result.stdout.strip().splitlines() if result.returncode == 0 else []
        except (OSError, subprocess.TimeoutExpired):
            gpu["query_failed"] = True
    try:
        import cupy
        gpu["cuda_runtime_version"] = cupy.cuda.runtime.runtimeGetVersion()
        gpu["cuda_driver_version"] = cupy.cuda.runtime.driverGetVersion()
        gpu["active_cuda_device"] = cupy.cuda.runtime.getDevice()
    except (ImportError, RuntimeError):
        pass
    return {"python": platform.python_version(), "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "installed_cvxpy_solvers": cp.installed_solvers(), "versions": versions, "gpu": gpu}


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, help="Frozen-model NPZ, allow_pickle=False")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--channels", type=int, default=2722)
    ap.add_argument("--rank", type=int, default=70)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--estimate-tau", action="store_true")
    ap.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    ap.add_argument("--blas-threads", type=int, default=1)
    ap.add_argument("--map-maxiter", type=int, default=10000)
    ap.add_argument("--rho", type=float, default=10.0)
    ap.add_argument("--baseline-atol", type=float, default=2e-5,
                    help="Pilot absolute absorbance agreement threshold; must be justified for real data")
    ap.add_argument("--profile-only", action="store_true")
    a = ap.parse_args()
    if a.samples < 1 or a.blas_threads < 1:
        ap.error("samples and blas-threads must be positive")
    if not np.isfinite(a.baseline_atol) or a.baseline_atol <= 0:
        ap.error("baseline-atol must be finite and positive")
    if a.input:
        with np.load(a.input, allow_pickle=False) as data:
            wn, spectra, mean, components = (np.asarray(data[k], dtype=np.float64)
                                           for k in ["wavenumbers", "spectra", "mean", "components"])
        spectra = spectra[:a.samples]
        input_hash = hashlib.sha256(a.input.read_bytes()).hexdigest()
    else:
        wn, spectra, mean, components = synthetic_inputs(a.samples, a.channels, a.rank, a.seed)
        input_hash = None
    summary = {"schema_version": 1, "input_kind": "provided_frozen_model" if a.input else "synthetic",
               "input_sha256": input_hash, "samples": len(spectra), "channels": len(wn),
               "rank": components.shape[1], "machine": machine_info(), "source_hashes": source_hashes(),
               "blas_threads": a.blas_threads, "tau": None if a.estimate_tau else a.tau,
               "baseline_agreement_atol": a.baseline_atol,
               "claim_limit": "Numerical/performance pilot only; no real-filter accuracy or GPU speedup established."}
    with threadpool_limits(limits=a.blas_threads):
        expected, records, wall = profile_reference(wn, spectra, mean, components, summary["tau"])
        good = np.array([r["reference_accepted"] for r in records])
        summary["reference"] = {"wall_seconds": wall, "accepted": int(good.sum()),
                                "attempted": len(spectra), "elbo_wall_seconds": sum(r["elbo_wall_seconds"] for r in records),
                                "map_wall_seconds": sum(r["map_wall_seconds"] for r in records), "rows": records}
        if good.any():
            taus = np.array([r["tau"] for r in records])[good]
            sigmas = np.array([r["sigma"] for r in records])[good]
            cached, statuses, seconds, certificates, cached_wall = cached_cpu_map(spectra[good], mean, components, taus, sigmas)
            cached_good = np.array([c["verified"] for c in certificates])
            cached_delta = float(np.max(np.abs(cached[cached_good] - expected[good][cached_good]))) if cached_good.any() else None
            summary["cached_cpu_map"] = {"statuses": statuses, "wall_seconds_per_fit": seconds,
                                          "wall_seconds_including_setup": cached_wall,
                                          "cold_seconds": seconds[0], "remaining_seconds": sum(seconds[1:]),
                                          "certificates": certificates,
                                          "max_abs_baseline_difference": cached_delta,
                                          "comparison_passed": bool(cached_good.all() and cached_delta is not None and cached_delta <= a.baseline_atol)}
            if not a.profile_only:
                from vibes_map_batch import solve_pinball_map_batch
                # Report cold and warm batches separately. CuPy backend synchronizes.
                runs = []
                for label in ("cold", "warm"):
                    start = time.perf_counter()
                    result = solve_pinball_map_batch(spectra[good], mean, components, tau=taus, sigma=sigmas,
                                                    backend=a.backend, rho=a.rho, maxiter=a.map_maxiter)
                    elapsed = time.perf_counter() - start
                    successful = np.asarray(result.converged)
                    delta = result.baseline[successful] - expected[good][successful]
                    max_difference = float(np.max(np.abs(delta))) if delta.size else None
                    runs.append({"phase": label, "wall_seconds_including_setup": elapsed,
                                 "accepted": int(successful.sum()), "attempted": int(good.sum()),
                                 "max_abs_baseline_difference": max_difference,
                                 "comparison_passed": bool(successful.all() and max_difference is not None and max_difference <= a.baseline_atol),
                                 "rms_baseline_difference": float(np.sqrt(np.mean(delta**2))) if delta.size else None,
                                 "diagnostics": [{**d, "input_row": int(np.flatnonzero(good)[i])}
                                                 for i, d in enumerate(result.diagnostics)]})
                summary["experimental_map"] = {"backend": a.backend, "rho": a.rho, "maxiter": a.map_maxiter,
                                               "dtype": "float64", "runs": runs}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(finite_json(summary), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"input_kind": summary["input_kind"], "samples": len(spectra), "channels": len(wn),
                      "rank": components.shape[1], "reference_accepted": int(good.sum()),
                      "wall_seconds": wall, "elbo_seconds": summary["reference"]["elbo_wall_seconds"],
                      "map_seconds": summary["reference"]["map_wall_seconds"]}, indent=2))
    # A partial reference or candidate is never a successful benchmark.
    candidate_passed = "experimental_map" not in summary or all(
        r["comparison_passed"] for r in summary["experimental_map"]["runs"])
    cached_passed = summary.get("cached_cpu_map", {}).get("comparison_passed", False)
    return 0 if good.all() and cached_passed and candidate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
