"""Benchmark batched VIBES (vibes_batch.py) against the 19 Oct network run, CPU or GPU.

Inputs are the frozen 19 Oct network VIBES bundle (4,452 SPARTAN spectra on the
2,722-point model grid, the 87-blank rank-70 background) and that run's saved
sigma / ELBO / iterations / baselines. Nothing here changes the adapter or any
saved correction; outputs are benchmark tables only.

Stages:

    # on the Mac: build a self-contained bundle (code + inputs + saved reference)
    python research/ftir_hips_chem/workflows/benchmark_vibes_batch.py prepare
    # anywhere (the Colab VM, or locally with --backend numpy):
    python benchmark_vibes_batch.py run --bundle DIR --out DIR --backend cupy --rows all
    python benchmark_vibes_batch.py run --bundle DIR --out DIR --backend numpy --rows sample:96

``run`` writes summary.json (timings, hardware, agreement) and results.npz.
Agreement is reported three ways, because the reference is only defined to
optimiser tolerance (SciPy's relative-reduction stop is path dependent):

* map_given_saved_sigma: batched MAP fed the *saved* sigma vs the saved baseline.
  This isolates the MAP solver; it should agree to ~1e-6 absorbance.
* sigma vs saved: the batched ELBO path vs the saved path (expect a heavy tail).
* cpu_fused (optional, --cpu-ref N): the adapter's SciPy procedure with the fused
  objective on N rows of the same machine, as the same-machine CPU comparator and
  a measure of the reference's own path noise.
* vs_polished (when the bundle has polished_reference.csv): ELBO and sigma against
  the G4 fused-SciPy run with 5 polish restarts, the closest thing to the optimum
  we had before the Newton solver.

ELBO paths (--elbo): batch (vectorised L-BFGS, numpy/cupy), cpu (fused SciPy on a
process pool), newton (batched damped Newton, numpy/cupy) and jax (the batch
L-BFGS compiled with jax.jit; uses JAX's default device).
``latency`` times one optimiser iteration per path and batch size.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
SCRIPTS = HERE.parents[1] / "scripts"


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- prepare
def prepare(args):
    import pandas as pd

    root = HERE.parents[3]
    src = Path(args.source) if args.source else root / "research/ftir_hips_chem/output/tables/network_baselines_20261019"
    out = Path(args.out or root / "research/ftir_hips_chem/output/tables/vibes_gpu_bench_20261103/bundle")
    if out.exists():
        shutil.rmtree(out)
    (out / "code").mkdir(parents=True)
    shutil.copy2(src / "vibes_bundle/background_model.npz", out / "background_model.npz")
    shutil.copy2(src / "vibes_bundle/X_model_grid.npy", out / "X_model_grid.npy")
    shutil.copy2(src / "vibes_bundle/cases.csv", out / "cases.csv")
    diag = pd.read_csv(src / "vibes_fit_diagnostics.csv")
    keep = ["case_row", "sample_id", "success", "tau", "sigma", "elbo", "iterations",
            "retry_count", "message", "seconds"]
    diag[keep].to_csv(out / "saved_diagnostics.csv", index=False)
    shutil.copy2(src / "vibes_baselines_model_grid.npy", out / "saved_baselines.npy")
    shutil.copy2(SCRIPTS / "vibes_batch.py", out / "code/vibes_batch.py")
    shutil.copy2(SCRIPTS / "vibes_jax.py", out / "code/vibes_jax.py")
    if args.polished:
        shutil.copy2(args.polished, out / "polished_reference.csv")
    shutil.copy2(HERE, out / "code/benchmark_vibes_batch.py")
    (out / "requirements.txt").write_text("threadpoolctl\n")
    files = {p.name: _sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "MANIFEST.json").write_text(json.dumps({
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": src.name, "rows": int(len(diag)), "files": files}, indent=1))
    print(f"bundle -> {out} ({sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 1e6:.0f} MB)")


# --------------------------------------------------------------------------- run
def machine_info(backend):
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "cpus": os.cpu_count(), "versions": {}}
    for name in ("numpy", "scipy", "cupy-cuda12x", "cupy-cuda13x", "cupy"):
        try:
            info["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if shutil.which("nvidia-smi"):
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        info["gpu"] = p.stdout.strip().splitlines()
    try:
        from threadpoolctl import threadpool_info
        info["blas"] = [(i.get("internal_api"), i.get("num_threads")) for i in threadpool_info()]
    except ImportError:
        pass
    return info


def select_rows(spec, saved, seed=20261103):
    n = len(saved)
    ok = saved["success"].to_numpy(bool)
    if spec == "all":
        return np.arange(n)
    if ":" in spec and spec.split(":")[0].isdigit():
        a, b = spec.split(":")
        return np.arange(int(a), min(int(b), n))
    if spec.startswith("sample:"):
        k = int(spec.split(":")[1])
        rng = np.random.default_rng(seed)
        it = saved["iterations"].to_numpy()
        # Stratify so the slow tail and the suspicious short stops are present.
        short = np.flatnonzero(ok & (it < 30))
        slow = np.flatnonzero(ok & (it > 20000))
        pick = list(rng.choice(short, min(len(short), max(1, k // 8)), replace=False))
        pick += list(rng.choice(slow, min(len(slow), max(1, k // 8)), replace=False))
        rest = np.setdiff1d(np.flatnonzero(ok), pick)
        pick += list(rng.choice(rest, max(0, k - len(pick)), replace=False))
        return np.sort(np.asarray(pick[:k]))
    raise ValueError("rows must be all, a:b, or sample:N")


def _q(a, qs=(0.5, 0.9, 0.99, 1.0)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return {f"p{int(q * 100)}": float(np.quantile(a, q)) for q in qs}


def _rel_summary(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (b > 0) & (a > 0)
    r = np.abs(a[m] / b[m] - 1)
    ratio = np.maximum(a[m] / b[m], b[m] / a[m])
    return {"n": int(m.sum()), "rel_diff": _q(r), "frac_gt_1pct": float(np.mean(r > 0.01)),
            "frac_gt_10pct": float(np.mean(r > 0.1)), "frac_gt_2x": float(np.mean(ratio > 2))}


def _cpu_job(args):
    bundle, row, polish = args
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.path.insert(0, str(Path(bundle) / "code"))
    from vibes_batch import fit_elbo_scipy_fused
    m = np.load(Path(bundle) / "background_model.npz")
    X = np.load(Path(bundle) / "X_model_grid.npy", mmap_mode="r")
    t = time.perf_counter()
    r = fit_elbo_scipy_fused(np.asarray(X[row], float), m["mean"], m["components"], polish_restarts=polish)
    r["seconds"], r["row"] = time.perf_counter() - t, int(row)
    return r


def _import_code(bundle):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # share the GPU with CuPy
    sys.path.insert(0, str(bundle / "code") if (bundle / "code/vibes_batch.py").exists() else str(SCRIPTS))
    import vibes_batch as vb
    return vb


def run_elbo(args, vb, Y, mu, W, tau, progress=None):
    """One ELBO pass on the selected path; returns (ElboResult-like, seconds, extra)."""
    extra = {}
    if args.elbo == "newton":
        t = time.perf_counter()
        e = vb.fit_elbo_newton(Y, mu, W, tau=tau, backend=args.backend, chunk=args.newton_chunk,
                               hess_dtype=args.hess_dtype, progress=progress)
        return e, time.perf_counter() - t, extra
    if args.elbo == "jax":
        import jax
        import vibes_jax as vj
        extra["jax"] = {"version": jax.__version__, "backend": jax.default_backend(),
                        "devices": [str(d) for d in jax.devices()]}
        B = int(min(args.pool, vj._pad(len(Y))))
        extra["jax"]["engine"] = args.engine
        extra["jax"]["compile_seconds"] = vj.precompile(Y.shape[1], W.shape[1], tau=tau, pool=B, chunk_iters=args.chunk_iters,
                                                        min_pool=args.min_pool, engine=args.engine)
        t = time.perf_counter()
        e = vj.fit_elbo_jax(Y, mu, W, tau=tau, pool=B, chunk_iters=args.chunk_iters, min_pool=args.min_pool,
                            polish_restarts=args.polish, engine=args.engine, progress=progress)
        return e, time.perf_counter() - t, extra
    t = time.perf_counter()
    e = vb.fit_elbo_batch(Y, mu, W, tau=tau, backend=args.backend, pool=args.pool,
                          polish_restarts=args.polish, progress=progress)
    return e, time.perf_counter() - t, extra


def run(args):
    import pandas as pd

    bundle = Path(args.bundle)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vb = _import_code(bundle)

    m = np.load(bundle / "background_model.npz")
    mu, W = m["mean"].astype(float), m["components"].astype(float)
    X = np.load(bundle / "X_model_grid.npy", mmap_mode="r")
    saved = pd.read_csv(bundle / "saved_diagnostics.csv")
    saved_base = np.load(bundle / "saved_baselines.npy", mmap_mode="r")
    rows = select_rows(args.rows, saved)
    Y = np.asarray(X[rows], dtype=np.float64)
    tau = 0.1
    summary = {"schema": 1, "bundle_manifest_sha": _sha(bundle / "MANIFEST.json") if (bundle / "MANIFEST.json").exists() else None,
               "vibes_batch_sha256": _sha(Path(vb.__file__)),
               "rows_spec": args.rows, "rows": int(len(rows)), "backend": args.backend, "pool": args.pool,
               "polish_restarts": args.polish, "elbo_path": args.elbo, "map_chunk": args.map_chunk, "machine": machine_info(args.backend),
               "newton": {"chunk": args.newton_chunk, "hess_dtype": args.hess_dtype} if args.elbo == "newton" else None,
               "jax": {"chunk_iters": args.chunk_iters, "min_pool": args.min_pool} if args.elbo == "jax" else None,
               "claim_limit": "Benchmark only; saved corrections and the adapter are unchanged."}
    print(json.dumps({k: summary[k] for k in ("rows", "backend", "pool", "polish_restarts")}), flush=True)

    # Warm-up (CUDA context, kernels) on a few rows, reported separately.
    t = time.perf_counter()
    vb.fit_elbo_batch(Y[:4], mu, W, tau=tau, backend=args.backend, pool=4, maxiter=50, retry_failed=False)
    if args.elbo == "newton":
        vb.fit_elbo_newton(Y[:4], mu, W, tau=tau, backend=args.backend, maxiter=3, hess_dtype=args.hess_dtype)
    vb.solve_map_batch(Y[:4], mu, W, sigma=np.full(4, 1e-3), tau=tau, backend=args.backend)
    summary["warmup_seconds"] = time.perf_counter() - t

    # 1. MAP only, given the saved sigma: isolates the MAP solver.
    ok_saved = saved["success"].to_numpy(bool)[rows]
    sig_saved = saved["sigma"].to_numpy(float)[rows]
    t = time.perf_counter()
    mres = vb.solve_map_batch(Y[ok_saved], mu, W, sigma=sig_saved[ok_saved], tau=tau,
                              backend=args.backend, chunk=args.map_chunk)
    t_map = time.perf_counter() - t
    ref = np.asarray(saved_base[rows[ok_saved]], dtype=np.float64)
    dmax = np.nanmax(np.abs(mres.baseline - ref), axis=1)
    summary["map_given_saved_sigma"] = {
        "seconds": t_map, "rows": int(ok_saved.sum()), "certified": int(mres.converged.sum()),
        "fits_per_minute": 60 * ok_saved.sum() / t_map, "iterations": _q(mres.iterations),
        "max_abs_baseline_diff_vs_saved": _q(dmax),
        "note": "saved baselines are float32 (resolution ~1e-8 at 0.1 absorbance)"}
    print("MAP given saved sigma:", json.dumps(summary["map_given_saved_sigma"], default=float), flush=True)

    # 2. ELBO + MAP end to end.
    last = [time.perf_counter()]

    def progress(done, total):
        if time.perf_counter() - last[0] > 30:
            last[0] = time.perf_counter()
            print(f"  ELBO {done}/{total}", flush=True)

    extra = {}
    t = time.perf_counter()
    if args.elbo == "cpu":
        # Fused SciPy ELBO on every core (the recommended path), MAP on --backend.
        from concurrent.futures import ProcessPoolExecutor
        from types import SimpleNamespace
        workers = args.cpu_workers or os.cpu_count()
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_cpu_job, [(str(bundle), int(r), args.polish) for r in rows], chunksize=4))
        res = sorted(res, key=lambda r: rows.searchsorted(r["row"]))
        e = SimpleNamespace(
            sigma=np.array([r["sigma"] for r in res]), elbo=np.array([r["elbo"] for r in res]),
            success=np.array([r["success"] for r in res]),
            iterations=np.array([r.get("iterations", -1) for r in res]),
            message=[r["message"] for r in res], stats={"cpu_workers": workers})
        t_elbo = time.perf_counter() - t
    else:
        e, t_elbo, extra = run_elbo(args, vb, Y, mu, W, tau, progress)
    summary.update(extra)
    good = e.success
    t = time.perf_counter()
    m2 = vb.solve_map_batch(Y[good], mu, W, sigma=e.sigma[good], tau=tau, backend=args.backend, chunk=args.map_chunk)
    t_map2 = time.perf_counter() - t
    baseline = np.full(Y.shape, np.nan)
    baseline[good] = m2.baseline
    both = good & ok_saved
    ref_all = np.asarray(saved_base[rows], dtype=np.float64)
    dmax_e2e = np.nanmax(np.abs(baseline[both] - ref_all[both]), axis=1)
    corr_rms = np.sqrt(np.nanmean((Y[both] - ref_all[both]) ** 2, axis=1))
    drms = np.sqrt(np.nanmean((baseline[both] - ref_all[both]) ** 2, axis=1))
    summary["end_to_end"] = {
        "elbo_seconds": t_elbo, "map_seconds": t_map2, "total_seconds": t_elbo + t_map2,
        "fits_per_minute": 60 * len(rows) / (t_elbo + t_map2),
        "elbo_success": int(good.sum()), "map_certified": int(m2.converged.sum()),
        "elbo_stats": {k: v for k, v in e.stats.items() if np.isscalar(v)},
        "iterations_total": int(e.iterations.sum()),
        "iterations_saved_total": int(saved["iterations"].to_numpy()[rows].sum()),
        "sigma_vs_saved": _rel_summary(e.sigma[both], sig_saved[both]),
        "elbo_minus_saved": _q(e.elbo[both] - saved["elbo"].to_numpy()[rows][both], (0.01, 0.05, 0.5, 0.95, 0.99)),
        "max_abs_baseline_diff_vs_saved": _q(dmax_e2e),
        "rms_baseline_diff_over_corrected_rms": _q(drms / corr_rms),
        "messages": pd.Series(e.message).value_counts().to_dict()}
    if "pool_sizes" in e.stats:
        summary["end_to_end"]["pool_sizes"] = e.stats["pool_sizes"]
    if args.elbo == "newton":
        summary["end_to_end"]["newton_iterations"] = _q(e.iterations)
        summary["end_to_end"]["final_step_exact"] = int(np.sum(e.stats["final_step_exact"]))
    pol_path = bundle / "polished_reference.csv"
    if pol_path.exists():
        pol = pd.read_csv(pol_path).set_index("row").reindex(rows)
        okp = good & pol["success"].fillna(False).to_numpy(bool)
        dE = e.elbo[okp] - pol["elbo"].to_numpy()[okp]
        summary["vs_polished"] = {
            "n": int(okp.sum()), "elbo_minus_polished": _q(dE, (0.0, 0.01, 0.5, 0.99, 1.0)),
            "rows_below_polished_by_1e-6_rel": int(np.sum(dE < -1e-6 * np.abs(pol["elbo"].to_numpy()[okp]))),
            "sigma_vs_polished": _rel_summary(e.sigma[okp], pol["sigma"].to_numpy()[okp]),
            "reference": "G4 fused SciPy L-BFGS-B, 5 polish restarts (cpu_all_polish5.csv)"}
        print("vs polished:", json.dumps(summary["vs_polished"], default=float)[:400], flush=True)
    print("end to end:", json.dumps({k: summary["end_to_end"][k] for k in
                                      ("elbo_seconds", "map_seconds", "fits_per_minute", "elbo_success", "map_certified")}), flush=True)

    # 3. Same-machine CPU comparator (fused SciPy, process pool).
    if args.cpu_ref:
        from concurrent.futures import ProcessPoolExecutor
        sub = rows[np.linspace(0, len(rows) - 1, min(args.cpu_ref, len(rows))).astype(int)]
        workers = args.cpu_workers or os.cpu_count()
        t = time.perf_counter()
        with ProcessPoolExecutor(workers) as ex:
            cpu = pd.DataFrame(list(ex.map(_cpu_job, [(str(bundle), int(r), args.polish) for r in sub])))
        t_cpu = time.perf_counter() - t
        pos = np.searchsorted(rows, sub)
        summary["cpu_fused"] = {
            "rows": int(len(sub)), "workers": workers, "elbo_seconds": t_cpu,
            "elbo_fits_per_minute": 60 * len(sub) / t_cpu,
            "batch_elbo_fits_per_minute_same_rows_est": 60 * len(rows) / t_elbo,
            "sigma_batch_vs_cpu": _rel_summary(e.sigma[pos], cpu["sigma"].to_numpy()),
            "sigma_cpu_vs_saved": _rel_summary(cpu["sigma"].to_numpy(), sig_saved[pos]),
            "note": "ELBO only; add MAP (~0.6 s/fit/core with CLARABEL) for the CPU end-to-end rate"}
        cpu.to_csv(out / "cpu_fused.csv", index=False)
        print("cpu fused:", json.dumps({k: summary["cpu_fused"][k] for k in ("rows", "workers", "elbo_fits_per_minute")}), flush=True)

    np.savez_compressed(out / "results.npz", rows=rows, sigma=e.sigma, elbo=e.elbo, iterations=e.iterations,
                        elbo_success=e.success, map_certified=np.isin(np.arange(len(rows)), np.flatnonzero(good)[m2.converged]),
                        baseline=baseline.astype(np.float32), message=np.asarray(e.message))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(f"wrote {out}/summary.json", flush=True)


def latency(args):
    """Wall time per optimiser iteration vs batch size, after warm-up/compile."""
    import pandas as pd

    bundle = Path(args.bundle)
    vb = _import_code(bundle)
    m = np.load(bundle / "background_model.npz")
    mu, W = m["mean"].astype(float), m["components"].astype(float)
    X = np.load(bundle / "X_model_grid.npy", mmap_mode="r")
    saved = pd.read_csv(bundle / "saved_diagnostics.csv")
    # Slow rows keep every slot busy for the whole timed window.
    slow = np.flatnonzero(saved["iterations"].to_numpy() > 20000)
    res = {"machine": machine_info(args.backend)}
    for path in args.paths.split(","):
        for B in [int(b) for b in args.pools.split(",")]:
            rows = np.resize(slow, B)
            Y = np.asarray(X[rows], float)
            key = f"{path}_B{B}"
            if path == "jax":
                import vibes_jax as vj
                comp = vj.precompile(Y.shape[1], W.shape[1], pool=B, chunk_iters=args.chunk_iters, min_pool=B)
                e = vj.fit_elbo_jax(Y, mu, W, pool=B, chunk_iters=args.chunk_iters, min_pool=B,
                                    maxiter=args.iters, retry_failed=False)
                its = e.stats["device_iterations"]
                res[key] = {"compile_seconds": comp, "seconds": e.seconds, "iterations": its,
                            "ms_per_iteration": 1e3 * e.seconds / its,
                            "device_wait_ms_per_iteration": 1e3 * e.stats["device_wait_seconds"] / its,
                            "us_per_row_iteration": 1e6 * e.seconds / its / B}
            elif path == "newton":
                vb.fit_elbo_newton(Y[:4], mu, W, backend=args.backend, maxiter=2, hess_dtype=args.hess_dtype)
                e = vb.fit_elbo_newton(Y, mu, W, backend=args.backend, chunk=B, maxiter=args.newton_iters,
                                       hess_dtype=args.hess_dtype)
                its = e.stats["newton_iterations"]
                res[key] = {"seconds": e.seconds, "iterations": its, "hess_dtype": args.hess_dtype,
                            "ms_per_iteration": 1e3 * e.seconds / max(its, 1),
                            "row_hessians": e.stats["row_hessians"]}
            else:
                vb.fit_elbo_batch(Y[:4], mu, W, backend=args.backend, pool=4, maxiter=5, retry_failed=False)
                e = vb.fit_elbo_batch(Y, mu, W, backend=args.backend, pool=B, maxiter=args.iters, retry_failed=False)
                its = int(e.iterations.max())
                res[key] = {"seconds": e.seconds, "iterations": its, "ms_per_iteration": 1e3 * e.seconds / its,
                            "us_per_row_iteration": 1e6 * e.seconds / its / B}
            print(key, json.dumps(res[key], default=float), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out")
    p.add_argument("--source", help="network_baselines_20261019 folder (default: this checkout's output/tables)")
    p.add_argument("--polished", help="cpu_all_polish5.csv to ship as polished_reference.csv")
    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    r.add_argument("--rows", default="sample:96")
    r.add_argument("--pool", type=int, default=1024)
    r.add_argument("--map-chunk", type=int, default=512)
    r.add_argument("--polish", type=int, default=0)
    r.add_argument("--elbo", choices=["batch", "cpu", "newton", "jax"], default="batch",
                   help="batch: vectorised L-BFGS on --backend; cpu: fused SciPy on a process pool; "
                        "newton: batched damped Newton on --backend; jax: compiled L-BFGS (JAX device)")
    r.add_argument("--cpu-ref", type=int, default=0)
    r.add_argument("--cpu-workers", type=int, default=0)
    for q in (r,):
        q.add_argument("--newton-chunk", type=int, default=1024)
        q.add_argument("--hess-dtype", choices=["float64", "float32"], default="float64")
        q.add_argument("--chunk-iters", type=int, default=100)
        q.add_argument("--min-pool", type=int, default=8)
        q.add_argument("--engine", choices=["lbfgs", "lbfgsb"], default="lbfgs", help="for --elbo jax")
    lt = sub.add_parser("latency")
    lt.add_argument("--bundle", required=True)
    lt.add_argument("--out", required=True)
    lt.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    lt.add_argument("--paths", default="jax,newton,batch")
    lt.add_argument("--pools", default="64,512,2048")
    lt.add_argument("--iters", type=int, default=300)
    lt.add_argument("--newton-iters", type=int, default=300)
    lt.add_argument("--hess-dtype", choices=["float64", "float32"], default="float64")
    lt.add_argument("--chunk-iters", type=int, default=100)
    a = ap.parse_args()
    {"prepare": prepare, "run": run, "latency": latency}[a.stage](a)


if __name__ == "__main__":
    main()
