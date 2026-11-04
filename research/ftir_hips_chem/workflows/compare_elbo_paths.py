"""Head-to-head comparison of VIBES ELBO paths on the 19 Oct network bundle (bundle_v2).

Paths: batched damped Newton (``vibes_batch.fit_elbo_newton``, numpy or cupy),
JAX-compiled L-BFGS (``vibes_jax.fit_elbo_jax``, same optimiser as the adapter's
SciPy L-BFGS-B), and the fused SciPy reference on a CPU process pool.

Tests (each writes ``<out>/<test>.json``; a failing test is logged and skipped):

    scale       wall time vs job size (1 ... 17,808 spectra; > 4,452 = tiled + 1e-5 noise)
    single      one spectrum at a time (typical, slowest, premature-stop), interactive use
    robust      tau 0.05/0.1/0.2/0.3 and background rank 20/40/70 on a stratified 256 sample
    multistart  8 starts per spectrum (default + 7 random): are there other optima?
    warm        warm start after a small change (tau 0.1 -> 0.11; spectra + 1e-4 noise)
    tune        Newton chunk size; JAX pool size and chunk length (time, GPU memory)

    python compare_elbo_paths.py --bundle bundle_v2 --out cmp --tests scale,single,robust
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

C = {}  # shared context


# --------------------------------------------------------------------------- setup
def setup(args):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    bundle = Path(args.bundle)
    sys.path.insert(0, str(bundle / "code"))
    import pandas as pd
    import vibes_batch as vb
    from benchmark_vibes_batch import machine_info, select_rows

    m = np.load(bundle / "background_model.npz")
    C.update(args=args, vb=vb, pd=pd, select_rows=select_rows, bundle=bundle,
             mu=m["mean"].astype(float), W=m["components"].astype(float),
             X=np.asarray(np.load(bundle / "X_model_grid.npy"), dtype=np.float64),
             saved=pd.read_csv(bundle / "saved_diagnostics.csv"),
             cases=pd.read_csv(bundle / "cases.csv"), jax_compiled=set(),
             machine=machine_info(args.backend))
    if (bundle / "polished_reference.csv").exists():
        C["polished"] = pd.read_csv(bundle / "polished_reference.csv").set_index("row").sort_index()


def gpu_mem_reset():
    if C["args"].backend == "cupy":
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()


def gpu_mem():
    out = {}
    if C["args"].backend == "cupy":
        import cupy as cp
        out["cupy_pool_bytes"] = int(cp.get_default_memory_pool().total_bytes())
    if "vj" in C:
        import jax
        st = jax.local_devices()[0].memory_stats() or {}
        out["jax_peak_bytes"] = int(st.get("peak_bytes_in_use", 0))
    return out


def q(a, qs=(0.0, 0.01, 0.5, 0.99, 1.0)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return {f"p{round(x * 100)}": float(np.quantile(a, x)) for x in qs} if len(a) else None


def compare(e_ref, e_other, label):
    """ELBO of ``e_ref`` minus ``e_other`` where both succeeded."""
    m = np.asarray(e_ref.success, bool) & np.asarray(e_other.success, bool)
    d = e_ref.elbo[m] - e_other.elbo[m]
    sr = e_other.sigma[m] / e_ref.sigma[m]
    return {"vs": label, "n": int(m.sum()), "elbo_diff": q(d),
            "ref_lower_by_1e-6_rel": int(np.sum(d < -1e-6 * np.abs(e_other.elbo[m]))),
            "other_lower_by_1e-6_rel": int(np.sum(d > 1e-6 * np.abs(e_other.elbo[m]))),
            "sigma_gt_10pct": float(np.mean(np.abs(sr - 1) > 0.1)) if m.any() else None,
            "sigma_gt_2x": int(np.sum((sr > 2) | (sr < 0.5)))}


def brief(e, seconds, **extra):
    return {"seconds": seconds, "rows": int(len(e.success)), "success": int(np.sum(e.success)),
            "fits_per_minute": 60 * len(e.success) / seconds if seconds > 0 else None,
            "iterations": q(e.iterations, (0.5, 0.9, 1.0)),
            "evaluations_total": int(np.sum(e.evaluations)) if hasattr(e, "evaluations") else None,
            **extra}


# --------------------------------------------------------------------------- runners
def newton(Y, W=None, tau=0.1, x0=None, dtype=None, chunk=None, backend=None):
    vb = C["vb"]
    a = C["args"]
    t = time.perf_counter()
    e = vb.fit_elbo_newton(Y, C["mu"], C["W"] if W is None else W, tau=tau, x0=x0,
                           backend=backend or a.backend, chunk=chunk or a.newton_chunk,
                           hess_dtype=dtype or a.hess_dtype)
    return e, time.perf_counter() - t


def jaxfit(Y, W=None, tau=0.1, polish=0, x0=None, pool=2048, chunk_iters=100, min_pool=8):
    if "vj" not in C:
        import vibes_jax as vj
        C["vj"] = vj
    vj = C["vj"]
    W = C["W"] if W is None else W
    B = int(min(pool, vj._pad(len(Y))))
    key = (Y.shape[1], W.shape[1], float(tau), B, chunk_iters, min(min_pool, B))
    comp = 0.0
    if key not in C["jax_compiled"]:
        comp = vj.precompile(Y.shape[1], W.shape[1], tau=tau, pool=B, chunk_iters=chunk_iters,
                             min_pool=min(min_pool, B))
        C["jax_compiled"].add(key)
    t = time.perf_counter()
    e = vj.fit_elbo_jax(Y, C["mu"], W, tau=tau, pool=B, chunk_iters=chunk_iters,
                        min_pool=min(min_pool, B), polish_restarts=polish, x0=x0)
    return e, time.perf_counter() - t, comp


_G = {}


def _pool_init(code, mu, W):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.path.insert(0, code)
    import vibes_batch as vb
    _G.update(vb=vb, mu=mu, W=W)


def _scipy_one(job):
    y, tau, c, polish = job
    t = time.perf_counter()
    r = _G["vb"].fit_elbo_scipy_fused(y, _G["mu"], _G["W"][:, :c], tau=tau, polish_restarts=polish)
    r["seconds"] = time.perf_counter() - t
    return r


def scipy_pool(Y, tau=0.1, c=None, polish=0):
    from concurrent.futures import ProcessPoolExecutor
    from types import SimpleNamespace

    c = c or C["W"].shape[1]
    workers = C["args"].cpu_workers or max(1, (os.cpu_count() or 2) - 2)
    t = time.perf_counter()
    import multiprocessing as mp

    # spawn, not fork: forking after JAX/CUDA have started threads can deadlock.
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_pool_init,
                             initargs=(str(C["bundle"] / "code"), C["mu"], C["W"])) as ex:
        res = list(ex.map(_scipy_one, [(y, tau, c, polish) for y in Y], chunksize=1))
    sec = time.perf_counter() - t
    e = SimpleNamespace(sigma=np.array([r["sigma"] for r in res]), elbo=np.array([r["elbo"] for r in res]),
                        success=np.array([r["success"] for r in res]),
                        iterations=np.array([r.get("iterations", -1) for r in res]),
                        per_fit_seconds=np.array([r["seconds"] for r in res]))
    return e, sec, workers


# --------------------------------------------------------------------------- tests
def rows_for(n, seed=0):
    N = len(C["X"])
    rng = np.random.default_rng(seed)
    if n < N:
        r = np.sort(rng.choice(N, n, replace=False))
        return r, C["X"][r]
    reps = int(np.ceil(n / N))
    r = np.tile(np.arange(N), reps)[:n]
    Y = C["X"][r].copy()
    Y[N:] += rng.normal(0, 1e-5, Y[N:].shape)
    return r, Y


def t_scale():
    a = C["args"]
    out = {}
    newton(C["X"][:4], dtype="float64")  # warm up CUDA context and kernels
    newton(C["X"][:4], dtype="float32")
    for n in [int(s) for s in a.sizes.split(",")]:
        r, Y = rows_for(n)
        row = {"n": n, "slow_tail_rows": int(np.sum(C["saved"]["iterations"].to_numpy()[r] > 20000))}
        gpu_mem_reset()
        e64, s64 = newton(Y, dtype="float64")
        row["newton_gpu_f64"] = brief(e64, s64, **gpu_mem())
        e32, s32 = newton(Y, dtype="float32")
        row["newton_gpu_f32"] = brief(e32, s32)
        if a.cpu and n <= 4452:
            e, s = newton(Y, dtype="float64", backend="numpy", chunk=512)
            row["newton_cpu"] = brief(e, s)
        if a.jax and n <= a.jax_max_n:
            e, s, comp = jaxfit(Y)
            row["jax_faithful"] = brief(e, s, compile_seconds=comp, **gpu_mem())
            row["newton_vs_jax"] = compare(e64, e, "jax_faithful")
        if a.cpu and n <= a.scipy_max_n:
            e, s, w = scipy_pool(Y)
            row["scipy_cpu_faithful"] = brief(e, s, workers=w)
            row["newton_vs_scipy"] = compare(e64, e, "scipy_cpu_faithful")
        out[str(n)] = row
        print("scale", n, json.dumps({k: (v.get("seconds") if isinstance(v, dict) else v) for k, v in row.items()}), flush=True)
    return out


def t_single():
    from threadpoolctl import threadpool_limits

    vb, saved = C["vb"], C["saved"]
    it = saved["iterations"].to_numpy()
    ok = saved["success"].to_numpy(bool)
    pm = (C["cases"]["filter_type"] == "PM2.5").to_numpy() & ok
    picks = {"typical_pm25": int(np.flatnonzero(pm)[np.argmin(np.abs(it[pm] - np.median(it[pm])))]),
             "slowest": int(np.argmax(np.where(ok, it, -1))),
             "premature_stop": 466}
    out = {"rows": picks, "saved_iterations": {k: int(it[v]) for k, v in picks.items()}}
    # warm up imports, BLAS and the CUDA context so the first timed call is not penalised
    y0 = C["X"][picks["typical_pm25"]:picks["typical_pm25"] + 1]
    vb.fit_elbo_scipy_fused(y0[0], C["mu"], C["W"])
    newton(y0, backend="numpy")
    newton(y0)

    def best(fn, reps=3):
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            r = fn()
            ts.append(time.perf_counter() - t)
            if ts[-1] > 5:
                break
        return r, min(ts)

    for name, row in picks.items():
        y = C["X"][row:row + 1]
        res = {}
        r, s = best(lambda: vb.fit_elbo_scipy_fused(y[0], C["mu"], C["W"]), reps=1)
        res["scipy_1core"] = {"seconds": s, "elbo": r["elbo"], "iterations": r["iterations"]}
        r, s = best(lambda: vb.fit_elbo_scipy_fused(y[0], C["mu"], C["W"], polish_restarts=5), reps=1)
        res["scipy_1core_polish5"] = {"seconds": s, "elbo": r["elbo"], "iterations": r["iterations"]}
        for th in (1, os.cpu_count()):
            with threadpool_limits(th):
                e, s = best(lambda: newton(y, backend="numpy", dtype="float64")[0])
            res[f"newton_cpu_{th}threads"] = {"seconds": s, "elbo": float(e.elbo[0]), "iterations": int(e.iterations[0])}
        e, s = best(lambda: newton(y, dtype="float64")[0])
        res["newton_gpu"] = {"seconds": s, "elbo": float(e.elbo[0]), "iterations": int(e.iterations[0])}
        if C["args"].jax:
            jaxfit(y, pool=1)  # compile
            (e, s, _), _s = best(lambda: jaxfit(y, pool=1), reps=1)
            res["jax_gpu"] = {"seconds": s, "elbo": float(e.elbo[0]), "iterations": int(e.iterations[0])}
        out[name] = res
        print("single", name, json.dumps({k: round(v["seconds"], 3) for k, v in res.items()}), flush=True)
    return out


def t_robust():
    a = C["args"]
    r = C["select_rows"](C["args"].robust_rows, C["saved"])
    Y = C["X"][r]
    out = {"rows": len(r)}
    for tau, c in [(0.05, 70), (0.1, 70), (0.2, 70), (0.3, 70), (0.1, 40), (0.1, 20)]:
        W = C["W"][:, :c]
        key = f"tau{tau}_rank{c}"
        row = {}
        en, sn = newton(Y, W=W, tau=tau, dtype="float64")
        row["newton"] = brief(en, sn, messages=C["pd"].Series(en.message).value_counts().to_dict())
        if a.jax:
            ej, sj, comp = jaxfit(Y, W=W, tau=tau, polish=a.robust_polish, pool=256)
            row["jax_polish"] = brief(ej, sj, compile_seconds=comp, polish=a.robust_polish)
            row["newton_vs_jax"] = compare(en, ej, "jax_polish")
        if a.cpu:
            es, ss, w = scipy_pool(Y, tau=tau, c=c, polish=a.robust_polish)
            row["scipy_polish"] = brief(es, ss, workers=w, polish=a.robust_polish)
            row["newton_vs_scipy"] = compare(en, es, "scipy_polish")
        out[key] = row
        print("robust", key, json.dumps({k: (v.get("seconds") if "seconds" in v else v.get("elbo_diff")) for k, v in row.items()}, default=str)[:600], flush=True)
    return out


def t_multistart():
    a = C["args"]
    r = C["select_rows"](C["args"].multistart_rows, C["saved"])
    S = 8
    c = C["W"].shape[1]
    rng = np.random.default_rng(7)
    starts = np.empty((len(r), S, 2 * c))
    starts[:, 0] = np.r_[np.zeros(c), np.ones(c)]
    starts[:, 1:, :c] = rng.normal(0, 1, (len(r), S - 1, c))
    starts[:, 1:, c:] = np.exp(rng.uniform(np.log(1e-3), np.log(3), (len(r), S - 1, c)))
    Y = np.repeat(C["X"][r], S, axis=0)
    x0 = starts.reshape(-1, 2 * c)
    out = {"rows": len(r), "starts": S}

    def spread(e, label):
        el = e.elbo.reshape(len(r), S)
        sg = e.sigma.reshape(len(r), S)
        okm = np.asarray(e.success, bool).reshape(len(r), S)
        el = np.where(okm, el, np.nan)
        best = np.nanmax(el, axis=1)
        tol = 1e-6 * np.abs(best)
        hit = np.abs(el - best[:, None]) <= tol[:, None]
        n_opt = [len(np.unique(np.round(sg[i][okm[i]] / sg[i][okm[i]].max(), 4))) for i in range(len(r))]
        return {"method": label, "converged": int(okm.sum()), "of": int(okm.size),
                "spread_max_minus_min": q(np.nanmax(el, 1) - np.nanmin(el, 1)),
                "default_start_reaches_best": int(np.sum(hit[:, 0])),
                "starts_reaching_best": q(hit.sum(1), (0.0, 0.5, 1.0)),
                "spectra_all_starts_agree": int(np.sum(hit.all(1))),
                "distinct_sigma_values_per_spectrum": q(n_opt, (0.5, 0.9, 1.0))}

    en, sn = newton(Y, x0=x0, dtype="float64")
    out["newton"] = {**spread(en, "newton"), "seconds": sn}
    if a.jax:
        ej, sj, comp = jaxfit(Y, polish=5, x0=x0, pool=1024)
        out["jax_polish5"] = {**spread(ej, "jax_polish5"), "seconds": sj}
        bn = np.nanmax(np.where(np.asarray(en.success).reshape(len(r), S), en.elbo.reshape(len(r), S), np.nan), 1)
        bj = np.nanmax(np.where(np.asarray(ej.success).reshape(len(r), S), ej.elbo.reshape(len(r), S), np.nan), 1)
        out["best_newton_minus_best_jax"] = q(bn - bj)
    print("multistart", json.dumps(out, default=float)[:1500], flush=True)
    return out


def t_warm():
    a = C["args"]
    r = C["select_rows"](a.warm_rows, C["saved"])
    Y = C["X"][r]
    rng = np.random.default_rng(11)
    cases = {"tau_0.11": (Y, 0.11), "noise_1e-4": (Y + rng.normal(0, 1e-4, Y.shape), 0.1)}
    out = {"rows": len(r)}
    en0, s = newton(Y)
    out["newton_base"] = brief(en0, s)
    xn = np.c_[en0.nu, en0.d]
    if a.jax:
        ej0, s, comp = jaxfit(Y)
        out["jax_base"] = brief(ej0, s, compile_seconds=comp)
        xj = np.c_[ej0.nu, ej0.d]
    for name, (Yc, tau) in cases.items():
        row = {}
        ec, sc = newton(Yc, tau=tau)
        ew, sw = newton(Yc, tau=tau, x0=xn)
        row["newton_cold"], row["newton_warm"] = brief(ec, sc), brief(ew, sw)
        row["newton_warm_vs_cold"] = compare(ew, ec, "newton_cold")
        if a.jax:
            jc, sjc, _ = jaxfit(Yc, tau=tau)
            jw, sjw, _ = jaxfit(Yc, tau=tau, x0=xj)
            row["jax_cold"], row["jax_warm"] = brief(jc, sjc), brief(jw, sjw)
            row["jax_warm_vs_cold"] = compare(jw, jc, "jax_cold")
            row["newton_vs_jax_warm"] = compare(ew, jw, "jax_warm")
        out[name] = row
        print("warm", name, json.dumps({k: v.get("seconds") for k, v in row.items() if "seconds" in v}), flush=True)
    return out


def t_tune():
    a = C["args"]
    Y = C["X"]
    out = {"newton": {}, "jax": {}}
    for dt in a.tune_dtypes.split(","):
        for ch in [int(x) for x in a.tune_chunks.split(",")]:
            gpu_mem_reset()
            e, s = newton(Y, dtype=dt, chunk=ch)
            out["newton"][f"{dt}_chunk{ch}"] = brief(e, s, **gpu_mem())
            print("tune newton", dt, ch, round(s, 2), gpu_mem(), flush=True)
    if a.jax:
        for spec in a.tune_jax.split(","):
            if not spec:
                continue
            pool, ci = (int(x) for x in spec.split(":"))
            e, s, comp = jaxfit(Y, pool=pool, chunk_iters=ci)
            out["jax"][f"pool{pool}_chunk{ci}"] = brief(e, s, compile_seconds=comp, **gpu_mem(),
                                                         device_iterations=e.stats["device_iterations"],
                                                         host_seconds=e.stats["host_seconds"])
            print("tune jax", spec, round(s, 1), round(comp, 1), flush=True)
    return out


TESTS = {"scale": t_scale, "single": t_single, "robust": t_robust, "multistart": t_multistart,
         "warm": t_warm, "tune": t_tune}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tests", default="scale,single,robust,multistart,warm,tune")
    ap.add_argument("--backend", choices=["numpy", "cupy"], default="cupy")
    ap.add_argument("--hess-dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--newton-chunk", type=int, default=2048)
    ap.add_argument("--no-jax", dest="jax", action="store_false")
    ap.add_argument("--no-cpu", dest="cpu", action="store_false")
    ap.add_argument("--cpu-workers", type=int, default=0)
    ap.add_argument("--sizes", default="1,16,128,1024,4452,17808")
    ap.add_argument("--jax-max-n", type=int, default=17808)
    ap.add_argument("--scipy-max-n", type=int, default=1024)
    ap.add_argument("--robust-polish", type=int, default=5)
    ap.add_argument("--warm-rows", default="all")
    ap.add_argument("--robust-rows", default="sample:256")
    ap.add_argument("--multistart-rows", default="sample:128")
    ap.add_argument("--tune-chunks", default="256,512,1024,2048,4096")
    ap.add_argument("--tune-dtypes", default="float64,float32")
    ap.add_argument("--tune-jax", default="2048:25,2048:400,1024:100,4096:100")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup(args)
    (out / "machine.json").write_text(json.dumps({"machine": C["machine"], "args": vars(args)}, indent=1, default=str))
    for name in args.tests.split(","):
        t = time.perf_counter()
        try:
            res = TESTS[name]()
            res["test_seconds"] = time.perf_counter() - t
            (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
            print(f"== {name} done in {time.perf_counter() - t:.0f}s", flush=True)
        except Exception:  # keep going; the log has the traceback
            traceback.print_exc()
            print(f"== {name} FAILED", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
