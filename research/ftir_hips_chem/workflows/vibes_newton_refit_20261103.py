"""Refit every calibration and target spectrum with the batched Newton ELBO (3 Nov 2026).

Same inputs, background (87 blanks, rank 70) and tau as production VIBES; only the ELBO optimiser
changes (vibes_batch.fit_elbo_newton instead of SciPy L-BFGS-B), followed by the certified batched
MAP. Reads the raw spectra of the 22 Oct decision bundle, writes corrected spectra per group.

    python vibes_newton_refit_20261103.py fit --bundle DECISION_BUNDLE --background background_model.npz --out DIR
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import vibes_batch as vb  # noqa: E402

GROUPS = ("improve", "addis", "network")


def fit(a):
    bundle, out = Path(a.bundle), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = np.load(a.background)
    mu, W = m["mean"].astype(float), m["components"].astype(float)
    receipt = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "backend": a.backend, "groups": {}}
    for g in GROUPS:
        X = np.asarray(np.load(bundle / f"{g}_raw.npy"), dtype=np.float64)
        t = time.perf_counter()
        e = vb.fit_elbo_newton(X, mu, W, tau=0.1, backend=a.backend, chunk=a.chunk)
        te = time.perf_counter() - t
        t = time.perf_counter()
        good = e.success
        r = vb.solve_map_batch(X[good], mu, W, sigma=e.sigma[good], tau=0.1, backend=a.backend, chunk=a.map_chunk)
        tm = time.perf_counter() - t
        base = np.full(X.shape, np.nan)
        base[good] = r.baseline
        np.save(out / f"corrected_newton_{g}.npy", (X - base).astype(np.float32))
        pd.DataFrame({"row": np.arange(len(X)), "sigma": e.sigma, "elbo": e.elbo, "iterations": e.iterations,
                      "success": e.success, "message": e.message, "map_certified": np.isin(
                          np.arange(len(X)), np.flatnonzero(good)[r.converged])}).to_csv(out / f"fit_{g}.csv", index=False)
        receipt["groups"][g] = {"rows": len(X), "elbo_seconds": te, "map_seconds": tm, "elbo_success": int(good.sum()),
                                "map_certified": int(r.converged.sum())}
        print(g, json.dumps(receipt["groups"][g]), flush=True)
    (out / "fit_receipt.json").write_text(json.dumps(receipt, indent=1))


def anchorhp(a):
    """Newton + MAP on the PTFE-anchored spectra with the high-passed background of vibes_continuum_20261102."""
    from scipy.signal import savgol_filter

    vc, out = Path(a.continuum) / "bundle", Path(a.out)
    z = np.load(vc / "background_model.npz")
    W = z["components"].astype(float)
    Whp = W - savgol_filter(W.T, 41, 2, axis=1).T
    mean = np.zeros_like(z["mean"], dtype=float)
    rows = pd.read_csv(vc / "rows.csv")
    anc = np.load(vc / "anchored.npy", mmap_mode="r")
    for g in GROUPS:
        r = rows[rows.group.eq(g)].sort_values("group_row")
        X = np.asarray(anc[r.row.to_numpy()], dtype=np.float64)
        t = time.perf_counter()
        e = vb.fit_elbo_newton(X, mean, Whp, tau=0.1, backend=a.backend, chunk=a.chunk)
        m = vb.solve_map_batch(X, mean, Whp, sigma=e.sigma, tau=0.1, backend=a.backend, chunk=a.map_chunk)
        np.save(out / f"corrected_newtonanchor_{g}.npy", (X - m.baseline).astype(np.float32))
        pd.DataFrame({"sigma": e.sigma, "elbo": e.elbo, "iterations": e.iterations, "success": e.success,
                      "map_certified": m.converged}).to_csv(out / f"fit_newtonanchor_{g}.csv", index=False)
        print(g, len(X), f"{time.perf_counter() - t:.0f}s ok {int(e.success.sum())} certified {int(m.converged.sum())}", flush=True)


def transfer(a):
    """The transfer-diagnosis scoring (run_transfer_diagnosis_20261102) with Newton corrected spectra as extra models.

    That workflow's model list is fixed, so its two slots for the continuum variants carry the Newton arrays here;
    they are renamed in this stage's outputs."""
    area, out = Path(a.area), Path(a.out)
    sys.path[:0] = [str(area / "workflows"), str(area / "scripts"), str(area.parents[1])]
    import run_transfer_diagnosis_20261102 as rtd

    dest = out / "transfer"
    dest.mkdir(exist_ok=True)
    rtd.OUT, rtd.VC = dest, out
    rtd.VC_VARIANTS = {"vibes_deflate": "newton", "vibes_anchorhp": "newtonanchor"}
    bundle = Path(a.bundle)
    rtd.fit(bundle)
    rtd.tables(bundle)
    rename = {"vibes_deflate": "vibes_newton", "vibes_anchorhp": "anchorhp_newton"}
    for f in dest.glob("*.csv"):
        d = f.read_text()
        for k, v in rename.items():
            d = d.replace(k, v)
        f.write_text(d)


def inject(a):
    """The continuum injection test of run_transfer_diagnosis_20261102 with Newton behind vibes_baseline_matrix."""
    area, out = Path(a.area), Path(a.out)
    sys.path[:0] = [str(area / "workflows"), str(area / "scripts"), str(area.parents[1])]
    import run_transfer_diagnosis_20261102 as rtd
    import vibes_baseline

    def newton_matrix(wn, spectra, bg, *, sample_ids, tau, loss, maxiter):
        Y = np.asarray(spectra, float)
        e = vb.fit_elbo_newton(Y, bg.mean, bg.components, tau=tau)
        m = vb.solve_map_batch(Y, bg.mean, bg.components, sigma=e.sigma, tau=tau)
        return m.baseline, Y - m.baseline, None

    vibes_baseline.vibes_baseline_matrix = newton_matrix
    dest = out / "inject"
    dest.mkdir(exist_ok=True)
    rtd.OUT = dest
    rtd.inject(Path(a.bundle), workers=1)
    new = pd.read_csv(dest / "vibes_injection.csv")
    old = pd.read_csv(area / "output/tables/transfer_diagnosis_20261102/vibes_injection.csv")
    t = (old.groupby(["variant", "shape"]).value.median().rename("saved").to_frame()
         .join(new.groupby(["variant", "shape"]).value.median().rename("newton")))
    t.to_csv(dest / "summary.csv")
    print(t.round(5).to_string())


BANDS = {"O-H / N-H 3000-3600": (3600, 3000), "C-H 2800-3000": (3000, 2800), "carbonyl 1650-1800": (1800, 1650),
         "1600 band 1500-1650": (1650, 1500), "fingerprint 1250-1500": (1500, 1250), "Si-O 980-1100": (1100, 980)}


def bands(a):
    """Band areas of the corrected spectra (the functional-group windows of run_meeting_20261028): saved vs Newton."""
    bundle, out = Path(a.bundle), Path(a.out)
    wn = np.load(bundle / "wn_full.npy")
    rows = []
    per = {}
    for g in GROUPS:
        S = np.asarray(np.load(bundle / f"{g}_vibes.npy"), float)
        N = np.asarray(np.load(out / f"corrected_newton_{g}.npy"), float)
        for name, (hi, lo) in BANDS.items():
            m = (wn <= hi) & (wn >= lo)
            o = np.argsort(wn[m])
            A = np.trapezoid(S[:, m][:, o], wn[m][o], axis=1)
            B = np.trapezoid(N[:, m][:, o], wn[m][o], axis=1)
            ok = np.isfinite(A) & np.isfinite(B)
            rel = np.abs(B - A) / np.maximum(np.abs(A), np.median(np.abs(A[ok])) * 0.1)
            rows.append({"group": g, "band": name, "n": int(ok.sum()), "median_area": float(np.median(A[ok])),
                         "median_change": float(np.median(rel[ok])), "p99_change": float(np.quantile(rel[ok], .99)),
                         "gt5pct": int(np.sum(rel[ok] > .05)), "gt20pct": int(np.sum(rel[ok] > .2)),
                         "r": float(np.corrcoef(A[ok], B[ok])[0, 1])})
            per[f"{g}|{name}"] = np.c_[A, B]
    d = pd.DataFrame(rows)
    d.to_csv(out / "band_areas.csv", index=False)
    np.savez_compressed(out / "band_areas_per_spectrum.npz", **per)
    print(d.round(4).to_string(index=False))


def lots(a):
    """Every job of vibes_lot241_20261026 (blank residue and four-peak spikes across 23 banks) with Newton,
    then that workflow's own ``score`` on a mirror of its output folder."""
    area, out = Path(a.area), Path(a.out)
    sys.path[:0] = [str(area / "workflows"), str(area / "scripts")]
    import vibes_lot241_20261026 as lot

    src = area / "output/tables/vibes_lot241_20261026"
    dest = out / "lots"
    dest.mkdir(exist_ok=True)
    link = dest / "bundle"
    if not link.exists():
        link.symlink_to(src / "bundle")
    jobs = pd.read_csv(src / "bundle/jobs.csv")
    wn = np.load(src / "bundle/wn.npy")
    S, P = np.load(src / "bundle/spartan_blanks.npy"), np.load(src / "bundle/pm25.npy")
    shape = lot.spike_shape(wn)
    bgs = {f.stem: f for f in sorted(src.glob("results/*/run/backgrounds/*.npz"))}
    base = np.full((len(jobs), len(wn)), np.nan, np.float32)
    ok = np.zeros(len(jobs), bool)
    its = np.zeros(len(jobs), np.int64)
    sig = np.full(len(jobs), np.nan)
    t = time.perf_counter()
    for bank, g in jobs.groupby("bank"):
        z = np.load(bgs[bank])
        Y = np.stack([(S[r] if k == "blank" else P[r]) + am * shape for k, r, am in zip(g.kind, g.ref, g.amplitude)]).astype(float)
        e = vb.fit_elbo_newton(Y, z["mean"].astype(float), z["components"].astype(float), tau=0.1, chunk=512)
        m = vb.solve_map_batch(Y[e.success], z["mean"].astype(float), z["components"].astype(float), sigma=e.sigma[e.success],
                               tau=0.1, chunk=256)
        b = np.full(Y.shape, np.nan)
        b[e.success] = m.baseline
        idx = g.job.to_numpy()
        base[idx], its[idx], sig[idx] = b, e.iterations, e.sigma
        ok[idx] = np.isfinite(b).all(axis=1)
        print(bank, len(g), f"{time.perf_counter() - t:.0f}s ok {int(ok[idx].sum())}", flush=True)
    np.save(dest / "baselines.npy", base)
    pd.DataFrame({"job": jobs.job, "success": ok, "iterations": its, "sigma": sig}).to_csv(dest / "fit_diagnostics.csv", index=False)
    for f in ("backgrounds.csv",):
        (dest / f).write_bytes((src / f).read_bytes())
    lot.OUT = dest
    lot.score(argparse.Namespace())


def banks(a):
    """The four blank-bank re-corrections of 7 Oct with Newton, mirrored so run_oct1_remaining_20261007's own
    ``blanks`` stage can score them; the capped 87-blank arrays stay the saved ones."""
    area, out = Path(a.area), Path(a.out)
    sys.path[:0] = [str(area / "workflows"), str(area / "scripts"), str(area.parents[1])]
    from vibes_large_run import RunConfig, prepare_experiment

    drive, mirror = Path(a.blankbank), out / "blankbank_mirror"
    stage = area / "output/tables/vibes_colab_bundle_full500/stage"
    X = np.asarray(prepare_experiment(stage, RunConfig(profile="full", max_background_components=30))[1], dtype=np.float64)
    for policy in ("lot251_all", "all_labeled", "lot251_all_minus_LABE1", "all_labeled_minus_LABE1"):
        run = sorted((drive / policy / "persistent_results").glob("*/RUN_MANIFEST.json"))[-1].parent
        dest = mirror / policy / "persistent_results" / run.name
        dest.mkdir(parents=True, exist_ok=True)
        saved = np.load(run / "corrected_VIBES_full.npy")
        use = np.isfinite(saved).all(axis=1)
        z = np.load(run / "background_model.npz")
        mean, W = z["mean"].astype(float), z["components"].astype(float)
        t = time.perf_counter()
        e = vb.fit_elbo_newton(X[use], mean, W, tau=0.1, chunk=512)
        m = vb.solve_map_batch(X[use][e.success], mean, W, sigma=e.sigma[e.success], tau=0.1, chunk=256)
        corr = np.full(X.shape, np.nan, np.float32)
        idx = np.flatnonzero(use)[e.success]
        corr[idx] = (X[idx] - m.baseline).astype(np.float32)
        np.save(dest / "corrected_VIBES_full.npy", corr)
        for f in run.iterdir():
            if f.is_file() and f.name != "corrected_VIBES_full.npy":
                (dest / f.name).write_bytes(f.read_bytes())
        ch = np.sqrt(np.nanmean((corr[use] - saved[use]) ** 2, axis=1)) / np.sqrt(np.nanmean(saved[use] ** 2, axis=1))
        print(policy, "rank", W.shape[1], f"{time.perf_counter() - t:.0f}s", "ok", int(e.success.sum()), "/", int(use.sum()),
              "certified", int(m.converged.sum()), "|rel change| median %.2e, >5%% %d, >20%% %d" % (np.nanmedian(ch), np.nansum(ch > .05), np.nansum(ch > .2)), flush=True)
    import run_oct1_remaining_20261007 as oc

    oc.BLANKBANK, oc.OUT = mirror, out / "banks"
    oc.OUT.mkdir(exist_ok=True)
    sys.argv = ["oct1", "--stages", "blanks"]
    oc.main()


def publish(a):
    """network_baselines v2: the 19 Oct network with Newton, next to (never over) the frozen run.

    Same inputs and background; lot 241/241a analyses use the banks of vibes_lot241_20261026/apply as in the gallery.
    Every analysis carries the premature-stop flag and how far it moved from the 19 Oct result."""
    import hashlib

    area, out = Path(a.area), Path(a.out)
    src = area / "output/tables/network_baselines_20261019"
    dest = out
    dest.mkdir(parents=True, exist_ok=True)
    X = np.asarray(np.load(src / "vibes_bundle/X_model_grid.npy"), dtype=np.float64)
    cases = pd.read_csv(src / "vibes_bundle/cases.csv", dtype={"analysis_id": str, "media_id": str})
    z = np.load(src / "vibes_bundle/background_model.npz")
    mean, W = z["mean"].astype(float), z["components"].astype(float)
    saved = pd.read_csv(src / "vibes_fit_diagnostics.csv")
    saved_b = np.load(src / "vibes_baselines_model_grid.npy").astype(float)
    over = pd.read_csv(area / "output/tables/vibes_lot241_20261026/apply/cases.csv")
    bank_of = dict(zip(over.case_row, over.bank))
    groups = {"b87": [r for r in range(len(X)) if r not in bank_of]}
    for r, b in bank_of.items():
        groups.setdefault(b, []).append(int(r))
    base = np.full(X.shape, np.nan)
    d = pd.DataFrame({"case_row": np.arange(len(X)), "analysis_id": cases.analysis_id, "filter_id": cases.filter_id,
                      "filter_key": cases.filter_key, "code": cases.code, "filter_type": cases.filter_type})
    for c_ in ("vibes_model", "sigma", "elbo", "iterations", "success", "map_certified", "message"):
        d[c_] = None
    for bank, rows in groups.items():
        if bank == "b87":
            m_, W_ = mean, W
        else:
            f = next((area / "output/tables/vibes_lot241_20261026/results").glob(f"*/run/backgrounds/{bank}.npz"))
            zb = np.load(f)
            m_, W_ = zb["mean"].astype(float), zb["components"].astype(float)
        rows = np.asarray(rows)
        e = vb.fit_elbo_newton(X[rows], m_, W_, tau=0.1, chunk=512)
        r = vb.solve_map_batch(X[rows], m_, W_, sigma=e.sigma, tau=0.1, chunk=256)
        base[rows] = r.baseline
        d.loc[rows, "vibes_model"] = bank
        d.loc[rows, "sigma"], d.loc[rows, "elbo"], d.loc[rows, "iterations"] = e.sigma, e.elbo, e.iterations
        d.loc[rows, "success"], d.loc[rows, "map_certified"] = e.success, r.converged
        d.loc[rows, "message"] = e.message
        print(bank, len(rows), "ok", int(e.success.sum()), "certified", int(r.converged.sum()), flush=True)
    corr_rms = np.sqrt(np.nanmean((X - saved_b) ** 2, axis=1))
    d["saved_iterations"] = saved.iterations.to_numpy()
    d["saved_sigma"], d["saved_elbo"], d["saved_success"] = saved.sigma.to_numpy(), saved.elbo.to_numpy(), saved.success.to_numpy()
    d["premature_stop"] = saved.success.to_numpy(bool) & (saved.iterations.to_numpy() < 30)
    b87 = d.vibes_model.eq("b87").to_numpy()
    d["elbo_gain"] = np.where(b87, d.elbo.astype(float) - d.saved_elbo, np.nan)
    d["baseline_change_frac_rms"] = np.sqrt(np.nanmean((base - saved_b) ** 2, axis=1)) / corr_rms
    zv = np.load(a.pls)
    frozen = area / "output/tables/vibes_fullrange/interpretation_20261002/benchmark/pls_full_pool_uncapped_full.npz"
    zf = np.load(frozen)
    ec = lambda z_, b: (X - b - z_["x_mean"].ravel()) @ z_["coefficient"].ravel() + float(z_["intercept"].ravel()[0])
    d["ec_saved_baseline_frozen_model_ug"] = ec(zf, saved_b)
    d["ec_newton_baseline_frozen_model_ug"] = ec(zf, base)
    d["ec_newton_baseline_rebuilt_model_ug"] = ec(zv, base)
    d.to_csv(dest / "vibes_fit_diagnostics.csv", index=False)
    np.save(dest / "vibes_baselines_model_grid.npy", base.astype(np.float32))
    d[d.premature_stop | (d.baseline_change_frac_rms > 0.05)].to_csv(dest / "changed_or_premature.csv", index=False)
    per = d[d.filter_type.eq("PM2.5")].groupby(["filter_key", "code"], as_index=False).agg(
        analyses=("case_row", "size"), premature_stop=("premature_stop", "max"),
        ec_saved_ug=("ec_saved_baseline_frozen_model_ug", "mean"), ec_newton_ug=("ec_newton_baseline_rebuilt_model_ug", "mean"))
    per.rename(columns={"filter_key": "id"}).to_csv(dest / "network_ec.csv", index=False)
    (dest / "MANIFEST.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "supersedes": None,
        "sits_next_to": "network_baselines_20261019 (unchanged)",
        "method": "same objective, background, tau and inputs; ELBO solved to convergence by vibes_batch.fit_elbo_newton",
        "vibes_batch_sha256": hashlib.sha256(Path(vb.__file__).read_bytes()).hexdigest(),
        "analyses": len(d), "elbo_success": int(d.success.astype(bool).sum()), "map_certified": int(d.map_certified.astype(bool).sum()),
        "premature_stop_flagged": int(d.premature_stop.sum()), "pls_model": str(a.pls), "lot241_overrides": len(over)}, indent=1))
    print("wrote", dest, flush=True)


def calibration(a):
    """Rebuild the frozen all-IMPROVE VIBES PLS model with the procedure of vibes_large_run.compare_calibrations
    (site-grouped 5-fold curve, first major minimum, fit on the train split), first on the saved baselines as a
    check that the frozen model comes back, then on the Newton baselines."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import r2_score
    from pls_transfer import component_cv_curve, select_first_major_minimum

    out, area = Path(a.out), Path(a.area)
    bench = area / "output/tables/vibes_fullrange/interpretation_20261002"
    cases = pd.read_csv(bench / "cases.csv")
    cases = cases[cases.kind.eq("calibration")].reset_index(drop=True)
    train, test = cases.split.eq("train").to_numpy(), cases.split.eq("test").to_numpy()
    y = cases.y.to_numpy(float)
    arrays = {"saved": np.asarray(np.load(Path(a.bundle) / "improve_vibes.npy"), float),
              "newton": np.asarray(np.load(out / "corrected_newton_improve.npy"), float)}
    frozen = np.load(bench / "benchmark/pls_full_pool_uncapped_full.npz")
    rows, curves = [], []
    for name, X in arrays.items():
        t = time.perf_counter()
        curve = component_cv_curve(X[train], y[train], range(1, 31), groups=cases.Site[train], n_splits=5, random_state=42)
        k, _ = select_first_major_minimum(curve)
        k = int(k)
        curves.append(curve.assign(method=name, selected_k=k))
        for kk in sorted({k, 4}):
            model = PLSRegression(n_components=kk, scale=False, max_iter=1000, tol=1e-10).fit(X[train], y[train])
            pred = model.predict(X[test]).ravel()
            rows.append({"method": name, "k": kk, "selected": kk == k, "predictive_R2": r2_score(y[test], pred),
                         "RMSE": float(np.sqrt(np.mean((pred - y[test]) ** 2))), "MAE": float(np.mean(np.abs(pred - y[test]))),
                         "bias": float(np.mean(pred - y[test])), "n_train": int(train.sum()), "n_test": int(test.sum())})
            np.savez_compressed(out / f"pls_full_pool_{name}_k{kk}.npz", coefficient=model.coef_, x_mean=model._x_mean,
                                y_mean=model._y_mean, intercept=model.intercept_, components=np.array(kk))
            if name == "saved" and kk == int(frozen["components"]):
                print("frozen coefficient reproduced to", float(np.abs(model.coef_.ravel() - frozen["coefficient"].ravel()).max()),
                      "of", float(np.abs(frozen["coefficient"]).max()), flush=True)
        print(name, "k", k, f"{time.perf_counter() - t:.0f}s", flush=True)
    pd.concat(curves).to_csv(out / "calibration_cv_curves.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "calibration_scores.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--bundle", required=True)
    p.add_argument("--background", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", default="numpy", choices=["numpy", "cupy"])
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--map-chunk", type=int, default=256)
    h = sub.add_parser("anchorhp")
    h.add_argument("--continuum", required=True, help="output/tables/vibes_continuum_20261102")
    h.add_argument("--out", required=True)
    h.add_argument("--backend", default="numpy", choices=["numpy", "cupy"])
    h.add_argument("--chunk", type=int, default=512)
    h.add_argument("--map-chunk", type=int, default=256)
    x = sub.add_parser("transfer")
    x.add_argument("--bundle", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--area", required=True)
    b = sub.add_parser("bands")
    b.add_argument("--bundle", required=True)
    b.add_argument("--out", required=True)
    j = sub.add_parser("inject")
    j.add_argument("--bundle", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--area", required=True)
    for nm in ("lots", "banks"):
        q = sub.add_parser(nm)
        q.add_argument("--out", required=True)
        q.add_argument("--area", required=True)
        q.add_argument("--blankbank", help="Drive results/blankbank_20261007")
    u = sub.add_parser("publish")
    u.add_argument("--out", required=True)
    u.add_argument("--area", required=True)
    u.add_argument("--pls", required=True)
    c = sub.add_parser("calibration")
    c.add_argument("--bundle", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--area", required=True, help="research/ftir_hips_chem of the checkout holding output/")
    a = ap.parse_args()
    {"fit": fit, "anchorhp": anchorhp, "transfer": transfer, "bands": bands, "inject": inject, "lots": lots, "banks": banks, "publish": publish,
     "calibration": calibration}[a.stage](a)
