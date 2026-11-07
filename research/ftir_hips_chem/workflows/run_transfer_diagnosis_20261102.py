"""Why FTIR EC does not transfer to Addis (and which sites, filters and bands show it), 2 Nov 2026.

Reads the 22 Oct decision bundle (IMPROVE 12,393 + Addis 233 + SPARTAN network 3,517 filters,
each as raw / spline / VIBES spectra) and writes tables for ``transfer_diagnosis_20261102.ipynb``.

Stages (default: all but ``inject``)

* ``fit``     site-grouped 5-fold out-of-site PLS (k 1-25) on IMPROVE TOR EC for raw, spline, VIBES
              and the PTFE-anchored background (scripts/continuum_anchor.py), plus anchored variants
              (C-H window masked; OC/EC-balanced weights). Full-pool models predict Addis and the network.
* ``tables``  per-site readouts, loading bins, continuum vs loading, window attribution, Addis residuals.
* ``inject``  injects smooth continua into 16 light IMPROVE filters and reruns real VIBES (the saved
              87-blank rank-70 background, tau 0.1, PB) with the original and smooth-deflated component sets
              (~5 min on 12 cores).

    ../../.venv/bin/python workflows/run_transfer_diagnosis_20261102.py [--bundle DIR] [--stages fit tables inject]

The bundle defaults to ``<Drive>/Aethmodular VIBES/decision_workbook_20261022/bundle``; copy it to local disk
first if Drive reads stall (see the decision-bundle builder). Nothing here is tuned on Addis or SPARTAN HIPS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path[:0] = [str(AREA / "scripts"), str(ROOT)]
import continuum_anchor as ca  # noqa: E402
import data_paths  # noqa: E402
from candidate_metrics import ids_sha256, sha256_file, validate_physical_ids  # noqa: E402
from spectral_provenance import verify_corrected_spectra, verify_manifest_files  # noqa: E402
from aethmodular_cli import env  # noqa: E402

OUT = AREA / "output/tables/transfer_diagnosis_20261102"
VIBES_MODEL = AREA / "output/tables/network_baselines_20261019/vibes_bundle/background_model.npz"
# continuum-protected VIBES reruns (workflows/vibes_continuum_20261102.py, Colab); used when present
VC = AREA / "output/tables/vibes_continuum_20261102"
VC_VARIANTS = {"vibes_deflate": "deflate", "vibes_anchorhp": "anchorhp"}


def vc_spectra(variant):
    files = {g: VC / f"corrected_{variant}_{g}.npy" for g in ("improve", "addis", "network")}
    return {g: np.load(f, mmap_mode="r") for g, f in files.items()} if all(f.exists() for f in files.values()) else None
K = 25
LOAD_BINS = [0, 2, 5, 10, 20, 30, 45, 90]
ADDIS_BINS = [0, 30, 45, 90]
CH = (2790, 3010)
WINDOW_EDGES = [4000, 3700, 3400, 3100, 3000, 2800, 2600, 2400, 2300, 2000, 1800, 1700, 1600, 1500, 1425,
                1300, 1250, 1100, 1000, 900, 800, 700, 600, 500]


def default_bundle() -> Path:
    return data_paths.vibes_colab_results_dir().parent / "decision_workbook_20261022/bundle"


def load(bundle: Path):
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    for name in ("improve_meta.csv", "addis_meta.csv", "network_meta.csv"):
        if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"bundle file changed: {name}")
    im, ad, nw = (pd.read_csv(bundle / f"{g}_meta.csv") for g in ("improve", "addis", "network"))
    # everything per filter (µg); HIPS BC = Fabs x volume / MAC 10 (Mm-1 m3 / m2 g-1 = µg)
    im["hips_ug"] = im.fabs_Mm1 * im.volume_m3 / 10
    im["tor_ug"] = im.ec_ug
    im["ocec"] = im.tor_oc_ugm3 / im.tor_ec_ugm3
    for d in (ad, nw):
        d["hips_ug"] = d.fabs_Mm1 * d.volume_m3 / 10
    ad["pub_ug"] = ad.ec_published_ugm3 * ad.volume_m3
    nw["pub_ug"] = nw.ec_ftir_ugm3 * nw.volume_m3
    return im, ad, nw


def spectra(bundle: Path, prep: str):
    return {g: np.load(bundle / f"{g}_{prep}.npy", mmap_mode="r") for g in ("improve", "addis", "network")}


# --------------------------------------------------------------------------- fit
def fit(bundle: Path):
    from sklearn.model_selection import GroupKFold

    im, ad, nw = load(bundle)
    checked_inputs = {bundle / name: value for name, value in verify_manifest_files(
        bundle, ["wn_full.npy", "improve_meta.csv", "addis_meta.csv", "network_meta.csv", *[f"{group}_{prep}.npy" for group in ("improve", "addis", "network")
                                   for prep in ("raw", "spline", "vibes")]]).items()}
    checked_inputs[bundle / "MANIFEST.json"] = sha256_file(bundle / "MANIFEST.json")
    checked_inputs[VIBES_MODEL] = sha256_file(VIBES_MODEL)
    y, sites = im.tor_ug.to_numpy(), im.site.to_numpy()
    physical_ids = validate_physical_ids(im.filter_id.to_numpy())
    splits = list(GroupKFold(5).split(np.zeros((len(y), 1)), y, sites))
    fold_id = np.full(len(y), -1, dtype=int)
    for fold, (_, test) in enumerate(splits):
        fold_id[test] = fold
    # five IMPROVE wildfire filters with TOR 250-390 µg (HIPS ~50) dominate squared error; fit without TOR > 150
    ok = np.isfinite(y) & (y < 150)
    wn = np.load(bundle / "wn_full.npy")
    z = np.load(VIBES_MODEL)
    if not np.allclose(z["wn"], wn):
        raise ValueError("VIBES background grid differs from the bundle grid")
    bg = ca.AnchoredBackground(wn, z["mean"], z["components"])
    raw = spectra(bundle, "raw")
    anchored = {g: bg.correct(raw[g]) for g in raw}

    ocec = np.clip(im.ocec.to_numpy(), 0.1, 50)
    ocec[~np.isfinite(ocec)] = np.nanmedian(ocec)
    b1 = pd.qcut(np.log(ocec), 10, labels=False, duplicates="drop")
    b2 = pd.cut(np.log(np.clip(y, 0.05, None)), 6, labels=False)
    cell = pd.Series(1, index=pd.MultiIndex.from_arrays([b1, b2])).groupby(level=[0, 1]).transform("size").to_numpy()
    balanced = np.clip((1 / cell) / np.mean(1 / cell), 0, 20)
    not_ch = ~((wn > CH[0]) & (wn < CH[1]))

    models = {"raw": (raw, None, None), "spline": (spectra(bundle, "spline"), None, None),
              "vibes": (spectra(bundle, "vibes"), None, None), "anchored": (anchored, None, None),
              "anchored_noCH": (anchored, not_ch, None), "anchored_balanced": (anchored, None, balanced)}
    for name, v in VC_VARIANTS.items():
        X = vc_spectra(v)
        if X is not None:
            checked_inputs.update({VC / name: value for name, value in verify_corrected_spectra(
                VC, {"improve": im, "addis": ad, "network": nw}, wn, (v,)).items()})
            if any(np.isnan(np.asarray(a, float)).any() for a in X.values()):
                raise ValueError(f"{v}: corrected spectra contain NaN (failed VIBES fits)")
            models[name] = (X, None, None)
    preds = {"improve_filter_ids": physical_ids, "improve_site_ids": sites.astype(str), "improve_fold_id": fold_id}
    coefs = {}
    for name, (X, cols, w) in models.items():
        t = time.time()
        XI = np.asarray(X["improve"], np.float64)
        cols = np.ones(XI.shape[1], bool) if cols is None else cols
        XI = XI[:, cols]
        oof = np.full((len(y), K), np.nan)
        for tr, te in splits:
            tr = tr[ok[tr]]
            oof[te] = ca.pls_predict(ca.pls_fit(XI[tr], y[tr], K, None if w is None else w[tr]), XI[te])
        full = ca.pls_fit(XI[ok], y[ok], K, None if w is None else w[ok])
        preds[f"{name}|improve_oof"] = oof
        preds[f"{name}|addis"] = ca.pls_predict(full, np.asarray(X["addis"], float)[:, cols])
        preds[f"{name}|network"] = ca.pls_predict(full, np.asarray(X["network"], float)[:, cols])
        coefs[f"{name}|xm"], coefs[f"{name}|B"], coefs[f"{name}|cols"] = full[0], full[2], cols
        print(f"{name}: {time.time() - t:.0f}s", flush=True)
    if any(sha256_file(path) != value for path, value in checked_inputs.items()):
        raise RuntimeError("a spectral input changed during fitting; source predictions are not certified")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "predictions.npz", **preds)
    np.savez_compressed(OUT / "coefficients.npz", **coefs)
    for g, A in anchored.items():
        np.save(OUT / f"anchored_{g}.npy", A)
    (OUT / "fit_receipt.json").write_text(json.dumps({
        "bundle": env.display_path(bundle), "vibes_model": env.display_path(VIBES_MODEL), "k_max": K,
        "cv": "GroupKFold(5) by IMPROVE site", "fit_rows": int(ok.sum()), "excluded_tor_over_150": int((y >= 150).sum()),
        "spectral_inputs": {env.display_path(path): value for path, value in checked_inputs.items()},
        "source_identity": {
            "protocol": "physical_filter_order_and_site_folds_v1",
            "improve_metadata_sha256": sha256_file(bundle / "improve_meta.csv"),
            "filter_ids_sha256": ids_sha256(physical_ids),
            "predictions_sha256": sha256_file(OUT / "predictions.npz"),
            "split_population": "all bundled IMPROVE physical filters; training requires finite TOR <150 micrograms per filter",
        },
        "anchored": {"windows": ca.PTFE_WINDOWS, "highpass_window_channels": 41, "ridge": 1e-6},
        "balanced_weights": "inverse cell count over 10 OC/EC deciles x 6 log-EC bins, clipped at 20",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))


# --------------------------------------------------------------------------- tables
def _binmed(values, by, bins, mask=None):
    s = pd.Series(np.asarray(values, float))
    b = pd.cut(pd.Series(np.asarray(by, float)), bins)
    if mask is not None:
        s, b = s[mask], b[mask]
    g = s.groupby(b, observed=True)
    return pd.DataFrame({"median": g.median(), "n": g.size()})


def tables(bundle: Path):
    from sklearn.ensemble import HistGradientBoostingRegressor

    im, ad, nw = load(bundle)
    P = np.load(OUT / "predictions.npz")
    wn = np.load(bundle / "wn_full.npy")
    y = im.tor_ug.to_numpy()
    low = (im.ocec < 2.5).to_numpy()
    names = sorted({k.split("|")[0] for k in P.files}, key=lambda n: ["raw", "spline", "vibes", "anchored", "anchored_noCH",
                                                                       "anchored_balanced", "vibes_deflate", "vibes_anchorhp"].index(n))

    # 1. per-site published EC vs HIPS (µg per filter)
    ok = nw.fabs_Mm1.notna() & nw.ec_ftir_ugm3.notna() & ~nw.excluded
    rows = []
    for code, g in nw[ok].groupby("code"):
        s, i, r2, n = ca.ols(g.hips_ug, g.pub_ug)
        rows.append(dict(code=code, n=n, hips_ug_median=g.hips_ug.median(), hips_ug_p90=g.hips_ug.quantile(.9),
                         pub_over_hips=(g.pub_ug / g.hips_ug).median(), ols_slope=s, ols_intercept_ug=i,
                         intercept_rel=i / g.pub_ug.mean(), r2=r2,
                         oc_over_bc=(g.oc_ftir_ugm3 * g.volume_m3 / g.hips_ug).median()))
    pd.DataFrame(rows).sort_values("hips_ug_median").to_csv(OUT / "sites_published.csv", index=False)

    # 2. reference relation in IMPROVE: TOR / HIPS by loading and composition
    rel = {"all": None, "OC/EC<2.5": low, "OC/EC 2.5-5": ((im.ocec >= 2.5) & (im.ocec < 5)).to_numpy(),
           "OC/EC>=5": (im.ocec >= 5).to_numpy(), "OP>0": (im.tor_optr_ugm3 > 0).to_numpy(),
           "OP=0": (im.tor_optr_ugm3 <= 0).to_numpy()}
    t = []
    for nm, m in rel.items():
        d = _binmed(y / im.hips_ug, im.hips_ug, LOAD_BINS, m)
        d["subset"] = nm
        t.append(d)
    pd.concat(t).reset_index(names="hips_bin").assign(hips_bin=lambda d: d.hips_bin.astype(str)).to_csv(
        OUT / "improve_tor_over_hips.csv", index=False)

    # 3. model ratios by loading: IMPROVE (out of site) FTIR/TOR, FTIR/HIPS (all, low OC/EC); Addis FTIR/HIPS
    t = []
    for nm in names:
        for k in (5, 8, 12, 18):
            oof, pa = P[f"{nm}|improve_oof"][:, k - 1], P[f"{nm}|addis"][:, k - 1]
            for what, vals, by, bins, m in [("IMPROVE FTIR/TOR", oof / y, im.hips_ug, LOAD_BINS, None),
                                            ("IMPROVE FTIR/HIPS", oof / im.hips_ug, im.hips_ug, LOAD_BINS, None),
                                            ("IMPROVE low OC/EC FTIR/TOR", oof / y, im.hips_ug, LOAD_BINS, low),
                                            ("IMPROVE low OC/EC FTIR/HIPS", oof / im.hips_ug, im.hips_ug, LOAD_BINS, low),
                                            ("Addis FTIR/HIPS", pa / ad.hips_ug, ad.hips_ug, LOAD_BINS, None)]:
                d = _binmed(vals, by, bins, m)
                d["model"], d["k"], d["what"] = nm, k, what
                t.append(d)
            e = oof - y
            fitok = np.isfinite(y) & (y < 150)
            s, i, r2, n = ca.ols(ad.hips_ug, pa)
            t.append(pd.DataFrame([dict(median=np.nanmedian(np.abs(e[fitok])), n=int(fitok.sum()), model=nm, k=k,
                                        what="IMPROVE out-of-site median absolute error (µg/filter)")]))
            t.append(pd.DataFrame([dict(median=s, n=n, model=nm, k=k, what="Addis OLS slope vs HIPS"),
                                   dict(median=i, n=n, model=nm, k=k, what="Addis OLS intercept vs HIPS (µg)")]))
    pub = _binmed(ad.pub_ug / ad.hips_ug, ad.hips_ug, LOAD_BINS)
    pub["model"], pub["k"], pub["what"] = "published", 0, "Addis FTIR/HIPS"
    t.append(pub)
    out = pd.concat(t)
    out.index = out.index.astype(str)
    out.reset_index(names="hips_bin").to_csv(OUT / "ratios_by_loading.csv", index=False)

    # 4. continuum vs loading (raw, VIBES, spline, anchored), IMPROVE by lot and Addis
    raw, vib = spectra(bundle, "raw"), spectra(bundle, "vibes")
    spl, wns = spectra(bundle, "spline"), np.load(bundle / "wn_spline.npy")
    anc = {g: np.load(OUT / f"anchored_{g}.npy", mmap_mode="r") for g in ("improve", "addis")}
    t = []
    extra = [(n, vc_spectra(v), wn) for n, v in VC_VARIANTS.items() if vc_spectra(v) is not None]
    for prep, X, grid in [("raw", raw, wn), ("vibes", vib, wn), ("spline", spl, wns), ("anchored", anc, wn)] + extra:
        for wname, (lo, hi) in ca.CONTINUUM_WINDOWS.items():
            if not ((grid >= lo) & (grid <= hi)).any():
                continue
            vi, va = ca.window_mean(X["improve"], grid, lo, hi), ca.window_mean(X["addis"], grid, lo, hi)
            for grp, vals, by, m in [("IMPROVE lot 251", vi, im.hips_ug, (im.lot == 251).to_numpy()),
                                     ("IMPROVE lot 248", vi, im.hips_ug, (im.lot == 248).to_numpy()),
                                     ("Addis", va, ad.hips_ug, None)]:
                d = _binmed(vals, by, LOAD_BINS, m)
                d["prep"], d["window"], d["group"] = prep, wname, grp
                t.append(d)
    out = pd.concat(t)
    out.index = out.index.astype(str)
    out.reset_index(names="hips_bin").to_csv(OUT / "continuum_by_loading.csv", index=False)

    # 4b. continuum scatter on light filters in µg-EC equivalents (lot-mean subtraction vs anchored)
    light = (im.hips_ug < 1).to_numpy()
    rows = []
    for wname, (lo, hi) in ca.CONTINUUM_WINDOWS.items():
        vr, vaN = ca.window_mean(raw["improve"], wn, lo, hi), ca.window_mean(anc["improve"], wn, lo, hi)
        for lot in (251, 248):
            s = (im.lot == lot).to_numpy()
            slope = ca.ols(im.hips_ug[s & (im.hips_ug < 60)], vr[s & (im.hips_ug < 60)])[0]
            for prep, v in (("raw minus lot mean", vr), ("anchored", vaN)):
                x = v[s & light]
                mad = 1.4826 * np.median(np.abs(x - np.median(x)))
                rows.append(dict(window=wname, lot=lot, prep=prep, slope_per_ug=slope, mad=mad, noise_ug=mad / slope))
    pd.DataFrame(rows).to_csv(OUT / "continuum_noise.csv", index=False)

    # 5. window attribution: Addis minus matched IMPROVE (low OC/EC, HIPS 20-60 µg) contribution per HIPS µg, k 12
    match = low & (im.hips_ug > 20).to_numpy() & (im.hips_ug < 60).to_numpy() & np.isfinite(y) & (y < 150)
    C = np.load(OUT / "coefficients.npz")
    rows = []
    for nm, X, grid in [("raw", raw, wn), ("spline", spl, wns), ("vibes", vib, wn), ("anchored", anc, wn)] + extra:
        b, xm = C[f"{nm}|B"][:, 11], C[f"{nm}|xm"]
        ci = (np.asarray(X["improve"], float)[match] - xm) * b / im.hips_ug.to_numpy()[match, None]
        cA = (np.asarray(X["addis"], float) - xm) * b / ad.hips_ug.to_numpy()[:, None]
        for hi, lo in zip(WINDOW_EDGES[:-1], WINDOW_EDGES[1:]):
            m = (grid <= hi) & (grid > lo)
            if m.any():
                rows.append(dict(model=nm, window=f"{hi}-{lo}", hi=hi, lo=lo, addis=np.median(cA[:, m].sum(1)),
                                 improve_matched=np.median(ci[:, m].sum(1))))
    pd.DataFrame(rows).assign(difference=lambda d: d.addis - d.improve_matched).to_csv(OUT / "window_attribution.csv", index=False)
    pd.Series(match.sum(), index=["n"]).to_json(OUT / "window_attribution_n.json")

    # 6. per-site transfer vs the IMPROVE TOR yardstick (TOR/HIPS expected at the site's loading and OC/BC)
    im["lh"], nw["lh"] = np.log(im.hips_ug), np.log(nw.hips_ug)
    im["ocb"] = np.log(np.clip(im.tor_oc_ugm3 * im.volume_m3, 0.05, None) / im.hips_ug)
    nw["ocb"] = np.log(np.clip(nw.oc_ftir_ugm3 * nw.volume_m3, 0.05, None) / nw.hips_ug)
    okI = (im.hips_ug > 2) & np.isfinite(im.ocb) & (im.tor_ug > 0)
    q1, q99 = im.ocb[okI].quantile([.01, .99])
    hgb = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=.05, min_samples_leaf=30,
                                        random_state=0).fit(im.loc[okI, ["lh", "ocb"]], np.log(im.tor_ug / im.hips_ug)[okI])
    load_only = _binmed(y / im.hips_ug, im.hips_ug, LOAD_BINS, low)["median"]
    nw["exp_tor_hips"] = np.exp(hgb.predict(np.column_stack([nw.lh.fillna(0), nw.ocb.clip(q1, q99).fillna(0)])))
    nw["exp_tor_hips_lowocec"] = pd.cut(nw.hips_ug, LOAD_BINS).map(dict(zip(load_only.index, load_only.values))).astype(float)
    nw["ocb_outside_improve"] = (nw.ocb < q1) | (nw.ocb > q99)
    seasons = ad.set_index("filter_id").season
    nw["season"] = nw.filter_id.map(seasons)
    pick = {"published": nw.pub_ug.to_numpy()}
    for nm, k in [("raw", 18), ("spline", 18), ("vibes", 18), ("anchored", 18), ("anchored_noCH", 18), ("anchored_balanced", 12)] + \
            [(n, 18) for n in VC_VARIANTS if f"{n}|network" in P.files]:
        pick[f"{nm} k{k}"] = P[f"{nm}|network"][:, k - 1]
    keep = (nw.hips_ug > 2) & nw.ec_ftir_ugm3.notna() & nw.vibes_ok & ~nw.excluded & np.isfinite(nw.ocb)
    per = nw.loc[keep, ["filter_id", "code", "date", "season", "hips_ug", "ocb", "ocb_outside_improve", "exp_tor_hips",
                        "exp_tor_hips_lowocec"]].copy()
    for nm, v in pick.items():
        per[nm] = v[keep.to_numpy()] / per.hips_ug
    per.to_csv(OUT / "network_filters.csv", index=False)
    g = per.groupby("code")
    site = pd.concat([g.size().rename("n"), g.hips_ug.median().rename("hips_ug"), np.exp(g.ocb.median()).rename("oc_over_bc"),
                      g.ocb_outside_improve.mean().rename("share_oc_bc_outside_improve"),
                      g.exp_tor_hips.median().rename("expected_tor_over_hips")]
                     + [(per[nm] / per.exp_tor_hips).groupby(per.code).median().rename(nm) for nm in pick], axis=1)
    site.reset_index().to_csv(OUT / "sites_vs_tor_yardstick.csv", index=False)

    # 7. Addis residuals (anchored k18, after the within-Addis log-log loading trend)
    a = ad.merge(nw[["filter_id", "oc_ftir_ugm3", "pm25_ugm3", "si_ugm3", "al_ugm3", "k_xrf_ugm3", "fe_ugm3", "so4_ugm3"]],
                 on="filter_id", how="left")
    p = P["anchored|addis"][:, 17]
    lr, lh = np.log(np.clip(p, 0.1, None) / a.hips_ug), np.log(a.hips_ug)
    a["resid"] = lr - np.polyval(np.polyfit(lh, lr, 1), lh)
    a["month"] = pd.to_datetime(a.date).dt.month
    a.to_csv(OUT / "addis_residuals.csv", index=False)
    Z = np.asarray(anc["addis"], float) / a.hips_ug.to_numpy()[:, None]
    Zc, rc = Z - Z.mean(0), (a.resid - a.resid.mean()).to_numpy()
    rho = (Zc * rc[:, None]).sum(0) / np.sqrt((Zc ** 2).sum(0) * (rc ** 2).sum())
    pd.DataFrame({"wn": wn, "rho": rho}).to_csv(OUT / "addis_residual_spectrum.csv", index=False)
    print("tables written to", env.display_path(OUT))


# --------------------------------------------------------------------------- inject
def inject(bundle: Path, n251=10, n248=6, workers=12):
    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_limits
    from vibes_baseline import VibesBackground, vibes_baseline_matrix

    im, _, _ = load(bundle)
    z = np.load(VIBES_MODEL)
    wn, mu, W = z["wn"], z["mean"], z["components"]
    Xr = np.load(bundle / "improve_raw.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    pick = np.r_[rng.choice(np.where((im.hips_ug < 1) & (im.lot == 251))[0], n251, replace=False),
                 rng.choice(np.where((im.hips_ug < 1) & (im.lot == 248))[0], n248, replace=False)]
    x = wn / 2000.0

    def deflate(shapes):
        Q, _ = np.linalg.qr(np.column_stack(shapes))
        U, s, _ = np.linalg.svd(W - Q @ (Q.T @ W), full_matrices=False)
        r = W.shape[1] - len(shapes)
        return U[:, :r] * s[:r]

    variants = {"production (rank 70)": W, "minus nu^1 direction": deflate([x]),
                "minus 1, nu, nu^2": deflate([np.ones_like(x), x, x ** 2])}
    shapes = {"nu^1 (BC-like)": 0.05 * x, "flat": 0.05 * np.ones_like(x), "nu^2 (scattering-like)": 0.05 * x ** 2}

    def run(Wv, spec):
        bg = VibesBackground(wn, mu, Wv, pd.DataFrame(), tuple(), 0.0)
        with threadpool_limits(1):
            _, vib, _ = vibes_baseline_matrix(wn, spec[None].astype(float), bg, sample_ids=["s"], tau=0.1, loss="PB",
                                              maxiter=10000)
        return vib[0]

    jobs = []
    for vn, Wv in variants.items():
        for i in pick:
            spec = np.asarray(Xr[i], float)
            jobs.append((vn, int(i), "none", Wv, spec))
            jobs += [(vn, int(i), sn, Wv, spec + c) for sn, c in shapes.items()]
    t = time.time()
    res = Parallel(n_jobs=workers)(delayed(run)(j[3], j[4]) for j in jobs)
    got = {(j[0], j[1], j[2]): r for j, r in zip(jobs, res)}
    rows = []
    for vn in variants:
        for i in pick:
            b0 = got[(vn, int(i), "none")]
            rows.append(dict(variant=vn, row=int(i), lot=int(im.lot[i]), shape="light-filter residual RMS",
                             value=float(np.sqrt(np.mean(b0 ** 2)))))
            for sn, c in shapes.items():
                rows.append(dict(variant=vn, row=int(i), lot=int(im.lot[i]), shape=sn,
                                 value=float((got[(vn, int(i), sn)] - b0) @ c / (c @ c))))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "vibes_injection.csv", index=False)
    print(f"inject: {len(jobs)} VIBES fits in {time.time() - t:.0f}s")


# --------------------------------------------------------------------------- per-filter tables (gallery)
def perfilter(bundle: Path):
    """Per-filter predictions for the gallery's crossplots: every Addis filter (µg/m³) and IMPROVE filters with HIPS > 10 µg."""
    im, ad, nw = load(bundle)
    P = np.load(OUT / "predictions.npz")
    names = [k.split("|")[0] for k in P.files if k.endswith("|addis")]
    yard = pd.read_csv(OUT / "network_filters.csv").set_index("filter_id")
    a = ad[["filter_id", "date", "lot", "fabs_Mm1", "volume_m3", "hips_ug", "ec_published_ugm3", "excluded"]].copy()
    a["exp_tor_hips"] = a.filter_id.map(yard.exp_tor_hips)
    for nm in names:
        for k in (12, 18):
            a[f"{nm} k{k}"] = P[f"{nm}|addis"][:, k - 1] / a.volume_m3
    a.to_csv(OUT / "addis_predictions.csv", index=False)
    heavy = (im.hips_ug > 10).to_numpy() & np.isfinite(im.tor_ug.to_numpy())
    i = im.loc[heavy, ["filter_id", "site", "date", "lot", "tor_ug", "hips_ug", "ocec", "tor_optr_ugm3"]].copy()
    for nm in names:
        for k in (12, 18):
            i[f"{nm} k{k}"] = P[f"{nm}|improve_oof"][heavy, k - 1]
    i.to_csv(OUT / "improve_heavy_predictions.csv", index=False)
    print(f"perfilter: {len(a)} Addis, {len(i)} IMPROVE (HIPS > 10 µg), models {names}")


def main():
    global OUT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None, help="versioned output directory; use for closeout reruns to preserve the historical archives")
    p.add_argument("--stages", nargs="+", default=["fit", "tables"], choices=["fit", "tables", "inject", "perfilter"])
    a = p.parse_args()
    if a.output is not None:
        OUT = a.output
    bundle = a.bundle or default_bundle()
    for s in a.stages:
        {"fit": fit, "tables": tables, "inject": inject, "perfilter": perfilter}[s](bundle)


if __name__ == "__main__":
    main()
