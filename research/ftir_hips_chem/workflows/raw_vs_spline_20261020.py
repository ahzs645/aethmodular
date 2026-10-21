"""Baseline correction alone: the same all-IMPROVE lot-251 calibration on raw vs spline spectra (20 Oct 2026).

The AAAR poster's Step 1 compares the current SPARTAN calibration (raw spectra, SPARTAN's own calibration set and
settings) with our all-IMPROVE lot-251 calibration on spline (AIRSpec) baseline-corrected spectra. That changes two
things at once. Here only the spectra change: the calibration set (IMPROVE lot 251), the train/test site split, the
component-selection rule and the wavenumber window (1425-4000 cm-1) are identical to
``run_meeting_followup_20260917.stage_allimprove``; the raw spectra are the ones the spline correction was applied to,
from the frozen run's own bundle (content hash checked). A third variant, raw-full, uses the raw spectra over
4000-500 cm-1 (the window the September phase-3 explorer calibrations used), to separate the window from the baseline.

Result (20 Oct, 189 poster filters, TONT1 2022-11-19 excluded, see qc_keep): site-held-out raw-full 1.91x -4.75 (k 11),
raw 1425-4000 1.92x -5.19 (k 8), spline 0.59x -1.21 (k 8); Shiny-app protocol 2.42x -5.94 (k 24), 1.98x -5.38 (k 7),
0.63x -1.25 (k 11). Raw spectra on either window sit near SPARTAN's 2.30x -6.09; baseline correction moves the
intercept to about -1.2, with IMPROVE test R² unchanged. A first pass that kept TONT1 suggested the window, not the
baseline, did the work; that was the one corrupt spectrum (raw offset ~3 absorbance) steering the raw models.

Run from the repo root:  uv run python research/ftir_hips_chem/workflows/raw_vs_spline_20261020.py [app]
(``app`` = the same three spectra under the Shiny app's protocol, see ``main_app``)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AREA / "scripts"))
import analog_bias as ab  # noqa: E402

BUNDLE = AREA / "output/tables/vibes_colab_bundle/stage"
# same filters, same order, same values on 1425-4000; adds 1425-500 cm-1 (checked 20 Oct)
BUNDLE_FULL = AREA / "output/tables/vibes_colab_bundle_full500/stage"
OUT = AREA / "output/tables/raw_vs_spline_20261020"
DEPLOYED = (AREA / "output/tables/vibes_fullrange/gallery_rank70_20261002/meeting/addis_predictions.csv")
LOT = 251


def raw_rows(cases, rows, bundle=BUNDLE):
    """Raw spectra for frozen-run case rows, from the bundle the run corrected (or its full-range twin)."""
    px = np.load(bundle / "data/pool_raw.npy", mmap_mode="r")
    ex = np.load(bundle / "data/etad_raw.npy", mmap_mode="r")
    c = cases.iloc[np.asarray(rows)]
    out = np.empty((len(c), px.shape[1]))
    for src, arr in (("improve", px), ("etad", ex)):
        m = c.source.eq(src).to_numpy()
        out[m] = arr[c.source_row.to_numpy()[m].astype(int)]
    if not c.source.isin(["improve", "etad"]).all():
        raise ValueError("unexpected case source")
    return out


def qc_keep(X_raw_cut, ctx):
    """Calibration filters whose raw spectrum is physically plausible.

    One lot-251 IMPROVE filter (TONT1, 2022-11-19, 5.6 µg EC) has a raw 1425-4000 cm-1 spectrum sitting ~3 absorbance
    units high (row max 3.19 against a 99.9th percentile of 0.61); raw-spectra models predict it at ~1,400 µg and it
    dominates their CV error. Baseline correction hides the offset, so the frozen spline run kept it. Dropped here
    for every variant and both protocols so the comparison stays like for like (20 Oct).
    """
    keep = np.abs(X_raw_cut).max(axis=1) <= 1.5
    dropped = ctx.lib.loc[~keep, ["Site", "date", "lot"]].astype(str).values.tolist()
    if [d for d in dropped if d[2] == str(LOT)] != [["TONT1", "2022-11-19", str(LOT)]]:
        raise ValueError(f"QC drops more than the one known bad lot-{LOT} spectrum: {dropped}")
    return keep


def main():
    manifest = json.loads((ab.RUN / "RUN_MANIFEST.json").read_text())
    bundle = json.loads((BUNDLE / "BUNDLE_MANIFEST.json").read_text())
    if manifest["bundle_hash"] != bundle["content_hash"]:
        raise ValueError("bundle is not the one the frozen run corrected")
    if not np.allclose(np.load(BUNDLE / "data/wn.npy"), np.load(ab.RUN / "wn.npy")):
        raise ValueError("raw and corrected grids differ")

    ctx = ab.load_context(fullrange=False)
    cases = pd.read_csv(ab.RUN / "cases.csv")
    for f in ("pool_metadata.csv", "etad_metadata.csv"):
        if not pd.read_csv(BUNDLE / "data" / f).equals(pd.read_csv(BUNDLE_FULL / "data" / f)):
            raise ValueError(f"full-range bundle rows differ: {f}")
    # raw = raw spectra on the spline window (1425-4000 cm-1); raw-full = raw 4000-500 cm-1, as the September
    # phase-3 explorer calibrations used
    X = {"spline": ctx.X["AIRSpec"], "raw": raw_rows(cases, ctx.lib.row),
         "raw-full": raw_rows(cases, ctx.lib.row, BUNDLE_FULL)}
    XA = {"spline": ctx.XA["AIRSpec"], "raw": raw_rows(cases, ctx.addis.row),
          "raw-full": raw_rows(cases, ctx.addis.row, BUNDLE_FULL)}

    dep = pd.read_csv(DEPLOYED)
    dep = dep[dep.fit_id.eq("deployed|SPARTAN|current SPARTAN calibration")].dropna(subset=["ftir_ec_ugm3", "Fabs"])
    cohort = ctx.addis.MediaId.isin(dep.MediaId).to_numpy()          # the poster's 189 filters
    vol = ctx.addis.Volume_m3.to_numpy(float)
    qc = qc_keep(X["raw"], ctx)
    lot = np.flatnonzero(ctx.lib.lot.eq(LOT).to_numpy() & qc)
    sub = ctx.lib.iloc[lot]
    test = np.flatnonzero(ctx.lib.split.eq("test").to_numpy() & ctx.lib.lot.eq(LOT).to_numpy() & qc)

    rows, preds = [], []
    for name in ("raw-full", "raw", "spline"):
        f = ab.fit_cohort(X[name][lot], sub.y, sub.Site, train=sub.split.eq("train").to_numpy())
        pa = f.predict(XA[name]) / vol
        tor = ab.xy_stats(ctx.lib.y.to_numpy()[test], f.predict(X[name][test]))     # µg per filter
        for label, take in (("Addis 189 (poster)", cohort), ("Addis 233", np.ones(len(vol), bool))):
            a = ctx.addis.loc[take]
            st = ab.xy_stats(a.Fabs.to_numpy(float) / ab.MAC_VALUE, pa[take], lam=ctx.lam,
                             sigma_x=ctx.sigma_x, sigma_y=ab.SIGMA_Y_PROXY)
            rows.append({"spectra": name, "k": f.k, "population": label, "n": st["n"],
                         "deming_slope": st["deming_slope"], "deming_intercept": st["deming_intercept"],
                         "R2": st["R2"], "improve_test_TOR_R2": tor["R2"], "improve_test_n": tor["n"]})
        preds.append(ctx.addis[["MediaId", "date", "season", "Fabs"]].assign(ftir_ec_ugm3=pa, spectra=name))

    # component sweep: the selection rule picks very different k for raw and spline, and raw-spectra predictions are
    # known to depend strongly on k (1 Sep phase-3 test: network raw k=15 gave 1.94x -4.82). Every first-k submodel of
    # one k_max fit (pls_ksweep.prefix_predict), on lot 251 and on both lots, scored on the 189 poster filters.
    import pls_ksweep as pk
    from sklearn.cross_decomposition import PLSRegression
    ks = list(range(1, 21))
    sweep = []
    a189 = ctx.addis.loc[cohort]
    for lots, keep in (("lot 251", ctx.lib.lot.eq(LOT).to_numpy() & qc), ("lots 248+251", qc)):
        tr = np.flatnonzero(keep & ctx.lib.split.eq("train").to_numpy())
        te = np.flatnonzero(keep & ctx.lib.split.eq("test").to_numpy())
        for name in ("raw-full", "raw", "spline"):
            m = PLSRegression(n_components=max(ks), scale=False).fit(X[name][tr], ctx.lib.y.to_numpy()[tr])
            B = pk.prefix_coefs(m, ks)
            pa_k = pk.prefix_predict(m, XA[name][cohort], ks, B) / vol[cohort, None]
            pt_k = pk.prefix_predict(m, X[name][te], ks, B)
            for j, k in enumerate(ks):
                st = ab.xy_stats(a189.Fabs.to_numpy(float) / ab.MAC_VALUE, pa_k[:, j], lam=ctx.lam,
                                 sigma_x=ctx.sigma_x, sigma_y=ab.SIGMA_Y_PROXY)
                tor = ab.xy_stats(ctx.lib.y.to_numpy()[te], pt_k[:, j])
                sweep.append({"lots": lots, "spectra": name, "k": k, "n_train": len(tr),
                              "deming_slope": st["deming_slope"], "deming_intercept": st["deming_intercept"],
                              "R2": st["R2"], "improve_test_TOR_R2": tor["R2"]})

    # per-filter export for the comparison page: lot 251, the 189 poster filters, k 1-30, every variant
    ks30 = list(range(1, 31))
    tr = np.flatnonzero(ctx.lib.lot.eq(LOT).to_numpy() & ctx.lib.split.eq("train").to_numpy() & qc)
    page = {"x_hips_ec": np.round(a189.Fabs.to_numpy(float) / ab.MAC_VALUE, 4).tolist(),
            "season": a189.season.astype(str).tolist(), "ks": ks30, "variants": {}}
    dep_y = a189[["MediaId"]].merge(dep[["MediaId", "ftir_ec_ugm3"]], on="MediaId", how="left").ftir_ec_ugm3
    st = ab.xy_stats(a189.Fabs.to_numpy(float) / ab.MAC_VALUE, dep_y.to_numpy(float), lam=ctx.lam,
                     sigma_x=ctx.sigma_x, sigma_y=ab.SIGMA_Y_PROXY)
    page["spartan"] = {"y": np.round(dep_y.to_numpy(float), 4).tolist(),
                       **{k_: st[k_] for k_ in ("deming_slope", "deming_intercept", "R2", "n")}}
    for name in ("raw-full", "raw", "spline"):
        m = PLSRegression(n_components=max(ks30), scale=False).fit(X[name][tr], ctx.lib.y.to_numpy()[tr])
        B = pk.prefix_coefs(m, ks30)
        pa_k = pk.prefix_predict(m, XA[name][cohort], ks30, B) / vol[cohort, None]
        pt_k = pk.prefix_predict(m, X[name][test], ks30, B)
        stats = []
        for j, k in enumerate(ks30):
            s_ = ab.xy_stats(a189.Fabs.to_numpy(float) / ab.MAC_VALUE, pa_k[:, j], lam=ctx.lam,
                             sigma_x=ctx.sigma_x, sigma_y=ab.SIGMA_Y_PROXY)
            t_ = ab.xy_stats(ctx.lib.y.to_numpy()[test], pt_k[:, j])
            stats.append({"k": k, "slope": s_["deming_slope"], "intercept": s_["deming_intercept"],
                          "R2": s_["R2"], "improve_R2": t_["R2"]})
        chosen = next(r["k"] for r in rows if r["spectra"] == name)
        page["variants"][name] = {"protocol_k": chosen, "stats": stats,
                                  "y": np.round(pa_k.T, 4).tolist()}
    page["lambda"], page["n_improve_test"] = float(ctx.lam), int(len(test))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison_page_data.json").write_text(json.dumps(page))
    sw = pd.DataFrame(sweep)
    sw.to_csv(OUT / "addis_raw_vs_spline_ksweep.csv", index=False)
    print(sw.round(2).to_string(index=False))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "addis_raw_vs_spline.csv", index=False)
    pd.concat(preds).to_csv(OUT / "addis_predictions_raw_vs_spline.csv", index=False)
    print(table.round(3).to_string(index=False))


def main_app():
    """The same three spectra under the AQRC FTIR Calibration Shiny app's protocol (calibration_modes 'app').

    Interleaved 10-fold CV with no site grouping, up to 80 components, k = first within 5% of the RMSECV minimum,
    model fitted on every lot-251 filter (so there is no held-out IMPROVE test set; the IMPROVE number reported is
    the R² of the pooled interleaved-CV predictions, the same statistic as the site-held-out test R²). Writes comparison_page_data_app.json next to the
    site-held-out export.
    """
    import pls_ksweep as pk
    from sklearn.cross_decomposition import PLSRegression
    ctx = ab.load_context(fullrange=False)
    cases = pd.read_csv(ab.RUN / "cases.csv")
    X = {"spline": ctx.X["AIRSpec"], "raw": raw_rows(cases, ctx.lib.row),
         "raw-full": raw_rows(cases, ctx.lib.row, BUNDLE_FULL)}
    XA = {"spline": ctx.XA["AIRSpec"], "raw": raw_rows(cases, ctx.addis.row),
          "raw-full": raw_rows(cases, ctx.addis.row, BUNDLE_FULL)}
    dep = pd.read_csv(DEPLOYED)
    dep = dep[dep.fit_id.eq("deployed|SPARTAN|current SPARTAN calibration")].dropna(subset=["ftir_ec_ugm3", "Fabs"])
    cohort = ctx.addis.MediaId.isin(dep.MediaId).to_numpy()
    a189 = ctx.addis.loc[cohort]
    vol = ctx.addis.Volume_m3.to_numpy(float)
    lot = np.flatnonzero(ctx.lib.lot.eq(LOT).to_numpy() & qc_keep(X["raw"], ctx))
    sub = ctx.lib.iloc[lot]
    y = sub.y.to_numpy(float)
    ks = list(range(1, 81))
    fold = np.arange(len(y)) % 10          # the app's interleaved folds (calibration_modes.interleaved_cv_curve)
    out = {"protocol": "app", "n_train": int(len(lot)), "ks": ks, "variants": {}}
    for name in ("raw-full", "raw", "spline"):
        t0 = pd.Timestamp.now()
        f = ab.fit_cohort(X[name][lot], y, sub.Site.to_numpy(), mode="app", max_components=max(ks))
        assert f.train.all()
        m = PLSRegression(n_components=max(ks), scale=False).fit(X[name][lot], y)
        B = pk.prefix_coefs(m, ks)
        pa_k = pk.prefix_predict(m, XA[name][cohort], ks, B) / vol[cohort, None]
        # pooled interleaved-CV predictions at every k, scored with the same R² (xy_stats) as the site-held-out test
        cvp = np.zeros((len(y), len(ks)))
        for i in range(10):
            mi = PLSRegression(n_components=max(ks), scale=False).fit(X[name][lot][fold != i], y[fold != i])
            cvp[fold == i] = pk.prefix_predict(mi, X[name][lot][fold == i], ks)
        stats = []
        for j, k in enumerate(ks):
            s_ = ab.xy_stats(a189.Fabs.to_numpy(float) / ab.MAC_VALUE, pa_k[:, j], lam=ctx.lam,
                             sigma_x=ctx.sigma_x, sigma_y=ab.SIGMA_Y_PROXY)
            stats.append({"k": k, "slope": s_["deming_slope"], "intercept": s_["deming_intercept"],
                          "R2": s_["R2"], "improve_R2": ab.xy_stats(y, cvp[:, j])["R2"]})
        out["variants"][name] = {"protocol_k": f.k, "stats": stats, "y": np.round(pa_k.T, 4).tolist()}
        s_k = stats[f.k - 1]
        print(f"app {name}: k={f.k} Addis {s_k['slope']:.2f}x {s_k['intercept']:+.2f} R2 {s_k['R2']:.2f} "
              f"IMPROVE CV R2 {s_k['improve_R2']:.2f} ({(pd.Timestamp.now() - t0).total_seconds():.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison_page_data_app.json").write_text(json.dumps(out))


BANDS = [   # cm-1, high to low; labels for the comparison page
    (4000, 3600, "3600–4000 · no absorbers (baseline)"), (3600, 3000, "3000–3600 · O–H / N–H"),
    (3000, 2800, "2800–3000 · C–H"), (2800, 1800, "1800–2800 · baseline, CO₂"),
    (1800, 1650, "1650–1800 · carbonyl"), (1650, 1500, "1500–1650 · 1600 band"),
    (1500, 1425, "1425–1500"), (1425, 1100, "1100–1425 · Teflon (C–F)"),
    (1100, 900, "900–1100 · silicate (Si–O)"), (900, 500, "500–900 · Teflon, mineral"),
]


def main_contrib():
    """Which wavenumbers push the Addis predictions up: exact linear split of each PLS prediction.

    prediction (µg/filter) = ȳ + Σ_j b_j (x_j - x̄_j), so c_ij = b_j (x_ij - x̄_j) / volume_i is filter i's contribution
    at wavenumber j in µg/m³, and the terms add up exactly. Models: app protocol (all QC'd lot-251 filters), raw
    4000-500 and raw 1425-4000 at k 18 (the SPARTAN-like match), spline at k 18 and at the poster's k 8. Summaries
    avoid OLS: the mean contribution over the 189 poster filters (level), and the mean over the heaviest third by HIPS
    EC minus the lightest third (the gap that sets the slope; the HIPS gap itself is reported alongside).
    """
    import pls_ksweep as pk
    from sklearn.cross_decomposition import PLSRegression
    ctx = ab.load_context(fullrange=False)
    cases = pd.read_csv(ab.RUN / "cases.csv")
    X = {"spline": ctx.X["AIRSpec"], "raw": raw_rows(cases, ctx.lib.row),
         "raw-full": raw_rows(cases, ctx.lib.row, BUNDLE_FULL)}
    XA = {"spline": ctx.XA["AIRSpec"], "raw": raw_rows(cases, ctx.addis.row),
          "raw-full": raw_rows(cases, ctx.addis.row, BUNDLE_FULL)}
    wn = {"spline": np.load(BUNDLE / "data/wn.npy"), "raw": np.load(BUNDLE / "data/wn.npy"),
          "raw-full": np.load(BUNDLE_FULL / "data/wn.npy")}
    dep = pd.read_csv(DEPLOYED)
    dep = dep[dep.fit_id.eq("deployed|SPARTAN|current SPARTAN calibration")].dropna(subset=["ftir_ec_ugm3", "Fabs"])
    cohort = ctx.addis.MediaId.isin(dep.MediaId).to_numpy()
    a189 = ctx.addis.loc[cohort]
    hips = a189.Fabs.to_numpy(float) / ab.MAC_VALUE
    vol = ctx.addis.Volume_m3.to_numpy(float)[cohort]
    lo, hi = hips <= np.quantile(hips, 1 / 3), hips >= np.quantile(hips, 2 / 3)
    lot = np.flatnonzero(ctx.lib.lot.eq(LOT).to_numpy() & qc_keep(X["raw"], ctx))
    y = ctx.lib.y.to_numpy(float)[lot]
    out = {"n": int(cohort.sum()), "hips_mean": float(hips.mean()), "hips_gap": float(hips[hi].mean() - hips[lo].mean()),
           "n_lo": int(lo.sum()), "n_hi": int(hi.sum()), "bands": [b[2] for b in BANDS], "models": []}
    fits = {}
    for name, k in (("raw-full", 18), ("raw", 18), ("spline", 18), ("spline", 8)):
        if name not in fits:
            fits[name] = PLSRegression(n_components=18, scale=False).fit(X[name][lot], y)
        m = fits[name]
        b = pk.prefix_coefs(m, [k])[:, 0]
        xa = XA[name][cohort]
        C = (xa - m._x_mean) * b / vol[:, None]                     # µg/m³ per wavenumber
        const = float(np.ravel(m._y_mean)[0]) / vol                # the mean-spectrum term, µg/m³
        pred = C.sum(1) + const
        ref = pk.prefix_predict(m, xa, [k])[:, 0] / vol
        assert np.allclose(pred, ref, atol=1e-8), np.abs(pred - ref).max()
        st = ab.xy_stats(hips, pred, lam=ctx.lam, sigma_x=ctx.sigma_x, sigma_y=ab.SIGMA_Y_PROXY)
        w = wn[name]
        band_rows = []
        for top, bot, lab in BANDS:
            sel = (w <= top) & (w > bot)
            if not sel.any():
                band_rows.append({"band": lab, "mean": None, "gap": None}); continue
            cb = C[:, sel].sum(1)
            band_rows.append({"band": lab, "mean": float(cb.mean()), "gap": float(cb[hi].mean() - cb[lo].mean())})
        step = max(1, len(w) // 700)    # thin the per-wavenumber curves for the page
        if w[0] < w[-1]:
            raise ValueError("expected wavenumbers high to low")
        ix = np.unique(np.r_[np.arange(0, len(w), step), len(w) - 1])     # keep the last point: totals end exact
        out["models"].append({
            "id": f"{name}-k{k}", "spectra": name, "k": k,
            "fit": {"slope": st["deming_slope"], "intercept": st["deming_intercept"], "R2": st["R2"]},
            "pred_mean": float(pred.mean()), "pred_gap": float(pred[hi].mean() - pred[lo].mean()),
            "const_mean": float(const.mean()), "const_gap": float(const[hi].mean() - const[lo].mean()),
            "bands": band_rows,
            "wn": np.round(w[ix], 1).tolist(),
            # running totals from 4000 cm-1 down (exact cumsum, then thinned): where the prediction is built and cancelled
            "cum_mean": np.round(np.cumsum(C.mean(0))[ix], 4).tolist(),
            "cum_gap": np.round(np.cumsum(C[hi].mean(0) - C[lo].mean(0))[ix], 4).tolist(),
            "gross_gap": float(np.abs(C[hi].mean(0) - C[lo].mean(0)).sum()),
        })
        print(f"{name} k{k}: Addis {st['deming_slope']:.2f}x {st['deming_intercept']:+.2f}; mean pred "
              f"{pred.mean():.2f} (HIPS {hips.mean():.2f}); hi-lo gap {pred[hi].mean() - pred[lo].mean():.2f} "
              f"(HIPS {out['hips_gap']:.2f})", flush=True)
        for r in band_rows:
            if r["mean"] is not None:
                print(f"    {r['band']:38s} mean {r['mean']:+7.2f}  gap {r['gap']:+7.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contribution_page_data.json").write_text(json.dumps(out))


# The AQRC "FTIR Calibration" Shiny app (global.R, R/selectorServer.R, R/calibrateServer.R; the source Sean shared in
# July 2026, read 20 Oct 2026, see docs/spartan-ftir-ec-recreation.md): raw absorbance, every 4th point above 500 cm-1, 21 default
# IMPROVE sites ("All sites" off by default), field blanks included by default with the target set to 0, an optional
# biomass-burning selection from data/BiomassDetected_*.csv, pls::plsr up to 80 components with interleaved 10-fold
# CV, and k picked by the analyst on the RMSEP curve.
APP_SITES = ["SHMI1", "BIRM1", "TALL1", "PORE1", "WICA1", "BOND1", "KALM1", "MEAD1", "MELA1", "VIIS1", "FRES1",
             "DOSO1", "HAVO1", "EGBE1", "DOME1", "GRBA1", "VOYA2", "CHAS1", "THSI1", "FLAT1", "PHOE1"]
BIOMASS_DIR = OUT / "app_biomass"      # copies of the app's BiomassDetected_2019..2025.csv


def main_spartan_match():
    """Which of the app's options reproduce SPARTAN's published Addis EC best (raw 4000-500, lot 251).

    Grid: sites (all lot-251 / the app's 21), field blanks (left out / in at 0 µg), biomass (any / only / exclude).
    Each configuration is one PLS fit on all its filters (as the app ships), k 1-40, scored per filter against the
    published values for the 189 poster filters (RMSE, r) and on the Addis Deming fit.
    """
    import pls_ksweep as pk
    from sklearn.cross_decomposition import PLSRegression
    ctx = ab.load_context(fullrange=False)
    cases = pd.read_csv(ab.RUN / "cases.csv")
    X = raw_rows(cases, ctx.lib.row, BUNDLE_FULL)
    keep = qc_keep(raw_rows(cases, ctx.lib.row), ctx)
    XA = raw_rows(cases, ctx.addis.row, BUNDLE_FULL)
    pool = pd.read_csv(BUNDLE_FULL / "data/pool_metadata.csv")
    praw = np.load(BUNDLE_FULL / "data/pool_raw.npy", mmap_mode="r")
    blanks = pool[pool.FilterPurposeId.eq(2) & pool.LotNumber.eq(LOT) & pool.finite_spectrum].copy()
    XB = np.asarray(praw[blanks.index.to_numpy()], float)
    okb = np.abs(XB[:, :2002]).max(1) <= 1.5 if XB.shape[1] > 2002 else np.ones(len(XB), bool)
    blanks, XB = blanks[okb], XB[okb]
    bio = pd.concat([pd.read_csv(f) for f in sorted(BIOMASS_DIR.glob("BiomassDetected_*.csv"))])
    bio_ids = set(bio.FilterId.astype(int))
    dep = pd.read_csv(DEPLOYED)
    dep = dep[dep.fit_id.eq("deployed|SPARTAN|current SPARTAN calibration")].dropna(subset=["ftir_ec_ugm3", "Fabs"])
    cohort = ctx.addis.MediaId.isin(dep.MediaId).to_numpy()
    a189 = ctx.addis.loc[cohort]
    dep_y = a189[["MediaId"]].merge(dep[["MediaId", "ftir_ec_ugm3"]], on="MediaId", how="left").ftir_ec_ugm3.to_numpy()
    vol = ctx.addis.Volume_m3.to_numpy(float)[cohort]
    hips = a189.Fabs.to_numpy(float) / ab.MAC_VALUE
    lib = ctx.lib
    base = keep & lib.lot.eq(LOT).to_numpy()
    print(f"lot {LOT}: {base.sum()} samples, {lib.Site[base].nunique()} sites; app sites present: "
          f"{sorted(set(APP_SITES) & set(lib.Site[base]))} ({len(set(APP_SITES) & set(lib.Site[base]))} of 21); "
          f"biomass-listed samples: {lib.filter_id[base].isin(bio_ids).sum()}; field blanks: {len(blanks)} "
          f"({blanks.Site.isin(APP_SITES).sum()} at app sites); dates {lib.date[base].min().date()} to "
          f"{lib.date[base].max().date()}", flush=True)
    ks = list(range(1, 41))
    rows = []
    for sites in ("all", "app21"):
        for use_blanks in (False, True):
            for biomass in ("any", "only", "exclude"):
                m_ = base.copy()
                if sites == "app21":
                    m_ &= lib.Site.isin(APP_SITES).to_numpy()
                if biomass == "only":
                    m_ &= lib.filter_id.isin(bio_ids).to_numpy()
                elif biomass == "exclude":
                    m_ &= ~lib.filter_id.isin(bio_ids).to_numpy()
                Xt, yt = X[m_], lib.y.to_numpy(float)[m_]
                nb = 0
                if use_blanks:
                    bsel = blanks.Site.isin(APP_SITES).to_numpy() if sites == "app21" else np.ones(len(blanks), bool)
                    Xt, yt, nb = np.vstack([Xt, XB[bsel]]), np.r_[yt, np.zeros(bsel.sum())], int(bsel.sum())
                if m_.sum() < 60:
                    rows.append({"sites": sites, "blanks": use_blanks, "biomass": biomass, "n": int(m_.sum()), "n_blanks": nb}); continue
                kmax = min(max(ks), len(yt) - 2)
                m = PLSRegression(n_components=kmax, scale=False).fit(Xt, yt)
                kk = ks[:kmax]
                P = pk.prefix_predict(m, XA[cohort], kk) / vol[:, None]
                rm = np.sqrt(((P - dep_y[:, None]) ** 2).mean(0))
                j = int(np.argmin(rm))
                r = np.corrcoef(P[:, j], dep_y)[0, 1]
                st = ab.xy_stats(hips, P[:, j], lam=ctx.lam, sigma_x=ctx.sigma_x, sigma_y=ab.SIGMA_Y_PROXY)
                rows.append({"sites": sites, "blanks": use_blanks, "biomass": biomass, "n": int(m_.sum()),
                             "n_blanks": nb, "best_k": kk[j], "rmse_vs_spartan": float(rm[j]), "r_vs_spartan": float(r),
                             "addis_slope": st["deming_slope"], "addis_intercept": st["deming_intercept"],
                             "rmse_k18": float(rm[17]) if kmax >= 18 else np.nan,
                             "rmse_by_k": np.round(rm, 3).tolist()})
                print(f"{sites:5s} blanks={use_blanks!s:5s} biomass={biomass:7s} n={m_.sum():5d}+{nb:3d}: best k {kk[j]:2d} "
                      f"RMSE {rm[j]:.3f} r {r:.4f} | Addis {st['deming_slope']:.2f}x {st['deming_intercept']:+.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(OUT / "spartan_match_grid.json", orient="records")


FREEZE = OUT / "spartan_equivalent"
FREEZE_K = 18


def main_freeze():
    """Save the SPARTAN-equivalent EC model once, in the Shiny app's coefficient format.

    Recipe (20 Oct 2026, see main_spartan_match and the comparison page): raw absorbance 4000-500 cm-1 on the app's
    2,722-point grid, every QC'd lot-251 IMPROVE filter (TONT1 2022-11-19 excluded), no field blanks, no biomass
    selection, fitted on all filters as the app ships, k 18. Target: TOR EC in µg per filter. Matches SPARTAN's published
    Addis EC filter by filter (r 0.992, RMSE 0.30 µg/m³). Writes calibration-EC-spartan-equivalent-k18.csv
    ("", Wavenumber, b; Wavenumber 0 = intercept, as the app's "Export Coefficients"), the Addis comparison table, the
    page data, and a README. Apply as: µg per filter = b0 + Σ b_j x_j on raw spectra; ÷ air volume for µg/m³.
    """
    from sklearn.cross_decomposition import PLSRegression
    ctx = ab.load_context(fullrange=False)
    cases = pd.read_csv(ab.RUN / "cases.csv")
    X = raw_rows(cases, ctx.lib.row, BUNDLE_FULL)
    XA = raw_rows(cases, ctx.addis.row, BUNDLE_FULL)
    wn = np.load(BUNDLE_FULL / "data/wn.npy")
    sel = ctx.lib.lot.eq(LOT).to_numpy() & qc_keep(raw_rows(cases, ctx.lib.row), ctx)
    y = ctx.lib.y.to_numpy(float)[sel]
    m = PLSRegression(n_components=FREEZE_K, scale=False).fit(X[sel], y)
    b = np.ravel(m.coef_) if m.coef_.shape[0] == X.shape[1] else np.ravel(m.coef_.T)
    b0 = float(np.ravel(m._y_mean)[0] - m._x_mean @ b)
    FREEZE.mkdir(parents=True, exist_ok=True)
    coef = pd.DataFrame({"Wavenumber": np.r_[0.0, wn], "b": np.r_[b0, b]})
    coef.index = np.arange(1, len(coef) + 1).astype(str)
    coef.to_csv(FREEZE / f"calibration-EC-spartan-equivalent-k{FREEZE_K}.csv")
    # reload the saved file and apply it exactly as the app's coefficients would be applied
    c = pd.read_csv(FREEZE / f"calibration-EC-spartan-equivalent-k{FREEZE_K}.csv")
    rb0, rb = float(c.b.iloc[0]), c.b.to_numpy()[1:]
    pred_ug = rb0 + XA @ rb
    assert np.allclose(pred_ug, m.predict(XA).ravel(), atol=1e-8)
    dep = pd.read_csv(DEPLOYED)
    dep = dep[dep.fit_id.eq("deployed|SPARTAN|current SPARTAN calibration")].dropna(subset=["ftir_ec_ugm3", "Fabs"])
    t = ctx.addis[["MediaId", "ExternalFilterId", "date", "season", "Fabs", "Volume_m3"]].copy()
    t["hips_ec"] = t.Fabs / ab.MAC_VALUE
    t["recreation_ec"] = pred_ug / t.Volume_m3.to_numpy(float)
    t = t.merge(dep[["MediaId", "ftir_ec_ugm3"]].rename(columns={"ftir_ec_ugm3": "spartan_ec"}), on="MediaId", how="left")
    t.to_csv(FREEZE / "addis_spartan_vs_recreation.csv", index=False)
    c189 = t.dropna(subset=["spartan_ec"])
    fits = {}
    for col in ("spartan_ec", "recreation_ec"):
        st = ab.xy_stats(c189.hips_ec.to_numpy(float), c189[col].to_numpy(float), lam=ctx.lam, sigma_x=ctx.sigma_x,
                         sigma_y=ab.SIGMA_Y_PROXY)
        fits[col] = {k_: st[k_] for k_ in ("deming_slope", "deming_intercept", "R2", "n")}
    d = c189.recreation_ec - c189.spartan_ec
    seasons = {g: {"n": int(len(q)), "mean_diff": float((q.recreation_ec - q.spartan_ec).mean()),
                   "sd_diff": float((q.recreation_ec - q.spartan_ec).std()),
                   **{f"{col}_fit": dict(zip(("slope", "intercept", "R2"), [
                       ab.xy_stats(q.hips_ec.to_numpy(float), q[col].to_numpy(float), lam=ctx.lam, sigma_x=ctx.sigma_x,
                                   sigma_y=ab.SIGMA_Y_PROXY)[k_] for k_ in ("deming_slope", "deming_intercept", "R2")]))
                      for col in ("spartan_ec", "recreation_ec")}}
               for g, q in c189.groupby("season")}
    summary = {"k": FREEZE_K, "n_train": int(sel.sum()), "n": int(len(c189)), "fits": fits, "seasons": seasons,
               "r": float(np.corrcoef(c189.recreation_ec, c189.spartan_ec)[0, 1]), "rmse": float(np.sqrt((d ** 2).mean())),
               "mean_diff": float(d.mean()), "median_ratio": float((c189.recreation_ec / c189.spartan_ec).median()),
               "points": {"date": c189.date.astype(str).str[:10].tolist(), "season": c189.season.astype(str).tolist(),
                          "hips": np.round(c189.hips_ec, 4).tolist(), "spartan": np.round(c189.spartan_ec, 4).tolist(),
                          "recreation": np.round(c189.recreation_ec, 4).tolist()}}
    (FREEZE / "side_by_side_page_data.json").write_text(json.dumps(summary))
    (FREEZE / "README.md").write_text(f"""# SPARTAN-equivalent FTIR EC model (frozen 20 Oct 2026)

Built by `research/ftir_hips_chem/workflows/raw_vs_spline_20261020.py freeze`. A reconstruction, not AQRC's model.

- Spectra: raw absorbance, 4000-500 cm-1, the AQRC app's 2,722-point grid (every 4th point above 500 cm-1).
- Calibration set: {int(sel.sum()):,} lot-251 IMPROVE filters, all sites, no field blanks, no biomass selection;
  TONT1 2022-11-19 excluded (corrupt raw spectrum).
- PLS fitted on all of them (as the app ships its model), k = {FREEZE_K}; target TOR EC, µg per filter.
- `calibration-EC-spartan-equivalent-k{FREEZE_K}.csv` has the app's "Export Coefficients" layout: row Wavenumber 0 is
  the intercept. µg per filter = b0 + sum(b_j x_j) on raw spectra; divide by the filter's air volume for µg/m³.
- Against SPARTAN's published EC for the {len(c189)} poster Addis filters: r {summary['r']:.3f}, RMSE {summary['rmse']:.2f} µg/m³,
  mean difference {summary['mean_diff']:+.3f} µg/m³, median ratio {summary['median_ratio']:.3f}.
  Addis vs HIPS EC (Deming): SPARTAN {fits['spartan_ec']['deming_slope']:.2f}x {fits['spartan_ec']['deming_intercept']:+.2f},
  recreation {fits['recreation_ec']['deming_slope']:.2f}x {fits['recreation_ec']['deming_intercept']:+.2f}.
- Full write-up (sources, choices, match, what is unknown): docs/spartan-ftir-ec-recreation.md. Present SPARTAN's
  published values as SPARTAN's EC; this file is an analysis stand-in.
- Why these choices: see the "Matching SPARTAN" and "What SPARTAN's calibration is" sections of the comparison page
  (deliverables/aaar_poster_2026/addis_intercept.html). SPARTAN's own model (Anwar 2025: 22 IMPROVE sites, raw spectra)
  was trained on years we do not have spectra for, which is the likely source of the remaining ±3% per-filter scatter.
""")
    print(json.dumps({k_: v for k_, v in summary.items() if k_ != "points"}, indent=1, default=float))


if __name__ == "__main__":
    {"app": main_app, "contrib": main_contrib, "spartan": main_spartan_match, "freeze": main_freeze}.get(
        sys.argv[1] if sys.argv[1:] else "", main)()
