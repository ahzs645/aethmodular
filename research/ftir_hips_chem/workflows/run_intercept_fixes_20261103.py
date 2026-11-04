"""Can the Addis FTIR EC intercept be fixed without Addis labels? (3 Nov 2026)

Follows run_transfer_diagnosis_20261102.py. On the continuum-protected VIBES spectra (``vibes_anchorhp``) Addis still reads
1.88x HIPS - 4.88 µg/m³ (1.29x - 1.81 against the IMPROVE TOR yardstick). Addis HIPS BC spans only 3.4-6.8 µg/m³, so the
intercept is almost all gain: intercept ~ -(slope - 1) x mean. Stage ``attrib`` splits the Addis prediction into spectral
windows and gives each window's OLS slope and intercept against HIPS (they add up to the total), which says where the
excess day-to-day gain comes from.

Stage ``fit`` trains every candidate on IMPROVE TOR EC only (site-grouped 5-fold CV for the IMPROVE error, the full pool
for Addis and the SPARTAN network). Addis HIPS is never used to fit or choose anything; two candidates use the *unlabeled*
Addis spectra (di-PLS, importance weights). Spectra are averaged over channel pairs (2722 -> 1361) to keep di-PLS cheap;
``base`` repeats the full-resolution model on that grid to show the binning changes nothing.

* base          PLS on vibes_anchorhp (the current "after").
* cont_only     only the band-free continuum windows (3990-3800, 2650-2450, 2280-1850): the BC continuum, nothing else.
* mask_org      drops O-H/N-H/C-H 3700-2700 and 1800-1500, where the window attribution puts the excess gain.
* epo           external parameter orthogonalisation: the IMPROVE spectral signatures of OC, other PM, Fe and K (multivariate
                regression of the spectra on EC, OC, other PM, Fe, K) are projected out before PLS.
* iw            importance-weighted PLS: weights p/(1-p) from a logistic domain classifier (IMPROVE vs Addis spectra, 20 PCs).
* dipls_g{γ}    domain-invariant PLS (Nikzad-Langerodi 2018) with Addis as the unlabeled target; each weight vector is
                penalised by γ x |covariance difference| (scaled by its spectral norm); source centering for prediction.
* dipls_net     the same with the whole SPARTAN network as target (γ 3).

Writes output/tables/intercept_fixes_20261103/.

Stage ``aeth`` checks the reference itself: HIPS, published FTIR EC and the models against the MA350 aethalometer's IR BCc
at its four sites (ETAD, INDH, CHTS, USPA). The aethalometer is a separate instrument on a separate inlet, so it says which
of FTIR and HIPS follows Addis's day-to-day BC.

Stage ``hips`` asks whether HIPS saturates at heavy loading. Sampled volume is known exactly, so at a fixed concentration a
filter that drew more air carries more BC: a within-site negative effect of log volume on log(HIPS/FTIR) is a loading effect,
free of the errors-in-variables bias that a concentration regressor carries. A saturation constant Ls (HIPS = true / (1 +
true/Ls)) is fitted where that volume effect vanishes, on heavy sites with Ethiopia left out, and Addis is compared with HIPS
linearised by it.

Stage ``stress`` tests that reading, and it does not hold as stated. Delhi and Bhaktapur carry most of the volume signal, and
there volume is tied to concentration and to the sampling period; where volume is independent of concentration the effect
is small. So the stage also fits HIPS optical depth against aethalometer loading on the other sites and predicts Addis,
reads the Addis level from every anchor available, and asks whether the season effect survives the aethalometer.

    ../../.venv/bin/python workflows/run_intercept_fixes_20261103.py --bundle /tmp/xfer1102/bundle [--stages attrib fit tables aeth hips stress]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(AREA / "scripts"), str(AREA / "workflows"), str(AREA.parents[1])]
import continuum_anchor as ca  # noqa: E402
from aethmodular_cli import env  # noqa: E402
import run_transfer_diagnosis_20261102 as td  # noqa: E402

OUT = AREA / "output/tables/intercept_fixes_20261103"
XD = td.OUT
K = 20
LAMBDA_ADDIS = 2.96
GAMMAS = (1, 3, 10, 30)
CONT = ((3800, 3990), (2450, 2650), (1850, 2280))
ORG = ((2700, 3700), (1500, 1800))
WINDOW_EDGES = td.WINDOW_EDGES
# MA350 IR BCc per filter (overlap-weighted daily bins, >= 12 h; gallery/data/export_aethalometer.py, checked against HIPS 27 Oct)
AETH = AREA.parents[1] / "gallery/app/public/data/aethalometer"
LAMBDA_AETH = 1.0   # aethalometer error unknown: Deming with equal error variances (symmetric)


def deming(x, y, lam=LAMBDA_ADDIS):
    """Deming slope, intercept and R² (lam = error variance of y over that of x)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    xc, yc = x - x.mean(), y - y.mean()
    sxx, syy, sxy = xc @ xc, yc @ yc, xc @ yc
    d = syy - lam * sxx
    s = (d + np.sqrt(d * d + 4 * lam * sxy * sxy)) / (2 * sxy)
    return s, y.mean() - s * x.mean(), np.corrcoef(x, y)[0, 1] ** 2


def aeth_ir() -> pd.Series:
    """IR BCc (µg/m³, MA350 cross-section) for every filter matched to an aethalometer (ETAD, INDH, CHTS, USPA)."""
    fields = json.loads((AETH / "index.json").read_text())["fields"]
    col = fields.index("IR BCc (aethalometer)")
    out = {}
    for f in sorted((AETH / "filters").glob("*.json")):
        for fid, row in json.loads(f.read_text())["rows"].items():
            out[fid] = row[col]
    s = pd.Series(out, dtype=float)
    return s.where(s > 0)


def bin2(X, wn):
    n = (X.shape[1] // 2) * 2
    return np.asarray(X[:, :n], float).reshape(len(X), -1, 2).mean(2), wn[:n].reshape(-1, 2).mean(1)


def addis_frame(bundle):
    im, ad, nw = td.load(bundle)
    a = pd.read_csv(XD / "addis_predictions.csv")[["filter_id", "exp_tor_hips"]]
    ad = ad.merge(a, on="filter_id", how="left")
    ad["hips_bc"] = ad.fabs_Mm1 / 10
    ad["tor_equiv"] = ad.hips_bc * ad.exp_tor_hips
    ad["keep"] = ~ad.excluded & ad.fabs_Mm1.notna() & ad.exp_tor_hips.notna()
    ad["aeth_ir"] = ad.filter_id.map(aeth_ir())
    return im, ad, nw


# --------------------------------------------------------------------------- attrib
def attrib(bundle):
    """Per-window OLS slope/intercept of the Addis prediction against HIPS BC and the TOR yardstick (µg/m³)."""
    _, ad, _ = addis_frame(bundle)
    wn = np.load(bundle / "wn_full.npy")
    C, P = np.load(XD / "coefficients.npz"), np.load(XD / "predictions.npz")
    keep, v = ad.keep.to_numpy(), ad.volume_m3.to_numpy()
    rows = []
    for nm, X in [("vibes", td.spectra(bundle, "vibes")["addis"]), ("vibes_anchorhp", td.vc_spectra("anchorhp")["addis"])]:
        X = np.asarray(X, float)
        for k in (12, 18):
            b, xm = C[f"{nm}|B"][:, k - 1], C[f"{nm}|xm"]
            c = (X - xm) * b / v[:, None]
            const = P[f"{nm}|addis"][:, k - 1] / v - c.sum(1)        # y mean / volume
            parts = [(f"{hi}-{lo}", hi, lo, c[:, (wn <= hi) & (wn > lo)].sum(1)) for hi, lo in zip(WINDOW_EDGES[:-1], WINDOW_EDGES[1:])]
            parts.append(("mean term", np.nan, np.nan, const))
            for w, hi, lo, y in parts:
                r = dict(model=nm, k=k, window=w, hi=hi, lo=lo, mean=y[keep].mean())
                for xn in ("hips_bc", "tor_equiv"):
                    s, i = np.polyfit(ad[xn][keep], y[keep], 1)
                    r[f"slope_{xn}"], r[f"intercept_{xn}"] = s, i
                rows.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "window_gain.csv", index=False)
    print("attrib: window_gain.csv")


# --------------------------------------------------------------------------- models
def dipls_fit(Xs, y, Xt, kmax, gamma, w=None):
    """di-PLS1 (source and target each centred on their own mean for the covariance penalty; prediction uses source centering)."""
    xm, ym = Xs.mean(0), y.mean()
    E, f, T = Xs - xm, y - ym, Xt - Xt.mean(0)
    ns, nt, p = len(E), len(T), E.shape[1]
    W, P, q = np.zeros((p, kmax)), np.zeros((p, kmax)), np.zeros(kmax)
    for a in range(kmax):
        wv = E.T @ f / (f @ f)
        if gamma > 0:
            D = E.T @ E / (ns - 1) - T.T @ T / (nt - 1)
            ev, V = np.linalg.eigh(D)
            Dabs = (V * np.abs(ev)) @ V.T / np.abs(ev).max()
            wv = np.linalg.solve(np.eye(p) + gamma * Dabs, wv)
        wv /= np.linalg.norm(wv)
        t = E @ wv
        tt = t @ t
        P[:, a], q[a], W[:, a] = E.T @ t / tt, f @ t / tt, wv
        E -= np.outer(t, P[:, a])
        f = f - q[a] * t
        tT = T @ wv
        T -= np.outer(tT, T.T @ tT / max(tT @ tT, 1e-300))
    B = np.zeros((p, kmax))
    for k in range(1, kmax + 1):
        B[:, k - 1] = W[:, :k] @ np.linalg.solve(P[:, :k].T @ W[:, :k], q[:k])
    return xm, ym, B


def epo_basis(X, im, ok, bundle):
    """Orthonormal basis of the OC, other-PM, Fe and K spectral signatures (IMPROVE multivariate regression, per filter µg)."""
    v = im.volume_m3.to_numpy()
    oc = im.tor_oc_ugm3.to_numpy() * v
    ec = im.tor_ug.to_numpy()
    pm = pd.read_csv(bundle / "improve_mass.csv").set_index("filter_id").pm25_ugm3
    other = (im.filter_id.map(pm).to_numpy() * v) - 1.8 * oc - ec
    Z = np.column_stack([ec, oc, other, im.fe_ugm3.to_numpy() * v, im.k_xrf_ugm3.to_numpy() * v])
    m = ok & np.isfinite(Z).all(1)
    Zc = Z[m] - Z[m].mean(0)
    Xc = X[m] - X[m].mean(0)
    S = np.linalg.lstsq(Zc, Xc, rcond=None)[0]            # 5 x p signatures
    Q, _ = np.linalg.qr(S[1:].T)                          # everything but EC
    return Q, S, int(m.sum())


def domain_weights(XI, XA, n_pc=20):
    from sklearn.linear_model import LogisticRegression
    mu = XI.mean(0)
    _, _, Vt = np.linalg.svd(np.vstack([XI[::4], XA]) - mu, full_matrices=False)
    V = Vt[:n_pc].T
    Z = np.vstack([XI, XA]) @ V
    Z = (Z - Z.mean(0)) / Z.std(0)
    lab = np.r_[np.zeros(len(XI)), np.ones(len(XA))]
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Z, lab)
    p = clf.predict_proba(Z[: len(XI)])[:, 1]
    w = np.clip(p / (1 - p), 0, None)
    w = np.clip(w / w.mean(), 0.05, 20)
    return w / w.mean()



def fit(bundle):
    from sklearn.model_selection import GroupKFold

    im, ad, nw = addis_frame(bundle)
    y, sites = im.tor_ug.to_numpy(), im.site.to_numpy()
    ok = np.isfinite(y) & (y < 150)
    wn = np.load(bundle / "wn_full.npy")
    S = td.vc_spectra("anchorhp")
    XI, wb = bin2(S["improve"], wn)
    XA, _ = bin2(S["addis"], wn)
    XN, _ = bin2(S["network"], wn)
    cont = ca.window_mask(wb, CONT)
    noorg = ~ca.window_mask(wb, ORG)
    Q, sig, n_epo = epo_basis(XI, im, ok, bundle)
    proj = lambda X: X - (X @ Q) @ Q.T  # noqa: E731
    iw = domain_weights(XI, XA)
    ess = iw.sum() ** 2 / (iw ** 2).sum()

    cands = {"base": dict(), "cont_only": dict(cols=cont, k=10), "mask_org": dict(cols=noorg), "epo": dict(tf=proj),
             "iw": dict(w=iw)}
    cands.update({f"dipls_g{g}": dict(dipls=g, target="addis") for g in GAMMAS})
    cands["dipls_net"] = dict(dipls=3, target="network")
    preds, receipt = {}, {}
    for name, c in cands.items():
        t0 = time.time()
        kmax = c.get("k", K)
        cols = c.get("cols", np.ones(XI.shape[1], bool))
        tf = c.get("tf", lambda X: X)
        A, Ad, Nt = tf(XI)[:, cols], tf(XA)[:, cols], tf(XN)[:, cols]
        w = c.get("w")
        tgt = {"addis": Ad, "network": Nt}.get(c.get("target"))

        def fitone(rows):
            if "dipls" in c:
                return dipls_fit(A[rows], y[rows], tgt, kmax, c["dipls"])
            return ca.pls_fit(A[rows], y[rows], kmax, None if w is None else w[rows])

        oof = np.full((len(y), kmax), np.nan)
        for tr, te in GroupKFold(5).split(A, y, sites):
            tr = tr[ok[tr]]
            oof[te] = ca.pls_predict(fitone(tr), A[te])
        full = fitone(np.where(ok)[0])
        preds[f"{name}|improve_oof"], preds[f"{name}|addis"] = oof, ca.pls_predict(full, Ad)
        preds[f"{name}|network"], preds[f"{name}|B"], preds[f"{name}|xm"] = ca.pls_predict(full, Nt), full[2], full[0]
        preds[f"{name}|cols"] = cols
        receipt[name] = dict(k_max=kmax, channels=int(cols.sum()), seconds=round(time.time() - t0, 1))
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "predictions.npz", wn_bin2=wb, epo_signatures=sig, iw=iw, **preds)
    (OUT / "fit_receipt.json").write_text(json.dumps(dict(
        spectra="vibes_continuum_20261102 corrected_anchorhp_*, channel pairs averaged", cv="GroupKFold(5) by IMPROVE site",
        fit_rows=int(ok.sum()), epo_rows=n_epo, iw_effective_n=round(float(ess), 1), gammas=GAMMAS, cont_windows=CONT,
        organic_mask=ORG, models=receipt, addis_hips_used_for_fitting=False,
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1))


# --------------------------------------------------------------------------- tables
def tables(bundle):
    im, ad, nw = addis_frame(bundle)
    P = np.load(OUT / "predictions.npz")
    Pd = np.load(XD / "predictions.npz")
    y = im.tor_ug.to_numpy()
    fitok = np.isfinite(y) & (y < 150)
    heavy_low = fitok & (im.hips_ug > 20).to_numpy() & (im.ocec < 2.5).to_numpy()
    keep, v = ad.keep.to_numpy(), ad.volume_m3.to_numpy()
    yard = pd.read_csv(XD / "network_filters.csv")
    nrow = nw.reset_index().set_index("filter_id")["index"]
    yi = nrow.loc[yard.filter_id].to_numpy()
    names = [k.split("|")[0] for k in P.files if k.endswith("|addis")]
    srcs = [("vibes", Pd, 25), ("vibes_anchorhp_full", Pd, 25)] + [(n, P, P[f"{n}|addis"].shape[1]) for n in names]
    rows, addis_cols = [], {}
    for nm, src, kmax in srcs:
        key = "vibes_anchorhp" if nm == "vibes_anchorhp_full" else nm
        for k in range(1, kmax + 1):
            oof, pa, pn = src[f"{key}|improve_oof"][:, k - 1], src[f"{key}|addis"][:, k - 1] / v, src[f"{key}|network"][:, k - 1]
            r = dict(model=nm, k=k, improve_mae=np.median(np.abs(oof - y)[fitok]),
                     improve_heavy_lowocec_ratio=np.median((oof / y)[heavy_low]))
            for xn in ("hips_bc", "tor_equiv"):
                s, i, r2 = deming(ad[xn][keep], pa[keep])
                r[f"slope_{xn}"], r[f"intercept_{xn}"], r[f"ratio_{xn}"] = s, i, np.median(pa[keep] / ad[xn][keep])
            r["r2"] = r2
            ka = keep & ad.aeth_ir.notna().to_numpy()
            s, i, r2a = deming(ad.aeth_ir[ka], pa[ka], LAMBDA_AETH)
            r.update(slope_aeth=s, intercept_aeth=i, intercept_aeth_rel=i / pa[ka].mean(), r2_aeth=r2a,
                     loglog_aeth=np.polyfit(np.log(ad.aeth_ir[ka]), np.log(np.clip(pa[ka], 0.05, None)), 1)[0])
            rat = pd.Series(pn[yi] / yard.hips_ug.to_numpy() / yard.exp_tor_hips.to_numpy()).groupby(yard.code.to_numpy()).median()
            r["sites_median_abs_log"] = float(np.median(np.abs(np.log(rat.drop("ETAD", errors="ignore")))))
            r["sites_under_0p7"] = int((rat.drop("ETAD", errors="ignore") < 0.7).sum())
            rows.append(r)
            if k in (6, 10, 12, 18) and k <= kmax:
                addis_cols[f"{nm} k{k}"] = pa
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "summary.csv", index=False)
    a = ad[["filter_id", "date", "lot", "hips_bc", "tor_equiv", "exp_tor_hips", "aeth_ir", "excluded"]].copy()
    for c, vals in addis_cols.items():
        a[c] = vals
    a.to_csv(OUT / "addis_predictions.csv", index=False)
    pd.set_option("display.width", 200)
    sel = out[out.k.isin([6, 10, 12, 18])]
    print(sel[["model", "k", "improve_mae", "improve_heavy_lowocec_ratio", "slope_hips_bc", "intercept_hips_bc", "ratio_hips_bc",
               "slope_tor_equiv", "intercept_tor_equiv", "ratio_tor_equiv", "intercept_aeth", "loglog_aeth", "sites_median_abs_log"]].round(3).to_string(index=False))


# --------------------------------------------------------------------------- aeth
def aeth(bundle):
    """HIPS, published FTIR and the IMPROVE-trained models against the MA350 at its four sites (µg/m³)."""
    im, ad, nw = addis_frame(bundle)
    P, Pd = np.load(OUT / "predictions.npz"), np.load(XD / "predictions.npz")
    v = nw.volume_m3.to_numpy()
    ys = {"HIPS BC (MAC 10)": nw.fabs_Mm1.to_numpy() / 10, "SPARTAN published FTIR EC": nw.ec_ftir_ugm3.to_numpy(),
          "VIBES k18 (before)": Pd["vibes|network"][:, 17] / v, "VIBES anchored k18 (after)": Pd["vibes_anchorhp|network"][:, 17] / v,
          "EPO k18": P["epo|network"][:, 17] / v}
    nw["aeth_ir"] = nw.filter_id.map(aeth_ir())
    rows = []
    for code, g in nw[nw.aeth_ir.notna() & ~nw.excluded & nw.fabs_Mm1.notna() & nw.ec_ftir_ugm3.notna()].groupby("code"):
        x = g.aeth_ir.to_numpy()
        for nm, y in ys.items():
            y = y[g.index]
            ok = np.isfinite(y)
            s, i, r2 = deming(x[ok], y[ok], LAMBDA_AETH)
            lr = np.log(np.clip(y[ok], 0.05, None) / x[ok])
            rows.append(dict(code=code, n=int(ok.sum()), hips_ug=float(np.median(nw.fabs_Mm1[g.index] * v[g.index] / 10)),
                             aeth_median=float(np.median(x)), y=nm, slope=s, intercept=i, intercept_rel=i / y[ok].mean(), r2=r2,
                             loglog=np.polyfit(np.log(x[ok]), np.log(np.clip(y[ok], 0.05, None)), 1)[0],
                             # same denominator for every y: a noisy aethalometer would make every ratio fall with it
                             rho_ratio_vs_aeth=np.corrcoef(lr, np.log(x[ok]))[0, 1]))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "aeth_sites.csv", index=False)
    print(out.round(3).to_string(index=False))


# --------------------------------------------------------------------------- hips
def _volume_coef(g, ref, Ls=np.inf):
    """Within-site effect of log sampled volume on log(HIPS_c / FTIR) at fixed log concentration (site fixed effects).

    Volume is known exactly, so unlike concentration it carries no errors-in-variables bias: at a fixed BC concentration a
    filter that sampled more air carries more BC, and a loading (saturation) effect shows as a negative coefficient."""
    h = g.hips_ug.to_numpy()
    hc = h / np.clip(1 - h / Ls, 0.05, None)
    y = np.log(hc / g[ref].to_numpy())
    X = np.column_stack([np.log(g[ref] / g.volume_m3), np.log(g.volume_m3), pd.get_dummies(g.code).astype(float).to_numpy()])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    se = np.sqrt(np.diag(r @ r / (len(y) - X.shape[1]) * np.linalg.pinv(X.T @ X)))[1]
    return b[1], se


def _solve_ls(g, ref, lo=30.0, hi=1e6):
    """Saturation constant (µg per filter) at which the corrected HIPS has no volume effect; inf/lo when not bracketed."""
    f = lambda L: _volume_coef(g, ref, L)[0]  # noqa: E731
    a, b = np.log(lo), np.log(hi)
    fa, fb = f(np.exp(a)), f(np.exp(b))
    if fa * fb >= 0:
        return np.inf if fb < 0 else lo
    for _ in range(32):
        m = (a + b) / 2
        a, b = (m, b) if f(np.exp(m)) > 0 else (a, m)
    return float(np.exp((a + b) / 2))


def hips(bundle, n_boot=200):
    """Is HIPS saturating at heavy loading? Volume test, a saturation constant fitted without Ethiopia, and Addis against
    HIPS linearised with it (HIPS_c = HIPS / (1 - HIPS/Ls), per filter, MAC 10)."""
    im, ad, nw = addis_frame(bundle)
    Pd, P = np.load(XD / "predictions.npz"), np.load(OUT / "predictions.npz")
    nw["ftir_ug"] = Pd["vibes_anchorhp|network"][:, 17]
    nw["pub_ug"] = nw.ec_ftir_ugm3 * nw.volume_m3
    ok = (~nw.excluded) & (nw.hips_ug > 0) & (nw.ftir_ug > 0.5) & (nw.pub_ug > 0.5)
    g = nw[ok].copy()
    med = g.groupby("code").hips_ug.median()
    refs = {"ftir_ug": "VIBES anchored k18", "pub_ug": "SPARTAN published"}

    # 1. volume test by site loading class (sites chosen by their median, never by the filter's own HIPS) and single sites
    rows = []
    groups = [(f"sites with median {lo}–{hi} µg", med[(med >= lo) & (med < hi)].index) for lo, hi in ((0, 8), (8, 14), (14, 100))]
    groups += [(f"{c} only", [c]) for c in ("ETAD", "INDH", "BDDU", "AEAZ", "CLTA", "AUMN")]
    for name, codes in groups:
        sub = g[g.code.isin(codes)]
        sub = sub[sub.code.map(sub.code.value_counts()) >= 10]
        lv = np.log(sub.volume_m3)
        for ref, rl in refs.items():
            c, se = _volume_coef(sub, ref)
            rows.append(dict(group=name, reference=rl, n=len(sub), sites=sub.code.nunique(), median_hips_ug=float(sub.hips_ug.median()),
                             sd_log_volume_within=float((lv - lv.groupby(sub.code).transform("mean")).std()), coef=c, se=se))
    pd.DataFrame(rows).to_csv(OUT / "hips_volume_test.csv", index=False)

    # 2. saturation constant from heavy sites (median >= 10 µg), Ethiopia (ETAD, ETBI) left out; site bootstrap
    heavy = g[g.code.isin(med[med >= 10].index) & ~g.code.isin(["ETAD", "ETBI"])]
    heavy = heavy[heavy.code.map(heavy.code.value_counts()) >= 10]
    rng = np.random.default_rng(0)
    codes = heavy.code.unique()
    rows = []
    for ref, rl in refs.items():
        L = _solve_ls(heavy, ref)
        bs = []
        for _ in range(n_boot):
            pick = rng.choice(codes, len(codes))
            bg = pd.concat([heavy[heavy.code == c].assign(code=f"{c}_{i}") for i, c in enumerate(pick)])
            bs.append(_solve_ls(bg, ref))
        q = np.percentile(bs, [10, 50, 90])
        rows.append(dict(reference=rl, Ls_ug=L, boot_p10=q[0], boot_p50=q[1], boot_p90=q[2], sites=", ".join(sorted(codes)), n=len(heavy)))
        print(f"hips: Ls ({rl}) = {L:.0f} µg; bootstrap 10/50/90 {np.round(q, 0)}")
    sat = pd.DataFrame(rows)
    sat.to_csv(OUT / "hips_saturation.csv", index=False)
    Ls = float(sat.Ls_ug.mean())

    # 3. per site: HIPS-vs-FTIR log-log (reduced major axis) against median loading, with the prediction 1 - L/Ls
    ae = aeth_ir()
    rows = []
    for c, s in g.groupby("code"):
        if len(s) < 25:
            continue
        lh = np.log(s.hips_ug)
        r = dict(code=c, n=len(s), median_hips_ug=float(s.hips_ug.median()), predicted=1 - float(s.hips_ug.median()) / Ls)
        for ref, rl in refs.items():
            lf = np.log(s[ref])
            rr = np.corrcoef(lf, lh)[0, 1]
            r[f"rma_vs_{ref}"], r[f"r_{ref}"] = float(np.sign(rr) * lh.std() / lf.std()), float(rr)
        x = s.filter_id.map(ae)
        m = x.notna().to_numpy()
        if m.sum() >= 20:
            r["loglog_vs_aeth"] = float(np.polyfit(np.log(x[m]), np.log(s.fabs_Mm1[m] / 10), 1)[0])
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "hips_sites_loglog.csv", index=False)

    # 4. Addis against linearised HIPS over a grid of Ls (the level of HIPS_c is not trusted, the shape is)
    keep = ad.keep.to_numpy()
    h = (ad.fabs_Mm1 * ad.volume_m3 / 10).to_numpy()
    v = ad.volume_m3.to_numpy()
    models = {"SPARTAN published": ad.ec_published_ugm3.to_numpy(), "VIBES k18 (before)": Pd["vibes|addis"][:, 17] / v,
              "VIBES anchored k18 (after)": Pd["vibes_anchorhp|addis"][:, 17] / v, "EPO k18": P["epo|addis"][:, 17] / v}
    rows = []
    for L in sorted({40, 48, 55, 62, 66, 75, 90, 110, 144, 200, 300, 600, Ls, float(sat.boot_p10.mean()), float(sat.boot_p90.mean())} | {np.inf}):
        hc = h / np.clip(1 - h / L, 0.05, None) / v
        # HIPS error grows by d(HIPS_c)/d(HIPS) = 1 / (1 - HIPS/Ls)^2, so the error-variance ratio shrinks by its mean square
        lam = LAMBDA_ADDIS / np.mean(1 / np.clip(1 - h[keep] / L, 0.05, None) ** 4)
        ka = keep & ad.aeth_ir.notna().to_numpy()
        base = dict(Ls=L, hips_c_mean=float(hc[keep].mean()), hips_raw_mean=float((h / v)[keep].mean()),
                    hips_c_loglog_aeth=float(np.polyfit(np.log(ad.aeth_ir[ka]), np.log(hc[ka]), 1)[0]))
        for nm, y in models.items():
            k2 = keep & np.isfinite(y)
            s_, i_, r2 = deming(hc[k2], y[k2], lam)
            rows.append(dict(base, model=nm, slope=s_, intercept=i_, intercept_rel=i_ / y[k2].mean(), r2=r2, ratio=float(np.median(y[k2] / hc[k2]))))
    pd.DataFrame(rows).to_csv(OUT / "addis_vs_linearised_hips.csv", index=False)
    ad[["filter_id"]].assign(hips_lin=h / np.clip(1 - h / Ls, 0.05, None) / v).to_csv(OUT / "addis_hips_linearised.csv", index=False)
    (OUT / "hips_receipt.json").write_text(json.dumps(dict(Ls_used=Ls, form="HIPS_meas = true / (1 + true / Ls), per filter µg at MAC 10",
        fitted_on="sites with median HIPS >= 10 µg per filter, ETAD and ETBI left out", bootstrap=n_boot,
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1))
    print(f"hips: Ls used {Ls:.0f} µg; tables hips_*.csv, addis_vs_linearised_hips.csv")


# --------------------------------------------------------------------------- stress (is it saturation? what next?)
RAW_HIPS = AREA / "Filter Data/unified_filter_dataset.pkl"   # four sites only: HIPS_tau, HIPS_t, HIPS_r per filter
SEASON = {**{m: "Bega" for m in (10, 11, 12, 1)}, **{m: "Belg" for m in (2, 3, 4, 5)}, **{m: "Kiremt" for m in (6, 7, 8, 9)}}


def _within(s, ref, ctrl=None, quarter=False):
    """log(HIPS/ref) on log concentration and log volume, within site (or site x quarter). Returns coef, se, n."""
    c = ctrl or ref
    s = s[(s[ref] > 0) & (s[c] > 0)].copy()
    s["fe"] = s.code + ("|" + pd.to_datetime(s.date).dt.to_period("Q").astype(str) if quarter else "")
    s = s[s.fe.map(s.fe.value_counts()) >= 4]
    d = pd.DataFrame(dict(y=np.log(s.hips_ug / s[ref]), lc=np.log(s[c] / s.volume_m3), lv=np.log(s.volume_m3), fe=s.fe))
    d[["y", "lc", "lv"]] -= d.groupby("fe")[["y", "lc", "lv"]].transform("mean")
    X = d[["lc", "lv"]].to_numpy()
    b = np.linalg.lstsq(X, d.y, rcond=None)[0]
    r = d.y - X @ b
    se = np.sqrt(np.diag(r @ r / (len(d) - d.fe.nunique() - 2) * np.linalg.inv(X.T @ X)))
    return b[1], se[1], len(d)


def _curve_fit(f, d, p0):
    from scipy.optimize import least_squares
    return least_squares(lambda p: np.log(np.clip(f(p, d), 1e-4, None)) - np.log(d.tau), p0, bounds=(1e-6, 1e6)).x


def stress(bundle, n_boot=300):
    """Stress tests of the saturation reading, the HIPS optical curve against the aethalometer, the Addis level and seasons."""
    im, ad, nw = addis_frame(bundle)
    Pd = np.load(XD / "predictions.npz")
    nw["ftir_ug"] = Pd["vibes_anchorhp|network"][:, 17]
    nw["pub_ug"] = nw.ec_ftir_ugm3 * nw.volume_m3
    nw["pm_ug"] = nw.pm25_ugm3 * nw.volume_m3
    ok = (~nw.excluded) & (nw.hips_ug > 0) & (nw.ftir_ug > 0.5) & (nw.pub_ug > 0.5)
    g = nw[ok].copy()
    med = g.groupby("code").hips_ug.median()

    # 1. volume test: per heavy site, then sites where volume is independent of concentration vs those where it is not
    rows = []
    for c, s in g.groupby("code"):
        if len(s) < 25 or med[c] < 14:
            continue
        lv = np.log(s.volume_m3)
        rows.append(dict(code=c, n=len(s), median_hips_ug=med[c], sd_log_volume=lv.std(), r_volume_concentration=np.corrcoef(lv, np.log(s.fabs_Mm1))[0, 1],
                         coef_ftir=_within(s, "ftir_ug")[0], se_ftir=_within(s, "ftir_ug")[1], coef_pub=_within(s, "pub_ug")[0], se_pub=_within(s, "pub_ug")[1]))
    per = pd.DataFrame(rows).sort_values("sd_log_volume", ascending=False)
    per.to_csv(OUT / "hips_volume_by_site.csv", index=False)
    usable = per[per.sd_log_volume >= 0.1]
    clean, conf = usable[usable.r_volume_concentration.abs() < 0.3].code.tolist(), usable[usable.r_volume_concentration.abs() >= 0.3].code.tolist()
    rows = []
    for name, codes in (("volume independent of concentration", clean), ("volume tied to concentration", conf)):
        s = g[g.code.isin(codes)]
        for ref, rl in (("ftir_ug", "VIBES anchored k18"), ("pub_ug", "SPARTAN published")):
            for ctl, cl, q in ((None, "own concentration", False), ("pm_ug", "PM2.5 concentration", False), ("pm_ug", "PM2.5, site × quarter", True)):
                c_, se, n = _within(s, ref, ctl, q)
                rows.append(dict(group=name, sites=", ".join(codes), reference=rl, control=cl, coef=c_, se=se, n=n))
    pd.DataFrame(rows).to_csv(OUT / "hips_volume_stress.csv", index=False)

    # 2. HIPS optical depth against aethalometer loading at the four sites; curves fitted with Addis held out
    u = pd.read_pickle(RAW_HIPS)
    u = u[(u.DataSource == "HIPS") & (u.FilterType == "PM2.5") & u.Parameter.isin(["HIPS_tau", "HIPS_t", "HIPS_r"])]
    w = u.pivot_table(index="FilterId", columns="Parameter", values="Concentration", aggfunc="first").reset_index()
    w["filter_id"] = w.FilterId.str.replace(r"-\d$", "", regex=True)
    m = w.merge(nw[["filter_id", "code", "date", "volume_m3", "excluded", "ftir_ug", "pub_ug"]], on="filter_id")
    m["aeth"] = m.filter_id.map(aeth_ir())
    m = m[~m.excluded & m.aeth.notna() & (m.HIPS_tau > 0) & m.volume_m3.notna()].copy()
    m["L"], m["C"], m["V"], m["tau"] = m.aeth * m.volume_m3, m.aeth, m.volume_m3, m.HIPS_tau
    m["season"] = pd.to_datetime(m.date).dt.month.map(SEASON)
    m[["filter_id", "code", "date", "season", "aeth", "L", "volume_m3", "tau", "HIPS_t", "HIPS_r"]].rename(columns={"L": "aeth_ug"}).to_csv(OUT / "hips_tau_points.csv", index=False)
    fL = lambda p, d: p[0] * d.L / (1 + p[0] * d.L / p[1])  # noqa: E731
    fC = lambda p, d: p[0] * d.C / (1 + d.C / p[1]) * d.V / (100 * ca_area)  # noqa: E731
    fO = lambda p, d: p[0] * d.L  # noqa: E731
    ca_area = 3.53
    test = m[m.code == "ETAD"]
    rows, curves = [], {}
    for name, tr in (("Beijing + Pasadena + Delhi", m[m.code != "ETAD"]), ("Pasadena + Delhi", m[m.code.isin(["USPA", "INDH"])]), ("Beijing + Delhi", m[m.code.isin(["CHTS", "INDH"])])):
        for model, f, p0 in (("proportional", fO, [.02]), ("saturating with loading", fL, [.03, 2]), ("saturating with concentration", fC, [10, 30])):
            p = _curve_fit(f, tr, p0)
            e = np.log(np.clip(f(p, test), 1e-4, None)) - np.log(test.tau)
            rows.append(dict(trained_on=name, n=len(tr), model=model, p1=p[0], p2=p[1] if len(p) > 1 else np.nan, addis_bias_log=e.mean(), addis_rmse_log=np.sqrt((e ** 2).mean()),
                             predicted_loglog=np.polyfit(np.log(test.L), np.log(f(p, test)), 1)[0], observed_loglog=np.polyfit(np.log(test.L), np.log(test.tau), 1)[0]))
    hold = pd.DataFrame(rows)
    hold.to_csv(OUT / "hips_tau_holdout.csv", index=False)
    tr = m[m.code != "ETAD"]
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        b = pd.concat([x.sample(len(x), replace=True, random_state=int(rng.integers(1 << 31))) for _, x in tr.groupby("code")])
        boot.append(_curve_fit(fL, b, [.03, 2]))
    boot = np.array(boot)

    # 3. the Addis level: how far HIPS sits below the curve's straight start, and every way of reading FTIR's level
    p3 = hold[(hold.trained_on == "Beijing + Pasadena + Delhi") & (hold.model == "saturating with loading")].iloc[0]
    rows = []
    for name, a_, ti in [(f"curve, {r.trained_on}", r.p1, r.p2) for r in hold[hold.model == "saturating with loading"].itertuples()]:
        rows.append(dict(anchor=name, addis_true_over_hips=float(np.median(1 / np.clip(1 - test.tau / ti, 0.05, None)))))
    bridge = []
    for c, s in m.groupby("code"):
        lo = s[s.aeth <= s.aeth.quantile(.33)]
        bridge.append(dict(code=c, n=len(s), hips_over_aeth=float((s.tau * 35.3 / s.L).median()), hips_over_aeth_lightest_third=float((lo.tau * 35.3 / lo.L).median()),
                           ftir_over_aeth=float((s.ftir_ug / s.L).median()), published_over_aeth=float((s.pub_ug / s.L).median())))
    bridge = pd.DataFrame(bridge)
    bridge.to_csv(OUT / "aeth_bridge.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "addis_true_over_hips.csv", index=False)

    # 4. seasons at Addis once the aethalometer is taken out (log y = log-log x aethalometer + season)
    v = ad.volume_m3.to_numpy()
    a = ad[ad.keep & ad.aeth_ir.notna()].copy()
    a["season"] = pd.to_datetime(a.date).dt.month.map(SEASON)
    ys = {"HIPS BC": a.hips_bc, "VIBES k18 (before)": pd.Series(Pd["vibes|addis"][:, 17] / v, index=ad.index)[a.index],
          "VIBES anchored k18 (after)": pd.Series(Pd["vibes_anchorhp|addis"][:, 17] / v, index=ad.index)[a.index], "SPARTAN published": a.ec_published_ugm3}
    rows = []
    for nm, y in ys.items():
        k = (y > 0).to_numpy()
        D = pd.get_dummies(a.season[k]).astype(float)[["Bega", "Belg", "Kiremt"]].to_numpy()
        X = np.column_stack([np.log(a.aeth_ir[k]), D])
        ly = np.log(y[k]).to_numpy()
        b = np.linalg.lstsq(X, ly, rcond=None)[0]
        r = ly - X @ b
        C = r @ r / (k.sum() - 4) * np.linalg.inv(X.T @ X)
        rows.append(dict(y=nm, n=int(k.sum()), loglog=b[0], bega_vs_belg=np.exp(b[1] - b[2]) - 1, bega_se=np.sqrt(C[1, 1] + C[2, 2] - 2 * C[1, 2]),
                         kiremt_vs_belg=np.exp(b[3] - b[2]) - 1, kiremt_se=np.sqrt(C[3, 3] + C[2, 2] - 2 * C[2, 3])))
    pd.DataFrame(rows).to_csv(OUT / "addis_season_vs_aeth.csv", index=False)

    # 5. size checks
    e = nw[(nw.code == "ETAD") & ~nw.excluded]
    soil = 2.2 * e.al_ugm3 + 2.49 * e.si_ugm3 + 2.42 * e.fe_ugm3
    x = test.L.to_numpy()
    aic = lambda d: (lambda r: len(x) * np.log(r @ r / len(x)) + 2 * (d + 2))(test.tau.to_numpy() - np.polyval(np.polyfit(x, test.tau, d), x))  # noqa: E731
    (OUT / "stress_receipt.json").write_text(json.dumps(dict(
        clean_sites=clean, confounded_sites=conf, soil_median_ugm3=float(soil.median()), soil_absorption_Mm1_at_0p05=float(soil.median() * .05),
        hips_offset_Mm1=28.0, tau_inf_boot=np.percentile(boot[:, 1], [10, 50, 90]).tolist(), a_boot=np.percentile(boot[:, 0], [10, 50, 90]).tolist(),
        curve=dict(a=float(p3.p1), tau_inf=float(p3.p2)), addis_tau_dAIC_quadratic_minus_linear=float(aic(2) - aic(1)), raw_hips=env.display_path(RAW_HIPS),
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1))
    print(hold.round(3).to_string(index=False))
    print(pd.read_csv(OUT / "addis_true_over_hips.csv").round(2).to_string(index=False))
    print(bridge.round(2).to_string(index=False))
    print(pd.read_csv(OUT / "addis_season_vs_aeth.csv").round(3).to_string(index=False))
    print(pd.read_csv(OUT / "hips_volume_stress.csv").drop(columns="sites").round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, default=None)
    p.add_argument("--stages", nargs="+", default=["attrib", "fit", "tables"], choices=["attrib", "fit", "tables", "aeth", "hips", "stress"])
    a = p.parse_args()
    bundle = a.bundle or td.default_bundle()
    for s in a.stages:
        {"attrib": attrib, "fit": fit, "tables": tables, "aeth": aeth, "hips": hips, "stress": stress}[s](bundle)


if __name__ == "__main__":
    main()
