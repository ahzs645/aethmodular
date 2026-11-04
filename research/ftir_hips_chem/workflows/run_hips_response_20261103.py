"""How far can HIPS be trusted, across SPARTAN and IMPROVE, and what do the TOR filters we hold say? (3 Nov 2026)

Follows run_intercept_fixes_20261103.py, whose tables and helpers it reuses. Stages:

* ``flow``     is the aethalometer usable, given its flow split? The MA350 target is Flow2:Flow1 = 1:3 (Flow1/Flow2 = 3). Every
               site starts near 55:45; Beijing, Pasadena and Delhi were moved towards the target, and Addis (MA350-0238) never was.
               Flow on each site's filter days, every comparison repeated on the uncompensated spots (BC1, BC2), the Pasadena
               before/after and the Addis flow-ratio test. config.FLOW_FIX_PERIODS and FlowFix_BeforeAfter_Analysis.ipynb call the
               fixed state "degraded" and give 1.0 as ideal; that wording is out of date.
* ``network``  HIPS against each reference by optical depth, within sites, over the whole of IMPROVE and SPARTAN. Filters are
               binned on sqrt(tau_HIPS x tau_reference) so neither side is selected on. A response g(tau) = 1 / (1 + (tau - tau0)+ / c)
               is fitted on IMPROVE TOR alone (no aethalometer, no FTIR) with a site bootstrap, then applied to SPARTAN.
* ``tor``      the five Adama PTFE + quartz pairs: real VIBES on the anchored spectra, every model against TOR, Fabs/TOR, the
               spectrum-to-filter map sensitivity and the draft pass rule; then the rule's behaviour on simulated draws of about
               10 IMPROVE filters held out of their site.
* ``options``  per-filter anchored and EPO FTIR EC for the gallery's FTIR baseline switch.

    ../../.venv/bin/python workflows/run_hips_response_20261103.py --bundle /tmp/xfer1102/bundle --adama <dir> [--stages flow network tor options]

``--adama`` holds the batch 54 files copied from Drive (csu_amod_*, Carbon_concs_Batch54.csv). No Addis HIPS, TOR or aethalometer
value is used to fit anything here.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path[:0] = [str(AREA / "scripts"), str(AREA / "workflows"), str(ROOT)]
import continuum_anchor as ca  # noqa: E402
import run_intercept_fixes_20261103 as ix  # noqa: E402
import run_transfer_diagnosis_20261102 as td  # noqa: E402

OUT = ix.OUT
XD = td.OUT
AREA_SPARTAN, AREA_IMPROVE = 3.53, 3.5
SITES = {"ETAD": "Addis_Ababa", "INDH": "Delhi", "CHTS": "Beijing", "USPA": "JPL"}
TAU_BINS = [.05, .1, .2, .3, .4, .5, .6, .7, .8, 1, 1.25, 1.6, 3]
PASS_LOW, PASS_HIGH, GATE_UG, OCEC_LIMIT = 0.8, 1.25, 15.0, 2.5


# --------------------------------------------------------------------------- flow
def flow(bundle):
    _, ad, nw = ix.addis_frame(bundle)
    P = np.load(XD / "predictions.npz")
    nw["hips"], nw["pub"] = nw.fabs_Mm1 / 10, nw.ec_ftir_ugm3
    nw["anch"], nw["vib"] = P["vibes_anchorhp|network"][:, 17] / nw.volume_m3, P["vibes|network"][:, 17] / nw.volume_m3
    windows = json.loads((ROOT / "gallery/app/public/data/weather/filters.json").read_text())["windows"]
    cols = {"BCc": "IR BCc", "BCc_smoothed": "IR BCc smoothed", "BC1": "IR BC1 smoothed", "BC2": "IR BC2 smoothed", "F1": "Flow1 (mL/min)", "F2": "Flow2 (mL/min)"}
    rows = []
    for code, name in SITES.items():
        d = pd.read_pickle(AREA / f"processed_sites/df_{name}_9am_resampled.pkl")
        d = d.loc[:, ~d.columns.duplicated()]
        t = pd.to_datetime(d["datetime_local"]).dt.tz_localize(None)
        a, b = t.to_numpy(), (t + pd.Timedelta(hours=24)).to_numpy()
        V = np.column_stack([pd.to_numeric(d[c], errors="coerce") for c in cols.values()])
        for fid, w in windows.items():
            if not fid.startswith(code + "-"):
                continue
            lo, hi = np.datetime64(pd.Timestamp(w[0]).tz_localize(None)), np.datetime64(pd.Timestamp(w[1]).tz_localize(None))
            h = (np.minimum(b, hi) - np.maximum(a, lo)) / np.timedelta64(1, "h")
            u = h > 0
            if u.any() and h[u].sum() >= 12:
                v, k = V[u], np.isfinite(V[u])
                rows.append(dict(filter_id=fid, code=code, **dict(zip(cols, np.nansum(np.where(k, v, 0) * h[u][:, None], 0) / np.maximum((k * h[u][:, None]).sum(0), 1e-9)))))
    a = pd.DataFrame(rows).merge(nw[["filter_id", "hips", "anch", "vib", "pub", "excluded"]], on="filter_id")
    a = a[~a.excluded & a.hips.notna() & (a.BCc > 0) & (a.BC1 > 0) & (a.BC2 > 0)]
    a["ratio"] = a.F1 / a.F2
    g = a.groupby("code")
    fl = pd.concat([g.size().rename("n"), g.ratio.median().rename("flow_ratio"), g.ratio.quantile(.1).rename("p10"), g.ratio.quantile(.9).rename("p90"),
                    (a.BCc_smoothed / a.BC1).groupby(a.code).median().rename("bcc_over_bc1")], axis=1).reset_index()
    # the MA350 target is Flow2:Flow1 = 1:3, so Flow1/Flow2 = 3 (75:25). Sites start near 55:45 (1.2) until the flow fix
    fl["f1_share"] = 100 * fl.flow_ratio / (1 + fl.flow_ratio)
    fl["flow_status"] = np.select([fl.flow_ratio < 1.5, fl.flow_ratio < 2.7], ["unfixed", "fixed, below 1:3"], "near 1:3")
    fl.loc[fl.code == "INDH", "flow_status"] = "rising, fix unconfirmed"     # no step in Delhi's record: 1.9 to 2.9 across it
    fl.to_csv(OUT / "aeth_flow_by_site.csv", index=False)

    # what the split does: Pasadena is the only site with filters on both sides of its fix; Addis has only its small wander
    u = a[a.code == "USPA"].copy()
    u["era"], u["month"] = np.where(u.ratio < 1.5, "before fix", "after fix"), pd.to_datetime(u.filter_id.map(nw.set_index("filter_id").date)).dt.month
    rows = []
    for label, s in (("all months", u), ("June to September only", u[u.month.isin([6, 7, 8, 9])])):
        for era, g in s.groupby("era"):
            rows.append(dict(months=label, era=era, n=len(g), flow_ratio=g.ratio.median(), hips_over_bcc=(1000 * g.hips / g.BCc).median(), hips_over_bc1=(1000 * g.hips / g.BC1).median(),   # spot BC is in ng/m3
                             published_over_bcc=(1000 * g["pub"] / g.BCc).median(), published_over_bc1=(1000 * g["pub"] / g.BC1).median(), bcc_over_bc1=(g.BCc_smoothed / g.BC1).median()))
    pd.DataFrame(rows).to_csv(OUT / "aeth_fix_pasadena.csv", index=False)
    e = a[a.code == "ETAD"]
    tests = dict(bcc_over_bc1=(e.BCc_smoothed / e.BC1).quantile([.1, .5, .9]).tolist(), rho_compensation_vs_bc1=float(np.corrcoef(np.log(e.BCc_smoothed / e.BC1), np.log(e.BC1))[0, 1]),
                 flow_ratio_p05_p95=e.ratio.quantile([.05, .95]).tolist())
    # does the split change the filter / aethalometer ratio once BCc's dual-spot compensation is applied? Same fit on the
    # compensated BCc and the uncompensated spots, at every site, so BCc is judged on its own record
    a["date"] = pd.to_datetime(a.filter_id.map(nw.set_index("filter_id").date))
    rows = []
    for (c, x, y), s0 in ((k, a[a.code == k[0]]) for k in itertools.product(a.code.unique(), ("BCc", "BC1", "BC2"), ("hips", "pub", "anch"))):
        s = s0[s0[y] > .05]
        if len(s) < 12 or s.ratio.std() < .02:
            continue
        lr = np.log(1000 * s[y] / s[x]).to_numpy()
        for controls, X in (("loading", [np.log(s[x])]), ("loading and month", [np.log(s[x])] + [(s.date.dt.month == m).astype(float) for m in sorted(s.date.dt.month.unique())[1:]])):
            X = np.column_stack([s.ratio, *X, np.ones(len(s))])
            if np.linalg.matrix_rank(X) < X.shape[1] or len(s) - X.shape[1] < 6:
                continue
            b = np.linalg.lstsq(X, lr, rcond=None)[0]
            r = lr - X @ b
            se = float(np.sqrt(np.diag(r @ r / (len(s) - X.shape[1]) * np.linalg.inv(X.T @ X)))[0])
            span = float(s.ratio.quantile(.9) - s.ratio.quantile(.1))
            rows.append(dict(code=c, x=x, y=y, controls=controls, n=len(s), ratio_p10=s.ratio.quantile(.1), ratio_p90=s.ratio.quantile(.9), coef=b[0], se=se,
                             pct_over_range=100 * (np.exp(b[0] * span) - 1), pct_lo=100 * (np.exp((b[0] - 1.645 * se) * span) - 1), pct_hi=100 * (np.exp((b[0] + 1.645 * se) * span) - 1)))
    pd.DataFrame(rows).to_csv(OUT / "aeth_flow_coef.csv", index=False)
    # the same comparison paired: how far BCc/BC1 itself moves with the split, which needs no filter at all
    tests["compensation_vs_flow"] = {}
    for c, s in a.groupby("code"):
        X, lr = np.column_stack([s.ratio, np.log(s.BC1), np.ones(len(s))]), np.log(s.BCc / s.BC1).to_numpy()
        b = np.linalg.lstsq(X, lr, rcond=None)[0]
        r = lr - X @ b
        se = float(np.sqrt(np.diag(r @ r / (len(s) - 3) * np.linalg.inv(X.T @ X)))[0])
        span = float(s.ratio.quantile(.9) - s.ratio.quantile(.1))
        tests["compensation_vs_flow"][c] = [float(100 * (np.exp(k * span) - 1)) for k in (b[0], b[0] - 1.645 * se, b[0] + 1.645 * se)]
    (OUT / "aeth_flow_tests.json").write_text(json.dumps(tests, indent=1))
    rows = []
    for c, s0 in a.groupby("code"):
        for x, y in itertools.product(("BCc", "BC1", "BC2"), ("hips", "pub", "anch", "vib")):
            s = s0[(s0[y] > .05)]
            lx, ly = np.log(s[x]), np.log(s[y])
            sl, it = np.polyfit(s[x], s[y], 1)
            rows.append(dict(code=c, x=x, y=y, n=len(s), loglog=np.polyfit(lx, ly, 1)[0], intercept_rel=it / s[y].mean(), r=np.corrcoef(lx, ly)[0, 1], rho_ratio=np.corrcoef(ly - lx, lx)[0, 1]))
    pd.DataFrame(rows).to_csv(OUT / "aeth_spot_sensitivity.csv", index=False)
    print(fl.round(2).to_string(index=False))


# --------------------------------------------------------------------------- network
def _demean(y, groups):
    s = np.bincount(groups, y) / np.bincount(groups)
    return y - s[groups]


def _codes(x):
    return pd.factorize(x)[0]


def fit_response(d, area):
    """g(s) = 1 / (1 + (s - tau0)+ / c), log(HIPS/TOR) with site effects; s = geometric mean of the two optical depths."""
    from scipy.optimize import least_squares
    s, y, g = np.sqrt(d.h * d.ref).to_numpy() / (10 * area), np.log(d.h / d.ref).to_numpy(), _codes(d.site)
    res = lambda p: _demean(y + np.log1p(np.clip(s - p[0], 0, None) / p[1]), g)  # noqa: E731
    start = min(itertools.product((.15, .25, .3, .4, .5), (.6, 1, 1.4, 2)), key=lambda q: (res(q) ** 2).sum())
    return least_squares(res, start, bounds=([.05, .2], [.9, 6])).x


def correction(h_ug, p, area=AREA_SPARTAN):
    """true / HIPS at a measured HIPS loading: g is defined on s, and tau_HIPS = s sqrt(g(s)), so invert on a grid."""
    s = np.linspace(.01, 8, 4000)
    g = 1 / (1 + np.clip(s - p[0], 0, None) / p[1])
    return np.interp(np.asarray(h_ug, float) / (10 * area), s * np.sqrt(g), 1 / g)


def _curve(d, area, name):
    """Median-free stitching: log(HIPS/ref) on tau-bin dummies with site effects, relative to the 0.2-0.3 bin."""
    d = d[(d.h > 0) & (d.ref > .3)].copy()
    d["s"] = np.sqrt(d.h * d.ref) / (10 * area)
    d["b"] = pd.cut(d.s, TAU_BINS)
    d = d[d.b.notna()]
    d = d[d.site.map(d.site.value_counts()) >= 8]
    d["lab"] = d.b.astype(str)
    D = pd.get_dummies(d.lab).astype(float)
    base = "(0.2, 0.3]"
    X = pd.concat([D.drop(columns=base), pd.get_dummies(d.site).astype(float)], axis=1).to_numpy()
    y = np.log(d.h / d.ref).to_numpy()
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    se = np.sqrt(np.diag(r @ r / (len(y) - X.shape[1]) * np.linalg.pinv(X.T @ X)))
    names = list(D.drop(columns=base).columns)
    rows = []
    for c in sorted(D.columns, key=lambda k: float(k.strip("(]").split(",")[0])):
        lo, hi = (float(v) for v in c.strip("(]").split(","))
        v, e = (0.0, 0.0) if c == base else (b[names.index(c)], se[names.index(c)])
        rows.append(dict(curve=name, lo=lo, hi=hi, n=int(D[c].sum()), ratio=float(np.exp(v)), ratio_lo=float(np.exp(v - 1.645 * e)), ratio_hi=float(np.exp(v + 1.645 * e))))
    return rows


def network(bundle, n_boot=300):
    im, ad, nw = ix.addis_frame(bundle)
    P = np.load(XD / "predictions.npz")
    im["h"], im["ref"], im["ocec2"] = im.fabs_Mm1 * im.volume_m3 / 10, im.tor_ug, im.tor_oc_ugm3 / im.tor_ec_ugm3
    im["oof"] = P["vibes_anchorhp|improve_oof"][:, 17]
    nw["h"], nw["site"] = nw.fabs_Mm1 * nw.volume_m3 / 10, nw.code
    nw["anch"], nw["vib"], nw["pub"] = P["vibes_anchorhp|network"][:, 17], P["vibes|network"][:, 17], nw.ec_ftir_ugm3 * nw.volume_m3
    I = im[np.isfinite(im.ref) & (im.ref < 150)]
    op = I[I.tor_optr_ugm3 > 0]
    n = nw[~nw.excluded & (nw.h > 0)]
    noeth = n[~n.code.isin(["ETAD", "ETBI"])]
    rows = _curve(op, AREA_IMPROVE, "IMPROVE: HIPS / TOR")
    rows += _curve(op[op.ocec2 < OCEC_LIMIT], AREA_IMPROVE, "IMPROVE: HIPS / TOR, OC/EC < 2.5")
    rows += _curve(I.assign(ref=I.oof), AREA_IMPROVE, "IMPROVE: HIPS / FTIR, out of site")
    rows += _curve(noeth.assign(ref=noeth.anch), AREA_SPARTAN, "SPARTAN without Ethiopia: HIPS / anchored FTIR")
    rows += _curve(noeth.assign(ref=noeth["pub"]), AREA_SPARTAN, "SPARTAN without Ethiopia: HIPS / published FTIR")
    pd.DataFrame(rows).to_csv(OUT / "hips_tau_curves.csv", index=False)

    use = op[(op.h > 0) & (op.ref > .3)]
    use = use[use.site.map(use.site.value_counts()) >= 8]
    sets = {"all filters": use, "OC/EC < 2.5": use[use.ocec2 < OCEC_LIMIT]}
    rng = np.random.default_rng(0)
    fits, pars = [], {}
    for name, d in sets.items():
        p = fit_response(d, AREA_IMPROVE)
        pars[name] = p
        sites = d.site.unique()
        idx = {s: g for s, g in d.groupby("site")}
        boot = []
        for _ in range(n_boot):
            pick = rng.choice(sites, len(sites))
            boot.append(fit_response(pd.concat([idx[s].assign(site=f"{s}_{i}") for i, s in enumerate(pick)]), AREA_IMPROVE))
        q = np.percentile(boot, [10, 90], axis=0)
        fits.append(dict(fit=name, n=len(d), sites=len(sites), tau0=p[0], tau0_lo=q[0, 0], tau0_hi=q[1, 0], c=p[1], c_lo=q[0, 1], c_hi=q[1, 1]))
    fit = pd.DataFrame(fits)
    fit.to_csv(OUT / "hips_response_fit.csv", index=False)

    # Addis and every site, raw HIPS against HIPS on the IMPROVE TOR response
    ae = ix.aeth_ir()
    n = n.copy()
    n["aeth"] = n.filter_id.map(ae)
    e = n[n.code == "ETAD"]
    rows = []
    for name, p in pars.items():
        f = correction(e.h, p)
        hc = e.h * f / e.volume_m3
        ka = e.aeth.notna().to_numpy()
        base = dict(fit=name, correction_median=float(np.median(f)), correction_p05=float(np.quantile(f, .05)), correction_p95=float(np.quantile(f, .95)),
                    hips_loglog_aeth=np.polyfit(np.log(e.aeth[ka]), np.log(hc[ka]), 1)[0], raw_loglog_aeth=np.polyfit(np.log(e.aeth[ka]), np.log((e.h / e.volume_m3)[ka]), 1)[0])
        for m, lab in (("vib", "VIBES k18 (before)"), ("anch", "VIBES anchored k18 (after)"), ("pub", "SPARTAN published")):
            k = (e[m] > 0).to_numpy()
            y = (e[m] / e.volume_m3)[k]
            s0, i0, _ = ix.deming((e.h / e.volume_m3)[k], y, ix.LAMBDA_ADDIS)
            s1, i1, _ = ix.deming(hc[k], y, ix.LAMBDA_ADDIS / np.mean(f[k] ** 4))
            rows.append(dict(base, model=lab, raw_slope=s0, raw_intercept=i0, slope=s1, intercept=i1, raw_ratio=float(np.median(e[m][k] / e.h[k])), ratio=float(np.median(e[m][k] / (e.h[k] * f[k])))))
    pd.DataFrame(rows).to_csv(OUT / "addis_vs_hips_response.csv", index=False)
    pa = pars["all filters"]
    pd.DataFrame(dict(filter_id=e.filter_id, hips_resp=e.h * correction(e.h, pa) / e.volume_m3, hips_resp_low=e.h * correction(e.h, pars["OC/EC < 2.5"]) / e.volume_m3)).to_csv(OUT / "addis_hips_response.csv", index=False)

    rows = []
    s_all = n[(n.anch > .5)]
    for c, s in s_all.groupby("code"):
        if len(s) < 25:
            continue
        v = s.volume_m3
        y = s.anch / v
        h0, h1 = s.h / v, s.h * correction(s.h, pa) / v
        a0, b0, _ = ix.deming(h0, y, 1.0)
        a1, b1, _ = ix.deming(h1, y, 1.0)
        t = s.h / (10 * AREA_SPARTAN)
        rows.append(dict(code=c, n=len(s), tau_median=float(t.median()), share_over_0p3=float((t > .3).mean()), share_over_0p5=float((t > .5).mean()), share_over_1=float((t > 1).mean()),
                         correction_median=float(np.median(correction(s.h, pa))), raw_slope=a0, raw_intercept_rel=b0 / y.mean(), slope=a1, intercept_rel=b1 / y.mean()))
    t = pd.DataFrame(rows).sort_values("tau_median", ascending=False)
    t.to_csv(OUT / "hips_site_response.csv", index=False)
    heavy = t[t.tau_median > .35]
    (OUT / "hips_response_receipt.json").write_text(json.dumps(dict(
        form="HIPS/true = g(s) = 1/(1 + (s - tau0)+/c), s = sqrt(tau_HIPS tau_true); fitted on IMPROVE TOR, OP > 0, site effects",
        heavy_sites=len(heavy), heavy_abs_intercept_raw=float(heavy.raw_intercept_rel.abs().median()), heavy_abs_intercept_corrected=float(heavy.intercept_rel.abs().median()),
        improve_n_over_0p5=int((I.h / 35 > .5).sum()), improve_n_over_1=int((I.h / 35 > 1).sum()), bootstrap=n_boot,
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1))
    print(fit.round(3).to_string(index=False))
    print(pd.read_csv(OUT / "addis_vs_hips_response.csv").round(3).to_string(index=False))


# --------------------------------------------------------------------------- tor
def _on_grid(frame, wn):
    src = np.array([float(c) for c in frame.columns[1:]])
    i = np.abs(src[:, None] - wn[None, :]).argmin(axis=0)
    if len(set(i)) != len(wn) or np.abs(src[i] - wn).max() > 1e-3:
        raise ValueError("spectral grid mismatch")
    return frame.iloc[:, 0].astype(str).tolist(), frame.iloc[:, 1:].to_numpy(float)[:, i]


def _rule(ratio, rng, n_boot=2000):
    """The proposed rule on a set of filters that already passed the gates: pass, fail (low or high) or inconclusive, with the 90 % interval."""
    ratio = np.asarray(ratio, float)
    b = np.median(rng.choice(ratio, (n_boot, len(ratio))), axis=1)
    lo, hi = np.percentile(b, [5, 95])
    return ("pass" if lo >= PASS_LOW and hi <= PASS_HIGH else "fail" if hi < PASS_LOW or lo > PASS_HIGH else "inconclusive"), lo, hi


def tor(bundle, adama: Path):
    from threadpoolctl import threadpool_limits
    from vibes_baseline import VibesBackground, vibes_baseline_matrix
    import vibes_continuum_20261102 as vc

    wn = np.load(bundle / "wn_full.npy")
    z = np.load(td.VIBES_MODEL)
    C, Pd = np.load(XD / "coefficients.npz"), np.load(XD / "predictions.npz")
    Px = np.load(OUT / "predictions.npz")
    ids, Xraw = _on_grid(pd.read_csv(adama / "csu_amod_Batch_54_ShipDate_2026-05-29_spectra.csv"), wn)
    Xanc = ca.AnchoredBackground(wn, z["mean"], z["components"]).correct(Xraw).astype(float)
    m, W = vc.backgrounds(wn, z["mean"], z["components"])["anchorhp"]
    bg = VibesBackground(wn, m, W, pd.DataFrame(), tuple(), 0.0)
    with threadpool_limits(limits=4):
        _, Xhp, diag = vibes_baseline_matrix(wn, Xanc, bg, sample_ids=ids, tau=0.1, loss="PB", maxiter=10000)
    if not diag.success.astype(bool).all():
        raise RuntimeError("VIBES failed on an Adama spectrum")
    Xvib = np.load(AREA / "output/tables/vibes_fullrange/gallery_rank70_20261002/external/corrected_full.npy")[:5].astype(float)   # production rank 70

    def pred(nm, X, ref, k=18):
        b, xm = C[f"{nm}|B"][:, k - 1], C[f"{nm}|xm"]
        return (X - xm) @ b + np.median(Pd[f"{nm}|addis"][:, k - 1] - (ref - xm) @ b)

    addis_h = np.load(ix.td.VC / "corrected_anchorhp_addis.npy").astype(float)
    preds = {"VIBES k18 (before)": pred("vibes", Xvib, np.load(bundle / "addis_vibes.npy").astype(float)),
             "VIBES anchored k18 (after)": pred("vibes_anchorhp", Xhp, addis_h),
             "raw k18": pred("raw", Xraw, np.load(bundle / "addis_raw.npy").astype(float))}
    Xb, _ = ix.bin2(Xhp, wn)
    Ab, _ = ix.bin2(addis_h, wn)
    Q, _r = np.linalg.qr(Px["epo_signatures"][1:].T)
    proj = lambda A: A - (A @ Q) @ Q.T  # noqa: E731
    be, xe = Px["epo|B"][:, 17], Px["epo|xm"]
    ye = np.median(Px["epo|addis"][:, 17] - (proj(Ab) - xe) @ be)
    preds["EPO k18"] = (proj(Xb) - xe) @ be + ye

    meta = pd.read_csv(bundle / "adama_meta.csv")
    hips = pd.read_csv(adama / "csu_amod_HIPS_Batch_54.csv")[["FilterId", "tau", "Fabs", "Uncertainty"]]
    d = meta.merge(hips, left_on="ptfe_id", right_on="FilterId")
    d["analysis"] = d.sample_id.str.split(":").str[1]
    d["unflagged"] = d.flag.isna()
    order = [ids.index(a) for a in d.analysis]
    d["tor_ug"] = d.ec_tr_ugm3 * d.volume_m3
    for nm, v in preds.items():
        d[nm] = v[order] / d.volume_m3
    d["ocec"] = d.oc_tr_ugm3 / d.ec_tr_ugm3
    d["fabs_over_tor"] = d.Fabs / d.ec_tr_ugm3
    d["hips_bc"] = d.Fabs / 10
    d.drop(columns=["lat", "lon", "device", "pairing_status", "FilterId", "sample_id"], errors="ignore").to_csv(OUT / "adama_tor_test.csv", index=False)

    # summaries: all five and the three unflagged, plus every spectrum-to-filter assignment (the map is provisional)
    rng = np.random.default_rng(0)
    rows = []
    un = d.unflagged.to_numpy()
    for nm, v in preds.items():
        v = np.asarray(v) / 1.0
        per = np.array([np.median((v[list(p)][un] / d.volume_m3.to_numpy()[un]) / d.ec_tr_ugm3.to_numpy()[un]) for p in itertools.permutations(range(5))])
        r = (d[nm] / d.ec_tr_ugm3).to_numpy()
        rows.append(dict(model=nm, median_all=float(np.median(r)), median_unflagged=float(np.median(r[un])), mapped_lo=float(per.min()), mapped_hi=float(per.max()),
                         mean_diff_ug_unflagged=float(((d[nm] - d.ec_tr_ugm3) * d.volume_m3)[un].mean())))
    pd.DataFrame(rows).to_csv(OUT / "adama_tor_summary.csv", index=False)

    # the rule on what we hold: both gates decide before any ratio is read
    eligible = (d.tor_ug >= GATE_UG) & (d.ocec < OCEC_LIMIT)

    # how the rule behaves with about 10 filters: IMPROVE filters like Addis, each predicted out of its site
    im, _, _ = ix.addis_frame(bundle)
    y = im.tor_ug.to_numpy()
    ok = np.isfinite(y) & (y >= GATE_UG) & (y < 150) & (im.tor_oc_ugm3 / im.tor_ec_ugm3 < OCEC_LIMIT).to_numpy()
    models = {"VIBES k18 (before)": Pd["vibes|improve_oof"][:, 17], "VIBES anchored k18 (after)": Pd["vibes_anchorhp|improve_oof"][:, 17], "EPO k18": Px["epo|improve_oof"][:, 17]}
    rows = []
    for nm, oof in models.items():
        r = (oof / y)[ok]
        r = r[np.isfinite(r)]
        for n_f in (5, 10, 20):
            for true_level in (1.3, 1.0, 0.8, 0.6):
                out = {"pass": 0, "fail": 0, "inconclusive": 0}
                adj = r / np.median(r) * true_level     # the rule's question: given this scatter, what happens if the true level is L?
                for _ in range(1000):
                    out[_rule(rng.choice(adj, n_f, replace=False), rng, 400)[0]] += 1
                rows.append(dict(model=nm, n_filters=n_f, true_level=true_level, eligible=int(len(r)), observed_median=float(np.median(r)), **{k: v / 1000 for k, v in out.items()}))
    pd.DataFrame(rows).to_csv(OUT / "pass_rule_simulation.csv", index=False)
    (OUT / "tor_receipt.json").write_text(json.dumps(dict(
        adama_filters=len(d), adama_eligible=int(eligible.sum()), adama_max_tor_ug=float(d.tor_ug.max()), adama_median_ocec=float(d.ocec.median()), gate_ug=GATE_UG, ocec_limit=OCEC_LIMIT,
        pass_band=[PASS_LOW, PASS_HIGH], fabs_over_tor_median=float(d.fabs_over_tor.median()), fabs_over_tor_unflagged=float(d.fabs_over_tor[un].median()),
        adama_tau_range=[float(d.tau.min()), float(d.tau.max())], improve_eligible=int(ok.sum()), addis_city_filters_in_hand=0,
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1))
    print(d[["date", "ptfe_id", "flag", "tau", "ec_tr_ugm3", "tor_ug", "ocec", "hips_bc", "fabs_over_tor"] + list(preds)].round(2).to_string(index=False))
    print(pd.read_csv(OUT / "adama_tor_summary.csv").round(2).to_string(index=False))
    print(pd.read_csv(OUT / "pass_rule_simulation.csv").query("n_filters == 10").round(2).to_string(index=False))


# --------------------------------------------------------------------------- options
def options(bundle):
    _, _, nw = ix.addis_frame(bundle)
    Pd, Px = np.load(XD / "predictions.npz"), np.load(OUT / "predictions.npz")
    v = nw.volume_m3.where(nw.volume_m3 > 0).to_numpy()      # a few filters log zero volume: no concentration, not infinity
    out = pd.DataFrame(dict(filter_id=nw.filter_id, anchored=Pd["vibes_anchorhp|network"][:, 17] / v, epo=Px["epo|network"][:, 17] / v))
    out[out.anchored.notna()].to_csv(OUT / "network_candidates.csv", index=False)
    print(f"options: {out.anchored.notna().sum()} filters")

    # the scoreboard behind "why this baseline": IMPROVE-held-out tests only need the out-of-site predictions
    im, ad, _ = ix.addis_frame(bundle)
    y = im.tor_ug.to_numpy()
    oe = (im.tor_oc_ugm3 / im.tor_ec_ugm3).to_numpy()
    fit, like = np.isfinite(y) & (y < 150), np.isfinite(y) & (y >= GATE_UG) & (y < 150) & (oe < OCEC_LIMIT)
    light = np.isfinite(y) & (y > 3.5) & (y < 10) & (oe > 4.5) & (oe < 8)
    oof = {"raw": Pd["raw|improve_oof"][:, 17], "spline": Pd["spline|improve_oof"][:, 17], "vibes": Pd["vibes|improve_oof"][:, 17], "vibes_deflate": Pd["vibes_deflate|improve_oof"][:, 17],
           "vibes_anchorhp": Pd["vibes_anchorhp|improve_oof"][:, 17], "epo": Px["epo|improve_oof"][:, 17]}
    pd.DataFrame([dict(model=m, improve_error=float(np.median(np.abs(o - y)[fit])), improve_heavy_ratio=float(np.median((o / y)[like])), improve_light_ratio=float(np.median((o / y)[light])),
                       n_heavy=int(like.sum()), n_light=int(light.sum())) for m, o in oof.items()]).to_csv(OUT / "baseline_scoreboard.csv", index=False)

    # one Addis spectrum story: heavy and light thirds, raw and each corrected spectrum (every 4th channel)
    wn = np.load(bundle / "wn_full.npy")
    raw, vib = np.load(bundle / "addis_raw.npy", mmap_mode="r"), np.load(bundle / "addis_vibes.npy", mmap_mode="r")
    hp = np.load(td.VC / "corrected_anchorhp_addis.npy", mmap_mode="r")
    h = ad.hips_bc.to_numpy()
    k = ad.keep.to_numpy()
    lo, hi = np.nanquantile(h[k], [1 / 3, 2 / 3])
    rows = {"wn": wn[::4]}
    for name, m in (("light", k & (h <= lo)), ("heavy", k & (h >= hi))):
        rows[f"raw_{name}"] = np.median(np.asarray(raw)[m], 0)[::4]
        rows[f"vibes_{name}"] = np.median(np.asarray(vib)[m], 0)[::4]
        rows[f"anchored_{name}"] = np.median(np.asarray(hp)[m], 0)[::4]
    pd.DataFrame(rows).to_csv(OUT / "addis_spectra_story.csv", index=False)

    # is anchored carrying a lot offset at Addis? (log FTIR/aethalometer by lot, within season, season held fixed)
    a = ad[ad.keep & ad.aeth_ir.notna()].copy()
    v = ad.volume_m3.to_numpy()
    for nm, col in (("VIBES k18 (before)", "vibes"), ("VIBES anchored k18 (after)", "vibes_anchorhp")):
        a[nm] = pd.Series(Pd[f"{col}|addis"][:, 17] / v, index=ad.index)[a.index]
    a["EPO k18"] = pd.Series(Px["epo|addis"][:, 17] / v, index=ad.index)[a.index]
    a["season"] = pd.to_datetime(a.date).dt.month.map(ix.SEASON)
    rows = []
    for nm in ("VIBES k18 (before)", "VIBES anchored k18 (after)", "EPO k18"):
        s = a[a[nm] > 0]
        lr = np.log(s[nm] / s.aeth_ir)
        for (lot, se), g in s.groupby(["lot", "season"]):
            rows.append(dict(model=nm, lot=str(lot), season=se, n=len(g), ratio=float(np.exp(lr[g.index].median()))))
    pd.DataFrame(rows).to_csv(OUT / "addis_lot_vs_aeth.csv", index=False)
    print(pd.DataFrame(rows).pivot_table(index=["lot", "season"], columns="model", values="ratio").round(2).join(pd.DataFrame(rows).groupby(["lot", "season"]).n.first()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--adama", type=Path, default=None)
    p.add_argument("--stages", nargs="+", default=["flow", "network", "tor", "options"], choices=["flow", "network", "tor", "options"])
    a = p.parse_args()
    for s in a.stages:
        tor(a.bundle, a.adama) if s == "tor" else {"flow": flow, "network": network, "options": options}[s](a.bundle)


if __name__ == "__main__":
    main()
