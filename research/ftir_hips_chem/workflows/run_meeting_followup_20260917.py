"""Tasks from the 2026-09-17 Ann/Satoshi meeting (Ann's edited list, 2026-09-18).

Run from the repo root:  uv run python research/ftir_hips_chem/workflows/run_meeting_followup_20260917.py

Stages (all fast; every PLS fit is ~0.3 s on a 500-filter cohort):
  selections  seasonal correlation analogs, both baselines, locked + meeting masks
  seasonal    one calibration per season (protocol site split); analogs AND Addis
              predicted by the same calibration
  allimprove  the all-IMPROVE lot-251 model (and the frozen both-lots Colab model)
              predicting the seasonal analogs and Addis
  combined    union of the three seasonal analog sets as one calibration, against
              VIP/Euclidean analogs, the seasonal models and the all-IMPROVE model
  repeats     100 independent site splits per season: slope/intercept spread
The long category grid is ``run_meeting_grid_20260917.py``.

Nothing here tunes on Addis outcomes. Addis crossplots are FTIR EC vs HIPS
Fabs/MAC with MAC fixed at 10 (a comparison proxy, not thermal EC).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / "scripts"))
import analog_bias as ab  # noqa: E402

OUT = AREA / "output/tables/meeting_followup_20260917"
OUT.mkdir(parents=True, exist_ok=True)
MASKS = {"locked": ab.LOCKED_MASK, "meeting": ab.MEETING_MASK}
N_REPEATS = 100
LOT = 251   # 191 of the 233 Addis filters are lot 251; Ann: "the all-IMPROVE samples from that lot"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(frame, name):
    frame.to_csv(OUT / f"{name}.csv", index=False)


def slug(s):
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


REF_COLS = ["filter_id", "Site", "lot", "split", "date", "y", "tor_volume_m3", "tor_ec_ugm3",
            "tor_oc_ugm3", "tor_ec1_ugm3", "tor_optr_ugm3", "tor_optt_ugm3", "fabs_Mm1",
            "tor_oc_ec", "mac_ratio", "op_tor_frac"]


def ref_rel_uncertainty(ctx):
    """Relative 1-sigma of a reference measurement, from ETAD HIPS_Uncertainty.

    IMPROVE exports carry no per-filter fAbs or EC uncertainty, so the ETAD HIPS
    relative uncertainty (median sigma / median Fabs) stands in for both HIPS and
    TOR on the IMPROVE panels. It is small next to FTIR's held-out error, so the
    FTIR panels' Deming lambda is large and Deming sits close to OLS, the same
    logic as the Addis panels (lambda* = 3.3).
    """
    return ctx.sigma_x / float(np.median(ctx.addis.Fabs.to_numpy(float) / ab.MAC_VALUE))


def analog_panels(ctx, rows, pred_ugm3, rmse_ug=None, boot=True):
    """The three IMPROVE comparisons Ann asked for, on one set of analog rows.

    ``rmse_ug`` is the calibration's held-out TOR RMSE (µg/filter): FTIR's error
    for the Deming error ratio on the two FTIR panels.
    """
    lib = ctx.lib.iloc[rows]
    tor = lib.tor_ec_ugm3.to_numpy(float)
    fabs10 = lib.fabs_Mm1.to_numpy(float) / ab.MAC_VALUE
    sites = lib.Site.to_numpy()
    rel = ref_rel_uncertainty(ctx)
    s_fabs, s_tor = rel * np.nanmedian(fabs10), rel * np.nanmedian(tor)
    s_ftir = (rmse_ug / np.nanmedian(lib.tor_volume_m3)) if rmse_ug else s_tor

    def fit(x, y, sx, sy):
        st = ab.xy_stats(x, y, lam=(sy / sx) ** 2, sigma_x=sx, sigma_y=sy, groups=sites, boot=boot)
        st["deming_lambda"] = (sy / sx) ** 2
        return st

    out = {
        "tor_vs_fabs": fit(fabs10, tor, s_fabs, s_tor),
        "ftir_vs_tor": fit(tor, pred_ugm3, s_tor, s_ftir),
        "ftir_vs_fabs": fit(fabs10, pred_ugm3, s_fabs, s_ftir),
    }
    t = out["tor_vs_fabs"]
    if "deming_slope" in t:
        t["implied_mac_deming"] = ab.MAC_VALUE / t["deming_slope"]
        ok = np.isfinite(tor) & np.isfinite(fabs10) & (tor > 0)
        t["mac_ratio_median"] = float(np.median(lib.fabs_Mm1.to_numpy(float)[ok] / tor[ok]))
        t["mac_percentile_median"] = float(np.median(
            ab.mac_percentile(ctx, lib.fabs_Mm1.to_numpy(float)[ok], tor[ok])))
    return out


def addis_panel(ctx, take, pred_ugm3, boot=True):
    a = ctx.addis.loc[take]
    return ab.xy_stats(a.Fabs.to_numpy(float) / ab.MAC_VALUE, pred_ugm3[take],
                       lam=ctx.lam, sigma_x=ctx.sigma_x, sigma_y=ab.SIGMA_Y_PROXY,
                       groups=a.month_block.to_numpy(), boot=boot)


def stat_rows(base, panels):
    return [{**base, "panel": p, **st} for p, st in panels.items()]


# --------------------------------------------------------------------------- stages
def stage_selections(ctx, gm):
    sel, members, overlaps = {}, [], []
    for method in ctx.methods:
        for mname, mask in MASKS.items():
            for g, take in gm.items():
                pos, score = ab.correlation_analogs(ctx, method, take, mask)
                sel[(method, mname, g)] = pos
                f = ctx.lib.iloc[pos][["filter_id", "Site", "lot", "split"]].copy()
                f["method"], f["mask"], f["group"] = method, mname, g
                f["rank"] = np.arange(1, len(pos) + 1)
                f["pearson_r"] = score
                members.append(f)
            for i, a in enumerate(ab.SEASONS):
                for b in ab.SEASONS[i + 1:]:
                    sa = set(ctx.lib.filter_id.values[sel[(method, mname, a)]])
                    sb = set(ctx.lib.filter_id.values[sel[(method, mname, b)]])
                    overlaps.append({"method": method, "mask": mname, "a": a, "b": b,
                                     "shared": len(sa & sb)})
        for g in gm:
            s1 = set(ctx.lib.filter_id.values[sel[(method, "locked", g)]])
            s2 = set(ctx.lib.filter_id.values[sel[(method, "meeting", g)]])
            overlaps.append({"method": method, "mask": "locked_vs_meeting", "a": g, "b": g,
                             "shared": len(s1 & s2)})
    for g in gm:
        s1 = set(ctx.lib.filter_id.values[sel[("AIRSpec", "locked", g)]])
        s2 = set(ctx.lib.filter_id.values[sel[("VIBES", "locked", g)]])
        overlaps.append({"method": "AIRSpec_vs_VIBES", "mask": "locked", "a": g, "b": g,
                         "shared": len(s1 & s2)})
    save(pd.concat(members, ignore_index=True), "analog_membership")
    save(pd.DataFrame(overlaps), "analog_overlap")
    return sel


def stage_seasonal(ctx, gm, sel):
    """One calibration per selection group; Addis and its analogs predicted by it."""
    fits, stats, analog_pred, addis_pred = [], [], [], []
    vol = ctx.addis.Volume_m3.to_numpy(float)
    for method in ctx.methods:
        for mname in MASKS:
            for g in gm:
                pos = sel[(method, mname, g)]
                lib = ctx.lib.iloc[pos]
                f = ab.fit_cohort(ctx.X[method][pos], lib.y, lib.Site)
                fit_id = f"seasonal|{method}|{mname}|{g}"
                conc = f.predict(ctx.X[method][pos]) / lib.tor_volume_m3.to_numpy(float)
                pa = f.predict(ctx.XA[method]) / vol
                test = ~f.train
                rec = {"fit_id": fit_id, "family": "seasonal", "method": method, "mask": mname,
                       "group": g, "k": f.k, "cohort_n": len(pos), "train_n": int(f.train.sum()),
                       "test_n": int(test.sum()), "test_sites": int(lib.Site[test].nunique())}
                tor_test = ab.xy_stats(lib.y.to_numpy()[test], f.predict(ctx.X[method][pos])[test])
                rec["TOR_test_R2"] = tor_test.get("R2")
                rec["TOR_test_RMSE_ug"] = tor_test.get("RMSE")
                fits.append(rec)
                base = {k: rec[k] for k in ("fit_id", "family", "method", "mask", "group", "k")}
                full = mname == "locked"
                rmse = rec["TOR_test_RMSE_ug"]
                stats += stat_rows({**base, "population": "analog_test"},
                                   analog_panels(ctx, pos[test], conc[test], rmse, boot=full))
                stats += stat_rows({**base, "population": "analog_train"},
                                   analog_panels(ctx, pos[f.train], conc[f.train], rmse, boot=False))
                evals = [g] if g != "All Addis" else list(gm)
                for eg in evals:
                    stats.append({**base, "population": "addis", "eval_group": eg,
                                  "panel": "ftir_vs_fabs", **addis_panel(ctx, gm[eg], pa, boot=full)})
                a = lib[REF_COLS].copy()
                a["role"] = np.where(f.train, "train", "test")
                a["ftir_ec_ugm3"] = conc
                a["fit_id"] = fit_id
                analog_pred.append(a)
                p = ctx.addis[["MediaId", "ExternalFilterId", "date", "season", "Fabs", "month_block"]].copy()
                p["ftir_ec_ugm3"], p["fit_id"] = pa, fit_id
                addis_pred.append(p)
                log(f"{fit_id}: k={f.k} TOR R2={rec['TOR_test_R2']:.3f}")
    return fits, stats, analog_pred, addis_pred


def frozen_full_pool(method):
    z = np.load(ab.RUN / f"pls_full_pool_{method}.npz")
    coef, xm, b = z["coefficient"].ravel(), z["x_mean"], float(z["intercept"][0])
    return lambda X: (np.asarray(X, float) - xm) @ coef + b, int(z["components"])


def stage_allimprove(ctx, gm, sel):
    """Ann: predict the analogs with the all-IMPROVE model for that lot."""
    fits, stats, analog_pred, addis_pred = [], [], [], []
    vol = ctx.addis.Volume_m3.to_numpy(float)
    lot = ctx.lib.lot.eq(LOT).to_numpy()
    for method in ctx.methods:
        idx = np.flatnonzero(lot)
        sub = ctx.lib.iloc[idx]
        t0 = time.time()
        f = ab.fit_cohort(ctx.X[method][idx], sub.y, sub.Site,
                          train=sub.split.eq("train").to_numpy())
        predictors = {f"all-IMPROVE lot {LOT}": (f.predict, f.k, int(f.train.sum()), True)}
        if method in ab.METHODS:   # the frozen Colab full-pool models exist for the first run's baselines only
            predictors["all-IMPROVE both lots (frozen Colab)"] = (*frozen_full_pool(method), 10066, False)
        log(f"all-IMPROVE lot {LOT} {method}: k={f.k} in {time.time() - t0:.0f}s")
        for label, (predict, k, train_n, lot_only) in predictors.items():
            fit_id = f"allimprove|{method}|{label}"
            conc_all = predict(ctx.X[method]) / ctx.lib.tor_volume_m3.to_numpy(float)
            pa = predict(ctx.XA[method]) / vol
            pool_test = np.flatnonzero(ctx.lib.split.eq("test").to_numpy() & (lot | (not lot_only)))
            gtest = ab.xy_stats(ctx.lib.y.to_numpy()[pool_test],
                                predict(ctx.X[method][pool_test]))
            fits.append({"fit_id": fit_id, "family": "allimprove", "method": method,
                         "mask": "n/a", "group": label, "k": k,
                         "train_n": train_n,
                         "test_n": len(pool_test), "TOR_test_R2": gtest.get("R2"),
                         "TOR_test_RMSE_ug": gtest.get("RMSE")})
            base = {"fit_id": fit_id, "family": "allimprove", "method": method, "mask": "locked",
                    "k": k}
            rmse = gtest.get("RMSE")
            stats += stat_rows({**base, "group": "IMPROVE pool", "population": "pool_test"},
                               analog_panels(ctx, pool_test, conc_all[pool_test], rmse, boot=False))
            for g in gm:
                pos = sel[(method, "locked", g)]
                # analogs in the model's own test split were never seen by it
                held = pos[ctx.lib.split.to_numpy()[pos] == "test"]
                seen = pos[ctx.lib.split.to_numpy()[pos] == "train"]
                stats += stat_rows({**base, "group": g, "population": "analog_test"},
                                   analog_panels(ctx, held, conc_all[held], rmse))
                stats += stat_rows({**base, "group": g, "population": "analog_all"},
                                   analog_panels(ctx, pos, conc_all[pos], rmse, boot=False))
                stats += stat_rows({**base, "group": g, "population": "analog_train"},
                                   analog_panels(ctx, seen, conc_all[seen], rmse, boot=False))
                stats.append({**base, "group": g, "population": "addis", "eval_group": g,
                              "panel": "ftir_vs_fabs", **addis_panel(ctx, gm[g], pa)})
                a = ctx.lib.iloc[pos][REF_COLS].copy()
                a["role"] = np.where(ctx.lib.split.to_numpy()[pos] == "test", "test", "train")
                a["ftir_ec_ugm3"] = conc_all[pos]
                a["fit_id"] = f"{fit_id}|{g}"
                analog_pred.append(a)
            p = ctx.addis[["MediaId", "ExternalFilterId", "date", "season", "Fabs", "month_block"]].copy()
            p["ftir_ec_ugm3"], p["fit_id"] = pa, fit_id
            addis_pred.append(p)
    return fits, stats, analog_pred, addis_pred


def vip_analog_positions(ctx, n=500):
    """The earlier VIP/Euclidean analogs (corrected-space ranking, explorer cache)."""
    z = np.load(ROOT / "calibration_explorer/cache/analog_corrected_ranking.npz")
    p = np.load(ROOT / "research/ftir_ec_phase3/output/corrected/improve_pool_corrected_df6.npz",
                allow_pickle=True)
    a2f = dict(zip(p["analysis_id"].astype(int), p["filter_id"].astype(int)))
    f2pos = pd.Series(np.arange(len(ctx.lib)), index=ctx.lib.filter_id)
    ranked = [a2f[int(a)] for a in z["ids"] if int(a) in a2f]
    ranked = [f for f in dict.fromkeys(ranked) if f in f2pos.index]
    return f2pos.loc[ranked[:n]].to_numpy(int)


def stage_combined(ctx, gm, sel):
    fits, stats, addis_pred = [], [], []
    vol = ctx.addis.Volume_m3.to_numpy(float)
    for method in ctx.methods:
        union = np.array(list(dict.fromkeys(
            np.concatenate([sel[(method, "locked", s)] for s in ab.SEASONS]).tolist())), int)
        cohorts = {"three seasonal sets combined": union}
        if method == "AIRSpec":
            cohorts["VIP/Euclidean analogs (500)"] = vip_analog_positions(ctx)
        for label, pos in cohorts.items():
            lib = ctx.lib.iloc[pos]
            f = ab.fit_cohort(ctx.X[method][pos], lib.y, lib.Site)
            fit_id = f"combined|{method}|{label}"
            pa = f.predict(ctx.XA[method]) / vol
            test = ~f.train
            tt = ab.xy_stats(lib.y.to_numpy()[test], f.predict(ctx.X[method][pos])[test])
            fits.append({"fit_id": fit_id, "family": "combined", "method": method,
                         "mask": "locked", "group": label, "k": f.k, "cohort_n": len(pos),
                         "train_n": int(f.train.sum()), "test_n": int(test.sum()),
                         "TOR_test_R2": tt.get("R2"), "TOR_test_RMSE_ug": tt.get("RMSE")})
            conc = f.predict(ctx.X[method][pos]) / lib.tor_volume_m3.to_numpy(float)
            base = {"fit_id": fit_id, "family": "combined", "method": method, "mask": "locked",
                    "group": label, "k": f.k}
            stats += stat_rows({**base, "population": "analog_test"},
                               analog_panels(ctx, pos[test], conc[test], tt.get("RMSE")))
            for eg in gm:
                stats.append({**base, "population": "addis", "eval_group": eg,
                              "panel": "ftir_vs_fabs", **addis_panel(ctx, gm[eg], pa)})
            p = ctx.addis[["MediaId", "ExternalFilterId", "date", "season", "Fabs", "month_block"]].copy()
            p["ftir_ec_ugm3"], p["fit_id"] = pa, fit_id
            addis_pred.append(p)
            log(f"{fit_id}: n={len(pos)} k={f.k}")
    return fits, stats, addis_pred


def _one_repeat(ctx, gm, method, g, pos, r, train):
    lib = ctx.lib.iloc[pos]
    f = ab.fit_cohort(ctx.X[method][pos], lib.y, lib.Site, train=train)
    conc = f.predict(ctx.X[method][pos]) / lib.tor_volume_m3.to_numpy(float)
    test = ~train
    pa = f.predict(ctx.XA[method][gm[g]]) / ctx.addis.Volume_m3.to_numpy(float)[gm[g]]
    x = ctx.addis.Fabs.to_numpy(float)[gm[g]] / ab.MAC_VALUE
    ad = ab.xy_stats(x, pa, sigma_x=ctx.sigma_x, sigma_y=ab.SIGMA_Y_PROXY)
    tt = ab.xy_stats(lib.y.to_numpy()[test], f.predict(ctx.X[method][pos])[test])
    panels = analog_panels(ctx, pos[test], conc[test], tt.get("RMSE"), boot=False)
    row = {"method": method, "group": g, "repeat": r, "k": f.k, "test_n": int(test.sum()),
           "TOR_test_R2": tt.get("R2"),
           "addis_slope": ad["deming_slope"], "addis_intercept": ad["deming_intercept"],
           "addis_R2": ad["R2"]}
    for p, st in panels.items():
        row[f"{p}_slope"] = st.get("deming_slope")
        row[f"{p}_intercept"] = st.get("deming_intercept")
        row[f"{p}_ols_slope"] = st.get("ols_slope")
    return row, np.flatnonzero(gm[g]), pa


def stage_repeats(ctx, gm, sel):
    from joblib import Parallel, delayed

    jobs = []
    for method in ctx.methods:
        for g in gm:
            pos = sel[(method, "locked", g)]
            sites = ctx.lib.Site.to_numpy()[pos]
            for r, train in enumerate(ab.repeated_site_splits(sites, N_REPEATS)):
                jobs.append((method, g, pos, r, train))
    log(f"repeats: {len(jobs)} fits")
    out = Parallel(n_jobs=12, backend="threading")(
        delayed(_one_repeat)(ctx, gm, *j) for j in jobs)
    rows = [o[0] for o in out]
    # Stitch: split r of each season predicts that season's own Addis filters; the
    # three seasons together give one all-Addis crossplot per split.
    x = ctx.addis.Fabs.to_numpy(float) / ab.MAC_VALUE
    stitched = []
    for method in ctx.methods:
        for r in range(N_REPEATS):
            pred = np.full(len(ctx.addis), np.nan)
            for (row, idx, pa) in out:
                if row["method"] == method and row["repeat"] == r and row["group"] in ab.SEASONS:
                    pred[idx] = pa
            st = ab.xy_stats(x, pred, sigma_x=ctx.sigma_x, sigma_y=ab.SIGMA_Y_PROXY)
            stitched.append({"method": method, "repeat": r, "n": st["n"], "addis_slope": st["deming_slope"],
                             "addis_intercept": st["deming_intercept"], "addis_R2": st["R2"]})
    save(pd.DataFrame(stitched), "stitched_repeats")
    return pd.DataFrame(rows)


def stage_stitched(ctx, gm, addis_pred):
    """Each season's Addis filters predicted by that season's own calibration.

    Also the currently deployed SPARTAN calibration (EC_deployed), the poster's
    starting point. Both are read out exactly like every other Addis fit.
    """
    stats, preds = [], []
    frame = pd.concat(addis_pred, ignore_index=True)
    variants = {}
    for method in ctx.methods:
        pred = np.full(len(ctx.addis), np.nan)
        for g in ab.SEASONS:
            p = frame.loc[frame.fit_id.eq(f"seasonal|{method}|locked|{g}")].set_index("MediaId").prediction_ugm3 \
                if "prediction_ugm3" in frame else frame.loc[frame.fit_id.eq(f"seasonal|{method}|locked|{g}")].set_index("MediaId").ftir_ec_ugm3
            take = gm[g]
            pred[take] = p.loc[ctx.addis.MediaId[take]].to_numpy(float)
        variants[f"stitched|{method}|season calibrations"] = pred
    variants["deployed|SPARTAN|current SPARTAN calibration"] = ctx.addis.EC_deployed_ugm3.to_numpy(float)
    for fit_id, pred in variants.items():
        fam, method, label = fit_id.split("|")
        base = {"fit_id": fit_id, "family": fam, "method": method, "mask": "locked", "group": label, "k": None}
        for eg in gm:
            stats.append({**base, "population": "addis", "eval_group": eg, "panel": "ftir_vs_fabs",
                          **addis_panel(ctx, gm[eg], pred)})
        p = ctx.addis[["MediaId", "ExternalFilterId", "date", "season", "Fabs", "month_block"]].copy()
        p["ftir_ec_ugm3"], p["fit_id"] = pred, fit_id
        preds.append(p)
        log(f"{fit_id}: all-Addis Deming {stats[-4]['deming_slope']:.2f} / {stats[-4]['deming_intercept']:+.2f}")
    return stats, preds


def stage_op(ctx, sel):
    """Ann: does charring (OP) track where the analogs sit on TOR EC vs HIPS?"""
    from scipy.stats import spearmanr

    lib = ctx.lib
    ok = (lib.tor_ec_ugm3 > 0.02) & (lib.fabs_Mm1 > 0) & lib.op_tor_frac.notna()
    rows, terciles = [], []
    sets = {("pool", "IMPROVE pool"): np.flatnonzero(ok.to_numpy())}
    for method in ctx.methods:
        for g in ab.group_masks(ctx):
            pos = sel[(method, "locked", g)]
            sets[(method, g)] = pos[ok.to_numpy()[pos]]
    for (method, g), pos in sets.items():
        d = lib.iloc[pos]
        lmac = np.log(d.fabs_Mm1 / d.tor_ec_ugm3)
        rho = spearmanr(d.op_tor_frac, lmac).correlation
        rows.append({"method": method, "group": g, "n": len(d), "spearman_op_vs_log_mac": rho,
                     "op_frac_median": d.op_tor_frac.median(),
                     "optt_over_ec_median": (d.tor_optt_ugm3 / d.tor_ec_ugm3).median(),
                     "mac_ratio_median": (d.fabs_Mm1 / d.tor_ec_ugm3).median(),
                     "tor_oc_ec_median": d.tor_oc_ec.median(),
                     "tor_ec_ugm3_median": d.tor_ec_ugm3.median(),
                     "loading_ug_median": d.y.median()})
        q = pd.qcut(d.op_tor_frac, 3, labels=["low OP", "mid OP", "high OP"])
        for t, part in d.groupby(q, observed=True):
            terciles.append({"method": method, "group": g, "op_tercile": t, "n": len(part),
                             "op_frac_max": part.op_tor_frac.max(),
                             "mac_ratio_median": (part.fabs_Mm1 / part.tor_ec_ugm3).median()})
    save(pd.DataFrame(rows), "op_summary")
    save(pd.DataFrame(terciles), "op_terciles")


def stage_reproduction(ctx, gm):
    """Refit the meeting's own seasonal cohorts (Sept-10 run, CO2-only mask) here."""
    src = AREA / "output/tables/ann_weekly_20260910"
    met = pd.read_csv(src / "regression_metrics.csv")
    f2pos = pd.Series(np.arange(len(ctx.lib)), index=ctx.lib.filter_id)
    rows = []
    for g in ab.SEASONS + ["All Addis"]:
        fit_id = f"no_co2__{slug(g)}"
        co = pd.read_csv(src / f"cohort_{fit_id}.csv")
        slide = met.loc[met.fit_id.eq(fit_id) & met.evaluation_group.eq(g)].iloc[0]
        inlib = co.FilterId.isin(f2pos.index).to_numpy()
        pos = f2pos.loc[co.FilterId[inlib]].to_numpy()
        f = ab.fit_cohort(ctx.X["AIRSpec"][pos], co.TOR_EC_loading_ug[inlib], co.Site[inlib],
                          train=co.role[inlib].eq("train").to_numpy(), k=int(co.k.iloc[0]))
        pa = f.predict(ctx.XA["AIRSpec"]) / ctx.addis.Volume_m3.to_numpy(float)
        st = addis_panel(ctx, gm[g], pa, boot=False)
        rows.append({"group": g, "slide_k": int(co.k.iloc[0]), "slide_n_addis": int(slide.n),
                     "slide_slope": slide.deming_slope, "slide_intercept": slide.deming_intercept,
                     "cohort_in_frozen_run": int(inlib.sum()), "here_n_addis": st["n"],
                     "here_slope": st["deming_slope"], "here_intercept": st["deming_intercept"]})
    save(pd.DataFrame(rows), "meeting_reproduction")


def main():
    t0 = time.time()
    ctx = ab.load_context()
    gm = ab.group_masks(ctx)
    log(f"context: {len(ctx.lib)} IMPROVE, {len(ctx.addis)} Addis, lambda*={ctx.lam:.2f}; {ctx.notes}")
    ref = ctx.lib[REF_COLS].copy()
    save(ref, "improve_references")
    save(ctx.addis[["MediaId", "ExternalFilterId", "date", "season", "Fabs", "Volume_m3",
                    "month_block", "pmf_source", "LotId"]], "addis_evaluation")
    sel = stage_selections(ctx, gm)
    log("selections done")
    f1, s1, a1, p1 = stage_seasonal(ctx, gm, sel)
    f2, s2, a2, p2 = stage_allimprove(ctx, gm, sel)
    f3, s3, p3 = stage_combined(ctx, gm, sel)
    s4, p4 = stage_stitched(ctx, gm, p1)
    save(pd.DataFrame(f1 + f2 + f3), "fits")
    save(pd.DataFrame(s1 + s2 + s3 + s4), "panel_stats")
    save(pd.concat(a1 + a2, ignore_index=True), "analog_predictions")
    save(pd.concat(p1 + p2 + p3 + p4, ignore_index=True), "addis_predictions")
    stage_op(ctx, sel)
    stage_reproduction(ctx, gm)
    rep = stage_repeats(ctx, gm, sel)
    save(rep, "repeated_splits")
    # spectra summaries (median + IQR) for the crossplot-beside-spectra slides
    spectra = {"wn": ctx.wn.tolist(), **{f"wn|{m}": ctx.wn_of(m).tolist() for m in ctx.methods if m not in ab.METHODS}}
    for method in ctx.methods:
        for g, take in gm.items():
            pos = sel[(method, "locked", g)]
            for name, arr in (("improve", ctx.X[method][pos]), ("addis", ctx.XA[method][take])):
                q = np.percentile(arr, [10, 25, 50, 75, 90], axis=0)
                spectra[f"{method}|{g}|{name}"] = [np.round(v, 6).tolist() for v in q]
    (OUT / "spectra_quantiles.json").write_text(json.dumps(spectra))
    summary = {"run": str(ab.RUN.relative_to(ROOT)), "notes": ctx.notes, "lambda": ctx.lam,
               "methods": list(ctx.methods),
               "ref_rel_uncertainty": ref_rel_uncertainty(ctx),
               "sigma_x": ctx.sigma_x, "sigma_y": ab.SIGMA_Y_PROXY, "mac": ab.MAC_VALUE,
               "locked_mask": ab.LOCKED_MASK, "meeting_mask": ab.MEETING_MASK,
               "n_analogs": ab.N_ANALOGS, "n_repeats": N_REPEATS, "lot_model": LOT,
               "addis_n": len(ctx.addis), "improve_n": len(ctx.lib),
               "seconds": round(time.time() - t0)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
