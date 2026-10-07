"""Satoshi's questions after the 1 Oct 2026 meeting (email of 2 Oct 2026).

Run from the repo root:
    uv run --no-sync python research/ftir_hips_chem/workflows/run_satoshi_followup_20261002.py

Inputs are the completed rank-70 context the gallery's meeting tab uses
(``vibes_current.load_context``): identical IMPROVE filters, Addis filters, split
and references. Baselines: spline (AIRSpec, 4000-1425) and VIBES (rank 70,
4000-500). Stages:

  selections  seasonal correlation analogs for both baselines and both
              similarity regions (as presented 1 Oct; the region agreed 1 Oct)
  overlap     shared / spline-only / VIBES-only counts and what differs
  swap        select on one baseline, calibrate on the other (protocol split +
              100 site-disjoint repeats): training-set effect vs baseline effect
  models      the standard all-IMPROVE lot-251 model and each analog model:
              VIP, the fit -> cross-validation -> IMPROVE test -> Addis ladder,
              and score-space extrapolation of Addis
  envelope    where the analogs sit in the standard model's out-of-sample
              residual distribution (the "lower envelope" question)

Nothing is tuned on Addis. Addis comparisons are FTIR EC vs HIPS Fabs / MAC 10.
"""

from __future__ import annotations

import argparse
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
import analog_transfer_checks as ad  # noqa: E402
from vibes_current import LABELS, MIGRATION, load_context  # noqa: E402

OUT = AREA / "output/tables/satoshi_followup_20261002"
SPLINE, VIBES = "AIRSpec", "VIBES-full-r70"
BASELINES = (SPLINE, VIBES)
LOT = 251
N_REPEATS = 100


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(frame, name):
    frame.to_csv(OUT / f"{name}.csv", index=False)


def rel_uncertainty(ctx):
    """Reference relative 1-sigma (ETAD HIPS), as in the 17 Sep meeting panels."""
    return ctx.sigma_x / float(np.median(ctx.addis.Fabs.to_numpy(float) / ab.MAC_VALUE))


# --------------------------------------------------------------------------- selections
def stage_selections(ctx, gm):
    sel, scores, rows = {}, {}, []
    for method in BASELINES:
        for mask in ad.MASKS:
            for g, take in gm.items():
                pos, sc = ad.select_analogs(ctx, method, take, mask)
                sel[(method, mask, g)] = pos
                scores[(method, mask, g)] = sc
    # the spline grid starts at 1425, so the agreed region changes nothing for it
    for g in gm:
        assert np.array_equal(sel[(SPLINE, "presented", g)], sel[(SPLINE, "oct1", g)])
    # "presented" must reproduce the selections behind the meeting tab
    prior = pd.read_csv(MIGRATION / "meeting/analog_membership.csv")
    for method in BASELINES:
        for g in gm:
            want = prior.loc[prior.method.eq(method) & prior["mask"].eq("locked") & prior.group.eq(g)]
            got = ctx.lib.filter_id.to_numpy()[sel[(method, "presented", g)]]
            if not np.array_equal(want.sort_values("rank").filter_id.to_numpy(), got):
                raise ValueError(f"selection does not reproduce the meeting tab: {method} {g}")
            rows.append({"method": method, "group": g, "reproduces_meeting_selection": True})
    return sel, scores, pd.DataFrame(rows)


def subset_props(lib, pos):
    d = lib.iloc[pos]
    if not len(d):
        return {"n": 0}
    return {"n": len(d), "n_sites": int(d.Site.nunique()),
            "tor_ec_median": d.tor_ec_ugm3.median(), "tor_ec_p90": d.tor_ec_ugm3.quantile(.9),
            "oc_ec_median": d.tor_oc_ec.median(), "mac_ratio_median": d.mac_ratio.median(),
            "loading_median_ug": d.y.median(), "lot251_fraction": float(d.lot.eq(LOT).mean()),
            "top_sites": ", ".join(f"{s} {n}" for s, n in d.Site.value_counts().head(4).items())}


def stage_overlap(ctx, gm, sel, scores):
    lib = ctx.lib
    fid = lib.filter_id.to_numpy()
    summary, members = [], []
    for mask in ad.MASKS:
        for g in gm:
            a, v = sel[(SPLINE, mask, g)], sel[(VIBES, mask, g)]
            sa, sv = set(a), set(v)
            parts = {"shared": sorted(sa & sv), "spline only": sorted(sa - sv), "VIBES only": sorted(sv - sa)}
            ra = pd.Series(np.arange(1, len(a) + 1), index=a)
            rv = pd.Series(np.arange(1, len(v) + 1), index=v)
            shared = parts["shared"]
            rank_rho = (pd.Series(ra.loc[shared].to_numpy()).corr(pd.Series(rv.loc[shared].to_numpy()), method="spearman")
                        if len(shared) > 2 else np.nan)
            base = {"mask": mask, "group": g, "n_spline": len(a), "n_vibes": len(v),
                    "shared": len(shared), "spline_only": len(parts["spline only"]),
                    "vibes_only": len(parts["VIBES only"]),
                    "jaccard": len(shared) / len(sa | sv), "shared_rank_spearman": rank_rho,
                    "shared_sites": len(set(lib.Site.iloc[shared])),
                    "site_overlap": len(set(lib.Site.iloc[a]) & set(lib.Site.iloc[v]))}
            summary.append(base)
            for part, pos in parts.items():
                members.append(pd.DataFrame({
                    "mask": mask, "group": g, "part": part, "filter_id": fid[pos],
                    "Site": lib.Site.iloc[pos].to_numpy(), "lot": lib.lot.iloc[pos].to_numpy(),
                    "date": lib.date.iloc[pos].dt.strftime("%Y-%m-%d").to_numpy(),
                    "rank_spline": ra.reindex(pos).to_numpy(), "rank_vibes": rv.reindex(pos).to_numpy(),
                    "r_spline": scores[(SPLINE, mask, g)][pos], "r_vibes": scores[(VIBES, mask, g)][pos],
                    "tor_ec_ugm3": lib.tor_ec_ugm3.iloc[pos].to_numpy(),
                    "tor_oc_ec": lib.tor_oc_ec.iloc[pos].to_numpy(),
                    "mac_ratio": lib.mac_ratio.iloc[pos].to_numpy(), "loading_ug": lib.y.iloc[pos].to_numpy()}))
    # VIBES: what the agreed region changes
    for g in gm:
        p, o = set(sel[(VIBES, "presented", g)]), set(sel[(VIBES, "oct1", g)])
        summary.append({"mask": "presented_vs_oct1", "group": g, "n_spline": None, "n_vibes": len(o),
                        "shared": len(p & o), "spline_only": None, "vibes_only": len(o - p),
                        "jaccard": len(p & o) / len(p | o)})
    props = []
    m = pd.concat(members, ignore_index=True)
    f2pos = pd.Series(np.arange(len(lib)), index=fid)
    for (mask, g, part), d in m.groupby(["mask", "group", "part"], sort=False):
        props.append({"mask": mask, "group": g, "part": part,
                      **subset_props(lib, f2pos.loc[d.filter_id].to_numpy()),
                      "r_spline_median": d.r_spline.median(), "r_vibes_median": d.r_vibes.median()})
    save(pd.DataFrame(summary), "overlap_summary")
    save(pd.DataFrame(props), "overlap_properties")
    save(m, "overlap_membership")


# --------------------------------------------------------------------------- swap
SWAP_SELECTIONS = ((SPLINE, "presented"), (VIBES, "presented"), (VIBES, "oct1"))


def _addis_stats(ctx, gm, g, pa):
    x = ctx.addis.Fabs.to_numpy(float)[gm[g]] / ab.MAC_VALUE
    return ab.xy_stats(x, pa[gm[g]], sigma_x=ctx.sigma_x, sigma_y=ab.SIGMA_Y_PROXY)


def _swap_fit(ctx, gm, sel_m, mask, cal, g, pos, repeat, train):
    lib = ctx.lib.iloc[pos]
    f = ab.fit_cohort(ctx.X[cal][pos], lib.y, lib.Site, train=train)
    pa = f.predict(ctx.XA[cal]) / ctx.addis.Volume_m3.to_numpy(float)
    st = _addis_stats(ctx, gm, g, pa)
    test = ~f.train
    tt = ab.xy_stats(lib.y.to_numpy()[test], f.predict(ctx.X[cal][pos])[test])
    return {"selected_on": sel_m, "mask": mask, "calibrated_on": cal, "group": g, "repeat": repeat,
            "k": f.k, "addis_slope": st.get("deming_slope"), "addis_intercept": st.get("deming_intercept"),
            "addis_R2": st.get("R2"), "addis_n": st.get("n"), "improve_test_R2": tt.get("R2"),
            "improve_test_RMSE_ug": tt.get("RMSE")}


def stage_swap(ctx, gm, sel, n_repeats):
    from joblib import Parallel, delayed

    jobs = []
    for sel_m, mask in SWAP_SELECTIONS:
        for g in gm:
            pos = sel[(sel_m, mask, g)]
            sites = ctx.lib.Site.to_numpy()[pos]
            splits = [(-1, None)] + list(enumerate(ab.repeated_site_splits(sites, n_repeats)))
            for cal in BASELINES:
                for r, train in splits:
                    jobs.append((sel_m, mask, cal, g, pos, r, train))
    log(f"swap: {len(jobs)} fits")
    rows = Parallel(n_jobs=12, backend="threading")(delayed(_swap_fit)(ctx, gm, *j) for j in jobs)
    frame = pd.DataFrame(rows)
    # the diagonal of the protocol fits is the meeting tab's seasonal calibration
    stats = pd.read_csv(MIGRATION / "meeting/panel_stats.csv")
    fits = pd.read_csv(MIGRATION / "meeting/fits.csv")
    check = []
    for cal in BASELINES:
        for g in gm:
            here = frame.loc[frame.repeat.eq(-1) & frame.selected_on.eq(cal) & frame["mask"].eq("presented")
                             & frame.calibrated_on.eq(cal) & frame.group.eq(g)].iloc[0]
            fid = f"seasonal|{cal}|locked|{g}"
            there = stats.loc[stats.fit_id.eq(fid) & stats.population.eq("addis") & stats.eval_group.eq(g)].iloc[0]
            k = int(fits.loc[fits.fit_id.eq(fid), "k"].iloc[0])
            check.append({"baseline": cal, "group": g, "k_meeting": k, "k_here": int(here.k),
                          "slope_meeting": there.deming_slope, "slope_here": here.addis_slope,
                          "intercept_meeting": there.deming_intercept, "intercept_here": here.addis_intercept})
    check = pd.DataFrame(check)
    if not (np.allclose(check.slope_meeting, check.slope_here, atol=1e-6)
            and np.allclose(check.intercept_meeting, check.intercept_here, atol=1e-6)
            and check.k_meeting.eq(check.k_here).all()):
        raise ValueError(f"swap diagonal does not reproduce the meeting tab:\n{check}")
    save(check, "swap_reproduction")
    save(frame, "swap_fits")
    return frame


# --------------------------------------------------------------------------- models
def _model_record(ctx, gm, model_id, family, cal, mask, group, pos, f, rel, vip_store, ladders, extra):
    """VIP, fit ladder and extrapolation for one fitted calibration on rows ``pos``."""
    lib = ctx.lib.iloc[pos]
    vol = lib.tor_volume_m3.to_numpy(float)
    tor = lib.tor_ec_ugm3.to_numpy(float)
    X = ctx.X[cal][pos]
    sites = lib.Site.to_numpy()
    train, test = f.train, ~f.train
    fitted = f.predict(X) / vol
    oof = np.full(len(pos), np.nan)
    oof[train] = ad.out_of_fold(X[train], lib.y.to_numpy(float)[train], sites[train], f.k) / vol[train]
    rmsecv = float(f.curve.loc[f.curve.n_components.eq(f.k), "rmsecv"].iloc[0])
    s_ftir = rmsecv / float(np.nanmedian(vol))

    def stage(name, take, x, y, population):
        s_ref = rel * float(np.nanmedian(x[take]))
        ladders.append({"model_id": model_id, "family": family, "baseline": cal, "mask": mask, "group": group,
                        "population": population, "stage": name, "k": f.k,
                        **ad.fit_stats(x[take], y[take], s_ref, s_ftir)})

    stage("calibration fit", train, tor, fitted, "own")
    stage("cross-validation", train, tor, oof, "own")
    if test.any():
        stage("IMPROVE test sites", test, tor, fitted, "own")
    pa = f.predict(ctx.XA[cal]) / ctx.addis.Volume_m3.to_numpy(float)
    ex = ad.extrapolation(f.model, X[train], ctx.XA[cal])
    ex_test = ad.extrapolation(f.model, X[train], X[test]) if test.any() else None
    ytrain = lib.y.to_numpy(float)[train]
    implied = ctx.addis.Fabs.to_numpy(float) / ab.MAC_VALUE * ctx.addis.Volume_m3.to_numpy(float)
    for g in (gm if group in ("standard", "All Addis") else [group]):
        take = gm[g]
        st = ab.xy_stats(ctx.addis.Fabs.to_numpy(float)[take] / ab.MAC_VALUE, pa[take],
                         sigma_x=ctx.sigma_x, sigma_y=ab.SIGMA_Y_PROXY)
        ladders.append({"model_id": model_id, "family": family, "baseline": cal, "mask": mask, "group": group,
                        "population": "addis", "eval_group": g, "stage": "Addis test set", "k": f.k,
                        "n": st.get("n"), "R2": st.get("R2"), "RMSE": st.get("RMSE"), "bias": st.get("bias"),
                        "deming_slope": st.get("deming_slope"), "deming_intercept": st.get("deming_intercept"),
                        "x_median": st.get("x_median"),
                        "t2_beyond_p95": float(np.mean(ex["t2"][take] > ex["t2_p95"])),
                        "q_beyond_p95": float(np.mean(ex["q"][take] > ex["q_p95"])),
                        "t2_median_ratio": float(np.median(ex["t2"][take]) / ex["t2_p95"]),
                        "q_median_ratio": float(np.median(ex["q"][take]) / ex["q_p95"]),
                        "improve_test_t2_beyond_p95": float(np.mean(ex_test["t2"] > ex_test["t2_p95"])) if ex_test else None,
                        "improve_test_q_beyond_p95": float(np.mean(ex_test["q"] > ex_test["q_p95"])) if ex_test else None,
                        "loading_beyond_train_p95": float(np.mean(implied[take] > np.percentile(ytrain, 95))),
                        "loading_beyond_train_max": float(np.mean(implied[take] > ytrain.max())),
                        "addis_loading_median_ug": float(np.median(implied[take])),
                        "train_loading_median_ug": float(np.median(ytrain)),
                        "train_loading_p95_ug": float(np.percentile(ytrain, 95))})
    vip = ad.vip_scores(f.model)
    vip_store[model_id] = {"baseline": cal, "family": family, "mask": mask, "group": group, "k": f.k,
                           "vip": vip, "coef": f.model.coef_.ravel(),
                           "bands": ad.vip_band_table(ctx.wn_of(cal), vip)}
    extra[model_id] = {"fitted": fitted, "oof": oof, "train": train, "pos": pos, "pa": pa,
                       "t2": ex["t2"] / ex["t2_p95"], "q": ex["q"] / ex["q_p95"]}


def stage_models(ctx, gm, sel):
    rel = rel_uncertainty(ctx)
    ladders, vip_store, extra = [], {}, {}
    lot = np.flatnonzero(ctx.lib.lot.eq(LOT).to_numpy())
    for cal in BASELINES:
        t0 = time.time()
        sub = ctx.lib.iloc[lot]
        f = ab.fit_cohort(ctx.X[cal][lot], sub.y, sub.Site, train=sub.split.eq("train").to_numpy())
        mid = f"standard|{cal}"
        _model_record(ctx, gm, mid, "standard", cal, "n/a", "standard", lot, f, rel, vip_store, ladders, extra)
        log(f"{mid}: k={f.k} in {time.time() - t0:.0f}s")
        # how the standard model handles each season's analogs (selected on its own baseline)
        e = extra[mid]
        pos_in_pool = pd.Series(np.arange(len(lot)), index=lot)
        oos = np.where(e["train"], e["oof"], e["fitted"])     # out-of-sample for every pool filter
        tor = ctx.lib.tor_ec_ugm3.to_numpy(float)[lot]
        rmsecv = float(f.curve.loc[f.curve.n_components.eq(f.k), "rmsecv"].iloc[0])
        s_ftir = rmsecv / float(np.nanmedian(ctx.lib.tor_volume_m3.to_numpy(float)[lot]))
        for mask in ad.MASKS:
            if cal == SPLINE and mask == "oct1":
                continue
            for g in gm:
                analog = sel[(cal, mask, g)]
                inlot = analog[np.isin(analog, lot)]
                j = pos_in_pool.loc[inlot].to_numpy()
                for name, take, y in (("calibration fit", e["train"][j], e["fitted"][j]),
                                      ("cross-validation", e["train"][j], e["oof"][j]),
                                      ("IMPROVE test sites", ~e["train"][j], e["fitted"][j]),
                                      ("out-of-sample (all)", np.ones(len(j), bool), oos[j])):
                    s_ref = rel * float(np.nanmedian(tor[j][take])) if take.any() else 1.0
                    ladders.append({"model_id": mid, "family": "standard", "baseline": cal, "mask": mask,
                                    "group": g, "population": "analogs", "stage": name, "k": f.k,
                                    "analogs_outside_lot": int(len(analog) - len(inlot)),
                                    **ad.fit_stats(tor[j][take], y[take], s_ref, s_ftir)})
        for mask in ad.MASKS:
            if cal == SPLINE and mask == "oct1":
                continue
            for g in gm:
                pos = sel[(cal, mask, g)]
                lib = ctx.lib.iloc[pos]
                fa = ab.fit_cohort(ctx.X[cal][pos], lib.y, lib.Site)
                _model_record(ctx, gm, f"analog|{cal}|{mask}|{g}", "analog", cal, mask, g, pos, fa, rel,
                              vip_store, ladders, extra)
        log(f"{cal}: analog models done")
    ladder = pd.DataFrame(ladders)
    ladder["intercept_rel"] = ladder.deming_intercept / ladder.x_median
    save(ladder, "fit_ladder")
    points = []
    for mid, e in extra.items():
        if not mid.startswith("analog|"):
            continue
        lib = ctx.lib.iloc[e["pos"]]
        points.append(pd.DataFrame({"model_id": mid, "filter_id": lib.filter_id.to_numpy(),
                                    "Site": lib.Site.to_numpy(), "role": np.where(e["train"], "train", "test"),
                                    "tor_ec_ugm3": lib.tor_ec_ugm3.to_numpy(), "fitted_ugm3": e["fitted"],
                                    "cv_ugm3": e["oof"]}))
    save(pd.concat(points, ignore_index=True), "ladder_points")
    bands = []
    for mid, v in vip_store.items():
        b = v["bands"].copy()
        b.insert(0, "model_id", mid)
        bands.append(b)
    save(pd.concat(bands, ignore_index=True), "vip_bands")
    # VIP similarity of each analog model to the standard model on the same baseline
    from scipy.stats import spearmanr
    sim = []
    for mid, v in vip_store.items():
        if v["family"] != "analog":
            continue
        s = vip_store[f"standard|{v['baseline']}"]["vip"]
        top_s, top_a = set(np.argsort(s)[-200:]), set(np.argsort(v["vip"])[-200:])
        sim.append({"model_id": mid, "baseline": v["baseline"], "mask": v["mask"], "group": v["group"],
                    "k": v["k"], "spearman_vs_standard": float(spearmanr(s, v["vip"]).statistic),
                    "top200_overlap": len(top_s & top_a) / 200,
                    "vip_ge1_both": int(np.sum((s >= 1) & (v["vip"] >= 1))),
                    "vip_ge1_standard": int(np.sum(s >= 1)), "vip_ge1_analog": int(np.sum(v["vip"] >= 1))})
    save(pd.DataFrame(sim), "vip_similarity")
    payload = {"wn": {m: ctx.wn_of(m).tolist() for m in BASELINES},
               "models": {mid: {k: (np.round(v[k], 5).tolist() if k in ("vip", "coef") else v[k])
                                for k in ("baseline", "family", "mask", "group", "k", "vip", "coef")}
                          for mid, v in vip_store.items()}}
    (OUT / "vip_profiles.json").write_text(json.dumps(payload))
    return extra


# --------------------------------------------------------------------------- envelope
def stage_envelope(ctx, gm, sel, extra):
    """Out-of-sample residual percentile of the analogs under the standard model."""
    lot = np.flatnonzero(ctx.lib.lot.eq(LOT).to_numpy())
    pts, summary = [], []
    years = []
    for cal in BASELINES:
        e = extra[f"standard|{cal}"]
        tor = ctx.lib.tor_ec_ugm3.to_numpy(float)[lot]
        oos = np.where(e["train"], e["oof"], e["fitted"])
        resid = oos - tor
        pct = ad.conditional_percentile(tor, resid, tor, resid)
        pts.append(pd.DataFrame({"baseline": cal, "filter_id": ctx.lib.filter_id.to_numpy()[lot],
                                 "Site": ctx.lib.Site.to_numpy()[lot], "role": np.where(e["train"], "cv", "test"),
                                 "tor_ec_ugm3": tor, "ftir_oos_ugm3": oos, "residual_pct": pct}))
        pos_in_pool = pd.Series(np.arange(len(lot)), index=lot)
        for mask in ad.MASKS:
            if cal == SPLINE and mask == "oct1":
                continue
            for g in gm:
                analog = sel[(cal, mask, g)]
                j = pos_in_pool.loc[analog[np.isin(analog, lot)]].to_numpy()
                p = pct[j]
                summary.append({"baseline": cal, "mask": mask, "group": g, "n": len(j),
                                "pct_median": float(np.nanmedian(p)),
                                "below_p25": float(np.nanmean(p < 25)), "below_p10": float(np.nanmean(p < 10)),
                                "above_p75": float(np.nanmean(p > 75)),
                                "residual_median_ugm3": float(np.nanmedian(resid[j])),
                                "ratio_median": float(np.nanmedian(oos[j] / tor[j])),
                                "tor_ec_median": float(np.nanmedian(tor[j]))})
    for mask in ad.MASKS:
        for cal in BASELINES:
            if cal == SPLINE and mask == "oct1":
                continue
            for g in gm:
                d = ctx.lib.iloc[sel[(cal, mask, g)]]
                for y, n in d.date.dt.year.value_counts().sort_index().items():
                    years.append({"set": f"{cal}|{mask}|{g}", "year": int(y), "n": int(n)})
    for y, n in ctx.lib.date.dt.year.value_counts().sort_index().items():
        years.append({"set": "IMPROVE pool", "year": int(y), "n": int(n)})
    save(pd.concat(pts, ignore_index=True), "envelope_points")
    save(pd.DataFrame(summary), "envelope_summary")
    save(pd.DataFrame(years), "analog_years")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--skip-swap", action="store_true", help="keep the saved swap_fits.csv (the slow stage)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx, data = load_context()
    gm = ab.group_masks(ctx)
    log(f"context: {len(ctx.lib)} IMPROVE, {len(ctx.addis)} Addis, methods {ctx.methods}")
    sel, scores, repro = stage_selections(ctx, gm)
    save(repro, "selection_reproduction")
    stage_overlap(ctx, gm, sel, scores)
    log("overlap done")
    extra = stage_models(ctx, gm, sel)
    stage_envelope(ctx, gm, sel, extra)
    log("models + envelope done")
    if args.skip_swap:
        prior = json.loads((OUT / "summary.json").read_text())
        args.repeats = prior["n_repeats"]
        seconds = prior["seconds"]          # keep the full run's time, not this partial rerun's
    else:
        stage_swap(ctx, gm, sel, args.repeats)
        seconds = None
    summary = {"question_source": "Satoshi Takahama email, 2 Oct 2026 (reply to the 1 Oct task list)",
               "baselines": {m: LABELS[m] for m in BASELINES}, "masks": ad.MASK_LABELS,
               "mask_definitions": ad.MASKS, "n_analogs": ab.N_ANALOGS, "n_repeats": args.repeats,
               "standard_model": f"all-IMPROVE lot {LOT}, frozen site-disjoint train split, site-grouped CV, first major minimum",
               "lambda_addis": ctx.lam, "sigma_x": ctx.sigma_x, "sigma_y": ab.SIGMA_Y_PROXY,
               "ref_rel_uncertainty": rel_uncertainty(ctx), "mac": ab.MAC_VALUE,
               "addis_n": len(ctx.addis), "improve_n": len(ctx.lib),
               "improve_dates": [str(ctx.lib.date.min().date()), str(ctx.lib.date.max().date())],
               "source_sha256": data["sources"], "seconds": seconds or round(time.time() - t0)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
