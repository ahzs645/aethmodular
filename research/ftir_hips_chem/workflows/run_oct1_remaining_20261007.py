"""The open items from the 1 Oct 2026 Ann/Satoshi task list (run 7 Oct 2026).

Run from the repo root:
    uv run --no-sync python research/ftir_hips_chem/workflows/run_oct1_remaining_20261007.py

Two stages, both on the same rank-70 context as the Satoshi 2 Oct tab
(``vibes_current.load_context``: identical IMPROVE filters, Addis filters,
site-disjoint split and references):

  blanks   VIBES re-corrected (Colab, ``vibes_fullrange_run.py --blank-policy``;
           results on Drive under ``data_paths.vibes_colab_results_dir()``)
           with every labeled lot-251 blank, and with every labeled blank in the
           bundle (lots 248 + 251, all sites). Compared with the capped 87-blank
           bank behind every VIBES result so far: background rank, held-out blank
           residuals, the standard all-IMPROVE lot-251 model, and seasonal analog
           calibrations selected on the agreed 1 Oct region ("oct1"), each with
           100 site-grouped splits for slope AND intercept.
  groups   within each Addis season, correlate the spectra over the fingerprint
           region (600-1500 cm-1, PTFE 1050-1250 dropped; and a with-PTFE
           sensitivity), cluster, and check whether the clusters line up with
           HIPS EC, the standard model's error, or SPARTAN composition.

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
sys.path.insert(0, str(AREA / "workflows"))
import analog_bias as ab  # noqa: E402
import analog_transfer_checks as ad  # noqa: E402
import run_satoshi_followup_20261002 as sf  # noqa: E402
from data_paths import vibes_colab_results_dir  # noqa: E402
from vibes_current import load_context, sha  # noqa: E402

OUT = AREA / "output/tables/oct1_remaining_20261007"
# The blank-bank re-corrections are on Drive (My Drive/Aethmodular VIBES/results/...).
BLANKBANK = vibes_colab_results_dir() / "blankbank_20261007"
SPLINE, CAPPED = "AIRSpec", "VIBES-full-r70"
POLICIES = {  # method key -> (run folder, label)
    "VIBES-lot251": ("lot251_all", "VIBES, every labeled lot-251 blank"),
    "VIBES-allblanks": ("all_labeled", "VIBES, every labeled blank (lots 248 + 251)"),
    # 7 Oct reruns without improve:1982003 (LABE1), the blank that set the one-SE rank to 13/15
    "VIBES-lot251-x1": ("lot251_all_minus_LABE1", "VIBES, lot-251 blanks minus the LABE1 outlier"),
    "VIBES-allblanks-x1": ("all_labeled_minus_LABE1", "VIBES, every labeled blank minus the LABE1 outlier"),
}
LABELS = {SPLINE: "Spline baseline", CAPPED: "VIBES, capped bank (87 blanks, rank 70)",
          **{k: v[1] for k, v in POLICIES.items()}}
N_REPEATS = 100
LOT = 251
# Fingerprint regions for the within-season grouping (cm-1).
REGIONS = {
    "fingerprint": {"low": 600, "high": 1500, "drop": [(1050, 1250)]},
    "fingerprint_with_ptfe": {"low": 600, "high": 1500, "drop": []},
}
REGION_LABELS = {"fingerprint": "600–1500 cm⁻¹, PTFE 1050–1250 dropped",
                 "fingerprint_with_ptfe": "600–1500 cm⁻¹, PTFE kept"}
CHEM = {  # SPARTAN parameter -> short name (µg/m³)
    "ChemSpec_Nitrate_Ion_PM2.5": "nitrate", "ChemSpec_Sulfate_Ion_PM2.5": "sulfate",
    "ChemSpec_Potassium_Ion_PM2.5": "K_ion", "ChemSpec_Calcium_Ion_PM2.5": "Ca_ion",
    "ChemSpec_Aluminum_PM2.5": "Al", "ChemSpec_Titanium_PM2.5": "Ti",
    "ChemSpec_Filter_PM2.5_mass": "PM25", "ChemSpec_OC_PM2.5": "OC",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(frame, name):
    frame.to_csv(OUT / f"{name}.csv", index=False)


# --------------------------------------------------------------------------- inputs
def load_policy_runs(ctx, data):
    """Attach the new VIBES arrays to the context; rows must match the frozen cases."""
    ref_ids = data["cases"].sample_id.tolist()
    runs, sources = {}, {}
    for method, (policy, _) in POLICIES.items():
        hits = sorted((BLANKBANK / policy / "persistent_results").glob("full500-rank85-*"))
        if len(hits) != 1:
            log(f"{method}: no completed run under {BLANKBANK / policy} (skipped)")
            continue
        run = hits[0]
        man = json.loads((run / "RUN_MANIFEST.json").read_text())
        cases = pd.read_csv(run / "cases.csv")
        if cases.sample_id.tolist() != ref_ids:
            raise ValueError(f"{method}: case rows differ from the frozen ledger")
        arr = np.load(run / "corrected_VIBES_full.npy", mmap_mode="r")
        wn = np.load(run / "wn.npy")
        if not np.array_equal(wn, ctx.wn_of(CAPPED)):
            raise ValueError(f"{method}: wavenumber grid differs from the capped run")
        lib_rows, addis_rows = ctx.lib.row.to_numpy(), ctx.addis.row.to_numpy()
        X, XA = np.asarray(arr[lib_rows], float), np.asarray(arr[addis_rows], float)
        bad_lib, bad_addis = ~np.isfinite(X).all(1), ~np.isfinite(XA).all(1)
        if bad_addis.any():
            raise ValueError(f"{method}: {int(bad_addis.sum())} Addis spectra failed")
        if bad_lib.any():
            # a failed correction leaves NaN; fill with the library mean so similarity scores stay
            # defined, and drop these rows from every selection and fit (ctx.failed)
            X[bad_lib] = np.nanmean(X[~bad_lib], axis=0)
        ctx.failed = {**getattr(ctx, "failed", {}), method: np.flatnonzero(bad_lib)}
        ctx.X[method], ctx.XA[method], ctx.wn_by[method] = X, XA, wn
        runs[method] = {"policy": policy, "run": run.name, "manifest": man, "cases": cases,
                        "array": arr, "failed_improve": np.flatnonzero(bad_lib),
                        "bmeta": pd.read_csv(run / "background_training_blanks.csv"),
                        "rank_cv": pd.read_csv(run / "background_rank_cv.csv")}
        for f in ("corrected_VIBES_full.npy", "cases.csv", "RUN_MANIFEST.json"):
            sources[f"drive:blankbank_20261007/{policy}/{run.name}/{f}"] = sha(run / f)
        log(f"{method}: {man['n_background_blanks']} blanks, rank {man['background_rank']}, "
            f"failed {man['n_failed']} ({len(runs[method]['failed_improve'])} in the IMPROVE library)")
    return runs, sources


def capped_run(data):
    run = next(ROOT / k for k in data["sources"] if k.endswith("/RUN_MANIFEST.json") and "rank85_resume" in k).parent
    return {"policy": "capped", "run": run.name, "manifest": json.loads((run / "RUN_MANIFEST.json").read_text()),
            "bmeta": pd.read_csv(run / "background_training_blanks.csv"),
            "rank_cv": pd.read_csv(run / "background_rank_cv.csv")}


# --------------------------------------------------------------------------- blanks
def held_out_blank_residuals(data, runs):
    """RMS of corrected blanks no bank trained on (same filters for every policy)."""
    cases = data["cases"]
    blank = cases.kind.eq("blank").to_numpy()
    used = np.zeros(len(cases), bool)
    for r in runs.values():
        used |= r["cases"].get("in_background_bank", pd.Series(False, index=cases.index)).fillna(False).to_numpy(bool)
    rows = np.flatnonzero(blank & ~used)
    wn = np.load(next(ROOT / k for k in data["sources"] if k.endswith("/wn.npy") and "rank85_resume" in k))
    bands = {"all (no CO₂)": ad.region_mask(wn, 500, 4000, [(1800, 2500)]),
             "PTFE 1050–1250": ad.region_mask(wn, 1050, 1250),
             "fingerprint 600–1500": ad.region_mask(wn, 600, 1500, [(1050, 1250)]),
             "functional 1500–4000": ad.region_mask(wn, 1500, 4000, [(1800, 2500)])}
    arrays = {CAPPED: data["arrays"][CAPPED], **{m: r["array"] for m, r in runs.items()}}
    out = []
    for m, arr in arrays.items():
        v = np.asarray(arr[rows], float)
        for band, keep in bands.items():
            rms = np.sqrt(np.nanmean(v[:, keep] ** 2, axis=1))
            for src in ("improve", "etad"):
                take = cases.source.to_numpy()[rows] == src
                if take.any():
                    out.append({"baseline": m, "band": band, "source": src, "n": int(take.sum()),
                                "rms_median": float(np.nanmedian(rms[take])),
                                "rms_p90": float(np.nanpercentile(rms[take], 90))})
    return pd.DataFrame(out)


def stage_background(data, runs):
    rows, curves = [], []
    for m, r in {CAPPED: capped_run(data), **runs}.items():
        man, b = r["manifest"], r["bmeta"]
        rows.append({"baseline": m, "label": LABELS[m], "policy": r["policy"], "run": r["run"],
                     "n_blanks": len(b), "n_improve": int(b.source.eq("IMPROVE").sum()),
                     "n_etad": int(b.source.eq("ETAD").sum()),
                     "n_lot251": int(b.LotNumber.eq(251).sum()), "n_lot248": int(b.LotNumber.eq(248).sum()),
                     "n_test_site": int(b.split.eq("test").sum()), "rank": man["background_rank"],
                     "rank_cap": man["config"]["max_background_components"],
                     "n_failed": man["n_failed"], "n_failed_improve": len(r.get("failed_improve", [])), "hours": round(man["correction_wall_seconds"] / 3600, 2)})
        c = r["rank_cv"].copy()
        c.insert(0, "baseline", m)
        curves.append(c)
    save(pd.DataFrame(rows), "blank_banks")
    save(pd.concat(curves, ignore_index=True), "blank_rank_cv")
    save(held_out_blank_residuals(data, runs), "blank_residuals")


def _split_fit(ctx, gm, method, g, pos, repeat, train):
    """One seasonal analog calibration: Addis + held-out IMPROVE slope and intercept."""
    lib = ctx.lib.iloc[pos]
    f = ab.fit_cohort(ctx.X[method][pos], lib.y, lib.Site, train=train)
    pa = f.predict(ctx.XA[method]) / ctx.addis.Volume_m3.to_numpy(float)
    st = sf._addis_stats(ctx, gm, g, pa)
    test = ~f.train
    vol = lib.tor_volume_m3.to_numpy(float)
    pt = f.predict(ctx.X[method][pos]) / vol
    tor = lib.tor_ec_ugm3.to_numpy(float)
    tt = ab.xy_stats(tor[test], pt[test]) if test.sum() > 3 else {}
    return {"baseline": method, "group": g, "repeat": repeat, "k": f.k,
            "addis_slope": st.get("deming_slope"), "addis_intercept": st.get("deming_intercept"),
            "addis_R2": st.get("R2"), "improve_slope": tt.get("deming_slope"),
            "improve_intercept": tt.get("deming_intercept"), "improve_R2": tt.get("R2"),
            "improve_n": int(test.sum())}


def stage_blank_models(ctx, gm, methods, n_repeats):
    from joblib import Parallel, delayed

    rel = sf.rel_uncertainty(ctx)
    ladders, vip_store, extra, sel = [], {}, {}, {}
    lot = np.flatnonzero(ctx.lib.lot.eq(LOT).to_numpy())
    for m in methods:
        ok = np.ones(len(ctx.lib), bool)
        ok[getattr(ctx, "failed", {}).get(m, [])] = False
        lot_m = lot[ok[lot]]
        sub = ctx.lib.iloc[lot_m]
        f = ab.fit_cohort(ctx.X[m][lot_m], sub.y, sub.Site, train=sub.split.eq("train").to_numpy())
        sf._model_record(ctx, gm, f"standard|{m}", "standard", m, "n/a", "standard", lot_m, f, rel,
                         vip_store, ladders, extra)
        for g, take in gm.items():
            pos, _ = ad.select_analogs(ctx, m, take, "oct1")
            pos = pos[ok[pos]]     # a failed filter is dropped (499 analogs instead of 500)
            sel[(m, g)] = pos
            lib = ctx.lib.iloc[pos]
            fa = ab.fit_cohort(ctx.X[m][pos], lib.y, lib.Site)
            sf._model_record(ctx, gm, f"analog|{m}|oct1|{g}", "analog", m, "oct1", g, pos, fa, rel,
                             vip_store, ladders, extra)
        log(f"{m}: standard k={f.k}; analog models done")
    ladder = pd.DataFrame(ladders)
    ladder["intercept_rel"] = ladder.deming_intercept / ladder.x_median
    save(ladder, "blank_fit_ladder")
    bands = []
    for mid, v in vip_store.items():
        b = v["bands"].copy()
        b.insert(0, "model_id", mid)
        bands.append(b)
    save(pd.concat(bands, ignore_index=True), "blank_vip_bands")
    (OUT / "blank_vip_profiles.json").write_text(json.dumps({
        "wn": {m: ctx.wn_of(m).tolist() for m in methods},
        "models": {mid: {"baseline": v["baseline"], "family": v["family"], "group": v["group"], "k": v["k"],
                         "vip": np.round(v["vip"], 4).tolist()} for mid, v in vip_store.items()}}))
    # analog overlap between baselines, per season
    ov = []
    for g in gm:
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                sa, sb = set(sel[(a, g)]), set(sel[(b, g)])
                ov.append({"group": g, "a": a, "b": b, "shared": len(sa & sb), "n": len(sa),
                           "jaccard": len(sa & sb) / len(sa | sb)})
    save(pd.DataFrame(ov), "blank_analog_overlap")
    # repeated site-grouped splits: slope and intercept for every baseline
    jobs = []
    for m in methods:
        for g in gm:
            pos = sel[(m, g)]
            sites = ctx.lib.Site.to_numpy()[pos]
            for r, train in enumerate(ab.repeated_site_splits(sites, n_repeats)):
                jobs.append((m, g, pos, r, train))
    log(f"repeats: {len(jobs)} fits")
    rows = Parallel(n_jobs=12, backend="threading")(delayed(_split_fit)(ctx, gm, *j) for j in jobs)
    save(pd.DataFrame(rows), "blank_repeats")


# --------------------------------------------------------------------------- groups
def spartan_chemistry(addis):
    from data_matching import load_filter_data

    f = load_filter_data()
    e = f.loc[f.Site.eq("ETAD") & f.Parameter.isin(CHEM)]
    wide = e.pivot_table(index="FilterId", columns="Parameter", values="Concentration", aggfunc="mean")
    wide = wide.rename(columns=CHEM)
    # SPARTAN chemistry is keyed by cartridge (ETAD-0001); the PTFE filter is ETAD-0001-1
    base = addis.filter_id.str.rsplit("-", n=1).str[0].rename("cartridge")
    return pd.DataFrame(base).merge(wide, left_on="cartridge", right_index=True, how="left")


def standard_predictions(ctx):
    """Out-of-sample Addis predictions of the standard spline model (FTIR EC, µg/m³)."""
    lot = np.flatnonzero(ctx.lib.lot.eq(LOT).to_numpy())
    sub = ctx.lib.iloc[lot]
    f = ab.fit_cohort(ctx.X[SPLINE][lot], sub.y, sub.Site, train=sub.split.eq("train").to_numpy())
    return f.predict(ctx.XA[SPLINE]) / ctx.addis.Volume_m3.to_numpy(float), f.k


def _deviations(X):
    """Unit-norm spectra minus their season mean, then PCA scores (90 % of variance).

    Raw correlations within a season are ~0.95, so clustering them only peels off
    outliers. Normalising removes loading; subtracting the season mean leaves the
    shape differences the grouping is about.
    """
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    Xd = Xn - Xn.mean(0)
    U, S, _ = np.linalg.svd(Xd, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    n = int(np.searchsorted(np.cumsum(var), 0.90) + 1)
    return U[:, :n] * S[:n], var[:n]


def _ward(scores, k):
    from scipy.cluster.hierarchy import fcluster, linkage

    return fcluster(linkage(scores, method="ward"), k, criterion="maxclust")


def cluster_season(scores, ks=(2, 3, 4), min_size=5):
    from scipy.cluster.hierarchy import leaves_list, linkage
    from sklearn.metrics import silhouette_score

    cands = []
    for k in ks:
        lab = _ward(scores, k)
        sizes = np.bincount(lab)[1:]
        cands.append({"k": k, "silhouette": float(silhouette_score(scores, lab)), "labels": lab,
                      "min_size": int(sizes.min())})
    # prefer splits with no tiny clusters; if every split isolates a few outliers, say so
    ok = [c for c in cands if c["min_size"] >= min_size] or cands
    best = max(ok, key=lambda c: c["silhouette"])
    best["leaves"] = leaves_list(linkage(scores, method="ward"))
    return best


def null_silhouette(scores, k, n=200, seed=20261007):
    """95th percentile silhouette for structureless data: Gaussian scores, same PC variances."""
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    sd = scores.std(axis=0)
    out = [silhouette_score(z, _ward(z, k)) for z in (rng.normal(size=scores.shape) * sd for _ in range(n))]
    return float(np.percentile(out, 95))


def _kw(col, idx, lab):
    from scipy.stats import kruskal

    if col is None:
        return np.nan
    v = col.to_numpy(float)[idx]
    parts = [v[(lab == l) & np.isfinite(v)] for l in np.unique(lab)]
    return float(kruskal(*parts).pvalue) if all(len(p) >= 3 for p in parts) else np.nan


def stage_groups(ctx, gm, methods):
    from scipy.stats import kruskal, spearmanr

    addis = ctx.addis
    fabs10 = addis.Fabs.to_numpy(float) / ab.MAC_VALUE
    pred, k_std = standard_predictions(ctx)
    chem = spartan_chemistry(addis)
    pct_all = pd.Series(fabs10).rank(pct=True).to_numpy() * 100
    members, summary, tests, spectra, heat = [], [], [], {}, {}
    seasons = [g for g in gm if g != "All Addis"]
    for m in methods:
        wn = ctx.wn_of(m)
        for region, spec in REGIONS.items():
            keep = ad.region_mask(wn, **spec)
            for g in seasons:
                idx = np.flatnonzero(gm[g])
                X = ctx.XA[m][idx][:, keep]
                R = np.corrcoef(X)
                sc, var = _deviations(X)
                c = cluster_season(sc)
                lab = c["labels"]
                # name clusters by median HIPS EC, low -> high
                med = {l: np.median(fabs10[idx][lab == l]) for l in np.unique(lab)}
                rename = {l: i + 1 for i, l in enumerate(sorted(med, key=med.get))}
                lab = np.array([rename[l] for l in lab])
                null95 = null_silhouette(sc, c["k"])
                resid = pred[idx] - fabs10[idx]
                groups = [fabs10[idx][lab == l] for l in np.unique(lab)]
                p_ec = float(kruskal(*groups).pvalue) if len(groups) > 1 else np.nan
                p_res = float(kruskal(*[resid[lab == l] for l in np.unique(lab)]).pvalue)
                # continuous check: first principal component of the shape deviations vs EC
                pc1 = sc[:, 0]
                rho = float(spearmanr(pc1, fabs10[idx]).statistic)
                rho_res = float(spearmanr(pc1, resid).statistic)
                tests.append({"baseline": m, "region": region, "group": g, "n": len(idx), "k": c["k"],
                              "n_pcs": int(sc.shape[1]), "pc1_variance": float(var[0]),
                              "silhouette": c["silhouette"], "silhouette_null_p95": null95,
                              "min_cluster": c["min_size"],
                              # a split that only isolates a few outliers is not a grouping
                              "structure_beyond_null": bool(c["silhouette"] > null95 and c["min_size"] >= 5),
                              "r_within_median": float(np.median(R[np.triu_indices(len(idx), 1)])),
                              "kruskal_p_ec": p_ec, "kruskal_p_residual": p_res,
                              "pc1_spearman_ec": rho, "pc1_spearman_residual": rho_res,
                              "kruskal_p_nitrate": _kw(chem.get("nitrate"), idx, lab),
                              "kruskal_p_Al": _kw(chem.get("Al"), idx, lab),
                              "kruskal_p_K_ion": _kw(chem.get("K_ion"), idx, lab)})
                top_q = fabs10[idx] >= np.percentile(fabs10[idx], 75)
                for l in np.unique(lab):
                    t = lab == l
                    row = {"baseline": m, "region": region, "group": g, "cluster": int(l), "n": int(t.sum()),
                           "ec_fabs10_median": float(np.median(fabs10[idx][t])),
                           "ec_pct_all_addis_median": float(np.median(pct_all[idx][t])),
                           "top_quartile_share": float(top_q[t].mean()),
                           "ftir_std_median": float(np.median(pred[idx][t])),
                           "residual_median": float(np.median(resid[t])),
                           "excluded_n": int(addis.is_excluded.to_numpy(bool)[idx][t].sum())}
                    for col in CHEM.values():
                        if col in chem:
                            row[f"{col}_median"] = float(np.nanmedian(chem[col].to_numpy(float)[idx][t]))
                    summary.append(row)
                    full = ctx.XA[m][idx][t]
                    spectra[f"{m}|{region}|{g}|{int(l)}"] = np.round(np.median(full, axis=0), 5).tolist()
                members.append(pd.DataFrame({"baseline": m, "region": region, "group": g,
                                             "filter_id": addis.filter_id.to_numpy()[idx], "cluster": lab,
                                             "leaf_order": np.argsort(c["leaves"]),
                                             "ec_fabs10": fabs10[idx], "ftir_std": pred[idx], "residual": resid,
                                             "pc1": pc1, "excluded": addis.is_excluded.to_numpy(bool)[idx]}))
                heat[f"{m}|{region}|{g}"] = {"order": c["leaves"].tolist(),
                                              "r": np.round(R[np.ix_(c["leaves"], c["leaves"])], 3).tolist()}
    save(pd.concat(members, ignore_index=True), "groups_members")
    save(pd.DataFrame(summary), "groups_summary")
    save(pd.DataFrame(tests), "groups_tests")
    (OUT / "groups_spectra.json").write_text(json.dumps(
        {"wn": {m: ctx.wn_of(m).tolist() for m in methods}, "median": spectra, "heat": heat,
         "standard_k": int(k_std), "regions": REGION_LABELS}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--stages", default="blanks,groups")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx, data = load_context()
    gm = ab.group_masks(ctx)
    runs, sources = load_policy_runs(ctx, data)
    methods = [SPLINE, CAPPED, *runs]
    stages = set(args.stages.split(","))
    if "blanks" in stages:
        stage_background(data, runs)
        stage_blank_models(ctx, gm, methods, args.repeats)
        log("blanks done")
    if "groups" in stages:
        stage_groups(ctx, gm, [CAPPED, *runs])
        log("groups done")
    summary = {"source": "1 Oct 2026 Ann/Satoshi task list; Satoshi email 2 Oct 2026",
               "baselines": {m: LABELS[m] for m in methods}, "regions": REGION_LABELS,
               "analog_mask": ad.MASK_LABELS["oct1"], "n_repeats": args.repeats, "n_analogs": ab.N_ANALOGS,
               "mac": ab.MAC_VALUE, "addis_n": len(ctx.addis), "improve_n": len(ctx.lib),
               "source_sha256": {**data["sources"], **sources}, "seconds": round(time.time() - t0)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
