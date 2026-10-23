"""Addis on every public SPARTAN filter (Ann, 19 Oct 2026: use all available QC'd data).

The AAAR 2026 poster keeps the 189 Addis filters that have a deployed SPARTAN FTIR EC (233 in the
meeting evaluation set). The public SPARTAN release v3.1.4 has 246 ETAD filters with both FTIR EC
(method 218) and HIPS BC (method 221). This reruns the spline-baseline (AIRSpec) calibrations on them:

  allimprove  the all-IMPROVE lot-251 model ('allimprove|AIRSpec|all-IMPROVE lot 251')
  seasonal    the locked seasonal-analog models ('seasonal|AIRSpec|locked|<season>'), twice:
              'kept'      analogs chosen from the 233-filter seasons (the poster's models, unchanged)
              'reselected' analogs chosen again from the 246-filter seasons, then refitted
  grid        the spline half of the calibration-explorer grid (3,138 configurations,
              run_meeting_grid_20260917.configs), rankings recomputed on the 246 filters

No spectrum is re-baselined: the frozen Colab run that the poster uses already corrected every Addis
target (253), including the 14 public filters that were left out of the evaluation set because the
in-house HIPS batch file had no Fabs for them. Outputs keep one row per Addis filter of the union of
the old 233 and the public 246 (247; ETAD-0246 is in the old set but not in the public release), so
the notebook can score either population. Fabs: 'Fabs_eval' (the poster's) and 'Fabs_public'
(10 x public BC PM2.5).

Run from the repo root (about 30-60 min for the grid; the other stages take a minute):
    ~/anaconda3/bin/python research/ftir_hips_chem/workflows/run_addis_public246_20261020.py all
The notebook is addis_public246_20261020.ipynb.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / "scripts"))
import analog_bias as ab  # noqa: E402
from config import season_for_month  # noqa: E402
from vibes_current import MIGRATION, load_context  # noqa: E402

OUT = AREA / "output/tables/addis_public246_20261020"
PUBLIC = ROOT / "data/spartan/raw/FilterBased/ChemSpecPM25/FilterBased_ChemSpecPM25_ETAD.csv"
FROZEN = MIGRATION / "meeting"
METHOD = "AIRSpec"
LOT = 251
LOCK = threading.Lock()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def public_etad():
    """One row per public ETAD filter with both FTIR EC and HIPS BC (µg/m³)."""
    d = pd.read_csv(PUBLIC, comment="#", low_memory=False)
    d = d[d.Parameter_Name.isin(["EC PM2.5", "BC PM2.5"])]
    methods = d.groupby("Parameter_Name").Method_Code.unique()
    if list(methods["EC PM2.5"]) != [218] or list(methods["BC PM2.5"]) != [221]:
        raise ValueError(f"unexpected method codes: {methods.to_dict()}")
    d = d.assign(date=pd.to_datetime(dict(year=d.Start_Year_local, month=d.Start_Month_local,
                                          day=d.Start_Day_local)))
    w = d.pivot_table(index=["Filter_ID", "date", "Hours_sampled"], columns="Parameter_Name",
                      values="Value", aggfunc="first").reset_index()
    w = w.rename(columns={"Filter_ID": "fid", "date": "date_public", "EC PM2.5": "EC_public",
                          "BC PM2.5": "BC_public"}).dropna(subset=["EC_public", "BC_public"])
    if w.fid.duplicated().any():
        raise ValueError("duplicate public filter ids")
    version = next(l.split("version", 1)[1].strip() for l in open(PUBLIC) if l.startswith("# Data version"))
    return w, version


def extended_context():
    """ctx with ctx.addis = union of the poster's 233 and the public 246 Addis filters."""
    ctx, data = load_context()
    ctx.methods = (METHOD,)
    cases = data["cases"]
    t = cases.loc[cases.kind.eq("target")].copy()
    t["MediaId"] = t.sample_id.str.split(":").str[1].astype(int)
    t["row"] = t.index.to_numpy()
    t["fid"] = t.filter_id.astype(str).str.rsplit("-", n=1).str[0]
    pub, version = public_etad()
    old = ctx.addis.copy()
    old["fid"] = old.ExternalFilterId.str.rsplit("-", n=1).str[0]
    missing = sorted(set(pub.fid) - set(t.fid))
    if missing:
        raise ValueError(f"public filters without a frozen corrected spectrum: {missing}")
    keep = t[t.fid.isin(set(pub.fid) | set(old.fid))]
    if keep.fid.duplicated().any():
        raise ValueError("a physical filter has two frozen targets")
    a = keep[["MediaId", "row", "fid", "filter_id", "date", "volume"]].rename(
        columns={"filter_id": "ExternalFilterId", "volume": "Volume_m3"})
    a["date"] = pd.to_datetime(a.date)
    a = a.merge(pub, on="fid", how="left").merge(
        old[["MediaId", "Fabs", "Volume_m3", "season", "row"]].rename(
            columns={"Fabs": "Fabs_eval", "Volume_m3": "Volume_eval", "season": "season_eval", "row": "row_eval"}),
        on="MediaId", how="left")
    a["in_old"], a["in_public"] = a.Fabs_eval.notna(), a.EC_public.notna()
    a["in_poster"] = a.MediaId.isin(set(pd.read_csv(FROZEN / "addis_predictions.csv").query(
        "fit_id == 'deployed|SPARTAN|current SPARTAN calibration'").dropna(subset=["ftir_ec_ugm3"]).MediaId))
    a["Fabs_public"] = 10 * a.BC_public
    a["season"] = a.date.dt.month.map(season_for_month)
    a["month_block"] = a.date.dt.strftime("%Y-%m")
    o = a[a.in_old]
    checks = {"row": (o.row == o.row_eval).all(), "volume": np.allclose(o.Volume_m3, o.Volume_eval),
              "season": (o.season == o.season_eval).all(),
              "date": (a.date[a.in_public] == a.date_public[a.in_public]).all()}
    if not all(checks.values()):
        raise ValueError(f"extended Addis frame disagrees with the evaluation set: {checks}")
    if a.Volume_m3.isna().any() or (a.Volume_m3 <= 0).any():
        raise ValueError("missing sampled volume")
    a = a.sort_values(["date", "MediaId"]).reset_index(drop=True)
    a = a.drop(columns=["row_eval", "Volume_eval", "date_public"])
    ctx.addis = a
    ctx.X = {METHOD: ctx.X[METHOD]}
    ctx.XA = {METHOD: np.asarray(data["arrays"][METHOD][a.row.to_numpy()], float)}
    ctx.meta = {"public_version": version, "n_union": len(a), "n_old": int(a.in_old.sum()),
                "n_public": int(a.in_public.sum()), "n_poster": int(a.in_poster.sum()),
                "old_not_public": a.loc[a.in_old & ~a.in_public, "fid"].tolist(),
                "public_not_old": a.loc[a.in_public & ~a.in_old, "fid"].tolist()}
    return ctx


def group_masks(ctx, population):
    """Selection groups over one population ('old' = the poster's 233, 'public' = the 246)."""
    on = ctx.addis[f"in_{population}"].to_numpy()
    g = {"All Addis": on}
    g.update({s: on & ctx.addis.season.eq(s).to_numpy() for s in ab.SEASONS})
    return g


def predictions_frame(ctx, pred, fit_id, **extra):
    p = ctx.addis[["MediaId", "ExternalFilterId", "fid", "date", "season"]].copy()
    p["ftir_ec_ugm3"], p["fit_id"] = pred, fit_id
    for k, v in extra.items():
        p[k] = v
    return p


def stage_allimprove(ctx):
    lot = ctx.lib.lot.eq(LOT).to_numpy()
    idx = np.flatnonzero(lot)
    sub = ctx.lib.iloc[idx]
    f = ab.fit_cohort(ctx.X[METHOD][idx], sub.y, sub.Site, train=sub.split.eq("train").to_numpy())
    pa = f.predict(ctx.XA[METHOD]) / ctx.addis.Volume_m3.to_numpy(float)
    fit_id = f"allimprove|{METHOD}|all-IMPROVE lot {LOT}"
    log(f"{fit_id}: k={f.k}")
    return predictions_frame(ctx, pa, fit_id, k=f.k), [{"fit_id": fit_id, "k": f.k, "train_n": int(f.train.sum())}]


def stage_seasonal(ctx):
    vol = ctx.addis.Volume_m3.to_numpy(float)
    preds, fits, members = [], [], []
    masks = {"kept": group_masks(ctx, "old"), "reselected": group_masks(ctx, "public")}
    chosen = {}
    for g in ["All Addis", *ab.SEASONS]:
        for variant, gm in masks.items():
            pos, _ = ab.correlation_analogs(ctx, METHOD, gm[g], ab.LOCKED_MASK)
            chosen[(variant, g)] = pos
            lib = ctx.lib.iloc[pos]
            f = ab.fit_cohort(ctx.X[METHOD][pos], lib.y, lib.Site)
            fit_id = f"seasonal|{METHOD}|locked|{g}"
            preds.append(predictions_frame(ctx, f.predict(ctx.XA[METHOD]) / vol, fit_id, variant=variant, k=f.k))
            fits.append({"fit_id": fit_id, "variant": variant, "group": g, "k": f.k, "cohort_n": len(pos),
                         "addis_n_selecting": int(gm[g].sum()), "train_n": int(f.train.sum()),
                         "test_sites": int(lib.Site[~f.train].nunique())})
            m = lib[["filter_id", "Site"]].assign(variant=variant, group=g, rank=np.arange(1, len(pos) + 1))
            members.append(m)
        shared = len(set(chosen[("kept", g)]) & set(chosen[("reselected", g)]))
        fits[-1]["analogs_shared_with_kept"] = fits[-2]["analogs_shared_with_kept"] = shared
        log(f"seasonal {g}: kept k={fits[-2]['k']}, reselected k={fits[-1]['k']}, {shared}/500 analogs shared")
    return pd.concat(preds, ignore_index=True), fits, pd.concat(members, ignore_index=True)


def grid_fit(ctx, method, cat, cut, mode, pos, sink, grid):
    lib = ctx.lib.iloc[pos]
    X = ctx.X[method][pos]
    f = ab.fit_cohort(X, lib.y, lib.Site, mode=mode)
    pa = f.predict(ctx.XA[method]) / ctx.addis.Volume_m3.to_numpy(float)
    row = {"key": grid.key(method, cat, cut, mode), "method": method, "category": cat, "label": grid.LABELS[cat],
           "cutoff": int(cut), "mode": mode, "k": f.k, "train_n": int(f.train.sum())}
    if mode == "site_heldout":
        test = ~f.train
        row["heldout_R2"] = ab.xy_stats(lib.y.to_numpy()[test], f.predict(X)[test]).get("R2")
    row["pred"] = np.round(pa, 6).tolist()
    with LOCK:
        sink.write(json.dumps(row) + "\n")
        sink.flush()
    return 1


def run_grid(ctx, n_jobs):
    grid = module(AREA / "workflows/run_meeting_grid_20260917.py", "public246_grid")
    full, XA_full = ctx.addis, ctx.XA
    pub = full.in_public.to_numpy()
    ctx.addis, ctx.XA = full[pub].reset_index(drop=True), {METHOD: XA_full[METHOD][pub]}
    try:
        configs = grid.configs(ctx)
    finally:
        ctx.addis, ctx.XA = full, XA_full
    results = OUT / "grid_predictions.jsonl"
    done = set()
    if results.exists():
        done = {json.loads(l)["key"] for l in results.read_text().splitlines() if l.strip()}
    todo = [c for c in configs if grid.key(*c[:4]) not in done]
    todo.sort(key=lambda c: -len(c[4]))
    log(f"grid: {len(configs)} spline configurations, {len(todo)} to fit ({len(done)} done)")
    t0 = time.time()
    with results.open("a") as sink, threadpool_limits(limits=1):
        n = 0
        for chunk in range(0, len(todo), 240):
            part = todo[chunk:chunk + 240]
            n += sum(Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(grid_fit)(ctx, *c, sink, grid) for c in part))
            log(f"grid [{time.time() - t0:.0f}s] {n}/{len(todo)}")
    return len(configs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["context", "fits", "grid", "all"])
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = extended_context()
    ctx.addis.drop(columns=["row"]).to_csv(OUT / "addis_filters.csv", index=False)
    summary = {**ctx.meta, "lambda_addis": ctx.lam, "sigma_x": ctx.sigma_x, "sigma_y": ab.SIGMA_Y_PROXY,
               "mac": ab.MAC_VALUE, "method": METHOD, "lot_model": LOT, "locked_mask": ab.LOCKED_MASK,
               "n_analogs": ab.N_ANALOGS, "seasons": ab.SEASONS}
    log(f"Addis: {ctx.meta}")
    if args.stage in ("fits", "all"):
        with threadpool_limits(limits=2):
            p1, f1 = stage_allimprove(ctx)
            p2, f2, members = stage_seasonal(ctx)
        pd.concat([p1, p2], ignore_index=True).to_csv(OUT / "addis_predictions.csv", index=False)
        pd.DataFrame(f1 + f2).to_csv(OUT / "fits.csv", index=False)
        members.to_csv(OUT / "seasonal_analog_membership.csv", index=False)
    if args.stage in ("grid", "all"):
        summary["grid_configs"] = run_grid(ctx, args.jobs)
    summary["seconds"] = round(time.time() - t0)
    path = OUT / "summary.json"
    prior = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**prior, **summary}, indent=2) + "\n")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
