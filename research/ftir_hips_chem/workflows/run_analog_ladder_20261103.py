"""Seasonal analog calibrations at every cohort size, for the gallery's analog picker.

The meeting's seasonal calibrations are fitted on the top 500 analogs. A different
count or selection rule is a different calibration, so filtering the saved
predictions would not answer "what if we used 87": each size here is its own PLS
fit, under the same protocol as ``run_meeting_followup_20260917.stage_seasonal``
(site-grouped split, k by the protocol rule, rank-70 inputs).

Rules: Pearson r to the group's median Addis spectrum over the 17 Sep or the 1 Oct
region, and lowest TOR OC/EC. The size-500 fits of the 17 Sep rule must reproduce the
saved calibrations or the run stops. A size the protocol cannot fit (its top filters
come from fewer than five sites, as low OC/EC does at small sizes) is recorded with
k empty, so the gallery can say so. One npz per rule, baseline and group, written as
each finishes; finished ones are skipped, so it can be stopped and resumed.

Run from the repo root:
    uv run --no-sync python research/ftir_hips_chem/workflows/run_analog_ladder_20261103.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")   # before numpy: threadpoolctl cannot reach Accelerate

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

AREA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AREA / "scripts"))
import analog_bias as ab  # noqa: E402
import analog_transfer_checks as atc  # noqa: E402
import vibes_current  # noqa: E402

OUT = AREA / "output/tables/analog_ladder_20261103"
SAVED = AREA / "output/tables/vibes_fullrange/gallery_rank70_20261002/meeting"
N_MIN, N_MAX = 20, 1000
N_JOBS = 12
GROUP_SLUG = {"All Addis": "all", "Dry (Oct-Feb)": "dry", "Belg (Mar-May)": "belg", "Kiremt (Jun-Sep)": "kiremt"}
METHOD_SLUG = {"AIRSpec": "spline", "VIBES-full-r70": "vibes-full", "VIBES-full-cut-r70": "vibes-cut"}


def rankings(ctx, gm):
    """Library positions in selection order, best first, per (rule, method, group)."""
    out = {}
    for method in ctx.methods:
        for g, take in gm.items():
            out["similarity-sep17", method, g] = ab.correlation_ranking(ctx, method, take)[:N_MAX]
            oct1 = atc.select_analogs(ctx, method, take, "oct1", n=N_MAX)[0]
            # grids that stop at 1425 cm⁻¹ have none of what 1 Oct changes: the same analogs, the same fits
            if not np.array_equal(oct1, out["similarity-sep17", method, g]):
                out["similarity-oct1", method, g] = oct1
        # a property of the filter, not of a season: one ranking for every group
        out["lowest-ocec", method, "All Addis"] = ab.ocec_ranking(ctx)[:N_MAX]
    return out


def sweep(path, X, XA, y, sites, volume, fabs10, tor, addis_volume, rel, positions, filter_id, sizes=None):
    """Every cohort size on one ranking: the fit's readouts and its per-filter predictions, saved to ``path``."""
    from threadpoolctl import threadpool_limits

    sizes = np.arange(N_MIN, N_MAX + 1) if sizes is None else sizes
    rows, addis, held_mask, held = [], [], [], []
    with threadpool_limits(limits=1):
        for n in sizes:
            try:
                f = ab.fit_cohort(X[:n], y[:n], sites[:n])
            except ValueError as e:
                if "fewer unique groups" not in str(e):
                    raise
                rows.append({"n": int(n), "k": None, "train_n": 0, "test_n": 0, "test_sites": 0, "TOR_test_R2": None,
                             "TOR_test_RMSE_ug": None, "lambda_tor_fabs": None, "lambda_ftir_tor": None, "lambda_ftir_fabs": None})
                addis.append(np.full(len(XA), np.nan))
                held_mask.append(np.zeros(N_MAX, bool))
                held.append(np.empty(0))
                continue
            whole = f.predict(X[:n])
            test = ~f.train
            pos = np.flatnonzero(test)
            score = ab.xy_stats(y[:n][test], whole[test]) if test.sum() > 2 else {}
            rmse = score.get("RMSE")
            # the error ratios of analog_panels, on the held-out filters of this size
            s_fabs, s_tor = rel * np.nanmedian(fabs10[pos]), rel * np.nanmedian(tor[pos])
            s_ftir = rmse / np.nanmedian(volume[pos]) if rmse else s_tor
            rows.append({"n": int(n), "k": f.k, "train_n": int(f.train.sum()), "test_n": int(test.sum()),
                         "test_sites": len(set(sites[:n][test])), "TOR_test_R2": score.get("R2"), "TOR_test_RMSE_ug": rmse,
                         "lambda_tor_fabs": (s_tor / s_fabs) ** 2, "lambda_ftir_tor": (s_ftir / s_tor) ** 2,
                         "lambda_ftir_fabs": (s_ftir / s_fabs) ** 2})
            addis.append(f.predict(XA) / addis_volume)
            mask = np.zeros(N_MAX, bool)
            mask[pos] = True
            held_mask.append(mask)
            held.append(whole[test] / volume[pos])
    fits = pd.DataFrame(rows)
    if path is None:
        return fits
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, positions=positions, filter_id=filter_id, fits=fits.to_json(orient="records"),
                        addis=np.array(addis, np.float32), held_mask=np.array(held_mask),
                        held=np.concatenate(held).astype(np.float32))
    tmp.rename(path)
    return path.name, int(fits.k.isna().sum())


def check_reproduces(ctx, gm):
    """Size 500 under the 17 Sep rule is the saved seasonal calibration, filter by filter."""
    fits = pd.read_csv(SAVED / "fits.csv").set_index("fit_id")
    saved = pd.read_csv(SAVED / "addis_predictions.csv")
    for method in ctx.methods:
        for g in gm:
            z = np.load(OUT / f"similarity-sep17.{METHOD_SLUG[method]}.{GROUP_SLUG[g]}.npz", allow_pickle=True)
            row = pd.read_json(str(z["fits"]), orient="records").set_index("n").loc[500]
            fit = fits.loc[f"seasonal|{method}|locked|{g}"]
            if row.k != fit.k or row.train_n != fit.train_n:
                raise ValueError(f"{method} {g}: size 500 k/train {row.k}/{row.train_n} vs saved {fit.k}/{fit.train_n}")
            ref = saved.loc[saved.fit_id.eq(f"seasonal|{method}|locked|{g}")].set_index("MediaId").ftir_ec_ugm3
            ours = pd.Series(z["addis"][500 - N_MIN], index=ctx.addis.MediaId)
            worst = float((ours - ref.reindex(ours.index)).abs().max())
            if not worst < 1e-3:
                raise ValueError(f"{method} {g}: size 500 Addis predictions differ by {worst:.4g} µg/m³")
    print("size 500 reproduces every saved seasonal calibration", flush=True)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ctx, data = vibes_current.load_context()
    gm = ab.group_masks(ctx)
    order = rankings(ctx, gm)
    print(f"context and {len(order)} rankings in {time.time() - t0:.0f} s", flush=True)
    lib = ctx.lib
    rel = ctx.sigma_x / float(np.median(ctx.addis.Fabs.to_numpy(float) / ab.MAC_VALUE))
    addis_volume = ctx.addis.Volume_m3.to_numpy(float)

    def path(rule, method, g):
        return OUT / f"{rule}.{METHOD_SLUG[method]}.{GROUP_SLUG[g]}.npz"

    def inputs(key):
        rule, method, g = key
        pos = order[key]
        sub = lib.iloc[pos]
        return (ctx.X[method][pos], ctx.XA[method], sub.y.to_numpy(float), sub.Site.to_numpy(),
                sub.tor_volume_m3.to_numpy(float), sub.fabs_Mm1.to_numpy(float) / ab.MAC_VALUE,
                sub.tor_ec_ugm3.to_numpy(float), addis_volume, rel, pos, lib.filter_id.values[pos])

    # the small sizes and a few across the range, every ranking, before the long run commits to them
    if "--skip-probe" not in sys.argv:
        for key in order:
            probe = sweep(None, *inputs(key), sizes=np.array([20, 21, 25, 30, 50, 87, 200, 500, 1000]))
            print(f"probe {key}: k {probe.k.tolist()}", flush=True)
    if "--probe" in sys.argv:
        return

    todo = [k for k in order if not path(*k).exists()]
    print(f"{len(todo)} of {len(order)} rankings to sweep, sizes {N_MIN}–{N_MAX}", flush=True)
    results = Parallel(n_jobs=N_JOBS, return_as="generator")(delayed(sweep)(path(*k), *inputs(k)) for k in todo)
    for name, failed in results:
        print(f"{name} done, {failed} sizes unfittable, {time.time() - t0:.0f} s", flush=True)
    check_reproduces(ctx, gm)
    vibes_current.receipt(OUT, data["sources"])


if __name__ == "__main__":
    main()
