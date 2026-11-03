"""Lot and FTIR-EC controls for the seasonal spectral ML (follow-up to run_seasonal_spectral_ml.py).

Three checks the first analysis did not run:

1. Lot. The 2022–23 training cycle has every Bega filter on lot 248 and every
   Kiremt filter on lot 251; both test cycles are lot 251. Lots are filled from
   SPARTAN's filter table, which also covers the filters the native lot map
   lacks. Season classifiers are then re-tested with lot held fixed.
2. The EC comparator. D = FTIR EC(X) − HIPS BC. A straight line on FTIR EC
   (no spectral channel) is scored on every saved disagreement fold.
3. Information beyond EC. Spectral models are fitted to D − line(EC) and the
   same grouped region permutation is run on that residual and on D.

Models, folds, windows and seeds are the first analysis's own (imported from
scripts/seasonal_spectral_ml.py). Spectra come from the native arrays when they
are available; otherwise from the per-filter curves exported to the gallery,
which reproduce the saved scores (the receipt records which source and how
closely). Run from the repository root:

    .venv/bin/python research/ftir_hips_chem/workflows/run_seasonal_ml_checks.py \
        --lots "<SPARTAN HIPS bundle>/spartan_Filters.csv"
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from seasonal_spectral_ml import (  # noqa: E402
    build_model, grouped_region_permutation, make_cycle_folds, make_feature_frame, spectral_regions,
)
from seasonal_ml_checks import (  # noqa: E402
    ECLine, fill_lot_groups, lot_season_counts, regression_scores, spartan_lot_map,
)
from run_seasonal_spectral_ml import NAMED_REGIONS, SCAN_QC_OMIT, SEED  # noqa: E402

TABLES = ROOT / "research/ftir_hips_chem/output/tables"
MAIN = TABLES / "seasonal_spectral_ml_20261103"
SPECTRA = ROOT / "gallery/app/public/data/research/baseline_lab/spectra"
SEASONS = ("Bega (Oct-Jan, dry)", "Belg (Feb-May, short rains)", "Kiremt (Jun-Sep, long rains)")
FAMILY_LABEL = {"rf": "Random forest", "linear": "PCA + ridge", "pls": "PLS"}
REP_LABEL = {"raw_full": "Raw · full spectrum", "raw_common": "Raw · 1560–4000 cm⁻¹",
             "spline_common": "Spline-corrected · 1560–4000 cm⁻¹",
             "removed_common": "Removed spline component · 1560–4000 cm⁻¹"}
# Raw-continuum linear/PLS models are pushed far off their training manifold by a
# block shuffle (importances of 3–15 µg/m³); only RF is tested on raw spectra.
IMPORTANCE_MODELS = (("rf", "spline_common"), ("linear", "spline_common"), ("pls", "spline_common"),
                     ("rf", "raw_full"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", flush=True)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if value is pd.NA:
        return None
    return value


def load_spectra(cohort, arrays_path, spectra_dir):
    """Per-physical-filter raw and spline-corrected spectra in cohort order."""
    if arrays_path and Path(arrays_path).is_file():
        with np.load(arrays_path, allow_pickle=False) as a:
            ids = a["physical_filter_ids"].astype(str)
            order = pd.Index(ids).get_indexer(cohort.id)
            if (order < 0).any():
                raise ValueError("native arrays lack some cohort filters")
            wn, shared = a["wavenumbers"].copy(), a["shared_mask"].copy()
            raw = a["physical_raw"][order].copy()
            cor = a["physical_corrected__spline_default"][order].copy()
        return wn, raw, cor, shared, {"kind": "native_method_arrays", "path": str(Path(arrays_path).relative_to(ROOT)),
                                      "sha256": sha256(arrays_path)}
    raw, cor, wn, digest = [], [], None, hashlib.sha256()
    for identifier in cohort.id:
        path = Path(spectra_dir) / f"{identifier}.json"
        blob = path.read_bytes()
        digest.update(blob)
        curves = {c["id"].removeprefix(identifier + "-"): c for c in json.loads(blob)["curves"]}
        w = np.asarray(curves["raw"]["wn"], float)
        if wn is None:
            wn = w
        elif not np.allclose(w, wn):
            raise ValueError("gallery spectra do not share one wavenumber grid")
        raw.append(np.asarray(curves["raw"]["y"], float))
        cor.append(np.asarray([np.nan if v is None else v for v in curves["spline"]["y"]], float))
    raw, cor = np.vstack(raw), np.vstack(cor)
    shared = np.isfinite(raw).all(axis=0) & np.isfinite(cor).all(axis=0)
    return wn, raw, cor, shared, {"kind": "gallery_baseline_lab_spectra", "path": str(Path(spectra_dir).relative_to(ROOT)),
                                  "sha256_of_cohort_files": digest.hexdigest(),
                                  "note": "Per-filter means exported from the native arrays; rounded for display."}


def representations(wn, raw, cor, shared):
    common = shared & (wn >= 1560) & (wn <= 4000)
    reps = {"raw_full": (raw, wn), "raw_common": (raw[:, common], wn[common]),
            "spline_common": (cor[:, common], wn[common]),
            "removed_common": ((raw - cor)[:, common], wn[common])}
    for key, (values, _) in reps.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{key} contains unsupported channels")
    return reps


def cohort_rows(cohort, name):
    if name == "direct232":
        return np.ones(len(cohort), bool)
    if name == "scan_qc229":
        return ~cohort.id.isin(SCAN_QC_OMIT).to_numpy()
    if name == "weather188":
        return np.isfinite(cohort[["rh", "log_rain"]]).all(axis=1).to_numpy()
    raise ValueError(f"unknown cohort {name}")


def folds_for(cohort, keep, calendar):
    use = cohort.loc[keep].reset_index(drop=True)
    folds, _ = make_cycle_folds(use.date.to_numpy(), use.id.to_numpy(), convention=calendar, mode="forward",
                                train_coverage="all_seasons", test_coverage="all_seasons", min_train=50, min_test=20)
    return use, {f.name: f for f in folds}


def ec_line_scores(cohort, main_scores):
    """EC-only line on every (calendar, cohort, fold) that has saved disagreement scores."""
    rows, preds = [], []
    contexts = (main_scores.loc[main_scores.task.eq("disagreement"), ["calendar", "cohort", "fold_id", "n_train", "n_test"]]
                .drop_duplicates())
    for ctx in contexts.itertuples(index=False):
        use, folds = folds_for(cohort, cohort_rows(cohort, ctx.cohort), ctx.calendar)
        fold = folds.get(ctx.fold_id)
        if fold is None or (len(fold.train), len(fold.test)) != (ctx.n_train, ctx.n_test):
            raise ValueError(f"cannot reproduce fold {ctx} for the EC line")
        tr, te = use.iloc[fold.train], use.iloc[fold.test]
        line = ECLine().fit(tr.ec_ftir_ugm3, tr.optical_disagreement_ugm3)
        p = line.predict(te.ec_ftir_ugm3)
        dummy = np.sqrt(np.mean((te.optical_disagreement_ugm3 - tr.optical_disagreement_ugm3.mean()) ** 2))
        rows.append({"task": "disagreement", "calendar": ctx.calendar, "cohort": ctx.cohort, "fold_id": ctx.fold_id,
                     "model_id": "ec_line", "n_train": len(tr), "n_test": len(te),
                     **regression_scores(te.optical_disagreement_ugm3, p), "dummy_rmse": float(dummy),
                     "slope": line.slope_, "intercept": line.intercept_})
        if ctx.calendar == "belg_feb" and ctx.cohort == "direct232":
            preds += [{"fold_id": ctx.fold_id, "id": i, "date": d, "season": s, "lot_group": lot, "actual": a,
                       "ec_ftir_ugm3": ec, "ec_line": q}
                      for i, d, s, lot, a, ec, q in zip(te.id, te.date, te.season, te.lot_group,
                                                       te.optical_disagreement_ugm3, te.ec_ftir_ugm3, p)]
    return pd.DataFrame(rows), pd.DataFrame(preds)


def residual_models(cohort, reps, main_scores, repeats, jobs):
    """D directly vs EC line + spectral residual, and region permutation on both targets."""
    use, folds = folds_for(cohort, cohort_rows(cohort, "direct232"), "belg_feb")
    D, EC = use.optical_disagreement_ugm3.to_numpy(), use.ec_ftir_ugm3.to_numpy()
    sub = main_scores.loc[main_scores.task.eq("disagreement") & main_scores.calendar.eq("belg_feb")
                          & main_scores.cohort.eq("direct232")]
    saved = {(f, m): r for f, m, r in zip(sub.fold_id, sub.model_id, sub.rmse)}
    scores, preds, importance = [], [], []
    for fold_id, fold in folds.items():
        tr, te = fold.train, fold.test
        line = ECLine().fit(EC[tr], D[tr])
        line_tr, line_te = line.predict(EC[tr]), line.predict(EC[te])
        for rep in ("raw_full", "raw_common", "spline_common", "removed_common"):
            X, cols = make_feature_frame(reps[rep][0], physical_ids=use.id.to_numpy())
            for family in ("rf", "linear", "pls"):
                model_id = f"{family}_{rep}"
                for target, y_tr in (("disagreement", D[tr]), ("residual", D[tr] - line_tr)):
                    est = build_model(task="regression", model=family, spectral_columns=cols, n_jobs=jobs)
                    est.fit(X.iloc[tr], y_tr)
                    raw_pred = est.predict(X.iloc[te])
                    final = raw_pred if target == "disagreement" else line_te + raw_pred
                    row = {"fold_id": fold_id, "model_id": model_id, "family": family, "representation": rep,
                           "target": target, "n_train": len(tr), "n_test": len(te), **regression_scores(D[te], final)}
                    if target == "disagreement":
                        row["saved_rmse"] = saved.get((fold_id, model_id))
                    scores.append(row)
                    preds += [{"fold_id": fold_id, "model_id": model_id, "target": target, "id": i, "season": s,
                               "actual": a, "predicted": q}
                              for i, s, a, q in zip(use.id.iloc[te], use.season.iloc[te], D[te], final)]
                    if (family, rep) not in IMPORTANCE_MODELS:
                        continue
                    y_te = D[te] if target == "disagreement" else D[te] - line_te
                    regions = spectral_regions(reps[rep][1], cols, bin_width=100, named_regions=NAMED_REGIONS,
                                               include_whole=True)
                    subsets = [(None, np.arange(len(te)))]
                    if fold.complete_test:
                        subsets += [(s, np.flatnonzero(use.season.iloc[te].to_numpy() == s)) for s in SEASONS]
                    for season, rows_ in subsets:
                        r = pd.DataFrame(grouped_region_permutation(est, X.iloc[te].iloc[rows_], y_te[rows_], regions,
                                                                    task="regression", repeats=repeats, seed=SEED))
                        g = r.groupby("region")
                        summary = pd.DataFrame({"delta_mean": g.importance.mean(), "delta_sd": g.importance.std(ddof=1),
                                                "q05": g.importance.quantile(.05), "q95": g.importance.quantile(.95),
                                                "baseline_loss": g.baseline_loss.first(), "n_channels": g.n_channels.first(),
                                                "n_test": g.n_test.first()}).reset_index()
                        for item in summary.itertuples(index=False):
                            ch = regions[item.region]
                            idx = [cols.index(c) for c in ch]
                            w = reps[rep][1][idx]
                            importance.append({"fold_id": fold_id, "model_id": model_id, "family": family,
                                               "representation": rep, "target": target, "season": season,
                                               "region_id": item.region,
                                               "region_label": item.region,
                                               "low": None if item.region == "whole_spectrum" else float(w.min()),
                                               "high": None if item.region == "whole_spectrum" else float(w.max()),
                                               "n_channels": int(item.n_channels), "n_test": int(item.n_test),
                                               "delta_mean": item.delta_mean, "delta_sd": item.delta_sd,
                                               "q05": item.q05, "q95": item.q95, "repeats": repeats,
                                               "baseline_loss": item.baseline_loss, "metric": "rmse"})
            log(f"  residual models done for {fold_id} · {rep}")
    return pd.DataFrame(scores), pd.DataFrame(preds), pd.DataFrame(importance)


def lot_controls(cohort, reps, main_predictions, jobs):
    """Season classification with lot held fixed, plus lot separability inside one season."""
    short = cohort.season.str.split().str[0]
    lot, cyc = cohort.lot_group.astype(str), cohort.cycle_end.to_numpy()
    tests = [("forward_2024", "Trained 2022–23 (lots 248+251) → tested 2023–24 (lot 251)", cyc == 2023, cyc == 2024),
             ("rev_2023_lot248", "Trained 2023–24 (lot 251) → 2022–23 lot-248 filters", cyc == 2024, (cyc == 2023) & lot.eq("248").to_numpy()),
             ("rev_2023_lot251", "Trained 2023–24 (lot 251) → 2022–23 lot-251 filters", cyc == 2024, (cyc == 2023) & lot.eq("251").to_numpy()),
             ("fwd_2025_lot251", "Trained 2023–24 (lot 251) → 2024–25 (lot 251)", cyc == 2024, cyc == 2025),
             ("fwd_2026_lot253", "Trained 2023–24 (lot 251) → 2025–26 Bega (lot 253)", cyc == 2024, (cyc == 2026) & lot.eq("253").to_numpy())]
    rows = []
    for family in ("rf", "linear"):
        for rep in ("raw_common", "spline_common", "removed_common"):
            X, cols = make_feature_frame(reps[rep][0], physical_ids=cohort.id.to_numpy())
            for test_id, label, tr, te in tests:
                if not te.any():
                    continue
                est = build_model(task="classification", model=family, spectral_columns=cols, n_jobs=jobs)
                est.fit(X[tr], cohort.season[tr])
                yp, yt = est.predict(X[te]), cohort.season[te].to_numpy()
                present = [s for s in SEASONS if (yt == s).any()]
                rec = recall_score(yt, yp, labels=present, average=None, zero_division=0)
                rows.append({"test_id": test_id, "test_label": label, "family": family, "representation": rep,
                             "model_id": f"{family}_{rep}", "n_train": int(tr.sum()), "n_test": int(te.sum()),
                             "lots_train": "+".join(sorted(set(lot[tr]))), "lots_test": "+".join(sorted(set(lot[te]))),
                             "accuracy": float(np.mean(yp == yt)),
                             "balanced_accuracy": float(balanced_accuracy_score(yt, yp)) if len(present) > 1 else None,
                             **{f"recall__{s.split()[0]}": float(r) for s, r in zip(present, rec)},
                             **{f"n__{s.split()[0]}": int((yt == s).sum()) for s in present}})
    sep = []
    belg = np.flatnonzero((cyc == 2023) & short.eq("Belg").to_numpy())
    for family in ("rf", "linear"):
        for rep in ("raw_common", "spline_common", "removed_common"):
            X, cols = make_feature_frame(reps[rep][0][belg])
            yp = cross_val_predict(build_model(task="classification", model=family, spectral_columns=cols, n_jobs=jobs),
                                   X, lot.iloc[belg].to_numpy(), cv=LeaveOneOut())
            sep.append({"family": family, "representation": rep, "model_id": f"{family}_{rep}",
                        "n": len(belg), "lots": "248 vs 251",
                        "date_range_248": [str(cohort.date[belg][lot.iloc[belg].eq("248").to_numpy()].min().date()),
                                           str(cohort.date[belg][lot.iloc[belg].eq("248").to_numpy()].max().date())],
                        "date_range_251": [str(cohort.date[belg][lot.iloc[belg].eq("251").to_numpy()].min().date()),
                                           str(cohort.date[belg][lot.iloc[belg].eq("251").to_numpy()].max().date())],
                        "balanced_accuracy": float(balanced_accuracy_score(lot.iloc[belg], yp))})
    # The saved forward-2024 predictions, per season, for the saved metadata (loading + lot) classifiers.
    p = main_predictions.loc[main_predictions.task.eq("season") & main_predictions.calendar.eq("belg_feb")
                             & main_predictions.cohort.eq("direct232") & main_predictions.fold_id.eq("forward_2024")]
    saved = (p.assign(ok=p.actual.eq(p.predicted)).groupby(["model_id", "actual"]).ok.mean()
             .rename("recall").reset_index())
    return pd.DataFrame(rows), pd.DataFrame(sep), saved


def run(args):
    started, started_at = time.monotonic(), datetime.now(timezone.utc).isoformat()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    inputs = {p: sha256(p) for p in [args.main_dir / "physical_filter_cohort.csv", args.main_dir / "model_scores.csv",
                                     args.main_dir / "heldout_predictions.csv", args.lots, Path(__file__),
                                     ROOT / "research/ftir_hips_chem/scripts/seasonal_ml_checks.py",
                                     ROOT / "research/ftir_hips_chem/scripts/seasonal_spectral_ml.py",
                                     ROOT / "research/ftir_hips_chem/workflows/run_seasonal_spectral_ml.py"]}
    cohort = pd.read_csv(args.main_dir / "physical_filter_cohort.csv", parse_dates=["date"])
    cohort = cohort.sort_values(["date", "id"]).reset_index(drop=True)
    lots = spartan_lot_map(pd.read_csv(args.lots, dtype=str, encoding="utf-8-sig"))
    cohort = fill_lot_groups(cohort, lots)
    main_scores = pd.read_csv(args.main_dir / "model_scores.csv")
    main_predictions = pd.read_csv(args.main_dir / "heldout_predictions.csv", dtype={"actual": str, "predicted": str})
    wn, raw, cor, shared, spectra_source = load_spectra(cohort, args.arrays, args.spectra)
    reps = representations(wn, raw, cor, shared)
    log(f"{len(cohort)} filters · lots {cohort.lot_group.value_counts().to_dict()} · spectra from {spectra_source['kind']}")

    counts = lot_season_counts(cohort)
    ec_scores, ec_preds = ec_line_scores(cohort, main_scores)
    log("EC line scored on every saved disagreement fold")
    res_scores, res_preds, res_imp = residual_models(cohort, reps, main_scores, args.repeats, args.jobs)
    lot_ctl, lot_sep, saved_recall = lot_controls(cohort, reps, main_predictions, args.jobs)
    log("lot controls done")

    check = res_scores.dropna(subset=["saved_rmse"]) if "saved_rmse" in res_scores else res_scores.iloc[:0]
    reproduction = {"compared_models": int(len(check)),
                    "max_abs_rmse_difference": float((check.rmse - check.saved_rmse).abs().max()) if len(check) else None,
                    "common_channels": int(reps["spline_common"][0].shape[1])}
    tables = {"cohort_with_lots.csv": cohort, "lot_season_counts.csv": counts, "ec_line_scores.csv": ec_scores,
              "ec_line_predictions.csv": ec_preds, "residual_scores.csv": res_scores,
              "residual_predictions.csv": res_preds, "residual_region_importance.csv": res_imp,
              "lot_controlled_season.csv": lot_ctl, "lot_separability.csv": lot_sep,
              "saved_forward2024_recall.csv": saved_recall}
    for name, frame in tables.items():
        frame.to_csv(out / name, index=False)
    corr = {int(c): float(g.optical_disagreement_ugm3.corr(g.ec_ftir_ugm3)) for c, g in cohort.groupby("cycle_end")}
    sd = (cohort.groupby(["cycle_end", "season"]).optical_disagreement_ugm3.agg(["count", "mean", "std"])
          .reset_index().rename(columns={"count": "n", "std": "sd"}))
    inputs_end = {p: sha256(p) for p in inputs}
    payload = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "lots": {"counts": counts.to_dict("records"),
                 "sources": cohort.lot_source.value_counts().to_dict(),
                 "note": "Lots missing from the native lot map are filled from SPARTAN's filter table (ExternalLotId). "
                         "Recorded lots agree with it wherever both exist."},
        "ec_line": ec_scores.to_dict("records"),
        "disagreement_ec_correlation": corr,
        "disagreement_by_season": sd.to_dict("records"),
        "residual_scores": res_scores.to_dict("records"),
        "residual_importance": res_imp.to_dict("records"),
        "lot_control": lot_ctl.to_dict("records"),
        "lot_separability": lot_sep.to_dict("records"),
        "saved_forward2024_recall": saved_recall.to_dict("records"),
        "labels": {"family": FAMILY_LABEL, "representation": REP_LABEL, "named_regions": list(NAMED_REGIONS)},
        "reproduction": reproduction,
        "limitations": [
            "The EC line is fitted on training filters only and uses no spectral channel. It is a comparator, not a model of EC accuracy.",
            "The residual target is D minus the training-fold EC line. Region importance on it shows spectral information about the "
            "FTIR–HIPS disagreement that a straight line on FTIR EC does not already carry.",
            "Raw-spectrum region tests use RF only: a block shuffle of the raw continuum pushes linear and PLS models off their training "
            "manifold and gives meaningless importances.",
            "Lot-controlled season tests train on 2023–24 and test backwards in time on 2022–23. They ask whether season is recognisable "
            "across years when the lot is unchanged, not whether it can be forecast.",
            "Lot 248 vs 251 inside 2023 Belg is also early vs late Belg (2 Feb–22 Mar vs 29 Mar–29 May); separability there is indicative only.",
            "Held-out sets are 97 and 30 filters; one test year carries most of the weight. Permutation spread is Monte Carlo variation, not a confidence interval.",
        ],
        "provenance": {"started_at": started_at, "elapsed_seconds": time.monotonic() - started,
                       "inputs": {str(Path(p).relative_to(ROOT)) if Path(p).is_relative_to(ROOT) else Path(p).name: h
                                  for p, h in inputs.items()},
                       "inputs_unchanged_during_run": inputs == inputs_end, "spectra": spectra_source,
                       "settings": {"seed": SEED, "repeats": args.repeats, "folds": "first analysis's forward cycles",
                                    "importance_models": [f"{f}_{r}" for f, r in IMPORTANCE_MODELS]},
                       "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                                    "sklearn": sklearn.__version__}},
    }
    (out / "checks.json").write_text(json.dumps(clean(payload), ensure_ascii=False, allow_nan=False,
                                                separators=(",", ":")) + "\n")
    log(f"wrote {out.relative_to(ROOT)} · reproduction {reproduction}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    bundle = os.environ.get("AETHMODULAR_SPARTAN_HIPS_BUNDLE")
    parser.add_argument("--lots", type=Path, default=Path(bundle) / "spartan_Filters.csv" if bundle else None,
                        required=not bundle, help="SPARTAN spartan_Filters.csv (ExternalLotId)")
    parser.add_argument("--main-dir", type=Path, default=MAIN)
    parser.add_argument("--arrays", type=Path, default=TABLES / "baseline_existing_data_20261103/native_method_arrays.npz")
    parser.add_argument("--spectra", type=Path, default=SPECTRA)
    parser.add_argument("--output", type=Path, default=TABLES / "seasonal_ml_checks_20261103")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
