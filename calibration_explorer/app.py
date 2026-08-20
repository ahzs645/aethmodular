"""Calibration Iteration Explorer — a standalone local Flask app at the repo root.

The July-17 meeting left a grid of options to trial: cohort x selection cutoff x
baseline-corrected-or-not (both for the *selection* and for the *calibration*) x CV
scheme x component count, read out as the Addis crossplot under OLS/Deming at MAC 6/10.
Rather than one notebook per grid cell, this app drives the locked phase-3 machinery
(`research/ftir_ec_phase3/scripts/`, the committed cohort tables, the AIRSpec caches)
from an interactive page: pick a configuration, get its CV curve, click a k, read the
crossplot — then pin runs and compare intercepts across configurations.

This is a sibling of `research/spartan_ec_2026_06_16/recreation_app` (which recreates
the AQRC Shiny tool itself); this app is independent of it and iterates the phase-3
setup matrix instead.

Everything heavy is cached in ./cache/ keyed by the exact configuration (including the
resolved cohort membership), so only genuinely new curves are computed.

Run:  python calibration_explorer/app.py   →  http://127.0.0.1:5058
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PHASE3_DIR = REPO / "research" / "ftir_ec_phase3"
sys.path.insert(0, str(PHASE3_DIR / "scripts"))

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression

# All calibration math is the SAME code the notebooks run — nothing is re-derived here:
# the two CV protocols and the Deming λ convention come from
# research/ftir_ec_phase3/scripts/calibration_modes.py, and the estimators/selection
# metric from research/ftir_hips_chem/scripts/pls_transfer.py.
from phase3_common import (  # noqa: E402  (also puts phase-2 scripts on sys.path)
    PATHS, PHASE2_TABLES, load_addis_evaluation, load_pool_metadata, load_tor_loadings,
)
from calibration_modes import (  # noqa: E402
    DEMING_LAMBDA_MAC10, MAX_COMPONENTS, deming_lambda,
    protocol_cv_curve, protocol_select_k, protocol_train_mask,
)
from pls_transfer import (  # noqa: E402
    band_feature_distance, deming_regression, mahalanobis_distance_squared,
    pairwise_score_distance_squared, project_scores, regression_metrics,
    score_metric, select_components_cv, vip_scores,
)
from config import season_for_month  # noqa: E402  (phase-2 scripts)

CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(exist_ok=True)
# Evaluation targets beyond the built-in Addis set: drop a folder here containing
# spectra.csv (id column + the IMPROVE wavenumber columns) and reference.csv
# (id column + Fabs or EC_ugm3, Volume_m3, optional Group) — see README.
TARGETS_DIR = HERE / "targets"
TARGETS_DIR.mkdir(exist_ok=True)

COHORTS = {
    "pool": "Entire IMPROVE network (no selection)",
    "smoke": "Biomass-smoke (906)",
    "eth_shaped": "Ethiopia-shaped smoke",
    "analogs": "Spectral analogs",
    "ocec": "Lowest-OC/EC",
}

# Savitzky-Golay second derivative — identical parameters to ftir_20's comparison.
SAVGOL = dict(window_length=11, polyorder=2, deriv=2)
SPECTRA_LABEL = {"raw": "as-measured", "airspec": "AIRSpec df1=6 baselined",
                 "deriv2": "SG 2nd derivative of raw"}
DEFAULT_CUTOFF = {"eth_shaped": 300, "analogs": 500, "ocec": 800}
RANKED_COHORTS = set(DEFAULT_CUTOFF)

app = Flask(__name__, static_folder=str(HERE / "static"))

STATE = {"ready": False, "error": None, "message": "starting…", "checks": []}
D: dict = {}
COMPUTE_LOCK = threading.Lock()


# ----------------------------------------------------------------------------- #
# data loading (once, in a background thread)
# ----------------------------------------------------------------------------- #
def _read_pool_spectra(path, wcols, engine=None):
    """The 725 MB pool-spectra read, via polars' multithreaded reader when
    installed (~6x faster than the pandas read in the same-shape trial —
    research/package_trials/io/). polars parses correctly rounded, bit-identical
    to pandas float_precision='round_trip'; against the historical default
    pandas read it differed on 1 cell in 35.4M, by 1 float32 ulp on an ~1e-5
    value — far below every reported precision. Set CALIB_EXPLORER_CSV=pandas
    to force the fallback (the historical read, byte-for-byte).

    Returns ``(frame, engine)`` so the loader can report which path ran.
    ``engine`` forces one path explicitly (the /api/benchmark_pool_read
    diagnostic); None resolves it from the environment.
    """
    if engine is None:
        engine = os.environ.get("CALIB_EXPLORER_CSV", "polars")
    if engine == "polars":
        try:
            import polars as pl
            raw = pl.read_csv(path, columns=["AnalysisId"] + wcols,
                              schema_overrides={c: pl.Float64 for c in wcols})
            frame = pd.DataFrame(
                raw.select(wcols).to_numpy().astype(np.float32), columns=wcols)
            frame.insert(0, "AnalysisId", raw["AnalysisId"].to_numpy())
            return frame, "polars"
        except ImportError:
            pass
        except Exception as exc:   # never let the accelerator break the load
            import warnings
            warnings.warn(f"polars pool read failed ({exc!r}); using pandas")
    return pd.read_csv(path, usecols=["AnalysisId"] + wcols,
                       dtype={c: np.float32 for c in wcols}), "pandas"


def _load_all():
    try:
        STATE["message"] = "loading Addis evaluation set…"
        etad_eval, X_addis_raw, wn = load_addis_evaluation(season_for_month)
        wcols = list(etad_eval.attrs["wcols"])
        D["etad_eval"] = etad_eval
        D["X_addis_raw"] = X_addis_raw
        D["wn_raw"] = wn
        D["wcols"] = wcols
        D["fabs"] = etad_eval["Fabs"].to_numpy(float)
        D["volume"] = etad_eval["SampleVolume_m3"].to_numpy(float)
        D["fixed_mask"] = etad_eval["EC_deployed_ugm3"].notna().to_numpy()
        D["season"] = etad_eval["season"].fillna("unknown").astype(str).tolist()

        STATE["message"] = "loading the 13k-pool spectra (biggest file)…"
        pool_raw, csv_engine = _read_pool_spectra(
            PATHS.ftir_dir / "local_db/spectra_248_251.csv", wcols)
        pool_raw = pool_raw[~pool_raw["AnalysisId"].duplicated()].set_index("AnalysisId")
        pool_raw.index = pool_raw.index.astype(int)
        D["pool_raw"] = pool_raw

        STATE["message"] = "matching pool metadata to TOR…"
        meta = (load_pool_metadata()
                .merge(load_tor_loadings(), on=["Site", "date"], how="left",
                       validate="many_to_one"))
        # Training frame — identical construction to ftir_21.
        pool = (meta.query("TOR_EC_loading_ug > 0").drop_duplicates("FilterId").copy())
        pool["AnalysisId"] = pool["AnalysisId"].astype(int)
        pool = pool[pool["AnalysisId"].isin(pool_raw.index)].drop_duplicates("AnalysisId")
        pool = pool.set_index("AnalysisId")[
            ["Site", "date", "TOR_EC_loading_ug", "OC_EC_ratio",
             "TOR_EC_ugm3", "TOR_OC_ugm3"]]
        D["pool"] = pool

        # Lowest-OC/EC ranking — identical eligibility + ordering to ftir_11.
        eligible = (meta["TOR_EC_loading_ug"].gt(0) & meta["TOR_EC_ugm3"].gt(0)
                    & meta["TOR_OC_ugm3"].gt(0) & meta["OC_EC_ratio"].notna())
        ocec_frame = (meta[eligible].sort_values("OC_EC_ratio")
                      .drop_duplicates("FilterId"))
        D["ocec_ranked"] = ocec_frame["AnalysisId"].astype(int).to_numpy()
        D["ocec_metric"] = ocec_frame["OC_EC_ratio"].to_numpy(float)

        STATE["message"] = "loading cohort tables…"
        smoke = pd.read_csv(
            PHASE2_TABLES / "pls_calibration_phase2/smoke_cohort_spectral_selection.csv")
        D["smoke_ids"] = smoke["AnalysisId"].to_numpy(int)
        eth_sorted = smoke.sort_values("Addis_band_feature_distance")
        D["eth_ranked"] = eth_sorted["AnalysisId"].to_numpy(int)
        D["eth_metric"] = eth_sorted["Addis_band_feature_distance"].to_numpy(float)
        eth_locked = set(
            smoke.loc[smoke["selected_Ethiopia_shaped_smoke"], "AnalysisId"].astype(int))

        similarity = load_pool_metadata()
        analog_locked = set(pd.read_csv(
            PHASE2_TABLES / "pls_calibration_phase2/locked_analog_train_test_split.csv")
            ["AnalysisId"].astype(int))
        sim = similarity.drop_duplicates("AnalysisId").copy()
        sim["AnalysisId"] = sim["AnalysisId"].astype(int)
        asc = sim.sort_values("analog_rank_score")
        n_lock = len(analog_locked)
        overlap_asc = len(set(asc["AnalysisId"].to_numpy(int)[:n_lock]) & analog_locked)
        overlap_desc = len(set(asc["AnalysisId"].to_numpy(int)[::-1][:n_lock]) & analog_locked)
        if overlap_desc > overlap_asc:
            asc = asc.iloc[::-1]
        D["analog_ranked"] = asc["AnalysisId"].to_numpy(int)
        D["analog_metric"] = asc["analog_rank_score"].to_numpy(float)

        STATE["message"] = "loading filter-lot catalog…"
        catalog = pd.read_csv(PATHS.ftir_dir / "local_db/tables/ftir_catalog.csv",
                              usecols=["AnalysisId", "LotNumber"])
        catalog["AnalysisId"] = catalog["AnalysisId"].astype(int)
        catalog = catalog[catalog["AnalysisId"].isin(pool_raw.index)]
        D["lot"] = catalog.drop_duplicates("AnalysisId").set_index("AnalysisId")[
            "LotNumber"].astype(str).to_dict()
        D["lots_available"] = sorted(set(D["lot"].values()))

        STATE["message"] = "loading AIRSpec-corrected caches…"
        corr = np.load(PHASE3_DIR / "output/corrected/improve_pool_corrected_df6.npz",
                       allow_pickle=True)
        D["corr_pool"] = corr["corrected"]
        D["corr_wn"] = corr["wn"].astype(float)
        D["corr_row"] = {int(a): i for i, a in enumerate(corr["analysis_id"].astype(int))}
        etad_npz = np.load(PHASE3_DIR / "output/corrected/etad_corrected_df6.npz",
                           allow_pickle=True)
        etad_corr = pd.DataFrame(etad_npz["corrected"].astype(float))
        etad_corr["MediaId"] = etad_npz["media_id"].astype(int)
        D["X_addis_corr"] = (etad_corr.groupby("MediaId").mean()
                             .loc[etad_eval["MediaId"].astype(int)].to_numpy(float))

        # Per-filter lot of the evaluation set, from the HIPS LotId (Ann,
        # 2026-08-19: judge a lot-251 calibration on lot-251 filters — most
        # ETAD filters are 251, ~15% are 248, a few 253).
        STATE["message"] = "matching evaluation filters to HIPS lots…"
        hips_lots = (pd.read_csv(PATHS.spartan_hips_primary, encoding="cp1252",
                                 usecols=["Site", "FilterId", "LotId"])
                     .query("Site == 'ETAD'").drop_duplicates("FilterId")
                     .dropna(subset=["LotId"]))
        lot_by_filter = (hips_lots.set_index("FilterId")["LotId"]
                         .astype(int).astype(str).to_dict())
        eval_lots = [lot_by_filter.get(f, "?")
                     for f in etad_eval["ExternalFilterId"]]
        D["eval_lots"] = {str(k): int(v)
                          for k, v in pd.Series(eval_lots).value_counts().items()}

        # The built-in evaluation target. Custom targets load on demand from
        # TARGETS_DIR and join this registry; everything downstream reads it.
        dates = pd.to_datetime(etad_eval["SamplingStartDate"], errors="coerce")
        deployed = etad_eval["EC_deployed_ugm3"]
        D["targets"] = {"addis": {
            "label": "Addis (ETAD) — built-in",
            "ref_kind": "fabs",          # x = ref/MAC; "ec" targets use x = ref
            "ref": D["fabs"],
            "volume": D["volume"],
            "X_raw": D["X_addis_raw"],
            "X_corr": D["X_addis_corr"],
            "groups": D["season"],
            "fixed_mask": D["fixed_mask"],
            "dates": [d.date().isoformat() if pd.notna(d) else None for d in dates],
            "deployed": [round(float(v), 4) if pd.notna(v) else None for v in deployed],
            "lots": eval_lots,
        }}

        # Startup provenance checks against the locked cohorts (reported, not fatal).
        # Top-N is counted over TOR-eligible filters — the same rule the app's
        # cohort resolution uses (see _ranking).
        def top_eligible(ranked, n):
            r = np.asarray(ranked, dtype=int)
            return set(int(i) for i in r[np.isin(r, pool.index.to_numpy())][:n])

        checks = [{"name": f"pool spectra loaded via {csv_engine}", "ok": True,
                   "detail": "polars is the fast path when installed; "
                             "CALIB_EXPLORER_CSV=pandas forces the pandas read"}]
        eth300 = top_eligible(D["eth_ranked"], 300)
        checks.append({"name": "Ethiopia-shaped top-300 == locked selection",
                       "ok": eth300 == eth_locked,
                       "detail": f"{len(eth300 & eth_locked)}/300 overlap"})
        top_lock = top_eligible(D["analog_ranked"], n_lock)
        checks.append({"name": f"analog top-{n_lock} == locked analog cohort",
                       "ok": top_lock == analog_locked,
                       "detail": f"{len(top_lock & analog_locked)}/{n_lock} overlap"})
        committed_800 = set(pd.read_csv(
            PHASE3_DIR / "output/tables/ftir11/lowest_ocec_800_cohort.csv")
            ["AnalysisId"].astype(int))
        got_800 = top_eligible(D["ocec_ranked"], 800)
        checks.append({"name": "lowest-OC/EC top-800 == committed ftir_11 cohort",
                       "ok": got_800 == committed_800,
                       "detail": f"{len(got_800 & committed_800)}/800 overlap"})
        STATE["checks"] = checks
        STATE["ready"] = True
        STATE["message"] = "ready"
    except Exception as exc:  # surfaced via /api/status
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        STATE["message"] = "load failed"
        raise


threading.Thread(target=_load_all, daemon=True).start()


# ----------------------------------------------------------------------------- #
# evaluation targets
# ----------------------------------------------------------------------------- #
def list_targets():
    out = {"addis": "Addis (ETAD) — built-in"}
    for p in sorted(TARGETS_DIR.iterdir()) if TARGETS_DIR.exists() else []:
        if p.is_dir() and (p / "spectra.csv").exists():
            out[p.name] = (D["targets"][p.name]["label"]
                           if p.name in D.get("targets", {}) else f"{p.name} (custom)")
    return out


def get_target(name):
    name = name or "addis"
    if name in D["targets"]:
        return D["targets"][name]
    d = TARGETS_DIR / name
    if not (d / "spectra.csv").exists() or not (d / "reference.csv").exists():
        raise ValueError(f"target {name!r}: needs spectra.csv and reference.csv in "
                         f"calibration_explorer/targets/{name}/")
    sp = pd.read_csv(d / "spectra.csv")
    missing = [c for c in D["wcols"] if c not in sp.columns]
    if missing:
        raise ValueError(
            f"target {name!r}: spectra.csv must carry the IMPROVE wavenumber grid "
            f"({len(D['wcols'])} columns, like the local_db export); "
            f"{len(missing)} columns missing (e.g. {missing[:3]})")
    idc = sp.columns[0]
    ref = pd.read_csv(d / "reference.csv")
    ref = ref.rename(columns={ref.columns[0]: idc})
    m = sp.merge(ref, on=idc, how="inner", validate="one_to_one")
    m = m[m[D["wcols"]].notna().all(axis=1)]
    if "Fabs" in m.columns:
        ref_kind, ref_values = "fabs", m["Fabs"].to_numpy(float)
    elif "EC_ugm3" in m.columns:
        ref_kind, ref_values = "ec", m["EC_ugm3"].to_numpy(float)
    else:
        raise ValueError(f"target {name!r}: reference.csv needs a Fabs (Mm⁻¹) or "
                         "EC_ugm3 column")
    if "Volume_m3" not in m.columns or not (m["Volume_m3"] > 0).all():
        raise ValueError(f"target {name!r}: reference.csv needs a positive Volume_m3 "
                         "column (predictions are µg/filter ÷ volume)")
    keep = np.isfinite(ref_values)
    m = m[keep]
    if "Date" in m.columns:
        parsed = pd.to_datetime(m["Date"], errors="coerce")
        dates = [d.date().isoformat() if pd.notna(d) else None for d in parsed]
    else:
        dates = [None] * len(m)
    t = {"label": f"{name} (custom, n={len(m)})", "ref_kind": ref_kind,
         "ref": ref_values[keep], "volume": m["Volume_m3"].to_numpy(float),
         "X_raw": m[D["wcols"]].to_numpy(float), "X_corr": None,
         "groups": (m["Group"].astype(str).tolist() if "Group" in m.columns
                    else ["all"] * len(m)),
         "fixed_mask": None, "dates": dates, "deployed": None}
    D["targets"][name] = t
    return t


def _target_X(t, spectra):
    if spectra == "airspec":
        if t["X_corr"] is None:
            raise ValueError("this target has no AIRSpec-corrected spectra — "
                             "calibrate on raw or SG 2nd derivative instead")
        return t["X_corr"]
    if spectra == "deriv2":
        if "X_deriv2" not in t:
            t["X_deriv2"] = savgol_filter(np.asarray(t["X_raw"], float),
                                          axis=1, **SAVGOL)
        return t["X_deriv2"]
    return t["X_raw"]


# ----------------------------------------------------------------------------- #
# corrected-space Ethiopia-shaped selection (meeting item 1), lazily computed
# ----------------------------------------------------------------------------- #
def eth_corrected_ranking():
    if "eth_corr_ranked" not in D:
        ids = np.array([i for i in D["smoke_ids"] if int(i) in D["corr_row"]], int)
        X = D["corr_pool"][[D["corr_row"][int(i)] for i in ids]].astype(float)
        # the exact committed selection metric, in corrected space
        distance = band_feature_distance(X, D["corr_wn"], D["X_addis_corr"])
        order = np.argsort(distance)
        D["eth_corr_ranked"] = ids[order]
        D["eth_corr_metric"] = distance[order]
        D["eth_corr_coverage"] = f"{len(ids)}/{len(D['smoke_ids'])} smoke filters in corrected cache"
    return D["eth_corr_ranked"], D["eth_corr_metric"]


def analog_corrected_ranking():
    """The ftir_09 analog machinery re-run on AIRSpec-corrected spectra.

    Same recipe as the committed raw-space run (IMPROVE-HIPS PLS model → score-space
    nearest-Addis distance + VIP-weighted spectral RMSE → mean pct-rank), with two
    deliberate substitutions: the model, pool and Addis all use the corrected caches,
    and the raw recipe's 3900-4000 cm⁻¹ offset correction is dropped (the corrected
    spectra are already baselined). Cached to disk — first call takes ~20 s.
    """
    if "analog_corr_ranked" in D:
        return D["analog_corr_ranked"], D["analog_corr_metric"]
    cache = CACHE_DIR / "analog_corrected_ranking.npz"
    if cache.exists():
        z = np.load(cache)
        D["analog_corr_ranked"] = z["ids"].astype(int)
        D["analog_corr_metric"] = z["metric"].astype(float)
        D["analog_corr_coverage"] = str(z["coverage"])
        return D["analog_corr_ranked"], D["analog_corr_metric"]

    # 1. The IMPROVE-HIPS training set, exactly as in ftir_09 — apps EC set joined
    #    to results_hips → HIPS tau — but with corrected spectra.
    arrays = np.load(PATHS.ftir_dir / "apps/apps_data.npz", allow_pickle=True)
    rows = pd.DataFrame({"row": np.arange(len(arrays["EC_id"])),
                         "AnalysisId": arrays["EC_id"].astype(int),
                         "FilterId": arrays["EC_fid"].astype(int),
                         "Site": arrays["EC_site"].astype(str)})
    hips = pd.read_csv(PATHS.ftir_dir / "local_db/tables/results_hips.csv",
                       usecols=["MatchedFilterId", "Parameter", "Value",
                                "AverageFlowRate", "ElapsedTime", "SampleDepositArea"])
    hips = (hips[hips["Parameter"].str.casefold() == "fabs"]
            .drop_duplicates("MatchedFilterId"))
    join = rows.merge(hips, left_on="FilterId", right_on="MatchedFilterId",
                      how="left", validate="one_to_one")
    join["SampleVolume_m3"] = join["AverageFlowRate"] / 1000.0 * join["ElapsedTime"]
    join["HIPS_tau"] = (join["Value"] * join["SampleVolume_m3"]
                        / (100.0 * join["SampleDepositArea"]))
    eligible = (join["Value"].notna() & join["SampleVolume_m3"].gt(0)
                & join["SampleDepositArea"].gt(0) & join["HIPS_tau"].notna()
                & join["AnalysisId"].map(lambda a: int(a) in D["corr_row"]))
    train_ids = join.loc[eligible, "AnalysisId"].astype(int).to_numpy()
    Xc = D["corr_pool"][[D["corr_row"][int(a)] for a in train_ids]].astype(float)
    y_tau = join.loc[eligible, "HIPS_tau"].to_numpy(float)
    sites = join.loc[eligible, "Site"].to_numpy()

    k, _ = select_components_cv(Xc, y_tau, range(1, 26), groups=sites,
                                n_splits=5, random_state=42)
    model = PLSRegression(n_components=k, scale=False).fit(Xc, y_tau)
    vip = vip_scores(model)
    vip_weights = vip ** 2 / np.sum(vip ** 2)
    _, inverse_ss = score_metric(model)
    etad_scores = project_scores(model, D["X_addis_corr"])
    target = np.nanmedian(D["X_addis_corr"], axis=0)

    # 2. Screen the whole corrected pool.
    pool_scores = project_scores(model, D["corr_pool"].astype(float))
    metric_root = np.linalg.cholesky(inverse_ss)
    from scipy.spatial.distance import cdist
    nearest = cdist(pool_scores @ metric_root, etad_scores @ metric_root,
                    metric="sqeuclidean").min(axis=1)
    vip_rmse = np.sqrt(((D["corr_pool"].astype(float) - target) ** 2) @ vip_weights)
    rank = (pd.Series(nearest).rank(pct=True)
            + pd.Series(vip_rmse).rank(pct=True)).to_numpy() / 2

    all_ids = np.array(sorted(D["corr_row"], key=D["corr_row"].get), int)
    order = np.argsort(rank)
    coverage = (f"corrected-space analog model: k={k}, n_train={len(train_ids)} "
                f"(HIPS-tau target), pool screened={len(all_ids)}")
    D["analog_corr_ranked"] = all_ids[order]
    D["analog_corr_metric"] = rank[order]
    D["analog_corr_coverage"] = coverage
    np.savez_compressed(cache, ids=D["analog_corr_ranked"],
                        metric=D["analog_corr_metric"], coverage=coverage)
    return D["analog_corr_ranked"], D["analog_corr_metric"]


# ----------------------------------------------------------------------------- #
# cohort resolution
# ----------------------------------------------------------------------------- #
def _ranking(cohort, selection_space):
    """Ranked candidates for a cohort, restricted to the TOR-eligible pool.

    Rankings can list ids with no usable TOR row; the cutoff must count eligible
    filters, or "top 500" analogs resolves to 477 fitted filters (and the locked
    phase-2 analog cohort — which IS the top 500 eligible — never reproduces).
    """
    if cohort == "eth_shaped" and selection_space == "airspec":
        ranked, metric = eth_corrected_ranking()
        label = "band-feature distance to Addis (AIRSpec-corrected spectra)"
    elif cohort == "analogs" and selection_space == "airspec":
        ranked, metric = analog_corrected_ranking()
        label = "analog rank score (AIRSpec-corrected spectra)"
    else:
        ranked, metric, label = {
            "eth_shaped": (D["eth_ranked"], D["eth_metric"],
                           "band-feature distance to Addis (raw spectra)"),
            "analogs": (D["analog_ranked"], D["analog_metric"], "analog rank score"),
            "ocec": (D["ocec_ranked"], D["ocec_metric"], "TOR OC/EC ratio")}[cohort]
    ranked = np.asarray(ranked, dtype=int)
    keep = np.isin(ranked, D["pool"].index.to_numpy())
    return ranked[keep], np.asarray(metric, dtype=float)[keep], label


def resolve_cohort(cohort, cutoff, selection_space, spectra, lot="all"):
    pool_index = D["pool"].index
    if cohort == "pool":
        ids, label = pool_index.to_numpy(), COHORTS["pool"]
    elif cohort == "smoke":
        ids, label = D["smoke_ids"], COHORTS["smoke"]
    else:
        n = int(cutoff or DEFAULT_CUTOFF[cohort])
        ranked, _, _ = _ranking(cohort, selection_space)
        suffix = ", corrected-space selection" if (
            cohort in ("eth_shaped", "analogs") and selection_space == "airspec") else ""
        ids, label = ranked[:n], f"{COHORTS[cohort]} (top {n}{suffix})"
    keep = [i for i in dict.fromkeys(int(v) for v in ids)
            if i in pool_index and (spectra != "airspec" or i in D["corr_row"])
            and (lot in (None, "all") or D["lot"].get(i) == str(lot))]
    if lot not in (None, "all"):
        label += f" · lot {lot}"
    return np.array(keep, dtype=int), label


# ----------------------------------------------------------------------------- #
# crossplot readout (estimators are the shared pls_transfer / calibration_modes ones)
# ----------------------------------------------------------------------------- #
def crossplot_metrics(t, pred_ugm3):
    """OLS + Deming rows for each MAC (Fabs targets) and evaluation set the
    target supports. EC-reference targets get a single MAC=None row per set,
    with Deming at λ=1 (no HIPS-uncertainty λ* applies)."""
    rows = []
    masks = []
    if t.get("fixed_mask") is not None:
        masks.append(("fixed", t["fixed_mask"]))
    masks.append(("all", np.ones(len(pred_ugm3), bool)))
    macs = (10.0, 6.0) if t["ref_kind"] == "fabs" else (None,)
    for eval_set, mask in masks:
        for mac in macs:
            x = t["ref"][mask] / mac if mac else t["ref"][mask]
            y = pred_ugm3[mask]
            ols = regression_metrics(x, y)
            dm = deming_regression(x, y, deming_lambda(mac) if mac else 1.0)
            rows.append({
                "evaluation_set": eval_set, "MAC": mac, "n": int(np.isfinite(x * y).sum()),
                "ols_slope": round(float(ols["slope"]), 4),
                "ols_intercept": round(float(ols["intercept"]), 4),
                "R2": round(float(ols["R2"]), 4), "RMSE": round(float(ols["RMSE"]), 4),
                "deming_slope": round(float(dm["slope"]), 4),
                "deming_intercept": round(float(dm["intercept"]), 4),
            })
    return rows


# ----------------------------------------------------------------------------- #
# the calibration run (curve cached separately from the k-specific fit)
# ----------------------------------------------------------------------------- #
def _cache_key(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode())
        h.update(b"|")
    return h.hexdigest()[:16]


def _cohort_arrays(cohort, cutoff, selection_space, spectra, lot="all",
                   target="addis"):
    ids, label = resolve_cohort(cohort, cutoff, selection_space, spectra, lot)
    if len(ids) < 30:
        raise ValueError(f"cohort resolves to only {len(ids)} filters")
    y = D["pool"].loc[ids, "TOR_EC_loading_ug"].to_numpy(float)
    sites = D["pool"].loc[ids, "Site"].to_numpy()
    if spectra == "airspec":
        X = D["corr_pool"][[D["corr_row"][int(i)] for i in ids]].astype(float)
    elif spectra == "deriv2":
        X = savgol_filter(D["pool_raw"].loc[ids, D["wcols"]].to_numpy(float),
                          axis=1, **SAVGOL)
    else:
        X = D["pool_raw"].loc[ids, D["wcols"]].to_numpy(float)
    X_eval = _target_X(get_target(target), spectra)
    return ids, label, X, y, sites, X_eval


def run_config(cohort, cutoff, selection_space, spectra, mode, k_override,
               max_components, lot="all", target="addis", eval_lot="all"):
    t = get_target(target)
    ids, cohort_label, X, y, sites, X_eval = _cohort_arrays(
        cohort, cutoff, selection_space, spectra, lot, target)

    ids_hash = hashlib.sha1(np.sort(ids).tobytes()).hexdigest()[:12]
    curve_key = _cache_key("curve", cohort, selection_space, spectra, mode,
                           max_components, lot, ids_hash)   # target-independent
    curve_path = CACHE_DIR / f"{curve_key}.json"

    train = protocol_train_mask(mode, X, y, sites)

    if curve_path.exists():
        curve = pd.DataFrame(json.loads(curve_path.read_text()))
        curve_cached = True
    else:
        # NOTE: for the `app` protocol, interleaved folds are row-order dependent
        # (ftir_22) — the order here is the cohort's ranking/CSV order, matching the
        # committed runs' convention.
        curve = protocol_cv_curve(mode, X, y, sites, train,
                                  max_components=max_components)
        if "rmse_se" not in curve:
            curve["rmse_se"] = np.nan
        curve_path.write_text(curve.replace({np.nan: None}).to_json(orient="records"))
        curve_cached = False

    auto_k = protocol_select_k(mode, curve)
    k = int(k_override) if k_override else int(auto_k)
    k = max(1, min(k, int(curve["n_components"].max())))

    fit_key = _cache_key("fit", curve_key, k, target)
    fit_path = CACHE_DIR / f"{fit_key}.json"
    if fit_path.exists():
        fit = json.loads(fit_path.read_text())
    else:
        model = PLSRegression(n_components=k, scale=False).fit(X[train], y[train])
        pred = model.predict(np.asarray(X_eval, float)).ravel() / t["volume"]
        heldout = None
        if mode == "site_heldout":
            hm = regression_metrics(y[~train], model.predict(X[~train]).ravel())
            heldout = {m: round(float(hm[m]), 4)
                       for m in ("slope", "intercept", "R2", "RMSE")}
        fit = {"addis_ugm3": [round(float(v), 5) for v in pred],
               "heldout": heldout,
               "n_train": int(train.sum()),
               "n_train_sites": int(pd.Series(sites[train]).nunique())}
        fit_path.write_text(json.dumps(fit))

    pred = np.asarray(fit["addis_ugm3"], float)

    # Evaluation-lot mask (Ann, 2026-08-19): report the readout on filters of
    # one lot only. Applied after the cached fit — predictions always cover
    # every filter; only the crossplot/metrics view is restricted — so the
    # curve/fit caches stay lot-agnostic.
    t_view, pred_view = t, pred
    if eval_lot not in (None, "all"):
        lots = t.get("lots")
        if lots is None:
            raise ValueError(f"target '{target}' has no filter-lot information "
                             "(eval lot applies to the built-in Addis target)")
        m = np.array([str(l) == str(eval_lot) for l in lots])
        if m.sum() < 3:
            raise ValueError(f"only {int(m.sum())} evaluation filters on lot {eval_lot}")
        fixed = t["fixed_mask"][m] if t.get("fixed_mask") is not None else None
        if fixed is not None and fixed.sum() < 3:
            fixed = None                      # fixed subset too small on this lot
        t_view = {**t,
                  "ref": t["ref"][m],
                  "groups": [g for g, b in zip(t["groups"], m) if b],
                  "fixed_mask": fixed,
                  "dates": ([d for d, b in zip(t["dates"], m) if b]
                            if t.get("dates") else None),
                  "deployed": ([v for v, b in zip(t["deployed"], m) if b]
                               if t.get("deployed") else None)}
        pred_view = pred[m]

    floor_rows = curve.loc[curve["n_components"] == k, "rmsecv"]
    floor = float(floor_rows.iloc[0]) if len(floor_rows) else float("nan")
    return {
        "cohort_label": cohort_label, "n_cohort": int(len(ids)),
        "n_train": fit["n_train"], "n_train_sites": fit["n_train_sites"],
        "auto_k": int(auto_k), "k": k,
        "rmsecv_floor": round(floor, 4),
        "pct_rmsecv_floor": round(100 * floor / float(y[train].mean()), 2),
        "curve": json.loads(curve.replace({np.nan: None}).to_json(orient="records")),
        "curve_cached": curve_cached,
        "heldout": fit["heldout"],
        "eth_corr_coverage": D.get("eth_corr_coverage"),
        "analog_corr_coverage": D.get("analog_corr_coverage"),
        "target": {"name": target, "label": t["label"], "ref_kind": t["ref_kind"],
                   "has_fixed": t_view.get("fixed_mask") is not None,
                   "eval_lot": eval_lot or "all",
                   "n_eval": int(len(pred_view))},
        "eval": {"ref": [round(float(v), 4) for v in t_view["ref"]],
                 "pred": [round(float(v), 5) for v in pred_view],
                 "group": t_view["groups"],
                 "date": t_view.get("dates"),
                 "deployed": t_view.get("deployed"),
                 "fixed": ([bool(b) for b in t_view["fixed_mask"]]
                           if t_view.get("fixed_mask") is not None
                           else [False] * len(t_view["ref"]))},
        "metrics": crossplot_metrics(t_view, pred_view),
    }


# ----------------------------------------------------------------------------- #
# exhaustive batch optimizer — server-side, so it survives the browser tab and
# runs unattended (locally overnight, or in Colab with the cache copied back).
# Every (configuration, k) row is scored the same way the frontend leaderboard
# scores interactive runs and appended to cache/batch_results.jsonl.
# ----------------------------------------------------------------------------- #
BATCH_RESULTS_PATH = CACHE_DIR / "batch_results.jsonl"
BATCH = {"running": False, "stop": False, "done": 0, "total": 0, "current": "",
         "skipped": 0, "new_rows": 0, "started": None, "finished": None,
         "errors": []}
BATCH_LADDERS = {"eth_shaped": [200, 250, 300, 350, 400],
                 "analogs": [400, 450, 500, 550, 600],
                 "ocec": [600, 700, 800, 900, 1000]}


def _batch_row_key(r):
    return "|".join(str(r.get(f)) for f in
                    ("cohort", "cutoff", "selection_space", "spectra", "mode",
                     "lot", "target", "eval_lot")) + f"|k{r.get('k')}"


def _batch_sweep_ks(auto_k, curve_max):
    # mirror the frontend Sweep-k ladder: rule choice to ~double, capped at 20
    hi = min(max(2 * auto_k, auto_k + 6), 20, curve_max)
    return sorted({round(auto_k + i * (hi - auto_k) / 7) for i in range(8)})


def _batch_row(cfg, out):
    # same shape as the frontend's optRowFromRun, so saved rows merge straight
    # into the leaderboard / Pareto view
    def pick(mac):
        rows = ([m for m in out["metrics"]
                 if m["evaluation_set"] == "fixed" and m["MAC"] == mac]
                or [m for m in out["metrics"] if m["MAC"] == mac]
                or out["metrics"])
        return rows[0]
    f10, f6 = pick(10.0), pick(6.0)
    return {**cfg, "cohort_label": out["cohort_label"], "k": out["k"],
            "auto_k": out["auto_k"],
            "ols_slope": f10["ols_slope"], "ols_intercept": f10["ols_intercept"],
            "deming_slope": f10["deming_slope"],
            "deming_intercept": f10["deming_intercept"],
            "ols_intercept_mac6": f6["ols_intercept"],
            "deming_intercept_mac6": f6["deming_intercept"],
            "R2": f10["R2"],
            "heldout_R2": out["heldout"]["R2"] if out["heldout"] else None}


def _batch_configs(b):
    cohorts = b.get("cohorts") or ["eth_shaped", "analogs", "ocec", "smoke"]
    spectra = b.get("spectra") or ["raw", "airspec", "deriv2"]
    modes = b.get("modes") or ["site_heldout", "app", "app_fmm"]
    lots = b.get("lots") or ["all"]
    target = b.get("target", "addis")
    eval_lot = b.get("eval_lot", "all")
    corrsel = bool(b.get("corrsel"))
    ladder = b.get("cutoff_ladder", True)
    order = {"eth_shaped": 0, "analogs": 1, "ocec": 2, "smoke": 3, "pool": 4}
    cfgs = []
    for co in sorted(cohorts, key=lambda c: order.get(c, 9)):   # cheapest first
        if co in BATCH_LADDERS:
            cutoffs = BATCH_LADDERS[co] if ladder else [DEFAULT_CUTOFF.get(co)]
        else:
            cutoffs = [None]
        sels = (["raw", "airspec"] if corrsel and co in ("eth_shaped", "analogs")
                else ["raw"])
        for cut, sel, sp, mode, lot in itertools.product(
                cutoffs, sels, spectra, modes, lots):
            cfgs.append(dict(cohort=co, cutoff=cut, selection_space=sel,
                             spectra=sp, mode=mode, lot=lot, target=target,
                             eval_lot=eval_lot))
    return cfgs


def _batch_worker(cfgs, sweep):
    seen = set()
    if BATCH_RESULTS_PATH.exists():
        for line in BATCH_RESULTS_PATH.read_text().splitlines():
            try:
                seen.add(_batch_row_key(json.loads(line)))
            except Exception:
                pass
    try:
        with BATCH_RESULTS_PATH.open("a") as f:
            for i, cfg in enumerate(cfgs, 1):
                if BATCH["stop"]:
                    break
                BATCH["done"] = i - 1
                BATCH["current"] = (
                    f"{cfg['cohort']}/{cfg['cutoff'] or '-'} "
                    f"sel={cfg['selection_space']} cal={cfg['spectra']} "
                    f"{cfg['mode']} lot={cfg['lot']}")
                try:
                    # the lock is taken per fit, not per config, so interactive
                    # runs from the page interleave instead of queueing behind
                    # the whole batch
                    with COMPUTE_LOCK:
                        out = run_config(k_override=None,
                                         max_components=MAX_COMPONENTS, **cfg)
                    ks = [out["k"]]
                    if sweep:
                        ks = sorted(set(ks) | set(_batch_sweep_ks(
                            out["auto_k"], int(out["curve"][-1]["n_components"]))))
                    for k in ks:
                        if BATCH["stop"]:
                            break
                        if k == out["k"]:
                            o = out
                        else:
                            with COMPUTE_LOCK:
                                o = run_config(k_override=k,
                                               max_components=MAX_COMPONENTS,
                                               **cfg)
                        row = _batch_row(cfg, o)
                        key = _batch_row_key(row)
                        if key in seen:
                            continue
                        seen.add(key)
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        BATCH["new_rows"] += 1
                except Exception as exc:                       # noqa: BLE001
                    BATCH["skipped"] += 1
                    BATCH["errors"] = (BATCH["errors"]
                                       + [f"{BATCH['current']}: "
                                          f"{type(exc).__name__}: {exc}"])[-8:]
            else:
                BATCH["done"] = len(cfgs)
    finally:
        BATCH["running"] = False
        BATCH["current"] = ""
        BATCH["finished"] = time.time()


@app.route("/api/batch_start", methods=["POST"])
def api_batch_start():
    if not STATE["ready"]:
        return jsonify({"error": "data still loading"}), 503
    if BATCH["running"]:
        return jsonify({"error": "a batch is already running"}), 409
    b = request.get_json(force=True) or {}
    cfgs = _batch_configs(b)
    BATCH.update(running=True, stop=False, done=0, total=len(cfgs), current="",
                 skipped=0, new_rows=0, started=time.time(), finished=None,
                 errors=[])
    threading.Thread(target=_batch_worker,
                     args=(cfgs, bool(b.get("sweep_k", True))),
                     daemon=True).start()
    return jsonify({"started": True, "total": len(cfgs)})


@app.route("/api/batch_stop", methods=["POST"])
def api_batch_stop():
    BATCH["stop"] = True
    return jsonify({"stopping": True})


@app.route("/api/batch_status")
def api_batch_status():
    return jsonify(BATCH)


@app.route("/api/batch_results")
def api_batch_results():
    rows = []
    if BATCH_RESULTS_PATH.exists():
        for line in BATCH_RESULTS_PATH.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return jsonify({"rows": rows})


# ----------------------------------------------------------------------------- #
# routes
# ----------------------------------------------------------------------------- #
def _config_from(b):
    return dict(
        cohort=b.get("cohort", "ocec"),
        cutoff=b.get("cutoff"),
        selection_space=b.get("selection_space", "raw"),
        spectra=b.get("spectra", "raw"),
        mode=b.get("mode", "site_heldout"),
        max_components=int(b.get("max_components", MAX_COMPONENTS)),
        lot=b.get("lot", "all"),
        target=b.get("target", "addis"),
        eval_lot=b.get("eval_lot", "all"),
    )


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status")
def api_status():
    return jsonify({"ready": STATE["ready"], "message": STATE["message"],
                    "error": STATE["error"], "checks": STATE["checks"],
                    "cohorts": COHORTS, "default_cutoff": DEFAULT_CUTOFF,
                    "deming_lambda_mac10": DEMING_LAMBDA_MAC10,
                    "max_components_default": MAX_COMPONENTS,
                    "lots": D.get("lots_available", []),
                    "eval_lots": D.get("eval_lots", {}),
                    "targets": list_targets() if STATE["ready"] else {}})


@app.route("/api/run", methods=["POST"])
def api_run():
    if not STATE["ready"]:
        return jsonify({"error": "data still loading", "message": STATE["message"]}), 503
    b = request.get_json(force=True)
    started = time.time()
    with COMPUTE_LOCK:
        try:
            cfg = _config_from(b)
            out = run_config(k_override=b.get("k"), **cfg)
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
    out["elapsed_s"] = round(time.time() - started, 1)
    out["config"] = {**_config_from(b), "k": b.get("k")}
    return jsonify(out)


@app.route("/api/sweep", methods=["POST"])
def api_sweep():
    """Meeting item: scan k from the rule choice up to ~double / ~20."""
    if not STATE["ready"]:
        return jsonify({"error": "data still loading"}), 503
    b = request.get_json(force=True)
    ks = sorted({int(v) for v in b.get("ks", []) if int(v) >= 1})
    if not ks:
        return jsonify({"error": "pass ks: [..]"}), 400
    rows = []
    with COMPUTE_LOCK:
        try:
            cfg = _config_from(b)
            for k in ks:
                out = run_config(k_override=k, **cfg)

                def pick(mac):
                    return next(
                        (m for m in out["metrics"]
                         if m["evaluation_set"] == "fixed" and m["MAC"] == mac),
                        next((m for m in out["metrics"] if m["MAC"] == mac),
                             out["metrics"][0]))
                fixed10, fixed6 = pick(10), pick(6)
                rows.append({"k": k, "auto_k": out["auto_k"],
                             "rmsecv": out["rmsecv_floor"],
                             "ols_intercept": fixed10["ols_intercept"],
                             "ols_slope": fixed10["ols_slope"],
                             "deming_intercept": fixed10["deming_intercept"],
                             "deming_slope": fixed10["deming_slope"],
                             "ols_intercept_mac6": fixed6["ols_intercept"],
                             "deming_intercept_mac6": fixed6["deming_intercept"],
                             "R2": fixed10["R2"],
                             "heldout_R2": (out["heldout"] or {}).get("R2")})
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
    return jsonify({"rows": rows, "note": "intercept/slope at fixed 190, MAC 10 "
                                          "(*_mac6 fields carry the MAC 6 intercepts)"})


@app.route("/api/benchmark_pool_read", methods=["POST"])
def api_benchmark_pool_read():
    """Diagnostic for the polars fast path: re-read the pool CSV with both
    engines inside the app process (which holds the Drive access) and compare
    timings + values. Two full reads of the 725 MB export, so it takes a
    minute or two; runs under the compute lock so it can't race a calibration.
    """
    if not STATE["ready"]:
        return jsonify({"error": "data still loading"}), 503
    try:
        import polars  # noqa: F401
    except ImportError:
        return jsonify({"error": "polars is not installed in this interpreter"}), 400
    path = PATHS.ftir_dir / "local_db/spectra_248_251.csv"
    out = {"csv_mb": round(path.stat().st_size / 1e6, 1)}
    frames = {}
    with COMPUTE_LOCK:
        for engine in ("polars", "pandas"):
            t0 = time.time()
            frame, used = _read_pool_spectra(path, D["wcols"], engine=engine)
            out[f"{engine}_s"] = round(time.time() - t0, 2)
            assert used == engine
            frames[engine] = (frame[~frame["AnalysisId"].duplicated()]
                              .set_index("AnalysisId"))
    a = frames["polars"].to_numpy()
    b = frames["pandas"].to_numpy()
    differ = (a != b) & ~(np.isnan(a) & np.isnan(b))
    n = int(differ.sum())
    max_ulp = 0.0
    if n:
        max_ulp = float(np.max(
            np.abs(a[differ].astype(np.float64) - b[differ].astype(np.float64))
            / np.spacing(np.abs(b[differ]).astype(np.float32))))
    out.update({
        "rows": int(a.shape[0]), "cols": int(a.shape[1]),
        "speedup": round(out["pandas_s"] / out["polars_s"], 1),
        "index_equal": bool(frames["polars"].index.equals(frames["pandas"].index)),
        "cells_differing": n, "cells_total": int(a.size),
        "max_ulp_float32": round(max_ulp, 2),
    })
    return jsonify(out)


@app.route("/api/ranking", methods=["POST"])
def api_ranking():
    """The selection-cutoff diagnostic: metric vs rank, to eyeball jumps."""
    if not STATE["ready"]:
        return jsonify({"error": "data still loading"}), 503
    b = request.get_json(force=True)
    cohort = b.get("cohort")
    if cohort not in RANKED_COHORTS:
        return jsonify({"error": f"{cohort} is not a rank-based cohort"}), 400
    try:
        _, metric, label = _ranking(cohort, b.get("selection_space", "raw"))
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
    stride = max(1, len(metric) // 2000)
    idx = np.arange(0, len(metric), stride)
    # Distribution view: the whole IMPROVE candidate population, clipped at the
    # 99th percentile so the long tail doesn't flatten the interesting region.
    finite = metric[np.isfinite(metric)]
    clip_hi = float(np.percentile(finite, 99))
    counts, edges = np.histogram(np.clip(finite, None, clip_hi), bins=60)
    centers = (edges[:-1] + edges[1:]) / 2
    cutoff = min(int(b.get("cutoff") or DEFAULT_CUTOFF[cohort]), len(metric))
    return jsonify({"rank": idx.tolist(),
                    "metric": [round(float(metric[i]), 5) for i in idx],
                    "hist": {"centers": np.round(centers, 5).tolist(),
                             "counts": counts.tolist(),
                             "clipped_at_p99": clip_hi},
                    "cutoff_metric": round(float(metric[cutoff - 1]), 5),
                    "label": label, "n_total": int(len(metric)),
                    "default_cutoff": DEFAULT_CUTOFF[cohort]})


def _spectra_bands(M, sl):
    q = np.percentile(np.asarray(M, float), [25, 50, 75], axis=0)
    return {"q25": np.round(q[0][sl], 5).tolist(),
            "median": np.round(q[1][sl], 5).tolist(),
            "q75": np.round(q[2][sl], 5).tolist()}


@app.route("/api/spectra", methods=["POST"])
def api_spectra():
    """Meeting item: cohort spectra vs Addis, side by side (median + IQR band).

    With ``compare: true`` this is Ann's multi-cohort ask instead: the Ethiopia-shaped,
    spectral-analog and lowest-OC/EC cohorts overlaid against the Addis median, all in
    the requested spectra space.
    """
    if not STATE["ready"]:
        return jsonify({"error": "data still loading"}), 503
    b = request.get_json(force=True)
    cfg = _config_from(b)
    with COMPUTE_LOCK:
        wn = D["corr_wn"] if cfg["spectra"] == "airspec" else D["wn_raw"]
        stride = max(1, len(wn) // 900)
        sl = slice(None, None, stride)
        try:
            if b.get("compare"):
                series = []
                for cohort in ("eth_shaped", "analogs", "ocec"):
                    ids, label, X, _, _, X_eval = _cohort_arrays(
                        cohort, None,
                        cfg["selection_space"] if cohort in ("eth_shaped", "analogs") else "raw",
                        cfg["spectra"], cfg["lot"], cfg["target"])
                    series.append({"label": f"{label} (n={len(ids)})",
                                   **_spectra_bands(X, sl)})
                return jsonify({"wn": np.round(np.asarray(wn, float)[sl], 2).tolist(),
                                "series": series, "reference": _spectra_bands(X_eval, sl),
                                "reference_label": get_target(cfg["target"])["label"],
                                "spectra": cfg["spectra"], "compare": True})
            ids, label, X, _, _, X_eval = _cohort_arrays(
                cfg["cohort"], cfg["cutoff"], cfg["selection_space"], cfg["spectra"],
                cfg["lot"], cfg["target"])
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        if b.get("clusters"):
            # Satoshi's "possible later step": k-means sub-types within the cohort.
            from sklearn.cluster import KMeans
            n_clusters = max(2, min(int(b["clusters"]), 6))
            Xf = np.asarray(X, float)
            if len(Xf) > 2500:
                keep = np.random.default_rng(0).choice(len(Xf), 2500, replace=False)
                Xf = Xf[keep]
            labels = KMeans(n_clusters=n_clusters, n_init=4,
                            random_state=0).fit_predict(Xf)
            series = []
            for c in range(n_clusters):
                members = Xf[labels == c]
                if len(members) < 3:
                    continue
                series.append({"label": f"sub-type {c + 1} (n={len(members)})",
                               **_spectra_bands(members, sl)})
            series.sort(key=lambda s: -int(s["label"].split("n=")[1].rstrip(")")))
            return jsonify({"wn": np.round(np.asarray(wn, float)[sl], 2).tolist(),
                            "series": series, "reference": _spectra_bands(X_eval, sl),
                            "reference_label": get_target(cfg["target"])["label"],
                            "cohort_label": label, "n": int(len(ids)),
                            "spectra": cfg["spectra"], "compare": True})
        return jsonify({"wn": np.round(np.asarray(wn, float)[sl], 2).tolist(),
                        "cohort": _spectra_bands(X, sl),
                        "reference": _spectra_bands(X_eval, sl),
                        "reference_label": get_target(cfg["target"])["label"],
                        "cohort_label": label, "n": int(len(ids)),
                        "spectra": cfg["spectra"]})


@app.route("/api/cohort_info", methods=["POST"])
def api_cohort_info():
    """Characteristics of the calibration reference data the current cohort selects."""
    if not STATE["ready"]:
        return jsonify({"error": "data still loading"}), 503
    cfg = _config_from(request.get_json(force=True))
    with COMPUTE_LOCK:
        try:
            ids, label = resolve_cohort(cfg["cohort"], cfg["cutoff"],
                                        cfg["selection_space"], cfg["spectra"],
                                        cfg["lot"])
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        sub = D["pool"].loc[ids]
    lots = pd.Series([D["lot"].get(int(i), "?") for i in ids]).value_counts()
    sites = sub["Site"].value_counts()
    dates = pd.to_datetime(sub["date"], errors="coerce").dropna()

    def stats(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return None
        return {"min": round(float(s.min()), 2), "median": round(float(s.median()), 2),
                "max": round(float(s.max()), 2)}

    # Composition ruler: the whole pool's OC/EC distribution vs this cohort's,
    # with the Addis FTIR-derived OC/EC marker (ftir_30: 1.34; not thermal —
    # no Addis filter has TOR).
    pool_ratio = pd.to_numeric(D["pool"]["OC_EC_ratio"], errors="coerce").dropna()
    cohort_ratio = pd.to_numeric(sub["OC_EC_ratio"], errors="coerce").dropna()
    edges = np.linspace(0, 25, 61)
    pool_counts, _ = np.histogram(pool_ratio.clip(upper=25), bins=edges)
    cohort_counts, _ = np.histogram(cohort_ratio.clip(upper=25), bins=edges)
    centers = ((edges[:-1] + edges[1:]) / 2).round(3)
    composition = {"centers": centers.tolist(),
                   "pool": pool_counts.tolist(),
                   "cohort": cohort_counts.tolist(),
                   "pool_median": round(float(pool_ratio.median()), 2),
                   "addis_marker": 1.34}

    return jsonify({
        "composition": composition,
        "label": label, "n": int(len(ids)), "n_sites": int(sites.size),
        "top_sites": [f"{s} ({n})" for s, n in sites.head(5).items()],
        "lots": {str(k): int(v) for k, v in lots.items()},
        "ec_loading_ug": stats(sub["TOR_EC_loading_ug"]),
        "ec_ugm3": stats(sub["TOR_EC_ugm3"]),
        "ocec_ratio": stats(sub["OC_EC_ratio"]),
        "date_range": ([str(dates.min().date()), str(dates.max().date())]
                       if len(dates) else None),
    })


@app.route("/api/overlap", methods=["POST"])
def api_overlap():
    """Meeting item: are the selection methods picking the same filters?

    Pairwise membership overlap between the selection cohorts at the given cutoffs,
    including the raw- vs corrected-space Ethiopia-shaped selections (the July
    next-step: "see if they actually differ between the two sets").
    """
    if not STATE["ready"]:
        return jsonify({"error": "data still loading"}), 503
    b = request.get_json(force=True)
    cuts = {"eth_shaped": int(b.get("eth_cutoff") or DEFAULT_CUTOFF["eth_shaped"]),
            "analogs": int(b.get("analog_cutoff") or DEFAULT_CUTOFF["analogs"]),
            "ocec": int(b.get("ocec_cutoff") or DEFAULT_CUTOFF["ocec"])}
    with COMPUTE_LOCK:
        try:
            sets = {
                "Biomass-smoke (906)":
                    set(resolve_cohort("smoke", None, "raw", "raw")[0].tolist()),
                f"Ethiopia-shaped raw ({cuts['eth_shaped']})":
                    set(resolve_cohort("eth_shaped", cuts["eth_shaped"], "raw", "raw")[0].tolist()),
                f"Ethiopia-shaped corrected ({cuts['eth_shaped']})":
                    set(resolve_cohort("eth_shaped", cuts["eth_shaped"], "airspec", "raw")[0].tolist()),
                f"Spectral analogs raw ({cuts['analogs']})":
                    set(resolve_cohort("analogs", cuts["analogs"], "raw", "raw")[0].tolist()),
                f"Spectral analogs corrected ({cuts['analogs']})":
                    set(resolve_cohort("analogs", cuts["analogs"], "airspec", "raw")[0].tolist()),
                f"Lowest-OC/EC ({cuts['ocec']})":
                    set(resolve_cohort("ocec", cuts["ocec"], "raw", "raw")[0].tolist()),
            }
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
    names = list(sets)
    rows = []
    for i, a in enumerate(names):
        for bn in names[i + 1:]:
            rows.append({"a": a, "b": bn, "n_a": len(sets[a]), "n_b": len(sets[bn]),
                         "overlap": len(sets[a] & sets[bn])})
    return jsonify({"rows": rows})


if __name__ == "__main__":
    print("Calibration Iteration Explorer → http://127.0.0.1:5058")
    app.run(host="127.0.0.1", port=5058, debug=False, threaded=True)
