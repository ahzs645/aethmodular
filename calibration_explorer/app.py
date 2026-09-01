"""Calibration Iteration Explorer: a standalone local Flask app at the repo root.

The July-17 meeting left a grid of options to trial: cohort x selection cutoff x
baseline-corrected-or-not (both for the *selection* and for the *calibration*) x CV
scheme x component count, read out as the target crossplot under OLS/Deming at
MAC 6/10/17.
Rather than one notebook per grid cell, this app drives the locked phase-3 machinery
(`research/ftir_ec_phase3/scripts/`, the committed cohort tables, the AIRSpec caches)
from an interactive page: pick a configuration, get its CV curve, click a k, read the
crossplot: then pin runs and compare intercepts across configurations.

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
import subprocess
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
from sklearn.decomposition import PCA

# All calibration math is the SAME code the notebooks run: nothing is re-derived here:
# the two CV protocols and the Deming λ convention come from
# research/ftir_ec_phase3/scripts/calibration_modes.py, and the estimators/selection
# metric from research/ftir_hips_chem/scripts/pls_transfer.py.
from phase3_common import (  # noqa: E402  (also puts phase-2 scripts on sys.path)
    PATHS, PHASE2_SCRIPTS, PHASE2_TABLES,
    load_addis_evaluation, load_pool_metadata, load_tor_loadings,
)
from calibration_modes import (  # noqa: E402
    DEMING_LAMBDA_MAC10, MAX_COMPONENTS, deming_lambda,
    protocol_cv_curve, protocol_select_k, protocol_train_mask,
)
from optimizer_stability import (  # noqa: E402
    percentile_summary, selection_frequency, stratified_bootstrap_indices,
)
from pls_transfer import (  # noqa: E402
    band_feature_distance, deming_regression, mahalanobis_distance_squared,
    pairwise_score_distance_squared, project_scores, regression_metrics,
    score_metric, select_components_cv, spectral_q_residual, vip_scores,
)
from config import season_for_month  # noqa: E402  (phase-2 scripts)

CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_SCHEMA_VERSION = "2026-08-24-v3-q-residual"
TARGET_REGISTRY_PATH = HERE / "target_registry.json"
TARGET_REGISTRY = (json.loads(TARGET_REGISTRY_PATH.read_text())
                   if TARGET_REGISTRY_PATH.exists() else {})
try:
    GIT_COMMIT = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    GIT_DIRTY = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True,
        stderr=subprocess.DEVNULL,
    ).strip())
except (OSError, subprocess.CalledProcessError):
    GIT_COMMIT = os.environ.get("CALIB_EXPLORER_COMMIT", "unknown")
    GIT_DIRTY = None
# Evaluation targets beyond the built-in Addis set: drop a folder here containing
# spectra.csv (id column + the IMPROVE wavenumber columns) and reference.csv
# (id column + Fabs or EC_ugm3, Volume_m3, optional Group): see README.
TARGETS_DIR = HERE / "targets"
TARGETS_DIR.mkdir(exist_ok=True)

COHORTS = {
    "pool": "Entire IMPROVE network (no selection)",
    "smoke": "Biomass-smoke (906)",
    "eth_shaped": "Ethiopia-shaped smoke",
    "analogs": "Spectral analogs",
    "ocec": "Lowest-OC/EC",
}

# Savitzky-Golay second derivative: identical parameters to ftir_20's comparison.
SAVGOL = dict(window_length=11, polyorder=2, deriv=2)
SPECTRA_LABEL = {
    "raw": "as-measured",
    "airspec": "AIRSpec df1=6 baselined",
    "neutral": "neutral pspline-arPLS baselined",
    "deriv2": "SG 2nd derivative of raw",
}
DEFAULT_CUTOFF = {"eth_shaped": 300, "analogs": 500, "ocec": 800}
RANKED_COHORTS = set(DEFAULT_CUTOFF)

# Evaluation-view levers (all post-fit: they narrow the readout, never the fit).
# `early`/`late` are the date-ordered halves Ann asked for on 2026-08-27 - pick a
# configuration on one half, read it out on the half that never guided the
# choice. `odd`/`even` interleave in date order instead, so both halves share the
# same seasonal and temporal coverage: the control that separates a real time
# trend from ordinary sampling scatter. Every split carries the SAME n, so the
# two halves' R2 values are directly comparable.
EVAL_SPLITS = ("all", "early", "late", "odd", "even")
EVAL_SPLIT_COMPLEMENT = {"early": "late", "late": "early",
                         "odd": "even", "even": "odd"}

# The group lever can run on more than one grouping of the same filters. A
# target's `groups` list stays the DEFAULT scheme (season everywhere: Ethiopian
# calendar at Addis/Bishoftu, quarters at the SPARTAN sites) so everything that
# reads `t["groups"]` - crossplot colouring, the plausibility group medians, the
# stability bootstrap's strata - keeps its old meaning. Additional schemes live
# in `t["group_schemes"]`, one per-filter label list each, same length and order
# as `ref`. Addis gains Navid's PMF source apportionment there (group meeting
# 2026-08-27: read the calibration out on marine days against combustion days).
DEFAULT_GROUP_SCHEME = "season"
GROUP_SCHEME_LABELS = {DEFAULT_GROUP_SCHEME: "Season"}

app = Flask(__name__, static_folder=str(HERE / "static"))

STATE = {"ready": False, "error": None, "message": "starting…", "checks": []}
D: dict = {}
COMPUTE_LOCK = threading.Lock()


def _fingerprint_arrays(*arrays) -> str:
    """Content fingerprint for the modest-sized target arrays."""
    digest = hashlib.sha256()
    for value in arrays:
        if value is None:
            digest.update(b"none|")
            continue
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()[:20]


def _write_text_atomic(path: Path, payload: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(payload)
    temp.replace(path)


def target_meta(name: str) -> dict:
    """Registry metadata; unknown targets default to exploratory/no optimization."""
    default = {
        "display_name": name,
        "physical_site": name,
        "site_code": None,
        "role": "unregistered_custom",
        "provisional": True,
        "optimization_allowed": False,
        "cross_site_default": False,
    }
    return {**default, **TARGET_REGISTRY.get(name, {})}


# ----------------------------------------------------------------------------- #
# data loading (once, in a background thread)
# ----------------------------------------------------------------------------- #
def _read_pool_spectra(path, wcols, engine=None):
    """The 725 MB pool-spectra read, via polars' multithreaded reader when
    installed (~6x faster than the pandas read in the same-shape trial -
    research/package_trials/io/). polars parses correctly rounded, bit-identical
    to pandas float_precision='round_trip'; against the historical default
    pandas read it differed on 1 cell in 35.4M, by 1 float32 ulp on an ~1e-5
    value: far below every reported precision. Set CALIB_EXPLORER_CSV=pandas
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
        # Training frame: identical construction to ftir_21.
        pool = (meta.query("TOR_EC_loading_ug > 0").drop_duplicates("FilterId").copy())
        pool["AnalysisId"] = pool["AnalysisId"].astype(int)
        pool = pool[pool["AnalysisId"].isin(pool_raw.index)].drop_duplicates("AnalysisId")
        pool = pool.set_index("AnalysisId")[
            ["Site", "date", "TOR_EC_loading_ug", "OC_EC_ratio",
             "TOR_EC_ugm3", "TOR_OC_ugm3"]]
        D["pool"] = pool

        # Lowest-OC/EC ranking: identical eligibility + ordering to ftir_11.
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

        STATE["message"] = "loading neutral-baseline caches…"
        neutral_path = (PHASE3_DIR / "output/corrected/"
                        "improve_pool_neutral_pspline_arpls_lam1e6.npz")
        etad_neutral_path = (PHASE3_DIR / "output/corrected/"
                             "etad_neutral_pspline_arpls_lam1e6.npz")
        neutral = np.load(neutral_path, allow_pickle=True)
        D["neutral_pool"] = neutral["corrected"]
        D["neutral_wn"] = neutral["wn"].astype(float)
        D["neutral_row"] = {
            int(a): i for i, a in enumerate(neutral["analysis_id"].astype(int))
        }
        etad_neutral = np.load(etad_neutral_path, allow_pickle=True)
        etad_neutral_frame = pd.DataFrame(etad_neutral["corrected"].astype(float))
        etad_neutral_frame["MediaId"] = etad_neutral["media_id"].astype(int)
        D["X_addis_neutral"] = (
            etad_neutral_frame.groupby("MediaId").mean()
            .loc[etad_eval["MediaId"].astype(int)].to_numpy(float)
        )

        # Per-filter lot of the evaluation set, from the HIPS LotId (Ann,
        # 2026-08-19: judge a lot-251 calibration on lot-251 filters: most
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
        iso_dates = [d.date().isoformat() if pd.notna(d) else None for d in dates]

        # Second grouping axis: Navid's ETAD PMF source apportionment, joined on
        # the exact sampling date (102 PMF days of calendar 2023 against 239
        # evaluation filters, so most filters come back `unmatched`). Imported
        # here rather than at module scope on purpose - research/ftir_hips_chem
        # and its Filter Data CSV are not guaranteed to be present, and a missing
        # source table must cost the app its PMF axis, not its startup.
        STATE["message"] = "attaching PMF source groups…"
        group_schemes = {DEFAULT_GROUP_SCHEME: D["season"]}
        try:
            from pmf_source_groups import SCHEME_LABELS as PMF_SCHEME_LABELS
            from pmf_source_groups import pmf_group_schemes
            group_schemes.update(pmf_group_schemes(iso_dates))
            GROUP_SCHEME_LABELS.update(PMF_SCHEME_LABELS)
            matched = sum(1 for v in group_schemes["pmf_source"] if v != "unmatched")
            pmf_check = {"name": "PMF source groups attached to Addis", "ok": True,
                         "detail": f"{matched}/{len(iso_dates)} evaluation filters "
                                   "have a PMF day (exact date match)"}
        except Exception as exc:                              # noqa: BLE001
            pmf_check = {"name": "PMF source groups attached to Addis", "ok": False,
                         "detail": f"{type(exc).__name__}: {exc} "
                                   "(the season scheme is unaffected)"}
        D["targets"] = {"addis": {
            "label": target_meta("addis")["display_name"],
            "ref_kind": "fabs",          # x = ref/MAC; "ec" targets use x = ref
            "ref": D["fabs"],
            "volume": D["volume"],
            "X_raw": D["X_addis_raw"],
            "X_corr": D["X_addis_corr"],
            "X_neutral": D["X_addis_neutral"],
            "groups": D["season"],
            "group_schemes": group_schemes,
            "fixed_mask": D["fixed_mask"],
            "dates": iso_dates,
            "deployed": [round(float(v), 4) if pd.notna(v) else None for v in deployed],
            "lots": eval_lots,
            "filter_ids": etad_eval["ExternalFilterId"].astype(str).tolist(),
        }}
        D["targets"]["addis"]["fingerprint"] = _fingerprint_arrays(
            D["targets"]["addis"]["X_raw"], D["targets"]["addis"]["X_corr"],
            D["targets"]["addis"]["X_neutral"],
            D["targets"]["addis"]["ref"], D["targets"]["addis"]["volume"],
        )

        # Content fingerprint of the arrays that determine fits and cohort
        # membership, plus the shared model code. Unlike path/mtime metadata,
        # this is stable when the same inputs move between a laptop and Colab.
        D["source_fingerprint"] = _fingerprint_arrays(
            pool_raw.index.to_numpy(np.int64), pool_raw.to_numpy(np.float32),
            pool.index.to_numpy(np.int64),
            np.asarray(pool["Site"].astype(str).tolist(), dtype="U"),
            pool["date"].to_numpy(dtype="datetime64[ns]"),
            pool["TOR_EC_loading_ug"].to_numpy(float),
            D["corr_pool"],
            np.array(sorted(D["corr_row"], key=D["corr_row"].get), dtype=np.int64),
            D["neutral_pool"],
            np.array(sorted(D["neutral_row"], key=D["neutral_row"].get), dtype=np.int64),
            D["smoke_ids"], D["eth_ranked"], D["analog_ranked"], D["ocec_ranked"],
            (PHASE3_DIR / "scripts/calibration_modes.py").read_bytes(),
            (PHASE2_SCRIPTS / "pls_transfer.py").read_bytes(),
        )

        # Startup provenance checks against the locked cohorts (reported, not fatal).
        # Top-N is counted over TOR-eligible filters: the same rule the app's
        # cohort resolution uses (see _ranking).
        def top_eligible(ranked, n):
            r = np.asarray(ranked, dtype=int)
            return set(int(i) for i in r[np.isin(r, pool.index.to_numpy())][:n])

        checks = [{"name": f"pool spectra loaded via {csv_engine}", "ok": True,
                   "detail": "polars is the fast path when installed; "
                             "CALIB_EXPLORER_CSV=pandas forces the pandas read"},
                  pmf_check]
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
def _lot_label(value) -> str:
    """'251' from 251, 251.0 or '251.0'; '?' when the lot is missing."""
    try:
        if value is None or pd.isna(value):
            return "?"
    except (TypeError, ValueError):
        pass
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def list_targets(*, cross_site_only: bool = False):
    out = {"addis": target_meta("addis")["display_name"]}
    for p in sorted(TARGETS_DIR.iterdir()) if TARGETS_DIR.exists() else []:
        if p.is_dir() and (p / "spectra.csv").exists():
            meta = target_meta(p.name)
            if cross_site_only and not meta["cross_site_default"]:
                continue
            out[p.name] = meta["display_name"]
    if cross_site_only:
        out = {name: label for name, label in out.items()
               if target_meta(name)["cross_site_default"]}
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
    # Per-filter lot, when the reference table carries one. build_spartan_target.py
    # joins it from the HIPS LotId; targets without it simply have no evaluation-lot
    # lever (the built-in Addis target gets its lots in _load_all instead).
    lot_col = next((c for c in ("LotId", "Lot", "LotNumber") if c in m.columns), None)
    t = {"label": target_meta(name)["display_name"], "ref_kind": ref_kind,
         "ref": ref_values[keep], "volume": m["Volume_m3"].to_numpy(float),
         "X_raw": m[D["wcols"]].to_numpy(float), "X_corr": None,
         "X_neutral": None,
         "groups": (m["Group"].astype(str).tolist() if "Group" in m.columns
                    else ["all"] * len(m)),
         "lots": ([_lot_label(v) for v in m[lot_col]] if lot_col else None),
         "fixed_mask": None, "dates": dates, "deployed": None,
         "filter_ids": (m["ExternalFilterId"].astype(str).tolist()
                        if "ExternalFilterId" in m.columns else None)}
    # optional AIRSpec-corrected spectra (id column + the CORRECTED wavenumber
    # grid, i.e. corr_wn: not the raw grid): enables "Calibrate on AIRSpec"
    # readouts against this target
    corr_path = d / "spectra_corrected.csv"
    if corr_path.exists():
        cp = pd.read_csv(corr_path)
        cid = cp.columns[0]
        num = list(cp.columns[1:])
        try:
            cw = np.array([float(c) for c in num])
            if (len(cw) == len(D["corr_wn"])
                    and np.allclose(cw, D["corr_wn"], atol=1e-3)
                    and set(m[idc]) <= set(cp[cid])):
                cp = cp.set_index(cid)
                t["X_corr"] = cp.loc[m[idc], num].to_numpy(float)
        except ValueError:
            pass
    neutral_path = d / "spectra_neutral.csv"
    if neutral_path.exists():
        neutral = pd.read_csv(neutral_path)
        neutral_id = neutral.columns[0]
        neutral_names = list(neutral.columns[1:])
        try:
            neutral_wn = np.array([float(value) for value in neutral_names])
            if (len(neutral_wn) == len(D["neutral_wn"])
                    and np.allclose(neutral_wn, D["neutral_wn"], atol=1e-3)
                    and set(m[idc]) <= set(neutral[neutral_id])):
                neutral = neutral.set_index(neutral_id)
                t["X_neutral"] = neutral.loc[m[idc], neutral_names].to_numpy(float)
        except ValueError:
            pass
    t["fingerprint"] = _fingerprint_arrays(
        t["X_raw"], t["X_corr"], t["X_neutral"], t["ref"], t["volume"],
    )
    D["targets"][name] = t
    return t


def _target_X(t, spectra):
    if spectra == "airspec":
        if t["X_corr"] is None:
            raise ValueError("this target has no AIRSpec-corrected spectra: "
                             "calibrate on raw or SG 2nd derivative instead")
        return t["X_corr"]
    if spectra == "neutral":
        if t["X_neutral"] is None:
            raise ValueError("this target has no neutral-baseline spectra: run "
                             "build_neutral_baseline_cache.py first")
        return t["X_neutral"]
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
    spectra are already baselined). Cached to disk: first call takes ~20 s.
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

    # 1. The IMPROVE-HIPS training set, exactly as in ftir_09: apps EC set joined
    #    to results_hips → HIPS tau: but with corrected spectra.
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
    phase-2 analog cohort: which IS the top 500 eligible: never reproduces).
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
            and (spectra != "neutral" or i in D["neutral_row"])
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
    # MAC 17 is a sensitivity readout requested in the Aug-2026 discussion. It
    # is deliberately a readout, not another grid dimension: choosing a MAC
    # after seeing the target slope would merely tune the x-axis scale.
    macs = (10.0, 6.0, 17.0) if t["ref_kind"] == "fabs" else (None,)
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
# evaluation view: which target filters the readout is reported on.
# Lot x season/group x equal-n split, all applied AFTER the cached fit, so the
# curve and fit caches stay view-agnostic and every view of one configuration is
# a cache hit. Predictions always cover every filter; only the crossplot narrows.
# ----------------------------------------------------------------------------- #
def _eval_order(t, indices):
    """Date order of a subset of target filters (falls back to file order)."""
    dates = t.get("dates")
    if not dates or not any(dates[i] for i in indices):
        return np.asarray(indices)
    key = np.array([dates[i] or "9999-12-31" for i in indices])
    return np.asarray(indices)[np.argsort(key, kind="stable")]


def _split_indices(t, indices, split):
    """Equal-n half of `indices` under one split rule.

    Both halves carry the same n, so their R2 and RMSE are comparable - the point
    Ann made about test sets of unequal size.

    The split is STRATIFIED by the fixed/all evaluation subset. Without that, a
    date-ordered halving lands most of the fixed-set filters in one half (at
    Addis: 119 fixed filters early against 70 late), so the readout the
    leaderboard actually scores would be compared across unequal n - exactly the
    thing the split exists to avoid. Splitting each stratum separately keeps both
    the fixed-set and all-pairs readouts equal-n between halves. A stratum too
    small to halve is dropped from both halves rather than unbalancing them.
    """
    if split in (None, "all"):
        return np.asarray(indices)
    if split not in EVAL_SPLITS:
        raise ValueError(f"unknown evaluation split {split!r} "
                         f"(expected one of {', '.join(EVAL_SPLITS)})")
    indices = np.asarray(indices)
    fixed = t.get("fixed_mask")
    if fixed is not None:
        fixed = np.asarray(fixed, bool)
        strata = [indices[fixed[indices]], indices[~fixed[indices]]]
    else:
        strata = [indices]
    picked = []
    for stratum in strata:
        order = _eval_order(t, stratum)
        half = len(order) // 2
        if half < 1:
            continue
        if split == "early":
            picked.append(order[:half])
        elif split == "late":
            picked.append(order[-half:])
        elif split == "odd":
            picked.append(order[1::2][:half])
        else:
            picked.append(order[0::2][:half])       # "even"
    out = np.concatenate(picked) if picked else np.asarray([], int)
    if len(out) < 3:
        raise ValueError(f"evaluation view has only {len(indices)} filters: "
                         "an equal-n split needs at least 6 in an evaluation subset")
    return np.sort(out)


def group_scheme_names(t):
    """The grouping schemes this target carries, default scheme first."""
    schemes = t.get("group_schemes") or {}
    return list(schemes) or [DEFAULT_GROUP_SCHEME]


def _group_values(t, group_scheme=DEFAULT_GROUP_SCHEME):
    """Per-filter labels under one named scheme, aligned with the target's rows.

    The default scheme falls back to `t["groups"]` so a target that predates
    named schemes (any custom target with a `Group` column) still answers to it.
    An unknown scheme RAISES rather than silently reverting to season: a readout
    labelled "combustion" that quietly reported a season would be worse than an
    error. Cross-site and batch callers pre-resolve through resolve_eval_view,
    which falls back to the default scheme for targets that lack the requested
    one, so this only bites a direct request.
    """
    scheme = group_scheme or DEFAULT_GROUP_SCHEME
    schemes = t.get("group_schemes") or {}
    if scheme in schemes:
        return list(schemes[scheme])
    if scheme == DEFAULT_GROUP_SCHEME:
        return list(t.get("groups") or [])
    raise ValueError(f"this target has no {scheme!r} grouping scheme "
                     f"(available: {', '.join(group_scheme_names(t))})")


def _eval_indices(t, eval_lot="all", eval_group="all", eval_split="all",
                  group_scheme=DEFAULT_GROUP_SCHEME):
    """Target-filter indices surviving lot -> group -> split, in that order.

    The split runs on the survivors, so `lot 251 + early` really is the first
    half of the lot-251 filters rather than the lot-251 members of the first
    half of everything.

    `group_scheme` picks which grouping the group lever selects within (season,
    or one of the PMF schemes at Addis). It is validated even when the group is
    "all" and the labels therefore go unused: an unsupported scheme silently
    accepted here is a request that did not do what it said.
    """
    idx = np.arange(len(t["ref"]))
    groups = _group_values(t, group_scheme)
    if eval_lot not in (None, "all"):
        lots = t.get("lots")
        if lots is None:
            raise ValueError("this target has no filter-lot information "
                             "(rebuild it with a LotId column to use the eval-lot lever)")
        idx = idx[np.asarray([str(lots[i]) == str(eval_lot) for i in idx])]
        if len(idx) < 3:
            raise ValueError(f"only {len(idx)} evaluation filters on lot {eval_lot}")
    if eval_group not in (None, "all"):
        known = {str(g) for g in groups}
        if str(eval_group) not in known:
            raise ValueError(
                f"no evaluation filter carries {group_scheme} group "
                f"{str(eval_group)!r} (available: {', '.join(sorted(known))})")
        idx = idx[np.asarray([str(groups[i]) == str(eval_group) for i in idx])]
        if len(idx) < 3:
            raise ValueError(f"only {len(idx)} evaluation filters in group {eval_group}")
    return _split_indices(t, idx, eval_split)


def _slice_target(t, idx):
    """A target dict restricted to `idx`, keeping every per-filter field aligned."""
    idx = np.asarray(idx)
    fixed = t["fixed_mask"][idx] if t.get("fixed_mask") is not None else None
    if fixed is not None and fixed.sum() < 3:
        fixed = None                        # fixed subset too small in this view
    def take(field):
        values = t.get(field)
        return [values[i] for i in idx] if values else None
    schemes = t.get("group_schemes")
    return {**t, "ref": t["ref"][idx], "groups": take("groups"),
            "group_schemes": ({name: [values[i] for i in idx]
                               for name, values in schemes.items()}
                              if schemes else None),
            "fixed_mask": fixed, "dates": take("dates"),
            "deployed": take("deployed"), "lots": take("lots"),
            "filter_ids": take("filter_ids")}


def _eval_view(t, pred, eval_lot="all", eval_group="all", eval_split="all",
               group_scheme=DEFAULT_GROUP_SCHEME):
    """(sliced target, sliced predictions, index array) for one readout view."""
    idx = _eval_indices(t, eval_lot, eval_group, eval_split, group_scheme)
    if len(idx) == len(t["ref"]) and (idx == np.arange(len(idx))).all():
        return t, pred, idx
    return _slice_target(t, idx), np.asarray(pred)[idx], idx


def resolve_eval_view(target_name, eval_lot="all", eval_group="all",
                      eval_split="all", group_scheme=DEFAULT_GROUP_SCHEME):
    """The view levers this target can actually honour, as (lot, group, split, scheme).

    Lots, grouping schemes and group names are site-specific: a lot-251, Belg or
    PMF-marine readout is meaningful at Addis and simply does not exist at
    Pasadena. Rather than error a whole cross-site row, fall back to "all" on the
    axes the target cannot support. A scheme the target lacks drops back to the
    default one together with its group, since a group name only means anything
    inside the scheme it came from. The equal-n split is site-independent and
    always applies.
    """
    try:
        t = get_target(target_name)
    except Exception:                                     # noqa: BLE001
        return "all", "all", eval_split, DEFAULT_GROUP_SCHEME
    options = eval_view_options(t)
    lot = (eval_lot if eval_lot in (None, "all")
           or str(eval_lot) in options["lots"] else "all")
    scheme = group_scheme or DEFAULT_GROUP_SCHEME
    if scheme not in options["group_schemes"]:
        scheme, eval_group = DEFAULT_GROUP_SCHEME, "all"
    group = (eval_group if eval_group in (None, "all")
             or str(eval_group) in options["group_schemes"][scheme] else "all")
    group = group or "all"
    # With no group selected the scheme is a no-op; report the default so two
    # requests that ask for the same readout carry the same view identity
    # (the batch's row key and the run cache key both include the scheme).
    return (lot or "all"), group, (eval_split or "all"), (
        scheme if group != "all" else DEFAULT_GROUP_SCHEME)


def eval_view_options(t):
    """Lot / group choices this target actually supports, with their counts.

    `groups` stays the DEFAULT (season) scheme's counts, unchanged, so a caller
    that predates named schemes keeps working. `group_schemes` carries every
    scheme including that one.
    """
    lots = t.get("lots")
    n = len(t["ref"])
    def counts(values):
        if not values:
            return {}
        out = {}
        for v in values:
            out[str(v)] = out.get(str(v), 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))
    try:
        split_n = int(len(_split_indices(t, np.arange(n), "early")))
    except ValueError:
        split_n = 0
    schemes = {name: counts(_group_values(t, name))
               for name in group_scheme_names(t)}
    return {"n": int(n),
            "lots": {k: v for k, v in counts(lots).items() if k != "?"},
            "groups": schemes.get(DEFAULT_GROUP_SCHEME, {}),
            "group_schemes": schemes,
            "group_scheme_labels": {name: GROUP_SCHEME_LABELS.get(name, name)
                                    for name in schemes},
            # Flask sorts JSON object keys, so `group_schemes` reaches the page
            # alphabetically (pmf_class first) whatever order it was built in.
            # Render the picker from this list to keep the default scheme first.
            "group_scheme_order": list(schemes),
            "default_group_scheme": DEFAULT_GROUP_SCHEME,
            "splits": [s for s in EVAL_SPLITS if s == "all" or split_n >= 3],
            "split_n": split_n}


# ----------------------------------------------------------------------------- #
# the calibration run (curve cached separately from the k-specific fit)
# ----------------------------------------------------------------------------- #
def _cache_key(*parts) -> str:
    h = hashlib.sha1()
    h.update(CACHE_SCHEMA_VERSION.encode())
    h.update(b"|")
    h.update(str(D.get("source_fingerprint", "source-loading")).encode())
    h.update(b"|")
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
    elif spectra == "neutral":
        X = D["neutral_pool"][
            [D["neutral_row"][int(i)] for i in ids]
        ].astype(float)
    elif spectra == "deriv2":
        X = savgol_filter(D["pool_raw"].loc[ids, D["wcols"]].to_numpy(float),
                          axis=1, **SAVGOL)
    else:
        X = D["pool_raw"].loc[ids, D["wcols"]].to_numpy(float)
    X_eval = _target_X(get_target(target), spectra)
    return ids, label, X, y, sites, X_eval


def run_config(cohort, cutoff, selection_space, spectra, mode, k_override,
               max_components, lot="all", target="addis", eval_lot="all",
               eval_group="all", eval_split="all",
               group_scheme=DEFAULT_GROUP_SCHEME):
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
        # (ftir_22): the order here is the cohort's ranking/CSV order, matching the
        # committed runs' convention.
        curve = protocol_cv_curve(mode, X, y, sites, train,
                                  max_components=max_components)
        if "rmse_se" not in curve:
            curve["rmse_se"] = np.nan
        _write_text_atomic(
            curve_path, curve.replace({np.nan: None}).to_json(orient="records"))
        curve_cached = False

    auto_k = protocol_select_k(mode, curve)
    k = int(k_override) if k_override else int(auto_k)
    k = max(1, min(k, int(curve["n_components"].max())))

    fit_key = _cache_key("fit", curve_key, k, target, t["fingerprint"])
    fit_path = CACHE_DIR / f"{fit_key}.json"
    fit = None
    if fit_path.exists():
        fit = json.loads(fit_path.read_text())
        if "extrap" not in fit or "q_residual" not in fit:
            fit = None            # pre-diagnostic cache entry: recompute in place
    if fit is None:
        model = PLSRegression(n_components=k, scale=False).fit(X[train], y[train])
        Xe = np.asarray(X_eval, float)
        pred = model.predict(Xe).ravel() / t["volume"]
        heldout = None
        if mode == "site_heldout":
            hm = regression_metrics(y[~train], model.predict(X[~train]).ravel())
            heldout = {m: round(float(hm[m]), 4)
                       for m in ("slope", "intercept", "R2", "RMSE")}
        # Reggente et al. (2016)-style extrapolation diagnostic: whitened
        # distance in the fitted model's score space, target vs training cloud.
        # A target filter far beyond the training p95 is an extrapolation -
        # its contribution to slope/intercept should not be trusted.
        Ttr = model.transform(X[train])
        Tev = model.transform(Xe)
        mu, sd = Ttr.mean(axis=0), Ttr.std(axis=0) + 1e-12
        d_tr = np.sqrt((((Ttr - mu) / sd) ** 2).sum(axis=1))
        d_ev = np.sqrt((((Tev - mu) / sd) ** 2).sum(axis=1))
        # Orthogonal spectral residual (Q): score distance only asks whether a
        # target is unusual *within* the retained latent space. Q catches a
        # spectrum that the retained PLS subspace cannot reconstruct at all.
        q_tr = spectral_q_residual(model, X[train])
        q_ev = spectral_q_residual(model, Xe)
        fit = {"addis_ugm3": [round(float(v), 5) for v in pred],
               "heldout": heldout,
               "n_train": int(train.sum()),
               "n_train_sites": int(pd.Series(sites[train]).nunique()),
               "extrap": {"train_p95": round(float(np.percentile(d_tr, 95)), 4),
                          "d": [round(float(v), 3) for v in d_ev]},
               "q_residual": {
                   "train_p95": round(float(np.percentile(q_tr, 95)), 8),
                   "q": [round(float(v), 8) for v in q_ev],
               }}
        _write_text_atomic(fit_path, json.dumps(fit))

    pred = np.asarray(fit["addis_ugm3"], float)

    # Evaluation view (Ann, 2026-08-19 and 2026-08-27): report the readout on a
    # subset of the target's filters - one lot, one group (a season, or a PMF
    # source class under group_scheme), and/or one equal-n half. Applied after
    # the cached fit: predictions always cover every filter, only the
    # crossplot/metrics view is restricted, so the curve/fit caches stay
    # view-agnostic and every view of a fitted configuration is a cache hit.
    t_view, pred_view, idx_view = _eval_view(t, pred, eval_lot, eval_group,
                                             eval_split, group_scheme)

    ex = fit.get("extrap")
    extrap_pct = None
    if ex:
        dv = np.asarray(ex["d"], float)[idx_view]
        extrap_pct = round(100.0 * float((dv > ex["train_p95"]).mean()), 1)

    q_diag = fit.get("q_residual")
    q_residual_pct = None
    if q_diag:
        qv = np.asarray(q_diag["q"], float)[idx_view]
        q_residual_pct = round(
            100.0 * float((qv > q_diag["train_p95"]).mean()), 1)

    # Every equal-n half of the CURRENT lot/season view, scored off the same
    # cached predictions. Costs a regression on <=250 points, so the "pick on one
    # half, read out on the half that never guided the choice" comparison is
    # always available without a second fit.
    split_check = []
    for split_name in EVAL_SPLITS:
        try:
            t_s, pred_s, _ = _eval_view(t, pred, eval_lot, eval_group,
                                        split_name, group_scheme)
        except ValueError:
            continue
        split_check.append({"split": split_name, "n": int(len(pred_s)),
                            "metrics": crossplot_metrics(t_s, pred_s)})

    finite_pred = pred_view[np.isfinite(pred_view)]
    if len(finite_pred):
        group_medians = {}
        # Deliberately the DEFAULT (season) scheme, not `group_scheme`: these
        # medians and their span are a plausibility check the leaderboard scores
        # rows on, so their meaning has to stay fixed as the group lever moves.
        groups_view = np.asarray(t_view["groups"], object)
        for group in dict.fromkeys(groups_view.tolist()):
            vals = pred_view[(groups_view == group) & np.isfinite(pred_view)]
            if len(vals):
                group_medians[str(group)] = round(float(np.median(vals)), 4)
        plausibility = {
            "negative_pct": round(100.0 * float((finite_pred < 0).mean()), 1),
            "above_8_pct": round(100.0 * float((finite_pred > 8).mean()), 1),
            "median": round(float(np.median(finite_pred)), 4),
            "group_medians": group_medians,
            "group_median_span": (round(float(max(group_medians.values())
                                               - min(group_medians.values())), 4)
                                  if len(group_medians) > 1 else 0.0),
        }
    else:
        plausibility = {"negative_pct": None, "above_8_pct": None,
                        "median": None, "group_medians": {},
                        "group_median_span": None}

    floor_rows = curve.loc[curve["n_components"] == k, "rmsecv"]
    floor = float(floor_rows.iloc[0]) if len(floor_rows) else float("nan")
    run_id = _cache_key("run", cohort, cutoff, selection_space, spectra, mode,
                        lot, target, eval_lot, eval_group, eval_split,
                        group_scheme, k, t["fingerprint"], ids_hash)
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
        "provenance": {
            "run_id": run_id,
            "cache_schema": CACHE_SCHEMA_VERSION,
            "git_commit": GIT_COMMIT,
            "git_dirty": GIT_DIRTY,
            "source_fingerprint": D.get("source_fingerprint"),
            "target_fingerprint": t["fingerprint"],
            "resolved_cohort_hash": ids_hash,
        },
        "target": {"name": target, "label": t["label"], "ref_kind": t["ref_kind"],
                   "has_fixed": t_view.get("fixed_mask") is not None,
                   "eval_lot": eval_lot or "all",
                   "eval_group": eval_group or "all",
                   "group_scheme": group_scheme or DEFAULT_GROUP_SCHEME,
                   "eval_split": eval_split or "all",
                   "n_eval": int(len(pred_view)),
                   "n_target": int(len(t["ref"])),
                   "extrap_pct": extrap_pct,
                   "extrap_p95": ex["train_p95"] if ex else None,
                   "q_residual_pct": q_residual_pct,
                   "q_residual_p95": (q_diag["train_p95"]
                                      if q_diag else None)},
        "eval": {"ref": [round(float(v), 4) for v in t_view["ref"]],
                 "pred": [round(float(v), 5) for v in pred_view],
                 # `group` stays the season labels the crossplot has always
                 # coloured by; `scheme_group` is the same filters under the
                 # requested scheme (identical when that scheme IS season).
                 "group": t_view["groups"],
                 "scheme_group": _group_values(t_view, group_scheme),
                 "date": t_view.get("dates"),
                 "deployed": t_view.get("deployed"),
                 "fixed": ([bool(b) for b in t_view["fixed_mask"]]
                           if t_view.get("fixed_mask") is not None
                           else [False] * len(t_view["ref"]))},
        "plausibility": plausibility,
        "metrics": crossplot_metrics(t_view, pred_view),
        "split_check": split_check,
    }


# ----------------------------------------------------------------------------- #
# exhaustive batch optimizer: server-side, so it survives the browser tab and
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
    base = "|".join(str(r.get(f)) for f in
                    ("cohort", "cutoff", "selection_space", "spectra", "mode",
                     "lot", "target", "eval_lot"))
    # rows written before the evaluation-view levers existed carry neither field;
    # defaulting both to "all" keeps their keys stable against new rows
    view = "|".join(str(r.get(f) or "all") for f in ("eval_group", "eval_split"))
    # The scheme only names a readout when a group is actually selected, so rows
    # with eval_group "all" collapse onto the default: otherwise the same row
    # would be recomputed once per scheme. Rows written before schemes existed
    # carry no field and land on the same default.
    scheme = (str(r.get("group_scheme") or DEFAULT_GROUP_SCHEME)
              if r.get("eval_group") not in (None, "all") else DEFAULT_GROUP_SCHEME)
    return f"{base}|{view}|{scheme}|k{r.get('k')}"


def _batch_sweep_ks(auto_k, curve_max, k_min=1, k_max=30, dense=False):
    """Component candidates for unattended optimization.

    The old ladder stopped at 20 and could therefore never answer the meeting's
    explicit k=21 question. Dense mode evaluates every integer in the requested
    range. Sparse mode stays cheap but always includes the rule choice, 21 when
    in range, and the upper bound.
    """
    lo = max(1, int(k_min))
    hi = min(int(k_max), int(curve_max))
    if hi < lo:
        return []
    if dense:
        return list(range(lo, hi + 1))
    start = min(max(int(auto_k), lo), hi)
    ks = {start, hi}
    if lo <= 21 <= hi:
        ks.add(21)
    if hi > start:
        ks.update(round(start + i * (hi - start) / 7) for i in range(8))
    return sorted(ks)


def _batch_row(cfg, out):
    # same shape as the frontend's optRowFromRun, so saved rows merge straight
    # into the leaderboard / Pareto view. Carries the FULL readout: both
    # evaluation sets x both MACs x both estimators: so the leaderboard's
    # MAC / Fit / Addis-set toggles all apply to saved rows. (MAC-6 slopes are
    # not stored: slope scales by exactly 0.6 going 10->6, ftir_19.)
    def pick(es, mac):
        rows = ([m for m in out["metrics"]
                 if m["evaluation_set"] == es and m["MAC"] == mac]
                or [m for m in out["metrics"] if m["MAC"] == mac]
                or out["metrics"])
        return rows[0]
    ref_kind = out["target"]["ref_kind"]
    primary_mac = 10.0 if ref_kind == "fabs" else None
    secondary_mac = 6.0 if ref_kind == "fabs" else None
    f10, f6 = pick("fixed", primary_mac), pick("fixed", secondary_mac)
    a10, a6 = pick("all", primary_mac), pick("all", secondary_mac)
    return {**cfg, "cohort_label": out["cohort_label"], "k": out["k"],
            "auto_k": out["auto_k"], "ref_kind": ref_kind,
            "provenance": out.get("provenance"),
            "ols_slope": f10["ols_slope"], "ols_intercept": f10["ols_intercept"],
            "deming_slope": f10["deming_slope"],
            "deming_intercept": f10["deming_intercept"],
            "ols_intercept_mac6": f6["ols_intercept"],
            "deming_intercept_mac6": f6["deming_intercept"],
            "R2": f10["R2"], "RMSE": f10["RMSE"],
            "all_ols_slope": a10["ols_slope"],
            "all_ols_intercept": a10["ols_intercept"],
            "all_deming_slope": a10["deming_slope"],
            "all_deming_intercept": a10["deming_intercept"],
            "all_ols_intercept_mac6": a6["ols_intercept"],
            "all_deming_intercept_mac6": a6["deming_intercept"],
            "all_R2": a10["R2"], "all_RMSE": a10["RMSE"],
            "extrap_pct": out["target"].get("extrap_pct"),
            "q_residual_pct": out["target"].get("q_residual_pct"),
            "negative_pct": out.get("plausibility", {}).get("negative_pct"),
            "above_8_pct": out.get("plausibility", {}).get("above_8_pct"),
            "prediction_median": out.get("plausibility", {}).get("median"),
            "group_medians": out.get("plausibility", {}).get("group_medians", {}),
            "group_median_span": out.get("plausibility", {}).get("group_median_span"),
            "heldout_R2": out["heldout"]["R2"] if out["heldout"] else None}


def _batch_configs(b):
    cohorts = b.get("cohorts") or ["eth_shaped", "analogs", "ocec", "smoke"]
    spectra = b.get("spectra") or ["raw", "airspec", "deriv2"]
    modes = b.get("modes") or ["site_heldout", "app", "app_fmm"]
    lots = b.get("lots") or ["all"]
    # multi-target: one row per (config x target). Targets are the INNERMOST
    # loop so the fitted calibration (fit/curve caches are target-independent)
    # is reused across all sites back-to-back: evaluating a fitted config on
    # another site costs a prediction, not a refit. The evaluation-view levers
    # resolve per target - a lot or season a given site does not have falls back
    # to "all" rather than erroring the row out.
    targets = b.get("targets") or [b.get("target", "addis")]
    eval_lot = b.get("eval_lot", "all")
    # sweeping the split (e.g. ["early", "late"]) scores every configuration on
    # both blind halves in one pass, which is what makes "how far does the winner
    # move between halves" answerable without a second batch
    eval_groups = b.get("eval_groups") or [b.get("eval_group", "all")]
    eval_splits = b.get("eval_splits") or [b.get("eval_split", "all")]
    # one scheme per batch: the group names are only meaningful inside it, so
    # sweeping ["Marine", "Kiremt (Jun-Sep)"] in one pass would be a category
    # error. Targets without the scheme fall back to season + "all" per row.
    group_scheme = b.get("group_scheme", DEFAULT_GROUP_SCHEME)
    match_eval_lot = bool(b.get("match_eval_lot"))
    corrsel = bool(b.get("corrsel"))
    ladder = b.get("cutoff_ladder", True)
    # dense cutoff mode: step through the whole plausible range per ranked
    # cohort in `cutoff_step` intervals instead of the 5-point ladder
    step = int(b.get("cutoff_step") or 0)
    DENSE_RANGES = {"eth_shaped": (100, 600), "analogs": (250, 750),
                    "ocec": (300, 1500)}
    order = {"eth_shaped": 0, "analogs": 1, "ocec": 2, "smoke": 3, "pool": 4}
    cfgs = []
    for co in sorted(cohorts, key=lambda c: order.get(c, 9)):   # cheapest first
        if co in BATCH_LADDERS:
            if step:
                lo, hi = (b.get("cutoff_ranges") or {}).get(co, DENSE_RANGES[co])
                cutoffs = list(range(int(lo), int(hi) + 1, step))
            elif ladder:
                cutoffs = BATCH_LADDERS[co]
            else:
                cutoffs = [DEFAULT_CUTOFF.get(co)]
        else:
            cutoffs = [None]
        sels = (["raw", "airspec"] if corrsel and co in ("eth_shaped", "analogs")
                else ["raw"])
        for cut, sel, sp, mode, lot in itertools.product(
                cutoffs, sels, spectra, modes, lots):
            for tgt, grp, split in itertools.product(targets, eval_groups,
                                                     eval_splits):
                want_lot = (lot if match_eval_lot and lot != "all" else eval_lot)
                use_lot, use_grp, use_split, use_scheme = resolve_eval_view(
                    tgt, want_lot, grp, split, group_scheme)
                cfg = dict(cohort=co, cutoff=cut, selection_space=sel,
                           spectra=sp, mode=mode, lot=lot, target=tgt,
                           eval_lot=use_lot, eval_group=use_grp,
                           eval_split=use_split, group_scheme=use_scheme)
                if cfg not in cfgs:      # two views can collapse to the same row
                    cfgs.append(cfg)
    return cfgs


def _batch_worker(cfgs, sweep, k_min=1, k_max=30, dense_k=False):
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
                            out["auto_k"], int(out["curve"][-1]["n_components"]),
                            k_min=k_min, k_max=k_max, dense=dense_k)))
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
    requested_targets = b.get("targets") or [b.get("target", "addis")]
    blocked = [name for name in requested_targets
               if not target_meta(name)["optimization_allowed"]]
    if blocked:
        return jsonify({"error": "optimization is disabled for locked, provisional, "
                                 f"or unregistered targets: {', '.join(blocked)}"}), 400
    cfgs = _batch_configs(b)
    BATCH.update(running=True, stop=False, done=0, total=len(cfgs), current="",
                 skipped=0, new_rows=0, started=time.time(), finished=None,
                 errors=[], mode="grid")
    threading.Thread(target=_batch_worker,
                     args=(cfgs, bool(b.get("sweep_k", True)),
                           int(b.get("k_min", 1)), int(b.get("k_max", 30)),
                           bool(b.get("dense_k", False))),
                     daemon=True).start()
    return jsonify({"started": True, "total": len(cfgs)})


@app.route("/api/cross_site", methods=["POST"])
def api_cross_site():
    """Evaluate one configuration against every available target: the
    cross-site table (Addis / Bishoftu / Beijing / Delhi / Pasadena / custom)
    as a single call. Slopes come back with the extrapolation diagnostic so
    unreliable (out-of-domain) rows are visibly flagged."""
    if not STATE["ready"]:
        return jsonify({"error": "data still loading"}), 503
    b = request.get_json(force=True) or {}
    cfg = _config_from(b)
    base_view = (cfg.pop("eval_lot", "all"), cfg.pop("eval_group", "all"),
                 cfg.pop("eval_split", "all"),
                 cfg.pop("group_scheme", DEFAULT_GROUP_SCHEME))
    cfg.pop("target", None)
    rows = []
    include_variants = bool(b.get("include_variants", False))
    for name in list_targets(cross_site_only=not include_variants):
        try:
            lot, group, split, scheme = resolve_eval_view(name, *base_view)
            with COMPUTE_LOCK:
                out = run_config(k_override=b.get("k"), **cfg, target=name,
                                 eval_lot=lot, eval_group=group,
                                 eval_split=split, group_scheme=scheme)
            rows.append({"site": name, "label": out["target"]["label"],
                         "k": out["k"], "n": out["target"]["n_eval"],
                         "eval_lot": lot, "eval_group": group,
                         "group_scheme": scheme, "eval_split": split,
                         "metrics": out["metrics"],
                         "extrap_pct": out["target"].get("extrap_pct"),
                         "q_residual_pct": out["target"].get("q_residual_pct"),
                         "heldout_R2": (out["heldout"]["R2"]
                                        if out["heldout"] else None)})
        except Exception as exc:                     # noqa: BLE001
            rows.append({"site": name, "error": f"{type(exc).__name__}: {exc}"})
    return jsonify({"rows": rows})


def _stability_fit(x, y, estimator, mac):
    """Slope/intercept under the same crossplot convention as the explorer."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or not np.isfinite(x * y).all():
        return np.nan, np.nan
    if estimator == "deming":
        result = deming_regression(x, y, deming_lambda(mac) if mac else 1.0)
    else:
        result = regression_metrics(x, y)
    return float(result["slope"]), float(result["intercept"])


def _stability_view(target, eval_lot, evaluation_set, eval_group="all",
                    eval_split="all", group_scheme=DEFAULT_GROUP_SCHEME):
    """Indices/reference for one frozen target readout.

    Shares _eval_indices with run_config so the resampled view is exactly the
    view the leaderboard scored.
    """
    target_indices = _eval_indices(target, eval_lot, eval_group, eval_split,
                                   group_scheme)
    if len(target_indices) < 3:
        raise ValueError("evaluation view has fewer than three target filters")
    if evaluation_set == "fixed" and target.get("fixed_mask") is not None:
        fixed = np.asarray(target["fixed_mask"], bool)[target_indices]
        if fixed.sum() >= 3:
            target_indices = target_indices[fixed]
    return target_indices


@app.route("/api/stability", methods=["POST"])
def api_stability():
    """Conditional winner stability over target filters and source-site refits.

    This deliberately does not call the result an external validation: the
    candidate set and k values are frozen on entry, but the screening target has
    already been inspected.  The independent claim remains the outer source-site
    check plus a new, never-screened lot/target.
    """
    if not STATE["ready"]:
        return jsonify({"error": "data still loading"}), 503
    body = request.get_json(force=True) or {}
    candidates = body.get("candidates") or []
    if not 2 <= len(candidates) <= 8:
        return jsonify({"error": "pass 2–8 frozen finalist candidates"}), 400
    target_name = body.get("target", "addis")
    if not target_meta(target_name)["optimization_allowed"]:
        return jsonify({"error": "stability selection is disabled for locked or "
                                 f"provisional target: {target_name}"}), 400
    n_boot = max(10, min(int(body.get("n_boot", 100)), 500))
    seed = int(body.get("seed", 20260717))
    weight = float(body.get("weight", 5.0))
    estimator = body.get("estimator", "deming")
    if estimator not in ("deming", "ols"):
        return jsonify({"error": "estimator must be deming or ols"}), 400
    mac = float(body.get("mac", 10))
    evaluation_set = body.get("evaluation_set", "fixed")
    eval_lot = body.get("eval_lot", "all")
    eval_group = body.get("eval_group", "all")
    eval_split = body.get("eval_split", "all")
    group_scheme = body.get("group_scheme", DEFAULT_GROUP_SCHEME)

    target = get_target(target_name)
    try:
        view_indices = _stability_view(target, eval_lot, evaluation_set,
                                       eval_group, eval_split, group_scheme)
    except ValueError as exc:
        return jsonify({"error": f"ValueError: {exc}"}), 400
    reference = target["ref"][view_indices]
    x_reference = reference / mac if target["ref_kind"] == "fabs" else reference
    # Bootstrap strata stay the DEFAULT (season) scheme whatever the group lever
    # selects: they exist to keep each resample seasonally representative, which
    # is not something a PMF grouping should quietly redefine.
    target_groups = np.asarray(target["groups"], object)[view_indices]
    filter_draws = stratified_bootstrap_indices(target_groups, n_boot, seed)

    prepared = []
    try:
        with COMPUTE_LOCK:
            for number, raw in enumerate(candidates):
                cfg = {
                    "cohort": raw.get("cohort"),
                    "cutoff": raw.get("cutoff"),
                    "selection_space": raw.get("selection_space", "raw"),
                    "spectra": raw.get("spectra", "raw"),
                    "mode": raw.get("mode", "site_heldout"),
                    "lot": raw.get("lot", "all"),
                    "target": target_name,
                    "eval_lot": eval_lot,
                    "eval_group": eval_group,
                    "eval_split": eval_split,
                    "group_scheme": group_scheme,
                    "max_components": MAX_COMPONENTS,
                }
                k = int(raw.get("k"))
                out = run_config(k_override=k, **cfg)
                ids, label, X, y, sites, X_eval = _cohort_arrays(
                    cfg["cohort"], cfg["cutoff"], cfg["selection_space"],
                    cfg["spectra"], cfg["lot"], target_name,
                )
                train = protocol_train_mask(cfg["mode"], X, y, sites)
                source_sites = np.unique(sites[train])
                site_rows = {
                    site: np.flatnonzero(train & (sites == site)) for site in source_sites
                }
                pred = np.asarray(out["eval"]["pred"], float)
                # run_config already applied eval_lot and fixed is applied below;
                # construct its fixed/all view in the same order.
                fixed = np.asarray(out["eval"]["fixed"], bool)
                if evaluation_set == "fixed" and fixed.sum() >= 3:
                    pred = pred[fixed]
                if len(pred) != len(view_indices):
                    raise ValueError(f"{label}: target view alignment failed")
                prepared.append({
                    "index": number,
                    "label": label,
                    "cfg": cfg,
                    "k": k,
                    "X": X,
                    "y": y,
                    "sites": sites,
                    "X_eval": X_eval,
                    "train_sites": source_sites,
                    "site_rows": site_rows,
                    "pred": pred,
                    "volume": np.asarray(target["volume"], float),
                })
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400

    n_candidates = len(prepared)
    filter_scores = np.full((n_boot, n_candidates), np.nan)
    filter_slopes = np.full_like(filter_scores, np.nan)
    filter_intercepts = np.full_like(filter_scores, np.nan)
    for draw, take in enumerate(filter_draws):
        for candidate, item in enumerate(prepared):
            slope, intercept = _stability_fit(
                x_reference[take], item["pred"][take], estimator, mac
            )
            filter_slopes[draw, candidate] = slope
            filter_intercepts[draw, candidate] = intercept
            filter_scores[draw, candidate] = (
                abs(intercept) + weight * abs(slope - 1)
            )

    source_scores = np.full((n_boot, n_candidates), np.nan)
    source_slopes = np.full_like(source_scores, np.nan)
    source_intercepts = np.full_like(source_scores, np.nan)
    for candidate, item in enumerate(prepared):
        rng = np.random.default_rng(seed + 1009 * (candidate + 1))
        for draw in range(n_boot):
            chosen = rng.choice(
                item["train_sites"], size=len(item["train_sites"]), replace=True
            )
            rows = np.concatenate([item["site_rows"][site] for site in chosen])
            model = PLSRegression(n_components=item["k"], scale=False).fit(
                item["X"][rows], item["y"][rows]
            )
            prediction = model.predict(item["X_eval"]).ravel() / item["volume"]
            prediction = prediction[view_indices]
            slope, intercept = _stability_fit(
                x_reference, prediction, estimator, mac
            )
            source_slopes[draw, candidate] = slope
            source_intercepts[draw, candidate] = intercept
            source_scores[draw, candidate] = (
                abs(intercept) + weight * abs(slope - 1)
            )

    _, filter_frequency = selection_frequency(filter_scores)
    _, source_frequency = selection_frequency(source_scores)

    def rounded(summary):
        return {key: (round(value, 4) if value is not None else None)
                for key, value in summary.items()}

    rows = []
    for candidate, item in enumerate(prepared):
        rows.append({
            "candidate": candidate + 1,
            "label": item["label"],
            "config": {**item["cfg"], "k": item["k"]},
            "n_train_sites": int(len(item["train_sites"])),
            "target_filter_selection_pct": round(float(filter_frequency[candidate]), 1),
            "source_site_selection_pct": round(float(source_frequency[candidate]), 1),
            "target_filter_slope": rounded(percentile_summary(filter_slopes[:, candidate])),
            "target_filter_intercept": rounded(
                percentile_summary(filter_intercepts[:, candidate])
            ),
            "source_site_slope": rounded(percentile_summary(source_slopes[:, candidate])),
            "source_site_intercept": rounded(
                percentile_summary(source_intercepts[:, candidate])
            ),
        })
    return jsonify({
        "rows": rows,
        "n_boot": n_boot,
        "seed": seed,
        "target": target_name,
        "evaluation_set": evaluation_set,
        "estimator": estimator,
        "mac": mac,
        "objective_weight": weight,
        "interpretation": (
            "Conditional stability of a frozen finalist set. Target-filter frequency "
            "resamples within target groups and is descriptive, not independent. "
            "Source-site frequency refits PLS after resampling IMPROVE training sites. "
            "Neither replaces the untouched outer-site/lot confirmation."
        ),
    })


@app.route("/api/batch_stop", methods=["POST"])
def api_batch_stop():
    BATCH["stop"] = True
    return jsonify({"stopping": True})


@app.route("/api/batch_status")
def api_batch_status():
    return jsonify(BATCH)


# ----------------------------------------------------------------------------- #
# analog lab: compare the committed spectral-analog selection against
# literature-style similarity metrics (SAM/cosine, correlation, normalized
# Euclidean, nearest-neighbour cosine), in raw or AIRSpec-corrected space.
# One payload per space: full per-filter rank arrays, so the frontend can move
# the cutoff and recompute overlaps/memberships instantly client-side.
# ----------------------------------------------------------------------------- #
_ANALOG_LAB_CACHE = {}
_ANALOG_LAB_UNIVERSE = {}


def _analog_space_rows(space, ids):
    """Spectra matrix for the given AnalysisIds in the lab's spectra space."""
    if space == "airspec":
        return D["corr_pool"][[D["corr_row"][i] for i in ids]].astype(float)
    X = D["pool_raw"].loc[ids, D["wcols"]].to_numpy(float)
    if space == "deriv2":
        X = savgol_filter(X, axis=1, **SAVGOL)
    return X


def _analog_addis_rows(space):
    if space == "airspec":
        return np.asarray(D["X_addis_corr"], float)
    A = np.asarray(D["X_addis_raw"], float)
    if space == "deriv2":
        A = savgol_filter(A, axis=1, **SAVGOL)
    return A


def _band(M):
    q25, med, q75 = np.percentile(M, [25, 50, 75], axis=0)
    r = lambda v: [round(float(x), 6) for x in v]          # noqa: E731
    return {"q25": r(q25), "median": r(med), "q75": r(q75)}

ANALOG_LAB_METRICS = ["committed", "cosine_median", "corr_median",
                      "eucl_norm_median", "nearest_cosine", "mahalanobis_pca"]
ANALOG_LAB_LABELS = {
    "committed": "committed analog score (ftir_09)",
    "cosine_median": "cosine / SAM vs Addis median",
    "corr_median": "Pearson r vs Addis median (LOCAL's metric)",
    "eucl_norm_median": "Euclidean (L2-normalized) vs Addis median",
    "nearest_cosine": "nearest-neighbour cosine to any Addis filter",
    "mahalanobis_pca": "Mahalanobis to Addis centroid, PCA-10 scores (Reggente'16)",
}


def _analog_lab_payload(space):
    if space in _ANALOG_LAB_CACHE:
        return _ANALOG_LAB_CACHE[space]
    # universe: the committed analog ranking, restricted to TOR-eligible pool
    # filters (and, in corrected space, to filters with an AIRSpec cache row).
    # Its order IS the committed rank.
    pool_ids = set(int(i) for i in D["pool"].index)
    universe = [int(i) for i in D["analog_ranked"] if int(i) in pool_ids]
    if space == "airspec":
        universe = [i for i in universe if i in D["corr_row"]]
        X = D["corr_pool"][[D["corr_row"][i] for i in universe]].astype(float)
        A = np.asarray(D["X_addis_corr"], float)
    elif space == "deriv2":
        # LOCAL's classic representation: correlation on derivative spectra
        # kills the PTFE baseline before similarity is measured
        X = savgol_filter(D["pool_raw"].loc[universe, D["wcols"]]
                          .to_numpy(float), axis=1, **SAVGOL)
        A = savgol_filter(np.asarray(D["X_addis_raw"], float), axis=1, **SAVGOL)
    else:
        X = D["pool_raw"].loc[universe, D["wcols"]].to_numpy(float)
        A = np.asarray(D["X_addis_raw"], float)
    med = np.median(A, axis=0)

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    mn = med / (np.linalg.norm(med) + 1e-12)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Xc = X - X.mean(axis=1, keepdims=True)
    mc = med - med.mean()

    # every metric oriented so LOWER = more Addis-like
    p10 = PCA(n_components=10).fit(Xn)
    S, SA = p10.transform(Xn), p10.transform(An)
    sd = S.std(axis=0) + 1e-12
    vals = {
        "cosine_median": -(Xn @ mn),
        "corr_median": -(Xc @ mc) / (np.linalg.norm(Xc, axis=1)
                                     * np.linalg.norm(mc) + 1e-12),
        "eucl_norm_median": np.linalg.norm(Xn - mn, axis=1),
        "nearest_cosine": -(Xn @ An.T).max(axis=1),
        # the Reggente-2016 extrapolation diagnostic turned into a selector:
        # whitened score-space distance to the Addis centroid
        "mahalanobis_pca": np.linalg.norm((S - SA.mean(axis=0)) / sd, axis=1),
    }
    n = len(universe)
    ranks = {"committed": np.arange(n)}
    for name, v in vals.items():
        ranks[name] = np.argsort(np.argsort(v))

    # Spearman agreement with the committed ranking (rank arrays are already
    # ranks, so plain correlation of ranks = Spearman rho)
    agreement = {name: round(float(np.corrcoef(ranks["committed"], r)[0, 1]), 3)
                 for name, r in ranks.items() if name != "committed"}

    # 2-D PCA of normalized spectra: pool subsample + every Addis filter,
    # fitted on the combined cloud so both live in the same plane
    sample_idx = np.unique(np.linspace(0, n - 1, min(2500, n)).astype(int))
    pca = PCA(n_components=2).fit(np.vstack([Xn[sample_idx], An]))
    pool_xy = pca.transform(Xn[sample_idx])
    addis_xy = pca.transform(An)

    _ANALOG_LAB_UNIVERSE[space] = universe
    payload = {
        "space": space,
        "n": n,
        "metrics": ANALOG_LAB_METRICS,
        "labels": ANALOG_LAB_LABELS,
        "agreement": agreement,
        "ranks": {name: r.astype(int).tolist() for name, r in ranks.items()},
        "sample_idx": sample_idx.astype(int).tolist(),
        "pool_xy": [[round(float(a), 4), round(float(b), 4)] for a, b in pool_xy],
        "addis_xy": [[round(float(a), 4), round(float(b), 4)] for a, b in addis_xy],
        "explained": [round(float(v), 4) for v in pca.explained_variance_ratio_],
    }
    _ANALOG_LAB_CACHE[space] = payload
    return payload


@app.route("/api/analog_lab", methods=["POST"])
def api_analog_lab():
    if not STATE["ready"]:
        return jsonify({"error": "data still loading"}), 503
    b = request.get_json(force=True) or {}
    space = b.get("space", "raw")
    if space not in ("raw", "airspec", "deriv2"):
        return jsonify({"error": f"unknown space '{space}'"}), 400
    with COMPUTE_LOCK:
        try:
            return jsonify(_analog_lab_payload(space))
        except Exception as exc:                     # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


# ----------------------------------------------------------------------------- #
# cross-site target spectra: median + IQR per evaluation target, in a chosen
# baseline space. The baseline choice MATTERS: APRLssb/AIRSpec anchors segment 2
# at the minimum over [1520,1600] cm-1 (see scripts/airspec_baseline.py
# find_min_pos), which sits under a ~1617 band and suppresses it. "neutral" runs
# pybaselines pspline_arpls with no anchor window for an independent opinion.
# ----------------------------------------------------------------------------- #
_SITE_SPEC_CACHE = {}


def _neutral_baseline(wn_asc, Y_asc):
    """Independent penalized-spline baseline; PTFE-saturated regions masked."""
    from pybaselines import Baseline
    bad = ((wn_asc > 1100) & (wn_asc < 1300)) | (wn_asc < 700)
    ww = wn_asc[~bad]
    fitter = Baseline(x_data=ww)
    out = np.empty((Y_asc.shape[0], ww.size))
    for i in range(Y_asc.shape[0]):
        bl, _ = fitter.pspline_arpls(Y_asc[i, ~bad], lam=1e6)
        out[i] = Y_asc[i, ~bad] - bl
    return ww, out


@app.route("/api/site_spectra", methods=["POST"])
def api_site_spectra():
    """Median+IQR spectra for every evaluation target, one baseline space."""
    if not STATE["ready"]:
        return jsonify({"error": "data still loading"}), 503
    b = request.get_json(force=True) or {}
    space = b.get("space", "raw")           # raw | airspec | neutral
    if space not in ("raw", "airspec", "neutral"):
        return jsonify({"error": f"unknown space '{space}'"}), 400
    key = space
    if key in _SITE_SPEC_CACHE:
        return jsonify(_SITE_SPEC_CACHE[key])
    with COMPUTE_LOCK:
        try:
            series = []
            for name in list_targets():
                t = get_target(name)
                if space == "airspec":
                    X = t.get("X_corr")
                    wn = np.asarray(D["corr_wn"], float)
                    if X is None:
                        continue
                    X = np.asarray(X, float)
                else:
                    X = np.asarray(t["X_raw"], float)
                    wn = np.asarray(D["wn_raw"], float)
                o = np.argsort(wn)
                w, Xs = wn[o], X[:, o]
                if space == "neutral":
                    w, Xs = _neutral_baseline(w, Xs)
                q25, med, q75 = np.percentile(Xs, [25, 50, 75], axis=0)
                r = lambda v: [round(float(x), 6) for x in v]   # noqa: E731
                series.append({"name": name, "label": t["label"], "n": int(Xs.shape[0]),
                               "wn": [round(float(v), 2) for v in w],
                               "median": r(med), "q25": r(q25), "q75": r(q75)})
            payload = {"space": space, "series": series}
            _SITE_SPEC_CACHE[key] = payload
            return jsonify(payload)
        except Exception as exc:                     # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


@app.route("/api/analog_lab_spectra", methods=["POST"])
def api_analog_lab_spectra():
    """Median+IQR spectra of the top-`cutoff` cohort under a lab metric,
    alongside the committed cohort and Addis: in the lab's spectra space."""
    if not STATE["ready"]:
        return jsonify({"error": "data still loading"}), 503
    b = request.get_json(force=True) or {}
    space = b.get("space", "raw")
    metric = b.get("metric", "corr_median")
    cutoff = int(b.get("cutoff") or 500)
    if space not in ("raw", "airspec", "deriv2"):
        return jsonify({"error": f"unknown space '{space}'"}), 400
    with COMPUTE_LOCK:
        try:
            pay = _analog_lab_payload(space)
            if metric not in pay["ranks"]:
                return jsonify({"error": f"unknown metric '{metric}'"}), 400
            universe = np.asarray(_ANALOG_LAB_UNIVERSE[space])
            cutoff = min(cutoff, len(universe))
            alt_ids = universe[np.asarray(pay["ranks"][metric]) < cutoff]
            com_ids = universe[:cutoff]        # universe order = committed rank
            wn = D["corr_wn"] if space == "airspec" else np.asarray(D["wn_raw"], float)
            return jsonify({
                "wn": [round(float(v), 2) for v in wn],
                "n": int(cutoff),
                "alt": _band(_analog_space_rows(space, alt_ids.tolist())),
                "committed": _band(_analog_space_rows(space, com_ids.tolist())),
                "addis": _band(_analog_addis_rows(space)),
                "metric_label": ANALOG_LAB_LABELS.get(metric, metric),
            })
        except Exception as exc:                 # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


# ----------------------------------------------------------------------------- #
# cutoff refinement: hill-climb the cohort size in +-`step` intervals around
# each base configuration, following the score downhill until it stops
# improving. Shares the BATCH state/results plumbing, so refined rows land in
# the same leaderboard.
# ----------------------------------------------------------------------------- #
def _refine_bases(b):
    cohorts = [c for c in (b.get("cohorts") or ["eth_shaped", "analogs", "ocec"])
               if c in BATCH_LADDERS]
    spectra = b.get("spectra") or ["airspec"]
    modes = b.get("modes") or ["site_heldout"]
    lots = b.get("lots") or [b.get("lot", "all")]
    corrsel = bool(b.get("corrsel"))
    match_eval_lot = bool(b.get("match_eval_lot"))
    base_eval_lot = b.get("eval_lot", "all")
    bases = []
    for co in cohorts:
        sels = (["raw", "airspec"] if corrsel and co in ("eth_shaped", "analogs")
                else ["raw"])
        for sel, sp, mode, lot in itertools.product(sels, spectra, modes, lots):
            tgt = b.get("target", "addis")
            use_lot, use_grp, use_split, use_scheme = resolve_eval_view(
                tgt, (lot if match_eval_lot and lot != "all" else base_eval_lot),
                b.get("eval_group", "all"), b.get("eval_split", "all"),
                b.get("group_scheme", DEFAULT_GROUP_SCHEME))
            bases.append(dict(cohort=co, cutoff=DEFAULT_CUTOFF.get(co),
                              selection_space=sel, spectra=sp, mode=mode,
                              lot=lot, target=tgt, eval_lot=use_lot,
                              eval_group=use_grp, eval_split=use_split,
                              group_scheme=use_scheme))
    return bases


def _refine_worker(bases, w, min_r2, slope_min, slope_max, max_extrap,
                   max_q_residual, max_negative, step, max_steps):
    seen = set()
    if BATCH_RESULTS_PATH.exists():
        for line in BATCH_RESULTS_PATH.read_text().splitlines():
            try:
                seen.add(_batch_row_key(json.loads(line)))
            except Exception:
                pass
    try:
        with BATCH_RESULTS_PATH.open("a") as f:
            def evaluate(cfg):
                with COMPUTE_LOCK:
                    out = run_config(k_override=None,
                                     max_components=MAX_COMPONENTS, **cfg)
                row = _batch_row(cfg, out)
                key = _batch_row_key(row)
                if key not in seen:
                    seen.add(key)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    BATCH["new_rows"] += 1
                score = abs(row["deming_intercept"]) + w * abs(row["deming_slope"] - 1)
                fails_guardrail = (
                    (row["heldout_R2"] is None or row["heldout_R2"] < min_r2)
                    or row["deming_slope"] < slope_min
                    or row["deming_slope"] > slope_max
                    or (row.get("extrap_pct") is not None
                        and row["extrap_pct"] > max_extrap)
                    or (row.get("q_residual_pct") is not None
                        and row["q_residual_pct"] > max_q_residual)
                    or (row.get("negative_pct") is not None
                        and row["negative_pct"] > max_negative)
                )
                if fails_guardrail:
                    score += 1000.0          # guardrail failure: strongly penalized
                return score

            for bi, base in enumerate(bases, 1):
                if BATCH["stop"]:
                    break
                BATCH["done"] = bi - 1
                tag = (f"{base['cohort']} sel={base['selection_space']} "
                       f"cal={base['spectra']} {base['mode']}")
                scores = {}

                def sc_at(c, base=base, scores=scores, tag=tag):
                    if c is None or c < 50 or c in scores:
                        return scores.get(c)
                    BATCH["current"] = f"refine {tag}: cutoff {c}"
                    try:
                        scores[c] = evaluate({**base, "cutoff": c})
                    except Exception as exc:          # noqa: BLE001
                        scores[c] = None
                        BATCH["errors"] = (BATCH["errors"]
                                           + [f"{tag} @{c}: {exc}"])[-8:]
                    return scores[c]

                cut = base["cutoff"]
                s0 = sc_at(cut)
                if s0 is None:
                    BATCH["skipped"] += 1
                    continue
                up, dn = sc_at(cut + step), sc_at(cut - step)
                dirn, best = None, s0
                if up is not None and up < best:
                    dirn, best = step, up
                if dn is not None and dn < best:
                    dirn, best = -step, dn
                steps = 0
                while dirn and steps < max_steps and not BATCH["stop"]:
                    cut += dirn
                    steps += 1
                    nxt = sc_at(cut + dirn)
                    if nxt is None or nxt >= best:
                        break
                    best = nxt
            else:
                BATCH["done"] = len(bases)
    finally:
        BATCH["running"] = False
        BATCH["current"] = ""
        BATCH["finished"] = time.time()


@app.route("/api/refine_start", methods=["POST"])
def api_refine_start():
    if not STATE["ready"]:
        return jsonify({"error": "data still loading"}), 503
    if BATCH["running"]:
        return jsonify({"error": "a batch/refine job is already running"}), 409
    b = request.get_json(force=True) or {}
    target = b.get("target", "addis")
    if not target_meta(target)["optimization_allowed"]:
        return jsonify({"error": "optimization is disabled for locked, provisional, "
                                 f"or unregistered target: {target}"}), 400
    bases = _refine_bases(b)
    if not bases:
        return jsonify({"error": "no ranked cohorts selected (refine applies to "
                                 "Ethiopia-shaped, analogs and lowest-OC/EC)"}), 400
    BATCH.update(running=True, stop=False, done=0, total=len(bases), current="",
                 skipped=0, new_rows=0, started=time.time(), finished=None,
                 errors=[], mode="refine")
    threading.Thread(target=_refine_worker,
                     args=(bases, float(b.get("w", 5)),
                           float(b.get("min_r2", 0.85)),
                           float(b.get("slope_min", 0.7)),
                           float(b.get("slope_max", 1.3)),
                           float(b.get("max_extrap", 30)),
                           float(b.get("max_q_residual", 30)),
                           float(b.get("max_negative", 10)),
                           int(b.get("step", 10)), int(b.get("max_steps", 40))),
                     daemon=True).start()
    return jsonify({"started": True, "bases": len(bases)})


def _backfill_worker():
    """Upgrade every saved batch row to the full-readout shape by re-running
    its (config, k): cache hits, so this is re-deriving metrics, not
    recomputing calibrations. Rows that already carry the full readout pass
    through untouched; rows that error keep their original form."""
    try:
        lines = (BATCH_RESULTS_PATH.read_text().splitlines()
                 if BATCH_RESULTS_PATH.exists() else [])
        rows = [json.loads(line) for line in lines if line.strip()]
        BATCH["total"] = len(rows)
        out_rows = []
        for i, r in enumerate(rows, 1):
            BATCH["done"] = i - 1
            if (BATCH["stop"] or
                    ("all_deming_slope" in r and "negative_pct" in r
                     and "q_residual_pct" in r)):
                out_rows.append(r)
                continue
            cfg = {k: (r.get(k) or d) for k, d in
                   (("cohort", None), ("cutoff", None),
                    ("selection_space", "raw"), ("spectra", "raw"),
                    ("mode", "site_heldout"), ("lot", "all"),
                    ("target", "addis"), ("eval_lot", "all"),
                    ("eval_group", "all"), ("eval_split", "all"),
                    ("group_scheme", DEFAULT_GROUP_SCHEME))}
            BATCH["current"] = (f"backfill {i}/{len(rows)}: {cfg['cohort']}/"
                                f"{cfg['cutoff'] or '-'} k={r.get('k')}")
            try:
                with COMPUTE_LOCK:
                    o = run_config(k_override=r.get("k"),
                                   max_components=MAX_COMPONENTS, **cfg)
                out_rows.append(_batch_row(cfg, o))
                BATCH["new_rows"] += 1
            except Exception as exc:                     # noqa: BLE001
                BATCH["skipped"] += 1
                BATCH["errors"] = (BATCH["errors"]
                                   + [f"{BATCH['current']}: {exc}"])[-8:]
                out_rows.append(r)
        BATCH["done"] = len(rows)
        tmp = BATCH_RESULTS_PATH.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in out_rows))
        tmp.replace(BATCH_RESULTS_PATH)
    finally:
        BATCH["running"] = False
        BATCH["current"] = ""
        BATCH["finished"] = time.time()


@app.route("/api/batch_backfill", methods=["POST"])
def api_batch_backfill():
    if not STATE["ready"]:
        return jsonify({"error": "data still loading"}), 503
    if BATCH["running"]:
        return jsonify({"error": "a batch/refine job is already running"}), 409
    BATCH.update(running=True, stop=False, done=0, total=0, current="",
                 skipped=0, new_rows=0, started=time.time(), finished=None,
                 errors=[], mode="backfill")
    threading.Thread(target=_backfill_worker, daemon=True).start()
    return jsonify({"started": True})


@app.route("/api/batch_results")
def api_batch_results():
    target = request.args.get("target")
    require_heldout = request.args.get("require_heldout") == "1"
    limit = max(1, min(int(request.args.get("limit", 10000)), 50000))
    filters = {}
    for field in ("cohort", "spectra", "mode", "lot"):
        raw = request.args.get(field)
        if raw:
            filters[field] = set(raw.split(","))
    rows = []
    total = matched = 0
    if BATCH_RESULTS_PATH.exists():
        for line in BATCH_RESULTS_PATH.read_text().splitlines():
            try:
                row = json.loads(line)
                total += 1
                if target and row.get("target", "addis") != target:
                    continue
                if require_heldout and row.get("heldout_R2") is None:
                    continue
                if any(str(row.get(field, "all")) not in allowed
                       for field, allowed in filters.items()):
                    continue
                matched += 1
                if len(rows) < limit:
                    rows.append(row)
            except Exception:
                pass
    return jsonify({"rows": rows, "total": total, "matched": matched,
                    "limit": limit, "truncated": matched > len(rows)})


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
        eval_group=b.get("eval_group", "all"),
        eval_split=b.get("eval_split", "all"),
        group_scheme=b.get("group_scheme", DEFAULT_GROUP_SCHEME),
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
                    "targets": list_targets() if STATE["ready"] else {},
                    "target_meta": ({name: target_meta(name) for name in list_targets()}
                                    if STATE["ready"] else {}),
                    "provenance": {"cache_schema": CACHE_SCHEMA_VERSION,
                                   "git_commit": GIT_COMMIT,
                                   "git_dirty": GIT_DIRTY,
                                   "source_fingerprint": D.get("source_fingerprint")}})


@app.route("/api/eval_view_options", methods=["POST"])
def api_eval_view_options():
    """Which evaluation-view levers this target supports, with their counts.

    Lots and season/group names are per site, so the page asks per target
    instead of carrying one global list (which was Addis-only).
    """
    if not STATE["ready"]:
        return jsonify({"error": "data still loading"}), 503
    name = (request.get_json(force=True) or {}).get("target", "addis")
    try:
        options = eval_view_options(get_target(name))
    except Exception as exc:                              # noqa: BLE001
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
    return jsonify({"target": name, **options})


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
    """Scan an explicit component list (the UI now includes k=21 through 30)."""
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
                             "extrap_pct": out["target"].get("extrap_pct"),
                             "q_residual_pct": out["target"].get("q_residual_pct"),
                             "negative_pct": out.get("plausibility", {}).get("negative_pct"),
                             "above_8_pct": out.get("plausibility", {}).get("above_8_pct"),
                             "prediction_median": out.get("plausibility", {}).get("median"),
                             "group_medians": out.get("plausibility", {}).get("group_medians", {}),
                             "group_median_span": out.get("plausibility", {}).get("group_median_span"),
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
    # with the Addis FTIR-derived OC/EC marker (ftir_30: 1.34; not thermal -
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


# bare imports resolve when app.py runs as a script from its own directory;
# the package fallback covers importlib.import_module("calibration_explorer.app")
# (the Colab launcher) where calibration_explorer/ itself is not on sys.path
try:
    import hips_lab                                    # noqa: E402
except ModuleNotFoundError:
    from calibration_explorer import hips_lab          # noqa: E402
hips_lab.register(app, {
    "STATE": STATE, "COMPUTE_LOCK": COMPUTE_LOCK, "run_config": run_config,
    "list_targets": list_targets, "get_target": get_target,
    "_config_from": _config_from,
    "spartan_hips_path": PATHS.spartan_hips_primary,
})

try:
    import local_lab                                   # noqa: E402
except ModuleNotFoundError:
    from calibration_explorer import local_lab         # noqa: E402
local_lab.register(app, {
    "STATE": STATE, "COMPUTE_LOCK": COMPUTE_LOCK, "D": D,
    "get_target": get_target, "_target_X": _target_X,
    "crossplot_metrics": crossplot_metrics, "_cache_key": _cache_key,
    "CACHE_DIR": CACHE_DIR, "SAVGOL": SAVGOL, "list_targets": list_targets,
})


if __name__ == "__main__":
    _port = int(os.environ.get("PORT", 5058))
    print(f"Calibration Iteration Explorer → http://127.0.0.1:{_port}")
    app.run(host="127.0.0.1", port=_port, debug=False, threaded=True)
