"""Apply the frozen full-pool AIRSpec/VIBES models to Adama and Bishoftu.

Run from the repository root with
``uv run --locked --no-sync python research/ftir_hips_chem/workflows/run_adama_bishoftu_frozen_pilot.py``.
This is a diagnostic external application, not model selection or site EC
validation. Adama's spectrum-to-filter crosswalk and date-paired samplers are
provisional; Bishoftu has no thermal EC reference.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "research/ftir_hips_chem"
PHASE3 = ROOT / "research/ftir_ec_phase3"
sys.path[:0] = [str(ANALYSIS / "scripts"), str(PHASE3 / "scripts")]

from airspec_baseline import airspec_baseline_matrix  # noqa: E402
from config import MAC_VALUE  # noqa: E402
from outliers import apply_exclusion_flags, get_clean_data  # noqa: E402
from theory_test_suite import davis_root  # noqa: E402
from vibes_baseline import VibesBackground, vibes_baseline_matrix  # noqa: E402

RUN = ANALYSIS / "output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09"
OUT = ANALYSIS / "output/tables/adama_bishoftu_frozen_pilot"
OUT.mkdir(parents=True, exist_ok=True)
ADAMA_SPECTRA = davis_root() / "DAVIS/CSU_AMOD/csu_amod_Batch_54_ShipDate_2026-05-29_spectra.csv"
ADAMA_CARBON = davis_root() / "DAVIS/Adama TOR/Carbon_concs_Batch54.csv"
ADAMA_LEDGER = PHASE3 / "output/tables/ftir41/pairing_ledger.csv"
ETBI_SPECTRA = ROOT / "calibration_explorer/targets/etbi/spectra.csv"
ETBI_REFERENCE = ROOT / "calibration_explorer/targets/etbi/reference.csv"
SOURCE_FILES = [ADAMA_SPECTRA, ADAMA_CARBON, ADAMA_LEDGER, ETBI_SPECTRA,
                ETBI_REFERENCE, RUN / "background_model.npz", RUN / "wn.npy",
                RUN / "cases.csv", RUN / "addis_predictions.csv",
                RUN / "corrected_AIRSpec.npy", RUN / "corrected_VIBES.npy",
                RUN / "pls_full_pool_AIRSpec.npz", RUN / "pls_full_pool_VIBES.npz",
                RUN / "RUN_MANIFEST.json", Path(__file__),
                PHASE3 / "scripts/airspec_baseline.py",
                ANALYSIS / "scripts/vibes_baseline.py"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def on_grid(frame: pd.DataFrame, id_col: str, wn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select measured channels only; never interpolate a spectrum for a model."""
    columns = [c for c in frame if c != id_col]
    source_wn = np.array([float(c) for c in columns])
    indices = np.abs(source_wn[:, None] - wn[None, :]).argmin(axis=0)
    error = np.abs(source_wn[indices] - wn)
    if len(set(indices)) != len(wn) or error.max() > 1e-3:
        raise ValueError(f"Spectral grid mismatch: max channel error {error.max():g}")
    values = frame[columns].to_numpy(float)[:, indices]
    if not np.isfinite(values).all():
        raise ValueError("Non-finite external spectrum; inspect before prediction")
    return frame[id_col].to_numpy(), values


def predict_loading(corrected: np.ndarray, model_path: Path) -> np.ndarray:
    with np.load(model_path) as model:
        coefficient = model["coefficient"]
        x_mean = model["x_mean"]
        y_mean = model["y_mean"]
    if corrected.shape[1] != len(x_mean) or coefficient.shape != (1, len(x_mean)):
        raise ValueError("Frozen PLS model/spectrum dimension mismatch")
    return ((corrected - x_mean) @ coefficient.T + y_mean).ravel()


def correct_external(wn: np.ndarray, raw: np.ndarray, ids: list[str], bg: VibesBackground,
                     cache_key: str) -> dict[str, np.ndarray]:
    cache = OUT / "external_corrected.npz"
    if cache.exists():
        with np.load(cache) as saved:
            if str(saved["key"]) == cache_key and list(saved["ids"]) == ids:
                print(f"Reusing {len(ids)} frozen external corrections", flush=True)
                return {name: saved[name] for name in ("AIRSpec", "VIBES")}
    air = np.empty_like(raw)
    vib = np.empty_like(raw)
    diagnostics = []
    for i, (sid, row) in enumerate(zip(ids, raw)):
        _, air[i:i + 1] = airspec_baseline_matrix(wn, row[None], df1=6, df2=4)
        _, vib[i:i + 1], diag = vibes_baseline_matrix(
            wn, row[None], bg, sample_ids=[sid], tau=.1, loss="PB", maxiter=10000
        )
        diagnostics.append(diag.iloc[0].to_dict())
        print(f"Corrected {i + 1}/{len(ids)}: {sid}", flush=True)
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics.to_csv(OUT / "external_fit_diagnostics.csv", index=False)
    if not diagnostics.success.all() or not np.isfinite(vib).all():
        raise RuntimeError("External VIBES correction failed; inspect fit diagnostics")
    np.savez_compressed(cache, key=cache_key, ids=np.array(ids), AIRSpec=air, VIBES=vib)
    return {"AIRSpec": air, "VIBES": vib}


def band_height(values: np.ndarray, wn: np.ndarray, center: float,
                left: float, right: float) -> np.ndarray:
    """Mean 11-channel peak above a line between two local shoulder windows."""
    def window(x: float) -> tuple[float, np.ndarray]:
        mask = np.abs(wn - x) <= 7
        if mask.sum() < 3:
            raise ValueError(f"Missing spectral feature window {x}")
        return float(wn[mask].mean()), values[:, mask].mean(axis=1)
    x, y = window(center)
    xl, yl = window(left)
    xr, yr = window(right)
    return y - (yl + (yr - yl) * (x - xl) / (xr - xl))


def adama_table(ids: np.ndarray, predictions: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger = pd.read_csv(ADAMA_LEDGER, parse_dates=["date"])
    if len(ledger) != 5 or ledger.date.duplicated().any():
        raise ValueError("Expected five unique Adama candidate sampling dates")
    # The historical ftir_44 mapping is a hypothesis from sorted FilterIds,
    # not an exported SampleAnalysisId→FilterId crosswalk.
    sorted_ids = np.sort(ids.astype(int))
    sorted_filters = np.sort(ledger.FilterId_ptfe.to_numpy(str))
    provisional = dict(zip(sorted_ids, sorted_filters))
    inverse = {v: k for k, v in provisional.items()}
    ledger["SampleAnalysisId_provisional"] = ledger.FilterId_ptfe.map(inverse)
    if ledger.SampleAnalysisId_provisional.isna().any():
        raise ValueError("Adama candidate ID map incomplete")
    carbon = pd.read_csv(ADAMA_CARBON)
    carbon["date"] = pd.to_datetime(carbon.SampleDate).dt.normalize()
    carbon = carbon.pivot_table(index="date", columns="Parameter",
                                values="Concentration_ug_m3", aggfunc="first").reset_index()
    ledger = ledger.merge(carbon[["date", "ECTR", "ECTT"]], on="date", validate="one_to_one")
    ledger = apply_exclusion_flags(ledger, "Adama")
    if len(get_clean_data(ledger)) != 5:
        raise ValueError("Registered exclusions changed the five-pair Adama pilot")
    for method, mass in predictions.items():
        by_id = dict(zip(ids.astype(int), mass))
        ledger[f"{method}_loading_ug"] = ledger.SampleAnalysisId_provisional.map(by_id)
        ledger[f"{method}_EC_ugm3"] = ledger[f"{method}_loading_ug"] / ledger.Volume_m3
        for ref in ("ECTR", "ECTT"):
            ledger[f"{method}_over_{ref}"] = ledger[f"{method}_EC_ugm3"] / ledger[ref]
    ledger["pairing_status"] = "date candidate; sampler equivalence and spectrum ID map unconfirmed"
    ledger.to_csv(OUT / "adama_candidate_pairs.csv", index=False)

    # Vary *every* possible spectrum-to-filter assignment. This is a mapping
    # sensitivity bound, not a probability interval or alternate selected map.
    rows = []
    for method, mass in predictions.items():
        by_id = dict(zip(ids.astype(int), mass))
        for perm in itertools.permutations(sorted_ids):
            assignment = dict(zip(sorted_filters, perm))
            concentration = np.array([
                by_id[assignment[f]] / volume
                for f, volume in zip(ledger.FilterId_ptfe, ledger.Volume_m3)
            ])
            for reference in ("ECTR", "ECTT"):
                ratio = concentration / ledger[reference].to_numpy(float)
                for scope, mask in (("all_five", np.ones(5, bool)),
                                    ("three_sampling_unflagged", ledger.flag.isna().to_numpy())):
                    rows.append({"method": method, "reference": reference,
                                 "scope": scope, "mapping": ",".join(map(str, perm)),
                                 "median_prediction_over_reference": float(np.median(ratio[mask])),
                                 "mean_signed_error_ugm3": float(np.mean(
                                     concentration[mask] - ledger.loc[mask, reference]))})
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(OUT / "adama_id_mapping_sensitivity.csv", index=False)
    return ledger, sensitivity


def matched_sites(wn: np.ndarray, external: dict[str, np.ndarray],
                  external_ids: np.ndarray, predictions: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = pd.read_csv(ETBI_REFERENCE)
    if not ref.MediaId.is_unique:
        raise ValueError("Bishoftu reference repeats a physical filter")
    etbi = ref.set_index("MediaId").loc[external_ids.astype(int)].reset_index()
    etbi["date"] = pd.to_datetime(etbi.Date)
    etbi = apply_exclusion_flags(etbi, "Bishoftu")
    if len(get_clean_data(etbi)) != len(etbi):
        raise ValueError("Registered Bishoftu exclusions require review")
    etbi["HIPS_EC_equivalent_ugm3"] = etbi.Fabs / MAC_VALUE
    for method, mass in predictions.items():
        etbi[f"{method}_EC_ugm3"] = mass / etbi.Volume_m3.to_numpy(float)
        etbi[f"{method}_over_HIPS_equivalent"] = (
            etbi[f"{method}_EC_ugm3"] / etbi.HIPS_EC_equivalent_ugm3
        )
        etbi[f"{method}_band1617"] = band_height(external[method], wn, 1617, 1750, 1500)
        etbi[f"{method}_band2920"] = band_height(external[method], wn, 2920, 3050, 2800)
    etbi.to_csv(OUT / "bishoftu_frozen_predictions.csv", index=False)

    cases = pd.read_csv(RUN / "cases.csv")
    targets = cases[cases.kind.eq("target")].copy()
    targets["date"] = pd.to_datetime(targets.date)
    targets = apply_exclusion_flags(targets, "Addis_Ababa")
    targets = get_clean_data(targets)
    saved = pd.read_csv(RUN / "addis_predictions.csv")
    saved = saved[saved.cohort.eq("full_pool")]
    for method in ("AIRSpec", "VIBES"):
        rows = saved[saved.method.eq(method)][["sample_id", "prediction_ugm3"]]
        targets = targets.merge(rows.rename(columns={"prediction_ugm3": f"{method}_EC_ugm3"}),
                                on="sample_id", validate="one_to_one")
        corrected = np.load(RUN / f"corrected_{method}.npy", mmap_mode="r")
        if len(corrected) != len(cases):
            raise ValueError("Saved Addis corrected spectra/case order mismatch")
        index = cases.index[cases.kind.eq("target")]
        features = pd.DataFrame({"sample_id": cases.loc[index, "sample_id"],
                                 f"{method}_band1617": band_height(corrected[index], wn, 1617, 1750, 1500),
                                 f"{method}_band2920": band_height(corrected[index], wn, 2920, 3050, 2800)})
        targets = targets.merge(features, on="sample_id", validate="one_to_one")
    targets["HIPS_EC_equivalent_ugm3"] = targets.Fabs / MAC_VALUE
    for method in ("AIRSpec", "VIBES"):
        targets[f"{method}_over_HIPS_equivalent"] = (
            targets[f"{method}_EC_ugm3"] / targets.HIPS_EC_equivalent_ugm3
        )
    if len(targets) != 253 or not targets.sample_id.is_unique:
        raise ValueError("Frozen Addis target cohort changed; review before export")
    targets.to_csv(OUT / "addis_frozen_predictions.csv", index=False)

    # October–December and lot 251 are shared context, not concurrent dates.
    addis = targets[targets.lot.eq(251) & targets.date.dt.month.isin((10, 11, 12))
                    & targets.Fabs.notna()].copy()
    bish = etbi[etbi.LotId.eq(251) & etbi.date.dt.month.isin((10, 11, 12))
                & etbi.Fabs.notna()].copy()
    lower = max(addis.Fabs.min(), bish.Fabs.min())
    upper = min(addis.Fabs.max(), bish.Fabs.max())
    addis = addis[addis.Fabs.between(lower, upper)].reset_index(drop=True)
    bish = bish[bish.Fabs.between(lower, upper)].reset_index(drop=True)
    if addis.empty or bish.empty:
        raise ValueError("No lot/month HIPS loading overlap between Addis and Bishoftu")
    # One-to-one nearest loading, ≤20% symmetric log ratio. Dummy columns
    # favor the greatest number of valid matches, then minimize mismatch.
    cost = abs(np.log(bish.Fabs.to_numpy()[:, None] / addis.Fabs.to_numpy()[None, :]))
    allowed = cost <= np.log(1.2)
    assignment = np.c_[np.where(allowed, cost, 100.0),
                       np.ones((len(bish), len(bish)))]
    r, c = linear_sum_assignment(assignment)
    selected = [(i, j) for i, j in zip(r, c) if j < len(addis) and allowed[i, j]]
    if not selected:
        raise ValueError("No pairs satisfy the fixed ±20% optical-loading caliper")
    rows = []
    for i, j in selected:
        b, a = bish.iloc[i], addis.iloc[j]
        row = {"bishoftu_filter_id": b.ExternalFilterId,
               "addis_filter_id": a.filter_id,
               "bishoftu_date": str(b.date.date()), "addis_date": str(a.date.date()),
               "bishoftu_Fabs_Mm1": b.Fabs, "addis_Fabs_Mm1": a.Fabs,
               "Fabs_ratio": b.Fabs / a.Fabs,
               "lot": 251, "months": "Oct-Dec", "match_caliper": "20% log ratio"}
        for method in ("AIRSpec", "VIBES"):
            for key in ("EC_ugm3", "over_HIPS_equivalent", "band1617", "band2920"):
                row[f"bishoftu_{method}_{key}"] = b[f"{method}_{key}"]
                row[f"addis_{method}_{key}"] = a[f"{method}_{key}"]
        rows.append(row)
    matched = pd.DataFrame(rows)
    matched.to_csv(OUT / "addis_bishoftu_loading_matched.csv", index=False)
    return etbi, matched


def main() -> None:
    for path in SOURCE_FILES:
        if not path.exists():
            raise FileNotFoundError(path)
    wn = np.load(RUN / "wn.npy")
    with np.load(RUN / "background_model.npz") as z:
        if not np.array_equal(wn, z["wn"]):
            raise ValueError("Saved VIBES background grid differs from frozen PLS grid")
        bg = VibesBackground(wn, z["mean"], z["components"], pd.DataFrame(),
                             tuple(z["blank_ids"].astype(str)), 0.0)
    adama_raw = pd.read_csv(ADAMA_SPECTRA).rename(columns={"Wavelength": "SampleAnalysisId"})
    adama_ids, adama_x = on_grid(adama_raw, "SampleAnalysisId", wn)
    etbi_raw = pd.read_csv(ETBI_SPECTRA)
    etbi_ids, etbi_x = on_grid(etbi_raw, "MediaId", wn)
    if len(adama_ids) != 5 or len(etbi_ids) != 26:
        raise ValueError("External filter cohort size changed; review fixed pilot")
    ids = [f"adama:{int(i)}" for i in adama_ids] + [f"etbi:{int(i)}" for i in etbi_ids]
    raw = np.vstack([adama_x, etbi_x])
    key = hashlib.sha256(json.dumps({str(p): sha256(p) for p in
                                     (ADAMA_SPECTRA, ETBI_SPECTRA, RUN / "background_model.npz",
                                      RUN / "wn.npy", PHASE3 / "scripts/airspec_baseline.py",
                                      ANALYSIS / "scripts/vibes_baseline.py")},
                                    sort_keys=True).encode()).hexdigest()
    corrected = correct_external(wn, raw, ids, bg, key)
    models = {m: RUN / f"pls_full_pool_{m}.npz" for m in ("AIRSpec", "VIBES")}
    # Independently reproduce one already-saved Addis prediction as a portable
    # coefficient/orientation check before applying either model externally.
    cases = pd.read_csv(RUN / "cases.csv")
    saved = pd.read_csv(RUN / "addis_predictions.csv")
    target_idx = int(cases.index[cases.kind.eq("target")][0])
    sid = cases.loc[target_idx, "sample_id"]
    for method, model in models.items():
        prior = saved[(saved.sample_id.eq(sid)) & saved.method.eq(method)
                      & saved.cohort.eq("full_pool")].prediction_ugm3.iloc[0]
        spectrum = np.load(RUN / f"corrected_{method}.npy", mmap_mode="r")[target_idx:target_idx + 1]
        replay = predict_loading(spectrum, model)[0] / cases.loc[target_idx, "volume"]
        if not np.isclose(prior, replay, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Frozen {method} model fails saved Addis prediction replay")
    pred_adama = {m: predict_loading(x[:5], models[m]) for m, x in corrected.items()}
    pred_etbi = {m: predict_loading(x[5:], models[m]) for m, x in corrected.items()}
    adama, sensitivity = adama_table(adama_ids, pred_adama)
    etbi_corr = {m: x[5:] for m, x in corrected.items()}
    bish, matched = matched_sites(wn, etbi_corr, etbi_ids, pred_etbi)
    report = {
        "frozen_run_signature": json.loads((RUN / "RUN_MANIFEST.json").read_text())["signature"],
        "n_adama_candidate_pairs": len(adama),
        "n_adama_sampling_unflagged": int(adama.flag.isna().sum()),
        "n_bishoftu_external": len(bish),
        "n_addis_frozen_targets": 253,
        "n_addis_bishoftu_matched": len(matched),
        "source_sha256": {str(p): sha256(p) for p in SOURCE_FILES},
        "model_selection": "frozen full-pool AIRSpec and VIBES; no external refit or tuning",
        "adama_reference": "quartz TOR/TOT EC concentration on date-candidate co-samples",
        "adama_mapping": "sorted analysis IDs to sorted PTFE FilterIds, provisional ftir_44 hypothesis",
        "adama_sensitivity": "all 120 spectrum-to-PTFE assignments, not a confidence interval",
        "bishoftu_reference": "HIPS Fabs/MAC optical equivalent, not independent EC",
        "site_match": "lot 251, Oct-Dec, common HIPS Fabs support, one-to-one nearest, <=20% log ratio",
    }
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, indent=2))
    for method in models:
        print(method, "Adama provisional TOR median ratio (all/unflagged):",
              round(float(adama[f"{method}_over_ECTR"].median()), 3),
              round(float(adama.loc[adama.flag.isna(), f"{method}_over_ECTR"].median()), 3))
        print(method, "Bishoftu/Addis matched FTIR prediction medians:",
              round(float(matched[f"bishoftu_{method}_EC_ugm3"].median()), 3),
              round(float(matched[f"addis_{method}_EC_ugm3"].median()), 3))


if __name__ == "__main__":
    main()
