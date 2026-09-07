"""Reusable analyses for the phase-3 theory-test notebook suite.

The functions in this module deliberately keep source-data loading, quality
ledgers, statistical tests, and plotting-facing tables separate.  The reader-
facing notebooks ``ftir_37``--``ftir_40`` call these functions; no target label
is used to select a PLS model or an applicability-domain threshold.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import spearmanr, wilcoxon
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
P3 = HERE.parent
REPO = HERE.parents[2]
P2_SCRIPTS = REPO / "research/ftir_hips_chem/scripts"
if str(P2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(P2_SCRIPTS))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from calibration_modes import protocol_train_mask  # noqa: E402
from data_matching import base_filter_id, load_filter_data  # noqa: E402
from domain_invariant_pls import block_average  # noqa: E402
from phase3_common import (  # noqa: E402
    PATHS,
    load_pool_metadata,
    load_pool_spectra,
    load_tor_loadings,
)


DEFAULT_DAVIS_ROOT = Path(
    "/Users/ahmadjalil/Library/CloudStorage/"
    "GoogleDrive-ahzs645@gmail.com/My Drive/University/Research/Grad/Data/Davis Data"
)


def davis_root() -> Path:
    """Resolve the Davis source-data root without silently changing datasets."""
    root = Path(os.environ.get("AETHMODULAR_DAVIS_DIR", DEFAULT_DAVIS_ROOT)).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"Davis data root not found: {root}. Set AETHMODULAR_DAVIS_DIR."
        )
    return root


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(observed)) ** 2)))


def _site_balanced_rmse(frame: pd.DataFrame, residual_col: str) -> float:
    per_site = frame.groupby("Site")[residual_col].apply(
        lambda value: float(np.mean(np.square(value)))
    )
    return float(np.sqrt(per_site.mean()))


def load_improve_hips_tor_bridge() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join IMPROVE HIPS Fabs to TOR EC/OC with an explicit quality ledger."""
    tables = davis_root() / "FTIR/local_db/tables"
    hips_raw = pd.read_csv(
        tables / "results_hips.csv",
        usecols=["MatchedFilterId", "Parameter", "Value"],
        low_memory=False,
    )
    hips = hips_raw[hips_raw["Parameter"].str.casefold().eq("fabs")].copy()
    hips_duplicates = int(hips.duplicated("MatchedFilterId", keep=False).sum())
    hips = (
        hips.sort_values("MatchedFilterId")
        .drop_duplicates("MatchedFilterId", keep="first")
        .rename(columns={"MatchedFilterId": "FilterId", "Value": "Fabs_Mm1"})
    )

    catalog = pd.read_csv(
        tables / "ftir_catalog.csv", usecols=["FilterId", "SampleDate", "Site"]
    ).drop_duplicates("FilterId")
    catalog["date"] = pd.to_datetime(
        catalog["SampleDate"], format="mixed", errors="coerce"
    ).dt.normalize()

    tor = pd.read_csv(
        tables / "results_tor.csv",
        usecols=["Site", "SampleDate", "Parameter", "Value"],
        low_memory=False,
    )
    tor = tor[tor["Parameter"].isin(["EC", "OC"])].copy()
    tor["date"] = pd.to_datetime(
        tor["SampleDate"], format="mixed", errors="coerce"
    ).dt.normalize()
    tor_key_duplicates = int(tor.duplicated(["Site", "date", "Parameter"]).sum())
    tor_wide = (
        tor.drop_duplicates(["Site", "date", "Parameter"], keep="first")
        .pivot(index=["Site", "date"], columns="Parameter", values="Value")
        .reset_index()
    )

    step1 = hips.merge(catalog[["FilterId", "Site", "date"]], on="FilterId", how="left")
    bridge = step1.merge(tor_wide, on=["Site", "date"], how="left", validate="many_to_one")
    bridge["EC_ugm3"] = bridge["EC"] / 1000.0
    bridge["OC_ugm3"] = bridge["OC"] / 1000.0
    bridge["OC_EC"] = bridge["OC"] / bridge["EC"]
    bridge["implied_MAC"] = bridge["Fabs_Mm1"] / bridge["EC_ugm3"]
    positive = (
        bridge["Fabs_Mm1"].gt(0)
        & bridge["EC_ugm3"].gt(0)
        & bridge["OC_ugm3"].gt(0)
        & np.isfinite(bridge["OC_EC"])
    )
    clean = bridge[positive].copy()

    ledger = pd.DataFrame(
        [
            {"stage": "HIPS fAbs rows", "rows": len(hips_raw), "unique_filters": hips_raw["MatchedFilterId"].nunique()},
            {"stage": "HIPS first result per filter", "rows": len(hips), "unique_filters": hips["FilterId"].nunique()},
            {"stage": "catalog match", "rows": len(step1), "unique_filters": step1.loc[step1["Site"].notna(), "FilterId"].nunique()},
            {"stage": "TOR EC+OC match", "rows": len(bridge), "unique_filters": bridge.loc[bridge[["EC", "OC"]].notna().all(axis=1), "FilterId"].nunique()},
            {"stage": "positive finite analysis set", "rows": len(clean), "unique_filters": clean["FilterId"].nunique()},
        ]
    )
    ledger["hips_duplicate_rows_all_occurrences"] = hips_duplicates
    ledger["tor_duplicate_site_date_parameter_rows"] = tor_key_duplicates
    return clean, ledger


def hips_blank_domain() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Locate loaded SPARTAN filters relative to their lot/calibration blank domain."""
    path = davis_root() / "Spartan/SPARTAN_HIPS_Batch1-51.v2.csv"
    use = [
        "FilterId", "Site", "FilterType", "LotId", "T1", "R1", "Intercept",
        "Slope", "tau", "DepositArea", "Volume", "Fabs", "Uncertainty",
    ]
    raw = pd.read_csv(path, encoding="cp1252", usecols=use, low_memory=False)
    lot_key = raw["LotId"].map(
        lambda value: str(int(float(value)))
        if pd.notna(value) and str(value).replace(".", "", 1).isdigit()
        else str(value).strip()
    )
    raw["cal_key"] = list(
        zip(
            lot_key,
            raw["Intercept"].round(3),
            raw["Slope"].round(4),
        )
    )
    blanks = raw[raw["FilterType"].isin(["FB", "LB"])].dropna(subset=["R1", "T1"])
    blank_ranges = (
        blanks.groupby("cal_key")
        .agg(blank_n=("R1", "size"), blank_r1_min=("R1", "min"), blank_r1_max=("R1", "max"))
        .reset_index()
    )
    sample = raw[raw["FilterType"].eq("PM2.5")].drop_duplicates("FilterId").copy()
    sample = sample.merge(blank_ranges, on="cal_key", how="left", validate="many_to_one")
    width = (sample["blank_r1_max"] - sample["blank_r1_min"]).replace(0, np.nan)
    sample["r1_position"] = (sample["R1"] - sample["blank_r1_min"]) / width
    sample["below_blank_domain"] = sample["R1"] < sample["blank_r1_min"]
    sample["above_blank_domain"] = sample["R1"] > sample["blank_r1_max"]
    sample["outside_blank_domain"] = sample["below_blank_domain"] | sample["above_blank_domain"]
    sample["Fabs_quintile"] = pd.Series(pd.NA, index=sample.index, dtype="object")
    finite_fabs = np.isfinite(sample["Fabs"]) & sample["Fabs"].notna()
    sample.loc[finite_fabs, "Fabs_quintile"] = pd.qcut(
        sample.loc[finite_fabs, "Fabs"], 5, duplicates="drop"
    ).astype(str)
    loading = (
        sample.groupby("Fabs_quintile", observed=True)
        .agg(
            n=("FilterId", "size"),
            Fabs_median=("Fabs", "median"),
            Fabs_min=("Fabs", "min"),
            Fabs_max=("Fabs", "max"),
            below_blank_pct=("below_blank_domain", lambda v: 100 * v.mean()),
            outside_blank_pct=("outside_blank_domain", lambda v: 100 * v.mean()),
            r1_position_median=("r1_position", "median"),
        )
        .reset_index(drop=True)
        .sort_values("Fabs_median")
        .reset_index(drop=True)
    )
    quality = pd.DataFrame(
        [{
            "source_rows": len(raw),
            "sample_filters": len(sample),
            "blank_rows": len(blanks),
            "calibration_sets_with_blanks": len(blank_ranges),
            "sample_blank_domain_coverage_pct": 100 * sample["blank_n"].notna().mean(),
            "sample_outside_blank_domain_pct": 100 * sample["outside_blank_domain"].mean(),
            "hips_uncertainty_nonnull_pct": 100 * sample["Uncertainty"].notna().mean(),
        }]
    )
    return sample, loading, quality


def curvature_analysis(
    bridge: pd.DataFrame, *, seed: int = 20260901
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare linear, quadratic, and spline Fabs--TOR models by held-out site."""
    data = bridge.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Fabs_Mm1", "EC_ugm3", "OC_EC", "Site"]
    ).copy()
    bounds = {
        col: data[col].quantile([0.005, 0.995]).to_numpy()
        for col in ["Fabs_Mm1", "EC_ugm3", "OC_EC"]
    }
    keep = np.ones(len(data), dtype=bool)
    for col, (lo, hi) in bounds.items():
        keep &= data[col].between(lo, hi).to_numpy()
    data = data.loc[keep].copy()
    ec_scale = float(data["EC_ugm3"].quantile(0.75) - data["EC_ugm3"].quantile(0.25))
    ec_center = float(data["EC_ugm3"].median())
    data["ec_z"] = (data["EC_ugm3"] - ec_center) / ec_scale
    data["log_ocec"] = np.log1p(data["OC_EC"])
    counts = data["Site"].value_counts()
    weights = data["Site"].map(1 / counts).to_numpy(float)
    weights *= len(weights) / weights.sum()

    spline = SplineTransformer(n_knots=5, degree=2, include_bias=False)
    spline.fit(data[["ec_z"]])

    def matrix(kind: str, frame: pd.DataFrame) -> np.ndarray:
        ec = frame["ec_z"].to_numpy(float)
        ratio = frame["log_ocec"].to_numpy(float)
        if kind == "linear":
            return np.column_stack([ec, ratio, ec * ratio])
        if kind == "quadratic":
            return np.column_stack([ec, ec**2, ratio, ec * ratio])
        if kind == "spline":
            return np.column_stack([spline.transform(frame[["ec_z"]]), ratio])
        raise ValueError(kind)

    y = data["Fabs_Mm1"].to_numpy(float)
    predictions = {name: np.full(len(data), np.nan) for name in ("linear", "quadratic", "spline")}
    split = GroupKFold(n_splits=5)
    for train, test in split.split(data, groups=data["Site"]):
        for name in predictions:
            fitted = LinearRegression().fit(matrix(name, data.iloc[train]), y[train], sample_weight=weights[train])
            predictions[name][test] = fitted.predict(matrix(name, data.iloc[test]))

    cv_rows = []
    for name, pred in predictions.items():
        scored = data[["Site"]].copy()
        scored["residual"] = pred - y
        cv_rows.append({
            "model": name,
            "n": len(data),
            "sites": data["Site"].nunique(),
            "pooled_RMSE_Mm1": _rmse(y, pred),
            "site_balanced_RMSE_Mm1": _site_balanced_rmse(scored, "residual"),
            "pooled_R2": float(np.corrcoef(y, pred)[0, 1] ** 2),
        })
    cv = pd.DataFrame(cv_rows)
    base_rmse = float(cv.loc[cv["model"].eq("linear"), "site_balanced_RMSE_Mm1"].iloc[0])
    cv["site_balanced_RMSE_change_pct_vs_linear"] = 100 * (
        cv["site_balanced_RMSE_Mm1"] / base_rmse - 1
    )

    grid = pd.DataFrame({"EC_ugm3": np.linspace(data["EC_ugm3"].quantile(.01), data["EC_ugm3"].quantile(.99), 200)})
    grid["ec_z"] = (grid["EC_ugm3"] - ec_center) / ec_scale
    grid["log_ocec"] = float(data["log_ocec"].median())
    curves = []
    for name in predictions:
        fitted = LinearRegression().fit(matrix(name, data), y, sample_weight=weights)
        curves.append(grid.assign(model=name, predicted_Fabs=fitted.predict(matrix(name, grid))))
    curves = pd.concat(curves, ignore_index=True)

    site_rows = []
    for site, group in data.groupby("Site"):
        if len(group) < 200 or group["EC_ugm3"].nunique() < 20:
            continue
        local = group.sort_values("EC_ugm3").copy()
        z = (local["EC_ugm3"] - local["EC_ugm3"].median()) / (
            local["EC_ugm3"].quantile(.75) - local["EC_ugm3"].quantile(.25)
        )
        if not np.isfinite(z).all() or np.ptp(z) == 0:
            continue
        quadratic = LinearRegression().fit(np.column_stack([z, z**2]), local["Fabs_Mm1"])
        midpoint = len(local) // 2
        low, high = local.iloc[:midpoint], local.iloc[midpoint:]
        low_fit = np.polyfit(low["EC_ugm3"], low["Fabs_Mm1"], 1)
        high_fit = np.polyfit(high["EC_ugm3"], high["Fabs_Mm1"], 1)
        site_rows.append({
            "Site": site,
            "n": len(local),
            "median_EC_ugm3": local["EC_ugm3"].median(),
            "median_Fabs_Mm1": local["Fabs_Mm1"].median(),
            "quadratic_coefficient_per_IQR2": quadratic.coef_[1],
            "low_half_linear_intercept_Mm1": low_fit[1],
            "high_half_linear_intercept_Mm1": high_fit[1],
            "high_minus_low_intercept_Mm1": high_fit[1] - low_fit[1],
        })
    sites = pd.DataFrame(site_rows)
    if len(sites):
        q_test = wilcoxon(sites["quadratic_coefficient_per_IQR2"], zero_method="wilcox")
        i_test = wilcoxon(sites["high_minus_low_intercept_Mm1"], zero_method="wilcox")
        tests = pd.DataFrame([
            {"test": "within-site quadratic coefficient differs from zero", "n_sites": len(sites), "median_effect": sites["quadratic_coefficient_per_IQR2"].median(), "wilcoxon_p": q_test.pvalue},
            {"test": "high-minus-low loading-range intercept differs from zero", "n_sites": len(sites), "median_effect": sites["high_minus_low_intercept_Mm1"].median(), "wilcoxon_p": i_test.pvalue},
        ])
    else:
        tests = pd.DataFrame()
    return cv, curves, sites, tests


def _band_features(frame: pd.DataFrame) -> pd.DataFrame:
    identifiers = frame.iloc[:, 0].copy()
    wn = np.asarray([float(value) for value in frame.columns[1:]], float)
    X = frame.iloc[:, 1:].to_numpy(float)
    # Match the neutral-baseline adjudication: a genuine feature must be the
    # absolute local maximum inside 1560--1680, not a residual created by a
    # second local continuum. Delhi/Beijing peak at the 1680 edge (carbonyl
    # flank); Addis/Bishoftu peak at 1617.
    peak = (wn >= 1560) & (wn <= 1680)
    argmax = np.argmax(X[:, peak], axis=1)
    band_index = int(np.argmin(np.abs(wn - 1617)))
    ch_index = int(np.argmin(np.abs(wn - 2920)))
    band = X[:, band_index]
    ch = X[:, ch_index]
    return pd.DataFrame({
        frame.columns[0]: identifiers,
        "peak_cm1": wn[peak][argmax],
        "band_height": band,
        "CH_height": ch,
        "band_to_CH": band / np.where(np.abs(ch) > 1e-8, ch, np.nan),
        "interior_peak": (argmax >= 2) & (argmax <= peak.sum() - 3),
    })


def _partial_permutation(
    frame: pd.DataFrame,
    outcome: str,
    predictor: str,
    controls: list[str],
    *,
    permutations: int = 5000,
    seed: int = 20260901,
) -> dict:
    use = frame[[outcome, predictor] + controls].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < 20 or use[predictor].nunique() < 5:
        return {"n": len(use)}
    Z = np.column_stack([np.ones(len(use))] + [use[col].to_numpy(float) for col in controls])
    y = use[outcome].to_numpy(float)
    x = use[predictor].to_numpy(float)
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for index in range(permutations):
        null[index] = np.corrcoef(rx, rng.permutation(ry))[0, 1]
    p = float((1 + np.sum(np.abs(null) >= abs(r))) / (permutations + 1))
    rho, rho_p = spearmanr(use[predictor], use[outcome])
    return {"n": len(use), "partial_r": r, "permutation_p": p, "spearman_r": rho, "spearman_p": rho_p}


def neutral_band_and_pmf() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Test the neutral-baseline 1617 feature against Addis PMF source fractions."""
    targets = REPO / "calibration_explorer/targets"
    site_rows = []
    labels = {
        "addis_augmented": "Addis", "etbi": "Bishoftu", "indh": "Delhi",
        "chts": "Beijing", "uspa": "Pasadena",
    }
    addis = None
    for folder, label in labels.items():
        spectra = pd.read_csv(targets / folder / "spectra_neutral.csv")
        features = _band_features(spectra)
        median_spectrum = spectra.iloc[:, 1:].median(axis=0)
        median_frame = pd.DataFrame([["median", *median_spectrum.to_numpy()]], columns=spectra.columns)
        med = _band_features(median_frame).iloc[0]
        site_rows.append({
            "site": label,
            "n": len(features),
            "median_peak_cm1": med["peak_cm1"],
            "median_band_height": med["band_height"],
            "median_band_to_CH": med["band_to_CH"],
            "median_has_interior_peak": bool(med["interior_peak"]),
            "per_filter_interior_peak_pct": 100 * features["interior_peak"].mean(),
        })
        if folder == "addis_augmented":
            ref = pd.read_csv(targets / folder / "reference.csv")
            addis = features.merge(ref, on="MediaId", how="left", validate="one_to_one")

    if addis is None:
        raise RuntimeError("Addis neutral spectra were not loaded")
    factors = pd.read_csv(P2_SCRIPTS.parent / "Filter Data/ETAD Factor Contributions .csv")
    factors["date"] = pd.to_datetime(factors["oldDate"], errors="coerce").dt.normalize()
    total = factors[[f"GF{i}" for i in range(1, 6)]].sum(axis=1).replace(0, np.nan)
    names = {1: "sea_salt_frac", 2: "wood_frac", 3: "charcoal_frac", 4: "polluted_marine_frac", 5: "fossil_fuel_frac"}
    for index, name in names.items():
        factors[name] = factors[f"GF{index}"] / total
    addis["date"] = pd.to_datetime(addis["Date"], errors="coerce").dt.normalize()
    joined = addis.merge(factors[["date"] + list(names.values())], on="date", how="left", validate="many_to_one")
    month = joined["date"].dt.month.astype(float)
    joined["log_Fabs"] = np.log1p(joined["Fabs"].clip(lower=0))
    joined["sin_month"] = np.sin(2 * np.pi * month / 12)
    joined["cos_month"] = np.cos(2 * np.pi * month / 12)
    tests = []
    for predictor in names.values():
        tests.append({
            "predictor": predictor,
            **_partial_permutation(
                joined, "band_to_CH", predictor,
                ["log_Fabs", "sin_month", "cos_month"],
            ),
        })
    tests = pd.DataFrame(tests)
    tests["fdr_q"] = np.nan
    ok = tests["permutation_p"].notna()
    tests.loc[ok, "fdr_q"] = multipletests(tests.loc[ok, "permutation_p"], method="fdr_bh")[1]
    return pd.DataFrame(site_rows), joined, tests


def canonical_soil_tests() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test the canonical IMPROVE soil formula against locked residuals."""
    source = P3 / "output/tables/variation_closure/residual_chemistry_per_filter.csv"
    residuals = pd.read_csv(source)
    filters = load_filter_data()
    calcium = filters[filters["Parameter"].eq("ChemSpec_Calcium_PM2.5")][
        ["FilterId", "Site", "Concentration"]
    ].copy()
    calcium["base_filter_id"] = calcium["FilterId"].map(base_filter_id)
    calcium = calcium.drop_duplicates(["base_filter_id", "Site"]).rename(columns={"Concentration": "Ca"})
    joined = residuals.merge(
        calcium[["base_filter_id", "Site", "Ca"]],
        on=["base_filter_id", "Site"], how="left", validate="many_to_one",
    )
    for element in ["Al", "Si", "Ca", "Fe", "Ti"]:
        joined[f"{element}_nonnegative"] = joined[element].clip(lower=0)
    # Element concentrations are ng/m3 in the unified table; formula output is ug/m3.
    joined["IMPROVE_soil_ugm3"] = (
        2.20 * joined["Al_nonnegative"]
        + 2.49 * joined["Si_nonnegative"]
        + 1.63 * joined["Ca_nonnegative"]
        + 2.42 * joined["Fe_nonnegative"]
        + 1.94 * joined["Ti_nonnegative"]
    ) / 1000.0
    joined["log_soil"] = np.log1p(joined["IMPROVE_soil_ugm3"])
    joined["log_Kion"] = np.log1p(joined["K_ion"].clip(lower=0))
    joined["sin_month"] = np.sin(2 * np.pi * joined["month"] / 12)
    joined["cos_month"] = np.cos(2 * np.pi * joined["month"] / 12)

    rows = []
    for (config, target), group in joined.groupby(["config", "target"]):
        for predictor in ["log_soil", "dust_index", "log_Kion"]:
            rows.append({
                "config": config,
                "target": target,
                "predictor": predictor,
                **_partial_permutation(
                    group, "calibration_residual", predictor,
                    ["observed_bc_mac10_ugm3", "sin_month", "cos_month"],
                ),
            })
    tests = pd.DataFrame(rows)
    tests["fdr_q"] = np.nan
    for _, indices in tests.groupby(["config", "target"]).groups.items():
        indices = list(indices)
        ok = tests.loc[indices, "permutation_p"].notna()
        selected = np.asarray(indices)[ok.to_numpy()]
        if len(selected):
            tests.loc[selected, "fdr_q"] = multipletests(
                tests.loc[selected, "permutation_p"], method="fdr_bh"
            )[1]
    return joined, tests


def _load_training_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = load_pool_metadata().merge(
        load_tor_loadings(), on=["Site", "date"], how="left", validate="many_to_one"
    )
    pool = metadata.query("TOR_EC_loading_ug > 0").drop_duplicates("FilterId").copy()
    pool["AnalysisId"] = pool["AnalysisId"].astype(int)
    return metadata, pool.drop_duplicates("AnalysisId").set_index("AnalysisId")


def _resolve_source_ids(spec: pd.Series, metadata: pd.DataFrame, pool: pd.DataFrame) -> np.ndarray:
    if spec["cohort"] == "ocec":
        eligible = (
            metadata["TOR_EC_loading_ug"].gt(0)
            & metadata["TOR_EC_ugm3"].gt(0)
            & metadata["TOR_OC_ugm3"].gt(0)
            & metadata["OC_EC_ratio"].notna()
        )
        ranked = (
            metadata[eligible].sort_values("OC_EC_ratio")
            .drop_duplicates("FilterId")["AnalysisId"].astype(int).to_numpy()
        )
    else:
        ranked = np.load(REPO / "calibration_explorer/cache/analog_corrected_ranking.npz")["ids"].astype(int)
    ranked = ranked[np.isin(ranked, pool.index.to_numpy())]
    return np.array(list(dict.fromkeys(ranked[: int(spec["cutoff"])])), dtype=int)


def _source_representation(ids: np.ndarray, spectra: str, pool: pd.DataFrame, corrected) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if spectra == "airspec":
        row = {int(value): i for i, value in enumerate(corrected["analysis_id"].astype(int))}
        X = corrected["corrected"][[row[int(value)] for value in ids]].astype(float)
    else:
        columns = pd.read_csv(PATHS.ftir_dir / "local_db/spectra_248_251.csv", nrows=0).columns
        wcols = sorted(
            [value for value in columns if str(value).replace(".", "", 1).isdigit()],
            key=lambda value: -float(value),
        )
        raw = load_pool_spectra(ids, wcols).set_index("AnalysisId").loc[ids]
        X = savgol_filter(raw[wcols].to_numpy(float), 11, 2, deriv=2, axis=1)
    return (
        block_average(X, 8),
        pool.loc[ids, "TOR_EC_loading_ug"].to_numpy(float),
        pool.loc[ids, "Site"].astype(str).to_numpy(),
    )


def _target_representation(name: str, spectra: str) -> tuple[np.ndarray, np.ndarray]:
    path = REPO / "calibration_explorer/targets" / name / (
        "spectra_corrected.csv" if spectra == "airspec" else "spectra.csv"
    )
    frame = pd.read_csv(path)
    ids = frame.iloc[:, 0].to_numpy()
    X = frame.iloc[:, 1:].to_numpy(float)
    if spectra == "deriv2":
        X = savgol_filter(X, 11, 2, deriv=2, axis=1)
    return ids, block_average(X, 8)


def applicability_domain_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Source-only PCA T2/Q domain diagnostics, revealed against locked errors."""
    frozen = pd.read_csv(P3 / "output/tables/ftir36/frozen_protocol.csv")
    predictions = pd.read_csv(P3 / "output/tables/ftir36/locked_target_predictions.csv")
    predictions = predictions[predictions["variant"].eq("ordinary_pls")].copy()
    predictions["ExternalFilterId"] = predictions["ExternalFilterId"].astype(str)
    metadata, pool = _load_training_frame()
    corrected = np.load(P3 / "output/corrected/improve_pool_corrected_df6.npz", allow_pickle=True)
    target_map = {
        "addis_holdout": "addis_reconstructed_holdout",
        "delhi_holdout": "indh_reconstructed_holdout",
    }
    rows = []
    for _, spec in frozen.iterrows():
        ids = _resolve_source_ids(spec, metadata, pool)
        X, _, groups = _source_representation(ids, spec["spectra"], pool, corrected)
        train = protocol_train_mask("site_heldout", X, np.arange(len(X)), groups)
        scaler = StandardScaler().fit(X[train])
        Z_train = scaler.transform(X[train])
        Z_outer = scaler.transform(X[~train])
        pca_probe = PCA(n_components=min(20, len(Z_train) - 1, Z_train.shape[1])).fit(Z_train)
        k = int(np.searchsorted(np.cumsum(pca_probe.explained_variance_ratio_), .95) + 1)
        k = max(2, min(k, pca_probe.n_components_))
        pca = PCA(n_components=k).fit(Z_train)
        score_train = pca.transform(Z_train)
        score_outer = pca.transform(Z_outer)
        covariance = LedoitWolf().fit(score_train)

        def diagnostics(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            score = pca.transform(Z)
            centered = score - covariance.location_
            t2 = np.einsum("ij,jk,ik->i", centered, covariance.precision_, centered)
            reconstruction = pca.inverse_transform(score)
            q = np.sum((Z - reconstruction) ** 2, axis=1)
            return t2, q

        outer_t2, outer_q = diagnostics(Z_outer)
        for target, folder in target_map.items():
            target_ids, X_target = _target_representation(folder, spec["spectra"])
            t2, q = diagnostics(scaler.transform(X_target))
            t2_pct = np.array([(1 + np.sum(outer_t2 <= value)) / (len(outer_t2) + 1) for value in t2])
            q_pct = np.array([(1 + np.sum(outer_q <= value)) / (len(outer_q) + 1) for value in q])
            pred = predictions[
                predictions["config"].eq(spec["config"])
                & predictions["target"].eq(target)
            ].copy()
            pred["ExternalFilterId"] = pred["ExternalFilterId"].astype(str)
            for ident, t2_value, q_value, t2p, qp in zip(target_ids, t2, q, t2_pct, q_pct):
                rows.append({
                    "config": spec["config"], "target": target,
                    "ExternalFilterId": str(ident), "source_train_n": int(train.sum()),
                    "source_outer_n": int((~train).sum()), "pca_components_95pct": k,
                    "T2": t2_value, "Q_residual": q_value,
                    "T2_source_percentile": t2p, "Q_source_percentile": qp,
                    "outside_source_95": bool((t2p > .95) or (qp > .95)),
                })
            # Predictions are joined after rows are accumulated to keep target labels out
            # of the domain calculation above.
    per_filter = pd.DataFrame(rows).merge(
        predictions[["config", "target", "ExternalFilterId", "residual_ugm3"]],
        on=["config", "target", "ExternalFilterId"], how="left", validate="one_to_one",
    )
    per_filter["abs_error_ugm3"] = per_filter["residual_ugm3"].abs()
    summary_rows = []
    for (config, target), group in per_filter.groupby(["config", "target"]):
        complete = group.dropna(subset=["abs_error_ugm3"])
        rho, pvalue = spearmanr(
            complete[["T2_source_percentile", "Q_source_percentile"]].max(axis=1),
            complete["abs_error_ugm3"],
        )
        summary_rows.append({
            "config": config, "target": target, "n": len(group),
            "outside_source_95_pct": 100 * group["outside_source_95"].mean(),
            "median_T2_source_percentile": group["T2_source_percentile"].median(),
            "median_Q_source_percentile": group["Q_source_percentile"].median(),
            "rho_domain_percentile_vs_abs_error": rho,
            "rho_p": pvalue,
            "RMSE_inside": _rmse(0, complete.loc[~complete["outside_source_95"], "residual_ugm3"]) if (~complete["outside_source_95"]).any() else np.nan,
            "RMSE_outside": _rmse(0, complete.loc[complete["outside_source_95"], "residual_ugm3"]) if complete["outside_source_95"].any() else np.nan,
        })
    return per_filter, pd.DataFrame(summary_rows)


def campaign_design_analysis(
    *, simulations: int = 20000, seed: int = 20260901
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Power and allocation audit for the seasonal paired Teflon/quartz campaign."""
    ref = pd.read_csv(REPO / "calibration_explorer/targets/addis_augmented/reference.csv")
    ref = ref[ref["ReferenceSource"].eq("shipped")].copy()
    ref["Date"] = pd.to_datetime(ref["Date"], errors="coerce")
    month = ref["Date"].dt.month
    ref["season"] = np.select(
        [month.isin([10, 11, 12, 1, 2]), month.isin([3, 4, 5])],
        ["Dry (Oct-Feb)", "Belg (Mar-May)"], default="Kiremt (Jun-Sep)",
    )

    batch = pd.read_csv(
        davis_root() / "Spartan/SPARTAN_HIPS_Batch1-51.v2.csv",
        encoding="cp1252", usecols=["Site", "FilterType", "Uncertainty", "Fabs"],
        low_memory=False,
    )
    hips_sigma = float(batch.loc[
        batch["Site"].eq("ETAD") & batch["FilterType"].eq("PM2.5"), "Uncertainty"
    ].median() / 10.0)

    adama = pd.read_csv(davis_root() / "DAVIS/Adama TOR/OC_EC_concs_Batch54.csv")
    ec = adama[adama["Parameter"].isin(["ECTR", "ECTT"])].pivot(
        index="FilterId", columns="Parameter", values="Concentration_ug_m3"
    ).dropna()
    ec["protocol_difference"] = ec["ECTR"] - ec["ECTT"]
    tor_protocol_sigma = float(ec["protocol_difference"].std(ddof=1))
    # The five Adama filters quantify only the reflectance/transmittance split.
    # For design, retain the project's conservative TOR error model
    # (10% of EC at MAC=6 + 0.3 ug/m3) and add measured HIPS/protocol terms.
    paired_sigma = float(np.sqrt(hips_sigma**2 + tor_protocol_sigma**2))

    season_summary = ref.groupby("season").agg(
        historical_filters=("ExternalFilterId", "size"),
        Fabs_median=("Fabs", "median"), Fabs_p25=("Fabs", lambda v: v.quantile(.25)),
        Fabs_p75=("Fabs", lambda v: v.quantile(.75)),
    ).reset_index()
    season_summary["expected_MAC6_minus_MAC10_ugm3"] = season_summary["Fabs_median"] * (1/6 - 1/10)
    season_summary["HIPS_sigma_ugm3"] = hips_sigma
    season_summary["Adama_ECTR_minus_ECTT_sd_ugm3"] = tor_protocol_sigma
    season_summary["paired_sigma_ugm3"] = paired_sigma

    rng = np.random.default_rng(seed)
    power_rows = []
    for _, row in season_summary.iterrows():
        historical = ref.loc[ref["season"].eq(row["season"]), "Fabs"].to_numpy(float)
        for retained in [1.0, 0.5]:
            for n in [6, 9, 12, 15, 18]:
                detected_95 = np.zeros(simulations, dtype=bool)
                detected_5sigma = np.zeros(simulations, dtype=bool)
                for i in range(simulations):
                    fabs = rng.choice(historical, n, replace=True)
                    tor_sigma = 0.10 * (fabs / 6.0) + 0.3
                    sigma = np.sqrt(tor_sigma**2 + hips_sigma**2 + tor_protocol_sigma**2)
                    difference = retained * fabs * (1/6 - 1/10) + rng.normal(0, sigma)
                    se = difference.std(ddof=1) / np.sqrt(n)
                    z = difference.mean() / se if se > 0 else np.inf
                    detected_95[i] = z > 1.96
                    detected_5sigma[i] = z > 5.0
                power_rows.append({
                    "season": row["season"], "n_primary_pairs": n,
                    "effect_retained": retained,
                    "simulated_power_95pct": detected_95.mean(),
                    "simulated_power_5sigma": detected_5sigma.mean(),
                })
    power = pd.DataFrame(power_rows)

    allocation = pd.DataFrame([
        {"season": "Dry (Oct-Feb)", "months": "Oct, Nov, Dec, Jan, Feb", "primary_pairs": 13, "field_blanks": 2, "collocated_duplicates": 2},
        {"season": "Belg (Mar-May)", "months": "Mar, Apr, May", "primary_pairs": 12, "field_blanks": 2, "collocated_duplicates": 2},
        {"season": "Kiremt (Jun-Sep)", "months": "Jun, Jul, Aug, Sep", "primary_pairs": 11, "field_blanks": 2, "collocated_duplicates": 2},
    ])
    allocation["total_filters"] = allocation[["primary_pairs", "field_blanks", "collocated_duplicates"]].sum(axis=1)
    qc = pd.DataFrame([
        {"measurement": "paired Teflon HIPS + quartz TOR", "role": "primary endpoint", "n": 36},
        {"measurement": "field blanks, both media", "role": "season-specific contamination/blank correction", "n": 6},
        {"measurement": "collocated duplicate pairs", "role": "sampling + analytical precision", "n": 6},
        {"measurement": "TOR reflectance/transmittance split", "role": "protocol sensitivity retained from Batch 54", "n": 36},
    ])
    return season_summary, power, allocation, qc
