"""Reproduce Ann's Sept 3 follow-up. Run from repo root with uv run.

Correlation masks affect selection only; every PLS fit uses the same complete
AIRSpec grid. Data availability/positive TOR eligibility are recorded as flags.
No tuning against Addis outcomes. The proposed Addis split is scoped, unscored.
"""

from __future__ import annotations
import json
import os
import sys
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = ROOT / "research/ftir_hips_chem"
P3 = ROOT / "research/ftir_ec_phase3"
sys.path.insert(0, str(P3 / "scripts"))
sys.path.insert(0, str(ACTIVE / "scripts"))
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits
from config import MAC_VALUE, ETHIOPIA_SEASONS, season_for_month, season_convention_name
from data_matching import load_filter_data
from outliers import apply_exclusion_flags, get_clean_data
from plotting import PlotConfig
from plotting.overlays import crossplot_on_axes
from plotting.utils import calculate_regression_stats, deming_bootstrap
from seasonal_analogs import (
    spectral_region_mask,
    mean_correlation_scores,
    rank_unique_filters,
    seasonal_month_split,
)
from phase3_common import load_pool_metadata, load_tor_loadings, PATHS
from calibration_modes import protocol_train_mask, protocol_cv_curve, protocol_select_k
from pmf_source_groups import pmf_group_schemes

PlotConfig.set(sites="Addis_Ababa", layout="individual")
AGGREGATION = os.environ.get("AETHMODULAR_WEEKLY_AGGREGATION", "median")
if AGGREGATION not in ("median", "mean"):
    raise ValueError("AETHMODULAR_WEEKLY_AGGREGATION must be median or mean")
SUFFIX = "_mean_sensitivity" if AGGREGATION == "mean" else ""
OUT = ACTIVE / ("output/tables/ann_weekly_20260910" + SUFFIX)
PLOTS = ACTIVE / ("output/plots/ann_weekly_20260910" + SUFFIX)
DECK = ROOT / "deliverables/ann_weekly_2026-09-10"
for directory in [OUT, PLOTS, DECK]:
    directory.mkdir(parents=True, exist_ok=True)
MASKS = {
    "full": (None, None),
    "no_co2": (1800, None),
    "no_co2_max3600": (1800, 3600),
    "no_co2_max3500": (1800, 3500),
    "co2_1850": (1850, None),
    "co2_1850_max3600": (1850, 3600),
    "co2_1850_max3500": (1850, 3500),
}
PRIMARY = "no_co2"  # selected from meeting instruction, before inspecting outcomes
N_ANALOGS = 500
MODE = "site_heldout"
SEASONS = list(ETHIOPIA_SEASONS)
SEED = 20260910


def slug(value):
    return "".join(c if c.isalnum() else "_" for c in value).strip("_")


def save(frame, name):
    frame.to_csv(OUT / f"{name}.csv", index=False)


def log(message):
    print(message, flush=True)


def render_full_spectra(wn, calibration, targets, group):
    """Plot every actual training and target trace on shared axes."""
    col = ETHIOPIA_SEASONS[group]["color"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    axes[0].plot(wn, calibration.T, color="#70757a", lw=0.35, alpha=0.18)
    axes[1].plot(wn, targets.T, color=col, lw=0.55, alpha=0.3)
    axes[0].set_title(f"Actual calibration fit: {len(calibration)} IMPROVE filters")
    axes[1].set_title(f"All {len(targets)} Addis filters: {group}")
    for ax in axes:
        ax.axvspan(1800, 2500, color="#dddddd", alpha=0.5)
        ax.set(xlim=(wn.max(), wn.min()), xlabel="Wavenumber (cm$^{-1}$)")
    axes[0].set_ylabel("AIRSpec-corrected absorbance")
    fig.suptitle("Full spectra; grey band excluded from analog selection only")
    fig.tight_layout()
    fig.savefig(PLOTS / f"full_calibration_vs_addis_{slug(group)}.png", bbox_inches="tight")
    plt.close(fig)


def render_crossplots(e, pred, met):
    """Refresh the report regressions from saved physical-filter predictions."""
    x = e.Fabs.to_numpy(float) / MAC_VALUE
    sources = sorted(set(e.pmf_source) - {"unmatched"})
    group_masks = {s: e.season.eq(s).to_numpy() for s in SEASONS}
    group_masks.update({s: e.pmf_source.eq(s).to_numpy() for s in sources})
    # Crossplots show the full statistics, no intercept-only comparison.
    for kind, names in [("seasons", SEASONS), ("pmf", sources)]:
        ncols = 3
        nrows = int(np.ceil(len(names) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), squeeze=False)
        for ax, group in zip(axes.ravel(), names):
            fit_id = f"{PRIMARY}__{slug(group)}"
            p = pred.loc[pred.fit_id.eq(fit_id)]
            gm = group_masks[group]
            st = met.loc[met.fit_id.eq(fit_id) & met.evaluation_group.eq(group)].iloc[0].to_dict()
            crossplot_on_axes(
                ax,
                x[gm],
                p.prediction_ugm3.to_numpy()[gm],
                "HIPS Fabs/MAC (µg/m³)",
                "FTIR EC (µg/m³)",
                stats=st,
                stats_box=False,
                deming_line=True,
                color=ETHIOPIA_SEASONS[group]["color"] if kind == "seasons" else None,
            )
            ax.text(
                0.04,
                0.97,
                f"n={st['n']}   R²={st['R2']:.3f}\nOLS slope={st['slope']:.2f}\n"
                f"Deming slope={st['deming_slope']:.2f} [{st['slope_ci_low']:.2f}, {st['slope_ci_high']:.2f}]\n"
                f"Intercept={st['deming_intercept']:.2f} [{st['intercept_ci_low']:.2f}, {st['intercept_ci_high']:.2f}]",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
            ax.set_title(group)
        for ax in axes.ravel()[len(names) :]:
            ax.set_visible(False)
        fig.suptitle("Separate seasonal/source selections; 95% month-block bootstrap intervals")
        fig.tight_layout()
        fig.savefig(PLOTS / f"{kind}_crossplots.png", bbox_inches="tight")
        plt.close(fig)


def main():
    if os.environ.get("AETHMODULAR_WEEKLY_PLOTS_ONLY") == "1":
        e = pd.read_csv(OUT / "addis_evaluation.csv")
        render_crossplots(
            e,
            pd.read_csv(OUT / "predictions.csv"),
            pd.read_csv(OUT / "regression_metrics.csv"),
        )
        library = np.load(P3 / "output/corrected/improve_pool_corrected_df6.npz", allow_pickle=True)
        target = np.load(P3 / "output/corrected/etad_corrected_df6.npz", allow_pickle=True)
        positions = pd.Series(np.arange(len(library["analysis_id"])), index=library["analysis_id"])
        X = np.vstack(
            [
                target["corrected"][target["media_id"].astype(int) == m].mean(axis=0)
                for m in e.MediaId
            ]
        )
        for group in SEASONS:
            cohort = pd.read_csv(OUT / f"cohort_{PRIMARY}__{slug(group)}.csv")
            training = cohort.loc[cohort.role.eq("train"), "AnalysisId"]
            render_full_spectra(
                library["wn"],
                library["corrected"][positions.loc[training].to_numpy()],
                X[e.season.eq(group)],
                group,
            )
        return
    tor_path = Path(
        os.environ.get(
            "AETHMODULAR_WEEKLY_TOR_CSV", PATHS.ftir_dir / "local_db/tables/results_tor.csv"
        )
    )
    log("Loading staged, physical-filter Addis records and corrected scan cache")
    # Same shipped 239 filters as load_addis_evaluation, without rereading the
    # very large raw-spectrum CSV. Reconstructed HIPS extensions stay separate.
    ref = pd.read_csv(ROOT / "calibration_explorer/targets/addis_augmented/reference.csv")
    ref["evaluation_eligible"] = ref.ReferenceSource.eq("shipped")
    ref["date"] = pd.to_datetime(ref.Date)
    ref["season"] = ref.date.dt.month.map(season_for_month)
    ref["filter_id"] = ref.ExternalFilterId
    ref = apply_exclusion_flags(ref, "Addis_Ababa")
    save(ref, "addis_input_audit")
    e = get_clean_data(ref.loc[ref.evaluation_eligible]).reset_index(drop=True)
    assert not e.MediaId.duplicated().any() and not e.ExternalFilterId.duplicated().any()
    original = pd.read_csv(P3 / "output/tables/ftir11/addis_predictions.csv")
    assert set(e.MediaId) == set(original.MediaId), (
        "staged cohort differs from 239-filter evaluation"
    )
    assert np.allclose(e.Fabs, original.set_index("MediaId").loc[e.MediaId, "Fabs"])
    e = e.merge(original[["MediaId", "EC_deployed_ugm3"]], on="MediaId", validate="one_to_one")
    z = np.load(P3 / "output/corrected/etad_corrected_df6.npz", allow_pickle=True)
    wn = z["wn"].astype(float)
    X = np.vstack([z["corrected"][z["media_id"].astype(int) == m].mean(axis=0) for m in e.MediaId])
    assert np.isfinite(X).all()
    groups = pmf_group_schemes(e.date.dt.strftime("%Y-%m-%d").tolist())
    for k, v in groups.items():
        e[k] = v
    e["month_block"] = e.date.dt.to_period("M").astype(str)
    save(e, "addis_evaluation")
    log(f"Addis: {len(e)} physical filters; seasons {e.season.value_counts().to_dict()}")

    log("Loading IMPROVE metadata and TOR calibration targets")
    z = np.load(P3 / "output/corrected/improve_pool_corrected_df6.npz", allow_pickle=True)
    assert np.allclose(wn, z["wn"])
    L = z["corrected"].astype(float)
    meta = load_pool_metadata().merge(
        load_tor_loadings(tor_path), on=["Site", "date"], how="left", validate="many_to_one"
    )
    lib = pd.DataFrame(
        {
            "AnalysisId": z["analysis_id"].astype(int),
            "FilterId": z["filter_id"].astype(int),
            "Site": z["site"].astype(str),
        }
    )
    lib = lib.merge(
        meta[
            ["AnalysisId", "date", "TOR_EC_loading_ug", "TOR_EC_ugm3", "TOR_OC_ugm3", "OC_EC_ratio"]
        ],
        on="AnalysisId",
        how="left",
        validate="one_to_one",
    )
    lib["has_finite_spectrum"] = np.isfinite(L).all(axis=1)
    lib["calibration_eligible"] = lib.TOR_EC_loading_ug.gt(0) & lib.has_finite_spectrum
    lib["eligibility_reason"] = np.where(
        lib.calibration_eligible, "", "missing/nonpositive TOR EC or nonfinite spectrum"
    )
    save(lib, "library_eligibility")
    log(
        f"Library: {len(lib)} scans; {lib.loc[lib.calibration_eligible, 'FilterId'].nunique()} eligible physical filters"
    )

    # Reconstruct the historical winner's exact cohort ordering and fit split.
    valid_ocec = (
        meta.TOR_EC_loading_ug.gt(0)
        & meta.TOR_EC_ugm3.gt(0)
        & meta.TOR_OC_ugm3.gt(0)
        & meta.OC_EC_ratio.notna()
    )
    hist = meta.loc[valid_ocec].sort_values("OC_EC_ratio").drop_duplicates("FilterId")
    hist = hist.loc[hist.AnalysisId.isin(lib.loc[lib.calibration_eligible, "AnalysisId"])].head(440)
    positions = (
        lib.reset_index().set_index("AnalysisId").loc[hist.AnalysisId, "index"].to_numpy(int)
    )
    train = protocol_train_mask(MODE, L[positions], hist.TOR_EC_loading_ug, hist.Site)
    historical_train_n = int(train.sum())
    hist = hist.copy()
    hist["role"] = np.where(train, "train", "TOR_test")
    hist["cv_scheme"] = "site-grouped 5-fold"
    hist["mode"] = MODE
    hist["k"] = 8
    save(hist, "historical_addis_winner_membership")
    hist_roles = hist.set_index("FilterId").role.to_dict()
    hist_scan_roles = hist.set_index("AnalysisId").role.to_dict()

    group_masks = {"All Addis": np.ones(len(e), bool)}
    group_masks.update({s: e.season.eq(s).to_numpy() for s in SEASONS})
    sources = sorted(set(e.pmf_source) - {"unmatched"})
    group_masks.update({s: e.pmf_source.eq(s).to_numpy() for s in sources})
    lists, selections, sensitivity, pairs = [], {}, [], []
    mask_rows = []
    for mask_name, (low, upper) in MASKS.items():
        mask = spectral_region_mask(wn, low, upper)
        mask_rows.append(
            {
                "mask": mask_name,
                "co2_low": low,
                "co2_high": 2500 if low else None,
                "upper": upper,
                "n_channels": int(mask.sum()),
            }
        )
        log(f"Selecting {mask_name}: {mask.sum()} channels")
        for group, gm in group_masks.items():
            target_spectra = np.median(X[gm], axis=0)[None, :] if AGGREGATION == "median" else X[gm]
            scores = mean_correlation_scores(L, target_spectra, mask)
            ordered = rank_unique_filters(
                scores, lib.FilterId, lib.AnalysisId, lib.calibration_eligible
            )
            selected = ordered[:N_ANALOGS]
            selections[(mask_name, group)] = selected
            f = lib.iloc[selected].copy()
            f["rank"] = np.arange(1, len(f) + 1)
            f["mask"] = mask_name
            f["target_group"] = group
            f["aggregation"] = AGGREGATION
            f["selection_r"] = scores[selected]
            f["selection_r_squared"] = f.selection_r**2
            f["historical_winner_role"] = f.FilterId.map(hist_roles).fillna("outside_cohort")
            f["historical_winner_scan_role"] = f.AnalysisId.map(hist_scan_roles).fillna(
                "outside_cohort"
            )
            lists.append(f)
            sensitivity.append(
                {
                    "mask": mask_name,
                    "group": group,
                    "target_n": int(gm.sum()),
                    "selected_n": len(f),
                    "sites": f.Site.nunique(),
                    "median_selection_r": float(np.median(scores[selected])),
                    "historical_train_n": int(f.historical_winner_role.eq("train").sum()),
                }
            )
        for a_idx, a in enumerate(SEASONS):
            for b in SEASONS[a_idx + 1 :]:
                sa = set(lib.iloc[selections[(mask_name, a)]].FilterId)
                sb = set(lib.iloc[selections[(mask_name, b)]].FilterId)
                pairs.append(
                    {
                        "mask": mask_name,
                        "group_a": a,
                        "group_b": b,
                        "intersection": len(sa & sb),
                        "union": len(sa | sb),
                        "jaccard": len(sa & sb) / len(sa | sb),
                        "overlap_fraction": len(sa & sb) / N_ANALOGS,
                    }
                )
    all_lists = pd.concat(lists, ignore_index=True)
    save(all_lists, "analog_membership")
    save(pd.DataFrame(sensitivity), "selection_summary")
    save(pd.DataFrame(mask_rows), "mask_channels")
    save(pd.DataFrame(pairs), "seasonal_overlap")
    change = []
    for group in group_masks:
        full = set(lib.iloc[selections[("full", group)]].FilterId)
        base = set(lib.iloc[selections[(PRIMARY, group)]].FilterId)
        for mn in MASKS:
            current = set(lib.iloc[selections[(mn, group)]].FilterId)
            change.append(
                {
                    "group": group,
                    "mask": mn,
                    "shared_with_full": len(current & full),
                    "changed_vs_full": N_ANALOGS - len(current & full),
                    "shared_with_no_co2": len(current & base),
                    "changed_vs_no_co2": N_ANALOGS - len(current & base),
                }
            )
    save(pd.DataFrame(change), "mask_membership_changes")
    pmf_pairs = []
    for i, a in enumerate(sources):
        for b in sources[i + 1 :]:
            sa = set(lib.iloc[selections[(PRIMARY, a)]].FilterId)
            sb = set(lib.iloc[selections[(PRIMARY, b)]].FilterId)
            pmf_pairs.append(
                {
                    "group_a": a,
                    "group_b": b,
                    "intersection": len(sa & sb),
                    "jaccard": len(sa & sb) / len(sa | sb),
                }
            )
    save(pd.DataFrame(pmf_pairs), "pmf_overlap")

    # Old deck's five seasonal nearest spectra: exact AnalysisIds already saved.
    old = pd.read_csv(P3 / "output/tables/ftir52/season_analog_lists.csv")
    old = old.loc[old.search.eq("all filters") & old["rank"].le(5)].copy()
    old["historical_winner_scan_role"] = old.AnalysisId.map(hist_scan_roles).fillna(
        "outside_cohort"
    )
    old["FilterId"] = old.AnalysisId.map(lib.set_index("AnalysisId").FilterId)
    old["historical_winner_filter_role"] = old.FilterId.map(hist_roles).fillna("outside_cohort")
    save(old, "old_deck_top5_membership")
    # Representative whole-site analogs, including Bishoftu, for the revised mask.
    representative = []
    etbi = pd.read_csv(ROOT / "calibration_explorer/targets/etbi/spectra_corrected.csv")
    cols = [c for c in etbi if c not in ("MediaId", "ExternalFilterId")]
    assert np.allclose(np.array(cols, float), wn)
    for name, target in [("Addis", X), ("Bishoftu", etbi[cols].to_numpy(float))]:
        for mn in ["full", PRIMARY]:
            mask = spectral_region_mask(wn, *MASKS[mn])
            scores = mean_correlation_scores(L, np.median(target, axis=0)[None, :], mask)
            order = rank_unique_filters(scores, lib.FilterId, lib.AnalysisId)[:5]
            sub = lib.iloc[order].copy()
            sub["target"] = name
            sub["mask"] = mn
            sub["rank"] = np.arange(1, 6)
            sub["pearson_r"] = scores[order]
            sub["historical_winner_role"] = sub.FilterId.map(hist_roles).fillna("outside_cohort")
            representative.append(sub)
    save(pd.concat(representative), "representative_top5_membership")

    # HIPS uncertainty is a parameter row. This analysis uses MAC from config.
    filters = load_filter_data()
    u = filters.loc[
        filters.Site.eq("ETAD") & filters.Parameter.eq("HIPS_Uncertainty"), "Uncertainty"
    ]
    sigma_x = float(u.median() / MAC_VALUE)
    sigma_y = 0.531  # prior held-out TOR RMSE proxy, not per-sample truth
    lam = (sigma_y / sigma_x) ** 2
    x = e.Fabs.to_numpy(float) / MAC_VALUE
    fit_records = []
    metrics = []
    predictions = []
    envelopes = {}
    fit_targets = [
        (mn, g)
        for mn in ["full", PRIMARY, "no_co2_max3600", "no_co2_max3500"]
        for g in ["All Addis"] + SEASONS
    ]
    fit_targets += [(PRIMARY, g) for g in sources]
    fit_targets += [("historical_ocec440", "All Addis")]
    for mn, group in fit_targets:
        pos = positions if mn == "historical_ocec440" else selections[(mn, group)]
        cohort = lib.iloc[pos].copy()
        y = cohort.TOR_EC_loading_ug.to_numpy(float)
        site = cohort.Site.to_numpy()
        train = protocol_train_mask(MODE, L[pos], y, site)
        curve = protocol_cv_curve(MODE, L[pos], y, site, train, max_components=30)
        k = 8 if mn == "historical_ocec440" else protocol_select_k(MODE, curve)
        fit_id = f"{mn}__{slug(group)}"
        save(curve, f"cv_{fit_id}")
        model = PLSRegression(n_components=k, scale=False).fit(L[pos][train], y[train])
        pred = model.predict(X).ravel() / e.Volume_m3.to_numpy(float)
        test_pred = model.predict(L[pos][~train]).ravel()
        test_y = y[~train]
        tor_stats = calculate_regression_stats(test_y, test_pred)
        tor_q2 = 1 - np.sum((test_pred - test_y) ** 2) / np.sum((test_y - test_y.mean()) ** 2)
        cohort["fit_id"] = fit_id
        cohort["role"] = np.where(train, "train", "TOR_test")
        cohort["k"] = k
        save(cohort, f"cohort_{fit_id}")
        fr = {
            "fit_id": fit_id,
            "mask": mn,
            "selection_group": group,
            "k": k,
            "cohort_n": len(pos),
            "train_n": int(train.sum()),
            "train_sites": len(set(site[train])),
            "test_n": int((~train).sum()),
            "test_sites": len(set(site[~train])),
            "TOR_R2": tor_stats["R2"],
            "TOR_Q2": tor_q2,
            "TOR_RMSE_ug_filter": tor_stats["RMSE"],
            "cv_min_at_boundary": bool(
                int(curve.loc[curve.rmsecv.idxmin(), "n_components"])
                == int(curve.n_components.max())
            ),
        }
        fit_records.append(fr)
        log(
            f"Fit {fit_id}: k={k}, train={train.sum()}, TOR R2={tor_stats['R2']:.3f}, Q2={tor_q2:.3f}"
        )
        p = e[["MediaId", "ExternalFilterId", "date", "season", "pmf_source"]].copy()
        p["fit_id"] = fit_id
        p["prediction_ugm3"] = pred
        p["hips_equivalent_ugm3"] = x
        predictions.append(p)
        eval_groups = ["All Addis"] + SEASONS if group in ["All Addis"] + SEASONS else [group]
        if group == "All Addis":
            eval_groups += sources
        for eg in eval_groups:
            take = group_masks[eg]
            st = calculate_regression_stats(
                x[take], pred[take], errors_in_variables=True, sigma_x=sigma_x, sigma_y=sigma_y
            )
            ci = deming_bootstrap(
                x[take], pred[take], lam, groups=e.month_block.to_numpy()[take], seed=SEED
            )
            metrics.append({**fr, "evaluation_group": eg, **st, **ci})
        if mn == PRIMARY and group in SEASONS:
            gm = group_masks[group]
            render_full_spectra(wn, L[pos][train], X[gm], group)
            # Full-range summaries of ALL plotted spectra for editable deck chart.
            env = {"wavenumber": wn.tolist()}
            for name, arr in [("calibration", L[pos][train]), ("addis", X[gm])]:
                for q in [0, 25, 50, 75, 100]:
                    env[f"{name}_q{q}"] = np.percentile(arr, q, axis=0).tolist()
            env["train_n"] = int(train.sum())
            env["addis_n"] = int(gm.sum())
            envelopes[group] = env
    fits = pd.DataFrame(fit_records)
    met = pd.DataFrame(metrics)
    pred = pd.concat(predictions, ignore_index=True)
    save(fits, "calibration_fits")
    save(met, "regression_metrics")
    save(pred, "predictions")

    render_crossplots(e, pred, met)

    split = seasonal_month_split(e[["MediaId", "ExternalFilterId", "date", "season", "pmf_source"]])
    split["status"] = "proposed_retrospective_design_not_scored"
    save(split, "proposed_addis_split")
    save(
        split.groupby(["season", "role"])
        .agg(n=("MediaId", "size"), months=("month_block", "nunique"))
        .reset_index(),
        "split_counts",
    )
    lots = []
    for name in ["etbi", "etbi_reconstructed_holdout", "etbi_augmented"]:
        r = pd.read_csv(ROOT / f"calibration_explorer/targets/{name}/reference.csv")
        r["target"] = name
        lots.append(r)
    lots = pd.concat(lots, ignore_index=True)
    save(lots, "bishoftu_lot_audit")
    assert lots.LotId.notna().all() and lots.LotId.eq(251).all()
    summary = {
        "date": "2026-09-10",
        "aggregation": AGGREGATION,
        "addis_n": len(e),
        "library_scan_n": len(lib),
        "eligible_filter_n": int(lib.loc[lib.calibration_eligible, "FilterId"].nunique()),
        "season_convention": season_convention_name(),
        "primary_mask": PRIMARY,
        "n_analogs": N_ANALOGS,
        "sigma_x_ugm3": sigma_x,
        "sigma_y_proxy_ugm3": sigma_y,
        "deming_lambda": lam,
        "mac": MAC_VALUE,
        "pmf_counts": e.pmf_source.value_counts().to_dict(),
        "historical_train_n": historical_train_n,
        "fit_count": len(fits),
        "envelopes": envelopes,
    }
    # Capture source checksums without putting raw spectra into the presentation.
    hashes = {}
    for path in [
        P3 / "output/corrected/improve_pool_corrected_df6.npz",
        P3 / "output/corrected/etad_corrected_df6.npz",
        ROOT / "calibration_explorer/targets/addis_augmented/reference.csv",
        tor_path,
    ]:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary["source_sha256"] = hashes
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log("Completed all analyses and wrote auditable outputs")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
