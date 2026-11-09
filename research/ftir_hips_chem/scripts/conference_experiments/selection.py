"""Label-free reference-panel design and source-domain sampling simulations.

The representative component has known inclusion probabilities. Challenge
filters deliberately cover spectral/loading/stratum extremes and are kept
separate when estimating population accuracy. QC assays are additional budget
items; an unfilled blank request is never replaced with an invented specimen.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .coverage import (
    DEFAULT_OPTIONS, REFERENCE_KIND, SEED, calibration_scores, fit_source_model,
    paired_source_contrast, prepare_physical_spectra, spectral_coordinates,
)


LIMITS = [
    "Selection is a planning result; no new EC or organic-chemistry measurement has been made.",
    "Challenge samples are deliberately enriched and cannot by themselves estimate population accuracy.",
    "Substrate compatibility, retained material, storage, blanks and recovery require laboratory review.",
    "FTIR-derived EC/OC and HIPS-converted BC are not independent local EC references.",
]


def _is_reference_column(name):
    lower = name.lower()
    return ("tor" in lower.split("_")
            or any(token in lower for token in ("reference", "truth", "target", "ec_ug"))
            or lower in {"actual", "residual", "prediction"})


def _stratum_values(meta, strata):
    if any(column not in meta for column in strata):
        raise ValueError(f"stratification columns missing: {sorted(set(strata)-set(meta))}")
    if not strata:
        return pd.Series("all", index=meta.index)
    # JSON encodes the complete tuple without collisions between separators in
    # real site/lot labels. Unknown values remain explicit rather than inferred.
    return meta[list(strata)].astype("string").fillna("__UNKNOWN__").apply(
        lambda row: json.dumps(row.tolist()), axis=1)


def _representative_sample(strata, n, rng):
    """Proportional fixed-size stratification, or SRS if some strata would be absent."""
    labels = np.asarray(strata, str)
    groups = [(label, np.flatnonzero(labels == label)) for label in sorted(set(labels))]
    if n == 0:
        return np.array([], int), np.zeros(len(labels)), "none"
    if n < len(groups):
        selected = np.sort(rng.choice(len(labels), n, replace=False))
        return selected, np.repeat(n/len(labels), len(labels)), "simple_random_fallback_budget_below_stratum_count"
    # Guarantee representation, then allocate remaining slots toward proportional
    # targets. Sampling stays random within each stratum; n_h/N_h is exact.
    counts = np.ones(len(groups), int)
    sizes = np.asarray([len(indices) for _, indices in groups])
    ideal = n*sizes/sizes.sum()
    while counts.sum() < n:
        deficit = np.where(counts < sizes, ideal-counts, -np.inf)
        counts[int(np.argmax(deficit))] += 1
    probabilities, selected = np.zeros(len(labels)), []
    for count, (_, indices) in zip(counts, groups):
        probabilities[indices] = count/len(indices)
        selected.extend(rng.choice(indices, count, replace=False))
    return np.sort(selected), probabilities, "stratified_random"


def _coverage_sample(coordinates, loading, strata, selected, n):
    """Deterministic extremes plus farthest-point coverage of the remaining pool."""
    selected = list(map(int, selected))
    remaining = set(range(len(coordinates)))-set(selected)
    chosen, reasons = [], []

    def add(index, reason):
        chosen.append(int(index))
        reasons.append(reason)
        selected.append(int(index))
        remaining.remove(int(index))

    for reason, order in (
        ("low_loading_proxy", np.argsort(loading, kind="stable")),
        ("high_loading_proxy", np.argsort(-loading, kind="stable")),
        ("spectral_extreme", np.argsort(-np.linalg.norm(coordinates, axis=1), kind="stable")),
    ):
        if len(chosen) >= n:
            break
        index = next((int(i) for i in order if i in remaining), None)
        if index is not None:
            add(index, reason)
    while len(chosen) < n:
        available = np.asarray(sorted(remaining), int)
        covered = set(np.asarray(strata)[selected])
        uncovered = np.asarray([i for i in available if strata[i] not in covered], int)
        candidates = uncovered if len(uncovered) else available
        if selected:
            distances = np.sum((coordinates[candidates, None, :]-coordinates[np.asarray(selected)][None, :, :])**2, axis=2).min(axis=1)
        else:
            distances = np.sum(coordinates[candidates]**2, axis=1)
        add(candidates[int(np.argmax(distances))],
            "unrepresented_stratum" if len(uncovered) else "spectral_coverage")
    return np.asarray(chosen, int), reasons


def build_reference_panel(
    X, metadata, *, representative_n, challenge_n, physical_id_col="filter_id",
    site_col="site", strata=(), loading_proxy_col=None, blank_n=2, replicate_n=2,
    seed=SEED,
):
    """Return an auditable label-free sample manifest and additional QC assay plan.

    Only X, identities, explicit QC flags, ``strata`` and the optional pre-assay
    ``loading_proxy_col`` can affect selection. Extra reference/response columns
    are ignored. By default the loading proxy is the spectrum's L2 norm, whose
    units are absorbance-norm, not EC mass. The representative probability is
    recorded for every eligible sample, including unselected ones.
    """
    counts = (representative_n, challenge_n, blank_n, replicate_n)
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts):
        raise ValueError("panel and QC counts must be nonnegative integers")
    if representative_n+challenge_n == 0:
        raise ValueError("select at least one physical sample")
    if len(set(strata)) != len(strata):
        raise ValueError("strata must be distinct column names")
    if any(_is_reference_column(name) for name in strata) or (loading_proxy_col and _is_reference_column(loading_proxy_col)):
        raise ValueError("unknown independent reference values cannot be a selection proxy")
    stable = list(strata)+([loading_proxy_col] if loading_proxy_col else [])
    if "available_for_assay" in metadata:
        stable.append("available_for_assay")
    cohort = prepare_physical_spectra(X, metadata, physical_id_col=physical_id_col,
                                     site_col=site_col, stable_columns=stable)
    meta = cohort.metadata.copy()
    if "available_for_assay" in meta:
        known = meta.available_for_assay.notna()
        if not meta.loc[known, "available_for_assay"].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError("available_for_assay must be boolean or unknown")
        unavailable = known & meta.available_for_assay.eq(False)
        meta.loc[unavailable, "experiment_eligible"] = False
        meta.loc[unavailable, "eligibility_reason"] = meta.loc[unavailable, "eligibility_reason"].map(
            lambda reason: ";".join(filter(None, [reason, "specimen_unavailable_for_assay"])))
    pool = np.flatnonzero(meta.experiment_eligible.to_numpy(bool))
    if loading_proxy_col:
        proxy = pd.to_numeric(meta[loading_proxy_col], errors="raise").to_numpy(float)
        missing = ~np.isfinite(proxy)
        meta.loc[missing, "experiment_eligible"] = False
        meta.loc[missing, "eligibility_reason"] = meta.loc[missing, "eligibility_reason"].map(
            lambda reason: ";".join(filter(None, [reason, "nonfinite_loading_proxy"])))
        pool = np.flatnonzero(meta.experiment_eligible.to_numpy(bool))
    else:
        proxy = np.linalg.norm(cohort.X, axis=1)
    parameters = dict(representative_n=representative_n, challenge_n=challenge_n,
                      strata=list(strata), loading_proxy_col=loading_proxy_col,
                      blank_n=blank_n, replicate_n=replicate_n, seed=seed)
    # Availability and supplied-proxy flags apply to every scan of the specimen,
    # as do the original physical-filter exclusions.
    scan_ledger = cohort.scan_ledger.copy()
    physical_flags = meta.set_index(physical_id_col)
    for column in ("experiment_eligible", "eligibility_reason"):
        scan_ledger[column] = scan_ledger[physical_id_col].map(physical_flags[column])
    result = {"summary": {"status": "insufficient_data", "parameters": parameters,
                          "limitations": LIMITS, "eligible_pool": len(pool)},
              "tables": {"manifest": meta, "scan_ledger": scan_ledger}}
    if len(pool) < representative_n+challenge_n or len(pool) < 2:
        result["summary"]["reason"] = "fewer eligible physical filters than the requested disjoint sample budget"
        return result
    coordinates, descriptor = spectral_coordinates(cohort.X[pool], seed=seed)
    labels = _stratum_values(meta.iloc[pool], strata).to_numpy(str)
    rng = np.random.default_rng(seed)
    rep, probabilities, strategy = _representative_sample(labels, representative_n, rng)
    challenge, reasons = _coverage_sample(coordinates, proxy[pool], labels, rep, challenge_n)
    meta["panel"] = "not_selected"
    meta["selection_reason"] = "not_selected_by_prespecified_design"
    meta["selection_order"] = pd.Series(pd.NA, index=meta.index, dtype="Int64")
    meta["representative_inclusion_probability"] = np.nan
    meta["spectral_radius"] = np.nan
    meta["loading_proxy"] = proxy
    meta["sampling_stratum"] = "not_eligible"
    meta.loc[pool, "representative_inclusion_probability"] = probabilities
    meta.loc[pool, "spectral_radius"] = np.linalg.norm(coordinates, axis=1)
    meta.loc[pool, "sampling_stratum"] = labels
    meta.loc[pool[rep], "panel"] = "representative"
    meta.loc[pool[rep], "selection_reason"] = strategy
    meta.loc[pool[rep], "selection_order"] = np.arange(1, len(rep)+1)
    meta.loc[pool[challenge], "panel"] = "challenge"
    meta.loc[pool[challenge], "selection_reason"] = reasons
    meta.loc[pool[challenge], "selection_order"] = np.arange(1, len(challenge)+1)
    # Material suitability is not inferable from spectra or a filter identifier.
    meta["assay_readiness"] = "requires_substrate_material_storage_review"
    qc = []
    blank_available = meta.is_blank & ~meta.is_excluded & ~meta.is_outlier
    if "available_for_assay" in meta:
        blank_available &= ~meta.available_for_assay.eq(False)
    blanks = np.flatnonzero(blank_available.to_numpy(bool))
    chosen_blanks = rng.choice(blanks, min(blank_n, len(blanks)), replace=False) if len(blanks) else []
    for order in range(blank_n):
        i = chosen_blanks[order] if order < len(chosen_blanks) else None
        qc.append({"assay_role": "blank", "assay_order": order+1,
                   "filter_id": None if i is None else meta.at[i, physical_id_col],
                   "status": "specimen_not_available" if i is None else "requires_laboratory_review",
                   "requirement": "Confirm blank type and process with the same extraction/analysis batch"})
        if i is not None:
            meta.at[i, "panel"] = "qc_blank"
            meta.at[i, "selection_reason"] = "blank_QC_assay"
    sample_indices = np.r_[pool[rep], pool[challenge]]
    repeat_order = rng.permutation(sample_indices)
    for order in range(replicate_n):
        i = repeat_order[order % len(repeat_order)]
        qc.append({"assay_role": "blind_repeat", "assay_order": order+1,
                   "filter_id": meta.at[i, physical_id_col], "status": "requires_remaining_material_review",
                   "requirement": "Independent blinded assay aliquot; repeat FTIR scans are not new reference replicates"})
    qc.extend({"assay_role": name, "assay_order": 0, "filter_id": None, "status": "required_before_assay",
               "requirement": requirement} for name, requirement in (
        ("analytical_blank", "Include a laboratory procedural blank for each analytical batch"),
        ("reference_method", "Specify independent assay, substrate compatibility, mass units, calibration and detection limit"),
        ("recovery", "For extraction-based chemistry, specify storage checks, extraction blanks and recovery controls"),
    ))
    stratum_rows = []
    for label in sorted(set(labels)):
        members = pool[labels == label]
        stratum_rows.append({"stratum": label, "eligible_n": len(members),
            "representative_n": int(meta.loc[members, "panel"].eq("representative").sum()),
            "challenge_n": int(meta.loc[members, "panel"].eq("challenge").sum()),
            "representative_probability": float(meta.loc[members, "representative_inclusion_probability"].iloc[0])})
    result["tables"].update(manifest=meta, qc_plan=pd.DataFrame(qc), strata=pd.DataFrame(stratum_rows))
    result["summary"].update(status="completed", representative_strategy=strategy,
        representative_n=len(rep), challenge_n=len(challenge), blank_specimens_available=len(chosen_blanks),
        missing_blank_specimens=blank_n-len(chosen_blanks), reference_assays=representative_n+challenge_n,
        extra_requested_qc_assays=blank_n+replicate_n, spectral_descriptor=descriptor,
        selection_uses_reference_labels=False, readiness="planning_only_material_review_required",
        loading_proxy_definition=loading_proxy_col or "L2 norm of supplied spectrum, not EC mass")
    return result


def simulate_reference_selection(
    X, metadata, *, reference_kind, test_sites, budget, target_col="ec_ug",
    physical_id_col="filter_id", site_col="site", strata=(),
    strategies=("random", "coverage", "hybrid"), representative_fraction=.5,
    seeds=(SEED,), models=("ridge", "pls", "rf"), inner_folds=3,
    model_options=None, bootstrap_repeats=200, response_unit="ug/filter",
):
    """Compare equal label budgets on identical held-out source sites.

    Each selection sees only the candidate training spectra and pre-assay
    metadata. Measured EC is revealed after selecting the panel, solely for
    fitting and inner-site validation. Test features and labels cannot affect
    the selected sample identities. These simulations concern the source domain.
    """
    if reference_kind != REFERENCE_KIND:
        raise ValueError("sampling simulation requires independently measured thermal_optical_ec")
    if response_unit != "ug/filter":
        raise ValueError("sampling simulation is defined for EC loading in ug/filter")
    if "reference_kind" in metadata and not metadata.reference_kind.eq(reference_kind).all():
        raise ValueError("metadata reference_kind conflicts with the declared independent EC reference")
    if not isinstance(budget, int) or budget < 6 or not 0 <= representative_fraction <= 1:
        raise ValueError("budget must be at least six and representative_fraction in [0,1]")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide distinct sampling seeds")
    if not strategies or len(set(strategies)) != len(strategies) or set(strategies)-{"random", "coverage", "hybrid"}:
        raise ValueError("choose distinct random, coverage or hybrid strategies")
    if not models or len(set(models)) != len(models) or set(models)-set(DEFAULT_OPTIONS):
        raise ValueError("choose distinct ridge, pls or rf model families")
    if model_options is not None and (not isinstance(model_options, dict) or set(model_options)-set(models)):
        raise ValueError("model_options must be keyed only by requested model families")
    if not isinstance(bootstrap_repeats, int) or bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats must be nonnegative")
    if isinstance(test_sites, str) or not test_sites:
        raise ValueError("test_sites must list fixed held-out site identities")
    cohort = prepare_physical_spectra(X, metadata, physical_id_col=physical_id_col,
        site_col=site_col, target_col=target_col, stable_columns=strata)
    meta = cohort.metadata.copy()
    if not set(map(str, test_sites)) <= set(meta[site_col]):
        raise ValueError("requested test sites missing from input")
    meta["role"] = np.where(meta[site_col].isin(list(map(str, test_sites))), "test", "training_pool")
    train = np.flatnonzero(meta.experiment_eligible & meta.role.eq("training_pool"))
    test = np.flatnonzero(meta.experiment_eligible & meta.role.eq("test"))
    result = {"summary": {"status": "insufficient_data", "limitations": LIMITS+[
        "Sampling-strategy performance is source-domain evidence, not local EC validation.",
        "Intervals resample held-out sites and condition on fitted models; seed-level results remain separate."],
        "parameters": dict(reference_kind=reference_kind, test_sites=list(test_sites), budget=budget,
            target_col=target_col, response_unit="ug/filter", strata=list(strata), strategies=list(strategies),
            representative_fraction=representative_fraction, seeds=list(seeds), models=list(models),
            inner_folds=inner_folds, model_options=model_options, bootstrap_repeats=bootstrap_repeats)},
        "tables": {"physical_ledger": meta, "scan_ledger": cohort.scan_ledger}}
    if len(train) < budget or not len(test):
        result["summary"]["reason"] = "insufficient training-pool budget or no eligible test filters"
        return result
    y, sites, ids = meta[target_col].to_numpy(float), meta[site_col].to_numpy(str), meta[physical_id_col].to_numpy(str)
    low, high = np.quantile(y[train], [.2, .8])
    bins = np.where(y[test] <= low, "low", np.where(y[test] >= high, "high", "middle"))
    selections, predictions, scores, runs, tunings, folds, contrasts = [], [], [], [], [], [], []
    # Explicitly omit the response from the candidate panel's metadata.
    select_columns = list(dict.fromkeys([physical_id_col, site_col, "is_excluded", "exclusion_reason", "is_outlier", "outlier_reason", "is_blank", *strata]))
    for seed in seeds:
        saved = {}
        for strategy in strategies:
            rep_n = budget if strategy == "random" else 0 if strategy == "coverage" else int(round(budget*representative_fraction))
            panel = build_reference_panel(cohort.X[train], meta.iloc[train][select_columns],
                representative_n=rep_n, challenge_n=budget-rep_n, physical_id_col=physical_id_col,
                site_col=site_col, strata=strata, blank_n=0, replicate_n=0, seed=seed)
            ledger = panel["tables"]["manifest"]
            picked_ids = set(ledger.loc[ledger.panel.isin(("representative", "challenge")), physical_id_col])
            chosen = train[np.isin(ids[train], list(picked_ids))]
            if len(chosen) != budget:
                raise ValueError("sampling strategy changed the common physical-filter budget")
            selections.append(ledger.assign(strategy=strategy, seed=seed))
            for family in models:
                key = dict(strategy=strategy, seed=seed, model=family)
                try:
                    fitted, columns, tuning, membership = fit_source_model(
                        cohort.X[chosen], y[chosen], sites[chosen], ids[chosen], family=family,
                        inner_folds=inner_folds, seed=seed, options=(model_options or {}).get(family))
                    with threadpool_limits(limits=1):
                        pred = np.asarray(fitted.predict(pd.DataFrame(cohort.X[test], columns=columns))).reshape(-1)
                    calibration_scores(y[test], pred)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    runs.append({**key, "status": "failed", "reason": str(exc), "n_train": len(chosen)})
                    continue
                runs.append({**key, "status": "completed", "reason": "", "n_train": len(chosen)})
                tunings.append(tuning.assign(**key))
                folds.append(membership.assign(**key))
                saved[strategy, family] = pred
                for j, i in enumerate(test):
                    predictions.append({**key, "filter_id": ids[i], "site": sites[i], "actual": y[i],
                        "prediction": pred[j], "loading_bin": bins[j], "response_unit": "ug/filter"})
                for label in ("all", "low", "middle", "high"):
                    mask = np.ones(len(test), bool) if label == "all" else bins == label
                    if mask.any():
                        scores.append({**key, "loading_bin": label, "n_train": len(chosen),
                                       **calibration_scores(y[test][mask], pred[mask])})
        for family in models:
            baseline = saved.get(("random", family))
            if baseline is None:
                continue
            for strategy in strategies:
                candidate = saved.get((strategy, family))
                if strategy != "random" and candidate is not None:
                    contrasts.append({"strategy": strategy, "model": family, "seed": seed,
                        "comparison": "strategy_minus_random", **paired_source_contrast(
                            y[test], baseline, candidate, sites[test], repeats=bootstrap_repeats, seed=seed)})
    result["tables"].update(selection=pd.concat(selections, ignore_index=True),
        predictions=pd.DataFrame(predictions), metrics=pd.DataFrame(scores), runs=pd.DataFrame(runs),
        tuning=pd.concat(tunings, ignore_index=True) if tunings else pd.DataFrame(),
        inner_membership=pd.concat(folds, ignore_index=True) if folds else pd.DataFrame(),
        paired_contrasts=pd.DataFrame(contrasts))
    completed = sum(row["status"] == "completed" for row in runs)
    result["summary"].update(status="completed" if completed == len(runs) else "failed",
        completed_runs=completed, attempted_runs=len(runs), n_training_pool=len(train), n_test=len(test),
        fixed_test_filter_ids=ids[test].tolist(), loading_thresholds={"q20": low, "q80": high},
        selection_uses_reference_labels=False, reference_scope="source_domain_only")
    return result
