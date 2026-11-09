"""Prepare portable, hashed source-domain conference experiment inputs.

The default settings reproduce the bounded source experiment recorded for this
PR. Preparation freezes the site split before any experimental model is fitted;
the shared historical loader only verifies the existing archived calibration.
No target-domain EC reference, assay availability or local accuracy is inferred.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from .coverage import (
    REFERENCE_KIND, SEED, load_historical_source, prepare_physical_spectra,
    source_test_sites,
)
from .io import new_output_directory, sha256, write_json


NATIVE_SOURCE_FILES = (
    "apps/apps_data.npz",
    "notebook/extracted/EC_samples.csv",
    "notebook/extracted/EC_stats_chosen.csv",
    "notebook/extracted/EC_fitted_long.csv",
)

SOURCE_MODEL_OPTIONS = {
    "ridge": [{"n_components": 8, "alpha": 1.}, {"n_components": 8, "alpha": 10.}],
    "pls": [{"n_components": 4}, {"n_components": 8}, {"n_components": 16}],
    "rf": [{"rf_params": {"n_estimators": 100, "min_samples_leaf": 2}},
           {"rf_params": {"n_estimators": 100, "min_samples_leaf": 5}}],
}


def prepare_source_inputs(
    folder, output, *, budget=128, seed=SEED, test_site_fraction=.25,
    source_policy="documented_qc", inner_folds=3, bootstrap_repeats=500,
    sampling_seeds=None, model_options=None,
):
    """Write source spectra, metadata, receipt and executable manifest; return its path.

    ``folder`` is the reviewed native FTIR archive containing ``apps`` and
    ``notebook/extracted``. ``output`` must be new or empty. Defaults exactly
    specify the PR's 128-label, three-family, three-seed source experiment.
    The manifest contains relative paths and a fixed whole-site holdout. Raw
    archive hashes and canonical input hashes remain in ``source_receipt.json``.
    The full original cohort and documented exclusion reasons are retained.
    """
    integer_settings = (("budget", budget, 6), ("seed", seed, 0),
                        ("inner_folds", inner_folds, 2),
                        ("bootstrap_repeats", bootstrap_repeats, 0))
    for name, value, minimum in integer_settings:
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    if not 0 < test_site_fraction < 1:
        raise ValueError("test_site_fraction must be between zero and one")
    seeds = list(sampling_seeds) if sampling_seeds is not None else [seed, seed+1, seed+2]
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in seeds)):
        raise ValueError("sampling_seeds must be distinct nonnegative integers")
    options = deepcopy(SOURCE_MODEL_OPTIONS if model_options is None else model_options)
    if (not isinstance(options, dict) or set(options) != {"ridge", "pls", "rf"}
            or any(not candidates for candidates in options.values())):
        raise ValueError("model_options must specify candidates for ridge, pls and rf")
    folder = Path(folder)
    native_hashes = {name: sha256(folder / name) for name in NATIVE_SOURCE_FILES}
    destination = new_output_directory(Path(output))
    X, metadata, wavenumbers = load_historical_source(folder, source_policy=source_policy)
    if native_hashes != {name: sha256(folder / name) for name in NATIVE_SOURCE_FILES}:
        raise ValueError("native source inputs changed during preparation")
    cohort = prepare_physical_spectra(X, metadata, target_col="ec_ug")
    test_sites = source_test_sites(metadata, fraction=test_site_fraction, seed=seed)
    eligible = cohort.metadata.experiment_eligible
    test = cohort.metadata.site.isin(test_sites)
    train = cohort.metadata.loc[eligible & ~test]
    if len(train) < budget:
        raise ValueError("budget exceeds the eligible source training pool")
    # Fully depleted coverage arms need this many non-region samples. Validate
    # EC extremes now; the runner also checks the training-only spectral region.
    lower, upper = np.quantile(train.ec_ug, [.2, .8])
    if budget > min(int((train.ec_ug > lower).sum()), int((train.ec_ug < upper).sum())):
        raise ValueError("budget exceeds the depleted loading-region capacity")
    spectra_path = destination / "source.npz"
    metadata_path = destination / "source_metadata.csv"
    np.savez_compressed(spectra_path, X=X,
        analysis_ids=metadata.analysis_id.astype(str).to_numpy(dtype=str), wavenumbers=wavenumbers)
    metadata.to_csv(metadata_path, index=False)
    common = dict(reference_kind=REFERENCE_KIND, response_unit="ug/filter",
        test_sites=test_sites, budget=budget, models=["ridge", "pls", "rf"],
        inner_folds=inner_folds, model_options=options, bootstrap_repeats=bootstrap_repeats)
    manifest = {
        "schema_version": 1, "data_scope": "research",
        "datasets": {"source": {
            "kind": "spectra", "path": spectra_path.name, "metadata": metadata_path.name,
            "row_id_column": "analysis_id", "matrix_key": "X", "ids_key": "analysis_ids",
            "description": "Historical IMPROVE raw FTIR spectra with independent thermal-optical EC loading; native hashes and source policy in source_receipt.json.",
        }},
        "experiments": [
            {"id": "coverage", "kind": "calibration_coverage", "inputs": {"data": "source"},
             "parameters": {**common, "regions": ["low", "high", "spectral"],
                            "retain_fractions": [0., .5, 1.], "seed": seed}},
            {"id": "sampling", "kind": "selection_simulation", "inputs": {"data": "source"},
             "parameters": {**common, "strategies": ["random", "coverage", "hybrid"],
                            "representative_fraction": .5, "seeds": seeds, "strata": []}},
        ],
    }
    manifest_path = destination / "manifest.json"
    write_json(manifest_path, manifest)
    write_json(destination / "source_receipt.json", {
        "source_adapter": "conference_experiments.coverage.load_historical_source",
        "preparation": "conference_experiments.preparation.prepare_source_inputs",
        "preparation_code_sha256": sha256(Path(__file__)),
        "native_inputs": native_hashes,
        "canonical_inputs": {path.name: sha256(path) for path in (spectra_path, metadata_path)},
        "manifest_sha256": sha256(manifest_path), "source_policy": source_policy,
        "raw_filters": len(cohort.metadata), "eligible_filters": int(eligible.sum()),
        "source_sites": int(metadata.site.nunique()), "n_training_pool": len(train),
        "n_test": int((eligible & test).sum()), "split_seed": seed,
        "test_site_fraction": test_site_fraction, "test_sites_fixed_before_fit": test_sites,
        "interpretation": "Bounded source-domain coverage and assay-selection experiments, not Addis EC validation. Manifest settings and the split are frozen before experimental fitting.",
    })
    return manifest_path
