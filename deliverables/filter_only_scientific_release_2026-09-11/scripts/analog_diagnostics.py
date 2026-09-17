"""Exploratory diagnostics for fixed FTIR calibrations and analog selection.

HIPS Fabs/MAC is a comparison proxy, so discrepancies are not chemical-EC
prediction errors. Cluster intervals condition on the already fitted models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from seasonal_analogs import centered_unit_spectra, rank_unique_filters


def discrepancy_stats(reference, prediction):
    x, y = np.asarray(reference, float), np.asarray(prediction, float)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite([x, y]).all():
        raise ValueError("Expected paired, finite one-dimensional observations")
    if len(x) < 1:
        raise ValueError("At least one pair is required")
    d = y - x
    r2 = np.corrcoef(x, y)[0, 1] ** 2 if np.std(x) > 0 and np.std(y) > 0 else np.nan
    return dict(
        n=len(x),
        bias=float(d.mean()),
        rmse=float(np.sqrt(np.mean(d**2))),
        mae=float(np.mean(abs(d))),
        r_squared=float(r2),
    )


def month_resamples(blocks, n_boot, seed):
    """Sample complete observed year-month clusters, preserving repeated clusters."""
    labels = np.asarray(blocks)
    if pd.isna(labels).any():
        raise ValueError("Every observation needs a month block")
    unique = np.unique(labels)
    if len(unique) < 2:
        raise ValueError("At least two month blocks are required")
    members = [np.flatnonzero(labels == label) for label in unique]
    rng = np.random.default_rng(seed)
    for draw in rng.integers(0, len(unique), (n_boot, len(unique))):
        yield np.concatenate([members[i] for i in draw])


def paired_discrepancy_bootstrap(
    reference, baseline, candidate, blocks, *, n_boot=4000, seed=20260911
):
    """Candidate-minus-baseline discrepancy, using identical month draws for both."""
    x, old, new = map(lambda a: np.asarray(a, float), [reference, baseline, candidate])
    a, b = discrepancy_stats(x, old), discrepancy_stats(x, new)
    if len(blocks) != len(x):
        raise ValueError("Month blocks must align with observation pairs")
    draws = []
    for idx in month_resamples(blocks, n_boot, seed):
        d0, d1 = old[idx] - x[idx], new[idx] - x[idx]
        draws.append(
            [
                np.sqrt(np.mean(d1**2)) - np.sqrt(np.mean(d0**2)),
                np.mean(abs(d1)) - np.mean(abs(d0)),
                d1.mean() - d0.mean(),
            ]
        )
    q = np.quantile(draws, [0.025, 0.975], axis=0)
    result = dict(n=len(x), n_months=len(np.unique(blocks)), n_boot=n_boot)
    for j, metric in enumerate(["rmse", "mae", "bias"]):
        result.update(
            {
                f"baseline_{metric}": a[metric],
                f"candidate_{metric}": b[metric],
                f"delta_{metric}": b[metric] - a[metric],
                f"delta_{metric}_low": q[0, j],
                f"delta_{metric}_high": q[1, j],
            }
        )
    return result


def analog_month_stability(
    library, targets, mask, metadata, blocks, *, n_select=500, n_boot=200, seed=20260912
):
    """Recompute the target median and signed-Pearson ranks in every month draw.

    Physical-filter IDs are compared even if a duplicate scan changes rank.
    Frequencies measure sensitivity to this empirical sampling distribution.
    """
    lib = centered_unit_spectra(np.asarray(library)[:, mask])
    target = np.asarray(targets)[:, mask]
    centers = [np.median(target, axis=0)]
    centers.extend(np.median(target[idx], axis=0) for idx in month_resamples(blocks, n_boot, seed))
    scores = np.clip(lib @ centered_unit_spectra(np.asarray(centers)).T, -1, 1)
    ids, analyses = metadata.FilterId.to_numpy(), metadata.AnalysisId.to_numpy()
    eligible = metadata.calibration_eligible.to_numpy(bool)
    base_order = rank_unique_filters(scores[:, 0], ids, analyses, eligible)[:n_select]
    if len(base_order) < n_select:
        raise ValueError("Too few eligible physical filters")
    original = set(ids[base_order])
    counts, records = {}, []
    for b in range(n_boot):
        order = rank_unique_filters(scores[:, b + 1], ids, analyses, eligible)[:n_select]
        selected = set(ids[order])
        overlap = len(original & selected)
        records.append(
            dict(
                draw=b,
                shared=overlap,
                retained_fraction=overlap / n_select,
                jaccard=overlap / (2 * n_select - overlap),
            )
        )
        for fid in selected:
            counts[fid] = counts.get(fid, 0) + 1
    base_ranks = dict(zip(ids[base_order], np.arange(1, n_select + 1)))
    frequencies = pd.DataFrame(
        [
            dict(
                FilterId=fid,
                selection_frequency=counts.get(fid, 0) / n_boot,
                original_rank=base_ranks.get(fid, np.nan),
            )
            for fid in sorted(set(counts) | original)
        ]
    )
    return pd.DataFrame(records), frequencies, ids[base_order]


def site_concentration(membership):
    """Inverse-Simpson effective site count: 1/sum(site fraction squared)."""
    if membership.FilterId.duplicated().any():
        raise ValueError("Use one membership row per physical filter")
    counts = membership.Site.value_counts()
    p = counts / counts.sum()
    summary = dict(
        n_filters=int(counts.sum()),
        n_sites=len(counts),
        effective_sites=float(1 / np.sum(p**2)),
        top5_fraction=float(p.head(5).sum()),
    )
    curve = pd.DataFrame(
        dict(
            Site=counts.index,
            count=counts.values,
            rank=np.arange(1, len(counts) + 1),
            cumulative_fraction=p.cumsum().values,
        )
    )
    return summary, curve


def spectral_shape_projection(library, targets, mask, metadata, *, n_components=10):
    """Fit a descriptive PCA on unique eligible source filters; project Addis.

    Center/unit-normalize each masked spectrum exactly as in Pearson matching.
    Resolve the two duplicate source filters by lowest AnalysisId before PCA.
    The source 95th-percentile reconstruction residual is descriptive, not a
    calibrated acceptance threshold or a chemical-EC accuracy measure.
    """
    eligible = metadata.loc[metadata.calibration_eligible].copy()
    pos = eligible.sort_values("AnalysisId").drop_duplicates("FilterId").index.to_numpy()
    source = centered_unit_spectra(np.asarray(library)[pos][:, mask])
    target = centered_unit_spectra(np.asarray(targets)[:, mask])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=20260910)
    pca.fit(source)
    source_scores, target_scores = pca.transform(source), pca.transform(target)
    source_q = np.sum((source - pca.inverse_transform(source_scores)) ** 2, axis=1)
    target_q = np.sum((target - pca.inverse_transform(target_scores)) ** 2, axis=1)
    threshold = float(np.quantile(source_q, 0.95))
    return dict(
        source_positions=pos,
        source_scores=source_scores,
        target_scores=target_scores,
        source_q=source_q,
        target_q=target_q,
        source_q95=threshold,
        variance_ratio=pca.explained_variance_ratio_,
        components=pca.components_,
        source_mean=pca.mean_,
    )
