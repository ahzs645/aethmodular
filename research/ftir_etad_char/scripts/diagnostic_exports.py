"""Save aligned scientific arrays for gallery plots without refitting them."""
from __future__ import annotations

import numpy as np
import pandas as pd


def metadata_rows(meta):
    """Keep identifiers and descriptors; native wavenumber columns are spectra."""
    axis = pd.to_numeric(pd.Index(meta.columns), errors='coerce').to_numpy(float)
    spectral = np.isfinite(axis) & (axis >= 100) & (axis <= 5000)
    return meta.loc[:, ~spectral].reset_index(drop=True).copy()


def score_rows(scores, meta, population, analysis):
    scores = np.asarray(scores, float)
    meta = meta.reset_index(drop=True)
    if scores.ndim != 2 or scores.shape[1] < 2 or len(scores) != len(meta):
        raise ValueError('PCA scores and metadata must be row-aligned with at least two PCs')
    out = metadata_rows(meta)
    out['PC1'], out['PC2'] = scores[:, 0], scores[:, 1]
    out['population'], out['analysis'] = population, analysis
    if 'ExternalFilterId' in out:
        out['id'] = out['ExternalFilterId']
    elif 'is_addis' in out and 'sample' in out:
        out['id'] = out['sample'].where(out['is_addis'])
    return out


def envelope_rows(grid, values, population, representation):
    grid, values = np.asarray(grid, float), np.asarray(values, float)
    if values.ndim != 2 or values.shape[1] != len(grid) or not len(values):
        raise ValueError('An envelope needs nonempty spectra aligned to the wavenumber grid')
    low, high = np.percentile(values, [5, 95], axis=0)
    return pd.DataFrame({'wn': grid, 'mean': values.mean(axis=0), 'p05': low, 'p95': high,
                         'population': population, 'representation': representation, 'n': len(values)})
