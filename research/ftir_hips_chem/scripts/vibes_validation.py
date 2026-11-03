"""Independent signal-addition diagnostics for a frozen VIBES background.

These are descriptive controls, not acceptance thresholds or EC validation.
Keep broad additions distinct from the existing four-band injection experiment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def validation_signals(wn):
    wn = np.asarray(wn, float)
    if wn.ndim != 1 or len(wn) < 3 or not np.isfinite(wn).all():
        raise ValueError('Expected a finite wavenumber grid')
    if not (np.all(np.diff(wn) > 0) or np.all(np.diff(wn) < 0)):
        raise ValueError('Wavenumber grid must be strictly monotonic')
    bands = sum(a * np.exp(-.5 * ((wn - c) / w) ** 2)
                for a, c, w in zip([.7, .5, 1., .8], [1610., 1720., 2920., 3400.], [30., 22., 35., 130.]))
    return {'Four diagnostic bands': bands,
            'Broad hump': np.exp(-.5 * ((wn - 2100.) / 850.) ** 2),
            'Power-law continuum': wn / 2000.}


def incremental_recovery(wn, expected, unspiked, spiked):
    """Signed area/gain and relative RMS error of C(blank + signal) − C(blank).

    Unknown/failed channels make this entire window unavailable. A lost signal
    has recovery zero; inversion stays negative. Neither can appear successful
    by taking absolute areas or omitting the channels where a method failed.
    """
    arrays = [np.asarray(a, float) for a in (wn, expected, unspiked, spiked)]
    if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays) or len(arrays[0]) < 3:
        raise ValueError('Aligned one-dimensional spectral arrays are required')
    wn, expected, unspiked, spiked = arrays
    validation_signals(wn)  # validate the integration axis
    if not all(np.isfinite(a).all() for a in arrays):
        return {'available': False, 'area_recovery': np.nan, 'gain': np.nan,
                'relative_RMSE': np.nan, 'expected_area': np.nan, 'recovered_area': np.nan}
    order = np.argsort(wn)
    expected_area = np.trapezoid(expected[order], wn[order])
    energy = expected @ expected
    if energy <= 0 or expected_area <= 0:
        raise ValueError('The known addition must have positive energy and area')
    recovered = spiked - unspiked
    area = np.trapezoid(recovered[order], wn[order])
    return {'available': True, 'area_recovery': float(area / expected_area),
            'gain': float(expected @ recovered / energy),
            'relative_RMSE': float(np.linalg.norm(recovered - expected) / np.sqrt(energy)),
            'expected_area': float(expected_area), 'recovered_area': float(area)}


def diagnostic_flags(values):
    """Read CSV booleans explicitly: the string 'False' is never truthy success."""
    s = pd.Series(values).astype('string').str.strip().str.lower()
    if not s.isin(['true', 'false']).all():
        raise ValueError('Fit success must be explicitly true or false')
    return s.eq('true').to_numpy(bool)


def align_reference_diagnostics(cases, diagnostics):
    """The original solver logged row-number IDs; case_row joins physical IDs."""
    if not cases.sample_id.is_unique or not diagnostics.case_row.is_unique:
        raise ValueError('Reference identities must be unique')
    positions = pd.to_numeric(diagnostics.case_row, errors='raise').to_numpy(float)
    if (not np.isfinite(positions).all() or not np.equal(positions, np.floor(positions)).all()
            or set(positions) != set(range(len(cases)))):
        raise ValueError('Diagnostic case rows must cover the case ledger exactly')
    positions = positions.astype(int)
    physical = cases.sample_id.to_numpy()[positions]
    logged = diagnostics.sample_id.astype(str).to_numpy()
    if not (np.array_equal(logged, physical) or np.array_equal(logged, positions.astype(str))):
        raise ValueError('Diagnostic solver labels conflict with their case rows')
    return diagnostics.rename(columns={'sample_id': 'solver_label'}).assign(sample_id=physical)
