"""Broad-signal erasure and failed channels must not pass a recovery check."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'research/ftir_hips_chem/scripts'))
from vibes_validation import incremental_recovery, validation_signals, diagnostic_flags, align_reference_diagnostics


def test_incremental_recovery_distinguishes_preservation_erasure_and_inversion():
    wn = np.linspace(4000., 500., 120)
    signal = .05 * validation_signals(wn)['Power-law continuum']
    blank = .002 * np.sin(wn / 200.)
    perfect = incremental_recovery(wn, signal, blank, blank + signal)
    erased = incremental_recovery(wn, signal, blank, blank)
    inverted = incremental_recovery(wn, signal, blank, blank - signal)
    assert perfect['area_recovery'] == pytest.approx(1.)
    assert perfect['relative_RMSE'] == pytest.approx(0., abs=1e-14)
    assert erased['area_recovery'] == 0. and erased['relative_RMSE'] == pytest.approx(1.)
    assert inverted['area_recovery'] == pytest.approx(-1.)
    assert inverted['relative_RMSE'] == pytest.approx(2.)
    reverse = incremental_recovery(wn[::-1], signal[::-1], blank[::-1], (blank + signal)[::-1])
    assert reverse['area_recovery'] == pytest.approx(perfect['area_recovery'])


def test_failed_channels_are_unavailable_and_csv_false_is_not_success():
    wn = np.linspace(4000., 500., 120)
    y = np.ones(120)
    y[3] = np.nan
    result = incremental_recovery(wn, np.ones(120), np.zeros(120), y)
    assert result['available'] is False and np.isnan(result['area_recovery'])
    np.testing.assert_array_equal(diagnostic_flags(['False', 'true', False, True]), [False, True, False, True])
    with pytest.raises(ValueError):
        diagnostic_flags(['unknown'])
    with pytest.raises(ValueError):
        incremental_recovery(wn, wn[:-1], wn, wn)
    with pytest.raises(ValueError):
        incremental_recovery(wn, np.zeros(120), wn, wn)


def test_solver_row_labels_join_to_physical_filters_with_no_guessing():
    cases = pd.DataFrame({'sample_id': ['etad:459', 'etad:1017']})
    diag = pd.DataFrame({'sample_id': [1, 0], 'case_row': [1, 0]})
    assert align_reference_diagnostics(cases, diag).sample_id.tolist() == ['etad:1017', 'etad:459']
    with pytest.raises(ValueError):
        align_reference_diagnostics(cases, diag.assign(sample_id=[0, 1]))
    with pytest.raises(ValueError):
        align_reference_diagnostics(cases, diag.assign(case_row=[0, 0]))
