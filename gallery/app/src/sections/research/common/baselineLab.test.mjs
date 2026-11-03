import assert from 'node:assert/strict'
import { test } from 'node:test'
import { line } from 'd3'
import { labCalibrationRows, labChronologyRows, labCohortRows, labInjectionRows, labOpticalSourceIncluded,
  labRepeatValues, labSpectrumValues, selectLabCurves, validateLabData, validateLabSpectrum } from './baselineLab.ts'

const curve = (id, kind, method = 'spline', scan_id) => ({ id, label: id, kind, method, scan_id, wn: [1800, 1617, 1450], y: [.001, -.003, .002] })
const spectra = { schema_version: 1, id: 'ETAD-0139', source: 'native', curves: [curve('mean', 'corrected'), curve('raw', 'raw', 'raw'),
  curve('near-flat', 'corrected', 'spline', '1626'), curve('informative', 'corrected', 'spline', '1668'), curve('other-method', 'corrected', 'neutral')] }

test('native selection preserves signed values and separate scans without averaging or changing the recorded product', () => {
  const original = structuredClone(spectra)
  const mean = selectLabCurves(spectra, 'corrected', 'spline', 'mean')
  assert.deepEqual(mean.curves.map(c => c.id), ['mean'])
  assert.equal(mean.curves[0].y[1], -.003)
  const scans = selectLabCurves(spectra, 'corrected', 'spline', 'scans')
  assert.deepEqual(scans.curves.map(c => c.scan_id), ['1626', '1668'])
  assert.equal(selectLabCurves(spectra, 'corrected', 'all', 'both').curves.length, 4)
  assert.deepEqual(spectra, original)
})

test('raw-only packed spectra visibly become Raw; removed or corrected curves are never invented', () => {
  const packed = { ...spectra, source: 'packed', curves: [curve('packed raw', 'raw', 'raw')] }
  const selected = selectLabCurves(packed, 'corrected', 'spline', 'mean')
  assert.equal(selected.view, 'raw')
  assert.equal(selected.curves[0], packed.curves[0])
  assert.equal(selectLabCurves(packed, 'removed', 'all', 'mean').curves.length, 0)
  assert.equal(selectLabCurves(packed, 'raw', 'all', 'scans').curves.length, 0)
})

test('physical-filter identity, exact paired grids and finite signed data are verified before plotting', () => {
  assert.equal(validateLabSpectrum(spectra, 'ETAD-0139'), null)
  assert.match(validateLabSpectrum(spectra, 'ETAD-0037'), /identity/)
  assert.match(validateLabSpectrum({ ...spectra, curves: [{ ...spectra.curves[0], y: [0, 1] }] }, spectra.id), /grid/)
  assert.equal(validateLabSpectrum({ ...spectra, curves: [{ ...spectra.curves[0], y: [0, null, 1] }] }, spectra.id), null)
  for (const bad of [NaN, Infinity, '0']) assert.match(validateLabSpectrum({ ...spectra, curves: [{ ...spectra.curves[0], y: [0, bad, 1] }] }, spectra.id), /values/)
  assert.match(validateLabSpectrum({ ...spectra, curves: [{ ...spectra.curves[0], wn: [1800, 1450, 1617] }] }, spectra.id), /ordered/)
  assert.match(validateLabData({ schema_version: 1, filters: [{ id: 'ETAD-0139' }, { id: 'ETAD-0139' }], spectra_ids: ['ETAD-0139'] }), /duplicate/)
})

test('unsupported spectral channels stay in place and split lines and uncertainty bounds instead of bridging or becoming zero', () => {
  const original = [.001, -.002, null, .003, .004]
  const values = labSpectrumValues(original)
  assert.equal(values.length, original.length)
  assert.ok(Number.isNaN(values[2]))
  assert.equal(values[1], -.002)
  assert.equal(original[2], null)
  // The shared SpectrumPlot uses this d3 finite-value rule; two supported blocks must be two subpaths.
  const path = line().defined(v => Number.isFinite(v)).x((_, i) => i).y(v => v)(values)
  assert.equal((path.match(/M/g) ?? []).length, 2)
  assert.ok(Number.isNaN(labSpectrumValues([null, .001])[0]), 'band bounds use the same gap-preserving conversion')
  const withBandGap = { ...spectra, curves: [{ ...spectra.curves[0], y: [0, null, 1], lo: [-.1, null, .9], hi: [.1, null, 1.1] }] }
  assert.equal(validateLabSpectrum(withBandGap, spectra.id), null)
  assert.equal(validateLabData({ schema_version: 1, filters: [], spectra_ids: [], curves: withBandGap.curves }), null)
  assert.match(validateLabData({ schema_version: 1, filters: [], spectra_ids: [], curves: [{ ...spectra.curves[0], y: [0, Infinity, 1] }] }), /values/)
  assert.match(validateLabSpectrum({ ...spectra, curves: [{ ...spectra.curves[0], lo: [0, Infinity, 1] }] }, spectra.id), /values/)
})

test('combined optical sources admit public HIPS extensions but never SSR or an unspecified product', () => {
  assert.equal(labOpticalSourceIncluded({ optical_source: 'direct_hips_fabs' }, 'direct'), true)
  assert.equal(labOpticalSourceIncluded({ optical_source: 'public_hips_extension' }, 'direct'), false)
  assert.equal(labOpticalSourceIncluded({ optical_source: 'public_hips_extension' }, 'combined'), true)
  for (const source of ['public_ssr', 'HIPS_SSR_curve', 'none', '']) assert.equal(labOpticalSourceIncluded({ optical_source: source }, 'combined'), false)
})

test('injection graph never pools held-out parents and leave-lot-out parents at matching amplitudes', () => {
  const rows = [
    { method: 'spline', control: 'Held-out blanks · broad', amplitude: .02, recovery_fraction: .2, n_parents: 9, n_finite: 9, n_converged: 7, recovery_converged: .25 },
    { method: 'spline', control: 'Leave-lot-out · broad', amplitude: .02, recovery_fraction: .8, n_parents: 4 },
    { method: 'neutral', control: 'Held-out blanks · broad', amplitude: .01, recovery_fraction: .7, n_parents: 9 },
    { method: 'spline', control: 'Held-out blanks · broad', amplitude: .01, recovery_fraction: .1, n_parents: 9 },
  ]
  const before = structuredClone(rows)
  const result = labInjectionRows(rows, 'Held-out blanks · broad', 'spline')
  assert.deepEqual(result.map(r => [r.amplitude, r.recovery_fraction, r.n_parents]), [[.01, .1, 9], [.02, .2, 9]])
  assert.equal(result[1].n_converged, 7, 'finite nonconverged parent results remain visible, with their convergence counts')
  assert.equal(result[1].recovery_fraction, .2, 'the all-finite result is not silently replaced by converged-only recovery')
  assert.deepEqual(rows, before)
})

test('empty same-known-lot matches remain unavailable; true signed zero remains a valid estimate', () => {
  const rows = [{ comparison: 'same lot', group: 'Addis', n: 0, median_disagreement: null, median_abs_disagreement: null, note: 'No known same-lot matches' },
    { comparison: 'matched dates', group: 'Addis', n: 8, median_disagreement: 0, median_abs_disagreement: 1 }]
  const empty = labCohortRows(rows, 'same lot')
  assert.equal(empty.available.length, 0)
  assert.equal(empty.unavailable[0].note, 'No known same-lot matches')
  assert.equal(labCohortRows(rows, 'matched dates').available[0].median_disagreement, 0)
})

test('quarter curves keep chronology separate from pooled and secondary-population scores', () => {
  const rows = [{ population: 'combined', quarter: '2025Q4', n: 12 }, { population: 'direct', quarter: '2025Q4', n: 9 },
    { population: 'combined', quarter: 'all_test_quarters', n: 183 }, { population: 'combined', quarter: '2024Q1', n: 20 }]
  assert.deepEqual(labChronologyRows(rows, 'combined', false).map(r => r.quarter), ['2024Q1', '2025Q4'])
  assert.deepEqual(labChronologyRows(rows, 'combined', true).map(r => r.n), [183])
  assert.equal(rows[0].quarter, '2025Q4')
})

test('the 905 primary and 906 historical calibration cohorts never pool, and mixed units are refused', () => {
  const base = { method: 'spline', split: 'grouped CV', n: 905, rmse: 2, units: 'µg/filter', calibration_matched: true }
  const rows = [{ ...base, cohort: 'primary 905' }, { ...base, cohort: 'historical 906', n: 906, rmse: 1 },
    { ...base, cohort: 'primary 905', method: 'old coefficients', calibration_matched: false, rmse: 0 },
    { ...base, cohort: 'primary 905', method: 'unfinished', rmse: null }]
  const primary = labCalibrationRows(rows, 'primary 905', 'grouped CV')
  assert.deepEqual(primary.rows.map(r => [r.n, r.rmse]), [[905, 2]])
  assert.equal(primary.sameUnits, true)
  assert.equal(labCalibrationRows([...rows, { ...base, cohort: 'primary 905', units: 'µg/m³' }], 'primary 905', 'grouped CV').sameUnits, false)
})

test('field-blank repeat diagnostics use absolute RMS even when a relative metric was requested', () => {
  const blank = { filter_type: 'FB', raw_relative_difference_rms: 1e9, corrected_relative_difference_rms: 2e9,
    raw_difference_rms: .0001, corrected_difference_rms: .0002 }
  assert.deepEqual(labRepeatValues(blank, true), { x: .0001, y: .0002 })
  assert.deepEqual(labRepeatValues({ ...blank, filter_type: 'PM2.5' }, true), { x: 1e9, y: 2e9 })
  assert.equal(labRepeatValues({ ...blank, corrected_difference_rms: null }, true), null)
})
