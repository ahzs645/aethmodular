// The Linearising HIPS page recomputes, in the browser, what the 10 Nov closure workflow saved: the Deming fits of
// FTIR EC on HIPS (as measured and linearised at each ceiling), the season and lot ratios, and the predicted exponent.
// Inputs: research/hips_linearise/linearise.json (gallery/data/export_research_hips_linearise.py).
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { correction, fitAgainstHips, linearise, noiseGain, predictedExponent, ratioMedians, saturate } from './linearise.ts'

const data = JSON.parse(fs.readFileSync(new URL('../../../../public/data/research/hips_linearise/linearise.json', import.meta.url), 'utf8'))
const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`)

test('the inversion undoes the saturating curve, and its factors', () => {
  for (const ceiling of [1.71, 2.26, 2.7]) for (const u of [0.05, 0.5, 1.2, 3]) near(linearise(saturate(u, ceiling), ceiling), u, 1e-9, `u ${u} at ${ceiling}`)
  assert.ok(Number.isNaN(linearise(2.3, 2.26)))
  near(correction(0.97, 2.26), 1 / (1 - 0.97 / 2.26), 1e-12, 'correction')
  near(noiseGain(0.7 * 2.26, 2.26), 1 / 0.09, 1e-9, 'noise at 70 % of the ceiling')
})

test('the fit as measured and at every saved ceiling matches the workflow', () => {
  const m = fitAgainstHips(data.filters, data.lambda, null)
  assert.equal(m.n, data.fit_measured.n)
  near(m.slope, data.fit_measured.slope, 1e-4, 'measured slope')
  near(m.intercept, data.fit_measured.intercept, 1e-4, 'measured intercept')
  for (const r of data.fits_linearised) {
    const f = fitAgainstHips(data.filters, data.lambda, r.tau_inf)
    assert.equal(f.n, r.n)
    near(f.slope, r.slope_vs_linearised_hips, 1e-3, `slope at ${r.tau_inf}`)
    near(f.intercept, r.intercept_vs_linearised_hips, 1e-3, `intercept at ${r.tau_inf}`)
  }
})

test('season and lot medians at the central ceiling match the saved table', () => {
  const central = data.ceilings.find((c) => c.key === 'central').tau_inf
  for (const s of data.season_lot) {
    const own = data.filters.filter((f) => (s.season === 'all' || f.season === s.season) && (s.lot === 'all' || f.lot === s.lot))
    const r = ratioMedians(own, central)
    assert.equal(r.n, s.n, `${s.season} ${s.lot} n`)
    near(r.measured, s.published_over_hips, 1e-4, `${s.season} ${s.lot} measured`)
    near(r.linearised, s.published_over_linearised_hips, 1e-4, `${s.season} ${s.lot} linearised`)
    near(r.tau, s.median_optical_depth, 2e-4, `${s.season} ${s.lot} optical depth`)
  }
})

test('the predicted Addis exponent at the central ceiling', () => {
  const r = data.fits_linearised.find((f) => f.which === 'central')
  near(predictedExponent(data.filters.map((f) => f.tau), r.tau_inf), r.predicted_hips_exponent, 1e-3, 'predicted exponent')
})
