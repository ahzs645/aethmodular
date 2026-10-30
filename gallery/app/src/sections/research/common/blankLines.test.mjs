// ftir_47's blank lines per deployed line (blankLines.ts) on the gallery's HIPS optics export, against numpy on the
// same rows and against the notebook's displayed site table
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { exposeSamples, fitBlankLines, lineKey, linFit, networkLines, polyval, poolLots, quadFit, recomputeCheck, rSuppression, siteExposure } from './blankLines.ts'

const data = (f) => JSON.parse(fs.readFileSync(new URL(`../../../../public/data/${f}`, import.meta.url), 'utf8'))
const optics = data('research/hips_optics/filters.json')
const rows = optics.rows.map((r) => Object.fromEntries(optics.columns.map((c, i) => [c, r[i]])))
const network = data('calibration_runs.json').hips.blanks
const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`)

const lined = (o) => o.cal_intercept !== null && o.cal_slope !== null && o.R1 !== null && o.T1 !== null
const blanks = rows.filter((o) => o.type === 'FB' && lined(o)).map((o) => ({ lot: o.lot, intercept: o.cal_intercept, slope: o.cal_slope, R1: o.R1, T1: o.T1, date: o.analysis_date }))
const samples = rows.filter((o) => o.type !== 'FB' && lined(o) && o.fabs !== null && o.volume_m3 > 0 && o.area_cm2)
  .map((o) => ({ id: o.id, code: o.code, lot: o.lot, intercept: o.cal_intercept, slope: o.cal_slope, R1: o.R1, T1: o.T1, fabs: o.fabs, dv: (100 * o.area_cm2) / o.volume_m3 }))

test('the polynomial fits are np.polyfit', () => {
  const xs = [1, 2, 3, 4, 5], ys = xs.map((x) => 2 - 3 * x + 0.5 * x * x)
  quadFit(xs, ys).forEach((c, i) => near(c, [2, -3, 0.5][i], 1e-9, `quad c${i}`))
  linFit([0, 1, 2], [1, 3, 5]).forEach((c, i) => near(c, [1, 2][i], 1e-12, `lin c${i}`))
  assert.equal(polyval([1, 2, 3], 2), 17)
  assert.equal(lineKey({ lot: '251', intercept: 1416.2, slope: -2.783 }), '251|I=1416.200|S=-2.7830')
})

test('the focus sites\' field blanks refit per deployed line as numpy does', () => {
  const lines = fitBlankLines(blanks)
  assert.equal(lines.length, 5)
  const l = lines.find((x) => x.key === '251|I=1416.200|S=-2.7830')
  assert.equal(l.n, 45)
  near(l.lin[0], 1425.69765821, 1e-6, 'lin a0'); near(l.lin[1], -2.86578319, 1e-8, 'lin a1')
  near(l.quad[0], 1501.81201, 1e-3, 'quad a0'); near(l.quad[1], -3.7612882, 1e-6, 'quad a1'); near(l.quad[2], 2.61445135e-3, 1e-9, 'quad a2')
  near(l.rmsLin, 3.739413315, 1e-6, 'rms lin'); near(l.rmsQuad, 3.678123080, 1e-6, 'rms quad'); near(l.rmsDeployed, 3.923145408, 1e-6, 'rms deployed')
  // two lines have four blanks each here (the network has 16 and 63): no refit
  assert.deepEqual(lines.filter((x) => !x.lin).map((x) => x.key).sort(), ['248|I=2262.000|S=-5.3200', '251|I=2423.600|S=-6.0330'])
  const lot = poolLots(blanks, lines).find((x) => x.lot === '251')
  assert.equal(lot.n, 73)
  near(lot.rmsQuad, 19.33535611, 1e-5, 'lot 251 pooled quad rms')
  assert.equal(lot.shareInterval, true)
})

test('the deployed line reproduces the shipped Fabs', () => {
  const chk = recomputeCheck(exposeSamples(samples, [], []))
  assert.equal(chk.n, 546)
  assert.ok(chk.maxDiff < 0.02, `max |Δ| ${chk.maxDiff}`)
})

test('the network fits on the focus samples give ftir_47 §2\'s site table', () => {
  const lines = networkLines(network)
  assert.equal(lines.length, 9)
  const site = Object.fromEntries(siteExposure(exposeSamples(samples, lines, [])).map((s) => [s.code, s]))
  // numpy on the same rows
  near(site.ETAD.below, 0.44210526, 1e-8, 'ETAD below'); near(site.ETAD.med.line_quad, 0.207581837, 1e-6, 'ETAD median |ΔFabs|')
  near(site.ETAD.p90.line_quad, 0.464114745, 1e-6, 'ETAD p90'); near(site.USPA.above, 0.084615385, 1e-8, 'USPA above')
  near(site.CHTS.med.line_quad, 0.335959779, 1e-6, 'CHTS median'); near(site.INDH.p90.line_quad, 1.377548583, 1e-6, 'INDH p90')
  // the notebook's displayed table (whole SPARTAN batch: more Delhi samples than the gallery's)
  near(site.ETAD.below, 0.439, 0.01, 'ftir_47 ETAD below'); near(site.ETAD.med.line_quad, 0.206, 0.01, 'ftir_47 ETAD median')
  near(site.USPA.above, 0.076, 0.01, 'ftir_47 USPA above'); near(site.USPA.med.line_quad, 0.241, 0.01, 'ftir_47 USPA median')
  near(site.CHTS.above, 0.036, 0.01, 'ftir_47 CHTS above'); near(site.CHTS.med.line_quad, 0.313, 0.03, 'ftir_47 CHTS median')
})

test('R suppression against the deployed line', () => {
  const r = rSuppression({ R1: 100, T1: 1000, intercept: 1500, slope: -2.5 })
  near(r.rExpected, 200, 1e-12, 'R expected'); near(r.pct, 50, 1e-12, 'suppression')
})
