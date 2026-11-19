// The Linearising HIPS page recomputes, in the browser, what the 10 Nov closure workflow saved: the Deming fits of
// FTIR EC on HIPS (as measured and linearised at each ceiling), the season and lot ratios, and the predicted exponent.
// Inputs: hips_linearise/linearise.json (gallery/data/export_hips_linearise.py).
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { CENTRAL_CEILING, correction, DEPOSIT_AREA_CM2, EXPECTED_FROM, fitAgainstHips, HIPS_SITE_READINGS, IMPROVE_CEILINGS, linearLevel, linearise, noiseGain, opticalDepth, parseHipsScope, predictedExponent, previousLoadingGroups, ratioMedians, saturate, scopeSites, SEEN_SITES, siteExposure, statusOf, underRead } from './hipsLinearise.ts'
import { HIPS_TAU, HIPS_UNDER_READ, withHipsLoading } from './viewFields.ts'

const data = JSON.parse(fs.readFileSync(new URL('../../public/data/hips_linearise/linearise.json', import.meta.url), 'utf8'))
const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`)

test('the inversion undoes the saturating curve, and its factors', () => {
  for (const ceiling of [1.71, 2.26, 2.7]) for (const u of [0.05, 0.5, 1.2, 3]) near(linearise(saturate(u, ceiling), ceiling), u, 1e-9, `u ${u} at ${ceiling}`)
  assert.ok(Number.isNaN(linearise(2.3, 2.26)))
  near(correction(0.97, 2.26), 1 / (1 - 0.97 / 2.26), 1e-12, 'correction')
  near(noiseGain(0.7 * 2.26, 2.26), 1 / 0.09, 1e-9, 'noise at 70 % of the ceiling')
})

test('SPARTAN bins use the previous eligible sampling date within the same site', () => {
  const sample = (id, site, date, loading) => ({ id, site, date, loading, tau: loading / 10 })
  const samples = [
    sample('A4', 'AAAA', '2024-01-20', 3),
    sample('A1', 'AAAA', '2024-01-01', 1),
    sample('A2', 'AAAA', '2024-01-01', 3),
    sample('B1', 'BBBB', '2024-01-09', 0),
    sample('A3', 'AAAA', '2024-01-10', 6),
    sample('B2', 'BBBB', '2024-01-10', 1),
    sample('A5', 'AAAA', '2024-01-21', 9),
  ]
  assert.deepEqual(previousLoadingGroups(samples, [0, 2, 3]), [
    { lower: 2, upper: 3, rows: [['AAAA', '2024-01-10', 2, 6, 0.6, 'A3']] },
    { lower: 3, upper: null, rows: [['AAAA', '2024-01-21', 3, 9, 0.9, 'A5']] },
  ])
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

test('a site level keeps the filters past the ceiling instead of dropping them', () => {
  const filters = [{ tau: 0.5, hips_bc: 2 }, { tau: 1, hips_bc: 4 }, { tau: 1.5, hips_bc: 6 }]
  near(linearLevel(filters, null, 'median'), 4, 1e-12, 'as measured')
  near(linearLevel(filters, 2, 'median'), 4 * correction(1, 2), 1e-12, 'median at a ceiling above every filter')
  near(linearLevel(filters, 2, 'mean'), (2 * correction(0.5, 2) + 4 * correction(1, 2) + 6 * correction(1.5, 2)) / 3, 1e-12, 'mean')
  // one filter past the ceiling: the median is still the middle filter's value, the mean has no finite value
  near(linearLevel(filters, 1.4, 'median'), 4 * correction(1, 1.4), 1e-12, 'median with one filter past the ceiling')
  assert.equal(linearLevel(filters, 1.4, 'mean'), Infinity)
  assert.equal(linearLevel(filters, 0.9, 'median'), Infinity)
  // the Addis median on the export, at the central ceiling and as measured
  const central = data.ceilings.find((c) => c.key === 'central').tau_inf
  assert.ok(linearLevel(data.filters, central, 'median') / linearLevel(data.filters, null, 'median') > 1.6)
})

test('the ceilings the gallery-wide view offers are the committed IMPROVE ones, and the optical depth is the workflow\'s', () => {
  assert.equal(DEPOSIT_AREA_CM2, data.deposit_area_cm2)
  assert.deepEqual(IMPROVE_CEILINGS.map((c) => c.key), data.ceilings.map((c) => c.key))
  for (const c of data.ceilings) near(IMPROVE_CEILINGS.find((k) => k.key === c.key).tau_inf, c.tau_inf, 1e-9, `ceiling ${c.key}`)
  // each Addis filter's tau from its HIPS BC (x MAC) and volume, as the workflow's load() saved it
  for (const f of data.filters.filter((x) => x.volume_known).slice(0, 40)) near(opticalDepth(f.hips_bc * data.mac, f.volume_m3), f.tau, 2e-3, `tau of ${f.id}`)
})

// The linearised view can cover every site, the sites where the bend is expected, or those where it is seen. The
// readings are the 11 Nov screen's (docs/hips-network-2026-11-11.json, bundled as hipsSites.json; read here from the
// procedure page's bundle, which carries the same tables), and "expected" is recomputed in the browser from the
// filters themselves, so it must give the screen's own list on the same table.
const procedure = JSON.parse(fs.readFileSync(new URL('../../public/data/meetings/hips-procedure-20261111.json', import.meta.url), 'utf8')).tables
const recordsOf = (t) => t.code.map((_, i) => Object.fromEntries(Object.keys(t).map((k) => [k, t[k][i]])))
const screen = { identified: recordsOf(procedure.found), checked: recordsOf(procedure.checked) }
const spartan = JSON.parse(fs.readFileSync(new URL('../../public/data/spartan_reference.json', import.meta.url), 'utf8'))

test('the bundled site readings are the 11 Nov screen\'s', () => {
  assert.equal(HIPS_SITE_READINGS.length, screen.identified.length)
  const checked = new Map(screen.checked.map((r) => [r.code, r]))
  for (const r of screen.identified) {
    const own = HIPS_SITE_READINGS.find((x) => x.code === r.code)
    assert.equal(own.reading, r.reading, `${r.code} reading`)
    assert.equal(own.verdict, checked.get(r.code).verdict, `${r.code} verdict`)
    near(own.expected_under_read, r.expected_under_read, 1e-4, `${r.code} under-read`)
  }
  // the screen's ceiling is the central one: Addis's median filter, read back through it
  near(HIPS_SITE_READINGS[0].expected_under_read, underRead(HIPS_SITE_READINGS[0].median_tau, CENTRAL_CEILING), 1e-4, 'the screen\'s ceiling')
  assert.deepEqual([...SEEN_SITES].sort(), ['BDDU', 'CHTS', 'CLST', 'ETAD', 'INDH', 'TWKA', 'ZAJB', 'ZAPR'])
})

test('each site\'s exposure from its own filters gives the screen\'s medians and the same expected sites', () => {
  const rows = withHipsLoading(spartan.rows.filter((r) => !r.excluded))
  const exposure = siteExposure(rows.map((r) => ({ code: r.code, tau: r[HIPS_TAU] })))
  assert.equal(exposure.size, screen.identified.length)
  for (const r of screen.identified) {
    const e = exposure.get(r.code)
    assert.equal(e.n, r.filters, `${r.code} filters`)
    near(e.medianTau, r.median_tau, 1e-4, `${r.code} median optical depth`)
    near(e.underRead, r.expected_under_read, 1e-4, `${r.code} expected under-read`)
    assert.equal(e.expected, r.expected_under_read > EXPECTED_FROM, `${r.code} expected`)
    // every site the screen calls expected and seen is one the rule calls expected
    if (SEEN_SITES.has(r.code)) assert.ok(e.expected, `${r.code} is seen, so expected`)
  }
  assert.equal(scopeSites('all', exposure), null)
  assert.equal(scopeSites('expected', exposure).size, screen.identified.filter((r) => r.expected_under_read > EXPECTED_FROM).length)
  assert.deepEqual([...scopeSites('seen', exposure)].sort(), [...SEEN_SITES].sort())
  // a site that is not on file is in no scope, and an IMPROVE site has no reading, so "seen" leaves it as measured
  const light = siteExposure([{ code: 'ACAD1', tau: 0.05 }, { code: 'ACAD1', tau: 0.07 }, { code: 'DARK1', tau: 0.6 }, { code: 'DARK1', tau: NaN }])
  assert.deepEqual([...scopeSites('expected', light)], ['DARK1'])
  assert.equal(light.get('DARK1').n, 1)
  assert.equal(scopeSites('seen', light).size, 0)
  // the per-filter field is the under-read in percent at the central ceiling
  const addis = rows.find((r) => r.code === 'ETAD' && r[HIPS_TAU] > 0)
  near(addis[HIPS_UNDER_READ], 100 * underRead(addis[HIPS_TAU]), 1e-9, 'under-read field')
})

test('statuses and scopes', () => {
  assert.deepEqual([0.1, 0.2, 0.25, 0.3, 0.31].map(statusOf), ['use, linearised', 'use, linearised', 'caution', 'caution', 'shape only'])
  near(underRead(CENTRAL_CEILING * 2), 1, 1e-12, 'past the ceiling')
  assert.deepEqual(['expected', 'seen', 'all', null, 'nonsense'].map(parseHipsScope), ['expected', 'seen', 'all', 'all', 'all'])
})
