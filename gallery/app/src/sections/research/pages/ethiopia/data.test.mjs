// The Ethiopia page's July07 / spartan_ec joins (data.ts) against the notebooks' saved tables in the page's export
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { adamaPoints, galleryId, residualDiagnostics, smokePoints, smokeScores, SMOKE_SETS } from './data.ts'

const data = (f) => JSON.parse(fs.readFileSync(new URL(`../../../../../public/data/${f}`, import.meta.url), 'utf8'))
const page = data('research/spectral/ethiopia_calibrations.json')
const table = (name) => { const t = page.tables[name]; const cols = Object.keys(t); return t[cols[0]].map((_, i) => Object.fromEntries(cols.map((c) => [c, t[c][i]]))) }
const gallery = data('filters.json').rows.filter((r) => !r.excluded)
const tables = { addis: table('smoke_addis'), etad: table('smoke_etad'), lot251: table('lot251_set') }
const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`)

test('gallery ids drop the scan suffix', () => {
  assert.equal(galleryId('ETAD-0017-1'), 'ETAD-0017')
  assert.equal(galleryId('ETAD-0249-1'), 'ETAD-0249')
})

test('the scorecard reproduces 07-07 regular vs smoke on its own 259 spectra (Fabs / 10)', () => {
  const pts = tables.etad.map((t) => ({ set: 'spectrum', hips: t.Fabs / 10, cal0: t.EC_regular, smoke: t.EC_smoke, reported: t.EC_reported ?? NaN }))
  const s = smokeScores(pts, 10).filter((r) => r.set === SMOKE_SETS.spectrum)
  const saved = table('smoke_verdict_etad')
  for (const [i, j] of [[1, 0], [2, 1], [0, 2]]) {
    assert.equal(s[i].n, saved[j].n)
    near(s[i].slope, saved[j].slope, 0.005, `${saved[j].calibration} slope`)
    near(s[i].intercept, saved[j].intercept, 0.005, `${saved[j].calibration} intercept`)
    near(s[i].r2, saved[j].R2, 0.0005, `${saved[j].calibration} R²`)
    near(s[i].impliedMac, saved[j].implied_MAC, 0.05, `${saved[j].calibration} implied MAC`)
  }
  near(s[1].gap, saved[0].median_HIPS_minus_EC, 0.005, 'CAL-0 median gap')
  near(s[2].gap, saved[1].median_HIPS_minus_EC, 0.005, 'smoke median gap')
})

test('on the gallery rows: 189 Addis filters, 07-07 before-CAL-0 to two decimals, slopes follow the MAC', () => {
  const pts = smokePoints(gallery, tables, 10)
  const n = (set) => pts.filter((p) => p.set === set).length
  assert.equal(n('filter'), 189)
  // 02's 156 less ETAD-0248, the PM10 filter the gallery excludes
  assert.equal(n('lot251'), 155)
  // the 259 spectra less the 49 from ETAD-0249 on, which are not gallery rows
  assert.equal(n('spectrum'), tables.etad.filter((t) => gallery.some((r) => r.id === galleryId(t.ExternalFilterId))).length)
  const s = smokeScores(pts, 10).filter((r) => r.set === SMOKE_SETS.filter)
  const saved = table('smoke_verdict_addis')
  // that notebook regressed HIPS on EC; its 2024-08-07 point averages ETAD-0241 with the PM10 ETAD-0248, which the gallery excludes
  for (const [i, j] of [[0, 0], [2, 1], [1, 2]]) {
    near(s[i].slopeHipsOnEc, saved[j].slope, 0.01, `${saved[j].calibration} slope`)
    near(s[i].interceptHipsOnEc, saved[j].intercept, 0.03, `${saved[j].calibration} intercept`)
    near(s[i].r2, saved[j].R2, 0.005, `${saved[j].calibration} R²`)
  }
  const at6 = smokeScores(smokePoints(gallery, tables, 6), 6).filter((r) => r.set === SMOKE_SETS.filter)
  for (let i = 0; i < 3; i++) {
    near(at6[i].slope, s[i].slope * 0.6, 1e-9, 'slope × 6/10')
    near(at6[i].intercept, s[i].intercept, 1e-9, 'intercept stays')
    near(at6[i].impliedMac, s[i].impliedMac, 1e-9, 'implied MAC stays')
  }
})

test('Adama: one footing for every calibration, against the date-paired quartz TOR', () => {
  const pts = adamaPoints({ variants: table('adama_variants'), tor: table('adama_tor'), han: table('adama_han'), june15: table('adama'),
    calibration: table('adama_calibration'), faithful: table('adama_faithful'), optimized: table('adama_optimized'), stabilized: table('adama_stabilized') })
  const cals = [...new Set(pts.map((p) => p.calibration))]
  assert.equal(cals.length, 13)
  for (const c of cals) assert.equal(pts.filter((p) => p.calibration === c).length, 5, c)
  // June 02's general / TOR ratios, and the quartz each PTFE filter pairs with
  for (const r of table('adama_tor')) {
    const p = pts.find((q) => q.calibration.startsWith('Deployed general') && q.ptfe === r.ptfe_FilterId)
    assert.equal(p.quartz, r.quartz_FilterId)
    near(p.ratioConc, r['general/Tor'], 0.001, `${r.ptfe_FilterId} general/TOR`)
  }
  // spartan_ec 07's stabilized biomass / TOR is on µg per filter; June 15's NEW is µg/m³ as printed
  for (const r of table('adama_stabilized')) near(pts.find((q) => q.calibration.startsWith('Biomass stabilized') && q.ptfe === r.FilterId).ratioMass, r.EC_biomass / r.ECTR, 1e-3, r.FilterId)
  const variants = table('adama_variants')
  for (const r of table('adama')) {
    const ptfe = variants.find((v) => v.barcode === r.barcode).FilterId
    near(pts.find((q) => q.calibration.startsWith('NEW') && q.ptfe === ptfe).ftirConc, r['NEW lot-248+251'], 1e-9, ptfe)
  }
})

test("spartan_ec 08's residual diagnostics: the full-fit 3σ line is round 1's", () => {
  const d = residualDiagnostics(table('calibration_all'))
  const rounds = table('cleaning_rounds')
  if (d.points.length === 906) {
    near(d.threshold, rounds[0]['3σ threshold (µg)'], 0.05, '3σ')
    assert.equal(d.beyond, rounds[0].dropped)
  }
  assert.equal(d.points.filter((p) => p.removed).length, table('removed').length)
})
