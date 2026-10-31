// The Ethiopia page's spectra (spectra.ts) and per-spectrum EC summaries (variants.ts) against scipy and the June
// notebooks' saved / printed numbers
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { ALL, bandArea, flatten, groupOf, MONTH_SPLITS, OTHER, pool, profileStats, rankDesc, savgol, UNDATED } from './spectraMath.ts'
import { ecByCalibration, kSummary, medianRatio, ratioOfMedians } from './variants.ts'

const page = JSON.parse(fs.readFileSync(new URL('../../../../../public/data/research/spectral/ethiopia_calibrations.json', import.meta.url), 'utf8'))
const table = (name) => { const t = page.tables[name]; const cols = Object.keys(t); return t[cols[0]].map((_, i) => Object.fromEntries(cols.map((c) => [c, t[c][i]]))) }
const has = (name) => name in page.tables
const tracked = (f) => {
  const [head, ...lines] = fs.readFileSync(new URL(`../../../../../../../research/ftir_ec_calibration_2026_06_25/tables/${f}`, import.meta.url), 'utf8').trim().split('\n')
  const cols = head.split(',')
  return lines.map((l) => Object.fromEntries(l.split(',').map((v, i) => [cols[i] || 'index', Number.isNaN(Number(v)) ? v : Number(v)])))
}
const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`)

test('savgol reproduces scipy.signal.savgol_filter (interp edges, delta 1)', () => {
  const y = Array.from({ length: 20 }, (_, i) => Math.sin(i * 0.7) + 0.01 * i * i)
  const d2 = [0.0724927407, 0.0724927407, 0.0724927407, 0.0724927407, 0.0724927407, 0.0724927407, 0.1504264213, 0.167018518, 0.1144655084, 0.0174838942,
    -0.0783143562, -0.1278738286, -0.1078859289, -0.0277512785, 0.0748415443, 0.0748415443, 0.0748415443, 0.0748415443, 0.0748415443, 0.0748415443]
  savgol(y, 11, 2, 2).forEach((v, i) => near(v, d2[i], 1e-9, `w11 p2 d2 [${i}]`))
  const d1 = [1.0831109684, 0.5003717396, 0.0466136311, -0.2781633569, -0.5511330656, -0.5272710319, -0.2083936304, 0.2649324267, 0.6795008114, 0.8497398471,
    0.7049897677, 0.3227351099, -0.1078374754, -0.3748166609, -0.3432316515, -0.0185310581, 0.455979669, 0.7679888874, 1.0281332017, 1.2364126119]
  savgol(y, 7, 3, 1).forEach((v, i) => near(v, d1[i], 1e-9, `w7 p3 d1 [${i}]`))
  // a quadratic's second derivative is exact everywhere
  savgol(Array.from({ length: 30 }, (_, i) => 3 * i * i - 2 * i + 1), 11, 2, 2).forEach((v) => near(v, 6, 1e-9, 'quadratic'))
})

test('June 01: the Adama band areas and ranks come back from the spectra', { skip: !has('adama_spectra') }, () => {
  const spec = table('adama_spectra'), wn = spec.map((r) => r.wavenumber)
  for (const r of table('adama_peak_vs_ec')) {
    const flat = flatten(spec.map((s) => s[r.spectrum]), wn)
    near(bandArea(flat, wn, 2800, 3000) / 1000, r.CH_area_2800_3000, 2e-4, `${r.spectrum} CH`)
    near(bandArea(flat, wn, 1650, 1750) / 1000, r.CO_area_1650_1750, 2e-4, `${r.spectrum} C=O`)
  }
  const saved = table('adama_peak_vs_ec')
  assert.deepEqual(rankDesc(saved.map((r) => r.CH_area_2800_3000)), saved.map((r) => r.peak_rank))
  assert.deepEqual(rankDesc(saved.map((r) => r.EC_ftir_ug_m3)), saved.map((r) => r.EC_rank))
})

test('June 04: month splits pool back to the whole set; the placeholder split counts as 04 printed', { skip: !has('etad_month_moments') }, () => {
  const meta = table('etad_spectra_meta'), mom = table('etad_month_moments'), dist = table('etad_spectra_distribution')
  const n = (g) => meta.filter((r) => (r.month == null ? 'none' : String(r.month)) === g).length
  const groups = [...Array.from({ length: 12 }, (_, i) => String(i + 1)), 'none']
  const all = pool(groups.map((g) => [n(g), mom.map((r) => r[`mean ${g}`]), mom.map((r) => r[`sd ${g}`])]))
  assert.equal(all.n, 319)
  all.mean.forEach((v, j) => near(v, dist[j].mean, 0.1, `pooled mean [${j}]`))
  const split = MONTH_SPLITS.find((s) => s.key === 'placeholder')
  const count = (g) => meta.filter((r) => groupOf(r.month, split) === g).length
  assert.equal(count('high-biomass months (Oct–Feb)'), 121)
  assert.equal(count('diesel months (Jun–Aug)'), 70)
  assert.equal(count(OTHER) + count(UNDATED), 128)
  assert.equal(groupOf(5, MONTH_SPLITS[0]), ALL)
})

test("June 13 [9]: the per-spectrum organic-band fractions' medians are 13's printed ETAD profile", { skip: !has('etad_spectra_meta') }, () => {
  const printed = table('band_profile_printed')[0]
  for (const s of profileStats(table('etad_spectra_meta'))) near(s.median, printed[s.band], 0.0006, s.band)
})

test('07 [11], 09 [12] C and 10 [12] B from the per-spectrum EC', () => {
  const ec = ecByCalibration(table('per_filter'), table('calibrations'))
  const cal0 = ec.get('CAL-0 all-nofilter')
  for (const r of tracked('etad_variant_ratio_vs_cal0.csv')) near(medianRatio(ec.get(r.index), cal0), r.median_vs_CAL0, 0.0006, r.index)
  const labels = table('calibrations').filter((c) => String(c.family).startsWith('09')).map((c) => c.calibration)
  const sweep = kSummary(ec, labels)
  const saved = table('component_summary')
  assert.deepEqual(sweep.map((s) => s.k), [3, 10, 18, 21, 25, 40])
  sweep.forEach((s, i) => {
    near(s.median, saved[i].median_EC_ugm3, 0.0006, `k${s.k} median`)
    near(s.mean, saved[i].mean_EC_ugm3, 0.0006, `k${s.k} mean`)
    near(s.pctNegative, saved[i].pct_negative, 0.0006, `k${s.k} % negative`)
  })
  // 10 printed: adding the 29 lot-248 filters changes the median by 1.030×
  near(ratioOfMedians(ec.get('935 both lots, k30 (first major min)'), ec.get('906 lot 251 only, k30')), 1.030, 0.0006, 'lot-248 effect')
})
