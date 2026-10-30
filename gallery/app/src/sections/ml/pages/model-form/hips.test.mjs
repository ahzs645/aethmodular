import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { chosenK, sharedRuns, transferMetrics, vipBandShare, vipOverlap } from './hips.ts'

const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`)

test('transfer metrics: OLS of predicted on observed, errors against 1:1', () => {
  const m = transferMetrics([1, 2, 3, 4], [3, 5, 7, 9])
  close(m.slope, 2); close(m.intercept, 1); close(m.R2, 1)
  close(m.bias, 3.5); close(m.MAE, 3.5); close(m.median_bias, 3.5); close(m.RMSE, Math.sqrt((4 + 9 + 16 + 25) / 4))
  assert.equal(m.n, 4)
  assert.equal(transferMetrics([1, 2, NaN], [1, 2, 3]), null)
})

test('chosen k is the lowest mean RMSE, first on a tie', () => {
  assert.equal(chosenK([{ k: 1, rmse: 0.3 }, { k: 2, rmse: 0.2 }, { k: 3, rmse: 0.2 }]), 2)
  assert.equal(chosenK([]), null)
})

test('VIP overlap: Spearman with ties, VIP ≥ 1 sets, top-n and mass', () => {
  const a = [0.5, 1.2, 1.5, 0.9, 2.0], b = [0.4, 1.1, 0.8, 1.3, 1.3]
  const o = vipOverlap(a, b, 1, 2)
  assert.equal(o.important_a_n, 3); assert.equal(o.important_b_n, 3)
  close(o.important_jaccard, 2 / 4)
  assert.equal(o.top_n_overlap, 1) // top-2 of a {4, 2}, of b {3, 4}
  close(o.a_vip_mass_on_b_important, (1.44 + 0.81 + 4) / (0.25 + 1.44 + 2.25 + 0.81 + 4))
  // ranks a: 1 3 4 2 5; b (tie at 1.3 → 4.5): 1 3 2 4.5 4.5
  const ra = [1, 3, 4, 2, 5], rb = [1, 3, 2, 4.5, 4.5]
  const m = (v) => v.reduce((s, x) => s + x, 0) / v.length
  const cov = ra.reduce((s, x, i) => s + (x - m(ra)) * (rb[i] - m(rb)), 0)
  const sd = (v) => Math.sqrt(v.reduce((s, x) => s + (x - m(v)) ** 2, 0))
  close(o.spearman_r, cov / (sd(ra) * sd(rb)))
})

test('band share and shared runs', () => {
  close(vipBandShare([3, 2, 1], [1, 1, 2], 1, 3), (1 + 4) / 6)
  assert.deepEqual(sharedRuns([1, 1, 0, 1, 1, 1], [1, 2, 1, 0, 1, 1]), [[0, 1], [4, 5]])
})

// the exported ftir_08 tables, when the export carries them: the browser's maths reproduces the notebook's displays
const file = new URL('../../../../../public/data/ml/model-form.json', import.meta.url)
const bundle = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null
const t = bundle?.tables ?? {}
const rowsOf = (tab) => tab ? tab[Object.keys(tab)[0]].map((_, i) => Object.fromEntries(Object.keys(tab).map((k) => [k, tab[k][i]]))) : []

test('ftir_08 transfer metrics from the per-filter predictions match the notebook table', { skip: !t.hips_addis_filters && 'no hips_addis_filters in the export' }, () => {
  const f = rowsOf(t.hips_addis_filters), shown = rowsOf(t.hips_transfer)
  for (const [col, model] of [['improve', 'IMPROVE HIPS → Addis external test'], ['addis', 'Addis-only HIPS'], ['naive', 'Naive direct-Fabs sensitivity (unit mismatched)']]) {
    const m = transferMetrics(f.map((r) => r.Fabs), f.map((r) => r[col]))
    const s = shown.find((r) => r.model === model)
    for (const k of ['slope', 'intercept', 'R2', 'RMSE', 'bias', 'median_bias']) close(m[k], s[k], 5e-4 * Math.max(1, Math.abs(s[k])))
    assert.equal(m.n, s.n)
  }
})

test('ftir_08 VIP overlap and band shares from the exported profiles match the notebook', { skip: !t.vip_hips && 'no vip_hips in the export' }, () => {
  const a = t.vip_hips['IMPROVE HIPS'], b = t.vip_hips['Addis HIPS'], wn = t.vip_hips.wn
  const o = vipOverlap(a, b), shown = rowsOf(t.vip_overlap)[1]
  close(o.spearman_r, shown.spearman_r, 1e-4)
  for (const k of ['important_a_n', 'important_b_n', 'top_n_overlap']) assert.equal(o[k], shown[k])
  close(o.important_jaccard, shown.important_jaccard, 1e-4); close(o.a_vip_mass_on_b_important, shown.a_vip_mass_on_b_important, 1e-4)
  const edges = rowsOf(t.vip_band_edges), bands = rowsOf(t.vip_bands)
  for (const e of edges) close(vipBandShare(wn, a, e.low, e.high), bands.find((r) => r.model === 'IMPROVE HIPS' && r.band === e.band).vip2_share, 1e-4)
})
