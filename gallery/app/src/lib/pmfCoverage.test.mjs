import assert from 'node:assert/strict'
import { test } from 'node:test'
import fs from 'node:fs'
import { belowAbove, biomassPct, dominantShare, fieldCoverage, matchPmf } from './pmfCoverage.ts'

test('the dominant share is the largest source over the total; biomass is wood + charcoal of the total', () => {
  const labels = ['Wood Burning', 'Charcoal', 'Fossil']
  assert.deepEqual(dominantShare({ 'Wood Burning': 1, Charcoal: 3, Fossil: null }, labels), { source: 'Charcoal', share: 0.75 })
  assert.equal(dominantShare({ 'Wood Burning': 0, Charcoal: 0 }, labels), null)
  assert.equal(biomassPct({ 'Wood Burning': 1, Charcoal: 1, Fossil: 2 }, labels), 50)
  assert.equal(biomassPct({ 'Wood Burning': 1, Fossil: 2 }, labels), null)
})

test('filters match their PMF day by id, else within a day; each PMF day is used once', () => {
  const filters = [{ id: 'A-1', date: '2023-01-03' }, { id: 'A-2', date: '2023-01-06' }, { id: 'A-3', date: '2023-01-10' }, { id: 'A-4', date: '2023-01-11' }]
  const pmf = [{ id: 'A-1', date: '2023-01-03' }, { id: 'x', date: '2023-01-07' }, { id: 'y', date: '2023-01-11' }, { id: 'z', date: '2023-02-01' }]
  const { of, unmatched } = matchPmf(filters, pmf)
  assert.equal(of.get('A-1').id, 'A-1')
  assert.equal(of.get('A-2').id, 'x')
  // A-3 (one day off) and A-4 (same day) both want y: first come, the nearest
  assert.equal(of.get('A-3').id, 'y')
  assert.equal(of.has('A-4'), false)
  assert.deepEqual(unmatched.map((p) => p.id), ['z'])
})

test('below / above counts and per-field coverage', () => {
  assert.deepEqual(belowAbove([30, 31, 50, 70, NaN], [30, 50]), [{ threshold: 30, below: 1, above: 3 }, { threshold: 50, below: 3, above: 1 }])
  const rows = [{ id: 'a', EC: 1, Fe: null }, { id: 'b', EC: 2, Fe: 1 }, { id: 'c', EC: null, Fe: 3 }]
  assert.deepEqual(fieldCoverage(rows, ['EC', 'Fe'], (r) => r.id !== 'c'), [{ field: 'EC', all: 2, pmf: 2 }, { field: 'Fe', all: 2, pmf: 1 }])
})

// dominant_source_threshold_analysis c12 printed n = 102; GF mean 46.2 %, median 45.5 %, below/above 3/99, 31/71, 64/38,
// 92/10, 100/2; K_F mean 47.8 %, median 44.9 %, 3/99, 34/68, 66/36, 85/17, 94/8
const PMF = new URL('../../public/data/pmf.json', import.meta.url)
test('reproduces the notebook\'s dominance distribution from pmf.json', { skip: !fs.existsSync(PMF) }, () => {
  const pmf = JSON.parse(fs.readFileSync(PMF, 'utf8'))
  const labels = pmf.sources.map((s) => s.label)
  const pct = (key) => pmf.rows.map((r) => 100 * dominantShare(r[key], labels).share)
  const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length
  const median = (v) => { const s = [...v].sort((a, b) => a - b); return (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2 }
  const T = [30, 40, 50, 60, 70]
  const gf = pct('fraction'), kf = pct('ugm3')
  assert.equal(gf.length, 102)
  assert.equal(mean(gf).toFixed(1), '46.2')
  assert.equal(median(gf).toFixed(1), '45.5')
  assert.deepEqual(belowAbove(gf, T).map((b) => [b.below, b.above]), [[3, 99], [31, 71], [64, 38], [92, 10], [100, 2]])
  assert.equal(mean(kf).toFixed(1), '47.8')
  assert.equal(median(kf).toFixed(1), '44.9')
  assert.deepEqual(belowAbove(kf, T).map((b) => [b.below, b.above]), [[3, 99], [34, 68], [66, 36], [85, 17], [94, 8]])
  // the export's own dominant_fraction is the GF share
  pmf.rows.forEach((r, i) => assert.ok(Math.abs(100 * r.dominant_fraction - gf[i]) < 1e-3, r.id))
})
