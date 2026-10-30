import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { referenceQuantity, rankedReference, groupQuantiles, placeAmong, quantileSorted, filterValue, partsOf } from './referenceRank.ts'

const data = (p) => new URL(`../../public/data/${p}`, import.meta.url)
const ref = JSON.parse(fs.readFileSync(data('improve_site_quantiles.json'), 'utf8'))
const rows = JSON.parse(fs.readFileSync(data('filters.json'), 'utf8')).rows
const close = (a, b, tol) => Math.abs(a - b) <= tol

test('a ratio field splits into its parts only when both are known fields', () => {
  assert.deepEqual(partsOf('OC (FTIR) / EC (FTIR)'), ['OC (FTIR)', 'EC (FTIR)'])
  assert.deepEqual(partsOf('EC (FTIR)'), ['EC (FTIR)'])
  assert.deepEqual(partsOf('Alkane CH / EC (FTIR)'), ['Alkane CH / EC (FTIR)'])
  assert.ok('reason' in referenceQuantity('Alkane CH', 10))
  assert.ok('reason' in referenceQuantity('HIPS BC / HIPS Fabs', 10), 'Fabs over itself is no ratio')
})

test('FTIR carbon stands for TOR carbon, and HIPS BC follows the MAC through Fabs', () => {
  const q = referenceQuantity('OC (FTIR) / EC (FTIR)', 10)
  assert.equal(q.key, 'OC (TOR) / EC (TOR)')
  assert.equal(q.scale, 1)
  assert.equal(q.concPower, 0)
  assert.equal(q.methods.length, 2)
  const bc6 = referenceQuantity('HIPS BC', 6), fabs = referenceQuantity('HIPS Fabs', 6)
  assert.equal(bc6.key, 'HIPS Fabs')
  assert.equal(bc6.concPower, 1)
  const a = rankedReference(ref, 'paired', bc6, 50), b = rankedReference(ref, 'paired', fabs, 50)
  assert.equal(a.length, b.length)
  a.forEach((s, i) => assert.ok(close(s.p50, b[i].p50 / 6, 1e-12) && close(s.p25, b[i].p25 / 6, 1e-12)))
  // a ratio with HIPS BC below scales like a slope (× MAC), above like BC (÷ MAC)
  assert.equal(referenceQuantity('EC (FTIR) / HIPS BC', 6).scale, 6)
  assert.equal(referenceQuantity('HIPS BC / EC (FTIR)', 6).scale, 1 / 6)
  assert.equal(referenceQuantity('EC (FTIR) / PM2.5 mass', 6).concPower, 0)
})

test('ftir_51 11a/11b: the focus sites placed in the IMPROVE ranking (sites with ≥ 50 paired filters)', () => {
  // run_ftir_51 §11 printed these SPARTAN medians and n; the ranks are where searchsorted puts them among the
  // 211 IMPROVE sites (docs/research-questions-2026-10.md: Addis has the second-lowest OC/EC; the AAAR email:
  // less OC per unit of absorption than every one of the 211)
  const expect = {
    'OC (FTIR) / EC (FTIR)': { ETAD: [190, 1.343, 2], INDH: [61, 2.132, 3], CHTS: [160, 2.716, 9], USPA: [130, 3.730, 30] },
    'OC (FTIR) / HIPS Fabs': { ETAD: [190, 0.127, 1], INDH: [61, 0.233, 8], CHTS: [160, 0.268, 19], USPA: [130, 0.433, 101] },
  }
  for (const [field, sites] of Object.entries(expect)) {
    const q = referenceQuantity(field, 10)
    const ranked = rankedReference(ref, 'paired', q, 50)
    assert.equal(ranked.length, 211, field)
    const placed = groupQuantiles(rows, (r) => r.code, q, 'paired', 10)
    for (const [code, [n, median, rank]] of Object.entries(sites)) {
      const s = placed.find((p) => p.key === code)
      assert.equal(s.n, n, `${field} ${code} n`)
      assert.ok(close(s.p50, median, 5e-4), `${field} ${code} median ${s.p50}`)
      assert.equal(placeAmong(ranked, s.p50) + 1, rank, `${field} ${code} rank`)
    }
  }
})

test('per-filter ratios, not a ratio of quantiles; type-7 quantiles; searchsorted placement', () => {
  const rs = [{ a: 1, b: 1 }, { a: 4, b: 1 }, { a: 1, b: 4 }, { a: 10, b: 2 }]
  const [g] = groupQuantiles(rs, () => 'x', { parts: ['a', 'b'] }, 'own', 1)
  assert.deepEqual(g.values, [0.25, 1, 4, 5])
  assert.equal(g.p50, 2.5)                    // median(a)/median(b) would be 2.5/1.5
  assert.equal(quantileSorted([1, 2, 3, 4], 0.25), 1.75)
  assert.equal(filterValue({ a: 1, b: 0 }, { parts: ['a', 'b'] }, 'own'), null)
  assert.equal(filterValue({ a: -1 }, { parts: ['a'] }, 'own'), -1)
  assert.equal(filterValue({ a: 2, 'EC (FTIR)': 1, 'OC (FTIR)': 1 }, { parts: ['a'] }, 'paired'), null, 'paired needs Fabs')
  const ranked = [1, 2, 2, 3].map((p50) => ({ p50 }))
  assert.equal(placeAmong(ranked, 0.5), 0)
  assert.equal(placeAmong(ranked, 2), 1)
  assert.equal(placeAmong(ranked, 9), 4)
})
