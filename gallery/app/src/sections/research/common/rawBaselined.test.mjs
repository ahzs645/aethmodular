// ftir_52 §7 (rawBaselined.ts) against the notebook's printed cell [21] and displayed tables, on the exported matrices:
// raw_sites + raw_pairs (recomputed from the Davis Data raw spectra) and site_correlation (the notebook's own CSV)
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { logHistogram, members, nearest, neighbours, networkSummary, siteSummary, spaces } from './rawBaselined.ts'

const data = JSON.parse(fs.readFileSync(new URL('../../../../public/data/research/spectral/network_map.json', import.meta.url), 'utf8'))
const s = spaces(data.tables.raw_sites, data.tables.raw_pairs, data.tables.site_correlation)
const at = (name) => s.names.indexOf(name)

test('both spaces cover the same 151 IMPROVE sites and the five SPARTAN targets', () => {
  assert.ok(s)
  assert.equal(s.names.length, 156)
  assert.equal(members(s, false).length, 151)
  assert.deepEqual(s.names.filter((_, i) => s.spartan[i]), ['*Addis', '*Bishoftu', '*Delhi', '*Beijing', '*JPL'])
  assert.deepEqual(s.names.filter((_, i) => s.spartan[i]).map((n) => s.n[at(n)]), [239, 26, 152, 192, 158])
  for (const i of s.names.keys()) for (const j of s.names.keys()) if (i !== j) assert.ok(Number.isFinite(s.raw[i][j]) && Number.isFinite(s.baselined[i][j]), `${s.names[i]}–${s.names[j]}`)
})

test('ftir_52 [21]: ρ 0.587, median pair r 0.9990 raw and 0.9681 baselined, 139/151 sites change their nearest site', () => {
  const n = networkSummary(s, members(s, false))
  assert.equal(n.sites, 151)
  assert.equal(n.pairs, 151 * 150 / 2)
  assert.equal(Math.round(n.rho * 1000) / 1000, 0.587)
  assert.equal(Math.round(n.medianRaw * 1e4) / 1e4, 0.999)
  assert.equal(Math.round(n.medianBaselined * 1e4) / 1e4, 0.9681)
  assert.equal(n.changed, 139)
  // the figure's "all above the line": baselining separates nearly every pair
  assert.ok(n.separated / n.pairs > 0.99)
})

test('ftir_52 [21] displayed rows: nearest raw, nearest baselined, rank of the raw nearest after baselining', () => {
  const imp = members(s, false)
  const shown = { YOSE1: ['ACAD1', 'CABI1', 46], SAMA1: ['JARI1', 'SWAN1', 24], PMRF1: ['SENE1', 'HEGL1', 34], BRID1: ['MOZI1', 'BOLA1', 6],
    CEBL1: ['MELA1', 'DOSO1', 2], GRSM1: ['JARI1', 'CACR1', 3], CAPI1: ['GRBA1', 'CHIR1', 24], TALL1: ['WIMO1', 'GRRI1', 10],
    CHAS1: ['SAGO1', 'ROMA1', 8], TRCR1: ['MAKA2', 'BRID1', 14], BOLA1: ['WHRI1', 'BRID1', 12], GRBA1: ['CAPI1', 'WHRI1', 7] }
  for (const [site, [raw, base, rank]] of Object.entries(shown)) {
    const i = at(site)
    assert.equal(s.names[nearest(s.raw[i], i, imp)], raw, `${site} raw`)
    assert.equal(s.names[nearest(s.baselined[i], i, imp)], base, `${site} baselined`)
    assert.equal(neighbours(s, i, imp).find((n) => n.rankRaw === 1).rankBaselined, rank, `${site} rank after`)
  }
})

test('ftir_52 [23]: Addis\'s nearest IMPROVE site is PUSO1 (0.9996) raw and NOGA1 (0.9902) baselined', () => {
  const imp = members(s, false), a = at('*Addis')
  const sum = siteSummary(neighbours(s, a, imp), 10)
  assert.equal(s.names[sum.nearestRaw.j], 'PUSO1')
  assert.equal(Math.round((1 - sum.nearestRaw.raw) * 1e4) / 1e4, 0.9996)
  assert.equal(s.names[sum.nearestBaselined.j], 'NOGA1')
  assert.equal(Math.round((1 - sum.nearestBaselined.baselined) * 1e4) / 1e4, 0.9902)
  assert.equal(sum.partners, 151)
})

test('log histogram: 60 log bins, every value counted once', () => {
  const values = [1e-4, 2e-4, 5e-3, 0.01, 0.4]
  const h = logHistogram([values], 1e-4, 0.4, 60)
  assert.equal(h.edges.length, 61)
  assert.equal(h.counts[0].reduce((a, b) => a + b, 0), values.length)
  assert.equal(h.counts[0][0], 1)
  assert.equal(h.counts[0][59], 1)
})
