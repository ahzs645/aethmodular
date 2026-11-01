import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { referenceQuantity, rankedReference, groupQuantiles, placeAmong, filterValue, partsOf, rankNetworks, rankWithActiveSites, spartanQuantity, improveQuantity, NETWORK_RANK_PRESETS } from './referenceRank.ts'
import { quantileSorted } from './stats.ts'

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

const site = (code, p50, name = code) => ({ code, name, n: 100, p25: p50 / 2, p50, p75: p50 * 2 })

test('network modes rank their own unique sites, including when an IMPROVE site is also selected', () => {
  const im = [site('A', 2), site('B', 4), site('A', 2)]
  const sp = [site('ETAD', 1, 'Same label'), site('INDH', 3, 'Same label'), site('ETAD', 1)]
  const both = rankNetworks(im, sp, 'both')
  assert.deepEqual(both.map(s => [s.network, s.code, s.rank]), [['SPARTAN', 'ETAD', 1], ['IMPROVE', 'A', 2], ['SPARTAN', 'INDH', 3], ['IMPROVE', 'B', 4]])
  assert.deepEqual(rankNetworks(im, sp, 'spartan', 'reference').map(s => [s.code, s.rank]), [['ETAD', 1], ['INDH', 2]])
  assert.deepEqual(rankNetworks(im, sp, 'improve').map(s => [s.code, s.rank]), [['A', 1], ['B', 2]])
})

test('the original notebook placement remains available, with ties and both ends handled', () => {
  const ranked = rankNetworks([site('A', 2), site('B', 4)], [site('ETAD', 1), site('INDH', 2), site('CHTS', 8)], 'both', 'reference')
  assert.deepEqual(ranked.filter(s => s.network === 'SPARTAN').map(s => [s.code, s.rank]), [['ETAD', 1], ['INDH', 1], ['CHTS', 3]])
  assert.deepEqual(rankNetworks([site('A', 2)], [site('ETAD', 2), site('INDH', 3)], 'both').map(s => s.rank), [1, 1, 3])
})

test('every notebook ratio uses the corrected carbon mapping and has exported IMPROVE quantiles', () => {
  assert.equal(NETWORK_RANK_PRESETS.length, 6)
  for (const preset of NETWORK_RANK_PRESETS) {
    const q = referenceQuantity(preset.field, 10)
    assert.ok(!('reason' in q), preset.field)
    assert.ok(rankedReference(ref, 'paired', q, 50).length > 0, preset.field)
  }
  assert.deepEqual(spartanQuantity({ parts: ['OC (TOR)', 'EC (TOR)'] }), { parts: ['OC (FTIR)', 'EC (FTIR)'] })
  assert.deepEqual(spartanQuantity({ parts: ['Iron', 'HIPS Fabs'] }), { parts: ['Iron', 'HIPS Fabs'] })
  assert.deepEqual(improveQuantity({ parts: ['EC (FTIR)', 'PM2.5 mass'] }), { parts: ['EC (TOR)', 'PM2.5 mass'] })
  assert.deepEqual(improveQuantity({ parts: ['HIPS BC', 'EC (FTIR)'] }), { parts: ['HIPS BC', 'EC (TOR)'] })
})

test('active Samples sites keep their colours and one mark in every comparison background', () => {
  const im = [site('A', 2), site('B', 4)]
  const sp = [site('ETAD', 1), site('INDH', 3)]
  const active = [{ ...site('ETAD', 5), network: 'SPARTAN', color: '#abc123' }, { ...site('A', 1), network: 'IMPROVE', color: '#123abc' }]
  for (const mode of ['both', 'improve', 'spartan']) {
    const { sites, nReference } = rankWithActiveSites(im, sp, mode, 'together', active)
    assert.equal(nReference, mode === 'both' ? 4 : 2)
    for (const s of active) {
      const drawn = sites.filter(p => p.code === s.code && p.network === s.network)
      assert.equal(drawn.length, 1)
      assert.equal(drawn[0].active, true)
      assert.equal(drawn[0].color, s.color)
      assert.equal(drawn[0].p50, s.p50, 'the active filtered value replaces the exported site median')
    }
  }
  const onlyIm = rankWithActiveSites(im, sp, 'improve', 'together', active)
  assert.equal(onlyIm.sites.find(s => s.code === 'ETAD').rank, 3, 'active SPARTAN is placed beyond both IMPROVE medians')
  assert.equal(onlyIm.sites.find(s => s.code === 'A').rank, 1, 'a site is compared against the background other than itself')
})

test('active sites outside minimum-count thresholds do not enlarge the reference, including an empty background', () => {
  const active = [{ ...site('ETAD', 5), n: 1, network: 'SPARTAN', color: '#abc123' }]
  const noBackground = rankWithActiveSites([], [], 'improve', 'together', active)
  assert.equal(noBackground.nReference, 0)
  assert.equal(noBackground.sites.length, 1)
  const notebook = rankWithActiveSites([site('A', 2)], [site('INDH', 3)], 'both', 'reference', active)
  assert.equal(notebook.nReference, 1)
  assert.equal(notebook.sites.find(s => s.active).rank, 2)
})
