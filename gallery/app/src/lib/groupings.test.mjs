import assert from 'node:assert/strict'
import { test } from 'node:test'
import { BUILT_IN_GROUPINGS, SPLITS, applyGrouping, bandOf, colorsFor, cutsFor, groupingsParam, labelsFor, mergeGroupings, parseGroupings, rangeText, siteMedians } from './groupings.ts'

const def = (o) => ({ id: 'g1', name: 'Iron', field: 'Iron', kind: 'cutoffs', cuts: [], labels: [], ...o })

test('bands are half-open like pd.cut(right=False): a value on a cutoff goes up', () => {
  const cuts = [2, 4]
  assert.deepEqual([0, 1.99, 2, 3.9, 4, 10].map((v) => bandOf(cuts, v)), [0, 0, 1, 1, 2, 2])
  assert.equal(bandOf(cuts, null), null)
  assert.equal(bandOf(cuts, NaN), null)
  assert.equal(rangeText(cuts, 0), '< 2.00')
  assert.equal(rangeText(cuts, 1), '2.00 to < 4.00')
  assert.equal(rangeText(cuts, 2), '≥ 4.00')
})

test('quantile splits follow the values; labels default to Low / Mid / High', () => {
  const values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
  assert.deepEqual(cutsFor(def({ kind: 'median' }), values), [5])
  const t = cutsFor(def({ kind: 'tertiles' }), values)
  assert.equal(t.length, 2)
  assert.deepEqual(labelsFor(def({ kind: 'tertiles' }), t), ['Low', 'Mid', 'High'])
  // the reader's labels win, blanks fall back
  assert.deepEqual(labelsFor(def({ kind: 'median', labels: ['low iron', ' '] }), [5]), ['low iron', 'High'])
  // fixed cutoffs ignore the values and are sorted
  assert.deepEqual(cutsFor(def({ cuts: [4, 2] }), values), [2, 4])
})

test('applyGrouping labels rows and leaves rows without the field out', () => {
  const g = applyGrouping(def({ kind: 'median', labels: ['Low iron', 'High iron'] }), [{ Iron: 1 }, { Iron: 3 }, { Iron: 5 }, { Iron: null }, {}])
  assert.deepEqual(g.labels, ['Low iron', 'High iron'])
  assert.deepEqual([{ Iron: 1 }, { Iron: 5 }, {}].map(g.labelOf), ['Low iron', 'High iron', null])
  assert.equal(g.colors.length, 2)
})

test('groupings round-trip through the URL; junk is ignored', () => {
  const defs = [def({ cuts: [0.1, 0.5], labels: ['low', 'mid', 'high µg'] }), def({ id: 'g2', name: 'OC/EC tertiles', field: 'OC / EC', kind: 'tertiles' })]
  assert.deepEqual(parseGroupings(groupingsParam(defs)), defs)
  assert.equal(groupingsParam([]), null)
  assert.deepEqual(parseGroupings('not-base64!!'), [])
  assert.deepEqual(parseGroupings(null), [])
})

test('the notebook groupings are well formed', () => {
  const ids = BUILT_IN_GROUPINGS.map((d) => d.id)
  assert.equal(new Set(ids).size, ids.length)
  for (const d of BUILT_IN_GROUPINGS) {
    assert.ok(d.id.startsWith('nb-') && d.source, d.id)
    assert.ok(SPLITS.some((s) => s.kind === d.kind), d.id)
    if (d.kind === 'cutoffs') {
      assert.ok(d.cuts.length > 0 && d.cuts.every((c, i) => i === 0 || c > d.cuts[i - 1]), `${d.id}: cutoffs ascending`)
      assert.ok(d.labels.length === 0 || d.labels.length === d.cuts.length + 1, `${d.id}: one label per band`)
    }
  }
})

test('deciles get ten labels and ten distinct colours; merging keeps the first copy of an id', () => {
  const d = { id: 'x', name: '', field: 'v', kind: 'deciles', cuts: [], labels: [] }
  const cuts = cutsFor(d, Array.from({ length: 100 }, (_, i) => i))
  assert.equal(cuts.length, 9)
  assert.equal(labelsFor(d, cuts)[0], 'D1 (lowest)')
  assert.equal(new Set(colorsFor(10)).size, 10)
  const a = { ...d, name: 'link' }, b = { ...d, name: 'saved' }, c = { ...d, id: 'y' }
  assert.deepEqual(mergeGroupings([a], [b, c]).map((g) => g.name + g.id), ['linkx', 'y'])
})

test('equal values can join the band below, and the ranges say so', () => {
  const cuts = [2, 4]
  assert.deepEqual([1.99, 2, 2.01, 4, 4.01].map((v) => bandOf(cuts, v, 'below')), [0, 0, 1, 1, 2])
  assert.equal(rangeText(cuts, 0, { equal: 'below' }), '≤ 2.00')
  assert.equal(rangeText(cuts, 1, { equal: 'below' }), '> 2.00 to ≤ 4.00')
  assert.equal(rangeText(cuts, 2, { equal: 'below' }), '> 4.00')
  const rows = [{ id: 'a', Iron: 2 }, { id: 'b', Iron: 3 }]
  assert.equal(applyGrouping(def({ cuts: [2], equal: 'below' }), rows).labelOf(rows[0]), '≤ 2')
  assert.equal(applyGrouping(def({ cuts: [2] }), rows).labelOf(rows[0]), '≥ 2')
})

test('typed cutoffs read as typed; quantile cutoffs stay rounded', () => {
  assert.equal(rangeText([0.371], 0, { exact: true }), '< 0.371')
  assert.equal(rangeText([0.1 + 0.2], 1, { exact: true }), '≥ 0.3')
  assert.deepEqual(labelsFor(def({ cuts: [0.371] }), [0.371]), ['< 0.371', '≥ 0.371'])
  assert.equal(rangeText([0.37132], 0), '< 0.37')
})

test('band colours and the equal side round-trip through the URL; older links read the same', () => {
  const d = def({ cuts: [2], equal: 'below', colors: ['#123456', ''] })
  const [back] = parseGroupings(groupingsParam([d]))
  assert.equal(back.equal, 'below')
  assert.deepEqual(back.colors, ['#123456', ''])
  const g = applyGrouping(back, [])
  assert.equal(g.colors[0], '#123456')
  assert.equal(g.colors[1], colorsFor(2)[1])
  const [plain] = parseGroupings(groupingsParam([def({ cuts: [2] })]))
  assert.equal(plain.equal, undefined)
  assert.equal(plain.colors, undefined)
})

test("a per-site median split cuts each site at its own median (FTIR_Group_Talk [17]: iron <= the site's median is Low)", () => {
  const rows = [
    { site: 'Beijing', Iron: 0.1 }, { site: 'Beijing', Iron: 0.2 }, { site: 'Beijing', Iron: 0.3 }, { site: 'Beijing', Iron: 0.4 },
    { site: 'Delhi', Iron: 1 }, { site: 'Delhi', Iron: 2 }, { site: 'Delhi', Iron: 3 },
    { site: 'JPL', Iron: null },
  ]
  const m = siteMedians(rows, 'Iron')
  assert.ok(Math.abs(m.get('Beijing') - 0.25) < 1e-12)
  assert.equal(m.get('Delhi'), 2)
  assert.equal(m.has('JPL'), false)
  const nbDef = BUILT_IN_GROUPINGS.find((d) => d.id === 'nb-iron-site-median')
  assert.ok(nbDef && nbDef.kind === 'site-median' && nbDef.equal === 'below')
  const g = applyGrouping(nbDef, rows)
  assert.deepEqual(g.labels, ['Low Fe', 'High Fe'])
  assert.deepEqual(g.cuts, [])
  assert.deepEqual(g.siteCuts.get('Delhi'), [2])
  // every Delhi value is above every Beijing one, yet each site has a low and a high half; the median itself is Low
  assert.deepEqual(rows.map(g.labelOf), ['Low Fe', 'Low Fe', 'High Fe', 'High Fe', 'Low Fe', 'Low Fe', 'High Fe', null])
  // the pooled median split would put all of Delhi in the high band
  const pooled = applyGrouping(def({ kind: 'median', equal: 'below' }), rows)
  assert.deepEqual(rows.slice(4, 7).map(pooled.labelOf), ['High', 'High', 'High'])
  // unnamed: the halves say what they are; the kind survives the URL
  assert.deepEqual(labelsFor(def({ kind: 'site-median' }), []), ['Below its site median', 'Above its site median'])
  assert.equal(parseGroupings(groupingsParam([def({ kind: 'site-median' })]))[0].kind, 'site-median')
  assert.ok(SPLITS.some((x) => x.kind === 'site-median' && x.bands === 2))
})
