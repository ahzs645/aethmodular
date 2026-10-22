import assert from 'node:assert/strict'
import { test } from 'node:test'
import { BUILT_IN_GROUPINGS, SPLITS, applyGrouping, bandOf, colorsFor, cutsFor, groupingsParam, labelsFor, mergeGroupings, parseGroupings, rangeText } from './groupings.ts'

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
