import test from 'node:test'
import assert from 'node:assert/strict'
import { EXCL_MODES, crossPath, exclModeParam, exclusionReason, exportIn, markTip, parseExclMode, splitExcluded } from './exclusionMode.ts'
import { regression } from './stats.ts'

test('excl in the URL: 1 stays Include, mark is Mark, anything else Hide; Hide stays out of the link', () => {
  assert.equal(parseExclMode('1'), 'include')
  assert.equal(parseExclMode('include'), 'include')
  assert.equal(parseExclMode('mark'), 'mark')
  for (const v of [null, undefined, '', '0', 'hide', 'x']) assert.equal(parseExclMode(v), 'hide')
  for (const m of EXCL_MODES) assert.equal(parseExclMode(exclModeParam(m)), m)
  assert.equal(exclModeParam('hide'), null)
  assert.equal(exclModeParam('include'), '1')
})

const rows = [
  { id: 'A', x: 1, y: 2.1 },
  { id: 'B', x: 2, y: 3.9 },
  { id: 'C', x: 3, y: 6.2 },
  { id: 'D', x: 4, y: 7.8 },
  { id: 'E', x: 5, y: 30, excluded: true, exclusion_reason: 'flow fault' },
  { id: 'F', x: 6, y: 11.9, excluded: true, exclusion_reason: '' },
  { id: 'G', x: 7, y: 2 },
]
const state = (mode, extra = {}) => ({ mode, inclIds: new Set(), dropIds: new Set(['G']), dropWhy: { G: 'smudged' }, ...extra })
const ids = (rs) => rs.map((r) => r.id).join('')

test('which rows feed the statistics and which are drawn as marks', () => {
  const hide = splitExcluded(rows, state('hide'))
  assert.equal(ids(hide.counted), 'ABCD')
  assert.equal(ids(hide.marked), '')
  const mark = splitExcluded(rows, state('mark'))
  assert.equal(ids(mark.counted), 'ABCD', 'Mark counts exactly what Hide counts')
  assert.equal(ids(mark.marked), 'EFG', "the export's and the reader's exclusions are both marked")
  const include = splitExcluded(rows, state('include'))
  assert.equal(ids(include.counted), 'ABCDEF', "Include puts the export's back; a hand exclusion stays out")
  assert.equal(ids(include.marked), '')
})

test('a filter put back one at a time counts in every mode', () => {
  for (const m of EXCL_MODES) {
    const s = state(m, { inclIds: new Set(['E']) })
    assert.ok(splitExcluded(rows, s).counted.some((r) => r.id === 'E'), m)
    assert.ok(exportIn(rows[4], s))
  }
})

test('the reason a mark gives on hover', () => {
  const s = state('mark')
  assert.equal(exclusionReason(rows[0], s), null)
  assert.equal(exclusionReason(rows[4], s), 'flow fault')
  assert.equal(exclusionReason(rows[5], s), 'excluded in the data export')
  assert.equal(exclusionReason(rows[6], s), 'by hand (smudged)')
  assert.equal(exclusionReason(rows[6], { ...s, dropWhy: {} }), 'by hand')
  // an unknown filter excluded by hand only (a crossplot fed from a static export)
  assert.equal(exclusionReason({ id: 'G' }, s), 'by hand (smudged)')
  const tip = markTip({ id: 'E', site: 'Addis Ababa', date: '2023-01-02' }, 'flow fault', ['x: 5'])
  assert.deepEqual(tip.slice(0, 4), ['E', 'Addis Ababa · 2023-01-02', 'x: 5', 'excluded: flow fault'])
})

test('a fit on the counted rows is the same under Hide and Mark, and moves under Include', () => {
  const fit = (m) => { const { counted } = splitExcluded(rows, state(m)); return regression(counted.map((r) => r.x), counted.map((r) => r.y)) }
  const h = fit('hide'), k = fit('mark'), i = fit('include')
  assert.equal(k.slope, h.slope)
  assert.equal(k.intercept, h.intercept)
  assert.equal(k.r2, h.r2)
  assert.equal(k.n, h.n)
  assert.notEqual(i.slope, h.slope)
  assert.ok(i.r2 < h.r2)
})

test('the × path is two strokes centred on the point', () => {
  assert.equal(crossPath(10, 20, 2), 'M8,18L12,22M8,22L12,18')
})
