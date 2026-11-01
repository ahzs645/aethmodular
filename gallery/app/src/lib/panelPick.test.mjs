import test from 'node:test'
import assert from 'node:assert/strict'
import { NO_PANELS, addPanel, formatPanels, hidePanel, movePanel, movePanelTo, parsePanels, replacePanel, resolvePanels, showPanel } from './panelPick.ts'

const ALL = ['temp', 'prcp', 'rh', 'blh', 'ws', 'field']
const DEF = ['temp', 'prcp', 'rh', 'blh', 'field']

test('no value is the defaults; "-" is none; ids keep their order, once, and only those the chart draws', () => {
  assert.deepEqual(parsePanels(null, ALL, DEF), DEF)
  assert.deepEqual(parsePanels('', ALL, DEF), DEF)
  assert.deepEqual(parsePanels(NO_PANELS, ALL, DEF), [])
  assert.deepEqual(parsePanels('field|ws|ws|gone|temp', ALL, DEF), ['field', 'ws', 'temp'])
  // options still loading: the link's ids stand
  assert.deepEqual(parsePanels('f:EC|temp', [], DEF), ['f:EC', 'temp'])
})

test('the defaults are never written, and an empty choice is written as "-"', () => {
  assert.equal(formatPanels(DEF, DEF), '')
  assert.equal(formatPanels([], DEF), NO_PANELS)
  assert.equal(formatPanels(['ws', 'temp', 'ws'], DEF), 'ws|temp')
  for (const ids of [DEF, [], ['ws'], ['field', 'temp']]) assert.deepEqual(parsePanels(formatPanels(ids, DEF), ALL, DEF), ids)
})

test('add, show, hide, move and replace', () => {
  assert.deepEqual(addPanel(['a', 'b'], 'c'), ['a', 'b', 'c'])
  assert.deepEqual(addPanel(['a', 'b'], 'c', 0), ['c', 'a', 'b'])
  assert.deepEqual(addPanel(['a', 'b'], 'a'), ['a', 'b'])
  // shown among a fixed set: in the set's order
  assert.deepEqual(showPanel(['temp', 'field'], 'rh', ALL), ['temp', 'rh', 'field'])
  // a reordered choice: before the first shown panel that comes after it in the set
  assert.deepEqual(showPanel(['field', 'temp'], 'ws', ALL), ['ws', 'field', 'temp'])
  assert.deepEqual(showPanel(['temp'], 'other', ALL), ['temp', 'other'])
  assert.deepEqual(hidePanel(['a', 'b', 'c'], 'b'), ['a', 'c'])
  assert.deepEqual(movePanel(['a', 'b', 'c'], 'c', -1), ['a', 'c', 'b'])
  assert.deepEqual(movePanel(['a', 'b', 'c'], 'a', -1), ['a', 'b', 'c'])
  assert.deepEqual(movePanel(['a', 'b', 'c'], 'a', 5), ['b', 'c', 'a'])
  assert.deepEqual(movePanelTo(['a', 'b', 'c', 'd'], 'd', 'b'), ['a', 'd', 'b', 'c'])
  assert.deepEqual(movePanelTo(['a', 'b', 'c', 'd'], 'a', 'c'), ['b', 'c', 'a', 'd'])
  // a panel already shown trades places with the one being changed (the weather timeline's axis titles)
  assert.deepEqual(replacePanel(['temp', 'rh', 'field'], 'rh', 'ws'), ['temp', 'ws', 'field'])
  assert.deepEqual(replacePanel(['temp', 'rh', 'field'], 'rh', 'temp'), ['rh', 'temp', 'field'])
})

test('a legacy value of the same key, or a legacy key, still opens as it was', () => {
  const hips = [{ map: (v) => (v === 'with naive' ? ['improve', 'addis', 'naive'] : v === 'notebook pair' ? ['improve', 'addis'] : null) }]
  const all = ['improve', 'addis', 'naive'], def = ['improve', 'addis']
  assert.deepEqual(resolvePanels('with naive', hips, [''], all, def), ['improve', 'addis', 'naive'])
  assert.deepEqual(resolvePanels('notebook pair', hips, [''], all, def), def)
  assert.deepEqual(resolvePanels('naive|improve', hips, [''], all, def), ['naive', 'improve'])
  assert.deepEqual(resolvePanels('', hips, [''], all, def), def)
  const keyed = [{ key: 'old.shown', map: (v) => v.split(',') }]
  assert.deepEqual(resolvePanels('', keyed, ['addis,naive'], all, def), ['addis', 'naive'])
  // the current key wins over an old one
  assert.deepEqual(resolvePanels('improve', keyed, ['addis,naive'], all, def), ['improve'])
  assert.deepEqual(resolvePanels(NO_PANELS, keyed, ['addis'], all, def), [])
})
