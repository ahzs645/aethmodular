import test from 'node:test'
import assert from 'node:assert/strict'
import { UrlHistory, changedKeys } from './undoHistory.ts'

test('a burst of writes is one step, undone and redone as a whole', () => {
  const h = new UrlHistory()
  h.begin('#a=1')
  h.begin('#a=2') // a later write in the same burst keeps the burst's start
  assert.deepEqual(h.settle('#a=3'), { from: '#a=1', to: '#a=3' })
  assert.equal(h.undo('#a=3')?.from, '#a=1')
  assert.equal(h.redo('#a=1')?.to, '#a=3')
  assert.equal(h.redo('#a=3'), null)
})

test('a burst that ends where it began records nothing', () => {
  const h = new UrlHistory()
  h.begin('#a=1')
  assert.equal(h.settle('#a=1'), null)
  assert.equal(h.canUndo, false)
})

test('a new change after an undo drops the redo stack', () => {
  const h = new UrlHistory()
  h.begin('#s=1'); h.settle('#s=2')
  h.begin('#s=2'); h.settle('#s=3')
  assert.equal(h.undo('#s=3')?.from, '#s=2')
  h.begin('#s=2'); h.settle('#s=9')
  assert.equal(h.canRedo, false)
  assert.equal(h.undo('#s=9')?.from, '#s=2')
  assert.equal(h.undo('#s=2')?.from, '#s=1')
  assert.equal(h.undo('#s=1'), null)
})

test('undo during an open burst settles it first, so the change in flight is what is undone', () => {
  const h = new UrlHistory()
  h.begin('#x=1')
  assert.deepEqual(h.undo('#x=5'), { from: '#x=1', to: '#x=5' })
})

test('the stack keeps the most recent steps', () => {
  const h = new UrlHistory(2)
  for (let i = 0; i < 4; i++) { h.begin(`#n=${i}`); h.settle(`#n=${i + 1}`) }
  assert.equal(h.undo('#n=4')?.from, '#n=3')
  assert.equal(h.undo('#n=3')?.from, '#n=2')
  assert.equal(h.undo('#n=2'), null)
})

test('changedKeys names the settings that differ, in the query and the hash', () => {
  assert.deepEqual(changedKeys('?mapstat=Mean#tab=map&sites=A', '#tab=map&sites=A|B'), ['mapstat', 'sites'])
  assert.deepEqual(changedKeys('?t3.scatter.layout=Per+site&t4.x=1#tab=dashboard', '#tab=dashboard'), ['dashboard tile'])
})
