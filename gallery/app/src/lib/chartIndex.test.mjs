import assert from 'node:assert/strict'
import { test } from 'node:test'
import fs from 'node:fs'
import { buildIndex } from '../../scripts/build-chart-index.mjs'

test('the ⌘K chart index matches the source (run npm run index:charts after adding a chart)', () => {
  const saved = JSON.parse(fs.readFileSync(new URL('../chartIndex.json', import.meta.url), 'utf8'))
  assert.deepEqual(saved, JSON.parse(JSON.stringify(buildIndex())))
  // every chart has a title and a place
  for (const c of saved.charts) assert.ok(c.title && c.tab, JSON.stringify(c))
  assert.ok(saved.charts.length > 140)
  for (const id of ['vibes-validation-recovery', 'vibes-validation-errors',
    'vibes-validation-traces', 'vibes-validation-blanks', 'vibes-validation-runtime', 'vibes-validation-reference']) {
    assert.ok(saved.charts.some((c) => c.id === id && c.tab === 'baseline'), `Shared figure ${id} must be searchable on Baselines`)
  }
})
