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
})
