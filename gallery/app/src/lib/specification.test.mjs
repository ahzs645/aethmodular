import test from 'node:test'
import assert from 'node:assert/strict'
import { DEFAULT_HIGHLIGHT_LIMITS as limits, highlightMatches, selectionPanels } from './specification.ts'

test('highlight rules distinguish saved guardrails from partial criteria', () => {
  const good = { intercept: 0.5, slope: 0.7, cv: 0.85, guardrails: false }
  assert.equal(highlightMatches(good, 'guardrails', limits), false)
  assert.equal(highlightMatches(good, 'custom', limits), true)
  assert.equal(highlightMatches({ ...good, intercept: -0.5, slope: 1.3 }, 'custom', limits), true)
  assert.equal(highlightMatches({ ...good, intercept: -0.5001 }, 'custom', limits), false)
  assert.equal(highlightMatches({ ...good, intercept: -9 }, 'cv-slope', limits), true)
  assert.equal(highlightMatches({ ...good, cv: null }, 'cv-slope', limits), false)
  assert.equal(highlightMatches({ ...good, cv: null }, 'intercept', limits), true)
  assert.equal(highlightMatches({ ...good, slope: null }, 'custom', limits), false)
  assert.equal(highlightMatches({ ...good, intercept: NaN }, 'intercept', limits), false)
  assert.equal(highlightMatches({ ...good, intercept: 0.8 }, 'custom', { ...limits, intercept: 1 }), true)
})

test('separate selection curves partition the filtered population without reordering or duplicating it', () => {
  const shown = [0, 2, 3, 5]
  const panels = selectionPanels(shown, [0, 1, 1, 0, 2, 1], ['airspec', 'vibes', 'vibes-full'])
  assert.deepEqual(panels, [{ key: 'airspec', indices: [0, 3] }, { key: 'vibes', indices: [2, 5] }])
  assert.deepEqual(panels.flatMap(p => p.indices).sort((a,b) => a-b), shown)
  assert.deepEqual(selectionPanels([], [], ['airspec']), [])
})
