import assert from 'node:assert/strict'
import { test } from 'node:test'
import fs from 'node:fs'
import { medianOf, parseCuts, residuals, roundCut, siteMedianCuts, splitAt, splitTable, splitValue } from './splitFit.ts'

test('the variable is a field or a ratio; a zero or missing denominator has none', () => {
  assert.equal(splitValue({ Iron: 0.3, EC: 2 }, 'Iron', 'EC'), 0.15)
  assert.equal(splitValue({ Iron: 0.3, EC: 0 }, 'Iron', 'EC'), null)
  assert.equal(splitValue({ Iron: 0.3 }, 'Iron', 'EC'), null)
  assert.equal(splitValue({ Iron: 0.3 }, 'Iron', null), 0.3)
  assert.equal(splitValue({ Iron: null, EC: 1 }, 'Iron', 'EC'), null)
})

test('low is ≤ the cutoff, high is above it; a side needs three filters for a fit', () => {
  const items = [1, 2, 3, 4, 5, 6].map((t) => ({ t, x: t, y: 2 * t + (t % 2 ? 0.1 : -0.1), site: t < 4 ? 'A' : 'B' }))
  const { low, high } = splitAt(items, () => 3)
  assert.deepEqual(low.map((i) => i.t), [1, 2, 3])
  assert.deepEqual(high.map((i) => i.t), [4, 5, 6])
  const [row, few] = splitTable(items, [3, 5])
  assert.equal(row.nLow, 3)
  assert.ok(Math.abs(row.high.slope - 2) < 1e-9, String(row.high.slope))
  assert.ok(row.dR2 !== null)
  assert.equal(few.nHigh, 1)
  assert.equal(few.high, null)
  assert.equal(few.dR2, null)
})

test('per-site medians, residuals and typed cutoffs', () => {
  const items = [{ t: 1, site: 'A' }, { t: 3, site: 'A' }, { t: 10, site: 'B' }, { t: 20, site: 'B' }, { t: 40, site: 'B' }]
  assert.deepEqual([...siteMedianCuts(items)], [['A', 2], ['B', 20]])
  assert.equal(medianOf([4, 1, 3, 2]), 2.5)
  assert.deepEqual(residuals([{ x: 1, y: 3 }, { x: 2, y: 2 }], 'difference'), [2, 0])
  const r = residuals([{ x: 1, y: 2 }, { x: 2, y: 4.5 }, { x: 3, y: 6 }], 'fit')
  assert.ok(Math.abs(r.reduce((a, b) => a + b, 0)) < 1e-12, 'OLS residuals sum to zero')
  assert.deepEqual(parseCuts('0.2, 0.05 0.1;0.1 x'), [0.05, 0.1, 0.2])
  assert.equal(roundCut(0.135612), 0.136)
})

// Task_Analysis_Notebook c36: "HIGH Concentration Sites - HIPS vs FTIR EC" (Delhi + Addis Ababa), x = HIPS Fabs / MAC 10,
// y = FTIR EC, split on iron (ng/m³) / FTIR EC (µg/m³). Here iron is in µg/m³, so the notebook's 100 is 0.1.
test('reproduces the notebook\'s Fe/EC threshold table on the exported filters', { skip: !fs.existsSync(new URL('../../public/data/filters.json', import.meta.url)) }, () => {
  const { rows } = JSON.parse(fs.readFileSync(new URL('../../public/data/filters.json', import.meta.url), 'utf8'))
  const items = rows.filter((r) => (r.site === 'Delhi' || r.site === 'Addis Ababa') && !r.excluded).flatMap((r) => {
    const t = splitValue(r, 'Iron', 'EC (FTIR)')
    return typeof r['HIPS BC'] === 'number' && typeof r['EC (FTIR)'] === 'number' && t !== null ? [{ x: r['HIPS BC'], y: r['EC (FTIR)'], t, site: r.site }] : []
  })
  assert.equal(items.length, 215)
  const table = splitTable(items, [0.05, 0.1, 0.15, 0.2, 0.3])
  // notebook: threshold, low n, high n, high R², high slope (identical filters on the high side)
  const nb = [[50, 54, 161, 0.720, 1.582], [100, 174, 41, 0.612, 1.098], [150, 202, 13, 0.827, 1.045], [200, 208, 7, 0.227, 0.506], [300, 211, 4, 0.302, 0.567]]
  table.forEach((row, i) => {
    const [, nLow, nHigh, r2, slope] = nb[i]
    assert.equal(row.nLow, nLow)
    assert.equal(row.nHigh, nHigh)
    assert.ok(Math.abs(row.high.r2 - r2) < 0.002, `${nb[i][0]}: R² ${row.high.r2}`)
    assert.ok(Math.abs(row.high.slope - slope) < 0.002, `${nb[i][0]}: slope ${row.high.slope}`)
  })
  // the low side differs by a handful of HIPS values (notebook 0.762 / 1.776 at 100)
  assert.ok(Math.abs(table[1].low.r2 - 0.762) < 0.01)
})
