import assert from 'node:assert/strict'
import { test } from 'node:test'
import { colAt, groupStarts, monthColumns, rowAt, rowTops } from './heatMatrix.ts'

test('rows in groups get a gap before each new group, and the pointer finds rows but not gaps', () => {
  const { tops, height } = rowTops(['a', 'a', 'b', 'b', 'b'], 10, 4)
  assert.deepEqual(tops, [0, 10, 24, 34, 44])
  assert.equal(height, 54)
  assert.equal(rowAt(tops, 10, 0), 0)
  assert.equal(rowAt(tops, 10, 19.9), 1)
  assert.equal(rowAt(tops, 10, 21), -1, 'in the gap between groups')
  assert.equal(rowAt(tops, 10, 24), 2)
  assert.equal(rowAt(tops, 10, 53), 4)
  assert.equal(rowAt(tops, 10, 54), -1)
  assert.equal(rowAt(tops, 10, -1), -1)
  assert.deepEqual(rowTops([undefined, undefined], 8, 5).tops, [0, 8], 'no groups, no gaps')
})

test('columns are found by width and group starts are listed once per run', () => {
  assert.equal(colAt(5, 20, 0), 0)
  assert.equal(colAt(5, 20, 99), 4)
  assert.equal(colAt(5, 20, 100), -1)
  assert.equal(colAt(5, 20, -0.1), -1)
  assert.deepEqual(groupStarts(['2023', '2023', '2024', undefined, '2024']), [{ index: 0, group: '2023' }, { index: 2, group: '2024' }, { index: 4, group: '2024' }])
})

test('month columns: year-months grouped by year, calendar months named', () => {
  const ym = monthColumns(['2023-12', '2024-01'])
  assert.deepEqual(ym.map((c) => [c.key, c.label, c.group, c.title]), [['2023-12', '12', '2023', 'Dec 2023'], ['2024-01', '01', '2024', 'Jan 2024']])
  const cal = monthColumns([1, '02', 12])
  assert.deepEqual(cal.map((c) => [c.key, c.label, c.short, c.group]), [['1', 'Jan', 'J', undefined], ['02', 'Feb', 'F', undefined], ['12', 'Dec', 'D', undefined]])
})
