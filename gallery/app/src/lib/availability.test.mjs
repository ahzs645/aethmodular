import assert from 'node:assert/strict'
import { test } from 'node:test'
import { availabilityRecords, monthlyAvailability, recentCutoff, samplingStatus } from './availability.ts'

test('columnar IMPROVE histories preserve distinct POCs, zeros and missing fields', () => {
  const rows = availabilityRecords({ code: 'SITE1', date: ['2024-01-03', '2024-01-03', '2024-03-01'], poc: [1, 2, 1], EC: [1, 0, null], Iron: [-0.2, null, 3] })
  assert.equal(new Set(rows.map(r => r.id)).size, 3)
  assert.deepEqual(rows.map(r => r.EC), [1, 0, null])
  assert.deepEqual(monthlyAvailability(rows, 'EC').months.map(m => [m.samples, m.measured]), [[2, 2], [0, 0], [1, 0]])
  assert.deepEqual(availabilityRecords({ rows }), rows)
  assert.deepEqual(availabilityRecords({}), [])
})

test('recency uses inclusive dates and clamps leap-day thresholds', () => {
  assert.equal(recentCutoff('2026-10-23', 2), '2024-10-23')
  assert.equal(recentCutoff('2024-02-29', 1), '2023-02-28')
  assert.equal(samplingStatus('2024-10-23', '2024-10-23', '2026-10-23'), 'Recent samples')
  assert.equal(samplingStatus('2024-10-22', '2024-10-23', '2026-10-23'), 'No recent samples')
  assert.equal(samplingStatus('—', '2024-10-23', '2026-10-23'), 'No samples')
  assert.equal(samplingStatus('', '2024-10-23', '2026-10-23'), 'No samples')
  assert.equal(samplingStatus('2027-01-01', '2024-10-23', '2026-10-23'), 'After reference date')
})
test('sampling keeps zero months, distinguishes missing fields, and counts physical filters once', () => {
  const rows = [
    { id: 'a', date: '2023-12-03', ec: 0 },
    { id: 'a', date: '2023-12-03', ec: 0 },
    { id: 'b', date: '2024-03-01', ec: null },
    { id: 'c', date: '2024-03-02', ec: -0.2 },
    { id: 'd', date: 'invalid', ec: 1 },
  ]
  const result = monthlyAvailability(rows, 'ec')
  assert.equal(result.records, 3)
  assert.equal(result.measured, 2)
  assert.deepEqual(result.months.map(m => m.samples), [1, 0, 0, 2])
  assert.deepEqual(result.gaps, [{ start: '2024-01', end: '2024-02', months: 2 }])
  assert.deepEqual(result.jumps, [])
})
test('jumps require consecutive observed months and a meaningful count difference', () => {
  const rows = [2, 5, 0, 10, 9].flatMap((n, i) => Array.from({ length: n }, (_, j) => ({ id: `${i}-${j}`, date: `2024-0${i + 1}-01` })))
  assert.deepEqual(monthlyAvailability(rows).jumps, [{ period: '2024-02', before: 2, after: 5 }])
  assert.deepEqual(monthlyAvailability([]).months, [])
})
