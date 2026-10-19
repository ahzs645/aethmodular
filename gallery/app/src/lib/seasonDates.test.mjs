import test from 'node:test'
import assert from 'node:assert/strict'
import { seasonOfDate } from './seasons.ts'
import { PAPER_CALENDAR } from './pmfPaper.ts'

test('a calendar with day boundaries assigns by sampling date; month calendars are unchanged', () => {
  const of = (d) => seasonOfDate(d, Number(d.slice(5, 7)), PAPER_CALENDAR)?.name
  // the PMF paper's 2023 cut points: 25 Feb / 27 Feb, 27 Apr / 30 Apr, 30 Sep / 1 Oct
  assert.equal(of('2023-02-25'), 'First Dry')
  assert.equal(of('2023-02-27'), 'Less Rainy')
  assert.equal(of('2023-04-27'), 'Less Rainy')
  assert.equal(of('2023-04-30'), 'More Rainy')
  assert.equal(of('2023-09-30'), 'More Rainy')
  assert.equal(of('2023-10-01'), 'Second Dry')
  assert.equal(of('2023-12-17'), 'Second Dry')
  assert.equal(of('2024-01-03'), 'First Dry')
  const months = [{ name: 'Wet', months: [3, 4, 5], color: '#00f' }, { name: 'Dry', months: [1, 2, 6, 7, 8, 9, 10, 11, 12], color: '#f00' }]
  assert.equal(seasonOfDate('2023-04-30', 4, months)?.name, 'Wet')
  assert.equal(seasonOfDate(null, 2, months)?.name, 'Dry')
})
