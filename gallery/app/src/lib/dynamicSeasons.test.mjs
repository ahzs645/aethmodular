import test from 'node:test'
import assert from 'node:assert/strict'
import { dynamicCalendar, dynamicPlaceholder, MAX_FILL_DAYS } from './dynamicSeasons.ts'
import { seasonOfDate, isDatedCalendar } from './seasons.ts'

const days = (from, n) => Array.from({ length: n }, (_, i) => new Date(Date.parse(`${from}T00:00:00Z`) + i * 86400000).toISOString().slice(0, 10))
const WET_DRY = { kind: 'rules', minRun: 1, seasons: [{ name: 'Wet', when: [{ v: 'p', w: 1, op: '>', x: 5 }] }, { name: 'Dry', when: [] }] }

test('seasons cover the weather record only, not the open past and future', () => {
  const date = days('2023-01-01', 60)
  const cal = dynamicCalendar({ date, p: date.map((d) => (d < '2023-02-01' ? 10 : 0)) }, WET_DRY)
  assert.deepEqual(cal[0].spans, [['2023-01-01', '2023-02-01']])
  assert.deepEqual(cal[1].spans, [['2023-02-01', '2023-03-02']])
  assert.equal(seasonOfDate('2022-12-31', 12, cal), null)
  assert.equal(seasonOfDate('2023-03-05', 3, cal), null)
})

test('a short gap takes the nearest day, a long one stays unclassified', () => {
  const date = days('2023-01-01', 90)
  const short = date.map((d, i) => (i >= 20 && i < 20 + MAX_FILL_DAYS ? null : 10))
  const cal = dynamicCalendar({ date, p: short }, WET_DRY)
  assert.equal(seasonOfDate(date[22], 1, cal).name, 'Wet')
  const long = date.map((d, i) => (i >= 20 && i < 50 ? null : 10))
  const cal2 = dynamicCalendar({ date, p: long }, WET_DRY)
  assert.equal(seasonOfDate(date[21], 1, cal2).name, 'Wet')          // within the fill reach of day 19
  assert.equal(seasonOfDate(date[35], 2, cal2), null)                // 15 days from any weather
  // the nearest day, not always the one before: day 48 is 2 days from day 50 (dry) and 29 from day 19 (wet)
  const mixed = date.map((d, i) => (i >= 20 && i < 50 ? null : i < 20 ? 10 : 0))
  assert.equal(seasonOfDate(date[48], 2, dynamicCalendar({ date, p: mixed }, WET_DRY)).name, 'Dry')
})

test('while the weather loads nothing is classified', () => {
  const ph = dynamicPlaceholder(WET_DRY)
  assert.ok(isDatedCalendar(ph))
  assert.equal(seasonOfDate('2023-01-10', 1, ph), null)
  const onset = dynamicPlaceholder({ kind: 'onset', window: 30, onset: 1.5, heavy: 9 })
  assert.equal(seasonOfDate('2023-07-10', 7, onset), null)
})
