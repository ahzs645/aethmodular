import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { rainRuleCalendar, parseRainRule, rainRuleParam, DEFAULT_RAIN_RULE } from './rainSeasons.ts'
import { seasonOfDate } from './seasons.ts'

const daysBetween = (a, b) => Math.round((Date.parse(a) - Date.parse(b)) / 86400000)

test('the rain rule on Addis ERA5 reproduces the PMF paper 2023 boundaries within a week', () => {
  const daily = JSON.parse(readFileSync(new URL('../../public/data/weather/daily/ETAD.json', import.meta.url)))
  const cal = rainRuleCalendar(daily)
  const start = (name) => cal.find((s) => s.name === name).spans.find(([a]) => a.startsWith('2023'))[0]
  // paper: Less Rainy from 27 Feb, More Rainy from 30 Apr, Second Dry from 1 Oct
  assert.ok(Math.abs(daysBetween(start('Less Rainy'), '2023-02-26')) <= 3, start('Less Rainy'))
  assert.ok(Math.abs(daysBetween(start('More Rainy'), '2023-04-29')) <= 7, start('More Rainy'))
  assert.ok(Math.abs(daysBetween(start('Second Dry'), '2023-10-01')) <= 7, start('Second Dry'))
  // every day of every year belongs to exactly one season
  for (const d of ['2023-01-15', '2023-03-15', '2023-07-15', '2023-11-15', '2024-07-15', '2025-01-02']) {
    assert.equal(cal.filter((s) => s.spans.some(([a, b]) => d >= a && d < b)).length, 1, d)
  }
  assert.equal(seasonOfDate('2023-07-15', 7, cal).name, 'More Rainy')
  assert.equal(seasonOfDate('2023-12-01', 12, cal).name, 'Second Dry')
})

test('a year without rains is all First Dry; settings round-trip through the URL', () => {
  const date = Array.from({ length: 365 }, (_, i) => new Date(Date.UTC(2021, 0, 1 + i)).toISOString().slice(0, 10))
  const cal = rainRuleCalendar({ date, p: date.map(() => 0) })
  assert.deepEqual(cal[0].spans, [['2021-01-01', '2022-01-01']])
  assert.ok(cal.slice(1).every((s) => !s.spans.length))
  assert.deepEqual(parseRainRule(rainRuleParam({ window: 21, onset: 1, heavy: 8 })), { window: 21, onset: 1, heavy: 8 })
  assert.equal(parseRainRule('30,9,1.5'), null)   // heavy must exceed onset
  assert.deepEqual(parseRainRule(rainRuleParam(DEFAULT_RAIN_RULE)), DEFAULT_RAIN_RULE)
})

const days = (from, n) => Array.from({ length: n }, (_, i) => new Date(Date.parse(`${from}T00:00:00Z`) + i * 86400000).toISOString().slice(0, 10))

test('missing rain is no season, not a dry one', () => {
  const date = days('2021-01-01', 365)
  const cal = rainRuleCalendar({ date, p: date.map(() => null) })
  assert.ok(cal.every((s) => s.spans.length === 0))
  // a filter in that year has no rain-rule season, not the month's usual one
  assert.equal(seasonOfDate('2021-07-15', 7, cal), null)
})

test('the seasons stop where the rain record does, and skip a long gap', () => {
  // dry record from 1 Jul to 30 Sep only: Second Dry over those days, not the whole year
  const date = days('2021-07-01', 92)
  const cal = rainRuleCalendar({ date, p: date.map(() => 0) })
  assert.deepEqual(cal[3].spans, [['2021-07-01', '2021-10-01']])
  assert.equal(seasonOfDate('2021-12-15', 12, cal), null)
  assert.equal(seasonOfDate('2021-08-15', 8, cal).name, 'Second Dry')
  // a 60-day hole in a dry year: the rolling mean is missing there, so so is the season
  const year = days('2022-01-01', 365)
  const p = year.map((d) => (d >= '2022-05-01' && d < '2022-06-30' ? null : 0))
  const gap = rainRuleCalendar({ date: year, p })
  assert.equal(seasonOfDate('2022-05-31', 5, gap), null)
  assert.equal(seasonOfDate('2022-03-01', 3, gap).name, 'First Dry')
  assert.equal(seasonOfDate('2022-09-01', 9, gap).name, 'First Dry')
})
