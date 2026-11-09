import test from 'node:test'
import assert from 'node:assert/strict'
import { seasonOccurrence } from './seasons.ts'
import { seasonParent, seasonSegments, seasonSegmentLayout } from './seasonSegments.ts'
import { PAPER_CALENDAR } from './pmfPaper.ts'

const seasons = [
  { name: 'Bega (Oct-Jan)', months: [10, 11, 12, 1], color: '#f90' },
  { name: 'Belg (Feb-May)', months: [2, 3, 4, 5], color: '#0f0' },
  { name: 'Kiremt (Jun-Sep)', months: [6, 7, 8, 9], color: '#00f' },
]
const point = date => ({ d: new Date(`${date}T00:00:00Z`), row: { date, season: '' } })
const points = ['2022-10-15', '2023-01-20', '2023-03-15', '2023-07-15', '2023-10-15', '2024-01-20', '2024-03-15', '2024-07-15'].map(point)

test('cross-year seasons stay whole under their ending year, even with partial observations', () => {
  for (const date of ['2022-10-15', '2023-01-20']) {
    const o = seasonOccurrence(date, seasons)
    assert.equal(o.year, 2023)
    assert.equal(o.start, '2022-10-01')
    assert.equal(o.end, '2023-02-01')
  }
  assert.equal(seasonOccurrence('2024-10-15', seasons).year, 2025)
  assert.equal(seasonOccurrence('2024-01-15', seasons).start, '2023-10-01')
  const winter = [{ name: 'Winter', months: [12, 1, 2], color: '#00f' }]
  assert.equal(seasonOccurrence('2023-12-25', winter).year, 2024)
})

test('both grouping modes retain every filter once and change only the season-block order', () => {
  const annual = seasonSegments(points, seasons, 'Year, then season')
  const comparison = seasonSegments([...points].reverse(), seasons, 'Season, then year')
  const names = result => result.segs.map(s => `${s.year} ${s.label}`)
  assert.deepEqual(names(annual), ['2023 Bega', '2023 Belg', '2023 Kiremt', '2024 Bega', '2024 Belg', '2024 Kiremt'])
  assert.deepEqual(names(comparison), ['2023 Bega', '2024 Bega', '2023 Belg', '2024 Belg', '2023 Kiremt', '2024 Kiremt'])
  for (const result of [annual, comparison]) {
    assert.deepEqual(result.points.map(p => p.row.date).sort(), points.map(p => p.row.date).sort())
    result.segs.forEach((s, i) => assert.ok(result.points.slice(s.a, s.b).every(p => p.seg === i)))
    const bega = result.segs.find(s => s.year === '2023' && s.label === 'Bega')
    assert.equal(bega.b - bega.a, 2)
  }
})

test('day-boundary and weather calendars use their actual exclusive season boundaries', () => {
  assert.equal(seasonOccurrence('2023-12-17', PAPER_CALENDAR).year, 2023)
  const weather = [{ name: 'Dry', color: '#f90', months: [], spans: [['2022-10-15', '2023-02-03']] }]
  assert.equal(seasonOccurrence('2022-12-01', weather).year, 2023)
  assert.equal(seasonOccurrence('2023-02-03', weather), null)
})

test('disjoint months and all-year calendars do not join unrelated occurrences', () => {
  const disjoint = [{ name: 'Selected', months: [1, 5], color: '#00f' }]
  assert.notEqual(seasonOccurrence('2023-01-15', disjoint).start, seasonOccurrence('2023-05-15', disjoint).start)
  const all = [{ name: 'All year', months: Array.from({ length: 12 }, (_, i) => i + 1), color: '#00f' }]
  assert.deepEqual([seasonOccurrence('2023-09-15', all).start, seasonOccurrence('2023-09-15', all).year], ['2023-01-01', 2023])
})

test('missing years and seasons create no synthetic blocks or data', () => {
  const result = seasonSegments(['2022-03-01', '2024-03-01'].map(point), seasons, 'Season, then year')
  assert.deepEqual(result.segs.map(s => [s.label, s.year]), [['Belg', '2022'], ['Belg', '2024']])
  assert.equal(result.points.length, 2)
  assert.deepEqual(seasonSegments([], seasons, 'Year, then season'), { points: [], segs: [] })
})

test('block layout preserves room for labels and separates the year or season groups', () => {
  for (const order of ['Year, then season', 'Season, then year']) {
    const { segs } = seasonSegments(points, seasons, order)
    for (const width of [200, 1800]) {
      const layout = seasonSegmentLayout(segs, order, width)
      assert.ok(layout.width >= width)
      layout.widths.forEach(w => assert.ok(w >= 100 - 1e-9))
      layout.lefts.slice(1).forEach((left, j) => {
        const expected = seasonParent(segs[j], order) === seasonParent(segs[j + 1], order) ? 10 : 24
        assert.ok(Math.abs(left - layout.lefts[j] - layout.widths[j] - expected) < 1e-9)
      })
      assert.ok(Math.abs(layout.lefts.at(-1) + layout.widths.at(-1) - layout.width) < 1e-9)
    }
  }
})
