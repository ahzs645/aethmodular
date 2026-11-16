// The rounding rule behind "How many decimal places a linearised HIPS BC can carry" (pages/HipsDecimals.tsx), and the
// headline it gives on the 11 Nov procedure's saved site statistics: at the median, Addis is the only SPARTAN site
// whose linearised BC cannot carry a decimal. Inputs: meetings/hips-procedure-20261111.json
// (gallery/data/export_meeting_20261111_procedure.py).
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { claimOf, claimPlace, digitSplit, halfRange, placeName, writeAt, writtenPlace } from './claimDigits.ts'

const bundle = JSON.parse(fs.readFileSync(new URL('../../public/data/meetings/hips-procedure-20261111.json', import.meta.url), 'utf8'))
const rows = (table) => { const cols = Object.keys(table); return table[cols[0]].map((_, i) => Object.fromEntries(cols.map((c) => [c, table[c][i]]))) }

test('the last digit written is the one the uncertainty starts in', () => {
  assert.equal(claimPlace(1.29), 0)
  assert.equal(claimPlace(9.99), 0)
  assert.equal(claimPlace(13), 1)
  assert.equal(claimPlace(0.4), -1)
  assert.equal(claimPlace(0.1), -1)
  assert.equal(claimPlace(1), 0)
  assert.equal(claimPlace(0.039), -2)
  assert.ok(Number.isNaN(claimPlace(0)))
  assert.ok(Number.isNaN(claimPlace(NaN)))
  assert.equal(halfRange(7.19, 9.76), halfRange(9.76, 7.19))
})

test('a value is never written finer than it was reported', () => {
  assert.equal(writtenPlace(0.004), -2)
  assert.equal(writtenPlace(0.004, -3), -3)
  assert.equal(writtenPlace(0), -2)
  assert.equal(writtenPlace(0.4), -1)
  assert.equal(writtenPlace(2), 0)
})

test('writing and splitting a value at a place', () => {
  assert.equal(writeAt(8.0389, 0), '8')
  assert.equal(writeAt(9.761, 0), '10')
  assert.equal(writeAt(8.0325, -1), '8.0')
  assert.equal(writeAt(0.4812, -2), '0.48')
  assert.equal(writeAt(12.8, 1), '10')
  assert.equal(writeAt(NaN, 0), '—')
  assert.deepEqual(digitSplit(8.0389, 0), ['8', '.04'])
  assert.deepEqual(digitSplit(8.0325, -1), ['8.0', '3'])
  assert.deepEqual(digitSplit(0.4812, -2), ['0.48', ''])
  assert.deepEqual(digitSplit(12.84, 1), ['1', '2.84'])
  assert.deepEqual(digitSplit(7.2, 1), ['', '7.20'])
  assert.equal(placeName(0), 'whole numbers')
  assert.equal(placeName(-1), 'one decimal')
})

test('a value with its range, as it is written', () => {
  assert.deepEqual(claimOf(8.0389, 7.1898, 9.761), { uncertainty: halfRange(7.1898, 9.761), place: 0, text: '8 (7 to 10)' })
  assert.equal(claimOf(8.0325, 7.6909, 8.4824).text, '8.0 (7.7 to 8.5)')
  assert.equal(claimOf(0.48, 0.478, 0.486).text, '0.48 (0.48 to 0.49)')
})

test('on the saved site statistics, Addis is the only site whose median BC loses its decimal', () => {
  const sites = (key) => rows(bundle.tables.traced).filter((r) => r.key === key && r.linearised != null)
    .map((r) => ({ code: r.code, ...claimOf(r.linearised, r.lo, r.hi) }))
  const median = sites('median_bc')
  assert.equal(median.length, 30)
  assert.deepEqual(median.filter((s) => s.place >= 0).map((s) => s.code), ['ETAD'])
  assert.equal(median.find((s) => s.code === 'ETAD').text, '8 (7 to 10)')
  assert.equal(median.find((s) => s.code === 'INDH').text, '8.0 (7.7 to 8.5)')
  assert.equal(median.filter((s) => s.place === -1).length, 14)
  // a mean leans on the darkest filters, so more sites lose the decimal
  assert.deepEqual(sites('mean_bc').filter((s) => s.place >= 0).map((s) => s.code).sort(), ['BDDU', 'ETAD', 'INJA', 'INKA', 'NGIL'])
  // every site the chart draws has an optical depth to sit at
  const found = new Set(rows(bundle.tables.found).map((f) => f.code))
  assert.ok(median.every((s) => found.has(s.code)))
})
