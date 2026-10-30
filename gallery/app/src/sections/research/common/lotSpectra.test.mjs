// char_12 §4's lot comparison (lotSpectra.ts), against the tables char_12 saved and the export's own spectra
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { boxStats, columnsIn, decodeView, escapeOf, lotContrast, lotsOf, membersOf, notebookSeason, profile } from './lotSpectra.ts'

const dir = new URL('../../../../public/data/research/lot_spectra/', import.meta.url)
const exp = fs.existsSync(new URL('lot_spectra.json', dir)) ? JSON.parse(fs.readFileSync(new URL('lot_spectra.json', dir), 'utf8')) : null
const skip = exp ? false : 'research/lot_spectra is not exported'
const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`)
const view = (key) => {
  const b = fs.readFileSync(new URL(exp.views[key].file, dir))
  return decodeView(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), exp.filters.id.length, exp.views[key].points)
}

test('decodeView: per-filter scale times int16, filter-major', () => {
  const buf = new ArrayBuffer(2 * 4 + 2 * 3 * 2)
  new Float32Array(buf, 0, 2).set([0.5, 2])
  new Int16Array(buf, 8, 6).set([1, -2, 3, 4, 5, -6])
  assert.deepEqual([...decodeView(buf, 2, 3)], [0.5, -1, 1.5, 8, 10, -12])
  assert.throws(() => decodeView(buf, 2, 4))
})

test('profile: mean or median and numpy linear quantiles per column', () => {
  const v = Float32Array.from([1, 10, 2, 20, 3, 30, 4, 40, 100, 50])  // 5 filters × 2 points
  const p = profile(v, 2, [0, 1, 2, 3, 4], 0, 2, 'mean', [0.1, 0.9])
  near(p.mid[0], 22, 1e-9, 'mean')
  near(p.lo[0], 1.4, 1e-6, 'p10')   // numpy.percentile([1,2,3,4,100], 10) = 1.4
  near(p.hi[0], 61.6, 1e-5, 'p90')  // 61.6
  assert.equal(profile(v, 2, [0, 1, 2, 3, 4], 1, 2, 'median', null).mid[0], 30)
  assert.deepEqual(columnsIn([3000, 2900, 2800, 1800], 2800, 3000), [0, 3])
})

test('boxStats follows matplotlib: whiskers at the furthest data within 1.5 × IQR', () => {
  const b = boxStats([1, 2, 3, 4, 5, 6, 7, 8, 9, 40])
  assert.equal(b.median, 5.5)
  assert.equal(b.whi, 9)
  assert.deepEqual(b.fliers, [40])
})

test('the notebook set reproduces char_12 §4: lot × season n, medians and Mann–Whitney p', { skip }, () => {
  const { members } = membersOf(exp, 'notebook', [], new Set(), () => null)
  assert.equal(members.length, 239)
  assert.deepEqual(lotsOf(members), [251, 248])
  const dry = members.filter((m) => m.season === notebookSeason(12))
  const saved = Object.fromEntries(exp.check.saved.lot_effect.map((r) => [r.subset, r]))
  for (const [subset, set] of [['all filters', members], ['dry season only', dry]]) {
    const c = lotContrast(set, 248, 251), s = saved[subset]
    assert.equal(c.na, s.n248, `${subset} n 248`)
    assert.equal(c.nb, s.n251, `${subset} n 251`)
    near(c.medianA, s.median248, 1e-6, `${subset} median 248`)
    near(c.medianB, s.median251, 1e-6, `${subset} median 251`)
    near(c.test.p, s.p, 1e-5, `${subset} p`)
  }
  // 04's box plot groups: 27 / 7 / 46 / 110
  const n = (lot, isDry) => members.filter((m) => m.lot === lot && (m.season === 'Dry (Oct-Feb)') === isDry).length
  assert.deepEqual([n(248, true), n(248, false), n(251, true), n(251, false)], [27, 7, 46, 110])
  // escape_by_group: all and per season
  const g = Object.fromEntries(exp.check.saved.escape_by_group.map((r) => [r.group, r]))
  near(boxStats(members.map((m) => m.escape)).median, g['ETAD — all'].median, 5e-5, 'ETAD all')
  near(boxStats(dry.map((m) => m.escape)).median, g['ETAD — dry season'].median, 5e-5, 'ETAD dry')
  assert.equal(dry.length, g['ETAD — dry season'].n)
})

test('the band means reproduce char_12 §4b\'s lot table within the dry season', { skip }, () => {
  const dry = membersOf(exp, 'notebook', [], new Set(), () => null).members.filter((m) => m.season === 'Dry (Oct-Feb)')
  for (const s of exp.check.saved.lot_bands) {
    const v = exp.band_means[s.band]
    const c = lotContrast(dry.map((m) => ({ lot: m.lot, escape: v[m.i] })), 248, 251)
    near(c.medianA, s.median248, 1e-4, `${s.band} 248`)
    near(c.medianB, s.median251, 1e-4, `${s.band} 251`)
    near(c.test.p, s.p, 1e-5, `${s.band} p`)
  }
})

test('the shipped 2nd derivatives give back the exported escape scores', { skip }, () => {
  const v = view('deriv'), p = exp.views.deriv.points
  assert.equal(p, exp.improve.deriv_p05.length)
  let same = 0
  for (let i = 0; i < exp.filters.id.length; i++) if (Math.abs(escapeOf(v, p, i, exp.improve.deriv_p05, exp.improve.deriv_p95) - exp.filters.escape[i]) < 0.0034) same++
  // quantisation and the envelope's 5 significant figures may move a point across an edge: at most one in a few filters
  assert.ok(same / exp.filters.id.length > 0.97, `${same} of ${exp.filters.id.length}`)
})

test('the view join: gallery rows in view, plus the spectra the gallery has no row for', { skip }, () => {
  const i = exp.filters.code.indexOf('INDH')
  const rows = [{ id: exp.filters.id[i], code: 'INDH', season: 'S' }]
  const known = new Set([exp.filters.id[i]])
  const v = membersOf(exp, 'view', rows, known, () => 'X')
  assert.equal(v.members.length, 1)
  assert.equal(v.members[0].season, 'S')
  const plus = membersOf(exp, 'plus', rows, known, () => 'X')
  assert.ok(plus.members.length > 1 && plus.members.every((m) => m.code === 'INDH'))
  assert.equal(plus.members.filter((m) => m.season === 'X').length, plus.unmatched)
})
