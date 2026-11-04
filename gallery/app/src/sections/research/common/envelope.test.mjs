// The focus-site sides of the IMPROVE envelope (envelope.ts), recomputed from filters.json and the HIPS optics
// export, against the exporter's saved rows (research/improve_envelope/envelope.json)
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { addisMatched, focusFilter, opticsJoin, rowDeposit, rowFilter, rowOptics, siteMassSpace, siteMatched } from './envelope.ts'

const read = (f) => JSON.parse(fs.readFileSync(new URL(`../../../../public/data/${f}`, import.meta.url), 'utf8'))
const envelope = read('research/improve_envelope/envelope.json')
const opticsFile = read('research/hips_optics/filters.json')
const optics = new Map(opticsFile.rows.map((r) => Object.fromEntries(opticsFile.columns.map((c, i) => [c, r[i]]))).filter((o) => o.type !== 'FB').map((o) => [o.id, o]))
const filters = read('filters.json')
const focus = (rows) => rows.flatMap((r) => { const o = optics.get(r.id); const f = o ? focusFilter(r, o) : null; return f ? [f] : [] })
// the rows the exporter used: its exclusion registry (outliers.apply_exclusion_flags) is filters.json's but for the
// PM10 filter ETAD-0248, which it keeps: neither the gallery's Hide nor its Include
const kept = focus(filters.rows.filter((r) => !r.excluded || r.id === 'ETAD-0248'))
// the exporter rounds to 4 significant figures
const near = (a, b, what) => assert.ok(b === null ? a === null : Math.abs(a - b) <= Math.abs(b) * 6e-4 + 1e-9, `${what}: ${a} vs ${b}`)
const sameQ = (mine, saved, what) => { assert.equal(mine.n, saved.n, `${what} n`); for (const k of ['p05', 'p25', 'p50', 'p75', 'p95']) near(mine[k], saved[k], `${what} ${k}`) }

test('on the exporter\'s rows the Addis matched row is the saved one: quantiles and OLS of Fabs on EC', () => {
  const mine = addisMatched(kept), saved = envelope.matched[0]
  for (const k of ['fabs', 'fabs_ec', 'ec_conc', 'ec_ug']) sameQ(mine[k], saved[k], k)
  assert.equal(mine.fit.n, saved.fit.n)
  for (const k of ['slope', 'intercept', 'r2']) near(mine.fit[k], saved.fit[k], `fit ${k}`)
})

test('on the exporter\'s rows each focus site\'s mass-space rows are the saved ones', () => {
  const mine = siteMassSpace(kept)
  const saved = envelope.mass_space.filter((m) => m.group !== 'IMPROVE')
  assert.equal(mine.length, saved.length)
  for (const s of saved) sameQ(mine.find((m) => m.group === s.group && m.quantity === s.quantity), s, `${s.group} ${s.quantity}`)
})

test('under Hide the export\'s exclusions leave the Addis row; Include puts them back', () => {
  const hidden = focus(filters.rows.filter((r) => !r.excluded)), all = focus(filters.rows)
  assert.equal(addisMatched(hidden).fabs.n, addisMatched(kept).fabs.n - 1)
  assert.equal(addisMatched(all).fabs.n, addisMatched(kept).fabs.n)
})

test('a filter outside the optics export is read from its row', () => {
  const f = rowFilter({ code: 'ETBI', 'EC (FTIR)': 2, 'OC (FTIR)': 6, 'HIPS Fabs': 30, 'Sampled volume': 7 })
  assert.deepEqual(f, { code: 'ETBI', fabs: 30, ecConc: 2, ecUg: 14, ocUg: 42, volume: 7, area: 3.53 })
})

test('an IMPROVE row takes TOR EC and IMPROVE\'s deposit area', () => {
  const f = rowFilter({ code: 'YOSE1', 'EC (TOR)': 0.5, 'OC (TOR)': 2, 'HIPS Fabs': 4, 'Sampled volume': 33 })
  assert.equal(f.ecUg, 16.5)
  assert.equal(f.ocUg, 66)
  assert.equal(f.area, 3.5)
})

test('a row without Fabs, or with no EC, is not a filter here; no volume leaves the masses empty', () => {
  assert.equal(rowFilter({ code: 'X', 'EC (FTIR)': 1 }), null)
  assert.equal(rowFilter({ code: 'X', 'HIPS Fabs': 1 }), null)
  assert.equal(rowFilter({ code: 'X', 'EC (FTIR)': 1, 'HIPS Fabs': 2 }).ecUg, null)
})

test('the matched row works for any site', () => {
  const a = rowFilter({ code: 'CHTS', 'EC (FTIR)': 2, 'HIPS Fabs': 20, 'Sampled volume': 7 })
  const b = rowFilter({ code: 'INDH', 'EC (FTIR)': 3, 'HIPS Fabs': 30, 'Sampled volume': 7 })
  const m = siteMatched([a, b], 'CHTS', 'Beijing')
  assert.equal(m.group, 'Beijing')
  assert.equal(m.fabs.n, 1)
})

test('rowDeposit: SPARTAN\'s area for a filter with FTIR EC or no EC at all, IMPROVE\'s for a TOR row, none without a volume', () => {
  assert.deepEqual(rowDeposit({ 'EC (FTIR)': 2, 'Sampled volume': 7 }), { volume_m3: 7, area_cm2: 3.53, lot: null })
  assert.equal(rowDeposit({ 'Sampled volume': 7 }).area_cm2, 3.53)
  assert.equal(rowDeposit({ 'EC (TOR)': 0.5, 'Sampled volume': 33 }).area_cm2, 3.5)
  assert.deepEqual(rowDeposit({ 'EC (FTIR)': 2 }), { volume_m3: null, area_cm2: null, lot: null })
})

test('rowOptics: tau from Fabs, volume and area matches the export on every focus sample with a reading', () => {
  const samples = opticsFile.rows.map((r) => Object.fromEntries(opticsFile.columns.map((c, i) => [c, r[i]])))
    .filter((o) => o.type !== 'FB' && o.tau !== null && o.fabs > 0 && o.volume_m3 > 0)
  assert.equal(samples.length, 546)
  for (const o of samples) {
    const s = rowOptics({ id: o.id, code: o.code, date: o.date, 'HIPS Fabs': o.fabs, 'Sampled volume': o.volume_m3 })
    assert.ok(Math.abs(s.tau - o.tau) <= 1.5e-4, `${o.id}: ${s.tau} vs ${o.tau}`)
  }
})

test('rowOptics: no Fabs is no filter; no volume leaves tau and the area empty; the raw readings stay empty', () => {
  assert.equal(rowOptics({ id: 'a', code: 'ETBI', 'Sampled volume': 7 }), null)
  const f = rowOptics({ id: 'a', code: 'ETBI', 'HIPS Fabs': 20 })
  assert.equal(f.tau, null)
  assert.equal(f.area_cm2, null)
  const g = rowOptics({ id: 'a', code: 'ETBI', date: '2025-10-20', 'HIPS Fabs': 20, 'Sampled volume': 7 })
  assert.equal(g.tau, (20 * 7) / 353)
  for (const k of ['t', 'r', 'T1', 'R1', 'lot', 'cal_intercept', 'cal_slope', 'analysis_date']) assert.equal(g[k], null, k)
})

test('opticsJoin: the export\'s row where it has the filter, a stand-in only at a site it does not cover', () => {
  const rows = [{ id: 'ETAD-1', code: 'ETAD', type: 'PM2.5', fabs: 50 }, { id: 'ETAD-B', code: 'ETAD', type: 'FB', fabs: null }]
  const join = opticsJoin(rows)
  assert.equal(join({ id: 'ETAD-1', code: 'ETAD', 'HIPS Fabs': 1 }), rows[0])
  // a field blank is never a sample's optics, and a filter missing at a covered site does not become a stand-in
  assert.equal(join({ id: 'ETAD-B', code: 'ETAD', 'HIPS Fabs': 1, 'Sampled volume': 7 }), null)
  assert.equal(join({ id: 'ETAD-2', code: 'ETAD', 'HIPS Fabs': 1, 'Sampled volume': 7 }), null)
  assert.equal(join({ id: 'ETBI-1', code: 'ETBI', 'HIPS Fabs': 20, 'Sampled volume': 7 }).fabs, 20)
  assert.equal(join({ id: 'ETBI-2', code: 'ETBI' }), null)
})
