// The focus-site sides of the IMPROVE envelope (envelope.ts), recomputed from filters.json and the HIPS optics
// export, against the exporter's saved rows (research/improve_envelope/envelope.json)
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { addisMatched, focusFilter, siteMassSpace } from './envelope.ts'

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
