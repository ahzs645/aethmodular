// The rules of the "Linearising HIPS across SPARTAN" page (lib/hipsRules), held to the 11 Nov screen: the filters
// counted, each site's exponent as measured and after every filter is linearised (hipsSites.json, from
// docs/hips-network-2026-11-11.json), and what each rule can and cannot do to the bend.
// Inputs: spartan_reference.json (the screen's own input).
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { CENTRAL_CEILING, HIPS_SITE_READINGS, SEEN_SITES, correction } from './hipsLinearise.ts'
import { exponent, networkFilters, ruleFactors, RULE_KEYS, scoreRule, siteOutcomes, stepAt } from './hipsRules.ts'

const spartan = JSON.parse(fs.readFileSync(new URL('../../public/data/spartan_reference.json', import.meta.url), 'utf8'))
const filters = networkFilters(spartan.rows)
const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`)
const at = (rule, cut) => ({ rule, ceiling: CENTRAL_CEILING, cut })

test('the filters are the screen\'s, site by site', () => {
  assert.equal(new Set(filters.map((f) => f.code)).size, HIPS_SITE_READINGS.length)
  for (const s of HIPS_SITE_READINGS) assert.equal(filters.filter((f) => f.code === s.code).length, s.filters, `${s.code} filters`)
})

test('each site\'s exponent as measured and with every filter linearised is the screen\'s', () => {
  const sites = new Map(siteOutcomes(filters, at('filter')).map((s) => [s.code, s]))
  let tested = 0
  for (const s of HIPS_SITE_READINGS) {
    const own = sites.get(s.code)
    if (s.exponent == null) { assert.ok(Number.isNaN(own.b), `${s.code} has no exponent`); continue }
    tested++
    assert.equal(own.tested, s.tested_filters, `${s.code} tested filters`)
    near(own.b, s.exponent, 2e-3, `${s.code} as measured`)
    near(own.bAfter, s.exponent_after, 2e-3, `${s.code} linearised`)
    near(own.bAfter, own.bOwn, 1e-12, `${s.code} own`)
  }
  assert.equal(tested, 23)
})

test('a rule leaves a filter as measured or gives it a factor; the site rules follow the screen\'s readings', () => {
  const own = ruleFactors(filters, at('filter'))
  filters.forEach((f, i) => { const k = correction(f.tau, CENTRAL_CEILING); if (Number.isFinite(k)) near(own[i], k, 1e-12, f.id); else assert.equal(own[i], Infinity) })
  const seen = ruleFactors(filters, at('seen')), expected = ruleFactors(filters, at('expected'))
  filters.forEach((f, i) => {
    assert.equal(seen[i], SEEN_SITES.has(f.code) ? own[i] : 1, `${f.id} where seen`)
    // every site where it is seen is one where it is expected
    if (seen[i] !== 1) assert.equal(expected[i], own[i])
  })
  assert.equal(new Set(filters.filter((_, i) => expected[i] !== 1).map((f) => f.code)).size, 17)
  // the cut-off: nothing short of it moves, everything past it gets its own factor, and the smallest factor applied is the step
  const cut = ruleFactors(filters, at('cut', 0.2))
  const moved = cut.filter((k) => k !== 1)
  assert.ok(moved.length > 0 && Math.min(...moved) >= stepAt(0.2) - 1e-9)
  near(stepAt(0.2), 1.25, 1e-12, 'step at a fifth')
  // one factor per site: every filter of a site gets the same one
  const site = ruleFactors(filters, at('site'))
  const bySite = new Map()
  filters.forEach((f, i) => { if (!bySite.has(f.code)) bySite.set(f.code, site[i]); assert.equal(site[i], bySite.get(f.code)) })
})

test('one factor per site moves the level and leaves the bend where it was', () => {
  for (const s of siteOutcomes(filters, at('site')).filter((x) => Number.isFinite(x.b))) {
    near(s.bAfter, s.b, 1e-9, `${s.code} exponent`)
    assert.ok(s.fabsAfter >= s.fabs)
  }
  const addis = siteOutcomes(filters, at('site')).find((s) => s.code === 'ETAD')
  // at Addis the site factor lands on the per-filter median, and misses more than half the filters by over a tenth
  near(addis.fabsAfter / addis.fabsOwn, 1, 0.02, 'Addis median')
  assert.ok(addis.offTenth > 0.5)
})

test('the scoreboard: what each rule changes and what it does to the bend', () => {
  const score = Object.fromEntries(RULE_KEYS.map((r) => [r, scoreRule(filters, at(r, 0.2))]))
  assert.equal(score.filter.filters, 3733)
  assert.equal(score.filter.offTenth, 0)
  assert.equal(score.filter.tested, 23)
  // fewer filters are touched as the rule narrows from every filter to the sites where it is seen
  assert.ok(score.filter.corrected > score.expected.corrected && score.expected.corrected > score.seen.corrected)
  // one factor per site moves no site towards proportional; every filter by its own factor moves the most
  assert.equal(score.site.closer, 0)
  assert.equal(score.site.nearAfter, score.site.nearBefore)
  assert.ok(score.filter.closer >= score.seen.closer && score.seen.closer > 0)
  // "where seen" pushes no site that was not corrected, so it sends fewer sites away from proportional than every filter does
  assert.ok(score.seen.further <= score.filter.further)
})

test('the exponent needs 40 filters with FTIR EC and PM2.5', () => {
  assert.ok(Number.isNaN(exponent(filters.slice(0, 20)).b))
})
