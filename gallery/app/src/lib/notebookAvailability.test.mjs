import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { calendarMonths, daysInMonth, intersectDays, studyOverlap } from './notebookAvailability.ts'

test('day intersections deduplicate days without joining adjacent dates', () => {
  assert.deepEqual(intersectDays(['2024-01-01', '2024-01-01', '2024-01-02'], ['2024-01-02', '2024-01-03']), ['2024-01-02'])
  assert.deepEqual(calendarMonths('2023-11', '2024-02'), ['2023-11', '2023-12', '2024-01', '2024-02'])
  assert.deepEqual(calendarMonths('2024-02', '2023-11'), [])
  assert.deepEqual(calendarMonths('2024-13', '2025-01'), [])
})

test('unavailable completeness differs from a measured zero overlap; filter IDs differ from days', () => {
  const site = {
    ftir: { days: ['2024-01-01', '2024-01-02'] }, hips: { days: ['2024-01-01'] },
    aeth: { days: ['2024-01-02'], days50: [], days80: null },
    paired_filters: [{ id: 'a', days: ['2024-01-01'] }, { id: 'b', days: ['2024-01-01'] }],
  }
  const measured = studyOverlap(site, 'Any finite IR')
  assert.equal(measured.matched, 2)
  assert.equal(measured.both.length, 1)
  assert.deepEqual(measured.triple, [])
  assert.equal(measured.matchedWithAeth, 0)
  assert.deepEqual(studyOverlap(site, 'At least 50%').aeth, [])
  assert.equal(studyOverlap(site, 'At least 80%').triple, null)
  assert.equal(studyOverlap(site, 'At least 80%').matchedWithAeth, null)
  assert.equal(studyOverlap({ ...site, aeth: null }, 'Any finite IR').aeth, null)
})

const exported = JSON.parse(readFileSync(new URL('../../public/data/availability.json', import.meta.url), 'utf8'))
test('packaged monthly counts reconcile with each product population, including zero months', () => {
  const totals = new Map()
  const keys = new Set()
  for (const row of exported.network.monthly) {
    assert.ok(row.count > 0 && Number.isInteger(row.count))
    const key = `${row.code}|${row.product}`
    const monthKey = `${key}|${row.month}`
    assert.ok(!keys.has(monthKey), `Duplicate site/product/month: ${monthKey}`)
    keys.add(monthKey)
    totals.set(key, (totals.get(key) ?? 0) + row.count)
  }
  for (const site of exported.network.sites) for (const [product, stats] of Object.entries(site.products)) {
    assert.equal(totals.get(`${site.code}|${product}`) ?? 0, stats.n_samples ?? 0, `${site.code} · ${product}`)
  }
  assert.ok(exported.network.hips_inventory_total > exported.network.sites.reduce((sum, s) => sum + (s.products.HIPS?.n_samples ?? 0), 0), 'Distinct HIPS snapshots stay distinct')
  assert.ok(exported.network.hips_nonpositive > 0, 'Finite nonpositive values are included')
  for (const site of exported.network.sites) for (const stats of Object.values(site.products)) {
    if (stats.newer_date_max) assert.ok(stats.newer_date_max <= exported.network.public_release)
  }
})

test('study exports preserve sample dates, physical filters and missing completeness', () => {
  for (const site of exported.study) {
    for (const stream of [site.ftir, site.hips]) {
      assert.deepEqual(stream.days, [...new Set(stream.days)].sort())
      assert.ok(stream.days.length <= stream.reported_days)
      assert.ok(stream.days.length <= stream.filters)
    }
    assert.equal(site.date_conflicts, 0)
    assert.equal(new Set(site.paired_filters.map(f => f.id)).size, site.paired_filters.length)
    const all = studyOverlap(site, 'Any finite IR')
    assert.ok(all.triple.length <= all.both.length)
    assert.ok(all.matchedWithAeth <= all.matched)
  }
  const jpl = exported.study.find(s => s.code === 'USPA')
  assert.ok([...jpl.ftir.days, ...jpl.hips.days].every(day => day <= '2023-12-31'))
  assert.ok(jpl.ftir.analysis_last > '2023-12-31')
  const addis = exported.study.find(s => s.code === 'ETAD')
  assert.equal(studyOverlap(addis, 'At least 80%').aeth, null)
})

test('AERONET availability: monthly counts add up to each site, and photometer days fit their month', () => {
  assert.equal(daysInMonth('2024-02'), 29)
  assert.equal(daysInMonth('2023-02'), 28)
  assert.equal(daysInMonth('2024-13'), 0)
  const { sites, monthly } = exported.aeronet
  for (const site of sites) {
    const rows = monthly.filter(m => m.code === site.code)
    for (const key of ['filters', 'filters_aod', 'filters_inv', 'aod_days', 'inv_days']) {
      assert.equal(rows.reduce((sum, m) => sum + m[key], 0), site[key], `${site.code} · ${key}`)
    }
    for (const m of rows) {
      assert.ok(m.aod_days <= daysInMonth(m.month) && m.inv_days <= daysInMonth(m.month), `${site.code} · ${m.month}`)
      assert.ok(m.filters_inv <= m.filters && m.filters_aod <= m.filters)
    }
  }
})
