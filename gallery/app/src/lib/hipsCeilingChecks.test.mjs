// The ceiling checks (gallery/data/export_hips_ceiling_checks.py) against the browser: every Addis outcome the exporter
// wrote for a hyperbolic ceiling is the same fit lib/hipsLinearise makes from linearise.json, and the published central
// ceiling reproduces the closure's numbers.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { fitAgainstHips, macForSlopeOne } from './hipsLinearise.ts'

const read = (p) => JSON.parse(fs.readFileSync(new URL(`../../public/data/hips_linearise/${p}`, import.meta.url), 'utf8'))
const checks = read('ceiling_checks.json'), lin = read('linearise.json')

test('each hyperbolic ceiling gives the same Addis fit in the browser as in the exporter', () => {
  const rows = checks.addis.filter((r) => r.ceiling != null && !r.label.startsWith('Shape:') && r.past_ceiling === 0)
  assert.ok(rows.length >= 10)
  for (const r of rows) {
    const f = fitAgainstHips(lin.filters, lin.lambda, r.ceiling)
    assert.ok(Math.abs(f.slope - r.slope) < 2e-3 && Math.abs(f.intercept - r.intercept) < 5e-3, `${r.label}: ${f.slope} ${f.intercept} vs ${r.slope} ${r.intercept}`)
    assert.ok(Math.abs(macForSlopeOne(lin.filters, lin.lambda, r.ceiling, lin.mac) / r.mac_for_slope_one - 1) < 5e-3, r.label)
  }
  const measured = checks.addis.find((r) => r.group === 'measured')
  assert.ok(Math.abs(measured.slope - 2.31) < 0.01 && Math.abs(measured.intercept + 6.17) < 0.01)
  assert.equal(checks.instruments.find((r) => r.key === 'previous').tau_inf, 2.2596)
})
