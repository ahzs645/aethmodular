import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { test } from 'node:test'
import { regression } from './stats.ts'

const source = new URL('../../public/data/meeting/followup_20260917.json', import.meta.url)

test('meeting complete-pair counts and selectable Deming fits reproduce the saved analysis', { skip: !existsSync(source) }, () => {
  const data = JSON.parse(readFileSync(source, 'utf8'))
  const a = data.addis
  let checked = 0
  for (const row of data.stats.filter(s => s.population === 'addis')) {
    const ii = a.fit_id.map((id, i) => id === row.fit_id &&
      (row.eval_group === 'All Addis' || a.season[i] === row.eval_group) ? i : -1)
      .filter(i => i >= 0 && Number.isFinite(a.fabs10[i]) && Number.isFinite(a.ftir_ec_ugm3[i]))
    if (!ii.length) continue // Reproduction-only fits have no displayed predictions.
    assert.equal(ii.length, row.n, `${row.fit_id}: displayed pairs must match the score's denominator`)
    assert.ok(Math.abs(row.deming_lambda - data.summary.lambda) < 0.0001, 'Addis error ratio must survive export rounding')
    const stats = regression(ii.map(i => a.fabs10[i]), ii.map(i => a.ftir_ec_ugm3[i]), {
      errorsInVariables: true, lambda: row.deming_lambda,
    })
    for (const [actual, expected] of [[stats.demingSlope, row.deming_slope],
      [stats.demingIntercept, row.deming_intercept], [stats.r2, row.R2]]) {
      assert.ok(Math.abs(actual - expected) < 0.001, `${row.fit_id}: ${actual} != ${expected}`)
    }
    checked++
  }
  assert.ok(checked > 0, 'Check real saved fits, not an empty export')
})
