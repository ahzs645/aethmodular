// The explorer's saved MAC 10 fits, refitted from its per-filter readouts (explorerFit.ts), against calibration_runs.json
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { explorerFit, plausibilityOf } from './explorerFit.ts'

const runs = JSON.parse(fs.readFileSync(new URL('../../../../public/data/calibration_runs.json', import.meta.url), 'utf8'))
const LAMBDA_10 = 2.96
const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`)

const setOf = (run, set) => {
  const fabs = [], pred = []
  run.eval.ref.forEach((f, i) => {
    const y = run.eval.pred[i]
    if ((set === 'fixed' && !run.eval.fixed[i]) || !Number.isFinite(f) || !Number.isFinite(y)) return
    fabs.push(f); pred.push(y)
  })
  return { fabs, pred }
}

test('on the explorer\'s own set, the refit is its saved MAC 10 row (OLS, Deming at λ 2.96, R², n)', () => {
  let checked = 0
  for (const run of runs.runs) {
    for (const m of run.metrics.filter((x) => x.MAC === 10)) {
      const { fabs, pred } = setOf(run, m.evaluation_set)
      if (fabs.length !== m.n) continue // a set the eval readouts do not carry in full
      const f = explorerFit(fabs, pred, LAMBDA_10)
      const what = `${run.preset} ${run.target} ${m.evaluation_set}`
      near(f.ols_slope, m.ols_slope, 6e-4, `${what} OLS slope`)
      near(f.ols_intercept, m.ols_intercept, 6e-4, `${what} OLS intercept`)
      near(f.deming_slope, m.deming_slope, 6e-4, `${what} Deming slope`)
      near(f.deming_intercept, m.deming_intercept, 6e-4, `${what} Deming intercept`)
      near(f.R2, m.R2, 6e-4, `${what} R²`)
      checked++
    }
  }
  assert.ok(checked >= 20, `only ${checked} saved rows checked`)
})

test('leaving a filter out moves the refit, and too few filters give no fit', () => {
  const run = runs.runs.find((r) => r.preset === 'pool_raw' && r.target === 'addis')
  const { fabs, pred } = setOf(run, 'fixed')
  const all = explorerFit(fabs, pred, LAMBDA_10), less = explorerFit(fabs.slice(1), pred.slice(1), LAMBDA_10)
  assert.equal(less.n, all.n - 1)
  assert.notEqual(less.ols_slope, all.ols_slope)
  assert.equal(explorerFit(fabs.slice(0, 2), pred.slice(0, 2), LAMBDA_10), null)
})

test('the plausibility card over every dated Addis filter is the explorer\'s saved one', () => {
  let checked = 0
  for (const run of runs.runs.filter((r) => r.target === 'addis' && r.plausibility)) {
    const p = plausibilityOf(run.eval), s = run.plausibility
    near(p.median, s.median, 1e-4, `${run.preset} median`)
    near(p.negative_pct, s.negative_pct, 0.051, `${run.preset} negative %`)
    near(p.above_8_pct, s.above_8_pct, 0.051, `${run.preset} above 8 %`)
    for (const [g, v] of Object.entries(s.group_medians)) near(p.group_medians[g], v, 1e-4, `${run.preset} ${g}`)
    checked++
  }
  assert.ok(checked >= 5)
  const run = runs.runs.find((r) => r.preset === 'ocec800_airspec' && r.target === 'addis')
  assert.equal(plausibilityOf(run.eval, (i) => i !== 0).n, plausibilityOf(run.eval).n - 1)
})
