import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { metricRowLive, splitIndices, splitCheckLive } from './runFit.ts'

const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} != ${b}`)

test('equal-n halves are stratified by the fixed set and date-ordered, missing dates last', () => {
  // 6 fixed (dates out of order) + 4 non-fixed (one without a date)
  const ev = {
    id: [], ref: [], group: [], deployed: null,
    pred: Array(10).fill(1),
    date: ['2024-01-06', '2024-01-01', '2024-01-03', '2024-01-02', '2024-01-05', '2024-01-04', '2024-02-02', null, '2024-02-01', '2024-02-03'],
    fixed: [true, true, true, true, true, true, false, false, false, false],
  }
  const all = [...Array(10).keys()]
  // fixed in date order: 1,3,2,5,4,0; non-fixed: 8,6,9,7
  assert.deepEqual(splitIndices(ev, all, 'early'), [1, 2, 3, 6, 8])
  assert.deepEqual(splitIndices(ev, all, 'late'), [0, 4, 5, 7, 9])
  assert.deepEqual(splitIndices(ev, all, 'even'), [1, 2, 4, 8, 9])
  assert.deepEqual(splitIndices(ev, all, 'odd'), [0, 3, 5, 6, 7])
  // under 3 left: the explorer skips the split
  assert.equal(splitIndices(ev, [0, 1, 6, 7], 'early'), null)
})

test('a dropped filter leaves the fit', () => {
  const ev = { id: [], group: [], date: null, deployed: null, fixed: Array(6).fill(true),
    ref: [10, 20, 30, 40, 50, 60], pred: [1, 2, 3, 4, 5, 60] }
  const all = metricRowLive(ev, 'all', 10, 2.96)
  const kept = metricRowLive(ev, 'all', 10, 2.96, (i) => i !== 5)
  assert.equal(all.n, 6)
  assert.equal(kept.n, 5)
  near(kept.ols_slope, 1, 1e-12, 'slope without the outlier')
  near(kept.deming_slope, 1, 1e-12, 'Deming slope without the outlier')
  near(kept.RMSE, 0, 1e-12, 'RMSE on the 1:1 line')
})

// the exported runs (git-ignored): with every filter kept the browser reproduces the explorer's saved rows
const path = new URL('../../../public/data/calibration_runs.json', import.meta.url)
const runs = fs.existsSync(path) ? JSON.parse(fs.readFileSync(path, 'utf8')) : null
test('the saved metrics and blind halves are reproduced from the per-filter readouts', { skip: !runs && 'no calibration_runs.json export' }, () => {
  let checked = 0
  for (const run of runs.runs) {
    const ev = run.eval
    for (const row of run.metrics) {
      const live = metricRowLive(ev, row.evaluation_set, row.MAC, 2.96)
      assert.ok(live, `${run.preset}/${run.target} ${row.evaluation_set} @${row.MAC}`)
      assert.equal(live.n, row.n)
      for (const k of ['ols_slope', 'ols_intercept', 'deming_slope', 'deming_intercept', 'R2', 'RMSE']) near(live[k], row[k], 1e-4, `${run.preset}/${run.target} ${k}`)
      checked++
    }
    for (const mac of run.ref_kind === 'fabs' ? [10, 6] : [null]) {
      const live = splitCheckLive(ev, mac, 2.96)
      assert.deepEqual(live.map((s) => s.split), run.split_check.map((s) => s.split))
      for (const s of live) {
        const saved = run.split_check.find((x) => x.split === s.split)
        assert.equal(s.n, saved.n)
        const row = saved.metrics.find((r) => r.evaluation_set === s.set && r.MAC === mac)
        assert.ok(row, `${run.preset}/${run.target} ${s.split} ${s.set} @${mac}`)
        for (const k of ['deming_slope', 'deming_intercept', 'R2', 'RMSE']) near(s.m[k], row[k], 1e-4, `${run.preset}/${run.target} ${s.split} ${k}`)
        checked++
      }
    }
  }
  assert.ok(checked > 100)
})
