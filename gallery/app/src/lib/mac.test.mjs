import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { demingLambda, slopeAt, bcAt, macView, solveHipsMoments, rmseAtMac, meanResidualAtMac, meanBcFromResidual, rmseFromReadouts } from './macMath.ts'
import { regression } from './stats.ts'

test('λ scales with MAC², 1 off an Fabs axis', () => {
  assert.equal(demingLambda(10, 2.96), 2.96)
  assert.ok(Math.abs(demingLambda(6, 2.96) - 2.96 * 0.36) < 1e-12)
  assert.equal(demingLambda(null, 2.96), 1)
})

test('a fit against Fabs/MAC: slope scales by MAC/10, intercept stays (OLS and Deming with λ ∝ MAC²)', () => {
  let seed = 3
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
  const fabs = Array.from({ length: 60 }, () => 20 + 60 * rnd())
  const y = fabs.map((f) => 0.18 * f - 2 + rnd())
  const at = (mac) => regression(fabs.map((f) => f / mac), y, { errorsInVariables: true, lambda: demingLambda(mac, 2.96) })
  const s10 = at(10), s6 = at(6)
  assert.ok(Math.abs(slopeAt(s10.slope, 6) - s6.slope) < 1e-9)
  assert.ok(Math.abs(slopeAt(s10.demingSlope, 6) - s6.demingSlope) < 1e-9)
  assert.ok(Math.abs(s10.demingIntercept - s6.demingIntercept) < 1e-9)
  assert.ok(Math.abs(s10.intercept - s6.intercept) < 1e-9)
})

test('HIPS BC and the view object', () => {
  assert.equal(bcAt(5, 5), 10)
  assert.equal(bcAt(null, 5), null)
  const v = macView(12.5, 10, 2.96)
  assert.equal(v.changed, true)
  assert.equal(v.fromFabs(50), 4)
  assert.equal(v.slope(1), 1.25)
  assert.equal(v.label, 'MAC 12.5')
})

// ---- RMSE and mean residual against Fabs/MAC (the 1:1 line)

/** a synthetic evaluation set and many "calibrations" scored on it, with everything the explorer saves */
function synthetic(nConfigs = 40) {
  let seed = 11
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
  const fabs = Array.from({ length: 120 }, () => 15 + 70 * rnd())
  const at = (mac) => fabs.map((f) => f / mac)
  const configs = Array.from({ length: nConfigs }, (_, j) => {
    const b = 0.4 + 1.6 * rnd(), a = -6 + 8 * rnd(), s = 0.2 + 2 * rnd()
    return fabs.map((f) => a + b * (f / 10) + s * (rnd() - 0.5))
  })
  const ols = (x, y) => {
    const n = x.length, mx = x.reduce((p, v) => p + v) / n, my = y.reduce((p, v) => p + v) / n
    let sxy = 0, sxx = 0, syy = 0
    x.forEach((v, i) => { sxy += (v - mx) * (y[i] - my); sxx += (v - mx) ** 2; syy += (y[i] - my) ** 2 })
    return { slope: sxy / sxx, intercept: my - (sxy / sxx) * mx, r2: (sxy * sxy) / (sxx * syy) }
  }
  const rmse = (x, y) => Math.sqrt(y.reduce((p, v, i) => p + (v - x[i]) ** 2, 0) / y.length)
  const bias = (x, y) => y.reduce((p, v, i) => p + (v - x[i]), 0) / y.length
  const x10 = at(10), mean = x10.reduce((p, v) => p + v) / x10.length
  const variance = x10.reduce((p, v) => p + (v - mean) ** 2, 0) / x10.length
  return { at, configs, ols, rmse, bias, x10, mean, variance }
}

test('HIPS BC mean and variance are recovered from the saved fits alone', () => {
  const { configs, ols, rmse, x10, mean, variance } = synthetic()
  // what the explorer saves, rounded as it rounds (4 dp)
  const r4 = (v) => Math.round(v * 1e4) / 1e4
  const m = solveHipsMoments(configs.map((y) => { const f = ols(x10, y); return { slope: r4(f.slope), intercept: r4(f.intercept), r2: r4(f.r2), rmse: r4(rmse(x10, y)) } }))
  assert.ok(m)
  assert.ok(Math.abs(m.mean - mean) < 1e-3, `mean ${m.mean} vs ${mean}`)
  assert.ok(Math.abs(m.variance / variance - 1) < 2e-3, `variance ${m.variance} vs ${variance}`)
  assert.ok(m.miss < 1e-3)
})

test('RMSE and mean residual against Fabs/MAC at any MAC match a direct recomputation', () => {
  const { at, configs, ols, rmse, bias, x10, mean, variance } = synthetic(6)
  for (const y of configs) {
    const f = ols(x10, y), saved = { slope: f.slope, intercept: f.intercept, rmse: rmse(x10, y) }
    assert.equal(rmseAtMac(saved, { mean, variance }, 10), saved.rmse)
    for (const mac of [4.5, 6, 8.3, 13, 17]) {
      assert.ok(Math.abs(rmseAtMac(saved, { mean, variance }, mac) - rmse(at(mac), y)) < 1e-9)
      assert.ok(Math.abs(meanResidualAtMac(f.slope, f.intercept, mean, mac) - bias(at(mac), y)) < 1e-9)
    }
    assert.ok(Math.abs(meanBcFromResidual(f.slope, f.intercept, bias(x10, y)) - mean) < 1e-9 || Math.abs(f.slope - 1) < 0.05)
  }
})

test('a set the fits do not agree on, too few fits, or a slope near 1 give nothing rather than a guess', () => {
  const { configs, ols, rmse, x10, at } = synthetic(30)
  const mixed = configs.map((y, j) => { const x = j % 2 ? x10 : at(6); const f = ols(x, y); return { ...f, rmse: rmse(x, y) } })
  assert.equal(solveHipsMoments(mixed), null)
  assert.equal(solveHipsMoments(mixed.slice(0, 3)), null)
  assert.equal(meanBcFromResidual(1.02, -1, -0.9), null)
  assert.equal(rmseAtMac({ slope: 1, intercept: 0, rmse: 1 }, null, 6), null)
})

// the gallery's own exports: the grid's fits solve each evaluation set, and the result reproduces the
// explorer's saved MAC 6 and MAC 17 RMSEs, which the grid never carried
const data = (f) => JSON.parse(fs.readFileSync(new URL(`../../public/data/${f}`, import.meta.url), 'utf8'))

test('calibration.json: every evaluation set solves, and reproduces the explorer\'s MAC 6 and 17 RMSEs', () => {
  const calib = data('calibration.json'), runs = data('calibration_runs.json')
  const sets = new Map()
  for (const r of calib.grid) { const k = `${r.tg}|${r.el}`; if (!sets.has(k)) sets.set(k, []); sets.get(k).push(r) }
  const moments = new Map([...sets].map(([k, rows]) => [k, solveHipsMoments(rows.map((r) => ({ slope: r.om, intercept: r.ob, r2: r.r2, rmse: r.rmse })))]))
  for (const [k, m] of moments) assert.ok(m && m.miss < 1e-3, `${k} did not solve`)
  // the Addis fixed set, computed directly from its 190 filters' Fabs
  const addis = runs.runs.find((r) => r.target === 'addis' && r.eval.ref.length)
  const x = addis.eval.ref.filter((_, i) => addis.eval.fixed[i]).map((f) => f / 10)
  const mean = x.reduce((p, v) => p + v) / x.length, variance = x.reduce((p, v) => p + (v - mean) ** 2, 0) / x.length
  assert.ok(Math.abs(moments.get('addis|all').mean - mean) < 1e-3)
  assert.ok(Math.abs(moments.get('addis|all').variance / variance - 1) < 1e-3)
  let checks = 0
  for (const run of runs.runs) {
    const row = (set, mac) => run.metrics.find((m) => m.evaluation_set === set && m.MAC === mac)
    const set = row('fixed', 10) ? 'fixed' : 'all', f = row(set, 10)
    for (const mac of [6, 17]) {
      const want = row(set, mac)?.RMSE
      if (f == null || want == null) continue
      const got = rmseAtMac({ slope: f.ols_slope, intercept: f.ols_intercept, rmse: f.RMSE }, moments.get(`${run.target}|all`), mac)
      assert.ok(Math.abs(got - want) < 2e-3, `${run.preset} ${run.target} MAC ${mac}: ${got} vs ${want}`)
      checks++
    }
  }
  assert.ok(checks >= 80, `only ${checks} explorer readouts checked`)
})

test('ftir_36 and ftir_53: the holdout means solved from RMSE agree with those the bias columns imply', () => {
  const t = data('research/spectral/applicability.json').tables.target_summary
  for (const target of ['addis_holdout', 'delhi_holdout']) {
    const idx = t.target.map((v, i) => (v === target ? i : -1)).filter((i) => i >= 0)
    const m = solveHipsMoments(idx.map((i) => ({ slope: t.slope[i], intercept: t.intercept[i], r2: t.R2[i], rmse: t.RMSE_1to1[i] })))
    assert.ok(m, target)
    for (const i of idx) assert.ok(Math.abs(meanResidualAtMac(t.slope[i], t.intercept[i], m.mean, 10) - t.mean_bias[i]) < 0.01, `${target} row ${i}`)
  }
  // Bishoftu's accepted 26 are the grid's Bishoftu set: the mean its residual implies is the one the grid solves
  const b = data('research/spectral/committee.json').tables.bishoftu
  const accepted = b.group.findIndex((g) => g.startsWith('Bishoftu, accepted'))
  const grid = data('calibration.json').grid.filter((r) => r.tg === 'etbi' && r.el === 'all')
  const etbi = solveHipsMoments(grid.map((r) => ({ slope: r.om, intercept: r.ob, r2: r.r2, rmse: r.rmse })))
  assert.ok(Math.abs(meanBcFromResidual(b.ols_slope[accepted], b.ols_intercept[accepted], b.mean_residual[accepted]) - etbi.mean) < 0.01)
})

test('RMSE saved at MAC 6, 10 and 17 fixes it at every MAC (the blind-half splits): checked on the runs\' own filters', () => {
  const runs = data('calibration_runs.json')
  let checks = 0, worst = 0
  for (const run of runs.runs) {
    const { ref, pred, fixed } = run.eval
    for (const set of ['fixed', 'all']) {
      const saved = run.metrics.filter((m) => m.evaluation_set === set).map((m) => ({ mac: m.MAC, rmse: m.RMSE }))
      if (saved.length < 3) continue
      for (const mac of [4.5, 8, 12.5, 25]) {
        let sum = 0, n = 0
        ref.forEach((f, i) => { if ((set === 'all' || fixed[i]) && Number.isFinite(f) && Number.isFinite(pred[i])) { sum += (pred[i] - f / mac) ** 2; n++ } })
        if (!n) continue
        const err = Math.abs(rmseFromReadouts(saved, mac) - Math.sqrt(sum / n))
        worst = Math.max(worst, err); checks++
      }
    }
  }
  assert.ok(checks >= 200, `only ${checks} checks`)
  assert.ok(worst < 2e-3, `worst ${worst}`)
  console.log(`three-MAC rebuild: ${checks} checks, worst ${worst.toExponential(1)} µg/m³`)
  assert.equal(rmseFromReadouts([{ mac: 10, rmse: 1 }, { mac: 6, rmse: 2 }], 8), null)
  assert.equal(rmseFromReadouts([{ mac: 10, rmse: 1.25 }, { mac: 6, rmse: 2 }, { mac: 17, rmse: 3 }], 10), 1.25)
})
