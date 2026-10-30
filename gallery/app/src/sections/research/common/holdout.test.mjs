// ftir_36/ftir_39's locked holdouts at any MAC (holdout.ts), against the notebooks' saved tables
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { domainSummaryFrom, holdoutMoments, perFilterAtMac } from './holdout.ts'
import { rmseAtMac, solveHipsMoments } from '../../../lib/macMath.ts'

const data = (f) => JSON.parse(fs.readFileSync(new URL(`../../../../public/data/research/spectral/${f}`, import.meta.url), 'utf8'))
const rowsOf = (t) => t.config.map((_, i) => Object.fromEntries(Object.keys(t).map((k) => [k, t[k][i]])))
const tables = data('applicability.json').tables
const perFilter = rowsOf(tables.domain_filters), saved = rowsOf(tables.domain_summary), summary = rowsOf(tables.target_summary)
const fabs = data('holdout_fabs.json')
const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`)

test('the reconstructed Fabs are the notebook\'s: residual + Fabs/10 refits ftir_36\'s six ordinary-PLS lines', () => {
  assert.equal(Object.keys(fabs.rows).length, 40)
  for (const s of summary.filter((r) => r.variant === 'ordinary_pls')) {
    const g = perFilter.filter((r) => r.config === s.config && r.target === s.target)
    const x = g.map((r) => fabs.rows[String(r.ExternalFilterId)].fabs / 10), y = g.map((r, i) => r.residual_ugm3 + x[i])
    const mx = x.reduce((a, b) => a + b) / x.length, my = y.reduce((a, b) => a + b) / y.length
    let sxy = 0, sxx = 0, syy = 0
    x.forEach((v, i) => { sxy += (v - mx) * (y[i] - my); sxx += (v - mx) ** 2; syy += (y[i] - my) ** 2 })
    near(sxy / sxx, s.slope, 2e-3, `${s.config} ${s.target} slope`)
    near(my - (sxy / sxx) * mx, s.intercept, 2e-3, `${s.config} ${s.target} intercept`)
    near(sxy * sxy / (sxx * syy), s.R2, 2e-3, `${s.config} ${s.target} R²`)
  }
})

test('at MAC 10 the rebuilt summary is ftir_39\'s: RMSE inside and outside the source 95 %, ρ and its p', () => {
  const rebuilt = domainSummaryFrom(perFilterAtMac(perFilter, fabs, 10), saved)
  rebuilt.forEach((r, i) => {
    const s = saved[i]
    near(r.RMSE_inside, s.RMSE_inside, 1e-3, `${s.config} ${s.target} RMSE inside`)
    if (s.RMSE_outside == null) assert.equal(r.RMSE_outside, null)
    else near(r.RMSE_outside, s.RMSE_outside, 1e-3, `${s.config} ${s.target} RMSE outside`)
    near(r.rho_domain_percentile_vs_abs_error, s.rho_domain_percentile_vs_abs_error, 1e-3, `${s.config} ${s.target} ρ`)
    near(r.rho_p, s.rho_p, 2e-3, `${s.config} ${s.target} p`)
  })
})

test('the exact holdout moments agree with those solved from ftir_36\'s fits, and both routes give one RMSE at MAC 6', () => {
  const exact = holdoutMoments(perFilter, fabs)
  for (const target of ['addis_holdout', 'delhi_holdout']) {
    const fits = summary.filter((r) => r.target === target)
    const solved = solveHipsMoments(fits.map((r) => ({ slope: r.slope, intercept: r.intercept, r2: r.R2, rmse: r.RMSE_1to1 })))
    near(exact.get(target).mean, solved.mean, 5e-3, `${target} mean`)
    near(exact.get(target).variance / solved.variance, 1, 1e-2, `${target} variance`)
  }
  // per filter: moved residuals; summary route: the saved fit and RMSE with the exact moments
  const at6 = perFilterAtMac(perFilter, fabs, 6)
  for (const s of summary.filter((r) => r.variant === 'ordinary_pls')) {
    const e = at6.filter((r) => r.config === s.config && r.target === s.target).map((r) => r.residual_ugm3)
    const direct = Math.sqrt(e.reduce((a, v) => a + v * v, 0) / e.length)
    near(direct, rmseAtMac({ slope: s.slope, intercept: s.intercept, rmse: s.RMSE_1to1 }, exact.get(s.target), 6), 2e-3, `${s.config} ${s.target} RMSE at MAC 6`)
  }
})

test('a filter without Fabs leaves the panel at the notebook\'s MAC rather than mixing MACs', () => {
  const missing = { ...fabs, rows: { ...fabs.rows } }
  delete missing.rows[String(perFilter[0].ExternalFilterId)]
  assert.equal(perFilterAtMac(perFilter, missing, 6), null)
  assert.equal(perFilterAtMac(perFilter, null, 6), null)
  assert.equal(perFilterAtMac(perFilter, null, 10).length, perFilter.length)
})
