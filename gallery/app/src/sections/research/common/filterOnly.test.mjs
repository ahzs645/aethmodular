// The filter-only temporal-transfer refits (filterOnly.ts) against the release's own tables
// (deliverables/filter_only_scientific_release_2026-09-11, exported by gallery/data/export_research_filter_only.py)
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { POPULATIONS, denominatorSensitivity, evaluate, fitTraining, heldOutPrediction, id11Sensitivity, inPopulation, pairedInfluence, summarize } from './filterOnly.ts'

const data = (f) => JSON.parse(fs.readFileSync(new URL(`../../../../public/data/research/filter_only/${f}`, import.meta.url), 'utf8'))
const rowsOf = (t) => t[Object.keys(t)[0]].map((_, i) => Object.fromEntries(Object.keys(t).map((k) => [k, t[k][i]])))
const { points } = data('points.json')
const release = data('release.json')
const frozen = Object.fromEntries(Object.entries(release).filter(([k]) => !['generated', 'notebooks', 'sources'].includes(k)).map(([k, t]) => [k, rowsOf(t)]))
const SITES = ['Addis Ababa', 'Beijing', 'Delhi', 'JPL']
const OMIT = 'ETAD-0037'
// the release rounds its tables to 7 significant digits on export
const near = (a, b, what, tol = 2e-6) => {
  if (b === null || b === undefined) return assert.ok(a === null || !Number.isFinite(a), `${what}: ${a} vs missing`)
  assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${what}: ${a} vs ${b}`)
}
const cohortOf = (site, pop, variant) => points.filter((p) => p.site === site && inPopulation(p, pop) && !(variant === 'without_ETAD_0037' && p.id === OMIT))
const siteDiag = (site) => points.filter((p) => p.site === site)

test('every block of both schemes, all populations and the ETAD-0037 omission: splits, fits and errors', () => {
  let checked = 0
  for (const site of SITES) for (const { key: pop } of POPULATIONS) for (const variant of site === 'Addis Ababa' ? ['baseline', 'without_ETAD_0037'] : ['baseline']) {
    for (const scheme of ['leave_quarter_out', 'later_period']) {
      const mine = evaluate(siteDiag(site), cohortOf(site, pop, variant), scheme)
      const saved = frozen.blocks.filter((r) => r.site === site && r.population === pop && r.variant === variant && r.scheme === scheme)
      assert.deepEqual(mine.map((b) => b.block), saved.map((r) => r.block), `${site} ${pop} ${scheme} quarters`)
      mine.forEach((b, i) => {
        const r = saved[i], what = `${site} ${pop} ${variant} ${scheme} ${b.block}`
        assert.equal(b.status, r.status, `${what} status`)
        assert.equal(b.trainN, r.train_n, `${what} train n`); assert.equal(b.testN, r.test_n, `${what} test n`)
        near(b.fit.slope, r.ols_slope, `${what} slope`); near(b.fit.intercept, r.ols_intercept, `${what} intercept`); near(b.fit.median, r.training_median, `${what} median`)
        near(b.mae.ols, r.ols_mae, `${what} OLS MAE`); near(b.mae.median, r.median_mae, `${what} median MAE`); near(b.mse.ols, r.ols_mean_signed_error, `${what} OLS MSE`)
        if (b.outsideN !== null) assert.equal(b.outsideN, r.outside_training_ec_n, `${what} outside`)
        // the release leaves k unset on an empty quarter (its status is not 'evaluated')
        if (variant === 'baseline' && b.status === 'evaluated') { near(b.fit.k, r.proportional_k, `${what} k`); near(b.mae.proportional, r.proportional_mae, `${what} proportional MAE`) }
        checked++
      })
    }
  }
  assert.equal(checked, frozen.blocks.length)
})

test('the summaries: equal filters and equal quarters, improvement over the median', () => {
  for (const r of frozen.proportionality) {
    const s = summarize(evaluate(siteDiag(r.site), cohortOf(r.site, r.population, 'baseline'), r.scheme))
    const m = r.weighting === 'equal_filter' ? s.filters : s.quarters, what = `${r.site} ${r.population} ${r.scheme} ${r.weighting}`
    assert.equal(s.filters.n, r.evaluated_filters, `${what} n`); assert.equal(s.evaluatedBlocks, r.evaluated_blocks, `${what} blocks`)
    assert.equal(s.interceptBetterBlocks, r.intercept_better_blocks, `${what} intercept wins`)
    for (const k of ['median', 'proportional', 'ols']) near(m.mae[k], r[`${k}_mae`], `${what} ${k} MAE`)
    near(m.mse.ols, r.ols_mean_signed_error, `${what} OLS bias`); near(m.mse.proportional, r.proportional_mean_signed_error, `${what} prop bias`)
  }
  for (const r of frozen.performance) {
    const s = summarize(evaluate(siteDiag(r.site), cohortOf(r.site, r.population, r.variant), r.scheme))
    const what = `${r.site} ${r.population} ${r.variant} ${r.scheme}`
    near(s.improvementPct, r.mae_improvement_pct, `${what} improvement %`); assert.equal(s.improvedBlocks, r.improved_blocks, `${what} improved blocks`)
  }
})

test('full-cohort coefficients and the ETAD-0037 paired influence', () => {
  for (const r of frozen.coefficients) {
    const f = fitTraining(cohortOf(r.site, r.population, r.variant))
    near(f.slope, r.slope, `${r.site} ${r.population} ${r.variant} slope`); near(f.intercept, r.intercept, `${r.site} ${r.population} ${r.variant} intercept`)
  }
  for (const pop of POPULATIONS.map((p) => p.key)) for (const scheme of ['leave_quarter_out', 'later_period']) {
    const site = 'Addis Ababa'
    const mine = pairedInfluence(evaluate(siteDiag(site), cohortOf(site, pop, 'baseline'), scheme), evaluate(siteDiag(site), cohortOf(site, pop, 'without_ETAD_0037'), scheme))
    const saved = frozen.influence.filter((r) => r.population === pop && r.scheme === scheme)
    assert.equal(mine.length, saved.length, `${pop} ${scheme} rows`)
    mine.forEach((m, i) => {
      assert.equal(m.n, saved[i].common_n, `${pop} ${scheme} ${m.block} n`)
      near(m.change, saved[i].mae_change_without_minus_baseline, `${pop} ${scheme} ${m.block} change`, 1e-5)
    })
  }
})

test('the ID-11 training choice on common later ID-11 filters', () => {
  for (const { key: pop } of POPULATIONS) {
    const cohort = cohortOf('Addis Ababa', pop, 'baseline')
    const rows = id11Sensitivity(evaluate(siteDiag('Addis Ababa'), cohort, 'later_period'), cohort)
    for (const r of rows) for (const choice of ['all_earlier', 'id11_earlier']) {
      const s = frozen.id11.find((x) => x.population === pop && x.block === r.block && x.training_choice === choice), what = `${pop} ${r.block} ${choice}`
      assert.equal(r.status, s.status, `${what} status`); assert.equal(r.testN, s.test_n, `${what} test n`)
      assert.equal(r.allEarlierN, s.all_earlier_train_n); assert.equal(r.id11EarlierN, s.id11_earlier_train_n)
      const m = choice === 'all_earlier' ? r.all : r.id11
      near(m.mae.ols, s.ols_mae, `${what} OLS MAE`); near(m.mse.ols, s.ols_mean_signed_error, `${what} OLS bias`)
    }
  }
})

test('filter_only_diagnostics 05: counts and ratio medians at each MDL multiple', () => {
  for (const r of frozen.denominator) {
    const d = denominatorSensitivity(siteDiag(r.site)).find((x) => x.multiple === r.minimum_ec_mdl_multiple)
    assert.equal(d.n, r.n, `${r.site} ${r.minimum_ec_mdl_multiple}× n`)
    near(d.median, r.ratio_median, `${r.site} median`); near(d.q25, r.ratio_q25, `${r.site} q25`); near(d.q75, r.ratio_q75, `${r.site} q75`)
  }
  for (const r of frozen.cohorts) assert.equal(cohortOf(r.site, r.population, r.variant).length, r.n, `${r.site} ${r.population} ${r.variant} n`)
})

test('Mark: a hand-excluded filter is drawn at the prediction it would get in the cohort, and moves no fit', () => {
  const diag = siteDiag('Addis Ababa')
  let checked = 0
  for (const out of diag.filter((_, i) => i % 23 === 0)) {
    const without = diag.filter((p) => p !== out)
    // leave-quarter-out never trains on the filter's own quarter, so leaving it out does not change its prediction
    const withIt = evaluate(without, diag, 'leave_quarter_out').flatMap((b) => b.preds).find((q) => q.p === out)
    const blocks = evaluate(without, without, 'leave_quarter_out')
    const pred = heldOutPrediction(blocks, out)
    if (withIt && Number.isFinite(withIt.pred.ols)) { near(pred, withIt.pred.ols, `${out.id} prediction`, 1e-12); checked++ }
    // and the metrics are those of the cohort without it
    assert.ok(blocks.every((b) => b.preds.every((q) => q.p !== out)), `${out.id} in no block`)
  }
  assert.ok(checked > 3, `checked ${checked}`)
  assert.equal(heldOutPrediction([], { date: '2023-01-01', ec: 1 }), null)
})
