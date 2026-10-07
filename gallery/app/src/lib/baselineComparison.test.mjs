import assert from 'node:assert/strict'
import fs from 'node:fs'
import { baselineMetrics, pairedAbsoluteErrorDifference, retainedWavenumber } from './baselineComparison.ts'
import * as d3 from 'd3'
const data = JSON.parse(fs.readFileSync(new URL('../../public/data/baseline_comparison.json', import.meta.url), 'utf8'))
for (const cohort of ['full_pool', 'locked800']) {
  const rows = data.pairs.filter(r => r.cohort === cohort)
  const group = data.groups.find(g => g.cohort === cohort && g.dimension === 'overall')
  assert.equal(rows.length, group.n)
  const deltas = rows.map(pairedAbsoluteErrorDifference)
  assert.ok(Math.abs(d3.mean(deltas) - (group.VIBES_MAE - group.AIRSpec_MAE)) < 1e-10, 'Mean paired difference equals saved ΔMAE')
  for (const n of [10,20,40,80]) {
    const bound = d3.max(deltas, Math.abs) || 1
    const bins = d3.bin().domain([-bound,bound]).thresholds(d3.range(1,n).map(i=>(i-n/2)*2*bound/n))(deltas)
    assert.equal(d3.sum(bins,b=>b.length), rows.length, 'Every filter including both extreme tails is binned')
    assert.ok(bins.every(b=>!(b.x0<0 && b.x1>0)), 'No bin mixes improvement and worsening')
  }
  for (const method of ['AIRSpec', 'VIBES']) {
    const m = baselineMetrics(rows, method)
    for (const [computed, saved] of [['rmse','RMSE'],['mae','MAE'],['bias','bias'],['r2','predictive_R2']]) {
      assert.ok(Math.abs(m[computed] - group[`${method}_${saved}`]) < 1e-10, `${cohort} ${method} ${computed}`)
    }
  }
}
assert.equal(baselineMetrics([], 'AIRSpec'), null)
assert.equal(baselineMetrics([{y:2,AIRSpec:-2}], 'AIRSpec').rmse, 4, 'Negative predictions contribute full error')
assert.equal(baselineMetrics([{y:2,AIRSpec:2}], 'AIRSpec').r2, null, 'Constant truth has undefined predictive R²')
const full = data.historical_masks.find(m=>m.mask==='full')
const cut = data.historical_masks.find(m=>m.mask==='no_co2_max3600')
assert.equal(full.n_channels, 2002)
assert.equal(cut.n_channels, 1148)
assert.equal(cut.changed_vs_full, 61)
for (const mask of data.historical_masks) {
  assert.equal(data.wn.filter(wn=>retainedWavenumber(wn,mask)).length, mask.n_channels, mask.mask)
}
assert.equal(retainedWavenumber(1800,cut),false)
assert.equal(retainedWavenumber(2500,cut),false)
assert.equal(retainedWavenumber(3600,cut),true)
assert.equal(retainedWavenumber(3600.01,cut),false)
assert.equal(pairedAbsoluteErrorDifference({y:2,AIRSpec:-2,VIBES:-3}),1)
assert.equal(data.contributions.length,data.pairs.length)
const pairsByKey=new Map(data.pairs.map(r=>[`${r.cohort}|${r.sample_id}`,r]))
for(const r of data.contributions){
  const pair=pairsByKey.get(`${r.cohort}|${r.sample_id}`)
  assert.ok(pair,'Every contribution identifies a paired test filter')
  const terms=['1425_1799','1800_2499','2500_2999','3000_4000'].flatMap(b=>[r[`spectra_${b}`],r[`coefficients_${b}`]])
  assert.ok(Math.abs(terms.reduce((a,b)=>a+b,0)+r.intercept_and_centering_delta+r.centering_precision_delta-(pair.VIBES-pair.AIRSpec))<1e-8)
}
assert.equal(new Set(data.addis_predictions.filter(r=>r.cohort==='full_pool'&&r.method==='AIRSpec').map(r=>r.sample_id)).size,253)
const app=JSON.parse(fs.readFileSync(new URL('../../public/data/baseline_applicability.json',import.meta.url),'utf8'))
assert.equal(app.rows.length,2*(2327+253))
for(const method of ['AIRSpec','VIBES']){
  const test=app.rows.filter(r=>r.kind==='test'&&r.method===method)
  const addis=app.rows.filter(r=>r.kind==='addis'&&r.method===method)
  assert.deepEqual(new Set(test.map(r=>r.sample_id)),new Set(data.pairs.filter(r=>r.cohort==='full_pool').map(r=>r.sample_id)))
  assert.equal(addis.length,253)
  assert.ok(addis.every(r=>r.abs_error===null),'Addis has no independent TOR error')
  assert.ok([...test,...addis].every(r=>Number.isFinite(r.t2)&&Number.isFinite(r.q_rms)&&r.t2_train_percentile>=0&&r.t2_train_percentile<=100&&r.q_train_percentile>=0&&r.q_train_percentile<=100))
}
const followup=JSON.parse(fs.readFileSync(new URL('../../public/data/baseline_followup.json',import.meta.url),'utf8'))
assert.equal(followup.scope,'new_exploratory_followup_on_saved_corrections')
assert.equal(followup.common_test_n,2327)
assert.equal(followup.selection_metrics.length,12)
assert.equal(followup.routing_metrics.length,6)
assert.ok(followup.selection_metrics.every(r=>r.n===followup.common_test_n))
const frozenFull=data.groups.find(g=>g.cohort==='full_pool'&&g.dimension==='overall')
for(const method of ['AIRSpec','VIBES']){
  const row=followup.selection_metrics.find(r=>r.model==='full_pool'&&r.method===method)
  assert.ok(row)
  assert.ok(Math.abs(row.RMSE-frozenFull[`${method}_RMSE`])<1e-10)
  const restricted=followup.restricted_sensitivity.filter(r=>r.method===method)
  assert.equal(restricted.length,2)
  assert.equal(restricted.reduce((n,r)=>n+r.n,0),followup.common_test_n)
}
assert.equal(followup.addis_readiness.n_independent_thermal_matches,0)
console.log('Frozen comparison and notebook follow-up reconcile; Addis remains explicitly unlabelled.')
