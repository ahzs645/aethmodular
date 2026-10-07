import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'
import { csvParse } from 'd3'
import { baselineMetrics } from './baselineComparison.ts'
const file = name => new URL(`../../public/data/${name}`, import.meta.url)
const read = name => JSON.parse(fs.readFileSync(file(name), 'utf8'))
const migration = new URL('../../../../research/ftir_hips_chem/output/tables/vibes_fullrange/gallery_rank70_20261002/', import.meta.url)
const rows = name => csvParse(fs.readFileSync(new URL(name, migration), 'utf8'))
const close = (a,b) => assert.ok(Math.abs(a-b) < 1e-8, `${a} != ${b}`)

test('current seasonal and specification exports cover every completed fit and both windows', () => {
  const d=read('meeting-r70/followup_20260917.json')
  assert.equal(d.summary.background_rank,70)
  assert.deepEqual(d.summary.methods,['AIRSpec','VIBES-full-r70','VIBES-full-cut-r70'])
  assert.equal(d.grid_n,9010);assert.equal(d.repeats.method.length,1200)
  const grid=fs.readFileSync(new URL('meeting/category_grid.jsonl',migration),'utf8').trim().split('\n').map(JSON.parse)
  assert.equal(new Set(grid.map(r=>r.key)).size,9010)
  assert.equal(grid.filter(r=>r.method==='AIRSpec').length,3138)
  for(const m of ['VIBES-full-r70','VIBES-full-cut-r70']) assert.equal(grid.filter(r=>r.method===m).length,2936)
  assert.equal(read('meeting-r70/spec_curve_20260917.json').n,9010)
  assert.equal(d.spectra['wn|VIBES-full-r70'].length,908)
  assert.ok(Math.min(...d.spectra['wn|VIBES-full-r70']) <= 504)
  assert.equal(d.spectra['wn|VIBES-full-cut-r70'].length,668)
  assert.ok(Math.min(...d.spectra['wn|VIBES-full-cut-r70']) >= 1425)
})

test('current paired plots reproduce benchmark scores and every closed decomposition', () => {
  const d=read('baseline_paired_r70.json')
  assert.equal(d.run.background_rank,70);assert.equal(d.pairs.length,2464)
  const original=csvParse(fs.readFileSync(new URL('../../../../research/ftir_hips_chem/output/tables/vibes_fullrange/interpretation_20261002/benchmark/heldout_predictions.csv',import.meta.url),'utf8'))
  for(const cohort of ['full_pool','locked800']) {
    const pairs=d.pairs.filter(r=>r.cohort===cohort)
    assert.equal(new Set(pairs.map(r=>r.sample_id)).size,cohort==='full_pool'?2327:137)
    for(const method of ['AIRSpec','VIBES']) {
      const predictions=new Map(original.filter(r=>r.cohort===cohort&&r.method===method).map(r=>[r.sample_id,+r.prediction]))
      assert.equal(predictions.size,pairs.length)
      for(const r of pairs) close(r[method],predictions.get(r.sample_id))
      const score=baselineMetrics(pairs,method),group=d.groups.find(r=>r.cohort===cohort&&r.dimension==='overall')
      close(score.rmse,group[`${method}_RMSE`]);close(score.mae,group[`${method}_MAE`])
    }
  }
  const pairs=new Map(d.pairs.map(r=>[r.cohort+r.sample_id,r]))
  for(const r of d.contributions) {
    const pair=pairs.get(r.cohort+r.sample_id)
    close(r.prediction_delta,pair.VIBES-pair.AIRSpec)
    const total=Object.entries(r).filter(([k])=>k.startsWith('spectra_')||k.startsWith('coefficients_')).reduce((s,[,v])=>s+v,0)
    close(total+r.intercept_and_centering_delta+r.centering_precision_delta,r.prediction_delta)
  }
})

test('selection, routing and PCA use the completed source and unchanged paired populations', () => {
  const f=read('baseline_followup_r70.json'),a=read('baseline_applicability_r70.json')
  assert.equal(f.background_rank,70);assert.equal(a.background_rank,70);assert.equal(a.rows.length,5160)
  for(const method of ['AIRSpec','VIBES']) {
    assert.equal(a.rows.filter(r=>r.method===method&&r.kind==='test').length,2327)
    assert.equal(a.rows.filter(r=>r.method===method&&r.kind==='addis').length,253)
  }
  const predictions=rows('followup/common_test_predictions.csv')
  for(const metric of f.selection_metrics) {
    const part=predictions.filter(r=>r.method===metric.method&&r.model===metric.model)
    assert.equal(new Set(part.map(r=>r.sample_id)).size,2327)
    close(Math.sqrt(part.reduce((s,r)=>s+(+r.prediction- +r.y)**2,0)/part.length),metric.RMSE)
  }
  for(const name of ['routing_nested_training_predictions.csv','routing_prior_test_predictions.csv'])
    for(const r of rows(`followup/${name}`)) close(+r.routed,+r[r.choice])
  assert.ok(Object.keys(f.source_sha256).some(k=>k.endsWith('corrected_VIBES_full_recovered.npy')))
})

test('similarity binaries have independent full and shared channel grids', () => {
  const m=read('similarity/similarity.json');assert.equal(m.background_rank,70)
  for(const [method,n,channels] of [['VIBES-full-r70',341,2722],['VIBES-full-cut-r70',251,2002]]) {
    const grid=m.method_grids[method]
    assert.equal(grid.bin_wn.length,n);assert.equal(grid.masks.find(r=>r.key==='full').n_channels,channels)
    assert.equal(fs.statSync(file(`similarity/traces_${method}.bin`)).size,m.filters.sample_id.length*(n*2+4))
    assert.equal(fs.statSync(file(`similarity/pairs_${method}_full.bin`)).size,m.n_pairs*2)
  }
})

test('all 31 external corrections converged at rank 70 and the five physical-filter matches remain', () => {
  const p=read('rank70/baseline_external_pilot.json')
  assert.equal(p.background_rank,70);assert.equal(p.adama.length,5);assert.equal(p.bishoftu.length,26);assert.equal(p.matched.length,5)
  const diag=rows('external/external_fit_diagnostics.csv')
  assert.equal(new Set(diag.map(r=>r.sample_id)).size,31)
  for(const r of diag) {assert.equal(r.components,'70');assert.equal(r.success,'True')}
  const archived=read('baseline_external_pilot.json')
  assert.deepEqual(p.matched.map(r=>[r.addis_filter_id,r.bishoftu_filter_id]),archived.matched.map(r=>[r.addis_filter_id,r.bishoftu_filter_id]))
})
