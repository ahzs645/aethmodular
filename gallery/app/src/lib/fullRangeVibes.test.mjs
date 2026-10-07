import assert from 'node:assert/strict'
import { test } from 'node:test'
import fs from 'node:fs'
import { csvParse } from 'd3'

const read = path => JSON.parse(fs.readFileSync(new URL(path, import.meta.url), 'utf8'))
const data = read('../../public/data/vibes_fullrange_results.json')
const meta = read('../../public/data/similarity/traces_VIBES-full-r70.json')
const preliminary = read('../../public/data/similarity/traces_VIBES-full.json')

test('completed trace export distinguishes recovered rank 70 from frozen rank 30', () => {
  assert.equal(meta.presentation_status, 'completed_uncapped')
  assert.equal(meta.background_rank, 70)
  assert.equal(meta.rank_cap, 85)
  assert.equal(meta.n_recovered, 2)
  assert.equal(meta.n_corrected, 12808)
  assert.equal(meta.n_failed, 0)
  assert.equal(meta.n, 12646)
  assert.equal(meta.case_counts.calibration + meta.case_counts.target, meta.n)
  assert.equal(preliminary.presentation_status, 'preliminary_capped')
  assert.equal(preliminary.rank_cap, 30)
  const size = fs.statSync(new URL('../../public/data/similarity/traces_VIBES-full-r70.bin', import.meta.url)).size
  assert.equal(size, meta.n * (4 + 2 * meta.bin_wn.length))
})

test('gallery source-domain metrics reproduce every saved paired prediction cohort', () => {
  const predictions = csvParse(fs.readFileSync(new URL('../../../../research/ftir_hips_chem/output/tables/vibes_fullrange/interpretation_20261002/benchmark/heldout_predictions.csv', import.meta.url), 'utf8'))
  for (const score of data.scores) {
    const rows = predictions.filter(p => p.cohort === score.cohort && p.method === score.method)
    assert.equal(rows.length, score.n_test)
    assert.equal(new Set(rows.map(p => p.sample_id)).size, score.n_test)
    const rmse = Math.sqrt(rows.reduce((s, p) => s + (+p.prediction - +p.y) ** 2, 0) / rows.length)
    assert.ok(Math.abs(rmse - score.RMSE) < 1e-10)
    const interval = data.intervals.find(r => r.cohort === score.cohort && r.method === score.method)
    if (score.method !== 'AIRSpec') {
      const reference = data.scores.find(r => r.cohort === score.cohort && r.method === 'AIRSpec')
      assert.ok(Math.abs(interval.delta_rmse - (score.RMSE - reference.RMSE)) < 1e-10)
    }
  }
})

test('aggregate spectra retain each method grid and all paired physical filters', () => {
  for (const population of ['calibration', 'target']) {
    for (const [method, spectrum] of Object.entries(data.spectra[population])) {
      assert.equal(spectrum.n, population === 'calibration' ? 12393 : 253)
      assert.equal(spectrum.q.length, 3)
      assert.ok(spectrum.q.every(q => q.length === spectrum.wn.length && q.every(Number.isFinite)))
      assert.ok(spectrum.wn.every((w, i) => i === 0 || w < spectrum.wn[i - 1]))
      assert.ok(spectrum.wn.at(-1) > (method === 'AIRSpec' ? 1425 : 500))
      assert.ok(spectrum.wn.at(-1) < (method === 'AIRSpec' ? 1440 : 515))
    }
  }
})
