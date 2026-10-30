// Research → Python PMF (pythonPmf.ts) against the package's saved results (research/python_pmf/python_pmf.json, from
// gallery/data/export_python_pmf.py): the held-out total-OM scores, the archived-KF pattern r through the gallery's
// pmf.json, the supporting HIPS Fabs r through the gallery's filter rows, and the negative-cell RMSE bound.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { archivedMatch, baseId, disagrees, gapAfter, negativeRmseBound, pearson, qcCell, scores } from './pythonPmf.ts'

const read = (rel) => JSON.parse(fs.readFileSync(new URL(`../../../../public/data/${rel}`, import.meta.url), 'utf8'))
const data = read('research/python_pmf/python_pmf.json')
const pmf = read('pmf.json')
const filters = read('filters.json')
const cols = (t) => Object.keys(t).length ? t[Object.keys(t)[0]].map((_, i) => Object.fromEntries(Object.keys(t).map((k) => [k, t[k][i]]))) : []
const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`)

test('the held-out total-OM scores on all 158 filters are the report\'s 0.516 / 0.367 / 0.954', () => {
  const h = cols(data.tables.heldout)
  assert.equal(h.length, 158)
  const s = scores(h.map((r) => r.observed_OM_ug_m3), h.map((r) => r.predicted_OM_ug_m3))
  near(s.rmse, data.heldout_pooled.RMSE_ug_m3, 5e-5, 'RMSE'); near(s.mae, data.heldout_pooled.MAE_ug_m3, 5e-5, 'MAE')
  near(s.r2, data.heldout_pooled.R_squared, 5e-5, 'R²'); near(s.r, data.heldout_pooled.Pearson_r, 5e-5, 'r')
  assert.equal(s.rmse.toFixed(3), '0.516'); assert.equal(s.mae.toFixed(3), '0.367'); assert.equal(s.r2.toFixed(3), '0.954')
  // ETAD-0037-5 carries 19.60 % of the pooled squared error (report §8)
  const bad = h.find((r) => r.sample_id === 'ETAD-0037-5')
  near((bad.predicted_OM_ug_m3 - bad.observed_OM_ug_m3) ** 2 / s.sse, 0.196, 5e-4, 'ETAD-0037-5 share of SSE')
  // the training-mean baseline
  const b = scores(h.map((r) => r.observed_OM_ug_m3), h.map((r) => r.training_mean_baseline_ug_m3))
  near(b.rmse, 2.544, 5e-4, 'baseline RMSE'); near(b.r2, -0.126, 5e-4, 'baseline R²')
})

test('each held-out fold matches fold_metrics.csv, and the conditional (fixed-profile) RMSE is 0.497', () => {
  const h = cols(data.tables.heldout)
  for (const f of cols(data.tables.heldout_folds)) {
    const rs = h.filter((r) => r.fold === f.fold)
    const s = scores(rs.map((r) => r.observed_OM_ug_m3), rs.map((r) => r.predicted_OM_ug_m3))
    assert.equal(s.n, f.n)
    near(s.rmse, f.RMSE_ug_m3, 5e-4, `fold ${f.fold} RMSE`); near(s.r2, f.R_squared, 5e-4, `fold ${f.fold} R²`)
  }
  const c = cols(data.tables.conditional)
  const s = scores(c.map((r) => r.OM_ug_m3), c.map((r) => r.conditional_heldout_OM_prediction_ug_m3))
  near(s.rmse, data.conditional.heldout.rmse_ug_m3, 5e-5, 'conditional RMSE'); assert.equal(s.rmse.toFixed(3), '0.497')
})

test('every new factor against its archived KF factor, through the gallery\'s pmf.json, gives the saved pattern r', () => {
  const hist = cols(data.tables.historical)
  assert.equal(hist.length, 55)
  const label = (kf) => pmf.sources[kf - 1].label
  // the archived factor numbers are pmf.json's source order (comparison_manifest archived_factor_labels)
  for (const [k, l] of Object.entries(data.kf_labels)) assert.equal(label(+k).toLowerCase(), l.toLowerCase())
  for (const c of [...new Set(hist.map((r) => r.case))]) {
    const pairs = hist.filter((r) => r.case === c).map((r) => ({ factor: r.new_factor, kf: r.historical_factor }))
    const live = archivedMatch(data.contributions[c], pmf.rows, pairs, label)
    for (const m of live) {
      const saved = hist.find((r) => r.case === c && r.new_factor === m.factor)
      assert.equal(m.n, saved.n_common_physical_filters, `${c} factor ${m.factor} n`)
      near(m.r, saved.contribution_pearson_r, 1e-4, `${c} factor ${m.factor} r`)
    }
  }
  // a filter left out moves it
  const pairs = [{ factor: 1, kf: 5 }]
  const all = archivedMatch(data.contributions.etad246_huber, pmf.rows, pairs, label)[0]
  const less = archivedMatch(data.contributions.etad246_huber, pmf.rows, pairs, label, (id) => id !== 'ETAD-0035')[0]
  assert.equal(less.n, all.n - 1); assert.notEqual(less.r, all.r)
})

test('factor contributions against the gallery\'s HIPS Fabs on the package\'s 102 filters give the saved r', () => {
  const sup = cols(data.tables.supporting).filter((r) => r.reference_variable === 'hips_Fabs_Mm1')
  const rows = new Map(filters.rows.filter((r) => r.code === 'ETAD').map((r) => [r.id, r]))
  const keep = new Set(data.comparison_filters.map(baseId))
  for (const c of ['etad246_huber', 'etad102_huber']) {
    const k = data.contributions[c]
    for (let f = 1; f <= 5; f++) {
      const x = [], y = []
      k.id.forEach((id, i) => { const r = rows.get(baseId(id)); if (r && keep.has(baseId(id)) && r['HIPS Fabs'] != null) { x.push(k.g[f - 1][i]); y.push(r['HIPS Fabs']) } })
      assert.equal(x.length, 102)
      const saved = sup.find((r) => r.case === c && r.factor === f).pearson_r
      near(pearson(x, y), saved, 2e-4, `${c} factor ${f} vs Fabs`)
      // Fabs/MAC: r does not move with the MAC
      near(pearson(x, y.map((v) => v / 6)), saved, 2e-4, `${c} factor ${f} vs Fabs/6`)
    }
  }
})

test('the negative-cell lower bound is the report\'s 3.970 / 3.848 / 3.139', () => {
  const want = { etad246_huber: 3.970, etad102_huber: 3.848, etbi53_huber: 3.139 }
  for (const c of data.cases.filter((c) => c.key in want)) near(negativeRmseBound(c.negative_q_lower_bound, c.cells), want[c.key], 6e-4, c.key)
})

test('gaps, ids and the QC disagreement rule', () => {
  assert.deepEqual([...gapAfter(['2023-01-01', '2023-01-04', '2023-02-01', '2023-02-04'], 14)], [1])
  assert.equal(baseId('ETAD-0037-5'), 'ETAD-0037'); assert.equal(baseId('ETAD-0037'), 'ETAD-0037')
  assert.equal(disagrees(qcCell(false, true, true)), true) // ETAD-0158: held by the PMF, counted by the gallery
  assert.equal(disagrees(qcCell(false, true, false)), false) // ETAD-0248: out of both
  assert.equal(disagrees(qcCell(true, false, true)), false) // a modelled filter after the gallery's series ends
  assert.equal(disagrees(qcCell(true, true, false)), true) // a modelled filter the reader excluded
})
