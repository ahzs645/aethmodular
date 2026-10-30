// ftir_46's mass lead (massLead.ts) against the notebook's displayed tables, on the notebook's own 239 Addis filters:
// the explorer's locked 800 + AIRSpec predictions (calibration_runs.json) with the tracked ETAD mass and volume table
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { coefBoot, framed, lstsq, massPerArea, partialSlopes, siteFit } from './massLead.ts'

const runs = JSON.parse(fs.readFileSync(new URL('../../../../public/data/calibration_runs.json', import.meta.url), 'utf8'))
const csv = fs.readFileSync(new URL('../../../../../../research/ftir_ec_calibration_2026_06_25/tables/ethiopia_new_ec_by_component.csv', import.meta.url), 'utf8').trim().split('\n')
const head = csv[0].split(',')
const massOf = new Map()
for (const line of csv.slice(1)) {
  const r = Object.fromEntries(line.split(',').map((v, i) => [head[i], v]))
  const id = r.ExternalFilterId.split('-').slice(0, 2).join('-'), mass = Number(r.MassCollectedOnFilter_ug), volume = Number(r.SampleVolume_m3)
  if (r.MassCollectedOnFilter_ug && volume > 0 && !massOf.has(id)) massOf.set(id, { mass, volume })
}
const run = runs.runs.find((r) => r.preset === 'ocec800_airspec' && r.target === 'addis')
const addis = run.eval.id.flatMap((id, i) => {
  const t = massOf.get(id)
  return t ? [{ id, group: run.eval.group[i], month: Number(run.eval.date[i].slice(5, 7)), fabs: run.eval.ref[i], fx: run.eval.pred[i],
    mass: t.mass / t.volume, volume: t.volume, area: 3.53, lot: null }] : []
})
const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`)

test('the notebook\'s 239 Addis filters are all here', () => assert.equal(addis.length, 239))

test('ftir_46 §7 Addis, locked 800 + AIRSpec: Fabs ~ f(X) + mass and the residual\'s slopes, at MAC 10', () => {
  const v = siteFit(addis, 'volume', 10, null, { draws: 400 })
  near(v.massCoef.est, 0.6767, 5e-4, 'mass coefficient (Mm⁻¹ per µg/m³)')
  near(v.rPartial.est, -0.0677, 5e-4, 'residual partial slope')
  near(v.corrMassFx, 0.8386, 5e-4, 'corr(mass, f(X))')
  near(v.rSlope, -0.0006, 5e-4, 'raw residual slope')
  near(v.massCoef.lo, 0.4575, 0.06, 'bootstrap low end'); near(v.massCoef.hi, 0.9569, 0.06, 'bootstrap high end')
  // the Addis deposit-loading range is Addis's own (16.3–107.1 µg/cm²), so every filter sits in it
  const loads = addis.map(massPerArea)
  near(Math.min(...loads), 16.3, 0.05, 'range low'); near(Math.max(...loads), 107.1, 0.05, 'range high')
  const d = siteFit(addis, 'deposit', 10, [Math.min(...loads), Math.max(...loads)], { draws: 400 })
  near(d.massCoef.est, 0.0640, 5e-4, 'per-deposit coefficient (µg EC-eq per µg)')
  near(d.rPartial.est, -0.0640, 5e-4, 'per-deposit residual partial slope')
  assert.equal(d.nInRange, 239)
})

test('ftir_46 §5 partial slopes at MAC 10 (f(X) + season, and raw), both framings', () => {
  const v = partialSlopes(addis, 'volume', 10, { draws: 400 }), d = partialSlopes(addis, 'deposit', 10, { draws: 400 })
  near(v[0].slope.est, -0.0683, 5e-4, 'per volume | f(X), season'); near(v[0].slopeXIqr, -0.7652, 5e-3, '× IQR')
  near(v[2].slope.est, -0.0006, 5e-4, 'per volume raw')
  near(d[0].slope.est, -0.0640, 5e-4, 'per deposit | f(X), season'); near(d[0].slopeXIqr, -1.4015, 1e-2, '× IQR')
  near(d[2].slope.est, 0.0116, 5e-4, 'per deposit raw')
  near(v[0].slope.lo, -0.0974, 0.01, 'interval low'); near(v[0].slope.hi, -0.0448, 0.01, 'interval high')
})

test('the MAC: the per-volume coefficient does not move, the partial slope scales by 10/MAC, the raw slope does not', () => {
  const at10 = siteFit(addis, 'volume', 10, null, { draws: 50 }), at6 = siteFit(addis, 'volume', 6, null, { draws: 50 })
  near(at6.massCoef.est, at10.massCoef.est, 1e-12, 'mass coefficient')
  near(at6.rPartial.est, at10.rPartial.est * 10 / 6, 1e-12, 'partial slope')
  // the shortcut is the regression: r = f(X) − Fabs/6 on [f(X), mass] directly
  const F = framed(addis, 'volume', 6)
  near(lstsq([F.fx, F.m], F.r)[1], at6.rPartial.est, 1e-10, 'direct residual regression at MAC 6')
  assert.ok(Math.abs(at6.rSlope - at10.rSlope * 10 / 6) > 1e-3, 'the raw slope is not a rescaling')
  const d10 = siteFit(addis, 'deposit', 10, null, { draws: 50 }), d6 = siteFit(addis, 'deposit', 6, null, { draws: 50 })
  near(d6.massCoef.est, d10.massCoef.est * 10 / 6, 1e-12, 'per-deposit coefficient')
})

test('lstsq: a dummy column no resampled filter carries is skipped, not singular', () => {
  const x = [1, 2, 3, 4, 5, 6], lot = [0, 0, 0, 0, 0, 1], y = x.map((v, i) => 2 * v + 3 * lot[i] + 1)
  const all = lstsq([x, lot], y)
  near(all[0], 2, 1e-9, 'slope'); near(all[1], 3, 1e-9, 'dummy')
  const some = lstsq([x, lot], y, [0, 1, 2, 3, 4, 2])
  near(some[0], 2, 1e-9, 'slope without the dummy'); assert.ok(Number.isNaN(some[1]))
  assert.equal(coefBoot([x, lot], y, 0, { draws: 20 }).est.toFixed(6), '2.000000')
})
