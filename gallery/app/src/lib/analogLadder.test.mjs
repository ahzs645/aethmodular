import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import zlib from 'node:zlib'
import { ladderPath, ladderView } from './analogLadder.ts'

const DIR = new URL('../../public/data/meeting-r70/', import.meta.url)
const json = (name) => JSON.parse(fs.readFileSync(new URL(name, DIR), 'utf8'))
const ready = fs.existsSync(new URL('analog_ladder/manifest.json', DIR))

const open = (manifest, rule, method, group) => {
  const stem = ladderPath(manifest, rule, method, group)
  const raw = zlib.gunzipSync(fs.readFileSync(new URL(`analog_ladder/${stem}.bin.gz`, DIR)))
  return [json(`analog_ladder/${stem}.json`), raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)]
}

test('a rule that does not depend on the season reads one file for every season', () => {
  const m = { rules: [{ key: 'lowest-ocec', per_group: false }], method_slug: { AIRSpec: 'spline' }, group_slug: { 'All Addis': 'all', 'Dry (Oct-Feb)': 'dry' } }
  assert.equal(ladderPath(m, 'lowest-ocec', 'AIRSpec', 'Dry (Oct-Feb)'), 'lowest-ocec.spline.all')
  assert.equal(ladderPath(m, 'nope', 'AIRSpec', 'Dry (Oct-Feb)'), null)
  assert.equal(ladderPath({ ...m, rules: [{ key: 'a', per_group: true }], alias: { 'a.spline.dry': 'b.spline.dry' } }, 'a', 'AIRSpec', 'Dry (Oct-Feb)'), 'b.spline.dry')
})

test('the top 500 by the 17 Sep rule is the saved calibration, filter by filter', { skip: !ready }, () => {
  const manifest = json('analog_ladder/manifest.json'), data = json('followup_20260917.json')
  for (const method of data.summary.methods) for (const group of data.groups) {
    const fitId = `seasonal|${method}|locked|${group}`
    const [head, bin] = open(manifest, 'similarity-sep17', method, group)
    const v = ladderView(head, bin, manifest, 500, { addis: data.addis, fitId, model: 'seasonal calibration', method, group })
    const saved = data.fits.find((f) => f.fit_id === fitId)
    assert.equal(v.fit.k, saved.k); assert.equal(v.fit.test_n, saved.test_n); assert.equal(v.fit.train_n, saved.train_n)

    const a = data.analogs, rows = a.filter_id.flatMap((_, i) => a.model[i] === 'seasonal calibration' && a.method[i] === method && a.group[i] === group && a.role[i] === 'test' ? [i] : [])
    const want = new Map(rows.map((i) => [a.filter_id[i], a.ftir_ec_ugm3[i]]))
    assert.equal(v.analogs.filter_id.length, want.size)
    v.analogs.filter_id.forEach((id, i) => assert.ok(Math.abs(v.analogs.ftir_ec_ugm3[i] - want.get(id)) < 2e-3, `${fitId} ${id}`))

    const ad = data.addis, wantA = ad.fit_id.flatMap((f, i) => f === fitId ? [ad.ftir_ec_ugm3[i]] : [])
    assert.equal(v.addis.ftir_ec_ugm3.length, wantA.length)
    v.addis.ftir_ec_ugm3.forEach((y, i) => assert.ok(Math.abs(y - wantA[i]) < 2e-3, `${fitId} Addis ${i}`))
  }
})

test('every size draws only its own analogs, and they are the top n by rank', { skip: !ready }, () => {
  const manifest = json('analog_ladder/manifest.json'), data = json('followup_20260917.json')
  const method = 'AIRSpec', group = 'Dry (Oct-Feb)'
  const [head, bin] = open(manifest, 'similarity-oct1', method, group)
  for (const n of [manifest.n_min, 87, 500, manifest.n_max]) {
    const v = ladderView(head, bin, manifest, n, { addis: data.addis, fitId: `seasonal|${method}|locked|${group}`, model: 'seasonal calibration', method, group })
    const top = new Set(head.analogs.filter_id.slice(0, n))
    if (v.fit.k == null) continue
    assert.equal(v.analogs.filter_id.length, v.fit.test_n)
    assert.ok(v.analogs.filter_id.every((id) => top.has(id)))
    assert.equal(v.fit.train_n + v.fit.test_n, n)
    assert.ok(v.addis.ftir_ec_ugm3.every((y) => Number.isFinite(y)))
  }
  assert.throws(() => ladderView(head, bin, manifest, manifest.n_max + 1, { addis: data.addis, fitId: '', model: '', method, group }))
})
