import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { test } from 'node:test'
import { addisTargetSpectra, rankedAnalogSpectra } from './satoshiSpectra.ts'

const sources = ['satoshi/followup_20261002.json', 'similarity/similarity.json', 'similarity/etad_ids.json']
  .map(path => new URL(`../../public/data/${path}`, import.meta.url))
const skip = !sources.every(existsSync)
const read = url => JSON.parse(readFileSync(url, 'utf8'))

test('spectral overlay joins all 500 saved analogs, in saved rank order, under both regions and baselines', { skip }, () => {
  const [data, meta] = sources.map(read)
  for (const group of data.groups) for (const baseline of ['AIRSpec', 'VIBES-full-r70']) for (const region of ['presented', 'oct1']) {
    const { rows, missing } = rankedAnalogSpectra(data.members, meta, baseline, region, group)
    assert.deepEqual(missing, [])
    assert.equal(rows.length, 500, `${baseline}, ${region}, ${group}`)
    assert.equal(new Set(rows.map(row => row.filter)).size, 500)
    assert.deepEqual(rows.map(row => row.rank), Array.from({ length: 500 }, (_, i) => i + 1))
    assert.ok(rows.every(row => meta.filters.sample_id[row.filter].startsWith('improve:')))
    const column = baseline === 'AIRSpec' ? 'spline' : 'vibes'
    const first = data.members.filter_id.findIndex((_, i) => data.members.mask[i] === region && data.members.group[i] === group && data.members[`rank_${column}`][i] === 1)
    assert.equal(meta.filters.sample_id[rows[0].filter], `improve:${data.members.filter_id[first]}`)
    assert.equal(rows[0].r, data.members[`r_${column}`][first])
  }
  const ranks = (baseline, region) => rankedAnalogSpectra(data.members, meta, baseline, region, 'All Addis').rows.map(row => row.filter)
  assert.deepEqual(ranks('AIRSpec', 'presented'), ranks('AIRSpec', 'oct1'))
  assert.notDeepEqual(ranks('VIBES-full-r70', 'presented'), ranks('VIBES-full-r70', 'oct1'))
})

test('overlay targets the 233 frozen evaluation filters, with February in Dry, rather than all 253 available spectra', { skip }, () => {
  const [data, meta, ids] = sources.map(read)
  const all = addisTargetSpectra(meta, ids, 'All Addis')
  assert.equal(all.filters.length, 233)
  const seasons = data.seasons.map(group => addisTargetSpectra(meta, ids, group).filters)
  const savedCounts = data.seasons.map(group => data.ladder.find(row =>
    row.family === 'analog' && row.baseline === 'AIRSpec' && row.mask === 'presented' &&
    row.stage === 'Addis test set' && row.group === group && row.eval_group === group).n)
  assert.deepEqual(seasons.map(filters => filters.length), savedCounts)
  assert.deepEqual([...new Set(seasons.flat())].sort((a, b) => a - b), all.filters)
  const february = all.filters.filter(filter => new Date(meta.filters.date[filter]).getUTCMonth() === 1)
  assert.ok(february.length > 0)
  assert.ok(february.every(filter => seasons[0].includes(filter)))
})
