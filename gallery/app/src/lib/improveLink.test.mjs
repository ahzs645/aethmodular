import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { test } from 'node:test'
import { improveSampleFor, improveSiteDate, matchImproveRow, parseImproveRowId } from './improveLink.ts'

const data = (path) => new URL(`../../public/data/${path}`, import.meta.url)
const simUrl = data('similarity/similarity.json')
const skip = !existsSync(simUrl) || !existsSync(data('improve_samples/PINN1.json'))
const read = (url) => JSON.parse(readFileSync(url, 'utf8'))

const meta = {
  sites: [{ site: 'ETAD' }, { site: 'PINN1' }],
  filters: { sample_id: ['etad:1', 'improve:10', 'improve:11', 'improve:12'], site: [0, 1, 1, 1], date: ['2022-01-01', '2022-11-22', '2022-11-25', null], season: [0, 0, 0, 0] },
}

test('row ids parse into site, day and sampler', () => {
  assert.deepEqual(parseImproveRowId('PINN1-2022-11-22'), { code: 'PINN1', date: '2022-11-22', poc: 1 })
  assert.deepEqual(parseImproveRowId('PINN1-2022-11-22-2'), { code: 'PINN1', date: '2022-11-22', poc: 2 })
  // SPARTAN filter ids and spectral sample ids are not network rows
  assert.equal(parseImproveRowId('ETAD-0017'), null)
  assert.equal(parseImproveRowId('ETAD-0017-1'), null)
  assert.equal(parseImproveRowId('improve:2005726'), null)
})

test('a spectral sample resolves to its site and day, and an undated one does not', () => {
  assert.deepEqual(improveSiteDate(meta, 'improve:10'), { code: 'PINN1', date: '2022-11-22' })
  assert.equal(improveSiteDate(meta, 'improve:12'), null)
  assert.equal(improveSiteDate(meta, 'improve:99'), null)
  assert.equal(improveSiteDate(meta, 'etad:1'), null)
})

test('a day with two network rows (collocated samplers) is not guessed', () => {
  const rows = [{ id: 'PINN1-2022-11-22', code: 'PINN1', date: '2022-11-22' }, { id: 'PINN1-2022-11-25', code: 'PINN1', date: '2022-11-25' },
    { id: 'PINN1-2022-11-25-2', code: 'PINN1', date: '2022-11-25' }]
  assert.equal(matchImproveRow(rows, 'PINN1', '2022-11-22')?.id, 'PINN1-2022-11-22')
  assert.equal(matchImproveRow(rows, 'PINN1', '2022-11-25'), null)
  assert.equal(matchImproveRow(rows, 'PINN1', '2022-11-28'), null)
})

test('a network row finds its spectrum the other way, for the primary sampler only', () => {
  assert.equal(improveSampleFor(meta, 'PINN1-2022-11-22'), 'improve:10')
  assert.equal(improveSampleFor(meta, 'PINN1-2022-11-22-2'), null)
  assert.equal(improveSampleFor(meta, 'PINN1-2022-12-01'), null)
  assert.equal(improveSampleFor(meta, 'ETAD-0017'), null)
  const twice = { ...meta, filters: { ...meta.filters, date: ['2022-01-01', '2022-11-22', '2022-11-22', null] } }
  assert.equal(improveSampleFor(twice, 'PINN1-2022-11-22'), null)
})

test('the exported PINN1 filter links both ways, and nearly every IMPROVE spectrum has a network row', { skip }, () => {
  const sim = read(simUrl)
  const at = improveSiteDate(sim, 'improve:2005726')
  assert.deepEqual(at, { code: 'PINN1', date: '2022-11-22' })
  assert.equal(improveSampleFor(sim, 'PINN1-2022-11-22'), 'improve:2005726')

  const files = new Map()
  const rowsOf = (code) => {
    if (!files.has(code)) {
      const url = data(`improve_samples/${code}.json`)
      const f = existsSync(url) ? read(url) : { date: [], poc: [] }
      files.set(code, f.date.map((d, i) => ({ id: `${code}-${d}${f.poc[i] !== 1 ? `-${f.poc[i]}` : ''}`, code, date: d })))
    }
    return files.get(code)
  }
  let total = 0, linked = 0, roundTrip = 0
  sim.filters.sample_id.forEach((id) => {
    const sd = improveSiteDate(sim, id)
    if (!id.startsWith('improve:')) return
    total++
    const row = sd && matchImproveRow(rowsOf(sd.code), sd.code, sd.date)
    if (!row) return
    linked++
    if (improveSampleFor(sim, row.id) === id) roundTrip++
  })
  // 11,954 of 12,393 on 2 Nov 2026; the rest are days the network export does not carry
  assert.ok(linked / total > 0.9, `${linked} of ${total} linked`)
  assert.equal(roundTrip, linked)
})
