// Saved per-site tables under the Samples sites (savedSites.ts), on the labels the exports really carry
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { rowsOf } from '../../../lib/table.ts'
import { samplesCodes, savedSiteCode, savedSites, siteInView } from './savedSites.ts'

const read = (f) => JSON.parse(fs.readFileSync(new URL(`../../../../public/data/${f}`, import.meta.url), 'utf8'))
const FIVE = ['CHTS', 'ETAD', 'ETBI', 'INDH', 'USPA']

test('savedSiteCode reads every spelling the saved tables use', () => {
  const committee = read('research/spectral/committee.json').tables, band = read('research/spectral/band_1617.json').tables, net = read('research/spectral/network_map.json').tables
  const labels = [...rowsOf(committee.five_site).map((r) => r.site), ...rowsOf(band.site_bands).map((r) => r.site), ...rowsOf(net.class_share).map((r) => r.target), ...rowsOf(net.threshold).map((r) => r.target)]
  assert.ok(labels.length > 30)
  for (const l of labels) assert.ok(savedSiteCode(l), l)
  assert.deepEqual([...new Set(labels.map(savedSiteCode))].sort(), FIVE)
  assert.equal(savedSiteCode('Addis Ababa'), 'ETAD')
  assert.equal(savedSiteCode('Pasadena (USPA): screened'), 'USPA')
})

test('savedSiteCode: a pool names no one site', () => {
  assert.equal(savedSiteCode('pooled: 4 non-Addis sites'), null)
  assert.equal(savedSiteCode(null), null)
})

test('at the five calibration sites every saved row stays, in its order', () => {
  const codes = new Set(FIVE)
  const massTerm = rowsOf(read('research/spectral/committee.json').tables.mass_term)
  assert.deepEqual(savedSites(massTerm, codes, (r) => r.site), massTerm)
  const share = rowsOf(read('research/spectral/network_map.json').tables.class_share)
  assert.deepEqual(savedSites(share, codes, (r) => r.target), share)
})

test('a narrower pick keeps its sites and the pool', () => {
  const rows = [{ site: 'Addis Ababa' }, { site: 'Beijing' }, { site: 'pooled: 4 non-Addis sites' }]
  assert.deepEqual(savedSites(rows, new Set(['ETAD']), (r) => r.site).map((r) => r.site), ['Addis Ababa', 'pooled: 4 non-Addis sites'])
})

test('with no site of the table in Samples nothing stays, not even the pool', () => {
  const rows = [{ site: 'Addis Ababa' }, { site: 'pooled: 4 non-Addis sites' }]
  assert.deepEqual(savedSites(rows, new Set(['AEAZ']), (r) => r.site), [])
  assert.deepEqual(savedSites(rows, new Set(), (r) => r.site), [])
  assert.equal(siteInView(new Set(), 'pooled: 4 non-Addis sites'), true)
})

test('samplesCodes: the pick, or the rows in view where there is no pick (net=focus)', () => {
  assert.deepEqual([...samplesCodes(new Set(['ETAD']), [{ code: 'CHTS' }])], ['ETAD'])
  assert.deepEqual([...samplesCodes(new Set(), [{ code: 'CHTS' }, { code: 'CHTS' }, { code: 'INDH' }])], ['CHTS', 'INDH'])
  assert.equal(samplesCodes(new Set(), []).size, 0)
})
