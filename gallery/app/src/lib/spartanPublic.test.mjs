import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { csvFormatRows } from 'd3'
import { parsePublicSpartan, mergePublicSpartan, loadPublicSpartan, PUBLIC_SPARTAN_FIELDS } from './spartanPublic.ts'
import { replacePublicFilter, supplementFilter } from './filterMerge.ts'

const site = { code: 'ETAD', name: 'Addis Ababa', lat: 9, lon: 38 }
const checked = '2026-10-28T20:00:00Z'
const columns = ['Site_Code', 'Latitude', 'Longitude', 'Filter_ID', 'Start_Year_local', 'Start_Month_local', 'Start_Day_local', 'Parameter_Name', 'Value', 'Units', 'Method_Code', 'Analysis_Description', 'Collection_Description', 'Hours_sampled', 'Flag']
const record = (parameter = 'EC PM2.5', value = 2, changes = {}) => ({ Site_Code: 'ETAD', Latitude: 9, Longitude: 38, Filter_ID: 'ETAD-0001', Start_Year_local: 2024, Start_Month_local: 2, Start_Day_local: 29,
  Parameter_Name: parameter, Value: value, Units: 'Micrograms per cubic meter (ug/m3)', Method_Code: 218, Analysis_Description: 'FTIR', Collection_Description: 'SS5i sampler', Hours_sampled: 24, Flag: '', ...changes })
const csv = records => '\uFEFF# File Updated: 2026-10-27\r\n# Data version 3.1.4\r\n# Site: Addis, Ethiopia\r\n' + csvFormatRows([columns, ...records.map(r => columns.map(k => r[k]))])

test('live chemistry handles metadata, quoted CSV, units, repeat reports, zeros, negatives and missing values', () => {
  const data = parsePublicSpartan(csv([
    record(), record('EC PM2.5', 4, { Filter_ID: 'ETAD-0001-12' }),
    record('Iron PM2.5', 500, { Units: 'Nanograms per cubic meter (ng/m3)', Analysis_Description: 'XRF, revised' }),
    record('Sulfate Ion PM2.5', 0), record('Nitrate Ion PM2.5', -.3), record('OC PM2.5', ''), record('OM PM2.5', 'NaN'),
  ]), site, checked)
  assert.equal(data.rows.length, 1)
  const r = data.rows[0]
  assert.equal(r.id, 'ETAD-0001')
  assert.equal(r['EC (FTIR)'], 3)
  assert.equal(r['EC (ChemSpec FTIR)'], 3)
  assert.equal(r.Iron, .5)
  assert.equal(r.Sulfate, 0)
  assert.equal(r.Nitrate, -.3)
  assert.equal(r['OC (FTIR)'], null)
  assert.equal(r['OM (FTIR)'], null)
  assert.ok(Math.abs(r['Sampled volume'] - 7.2) < 1e-12)
  assert.equal(r.date, '2024-02-29')
  assert.equal(data.status.updated, '2026-10-27')
  assert.equal(data.status.version, '3.1.4')
  assert.equal(data.status.checked, checked)
})

test('direct HIPS, the HIPS–SSR estimate and SSR retain separate fields and source flags', () => {
  const data = parsePublicSpartan(csv([
    record('BC PM2.5', 1, { Method_Code: 219, Analysis_Description: 'HIPS' }),
    record('BC PM2.5', 3, { Method_Code: 221, Analysis_Description: 'HIPS' }),
    record('BC PM2.5', 9, { Method_Code: 220, Analysis_Description: 'Estimated from the HIPS-SSR BC curve' }),
    record('Equivalent BC PM2.5', 4, { Method_Code: 213, Analysis_Description: 'SSR', Flag: 'VIH' }),
  ]), site)
  const r = data.rows[0]
  assert.equal(r['BC (public HIPS)'], 2)
  assert.equal(r['BC (public SSR estimate)'], 9)
  assert.equal(r['Equivalent BC (SSR)'], 4)
  assert.equal(r['HIPS BC'], undefined)
  assert.equal(r.public_quality_flags, 'VIH')
  assert.match(r.public_method_codes, /219, 221, 220, 213/)
})

test('a changed schema, wrong site, ambiguous dates or unrecognized methods/units cannot masquerade as live chemistry', () => {
  for (const [changes, pattern] of [
    [{ Site_Code: 'ARCB' }, /identity/], [{ Filter_ID: 'ARCB-0001' }, /identity/],
    [{ Start_Year_local: 2023 }, /date/], [{ Units: 'mg/kg' }, /unit/],
    [{ Analysis_Description: 'TOR' }, /carbon method/], [{ Parameter_Name: 'Future new species' }, /Unmapped/],
    [{ Parameter_Name: 'BC PM2.5', Method_Code: 999, Analysis_Description: 'unknown' }, /BC method/],
  ]) assert.throws(() => parsePublicSpartan(csv([record('EC PM2.5', 1, changes)]), site), pattern)
  assert.throws(() => parsePublicSpartan(csv([record(), record('Iron PM2.5', 2, { Start_Day_local: 28 })]), site), /Conflicting/)
  assert.throws(() => parsePublicSpartan('<html>Missing file</html>', site), /header/)
  assert.throws(() => parsePublicSpartan(csv([]), site), /no filters/)
})

test('live merge updates public values, removes withdrawn values, adds filters and preserves lab results, predictions and flags', () => {
  const live = parsePublicSpartan(csv([record('EC PM2.5', 7), record('EC PM2.5', 8, { Filter_ID: 'ETAD-0003' })]), site)
  const bundled = [{ ...live.rows[0], 'EC (FTIR)': 2, Iron: 1, 'HIPS BC': 5, 'Alkane CH': 6, prediction: 1.234567,
    excluded: true, exclusion_reason: 'Known study exclusion' },
  { ...live.rows[0], id: 'ETAD-0002', 'EC (FTIR)': 3, 'HIPS BC': 4 }]
  const before = JSON.stringify(bundled)
  const merged = mergePublicSpartan(bundled, live)
  assert.equal(merged.length, 3)
  const r = merged.find(r => r.id === 'ETAD-0001')
  assert.equal(r['EC (FTIR)'], 7)
  assert.equal(r.Iron, null)
  assert.equal(r['HIPS BC'], 5)
  assert.equal(r['Alkane CH'], 6)
  assert.equal(r.prediction, 1.234567)
  assert.equal(r.excluded, true)
  assert.equal(r.exclusion_reason, 'Known study exclusion')
  assert.equal(merged.find(r => r.id === 'ETAD-0002')['EC (FTIR)'], null)
  assert.equal(merged.find(r => r.id === 'ETAD-0002')['HIPS BC'], 4)
  // App's second merge must apply the same replacement to focus-site rows.
  const focus = { ...bundled[0], 'EC (FTIR)': 9.123456 }
  const again = replacePublicFilter(supplementFilter(focus, r, PUBLIC_SPARTAN_FIELDS), r, PUBLIC_SPARTAN_FIELDS)
  assert.equal(again['EC (FTIR)'], 7)
  assert.equal(again.prediction, 1.234567)
  assert.equal(again.excluded, true)
  assert.equal(JSON.stringify(bundled), before)
  assert.throws(() => replacePublicFilter(bundled[0], { ...r, code: 'ARCB' }, PUBLIC_SPARTAN_FIELDS), /different physical/)
})

test('live fetches are anonymous, shared per site, refreshed explicitly, and retryable after failure', async () => {
  const original = globalThis.fetch, calls = []
  let fail = false
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options })
    return fail ? new Response('offline', { status: 503 }) : new Response(csv([record()]))
  }
  try {
    await Promise.all([loadPublicSpartan(site, 10), loadPublicSpartan(site, 10)])
    assert.equal(calls.length, 1)
    assert.match(calls[0].url, /ChemSpecPM25_ETAD.csv$/)
    assert.equal(calls[0].options.credentials, 'omit')
    assert.equal(calls[0].options.cache, 'no-cache')
    await loadPublicSpartan(site, 11)
    assert.equal(calls.length, 2)
    fail = true
    await assert.rejects(loadPublicSpartan(site, 12), /503/)
    fail = false
    await loadPublicSpartan(site, 12)
    assert.equal(calls.length, 4)
  } finally { globalThis.fetch = original }
})

test('the parser accepts every locally available public site file with the exporter parameter mapping', t => {
  const folder = new URL('../../../../data/spartan/raw/FilterBased/ChemSpecPM25/', import.meta.url)
  let files
  try { files = readdirSync(folder).filter(f => f.endsWith('.csv')) } catch { t.skip('Public mirror not present'); return }
  for (const file of files) {
    const code = file.match(/_([A-Z]{4})\.csv$/)[1]
    const result = parsePublicSpartan(readFileSync(new URL(file, folder), 'utf8'), { ...site, code })
    assert.ok(result.rows.length > 0, code)
    assert.ok(result.rows.every(r => r.code === code), code)
  }
  assert.equal(new Set(PUBLIC_SPARTAN_FIELDS).size, PUBLIC_SPARTAN_FIELDS.length)
})
