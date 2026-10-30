import test from 'node:test'
import assert from 'node:assert/strict'
import { builderDraft, builderFieldId, builderFields, builderRows, builderScales, builderSchema, readBuilderDraft } from './chartBuilder.ts'

const meta = {
  generated: '2026-10-30', mac_value: 10, basis: 'conc', season_convention: 'local_2026_10',
  sites: [{ code: 'ETAD', name: 'Addis Ababa', color: '#123456' }], seasons: [{ name: 'Addis Ababa · Belg', color: '#789abc' }],
  fields: ['EC (FTIR)', 'PM2.5', 'EC (ChemSpec FTIR)', 'Iron', 'Ratio'],
  field_units: { 'EC (FTIR)': 'ug/m3', 'EC (ChemSpec FTIR)': 'ug/m3', 'PM2.5': 'ug/m3', Iron: 'ug/m3' },
}
const row = { id: 'ETAD-0001', site: 'Addis Ababa', code: 'ETAD', date: '2023-03-01', year: 2023, month: 3,
  season: 'Addis Ababa · Belg', excluded: true, exclusion_reason: 'flow flag', color: '#123456',
  'EC (FTIR)': 0, 'PM2.5': null, 'EC (ChemSpec FTIR)': 7, Iron: Infinity, Ratio: '3', privatePrediction: 99 }
const chart = { visId: 'draft-1', name: 'Faceted comparison', config: { geoms: ['point'], defaultAggregated: false },
  layout: {}, encodings: { dimensions: [], measures: [], rows: [], columns: [], filters: [] } }

test('the adapter preserves zero, missingness, exclusions and calendar labels without exposing duplicate or undeclared measurements', () => {
  const before = structuredClone(row)
  const [data] = builderRows([row], meta)
  assert.equal(data[builderFieldId('EC (FTIR)', 'ug/m3')], 0)
  for (const [field, unit] of [['PM2.5', 'ug/m3'], ['Iron', 'ug/m3'], ['Ratio', '']]) assert.equal(data[builderFieldId(field, unit)], null)
  assert.equal(data.excluded, true)
  assert.equal(data.exclusion_reason, 'flow flag')
  assert.equal(data.season, row.season)
  assert.equal(data.date, row.date)
  assert.equal('privatePrediction' in data, false)
  assert.equal('color' in data, false)
  assert.equal(builderFieldId('EC (ChemSpec FTIR)', 'ug/m3') in data, false)
  assert.deepEqual(row, before)
})

test('measurement identifiers survive reordering and are safe for Vega paths and SQL; unit changes have a distinct identity', () => {
  const fields = builderFields(meta)
  const reordered = builderFields({ ...meta, fields: [...meta.fields].reverse() })
  assert.equal(builderSchema(fields), builderSchema(reordered))
  const ids = ['PM2.5', 'PM2_5', 'a[b]', 'a.b', 'a b', 'a/b', 'µg'].map(f => builderFieldId(f))
  assert.equal(new Set(ids).size, ids.length)
  assert.ok(ids.every(id => /^m_[a-z0-9_]+$/.test(id)))
  assert.notEqual(builderFieldId('EC (FTIR)', 'ug/m3'), builderFieldId('EC (FTIR)', 'ug/filter'))
  assert.equal(fields.find(f => f.fid === 'year').analyticType, 'dimension')
  assert.equal(fields.find(f => f.fid === 'date').semanticType, 'temporal')
  assert.match(fields.find(f => f.fid === builderFieldId('EC (FTIR)', 'ug/m3')).name, /µg\/m³/)
})

test('empty populations and missing or invalid dates remain empty instead of being coerced', () => {
  assert.deepEqual(builderRows([], meta), [])
  for (const date of ['', null, 'not-a-date']) assert.equal(builderRows([{ ...row, date }], meta)[0].date, null)
})

test('site and season encodings reuse metadata colours without imposing categorical colours on measurements', () => {
  const { color } = builderScales(meta)
  assert.deepEqual(color({ semanticType: 'nominal', values: ['Addis Ababa', 'Addis Ababa', null] }), { domain: ['Addis Ababa'], range: ['#123456'] })
  assert.deepEqual(color({ semanticType: 'nominal', values: ['Addis Ababa · Belg'] }), { domain: ['Addis Ababa · Belg'], range: ['#789abc'] })
  assert.deepEqual(color({ semanticType: 'quantitative', values: [1, 2] }), {})
  assert.deepEqual(color({ semanticType: 'nominal', values: ['new category'] }), {})
})

test('drafts round trip definitions and selection provenance without measurement values', () => {
  const fields = builderFields(meta)
  const draft = builderDraft([chart], fields, [row], meta, '#tab=builder&sites=Addis')
  const restored = readBuilderDraft(JSON.parse(JSON.stringify(draft)), fields)
  assert.deepEqual(restored.charts, [chart])
  assert.deepEqual(restored.context.filterIds, [row.id])
  assert.equal(restored.context.mac, 10)
  assert.equal(restored.context.seasonConvention, meta.season_convention)
  assert.equal('data' in restored, false)
  assert.equal('rows' in restored, false)
  assert.equal(JSON.stringify(restored).includes('privatePrediction'), false)
})

test('incompatible units, versions and malformed drafts cannot silently become gallery charts', () => {
  const fields = builderFields(meta)
  const draft = builderDraft([chart], fields, [row], meta, '#tab=builder')
  const mass = builderFields({ ...meta, field_units: { ...meta.field_units, 'EC (FTIR)': 'ug/filter' } })
  assert.throws(() => readBuilderDraft(draft, mass), /fields or units/)
  assert.throws(() => readBuilderDraft({ ...draft, version: 2 }, fields), /unsupported/)
  assert.throws(() => readBuilderDraft({ ...draft, graphicWalkerVersion: 'unreviewed' }, fields), /unsupported/)
  assert.throws(() => readBuilderDraft({ ...draft, charts: [] }, fields), /chart definitions/)
  assert.throws(() => readBuilderDraft({ ...draft, charts: [null] }, fields), /chart definitions/)
  assert.throws(() => readBuilderDraft({ ...draft, fields: [null] }, fields), /fields or units/)
  assert.throws(() => readBuilderDraft({ ...draft, context: {} }, fields), /filter selection/)
})
