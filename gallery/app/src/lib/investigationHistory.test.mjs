import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { execFileSync } from 'node:child_process'
import { EVIDENCE_LABELS, filterHistory, historyMermaid, historyNeighborhood, inspectHistoryExports } from './investigationHistory.ts'

const history = JSON.parse(fs.readFileSync(new URL('../sections/research/history.generated.json', import.meta.url), 'utf8'))

test('the research index preserves source gaps, evidence states and documented claim corrections', () => {
  const ids = new Set(history.records.map(r => r.id))
  assert.equal(ids.size, history.records.length)
  assert.ok(history.records.length > 100)
  assert.ok(history.records.every(r => r.evidence in EVIDENCE_LABELS && (!r.sourceDate || /^\d{4}-\d{2}-\d{2}$/.test(r.sourceDate))))
  assert.ok(history.edges.every(e => ids.has(e.from) && ids.has(e.to) && e.reason && e.source))
  assert.ok(history.records.some(r => r.evidence === 'missing' && !r.sourceAvailable))
  const reproduction = history.records.find(r => r.id === 'next_airspec_reproduction')
  assert.equal(reproduction.evidence, 'verified')
  const validation = history.records.find(r => r.id === 'next_addis_independent_validation')
  assert.equal(validation.evidence, 'blocked')
  const earlier = history.records.find(r => r.id === 'ftir_43')
  assert.ok(earlier.corrections.some(c => c.to === 'ftir_46' && c.reading.includes('includes zero')))
  assert.ok(history.edges.some(e => e.from === 'ftir_43' && e.to === 'ftir_46' && e.type === 'qualifies_claim'))
  assert.equal(history.schema, 2)
  assert.ok(history.records.every(r => r.charts.every(c => !('missingData' in c))))
  assert.deepEqual(history.records.find(r => r.id === 'transfer_diagnosis_20261102').charts[0].primaryExports, ['experimental/continuum_baseline.json'])
})

test('search combines words, research family and evidence without hiding undated source gaps by default', () => {
  assert.ok(filterHistory(history.records, 'paired zero', 'relationships').some(r => r.id === 'ftir_43'))
  assert.deepEqual(filterHistory(history.records, '', 'validation', 'blocked').map(r => r.id), ['next_addis_independent_validation'])
  assert.deepEqual(filterHistory(history.records, 'no-such-record-xyz'), [])
  assert.equal(filterHistory(history.records, '').length, history.records.length)
  assert.equal(filterHistory(history.records, '  FTIR_43  ')[0].id, 'ftir_43')
})

test('connections contain only documented direct neighbours, including other families, without inferred chronology edges', () => {
  const graph = historyNeighborhood(history, 'ftir_43')
  assert.ok(graph.records.some(r => r.id === 'ftir_46'))
  assert.ok(graph.edges.every(e => e.from === 'ftir_43' || e.to === 'ftir_43'))
  const isolated = { ...history, edges: [] }
  assert.deepEqual(historyNeighborhood(isolated, 'ftir_43').records.map(r => r.id), ['ftir_43'])
  assert.deepEqual(historyNeighborhood(isolated, 'ftir_43').edges, [])
})

test('Mermaid exports preserve relationship types, escape source labels and omit dangling edges', () => {
  const a = { id: 'a', title: 'A "quote" [bracket] <html> & line\nbreak' }, b = { id: 'b', title: 'B' }
  const output = historyMermaid([a, b], [
    { from: 'a', to: 'b', type: 'related_work' }, { from: 'b', to: 'a', type: 'qualifies_claim' }, { from: 'b', to: 'missing', type: 'data_dependency' },
  ])
  assert.match(output, /^flowchart LR\n/)
  assert.match(output, /n0 -\.->\|"Related work"\| n1/)
  assert.match(output, /n1 -->\|"Qualifies claim"\| n0/)
  assert.match(output, /#34;quote#34;/)
  assert.ok(!output.includes('<html>') && !output.includes('undefined'))
})

test('the checked-in catalog can be regenerated without executing research', () => {
  execFileSync(process.execPath, ['scripts/build-investigation-history.mjs', '--check'], { cwd: new URL('../../', import.meta.url), stdio: 'pipe' })
})

test('primary-export availability follows the supplied host or snapshot loader and can recover after restoration', async () => {
  const chart = { tab: 'experimental', page: 'continuum', label: 'Continuum', primaryExports: ['experimental/continuum_baseline.json'] }
  let restored = false
  const loader = async path => {
    assert.equal(path, 'experimental/continuum_baseline.json')
    if (!restored) throw new Error('HTTP 404')
    return { schema_version: 1, generated: '2026-11-04T03:44:59Z' }
  }
  assert.deepEqual(await inspectHistoryExports(chart, loader), [{ path: chart.primaryExports[0], error: 'HTTP 404' }])
  restored = true
  assert.deepEqual(await inspectHistoryExports(chart, loader), [{ path: chart.primaryExports[0], error: null, generated: '2026-11-04T03:44:59Z' }])
  assert.deepEqual(await inspectHistoryExports(chart, async () => { throw new Error('not included in this snapshot') }), [
    { path: chart.primaryExports[0], error: 'not included in this snapshot' },
  ])
})

test('a partial export set retains each failure and does not accept null or array placeholders', async () => {
  const chart = { primaryExports: ['ready.json', 'null.json', 'array.json', 'ready.json'] }
  const calls = []
  const results = await inspectHistoryExports(chart, async path => {
    calls.push(path)
    return path === 'ready.json' ? { schema_version: 1 } : path === 'null.json' ? null : []
  })
  assert.deepEqual(calls, ['ready.json', 'null.json', 'array.json'])
  assert.equal(results[0].error, null)
  assert.ok(results.slice(1).every(result => /not a JSON object/.test(result.error)))
})

test('undeclared pages and legacy checkout flags do not imply current availability', async () => {
  const load = async () => { assert.fail('A legacy missingData entry is not a declared primary export') }
  assert.deepEqual(await inspectHistoryExports({ primaryExports: undefined, missingData: ['old-checkout.json'] }, load), [])
  assert.deepEqual(await inspectHistoryExports({ primaryExports: [] }, load), [])
})
