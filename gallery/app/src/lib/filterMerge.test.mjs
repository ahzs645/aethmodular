import test from 'node:test'
import assert from 'node:assert/strict'
import { supplementFilter } from './filterMerge.ts'

const primary = { id: 'ETAD-0001', code: 'ETAD', date: '2023-01-01', site: 'Addis Ababa', excluded: false, exclusion_reason: null,
  'EC (FTIR)': null, 'OC (FTIR)': 1.23456, 'HIPS Fabs': 0, prediction: 1.5 }

test('network values fill missing measurements without replacing precision, zeros or frozen predictions', () => {
  const extra = { ...primary, 'EC (FTIR)': 2.3, 'OC (FTIR)': 1.23, 'HIPS Fabs': 40, 'Iron': .5, prediction: 9 }
  const before = JSON.stringify(primary)
  const merged = supplementFilter(primary, extra, ['EC (FTIR)', 'OC (FTIR)', 'HIPS Fabs', 'Iron'])
  assert.equal(merged['EC (FTIR)'], 2.3)
  assert.equal(merged['OC (FTIR)'], 1.23456)
  assert.equal(merged['HIPS Fabs'], 0)
  assert.equal(merged.Iron, .5)
  assert.equal(merged.prediction, 1.5)
  assert.equal(merged.network_filled_fields, 'EC (FTIR); Iron')
  assert.equal(JSON.stringify(primary), before)
  assert.equal(supplementFilter(primary, undefined, []), primary)
})

test('neither source can erase an exclusion, and different physical filters cannot merge', () => {
  const flagged = { ...primary, excluded: true, exclusion_reason: 'flow anomaly' }
  assert.equal(supplementFilter(flagged, primary, []).excluded, true)
  const merged = supplementFilter(primary, flagged, [])
  assert.equal(merged.excluded, true)
  assert.equal(merged.exclusion_reason, 'flow anomaly')
  assert.throws(() => supplementFilter(primary, { ...flagged, id: 'ETAD-0002' }, []), /physical filters/)
})
