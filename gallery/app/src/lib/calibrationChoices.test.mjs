import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { calibrationCohort, calibrationSubset, cohortConfig, cohortPreset } from './calibrationChoices.ts'

const read = file => JSON.parse(fs.readFileSync(new URL(`../../public/data/${file}`, import.meta.url), 'utf8'))
const grid = read('calibration.json'), runs = read('calibration_runs.json')
const start = { co: 'ocec', cut: 800, sel: 'raw', sp: 'airspec' }

test('per-chart choices preserve the original dataset and its settings', () => {
  assert.equal(calibrationSubset(grid, ''), grid)
  assert.equal(cohortConfig(start, grid, ''), start)
  assert.equal(cohortPreset(runs.presets, 'ocec800_airspec', '').key, 'ocec800_airspec')
})

test('cohort and protocol narrow saved grid rows while retaining the fixed-protocol factor sweeps', () => {
  const before = structuredClone(grid)
  const cohort = calibrationSubset(grid, 'eth_shaped')
  const protocol = calibrationSubset(cohort, '', 'app')
  assert.ok(protocol.grid.length > 0 && protocol.grid.length < cohort.grid.length)
  assert.ok(protocol.grid.every(r => r.co === 'eth_shaped' && r.mo === 'app'))
  assert.ok(cohort.sweeps.length > 0 && cohort.sweeps.every(r => r.co === 'eth_shaped'))
  assert.equal(protocol.sweeps, cohort.sweeps)
  assert.deepEqual(Object.keys(cohort.default_cutoff), ['eth_shaped'])
  assert.deepEqual(grid, before)
})

test('a cohort change preserves saved preprocessing when possible and never selects a foreign preset', () => {
  assert.equal(cohortPreset(runs.presets, 'ocec800_airspec', 'eth_shaped').key, 'eth300_airspec')
  assert.equal(cohortPreset(runs.presets, 'ocec800_airspec', 'ocec').key, 'ocec800_airspec')
  const smoke = cohortPreset(runs.presets, 'ocec800_airspec', 'smoke')
  assert.equal(smoke.cohort, 'smoke')
  assert.equal(smoke.spectra, 'raw')
  assert.equal(cohortPreset(runs.presets, 'ocec800_airspec', 'not-exported'), undefined)
})

test('grid selections resolve to actual exported configurations in each cohort', () => {
  for (const co of Object.keys(grid.cohorts)) {
    const config = cohortConfig(start, grid, co)
    assert.equal(config.co, co)
    assert.ok(grid.sweeps.some(r => r.co === config.co && r.cut === config.cut && r.sel === config.sel && r.sp === config.sp))
  }
  assert.equal(cohortConfig(start, grid, 'ocec'), start)
})

test('unavailable protocols remain empty instead of substituting another saved protocol', () => {
  const siteGroupedOnly = { ...grid, grid: grid.grid.filter(r => r.mo === 'site_heldout') }
  const subset = calibrationSubset(siteGroupedOnly, 'pool', 'app')
  assert.equal(subset.grid.length, 0)
  assert.ok(subset.sweeps.length > 0)
  assert.ok(subset.sweeps.every(r => r.co === 'pool'))
  assert.equal(cohortPreset(runs.presets, 'ocec800_airspec', 'pool').cutoff, null)
})

test('VIP analog cohort names match between the original and rank-70 searches', () => {
  assert.equal(calibrationCohort('vip_analogs'), calibrationCohort('analogs'))
  assert.equal(calibrationCohort('eth_shaped'), 'eth_shaped')
})
