// lib/vibes: the WebAssembly solver, the resampler and the file parser against values from the Python originals
// (scripts/vibes_batch.py fit_elbo_newton + solve_map_batch; workflows/network_baselines_20261019.py resample)
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { VibesSolver, truncate } from './solver.ts'
import { gridWavenumbers, resampleToGrid } from './resample.ts'
import { parseSpectrumText } from './parse.ts'

const near = (a, b, tol) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} vs ${b}`)
const wasm = readFileSync(new URL('./vibes_wasm.wasm', import.meta.url))
const GRID = { first: 3998.42320552162, last: 500.124317346017, n: 2722 }
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1))

// the fixture the Python values were computed on: a smooth background family, one band, a ripple
const p = 240, c = 6
const t = linspace(0, 1, p)
const W = new Float64Array(p * c), mean = new Float64Array(p)
for (let j = 0; j < p; j++) { mean[j] = 0.2 + 0.05 * t[j]; for (let k = 0; k < c; k++) W[j * c + k] = 0.03 * 0.4 ** k * Math.cos(Math.PI * k * t[j]) }
const weights = [0.5, -1.2, 0.8, 0.3, -0.6, 1.1]
const spectrum = (height) => Float64Array.from(t, (tj, j) => {
  let v = mean[j] + height * Math.exp(-(((tj - 0.4) / 0.03) ** 2)) + 2e-5 * Math.sin(97 * tj)
  for (let k = 0; k < c; k++) v += weights[k] * W[j * c + k]
  return v
})

test('the module in the gallery is the one its build record names', () => {
  const record = JSON.parse(readFileSync(new URL('./vibes_wasm.build.json', import.meta.url), 'utf8'))
  assert.equal(createHash('sha256').update(wasm).digest('hex'), record.wasm_sha256)
  assert.equal(wasm.length, record.bytes)
})

test('noise level and baseline match the Python solver', async () => {
  const solver = await VibesSolver.create(wasm, W, mean, p, c)
  const band = solver.fit(spectrum(0.01), 0.1)
  assert.equal(band.status, 'converged'); assert.ok(band.certified)
  near(band.sigma, 5.5895480974785986e-05, 1e-7)
  near(band.baseline[0], 0.20487697003532995, 1e-9); near(band.baseline[96], 0.22729074694376517, 1e-9); near(band.baseline[239], 0.28184568202686083, 1e-9)
  // a spectrum with no band fits down to the ripple
  near(solver.fit(spectrum(0), 0.1).sigma, 2.0318205579729797e-06, 1e-6)
  const wider = solver.fit(spectrum(0.01), 0.2)
  near(wider.sigma, 0.00011104182482171719, 1e-7); near(wider.baseline[96], 0.2273130827553038, 1e-9)
})

test('a lower rank is the first columns of the background', async () => {
  const W3 = truncate(W, p, c, 3)
  assert.equal(W3.length, p * 3); assert.equal(W3[3 * 5 + 2], W[c * 5 + 2])
  assert.equal(truncate(W, p, c, c), W)
  assert.throws(() => truncate(W, p, c, 7))
  const fit = (await VibesSolver.create(wasm, W3, mean, p, 3)).fit(spectrum(0.01), 0.1)
  near(fit.sigma, 0.00011478595715353256, 1e-7); near(fit.baseline[239], 0.28194669875203815, 1e-9)
})

test('the solver refuses a spectrum on another grid', async () => {
  const solver = await VibesSolver.create(wasm, W, mean, p, c)
  assert.throws(() => solver.fit(new Float64Array(p - 1)), /points/)
  assert.throws(() => solver.fit(spectrum(0), 1), /tau/)
  await assert.rejects(VibesSolver.create(wasm, W, mean.subarray(1), p, c))
})

test('resampling matches the network run, falling or rising input', () => {
  const wn = linspace(3997.8, 419.2, 2502), y = wn.map((w) => Math.sin(w / 300) + 1e-4 * w)
  const down = resampleToGrid(wn, y, GRID)
  assert.equal(down.extended, 1)
  const expected = [[0, 1.0884323852455253], [1, 1.0867617473930704], [1000, 0.6442893176128311], [2720, 1.0450852623435707], [2721, 1.0453778418560096]]
  for (const [i, v] of expected) near(down.values[i], v, 1e-12)
  const up = resampleToGrid([...wn].reverse(), [...y].reverse(), GRID)
  for (const [i, v] of expected) near(up.values[i], v, 1e-12)
})

test('a spectrum shorter than the grid is extended flat and counted', () => {
  const wn = linspace(3900, 600, 400), y = wn.map((w) => Math.cos(w / 150))
  const r = resampleToGrid(wn, y, GRID)
  assert.equal(r.extended, 155)
  for (const [i, v] of [[0, 0.6469193223286392], [50, 0.6469193223286392], [1500, 0.33119528823225824], [2721, -0.6536436208636119]]) near(r.values[i], v, 1e-12)
})

test('a straight line resamples to itself, and bad input is refused', () => {
  const grid = { first: 100, last: 10, n: 10 }
  const wn = linspace(120, 0, 61), r = resampleToGrid(wn, wn.map((w) => 2 * w + 1), grid)
  gridWavenumbers(grid).forEach((w, i) => near(r.values[i], 2 * w + 1, 1e-12))
  assert.equal(r.extended, 0)
  assert.throws(() => resampleToGrid([3, 2, 2, 1], [0, 0, 0, 0], grid), /one way/)
  assert.throws(() => resampleToGrid([3, 2, 1], [0, NaN, 0], grid), /not a number/)
})

test('a text file gives wavenumber and absorbance, header skipped, sorted by falling wavenumber', () => {
  const rows = Array.from({ length: 60 }, (_, i) => `${500 + i * 10}, ${(i / 100).toFixed(4)}`)
  const s = parseSpectrumText(['wavenumber,absorbance', '# comment', ...rows, ''].join('\r\n'))
  assert.equal(s.skipped, 2); assert.equal(s.wn.length, 60); assert.equal(s.wn[0], 1090); assert.equal(s.y[0], 0.59)
  assert.deepEqual(parseSpectrumText(rows.map((r) => r.replace(', ', '\t')).join('\n')).wn, s.wn)
  assert.deepEqual(parseSpectrumText(rows.map((r) => r.replace(', ', '   ')).join('\n')).y, s.y)
  assert.throws(() => parseSpectrumText(rows.slice(0, 10).join('\n')), /at least 50/)
  assert.throws(() => parseSpectrumText([...rows, rows[0]].join('\n')), /twice/)
})
