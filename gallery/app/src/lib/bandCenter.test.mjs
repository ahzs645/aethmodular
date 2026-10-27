import assert from 'node:assert/strict'
import { test } from 'node:test'
import { bandCenter } from './bandCenter.ts'

// a 2 cm⁻¹ grid from 1850 down to 1450, as the spectra are stored (high to low)
const wn = Array.from({ length: 201 }, (_, i) => 1850 - 2 * i)
const gauss = (c, h, s) => wn.map((w) => h * Math.exp(-((w - c) ** 2) / (2 * s * s)))

test('a peak on a sloping baseline is found at its centre, above the continuum', () => {
  const y = gauss(1618, 0.02, 12).map((v, i) => v + 0.05 + 0.0001 * (wn[i] - 1450))
  const r = bandCenter(wn, y)
  assert.equal(r.center, 1618)
  assert.ok(Math.abs(r.height - 0.02) < 1e-3, `height ${r.height}`)
  assert.equal(r.edge, false)
})

test('a neighbouring band leaking in at the window edge is flagged, not reported as a peak', () => {
  // the carbonyl shoulder: rising toward 1700 with nothing resolved inside 1550–1680
  const y = gauss(1720, 0.03, 25)
  const r = bandCenter(wn, y)
  assert.equal(r.center, 1680)
  assert.equal(r.edge, true)
})

test('missing anchor windows give no answer rather than a wrong one', () => {
  const short = wn.filter((w) => w < 1700)
  assert.equal(bandCenter(short, short.map(() => 0)), null)
})
