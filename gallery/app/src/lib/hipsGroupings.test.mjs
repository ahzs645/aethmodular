// The IMPROVE regrouping in the browser (lib/hipsGroupings) against the exporter's sweep, which fits exact medians:
// the same edges, the same groups, and a ceiling within the histogram medians' error.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { dropOne, fitCeiling, groupsOf, histMedian, ruleEdges, sweepRow } from './hipsGroupings.ts'

const file = JSON.parse(fs.readFileSync(new URL('../../public/data/hips_linearise/improve_groupings.json', import.meta.url), 'utf8'))

test('the published grouping rebuilds the 17 groups and the 2.26 ceiling', () => {
  const basis = file.bases.previous
  const { groups } = groupsOf(file, basis, ruleEdges('published', 0, basis), file.min_n)
  assert.equal(groups.length, 17)
  assert.equal(groups.reduce((a, g) => a + g.n, 0) <= file.samples, true)
  const fit = fitCeiling(groups, 'log')
  assert.ok(Math.abs(fit.tauInf - 2.2596) < 0.03, `τ∞ ${fit.tauInf}`)
  // the profile's lowest point is the fit
  const low = fit.profile.reduce((a, p) => (p.rms < a.rms ? p : a))
  assert.ok(Math.abs(Math.log(low.tauInf / fit.tauInf)) < 0.1)
})

test('every grouping in the sweep: the same edges, the same groups, and the ceiling within 4 %', () => {
  let checked = 0
  for (const row of file.sweep) {
    const basis = file.bases[row.basis]
    const edges = ruleEdges(row.rule, row.groups, basis)
    assert.deepEqual(edges.map((i) => basis.fine[i]), row.edges, `${row.basis} ${row.rule} ${row.groups}`)
    const { groups } = groupsOf(file, basis, edges, file.min_n)
    assert.equal(groups.length, row.used, `${row.basis} ${row.rule} ${row.groups} groups`)
    const fit = fitCeiling(groups, 'log')
    // three or four medians for two parameters: the fit swings with the last digit of a median, so only the grouping is checked
    if (row.used < 5) continue
    if (row.tau_inf > 90 || row.tau_inf < 0.25) { assert.ok(fit.atBound || fit.tauInf > 20 || fit.tauInf < 0.3, `${row.basis} ${row.rule} ${row.groups} at bound`); continue }
    assert.ok(Math.abs(fit.tauInf / row.tau_inf - 1) < 0.04, `${row.basis} ${row.rule} ${row.groups}: ${fit.tauInf} vs ${row.tau_inf}`)
    checked++
  }
  assert.ok(checked > 40)
  assert.equal(sweepRow(file, 'previous', 'count', 17, file.min_n, 'log')?.groups, 17)
  assert.equal(sweepRow(file, 'previous', 'count', 17, 100, 'log'), undefined)
})

test('a histogram median interpolates in log within its bin, and leaving out a group refits', () => {
  const bins = { lower: 1, ratio: 2, bins: 3 }
  assert.ok(Math.abs(histMedian([0, 2, 0], bins) - 2 ** 1.5) < 1e-12)
  const groups = [1, 2, 4, 8, 16, 32, 64].map((L) => ({ lower: 0, upper: 0, n: 200, loading: L, tau: (0.04 * L) / (1 + (0.04 * L) / 2) }))
  const fit = fitCeiling(groups, 'log')
  assert.ok(Math.abs(fit.tauInf - 2) < 0.01 && Math.abs(fit.a - 0.04) < 0.001)
  assert.ok(dropOne(groups).every((d) => Math.abs(d.tauInf - 2) < 0.05))
})
