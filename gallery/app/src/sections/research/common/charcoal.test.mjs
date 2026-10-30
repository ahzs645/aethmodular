// The char notebooks' per-filter summaries (charcoal.ts), against the tables the notebooks saved in the same export
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { byGroup, dryOf, galleryId, headline, joinScores, mannWhitney, monthlyRate } from './charcoal.ts'

const path = new URL('../../../../public/data/research/charcoal/scores.json', import.meta.url)
const data = fs.existsSync(path) ? JSON.parse(fs.readFileSync(path, 'utf8')) : null
const rowsOf = (t) => (t ? Object.values(t)[0].map((_, i) => Object.fromEntries(Object.keys(t).map((k) => [k, t[k][i]]))) : [])
const scores = rowsOf(data?.tables.per_filter)
const table = (name) => rowsOf(data?.tables[name])
const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`)
const skip = scores.length ? false : 'the export has no per-filter table (exported without the char outputs)'
const nb = () => joinScores(scores, [], 'notebook', () => null).scored

test('gallery ids drop the scan suffix', () => {
  assert.equal(galleryId('ETAD-0017-1'), 'ETAD-0017')
  assert.equal(galleryId(' ETAD-0249-12 '), 'ETAD-0249')
  assert.equal(galleryId('ETAD-0001'), 'ETAD-0001')
})

test('Mann–Whitney U and p as scipy computes them (ties, continuity correction)', () => {
  // scipy.stats.mannwhitneyu([1,2,2,3,5,8,9], [2,4,4,6,7,10,11,12]) → U 15, p 0.14619724467895154
  const r = mannWhitney([1, 2, 2, 3, 5, 8, 9], [2, 4, 4, 6, 7, 10, 11, 12])
  assert.equal(r.u, 15)
  near(r.p, 0.14619724467895154, 1e-9, 'p')
  // scipy's asymptotic p (its 'auto' would go exact for two tie-free samples this small; the pages' are larger)
  near(mannWhitney([1, 2, 3], [4, 5, 6, 7]).p, 0.05182992721790968, 1e-9, 'separated samples')
})

test('the join: view keeps gallery rows only, plus adds the rest with the reader\'s season, notebook keeps everything', { skip }, () => {
  const rows = [{ id: galleryId(scores[0].FilterId), date: scores[0].date, month: 12, season: 'Bega (Oct-Jan, dry)', 'HIPS Fabs': 1, 'EC (FTIR)': 2 }]
  const view = joinScores(scores, rows, 'view', () => 'X')
  assert.equal(view.scored.length, 1)
  assert.equal(view.outside, scores.length - 1)
  assert.equal(view.scored[0].fabs, 1)
  assert.equal(view.scored[0].season, 'Bega (Oct-Jan, dry)')
  const plus = joinScores(scores, rows, 'plus', () => 'X')
  assert.equal(plus.scored.length, scores.length)
  assert.equal(plus.scored.filter((s) => s.season === 'X').length, scores.length - 1)
  assert.equal(nb().length, scores.length)
})

test('char_05 by season and char_07 headline are reproduced from the per-filter scores', { skip }, () => {
  const all = nb()
  const seasons = byGroup(all, (s) => s.season, ['Dry Season', 'Belg Rainy Season', 'Kiremt Rainy Season'])
  for (const t of table('by_season_05')) {
    const g = seasons.find((s) => s.group === t.season)
    assert.equal(g.n, t.filters)
    near(g.rawMedian, t['best r vs raw feedstock'], 6e-4, `${t.season} raw`)
    near(g.charMedian, t['best r vs char'], 6e-4, `${t.season} char`)
    near(g.rawWins, t['share better matched by raw'], 6e-4, `${t.season} share`)
  }
  const dry = dryOf(all)
  const subsets = {
    'all filters (char_05)': all,
    'dry season excluded (primary)': all.filter((s) => !dry(s)),
    'char_06 anomalies excluded': all.filter((s) => !s.low),
  }
  for (const t of table('headline_07')) {
    const set = subsets[t.subset]
    if (!set) continue
    const h = headline(set)
    assert.equal(h.n, t.filters, t.subset)
    near(h.rawMedian, t['median best r — raw'], 6e-4, `${t.subset} raw`)
    near(h.charMedian, t['median best r — char'], 6e-4, `${t.subset} char`)
    near(h.rawWins, t['share raw > char'], 6e-4, `${t.subset} share`)
    near(h.deltaMedian, t['median delta'], 6e-4, `${t.subset} delta`)
  }
})

test('char_06: monthly rate, mixture counts, lot controls and the loading check', { skip }, () => {
  const all = nb(), dry = dryOf(all)
  const monthly = monthlyRate(all)
  for (const t of table('monthly_06')) {
    const m = monthly[t.month - 1]
    assert.equal(m.n, t.n, `month ${t.month}`)
    near(m.rate, t.rate, 6e-4, `month ${t.month}`)
  }
  for (const t of table('mixture_vs_season')) assert.equal(all.filter((s) => s.season === t.season && s.low).length, t['low-similarity'], t.season)
  // within lot 251, dry vs non-dry on delta: U=4284, p=9.42e-12 (printed)
  const lot251 = all.filter((s) => s.lot === 251)
  const u1 = mannWhitney(lot251.filter(dry).map((s) => s.delta), lot251.filter((s) => !dry(s)).map((s) => s.delta))
  assert.equal(u1.u, 4284)
  near(Math.log10(u1.p), Math.log10(9.42e-12), 0.01, 'lot 251 p')
  // within the dry season, lot 248 vs 251: U=634, p=0.886
  const dryLots = all.filter((s) => dry(s) && s.lot !== null)
  const u2 = mannWhitney(dryLots.filter((s) => s.lot === 248).map((s) => s.delta), dryLots.filter((s) => s.lot === 251).map((s) => s.delta))
  assert.equal(u2.u, 634)
  near(u2.p, 0.886, 6e-4, 'dry lots p')
  for (const t of table('lot_within_dry')) near(headline(dryLots.filter((s) => s.lot === t.LotId)).charMedian, t.median_r_char, 6e-4, `lot ${t.LotId}`)
  // within the dry season, low-similarity vs normal on Fabs and EC (filters with both)
  const model = all.filter((s) => dry(s) && s.fabs !== null && s.ec !== null)
  for (const t of table('loading_within_dry')) {
    const key = t.measure === 'Fabs' ? 'fabs' : 'ec'
    const r = mannWhitney(model.filter((s) => s.low).map((s) => s[key]), model.filter((s) => !s.low).map((s) => s[key]))
    near(r.p, t['Mann–Whitney p'], 6e-4, `${t.measure} p`)
  }
})
