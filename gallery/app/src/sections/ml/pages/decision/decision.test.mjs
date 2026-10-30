import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { crossCheck, flagTable, summarise } from './stats.ts'

const close = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`)

test('a site fit against Fabs/MAC: slope and ratios scale by MAC/10, the intercept stays', () => {
  let seed = 7
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
  const filters = Array.from({ length: 40 }, (_, i) => {
    const fabs = 10 + 60 * rnd()
    return { id: `ABCD-${String(i).padStart(4, '0')}`, code: 'ABCD', season: i % 2 ? 'Wet' : 'Dry', flagSpectrum: rnd(), flagHips: rnd(), outside: i % 4 === 0,
      ecModel: 0.05 * fabs + rnd(), ecPublished: 0.12 * fabs - 0.8 + 0.3 * rnd(), fabs }
  })
  const at10 = summarise('ABCD', '', filters, 10, 3.34), at6 = summarise('ABCD', '', filters, 6, 3.34)
  close(at6.slope, at10.slope * 0.6, 1e-9, 'slope')
  close(at6.intercept, at10.intercept, 1e-9, 'intercept')
  close(at6.publishedRatio, at10.publishedRatio * 0.6, 1e-9, 'median ratio')
  assert.equal(at10.outsideShare, 10 / 40)
  // site-seasons below 10 filters are left out, as in the notebook
  assert.equal(flagTable(filters, 'site-season', 10, 3.34).length, 2)
  assert.equal(flagTable(filters, 'site-season', 10, 3.34, 21).length, 0)
})

// the committed export: rebuilt from its per-filter table at MAC 10, the notebook's own site tables
const path = new URL('../../../../../public/data/ml/decision.json', import.meta.url)
const data = fs.existsSync(path) ? JSON.parse(fs.readFileSync(path, 'utf8')) : null
const rows = (t) => Object.keys(t).length ? t[Object.keys(t)[0]].map((_, i) => Object.fromEntries(Object.keys(t).map((c) => [c, t[c][i]]))) : []

test('the per-filter export reproduces the workbook\'s §6 site and site-season tables', { skip: !data?.tables.network_filters && 'decision.json was exported without the Drive' }, () => {
  const seasons = data.tables.seasons.season
  const filters = rows(data.tables.network_filters).map((r) => ({ id: r.filter_id, code: r.code, season: seasons[r.season], flagSpectrum: r.flag_spectrum,
    flagHips: r.flag_hips, outside: !!r.outside, ecModel: r.ec_model, ecPublished: r.ec_published, fabs: r.fabs }))
  const lambda10 = data.tables.settings.lambda_addis[0]
  for (const [by, table] of [['site', 'sites'], ['site-season', 'site_seasons']]) {
    const mine = flagTable(filters, by, 10, lambda10)
    const nb = rows(data.tables[table])
    assert.equal(mine.length, nb.length, `${table} rows`)
    for (const r of nb) {
      const m = mine.find((g) => g.code === r.code && (by === 'site' || g.season === (r.season ?? '')))
      assert.ok(m, `${r.code} ${r.season ?? ''}`)
      assert.equal(m.n, r.n)
      close(m.meanFlagHips, r['mean flag (+ HIPS)'], 1e-3, 'mean flag')
      close(m.flaggedShare, r['flagged share (+ HIPS)'], 1e-4, 'flagged share')
      if (r['published vs HIPS slope'] != null) {
        close(m.slope, r['published vs HIPS slope'], 2e-3 * Math.abs(r['published vs HIPS slope']), 'Deming slope')
        close(m.interceptRatio, r['intercept ÷ mean'], 5e-3, 'intercept ÷ mean')
      }
      if (r['published ÷ HIPS (median)'] != null) close(m.publishedRatio, r['published ÷ HIPS (median)'], 1e-3, 'published ÷ HIPS')
    }
  }
  // the notebook printed Spearman −0.33 and +0.25 over 54 site-seasons
  const cc = crossCheck(flagTable(filters, 'site-season', 10, lambda10))
  assert.equal(cc.n, 54)
  close(cc.intercept, -0.33, 0.005, 'ρ intercept')
  close(cc.ratio, 0.25, 0.01, 'ρ ratio')
})
