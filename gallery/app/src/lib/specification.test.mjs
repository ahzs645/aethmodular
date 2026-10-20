import test from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { choiceFilterDimensions, choiceFilterSets, matchesChoiceFilters, DEFAULT_HIGHLIGHT_LIMITS as limits, highlightMatches, selectionPanels } from './specification.ts'

test('highlight rules distinguish saved guardrails from partial criteria', () => {
  const good = { intercept: 0.5, slope: 0.7, cv: 0.85, guardrails: false }
  assert.equal(highlightMatches(good, 'guardrails', limits), false)
  assert.equal(highlightMatches(good, 'custom', limits), true)
  assert.equal(highlightMatches({ ...good, intercept: -0.5, slope: 1.3 }, 'custom', limits), true)
  assert.equal(highlightMatches({ ...good, intercept: -0.5001 }, 'custom', limits), false)
  assert.equal(highlightMatches({ ...good, intercept: -9 }, 'cv-slope', limits), true)
  assert.equal(highlightMatches({ ...good, cv: null }, 'cv-slope', limits), false)
  assert.equal(highlightMatches({ ...good, cv: null }, 'intercept', limits), true)
  assert.equal(highlightMatches({ ...good, slope: null }, 'custom', limits), false)
  assert.equal(highlightMatches({ ...good, intercept: NaN }, 'intercept', limits), false)
  assert.equal(highlightMatches({ ...good, intercept: 0.8 }, 'custom', { ...limits, intercept: 1 }), true)
})

test('separate selection curves partition the filtered population without reordering or duplicating it', () => {
  const shown = [0, 2, 3, 5]
  const panels = selectionPanels(shown, [0, 1, 1, 0, 2, 1], ['airspec', 'vibes', 'vibes-full'])
  assert.deepEqual(panels, [{ key: 'airspec', indices: [0, 3] }, { key: 'vibes', indices: [2, 5] }])
  assert.deepEqual(panels.flatMap(p => p.indices).sort((a,b) => a-b), shown)
  assert.deepEqual(selectionPanels([], [], ['airspec']), [])
})

for (const source of ['spec_curve.json', 'meeting-r70/spec_curve_20260917.json', 'meeting/spec_curve_20260917.json']) {
  const url = new URL(`../../public/data/${source}`, import.meta.url)
  test(`${source}: three cohort choices remain visible, while CV and factor choices narrow their union`, { skip: !existsSync(url) }, () => {
    const file = JSON.parse(readFileSync(url, 'utf8'))
    const dimensions = choiceFilterDimensions(file)
    const cohort = dimensions.flatMap((d, c) => d === 'co' ? [c] : []).slice(0, 3)
    assert.equal(cohort.length, 3)
    const shown = required => {
      const groups = choiceFilterSets(new Set(required), dimensions)
      return file.intercept.flatMap((_, i) => matchesChoiceFilters(file.masks, i, groups) ? [i] : [])
    }
    const union = [...new Set(cohort.flatMap(c => shown([c])))].sort((a, b) => a - b)
    assert.ok(union.length > 0)
    assert.deepEqual(shown(cohort), union)
    const cv = file.choices.indexOf('Site-grouped cross-validation')
    const factors = file.choices.indexOf('9 or fewer PLS factors')
    assert.equal(dimensions[cv], 'mo')
    assert.equal(dimensions[factors], 'k')
    const narrowed = union.filter(i => file.masks[cv][i] === '1' && file.masks[factors][i] === '1')
    assert.ok(narrowed.length > 0 && narrowed.length < union.length)
    assert.deepEqual(shown([...cohort, cv, factors]), narrowed)
    const otherCV = file.choices.indexOf('Interleaved cross-validation')
    assert.deepEqual(shown([...cohort, cv, otherCV]), union)
    const highFactors = file.choices.indexOf('15 or more PLS factors')
    assert.deepEqual(shown([...cohort, factors, highFactors]), union.filter(i => file.k[i] <= 9 || file.k[i] >= 15))
    assert.equal(dimensions[file.choices.indexOf('Cohort under 600 filters')], 'cut')
    const all = file.choices.indexOf('All IMPROVE filters, or smoke (906)')
    if (all >= 0) assert.equal(dimensions[all], 'co')
    assert.equal(shown([]).length, file.n)
  })
}

test('two searches merge into one curve without losing or inventing a configuration', async () => {
  const { mergeSpecFiles } = await import('./specification.ts')
  const a = { n: 2, n_pass: 1, guardrails: 'g', choices: ['Raw spectra', 'Lowest OC/EC'], groups: [{ title: 'Spectral preprocessing', start: 0, end: 1 }, { title: 'Calibration cohort', start: 1, end: 2 }],
    masks: ['10', '01'], ok: '10', intercept: [0.5, -1], slope: [1, 2], heldout: [0.9, null], olsIntercept: [0.4, -0.9], olsSlope: [1, 2], cut: [300, null], k: [5, 9],
    categories: { co: ['ocec', 'pool'], sp: ['raw', 'airspec'], sel: ['raw'], mo: ['site_heldout'], lot: ['all'] }, co: [1, 0], sp: [0, 1], sel: [0, 0], mo: [0, 0], lot: [0, 0], extra: { rmse: [1, 2] } }
  const b = { n: 1, n_pass: 0, guardrails: 'g', choices: ['VIBES baseline', 'Lowest OC/EC'], groups: [{ title: 'Spectral preprocessing', start: 0, end: 1 }, { title: 'Calibration cohort', start: 1, end: 2 }],
    masks: ['1', '1'], ok: '0', intercept: [0], slope: [1.1], heldout: [0.8], cut: [500], k: [7],
    categories: { co: ['ocec'], sp: ['vibes-full-r70'], sel: ['vibes-full-r70'], mo: ['app'], lot: ['all'] }, co: [0], sp: [0], sel: [0], mo: [0], lot: [0], extra: { qres: [3] } }
  const m = mergeSpecFiles([{ file: a, name: 'A' }, { file: b, name: 'B' }])
  assert.equal(m.n, 3)
  assert.deepEqual(m.intercept, [-1, 0, 0.5], 'back in intercept order')
  assert.deepEqual(m.slope, [2, 1.1, 1])
  assert.equal(m.ok, '001')
  assert.deepEqual(m.categories.sp, ['raw', 'airspec', 'vibes-full-r70'])
  assert.deepEqual(m.sp.map((c) => m.categories.sp[c]), ['airspec', 'vibes-full-r70', 'raw'])
  assert.deepEqual(m.co.map((c) => m.categories.co[c]), ['ocec', 'ocec', 'pool'])
  assert.deepEqual(m.olsIntercept, [-0.9, null, 0.4], 'a column one search lacks is missing, not zero')
  assert.deepEqual(m.extra.qres, [null, 3, null])
  const row = (label) => m.masks[m.choices.indexOf(label)]
  assert.equal(row('Lowest OC/EC'), '110', 'a shared choice is one row')
  assert.equal(row('VIBES baseline'), '010')
  assert.equal(row('From the B search'), '010')
  assert.deepEqual(m.groups.map((g) => g.title), ['Spectral preprocessing', 'Calibration cohort', 'Search'])
  assert.equal(m.groups.at(-1).end, m.choices.length)
})
