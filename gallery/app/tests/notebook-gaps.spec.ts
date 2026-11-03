import { test, expect } from '@playwright/test'
import { readFileSync } from 'node:fs'

const panel = (id: string) => `[data-chart-key="${id}"]`

test('processing comparison renders the full record and works on a phone', async ({ page }) => {
  const data = JSON.parse(readFileSync('public/data/aethalometer/quality/processing.json', 'utf8'))
  expect(data.tables.settings.minutes[0]).toBeGreaterThan(1_000_000)
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto('/?aq.topic=smoothing#tab=aethalometer&aep=quality&net=focus')
  await expect(page.locator(panel('processing-trace')).locator('svg[font-family]').first()).toBeVisible()
  await expect(page.locator(panel('processing-windows')).locator('svg[font-family]').first()).toBeVisible()
  await expect(page.locator(panel('processing-lag')).locator('svg[font-family]').first()).toBeVisible()
  await expect(page.locator(panel('processing-scope'))).toContainText('Jacros')
  await expect(page.locator(panel('processing-daily')).locator('svg[font-family]').first()).toBeVisible()
  await expect(page.locator(panel('processing-agreement')).locator('svg[font-family]').first()).toBeVisible()
  await page.setViewportSize({ width: 390, height: 844 })
  await expect(page.locator(panel('processing-trace'))).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  expect(errors).toEqual([])
})

test('downloaded native charcoal and EC/OC profiles render without fixtures', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto('/#tab=ml&aep=charcoal&net=focus')
  for (const id of ['mlc-pca', 'mlc-typology-pca', 'mlc-distance-cdf', 'mlc-envelope', 'mlc-escape']) {
    await expect(page.locator(panel(id)).locator('svg[font-family]').first()).toBeVisible()
  }
  await expect(page.locator(panel('mlc-envelope'))).toContainText('line: mean · band: 5th–95th percentiles')
  await expect(page.locator(panel('mlc-envelope')).getByText('line: mean · band: 5th–95th percentiles')).toBeVisible()
  await page.goto('/?mlc-envelope.representation=Second%20derivative&mlc-envelope.region=CH#tab=ml&aep=charcoal&net=focus')
  await expect(page.locator(panel('mlc-envelope'))).toContainText('2900')
  await page.goto('/?ml-model-form-vip-profiles.set=ec-oc#tab=ml&aep=model-form&net=focus')
  await expect(page.locator(panel('ml-model-form-vip-profiles')).locator('svg[font-family]').first()).toBeVisible()
  expect(errors).toEqual([])
})

test('native CV, domain diagnostics and uncertainty draws render without fixtures', async ({ page }) => {
  await page.goto('/#tab=ml&aep=components&net=focus')
  await expect(page.locator(panel('ml-components-2x2-curves')).locator('svg[font-family]').first()).toBeVisible()
  await page.goto('/#tab=ml&aep=protocols&net=focus')
  for (const id of ['ml-protocols-domain-locked-distance', 'ml-protocols-bootstrap-draws', 'ml-protocols-bootstrap-modes']) {
    await expect(page.locator(panel(id)).locator('svg[font-family]').first()).toBeVisible()
  }
  await page.goto('/#tab=ml&aep=residual&net=focus')
  await expect(page.locator(panel('ml-residual-null-draws')).locator('svg[font-family]').first()).toBeVisible()
})

test('per-filter diagnostics render and MAC choices survive a copied URL', async ({ page }) => {
  await page.goto('/#tab=ml&aep=protocols&net=focus&mac=5')
  await expect(page.locator(panel('ml-protocols-current-loading')).locator('svg[font-family]').first()).toBeVisible()
  await expect(page.locator(panel('ml-protocols-current-loading'))).toContainText('Fabs/5')
  await page.reload()
  await expect(page.locator(panel('ml-protocols-current-loading'))).toContainText('Fabs/5')
})

test('charcoal views consume saved arrays with negative PCA scores and percentile bands', async ({ page }) => {
  // Deliberately tiny synthetic interface fixture, never written into research data.
  const data = JSON.parse(readFileSync('public/data/ml/charcoal.json', 'utf8'))
  Object.assign(data.tables, {
    pca_scores: [{ PC1: -2, PC2: -1, population: 'Reference' }, { PC1: 2, PC2: 1, population: 'Reference' }],
    typology_scores: [{ PC1: -2, PC2: -1, source: 'Reference', analysis: 'Pass A' }],
    mahalanobis_per_filter: [{ distance: 1, population: 'Reference' }, { distance: 2, population: 'Reference' }],
    envelope_spectra: [{ wn: 2000, mean: 1, p05: .1, p95: 2, population: 'Reference', representation: 'Raw' },
      { wn: 2100, mean: 2, p05: .2, p95: 3, population: 'Reference', representation: 'Raw' }],
    per_filter_escape: [{ escape: .1, season_label: 'Dry Season' }],
  })
  await page.route('**/data/ml/charcoal.json', (route) => route.fulfill({ json: data }))
  await page.goto('/#tab=ml&aep=charcoal&net=focus')
  await expect(page.locator(panel('mlc-pca')).locator('svg[font-family]').first()).toBeVisible()
  await expect(page.locator(panel('mlc-pca'))).toContainText('-2')
  await expect(page.locator(panel('mlc-distance-cdf')).locator('svg[font-family]').first()).toBeVisible()
  await expect(page.locator(panel('mlc-envelope'))).toContainText('line: mean · band: 5th–95th percentiles')
})

test('full CV curves keep grouping and fold-count series separate', async ({ page }) => {
  const data = JSON.parse(readFileSync('public/data/ml/components.json', 'utf8'))
  data.tables.cv_2x2_curves = ['interleaved · 5-fold', 'interleaved · 10-fold', 'site-grouped · 5-fold', 'site-grouped · 10-fold']
    .flatMap((scheme) => [1, 2].map((k) => ({ n_components: k, pct_rmsecv: 100 / k, cohort: 'Fixture cohort', scheme })))
  await page.route('**/data/ml/components.json', (route) => route.fulfill({ json: data }))
  await page.goto('/#tab=ml&aep=components&net=focus')
  const chart = page.locator(panel('ml-components-2x2-curves'))
  await expect(chart.locator('svg[font-family]').first()).toBeVisible()
  for (const s of ['interleaved · 5-fold', 'interleaved · 10-fold', 'site-grouped · 5-fold', 'site-grouped · 10-fold']) await expect(chart.locator(`[data-export-series="${s}"]`)).toBeVisible()
})

test('EC/OC VIP remains selectable when its profiles are unavailable', async ({ page }) => {
  const data = JSON.parse(readFileSync('public/data/ml/model-form.json', 'utf8'))
  delete data.tables.vip_ec_oc
  await page.route('**/data/ml/model-form.json', (route) => route.fulfill({ json: data }))
  await page.goto('/?ml-model-form-vip-profiles.set=ec-oc#tab=ml&aep=model-form&net=focus')
  const chart = page.locator(panel('ml-model-form-vip-profiles'))
  await expect(chart).toContainText('The full VIP profiles for this model pair are not available')
  await page.goto('/?ml-model-form-vip-profiles.set=hips#tab=ml&aep=model-form&net=focus')
  await expect(chart.locator('svg[font-family]').first()).toBeVisible()
})
