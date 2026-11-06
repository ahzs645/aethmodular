import { test, expect } from '@playwright/test'
import { readFileSync } from 'node:fs'

const panel = (id: string) => `[data-chart-key="${id}"]`
// The counts of the local run (baseline_validation.json, gitignored and rebuilt per machine): a fit can fail on one
// machine and not another, and the page has to report whichever happened. docs/vibes-gallery-continuum-check.md
const evidence = () => JSON.parse(readFileSync('public/data/baseline_validation.json', 'utf8')).tables

test('native continuum results share signal/window controls and survive a copied URL', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto('/#tab=baseline&net=focus')
  const scope = page.locator(panel('vibes-validation-scope'))
  await expect(scope).toContainText('9 independent held-out Addis blanks')
  const { fits, run } = evidence()
  const passed = (fits.success as unknown[]).filter((v) => v === true || v === 1).length
  await expect(scope).toContainText(`${passed}/${fits.success.length} VIBES optimizer and finite-output checks succeeded`)
  await expect(scope).toContainText(`${run.map_optimal[0]}/${run.map_replayed[0]} reported optimal`)
  expect(run.map_optimal[0]).toBe(run.map_replayed[0])
  await expect(scope).toContainText('32,286-fit/G4 run awaits its own results export')
  for (const id of ['vibes-validation-recovery', 'vibes-validation-errors', 'vibes-validation-traces', 'vibes-validation-blanks']) {
    await expect(page.locator(panel(id)).locator('svg[font-family]').first()).toBeVisible()
  }
  await expect(page.locator(panel('vibes-validation-recovery')).getByText('Line: median across independent parent blanks · band: interquartile range')).toBeVisible()
  await expect(page.locator(panel('vibes-validation-traces')).getByText(/One declared example/)).toBeVisible()
  await scope.getByRole('radio', { name: 'Broad hump', exact: true }).click()
  expect(new URL(page.url()).searchParams.get('vibes-validation.signal')).toBe('Broad hump')
  await scope.getByRole('radio', { name: 'Full measured grid', exact: true }).click()
  await expect(page.locator(panel('vibes-validation-traces'))).toContainText('500')
  await page.reload()
  await expect(scope.getByRole('radio', { name: 'Broad hump', exact: true })).toHaveAttribute('aria-checked', 'true')
  await expect(scope.getByRole('radio', { name: 'Full measured grid', exact: true })).toHaveAttribute('aria-checked', 'true')
  await page.setViewportSize({ width: 390, height: 844 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  expect(errors).toEqual([])
})

test('missing older benchmark files do not hide the independent VIBES check', async ({ page }) => {
  await page.route('**/baseline_paired_r70.json', (route) => route.fulfill({ status: 404, body: 'missing' }))
  await page.goto('/#tab=baseline&net=focus')
  await expect(page.locator(panel('vibes-validation-traces')).locator('svg[font-family]').first()).toBeVisible()
  await expect(page.getByText('Saved baseline benchmark', { exact: true })).toBeVisible()
  await expect(page.locator(panel('vibes-validation-reference'))).toContainText('before recovery')
})
