import { test, expect } from '@playwright/test'

const panel = (id: string) => `[data-chart-key="${id}"]`
test('native continuum results share sidebar signal/window controls and survive a copied URL', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto('/#tab=baseline&net=focus')
  const sidebar = page.locator('.page-sidebar-section', { hasText: 'VIBES signal check' })
  await expect(page.locator(panel('vibes-validation-scope'))).toHaveCount(0)
  for (const id of ['vibes-validation-recovery', 'vibes-validation-errors', 'vibes-validation-traces', 'vibes-validation-blanks']) {
    await expect(page.locator(panel(id)).locator('svg[font-family]').first()).toBeVisible()
  }
  await expect(page.locator(panel('vibes-validation-recovery')).getByText('Line: median across independent parent blanks · band: interquartile range')).toBeVisible()
  await expect(page.locator(panel('vibes-validation-traces')).getByText(/One declared example/)).toBeVisible()
  await page.getByRole('button', { name: 'Samples', exact: false }).first().click()
  await sidebar.getByRole('radio', { name: 'Broad hump', exact: true }).click()
  expect(new URL(page.url()).searchParams.get('vibes-validation.signal')).toBe('Broad hump')
  await sidebar.getByRole('radio', { name: 'Full measured grid', exact: true }).click()
  await expect(page.locator(panel('vibes-validation-traces'))).toContainText('500')
  await page.reload()
  await expect(sidebar.getByRole('radio', { name: 'Broad hump', exact: true })).toHaveAttribute('aria-checked', 'true')
  await expect(sidebar.getByRole('radio', { name: 'Full measured grid', exact: true })).toHaveAttribute('aria-checked', 'true')
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
