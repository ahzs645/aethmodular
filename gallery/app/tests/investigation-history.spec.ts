import { test, expect } from '@playwright/test'

test('timeline searches findings, preserves selection in a shared URL and opens the mapped evidence graphs', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', e => errors.push(e.message))
  await page.goto('/#tab=investigations')
  await expect(page.getByRole('heading', { name: 'Research timeline' })).toBeVisible()
  await page.getByRole('searchbox', { name: 'Search experiments' }).fill('ftir_43')
  await page.getByRole('button', { name: /ftir_43.*FTIR/i }).click()
  const detail = page.getByRole('complementary', { name: 'Selected experiment' })
  await expect(detail).toContainText('includes zero')
  await expect(detail.getByRole('button', { name: 'Residual learners ↗' })).toBeVisible()
  await page.reload()
  await expect(page.getByRole('searchbox', { name: 'Search experiments' })).toHaveValue('ftir_43')
  await expect(detail).toContainText('ftir_43')
  await detail.getByRole('button', { name: 'Residual learners ↗' }).click()
  await expect(page).toHaveURL(/tab=ml/)
  await expect(page).toHaveURL(/aep=residual/)
  await expect(page.locator('[data-chart-key="ml-residual-null-draws"]')).toBeVisible()
  expect(errors).toEqual([])
})

test('connection navigation crosses filters and exports the displayed Mermaid graph', async ({ page }) => {
  await page.goto('/?investigations.record=ftir_13&investigations.family=airspec&investigations.view=connections#tab=investigations')
  const graph = page.getByRole('region', { name: 'Experiment connections' })
  await expect(graph.getByRole('group', { name: 'Research connections for ftir_13' })).toBeVisible()
  await graph.getByRole('button', { name: /ftir_11/ }).click()
  await expect(page.locator('.history-detail-kicker')).toContainText('ftir_11')
  await expect(page.getByRole('searchbox', { name: 'Search experiments' })).toHaveValue('')
  await graph.locator('summary', { hasText: 'Mermaid diagram' }).click()
  await expect(graph.locator('pre')).toContainText('Data dependency')
  const download = page.waitForEvent('download')
  await graph.getByRole('button', { name: 'Download Mermaid' }).click()
  expect((await download).suggestedFilename()).toBe('investigation-ftir_11.mmd')
  await page.reload()
  await expect(graph.getByRole('group', { name: 'Research connections for ftir_11' })).toBeVisible()
})

test('empty results, isolated records, original area cards and mobile layout remain usable', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/#tab=investigations')
  await page.getByRole('searchbox', { name: 'Search experiments' }).fill('no-such-investigation-xyz')
  await expect(page.getByText('No research records match these filters.')).toBeVisible()
  await page.getByRole('button', { name: 'Show all records' }).click()
  await page.goto('/?investigations.record=pr15_check_20261103&investigations.view=connections#tab=investigations')
  await expect(page.getByText('No documented connections for this record yet.', { exact: false })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.getByRole('radio', { name: 'Areas', exact: true }).click()
  await expect(page.getByRole('button', { name: /^Research diagnostics \d+ pages$/ })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
})

test('availability follows the active host and recovers after restoring an export without regenerating history', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 })
  const exportUrl = '**/data/experimental/continuum_baseline.json'
  let restored = false
  let requests = 0
  await page.route(exportUrl, async route => {
    requests++
    if (restored) await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ schema_version: 1, generated: '2026-11-04T03:44:59Z' }) })
    else await route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><html><body>Static-host fallback</body></html>' })
  })
  await page.goto('/?investigations.record=transfer_diagnosis_20261102#tab=investigations')
  const detail = page.getByRole('complementary', { name: 'Selected experiment' })
  const availability = detail.locator('.history-export-availability')
  await expect(availability.getByRole('status')).toHaveText('Could not load 1 of 1 primary exports in this build.')
  await availability.locator('summary').click()
  await expect(availability).toContainText('the data export is not available on this host')
  await expect(detail.locator('.history-badge').first()).toHaveText('Saved notebook outputs')

  restored = true
  await availability.getByRole('button', { name: 'Recheck exports for Continuum and the baseline' }).click()
  await expect(availability.getByRole('status')).toHaveText('Primary export loaded in this build.')
  await expect(availability).toContainText('generated 2026-11-04T03:44:59Z')
  await expect(detail.locator('.history-badge').first()).toHaveText('Saved notebook outputs')
  expect(requests).toBe(2)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await availability.scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('history-availability-restored-phone.png'), fullPage: false })

  await page.reload()
  await expect(availability.getByRole('status')).toHaveText('Primary export loaded in this build.')
  await page.goto('/?investigations.record=ann_followup_20261009#tab=investigations')
  await expect(detail.getByText('Open this page to check its data availability.', { exact: true })).toBeVisible()
})
