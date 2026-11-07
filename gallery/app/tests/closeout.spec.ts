import { test, expect } from '@playwright/test'
import { readFile } from 'node:fs/promises'

// Acceptance against the pinned November publication snapshot. These are UI
// checks, not a reproduction of the underlying scientific calculations.
test('legacy EPO evidence stays withheld and reference labels survive phone width with notes off', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/?view=single&chart=xc-source#tab=experimental&xp=continuum')
  const source = page.locator('#xc-source')
  await expect(source.locator('.chart-evidence')).toContainText('EPO withheld', { timeout: 45000 })
  await expect(source.locator('.chart-evidence')).toContainText('TOR EC reference')
  await expect(page.locator('html')).toHaveAttribute('data-explain', '0')
  await expect(source.locator('tbody tr').filter({ hasText: 'EPO' })).toContainText('Withheld')
  const width = await page.evaluate(() => ({ viewport: innerWidth, body: document.documentElement.scrollWidth }))
  expect(width.body).toBeLessThanOrEqual(width.viewport + 1)
  // A page can fit the viewport while its chart text is clipped or its table
  // headings wrap one character per line. Verify readable content as well.
  const clippedText = await source.locator('svg').evaluateAll(svgs => svgs.flatMap(svg => {
    const box = svg.getBoundingClientRect()
    return Array.from(svg.querySelectorAll('text')).filter(text => {
      const label = text.getBoundingClientRect()
      return label.left < box.left - 1 || label.right > box.right + 1
    }).map(text => text.textContent)
  }))
  expect(clippedText).toEqual([])
  const headerHeights = await source.locator('th').evaluateAll(headers => headers.map(header => header.getBoundingClientRect().height))
  expect(Math.max(...headerHeights)).toBeLessThanOrEqual(48)
  const tableRegion = source.getByRole('region', { name: 'Candidate counts and evidence' })
  expect(await tableRegion.evaluate(region => region.scrollWidth > region.clientWidth)).toBe(true)
  await page.screenshot({ path: testInfo.outputPath('source-phone.png'), fullPage: true })
  await tableRegion.evaluate(region => { region.scrollLeft = region.scrollWidth })
  await expect(source.locator('th').filter({ hasText: 'Evidence' })).toBeInViewport()
  await page.screenshot({ path: testInfo.outputPath('source-phone-evidence.png'), fullPage: true })
  await page.goto('/?view=single&chart=m1028-deficit#tab=meeting&meeting=meeting-20261028&net=focus')
  await expect(page.getByText(/FTIR residual from seasonal HIPS fit/).first()).toBeVisible({ timeout: 45000 })
})

test('six-figure review reopens in a fresh context and completes a seven-page PDF download', async ({ page, browser, context }, testInfo) => {
  test.setTimeout(240000)
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/#tab=experimental&xp=continuum')
  await page.getByRole('button', { name: 'Build six-figure review section' }).click({ timeout: 45000 })
  await expect(page.locator('.sec-card')).toHaveCount(6)
  await page.getByRole('button', { name: 'Copy link', exact: true }).click()
  await expect(page.getByRole('status').filter({ hasText: 'Link copied' })).toBeVisible()
  const link = await page.evaluate(() => navigator.clipboard.readText())
  expect(link).toContain('#tab=custom&sec=1.')

  const readerContext = await browser.newContext({ viewport: { width: 1440, height: 1000 } })
  const reader = await readerContext.newPage()
  reader.on('pageerror', error => errors.push(error.message))
  await reader.goto(link)
  await expect(reader.getByRole('heading', { name: 'FTIR transfer: existing-data closeout review' })).toBeVisible({ timeout: 45000 })
  await expect(reader.locator('.sec-card')).toHaveCount(6)
  await expect(reader.getByText(/not a frozen data package/)).toBeVisible()
  const cards = reader.locator('.sec-card')
  for (let i = 0; i < 6; i++) {
    await cards.nth(i).scrollIntoViewIfNeeded()
    await expect(cards.nth(i).locator('iframe.shown')).toBeVisible({ timeout: 45000 })
    await expect(cards.nth(i).frameLocator('iframe').locator('[data-single-active] .chart-evidence')).toBeVisible()
  }
  const downloadEvent = reader.waitForEvent('download', { timeout: 180000 })
  await reader.getByRole('button', { name: 'PDF', exact: true }).click()
  const download = await downloadEvent
  const destination = testInfo.outputPath('closeout-review.pdf')
  await download.saveAs(destination)
  await expect(reader.getByRole('status').filter({ hasText: /^Saved$/ })).toBeVisible()
  const pdf = await readFile(destination)
  expect(pdf.subarray(0, 5).toString()).toBe('%PDF-')
  expect(pdf.length).toBeGreaterThan(10000)
  expect(pdf.toString('latin1').match(/\/Type \/Page\b/g)).toHaveLength(7)
  await reader.getByRole('button', { name: 'Save a copy to change it' }).click()
  await expect(reader.getByRole('textbox', { name: 'Section name' })).toHaveValue('FTIR transfer: existing-data closeout review')
  await reader.reload()
  await expect(reader.locator('.sec-card')).toHaveCount(6)
  await reader.screenshot({ path: testInfo.outputPath('review-section.png'), fullPage: false })
  expect(errors).toEqual([])
  await readerContext.close()
})
