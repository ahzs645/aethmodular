import { test, expect, type APIRequestContext, type Page, type TestInfo } from '@playwright/test'
import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { readFile } from 'node:fs/promises'

interface ContinuumSnapshot {
  schema_version: number
  generated: string
  settings: {
    source_evaluation?: { status?: string; protocol?: string; population?: { n_common?: number; n_sites?: number } }
    target_predictions?: { generated?: string }
    source_refresh?: { base_generated?: string }
  }
  scoreboard: { model: string[]; [key: string]: (string | number | null)[] }
}

const manifestPath = new URL('../../review/closeout_20261107/manifest.json', import.meta.url)
const sourceUrl = '/?view=single&chart=xc-source#tab=experimental&xp=continuum'
const savedDate = (payload: ContinuumSnapshot) => (payload.settings.target_predictions?.generated ?? payload.settings.source_refresh?.base_generated ?? payload.generated).split('T')[0]

async function legacyContinuum() {
  if (existsSync(manifestPath) && !process.env.CLOSEOUT_LEGACY_EXPORT) {
    throw new Error('Set CLOSEOUT_LEGACY_EXPORT to the preserved original continuum JSON when a corrected overlay is present.')
  }
  const bytes = await readFile(process.env.CLOSEOUT_LEGACY_EXPORT ?? new URL('../public/data/experimental/continuum_baseline.json', import.meta.url))
  const payload = JSON.parse(bytes.toString()) as ContinuumSnapshot
  expect(payload.settings.source_evaluation?.status).not.toBe('corrected')
  return { bytes, payload }
}

async function activeContinuum(request: APIRequestContext) {
  const response = await request.get('/data/experimental/continuum_baseline.json')
  expect(response.ok()).toBe(true)
  const bytes = await response.body()
  const payload = JSON.parse(bytes.toString()) as ContinuumSnapshot
  const corrected = existsSync(manifestPath)
  if (process.env.CLOSEOUT_OVERLAY_APPLIED != null) expect(corrected).toBe(process.env.CLOSEOUT_OVERLAY_APPLIED === '1')
  if (corrected) {
    const manifest = JSON.parse(await readFile(manifestPath, 'utf8'))
    expect(manifest.scientific_status).toBe('fold_local_source_verified')
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(manifest.overlay.sha256)
    expect(payload.schema_version).toBe(2)
    expect(payload.settings.source_evaluation?.status).toBe('corrected')
    expect(payload.settings.source_evaluation?.protocol).toBe('fixed_k_common_physical_filter_metrics_v1')
  } else {
    expect(payload.settings.source_evaluation?.status).not.toBe('corrected')
  }
  return { payload, corrected }
}

async function readableSourceOnPhone(page: Page, testInfo: TestInfo, name: string) {
  const source = page.locator('#xc-source')
  await expect(source.locator('.chart-evidence')).toContainText('TOR EC reference')
  await expect(page.locator('html')).toHaveAttribute('data-explain', '0')
  await page.evaluate(() => document.fonts.ready.then(() => undefined))
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
  const tableRegion = source.getByRole('region', { name: 'Candidate errors, counts and evidence' })
  expect(await tableRegion.evaluate(region => region.scrollWidth > region.clientWidth)).toBe(true)
  await page.screenshot({ path: testInfo.outputPath(`${name}.png`), fullPage: true })
  await tableRegion.evaluate(region => { region.scrollLeft = region.scrollWidth })
  await expect(source.locator('th').filter({ hasText: 'Evidence' })).toBeInViewport()
  await page.screenshot({ path: testInfo.outputPath(`${name}-evidence.png`), fullPage: true })
}

// The historical guard always sees the preserved original snapshot, including
// its old EPO scalar. The actual-app/PDF tests below do not use a route fixture.
test('legacy EPO evidence stays withheld and reference labels survive phone width with notes off', async ({ page }, testInfo) => {
  const { bytes, payload } = await legacyContinuum()
  const epo = payload.scoreboard.model.indexOf('epo')
  expect(epo).toBeGreaterThanOrEqual(0)
  expect(typeof (payload.scoreboard.improve_error?.[epo] ?? payload.scoreboard.improve_median_absolute_error?.[epo])).toBe('number')
  await page.route('**/data/experimental/continuum_baseline.json*', route => route.fulfill({ contentType: 'application/json', body: bytes }))
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto(sourceUrl)
  const source = page.locator('#xc-source')
  await expect(source.locator('.chart-evidence')).toContainText('EPO withheld', { timeout: 45000 })
  await expect(source.locator('tbody tr').filter({ hasText: 'EPO' })).toContainText('Withheld')
  await readableSourceOnPhone(page, testInfo, 'source-phone-legacy')
  await page.goto('/?view=single&chart=m1028-deficit#tab=meeting&meeting=meeting-20261028&net=focus')
  await expect(page.getByText(/FTIR residual from seasonal HIPS fit/).first()).toBeVisible({ timeout: 45000 })
})

test('the actual export shows verified source counts when available and retains saved target dates', async ({ page }, testInfo) => {
  const { payload, corrected } = await activeContinuum(page.request)
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto(sourceUrl)
  const source = page.locator('#xc-source')
  const evidence = source.locator('.chart-evidence')
  const epoRow = source.locator('tbody tr').filter({ hasText: 'EPO' })
  if (corrected) {
    const population = payload.settings.source_evaluation!.population!
    expect(Number.isInteger(population.n_common)).toBe(true)
    expect(population.n_common).toBeGreaterThan(0)
    expect(Number.isInteger(population.n_sites)).toBe(true)
    expect(population.n_sites).toBeGreaterThan(0)
    await expect(evidence).toContainText('corrected retrospective comparison', { timeout: 45000 })
    await expect(evidence).toContainText(`${population.n_common!.toLocaleString('en-US')} common physical filters across ${population.n_sites!.toLocaleString('en-US')} sites`)
    await expect(evidence).not.toContainText('EPO withheld')
    await expect(epoRow).toContainText('Verified source CV')
    const epo = payload.scoreboard.model.indexOf('epo')
    expect(epo).toBeGreaterThanOrEqual(0)
    expect(payload.scoreboard.evaluation_status[epo]).toBe('corrected')
    for (const [key, column] of [['improve_mae', 1], ['improve_rmse', 2]] as const) {
      const value = payload.scoreboard[key][epo]
      expect(typeof value).toBe('number')
      expect(Number.isFinite(value)).toBe(true)
      await expect(epoRow.locator('td').nth(column)).toHaveText(Number(value).toFixed(2))
    }
    for (const label of ['Median abs. error (µg/filter)', 'Mean signed bias (µg/filter)', 'P95 abs. error (µg/filter)']) {
      await expect(source.locator('svg').getByText(label, { exact: true })).toBeVisible()
    }
  } else {
    await expect(evidence).toContainText('EPO withheld', { timeout: 45000 })
    await expect(epoRow).toContainText('Withheld')
  }
  await readableSourceOnPhone(page, testInfo, 'source-phone-active')
  await page.goto('/?view=single&chart=xc-addis#tab=experimental&xp=continuum')
  await expect(page.locator('#xc-addis .chart-evidence')).toContainText(`Saved study export ${savedDate(payload)}`, { timeout: 45000 })
  await expect(page.locator('#xc-addis .chart-evidence')).toContainText('Exploratory target comparison')
})

test('six-figure review reopens in a fresh context and completes a seven-page PDF download', async ({ page, browser, context }, testInfo) => {
  test.setTimeout(240000)
  const { payload, corrected } = await activeContinuum(page.request)
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
  await expect(cards.nth(1).locator('.sec-note')).toContainText(corrected ? 'Verified fold-safe EPO source scores are included' : 'EPO source scores remain withheld')
  await expect(cards.nth(1).frameLocator('iframe').locator('[data-single-active] .chart-evidence')).toContainText(corrected ? 'corrected retrospective comparison' : 'EPO withheld')
  await expect(cards.nth(5).frameLocator('iframe').locator('[data-single-active] .chart-evidence')).toContainText(`Saved study export ${savedDate(payload)}`)
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
