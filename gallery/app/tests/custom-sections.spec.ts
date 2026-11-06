import { test, expect, type Page } from '@playwright/test'

const picked = async (page: Page, url: string, frame = 'section[data-frame] button.section-add', nth = 0) => {
  await page.goto(url)
  const add = page.locator(frame).nth(nth)
  await expect(add).toBeVisible({ timeout: 45000 })
  await add.click()
  await expect(add).toHaveAttribute('aria-pressed', 'true')
}
const shown = (page: Page, n: number) => expect.poll(() => page.locator('.sec-frame iframe.shown').count(), { timeout: 90000 }).toBeGreaterThanOrEqual(n)
const openSection = async (page: Page) => {
  await page.locator('.section-menu-btn').click()
  await page.getByRole('button', { name: /^Open “/ }).click()
  await expect(page.locator('.sec-card').first()).toBeVisible({ timeout: 30000 })
}

test('charts picked on different pages keep their own settings, and one link carries the section', async ({ page, browser, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  const errors: string[] = []
  page.on('pageerror', e => errors.push(e.message))
  await picked(page, '/?scatter.colorBy=Season#tab=correlation&ssites=ETAD')
  await picked(page, '/#tab=calibration', '#run-crossplot button.section-add')
  await expect(page.locator('.section-count')).toHaveText('2')
  await openSection(page)
  await page.getByRole('textbox', { name: 'Section name' }).fill('Addis for Ann')
  await page.getByRole('textbox', { name: /^Note on Crossplot/ }).fill('Slope near 1, offset about −2.2')
  await shown(page, 2)
  const scatter = page.frameLocator('.sec-card iframe').first()
  await expect(scatter.locator('[data-single-active]')).toBeVisible()
  // only the chart: no stepping through its page, no sidebar, no second title
  await expect(scatter.locator('h2.single-head')).toBeHidden()
  await expect(scatter.locator('.app-header')).toBeHidden()

  await page.getByRole('button', { name: 'Copy link' }).click()
  await expect(page.getByRole('status').filter({ hasText: 'Link copied' })).toBeVisible()
  const link = await page.evaluate(() => navigator.clipboard.readText())
  expect(link).toContain('#tab=custom&sec=1.')

  // a reader with an empty browser sees the same section, read-only, until they keep a copy
  const reader = await (await browser.newContext({ baseURL: 'http://127.0.0.1:4173', viewport: { width: 1440, height: 1000 } })).newPage()
  await reader.goto(link)
  await expect(reader.getByRole('heading', { name: 'Addis for Ann' })).toBeVisible({ timeout: 45000 })
  await expect(reader.locator('.sec-card')).toHaveCount(2)
  await expect(reader.getByText('Slope near 1, offset about −2.2')).toBeVisible()
  await expect(reader.getByRole('button', { name: /Take .* out of the section/ })).toHaveCount(0)
  await shown(reader, 2)
  await expect(reader.frameLocator('.sec-card iframe').nth(1).locator('#run-crossplot [data-fid]').first()).toBeAttached()
  await reader.reload()
  await expect(reader.getByRole('heading', { name: 'Addis for Ann' })).toBeVisible({ timeout: 45000 })
  await reader.getByRole('button', { name: 'Save a copy to change it' }).click()
  await expect(reader.getByRole('textbox', { name: 'Section name' })).toHaveValue('Addis for Ann')
  await expect(reader).toHaveURL(/sec=s-/)

  // the sender's own link opens their section, which they can change
  await page.goto(link)
  await page.reload()
  await expect(page.getByRole('textbox', { name: 'Section name' })).toHaveValue('Addis for Ann', { timeout: 45000 })
  expect(errors).toEqual([])
})

test('a setting changed in a section chart is kept, and order, width and removal can be undone', async ({ page }) => {
  await picked(page, '/#tab=correlation')
  await picked(page, '/#tab=distribution')
  await openSection(page)
  await shown(page, 2)
  const first = page.frameLocator('.sec-card iframe').first()
  await first.getByRole('radio', { name: 'Per season' }).or(first.getByRole('button', { name: 'Per season' })).first().click()
  await expect.poll(() => page.evaluate(() => JSON.parse(localStorage.getItem('gallery.sections')!).sections[0].items[0].view.q['scatter.layout'])).toBe('Per season')

  const titles = () => page.locator('.sec-card input.sec-card-title').evaluateAll((els) => els.map((e) => (e as HTMLInputElement).value))
  const before = await titles()
  await page.locator('.sec-card').nth(1).getByRole('button', { name: /earlier$/ }).click()
  expect(await titles()).toEqual([before[1], before[0]])
  await page.locator('.sec-card').first().getByRole('button', { name: /out of the section$/ }).click()
  await expect(page.locator('.sec-card')).toHaveCount(1)
  await page.getByRole('button', { name: 'Undo' }).click()
  await expect(page.locator('.sec-card')).toHaveCount(2)
  await page.reload()
  await expect(page.locator('.sec-card')).toHaveCount(2, { timeout: 45000 })
  await shown(page, 2)
  // each chart is its own page, and still follows the theme of the page around it
  await page.evaluate(() => localStorage.setItem('aeth-gallery:theme', 'dark'))
  await expect(page.frameLocator('.sec-card iframe').first().locator('html')).toHaveAttribute('data-theme', 'dark')
})

test('a broken link and a section from another browser each say so', async ({ page }) => {
  await page.goto('/#tab=custom&sec=1.lz.notarealtoken')
  await expect(page.getByRole('heading', { name: 'This link could not be read' })).toBeVisible({ timeout: 45000 })
  await page.goto('about:blank')
  await page.goto('/#tab=custom&sec=s-notmine')
  await expect(page.getByRole('heading', { name: 'This section is in another browser' })).toBeVisible({ timeout: 45000 })
  await page.goto('about:blank')
  await page.goto('/#v=1.lz.notarealtoken')
  await expect(page.getByRole('alert').filter({ hasText: 'This link could not be read' })).toBeVisible({ timeout: 45000 })
})

test('a Dashboard tile, a text block and a set height join a section, and every kind survives the link', async ({ page, browser, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  await page.goto('/#tab=dashboard')
  const tile = page.locator('.dash-tile').filter({ has: page.locator('section.panel') }).first()
  await expect(tile).toBeVisible({ timeout: 45000 })
  await tile.hover()
  await tile.locator('.tile-menu-btn').click()
  await page.getByRole('button', { name: '+ Add to section' }).click()
  await picked(page, '/#tab=correlation')
  await openSection(page)
  await shown(page, 2)
  await expect(page.frameLocator('.sec-card iframe').first().locator('.dash-max [data-tile-id]')).toBeVisible()

  await page.locator('.sec-card').nth(1).getByRole('button', { name: /^Add text before/ }).click()
  await page.getByRole('textbox', { name: 'Heading', exact: true }).fill('The dry season')
  await page.getByRole('textbox', { name: /^Text under/ }).fill('The offset **grows**.')
  await page.locator('.sec-card').nth(1).getByRole('radio', { name: 'L', exact: true }).or(page.locator('.sec-card').nth(1).getByRole('button', { name: 'L', exact: true })).first().click()
  await expect(page.locator('.sec-card').nth(1).locator('.sec-frame')).toHaveCSS('height', '760px')
  await expect(page.locator('.section-count')).toHaveText('2')

  await page.getByRole('button', { name: 'Copy link' }).click()
  await expect(page.getByRole('status').filter({ hasText: 'Link copied' })).toBeVisible()
  const link = await page.evaluate(() => navigator.clipboard.readText())
  const reader = await (await browser.newContext({ baseURL: 'http://127.0.0.1:4173', viewport: { width: 1440, height: 1000 } })).newPage()
  await reader.goto(link)
  await expect(reader.getByRole('heading', { name: 'The dry season' })).toBeVisible({ timeout: 45000 })
  await expect(reader.locator('.sec-text strong')).toHaveText('grows')
  await shown(reader, 2)
  await expect(reader.locator('.sec-card').nth(1).locator('.sec-frame')).toHaveCSS('height', '760px')
})

test('copying a link to one chart gives a token that reopens that chart', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  await page.goto('/?scatter.colorBy=Season#tab=correlation&ssites=ETAD')
  const frame = page.locator('section[data-frame]').first()
  await expect(frame).toBeVisible({ timeout: 45000 })
  await frame.getByRole('button', { name: /^Ref/ }).click()
  await page.getByRole('menuitem', { name: /Copy link to this chart/ }).click()
  await expect(page.getByRole('menuitem', { name: /Link copied/ })).toBeVisible()
  const link = await page.evaluate(() => navigator.clipboard.readText())
  expect(link).toMatch(/#v=1\./)
  await page.goto('about:blank')
  await page.goto(link)
  await expect(page.locator('section.panel.focused')).toBeVisible({ timeout: 45000 })
  expect(new URL(page.url()).searchParams.get('scatter.colorBy')).toBe('Season')
})

test('a section saves as one PDF and one house-style deck', async ({ page }) => {
  await picked(page, '/#tab=calibration', '#run-crossplot button.section-add')
  await picked(page, '/#tab=distribution')
  await openSection(page)
  await page.getByRole('textbox', { name: 'Section name' }).fill('Addis — draft')
  await page.getByRole('textbox', { name: /^Note on Crossplot/ }).fill('Slope near 1 — offset −2.2')
  await page.getByRole('button', { name: 'Add text', exact: true }).click()
  await page.getByRole('textbox', { name: 'Heading', exact: true }).fill('Next steps')
  for (const [name, ext] of [['PDF', 'pdf'], ['Deck', 'pptx']]) {
    const file = page.waitForEvent('download', { timeout: 90000 })
    await page.getByRole('button', { name, exact: true }).click()
    const saved = await (await file).path()
    await (await file).saveAs(`../../tmp/section-export.${ext}`)
    expect(saved).toBeTruthy()
    await expect(page.getByRole('status').filter({ hasText: /^Saved$/ })).toBeVisible()
  }
})

test('a long section keeps only the charts near the screen alive, and one let go comes back when scrolled to', async ({ page }) => {
  await page.goto('/#tab=correlation')
  await expect(page.locator('section[data-frame]').first()).toBeVisible({ timeout: 45000 })
  const items = Array.from({ length: 14 }, (_, i) => ({ id: `i${i}`, title: `Chart ${i}`, chart: 'scatter', view: { h: { tab: 'correlation' }, q: { n: String(i) } } }))
  await page.evaluate((list) => localStorage.setItem('gallery.sections', JSON.stringify({ sections: [{ id: 's-long', name: 'Long', items: list }], active: 's-long' })), items)
  await page.goto('about:blank')
  await page.goto('/#tab=custom&sec=s-long')
  await expect(page.locator('.sec-card').first()).toBeVisible({ timeout: 45000 })
  await shown(page, 2)
  const frames = page.locator('.sec-frame iframe')
  for (let y = 0; y <= 4400; y += 400) { await page.evaluate((v) => window.scrollTo(0, v), y); await page.waitForTimeout(400) }
  await expect.poll(() => page.locator('.sec-card').last().locator('iframe.shown').count(), { timeout: 45000 }).toBe(1)
  expect(await frames.count()).toBeLessThanOrEqual(8)
  await expect(page.locator('.sec-card').first().locator('iframe')).toHaveCount(0)
  await page.evaluate(() => window.scrollTo(0, 0))
  await expect(page.locator('.sec-card').first().locator('iframe.shown')).toBeVisible({ timeout: 45000 })
  expect(await frames.count()).toBeLessThanOrEqual(9)
})

test('a section link copies on plain http, where browsers have no clipboard API', async ({ page }) => {
  // the published site is on http: this is what a page there has. Locally it is a secure context, and the bug does not show
  await page.addInitScript(() => {
    Object.defineProperty(Navigator.prototype, 'clipboard', { get: () => undefined })
    document.addEventListener('copy', () => { (window as unknown as { copied: string }).copied = (document.activeElement as HTMLTextAreaElement).value })
  })
  await picked(page, '/#tab=correlation')
  await openSection(page)
  await expect(page.getByRole('button', { name: 'Copy link', exact: true })).toBeEnabled()
  await page.getByRole('button', { name: 'Copy link', exact: true }).click()
  await expect(page.getByRole('status').filter({ hasText: 'Link copied' })).toBeVisible()
  expect(await page.evaluate(() => (window as unknown as { copied: string }).copied)).toContain('#tab=custom&sec=1.')

  await page.locator('.views-menu button[aria-label="Views"]').click()
  await page.locator('.views-pop').getByRole('button', { name: 'Copy link' }).click()
  await expect(page.locator('.views-pop [role=status]')).toContainText('Link copied')
})

test('a section link that cannot be copied is shown to copy by hand', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(Navigator.prototype, 'clipboard', { get: () => undefined })
    document.execCommand = () => false
  })
  await picked(page, '/#tab=correlation')
  await openSection(page)
  await expect(page.getByRole('button', { name: 'Copy link', exact: true })).toBeEnabled()
  await page.getByRole('button', { name: 'Copy link', exact: true }).click()
  await expect(page.getByRole('textbox', { name: 'Section link' })).toHaveValue(/#tab=custom&sec=1\./)
})
