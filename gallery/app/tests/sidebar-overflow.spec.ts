import { expect, test, type Locator } from '@playwright/test'

test.use({ reducedMotion: 'reduce' })

async function expectVerticalSidebar(sidebar: Locator) {
  await expect(sidebar).toBeVisible()
  await expect.poll(() => sidebar.evaluate(el => ({
    overflowX: getComputedStyle(el).overflowX,
    extraWidth: el.scrollWidth - el.clientWidth,
  }))).toEqual({ overflowX: 'hidden', extraWidth: 0 })
}

for (const side of ['left', 'right']) {
  test(`split-fit options fit in the ${side} column and scroll down to the last control`, async ({ page }) => {
    await page.setViewportSize({ width: 1465, height: 794 })
    await page.addInitScript(() => {
      localStorage.setItem('aeth-gallery:explain', '1')
      localStorage.setItem('aeth-gallery:density', 'readable')
    })
    await page.goto(`/?view=single&chart=split-fit&side=${side}#ci=1`)
    const options = page.locator(`[data-single-active][data-chart-key="split-fit"] > .controls-${side} > .frame-controls`)
    await expect(options.locator('.regression-summary-side.pointable').first()).toBeVisible()
    await page.evaluate(() => document.fonts.ready)
    await expectVerticalSidebar(options)

    const cutoffs = options.getByRole('textbox', { name: 'Table cutoffs' })
    await cutoffs.click()
    await expect(cutoffs).toBeFocused()
    expect(await options.evaluate(el => el.scrollTop)).toBeGreaterThan(0)
    expect(await options.evaluate(el => el.scrollLeft)).toBe(0)

    await page.setViewportSize({ width: 390, height: 794 })
    await expect.poll(() => options.evaluate(el => el.scrollWidth - el.clientWidth)).toBe(0)
  })
}

test('the Samples and Plot dock stays vertical on a narrow phone', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 794 })
  await page.goto('/#tab=correlation')
  const dock = page.locator('.subset-panel')

  for (const tab of ['Samples', 'Plot']) {
    await page.getByRole('tab', { name: new RegExp(`^${tab}`) }).click()
    await expectVerticalSidebar(dock)
    await dock.getByRole('button', { name: /^Close / }).click()
  }
})

test('the contents sidebar keeps its links within its column', async ({ page }) => {
  await page.goto('/?toc=shown#tab=baseline')
  await expectVerticalSidebar(page.locator('.toc'))
})
