import { test, expect, type Page } from '@playwright/test'

const url = '/?splitfit.at=0.099&view=single&chart=radar&clock.seasons=1&ranking.measure=r&ranking.minN=10&netrank.minN=100&netrank.scale=Log#tab=ranking&ssites=ETAD%2CINDH&f=HIPS+Fabs'

async function readableRadar(page: Page) {
  const svg = page.locator('#radar svg.animated')
  await expect(svg).toBeVisible()
  // SingleFit places the legend, refits, then debounces its zoom by 450 ms.
  // Wait through those passes: the old layout initially drew correctly, then collapsed.
  await page.waitForTimeout(1200)
  const geometry = await svg.evaluate(el => {
    const box = el.getBoundingClientRect()
    const labels = [...el.querySelectorAll('g > text')].map(t => t.getBoundingClientRect())
    const ticks = [...el.querySelectorAll(':scope > text')].map(t => t.getBoundingClientRect())
    return {
      radius: Number(el.querySelector('circle:nth-child(4)')?.getAttribute('r')),
      clipped: labels.filter(r => r.left < box.left - 1 || r.right > box.right + 1 || r.top < box.top - 1 || r.bottom > box.bottom + 1).length,
      overlaps: labels.some((r, i) => labels.slice(i + 1).some(s => r.left < s.right && r.right > s.left && r.top < s.bottom && r.bottom > s.top)),
      ticksOverlap: ticks.some(r => labels.some(s => r.left < s.right && r.right > s.left && r.top < s.bottom && r.bottom > s.top)),
    }
  })
  expect(geometry.radius).toBeGreaterThanOrEqual(59)
  expect(geometry.clipped).toBe(0)
  expect(geometry.overlaps).toBe(false)
  expect(geometry.ticksOverlap).toBe(false)
  await expect(page.locator('#radar .chart-wrap > .legend')).toContainText('Addis Ababa')
  await expect(page.locator('#radar .chart-wrap > .legend')).toContainText('Delhi')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
}

test('radar survives live resizing, including the linked focus view', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', e => errors.push(e.message))
  await page.setViewportSize({ width: 1920, height: 1080 })
  await page.goto(url.replace('#', '&focus=radar#'))
  for (const [width, height] of [[1920, 1080], [1440, 900], [1024, 768], [820, 900], [768, 900], [560, 800], [390, 844], [320, 640], [1440, 900]]) {
    await page.setViewportSize({ width, height })
    await readableRadar(page)
  }
  expect(errors).toEqual([])
})

test('controls stack above the radar when the Samples dock leaves a narrow card', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.goto(url)
  await readableRadar(page)
  await expect(page.locator('#radar .frame-body')).toHaveCSS('display', 'grid')
  await page.getByRole('tab', { name: /Samples/ }).click()
  await readableRadar(page)
  for (const width of [1100, 1024, 950, 901]) {
    await page.setViewportSize({ width, height: 900 })
    await readableRadar(page)
    await expect(page.locator('#radar .frame-body')).toHaveCSS('display', 'block')
    const bounds = await page.locator('#radar').evaluate(el => ({
      controls: el.querySelector('.frame-controls')!.getBoundingClientRect().bottom,
      plot: el.querySelector('svg.animated')!.getBoundingClientRect().top,
    }))
    expect(bounds.controls).toBeLessThanOrEqual(bounds.plot)
  }
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.getByRole('tab', { name: /Samples/ }).click()
  await readableRadar(page)
  await expect(page.locator('#radar .frame-body')).toHaveCSS('display', 'grid')
  await page.locator('#radar .legend-item').filter({ hasText: 'Delhi' }).click()
  await expect(page.locator('#radar path[data-export-site="Delhi"]')).toHaveCount(0)
  await page.locator('#radar .legend-item').filter({ hasText: 'Delhi' }).click()
  await expect(page.locator('#radar path[data-export-site="Delhi"]')).toHaveCount(1)
})
