import { test, expect, type Page } from '@playwright/test'

const view = '/?view=single&chart=composition&composition.bars=over+time&composition.units=composition+%28%25%29&composition.axis=Seasons+side+by+side&composition.shape=Daily+bars#tab=partOfWhole&ssites=ETAD&from=2022-10-01&to=2024-09-30'
const plot = (page: Page) => page.locator('#composition [data-composition-site="Addis Ababa"]')
const annual = (page: Page) => page.getByRole('radio', { name: 'Year, then season', exact: true })
const comparison = (page: Page) => page.getByRole('radio', { name: 'Season, then year', exact: true })
const blocks = (page: Page) => plot(page).locator('[data-season-block]').evaluateAll(els => els.map(el => ({
  name: el.getAttribute('data-season-block'), year: el.getAttribute('data-season-year'),
  start: el.getAttribute('data-season-start'), end: el.getAttribute('data-season-end'),
})))
const ready = async (page: Page) => {
  await page.goto(view)
  await expect(plot(page).locator('[data-season-block]')).toHaveCount(6)
  await page.evaluate(() => document.fonts.ready)
}

test('year groups contain their seasons, including the whole cross-year Bega', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('aeth-gallery:theme', 'dark'))
  const errors: string[] = []
  page.on('pageerror', e => errors.push(e.message))
  await ready(page)
  await expect(annual(page)).toBeChecked()
  expect(await blocks(page)).toEqual([
    { name: 'Bega', year: '2023', start: '2022-12-07', end: '2023-01-30' },
    { name: 'Belg', year: '2023', start: '2023-02-02', end: '2023-05-29' },
    { name: 'Kiremt', year: '2023', start: '2023-06-01', end: '2023-09-27' },
    { name: 'Bega', year: '2024', start: '2023-10-15', end: '2024-01-31' },
    { name: 'Belg', year: '2024', start: '2024-02-03', end: '2024-05-12' },
    { name: 'Kiremt', year: '2024', start: '2024-06-06', end: '2024-09-12' },
  ])
  expect(await plot(page).locator('[data-category-group]').evaluateAll(els => els.map(el => el.getAttribute('data-category-group')))).toEqual(['2023', '2024'])
  expect(errors).toEqual([])
})

test('season comparisons put each year side by side and persist on reload', async ({ page }) => {
  await ready(page)
  const count = await plot(page).getAttribute('data-composition-filters')
  const original = await blocks(page)
  await comparison(page).click()
  expect((await blocks(page)).map(b => `${b.name} ${b.year}`)).toEqual(['Bega 2023', 'Bega 2024', 'Belg 2023', 'Belg 2024', 'Kiremt 2023', 'Kiremt 2024'])
  expect((await blocks(page)).sort((a, b) => a.start!.localeCompare(b.start!))).toEqual(original.sort((a, b) => a.start!.localeCompare(b.start!)))
  await expect(plot(page)).toHaveAttribute('data-composition-filters', count!)
  await page.reload()
  await expect(comparison(page)).toBeChecked()
  await expect(plot(page).locator('[data-category-group]')).toHaveCount(3)
  await expect(plot(page).locator('[data-season-axis]')).toHaveAttribute('data-season-axis', 'Season, then year')
})

test('both label levels fit on phones and the timeline scrolls horizontally', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await ready(page)
  for (const control of [annual(page), comparison(page)]) {
    await control.click()
    await expect.poll(() => plot(page).evaluate(svg => {
      const outer = svg.getBoundingClientRect()
      return ['data-category-leaf', 'data-category-group'].every(attr => {
        const boxes = [...svg.querySelectorAll(`[${attr}] text`)].map(el => el.getBoundingClientRect())
        return boxes.every((a, i) => a.left >= outer.left && a.right <= outer.right && a.bottom <= outer.bottom
          && boxes.slice(i + 1).every(b => a.right + 2 <= b.left || b.right + 2 <= a.left))
      })
    })).toBe(true)
  }
  const scroller = page.locator('#composition .timeline-scroll')
  expect(await scroller.evaluate(el => el.scrollWidth > el.clientWidth)).toBe(true)
  await scroller.focus()
  await page.keyboard.press('ArrowRight')
  await expect.poll(() => scroller.evaluate(el => el.scrollLeft)).toBeGreaterThan(0)
  await expect(plot(page)).toBeVisible()
})

test('the hierarchy works for areas and the calendar axis keeps its existing controls', async ({ page }) => {
  await ready(page)
  await comparison(page).click()
  await page.getByRole('radio', { name: 'Stacked area', exact: true }).click()
  await expect(plot(page).locator('[data-season-axis]')).toHaveAttribute('data-season-axis', 'Season, then year')
  expect(await plot(page).locator('path').evaluateAll(els => els.every(el => !/NaN|Infinity/.test(el.getAttribute('d') ?? '')))).toBe(true)
  await page.getByRole('radio', { name: 'Calendar dates', exact: true }).click()
  await expect(plot(page).locator('[data-season-axis]')).toHaveCount(0)
  await expect(comparison(page)).toHaveCount(0)
  await page.getByRole('radio', { name: 'Seasons side by side', exact: true }).click()
  await expect(comparison(page)).toBeChecked()
})

test('the distribution chart retains its site and season hierarchy', async ({ page }) => {
  await page.goto('/?view=single&chart=distribution&distribution.by=Season&distribution.encoding=Violin#tab=distribution&ssites=CHTS,ETAD')
  const axis = page.locator('#distribution')
  await expect(axis.locator('[data-category-group]')).toHaveCount(2)
  expect(await axis.locator('[data-category-group]').evaluateAll(els => els.map(el => el.getAttribute('data-category-group')))).toEqual(['Beijing', 'Addis Ababa'])
  expect(await axis.locator('[data-category-leaf]').count()).toBeGreaterThanOrEqual(6)
})
