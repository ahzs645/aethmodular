import { test, expect } from '@playwright/test'

const url = '/?view=single&chart=pmf-residual-biomass&pmfbm.pair=FTIR+EC+%E2%88%92+HIPS+BC#tab=pmf'

for (const form of ['Absolute', 'Percent']) test(`${form} PMF difference fits follow the Plot sidebar in both layouts and survive reload`, async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(url.replace('#', `&pmfbm.form=${form}#`))
  const chart = page.locator('#pmf-residual-biomass')
  await expect(chart.locator('[data-regression="deming"]').first()).toBeVisible()
  await expect(chart).not.toContainText('OLS of the difference on the biomass share')
  await page.getByRole('tab', { name: 'Plot', exact: true }).click()
  const controls = page.getByRole('group', { name: 'Regression fits' })
  const ols = controls.getByRole('checkbox', { name: 'OLS', exact: true })
  const deming = controls.getByRole('checkbox', { name: 'Deming', exact: true })
  const york = controls.getByRole('checkbox', { name: 'York', exact: true })
  await expect(deming).toBeChecked()

  const groups = await chart.locator('[data-regression="deming"]').count()
  const points = await chart.locator('.scatter-dots circle').count()
  expect(groups).toBeGreaterThan(1)
  expect(points).toBeGreaterThan(3)
  const demingPath = await chart.locator('[data-regression="deming"]').first().getAttribute('d')
  const demingSlope = await chart.locator('tbody tr').first().locator('td').nth(3).innerText()
  await deming.uncheck()
  await ols.check()
  await expect(chart.locator('[data-regression="ols"]')).toHaveCount(groups)
  await expect(chart.getByRole('columnheader', { name: 'OLS slope (per % biomass)', exact: true })).toBeVisible()
  expect(await chart.locator('[data-regression="ols"]').first().getAttribute('d')).not.toBe(demingPath)
  // In concentration units the first season's slopes round to the same four decimals.
  if (form === 'Percent') expect(await chart.locator('tbody tr').first().locator('td').nth(3).innerText()).not.toBe(demingSlope)

  await york.check()
  await expect(chart.locator('[data-regression="york"]')).toHaveCount(groups)
  await expect(chart.getByRole('columnheader', { name: 'York slope (per % biomass)', exact: true })).toBeVisible()
  // The same constant-error ratio must give the same York and Deming coefficients.
  expect(await chart.locator('tbody tr').first().locator('td').nth(5).innerText()).toBe(demingSlope)
  await expect.poll(() => new URLSearchParams(new URL(page.url()).hash.slice(1)).get('fits')).toBe('ols|york')
  await page.reload()
  await expect(chart.locator('[data-regression="ols"]')).toHaveCount(groups)
  await expect(chart.locator('[data-regression="york"]')).toHaveCount(groups)
  await expect(ols).toBeChecked()
  await expect(york).toBeChecked()
  await expect(deming).not.toBeChecked()

  await deming.check()
  await expect(chart.locator('[data-regression]')).toHaveCount(groups * 3)
  await chart.getByRole('radio', { name: 'Per season', exact: true }).click()
  await expect(chart.locator('[data-regression]')).toHaveCount(groups * 3)
  await expect(chart.locator('svg').filter({ has: page.locator('[data-regression="york"]') })).toHaveCount(groups)

  // Hiding fits leaves the data and descriptive statistics available.
  await ols.uncheck()
  await deming.uncheck()
  await york.uncheck()
  await expect(chart.locator('[data-regression]')).toHaveCount(0)
  await expect(chart.getByRole('columnheader', { name: /slope|intercept/ })).toHaveCount(0)
  await expect(chart.locator('.scatter-dots circle')).toHaveCount(points)
  await expect(chart.getByRole('columnheader', { name: 'R²', exact: true })).toBeVisible()
  expect(errors).toEqual([])
})
