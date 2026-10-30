#!/usr/bin/env node
/** Upgrade smoke test: real chart construction and frozen draft imports, without private store calls. */
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { parseArgs } from 'node:util'
import { chromium } from 'playwright'
import { dragWalkerField, settleLayout, walkerWorkspace } from './graphic-walker-browser.mjs'

const { values } = parseArgs({ options: {
  'base-url': { type: 'string', default: 'http://localhost:4173' },
  'browser-path': { type: 'string' }, output: { type: 'string' },
} })
const output = values.output && path.resolve(values.output)
if (output) await fs.mkdir(output, { recursive: true })
const contract = JSON.parse(await fs.readFile(new URL('../src/lib/graphicWalkerContract.json', import.meta.url), 'utf8'))
const fixture = JSON.parse(await fs.readFile(new URL('../public/data/filters.json', import.meta.url), 'utf8'))
const expected = fixture.rows.filter(row => !row.excluded).length
const executablePath = values['browser-path'] ?? ['/usr/bin/chromium', '/usr/bin/chromium-browser'].find(existsSync)
const browser = await chromium.launch({ executablePath, args: ['--no-sandbox'] })
const context = await browser.newContext({ viewport: { width: 1500, height: 1050 } })
if (output) await context.tracing.start({ screenshots: true, snapshots: true })
const page = await context.newPage()
page.setDefaultTimeout(20000)
const errors = [], requests = [], checks = []
page.on('pageerror', error => errors.push(error.message))
page.on('request', request => requests.push(request.url()))
const snapshot = async name => { if (output) await page.screenshot({ path: path.join(output, `${name}.png`) }) }
const downloadedDraft = async () => {
  const pending = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download draft', exact: true }).click()
  const download = await pending
  return JSON.parse(await fs.readFile(await download.path(), 'utf8'))
}
const importDraft = draft => page.locator('.chart-builder-actions input[type="file"]').setInputFiles({
  name: 'draft.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(draft)),
})
const setSites = sites => page.evaluate(sites => {
  const params = new URLSearchParams(location.hash.slice(1)); params.set('sites', sites); location.hash = params.toString()
}, sites)
try {
  await page.goto(`${values['base-url'].replace(/\/$/, '')}/#net=focus`, { waitUntil: 'domcontentloaded' })
  const settings = () => page.getByRole('button', { name: 'Display and data settings', exact: true }).first()
  await settings().waitFor({ timeout: 60000 })
  assert.ok(!requests.some(url => /GraphicWalkerWorkspace/.test(url)), 'The ordinary gallery loaded the builder')
  await settings().click()
  await page.getByRole('button', { name: 'Interface', exact: true }).click()
  await page.getByRole('button', { name: 'On', exact: true }).click()
  assert.ok(!requests.some(url => /GraphicWalkerWorkspace/.test(url)), 'The settings switch eagerly loaded the builder')
  await snapshot('interface')
  await page.getByRole('button', { name: 'Open chart builder', exact: true }).click()
  let editor = await walkerWorkspace(page)
  assert.equal(await editor.getAttribute('data-walker-version'), contract.testedVersion)
  await editor.locator('style[data-gallery-walker-style]').waitFor({ state: 'attached' })
  const framing = await editor.locator('.App [role="tabpanel"][data-state="active"] > div.border').evaluate(el => {
    const style = getComputedStyle(el); return { margin: style.margin, padding: style.padding, border: style.borderWidth }
  })
  assert.deepEqual(framing, { margin: '0px', padding: '0px', border: '0px' }, 'Upstream layout changed: review the theme adapter')
  const initial = await downloadedDraft()
  assert.equal(initial.context.filterIds.length, expected)
  assert.equal(initial.charts[0].config.defaultAggregated, false)
  checks.push('optional loading, shadow-root contract and gallery framing')

  await dragWalkerField(page, editor, 'HIPS BC (µg/m³)', 'columns')
  await dragWalkerField(page, editor, 'EC (FTIR) (µg/m³)', 'rows')
  await dragWalkerField(page, editor, 'Iron (µg/m³)', 'rows')
  await dragWalkerField(page, editor, 'Site', 'columns', 'start')
  await dragWalkerField(page, editor, 'Season (selected calendar)', 'color')
  await editor.locator('.mark-symbol path').first().waitFor({ timeout: 30000 })
  const built = await downloadedDraft()
  assert.equal(built.charts[0].encodings.rows.length, 2)
  assert.equal(built.charts[0].encodings.columns[0].fid, 'site')
  assert.equal(built.charts[0].encodings.color[0].fid, 'season')
  checks.push('two measures, site facets and season colour constructed through real drag controls')
  await snapshot('builder')
  if (output) await fs.writeFile(path.join(output, 'builder-accessibility.txt'), await editor.ariaSnapshot())

  await page.getByRole('button', { name: /^Charts ·/ }).click()
  await page.getByRole('menuitem', { name: 'Distribution', exact: true }).click()
  await page.getByRole('button', { name: /^Charts ·/ }).click()
  await page.getByRole('menuitem', { name: 'Chart builder', exact: true }).click()
  editor = await walkerWorkspace(page)
  assert.deepEqual((await downloadedDraft()).charts, built.charts, 'Tab switch lost the draft')
  await importDraft({ ...built, version: 999 })
  await page.getByRole('alert').filter({ hasText: 'unsupported' }).waitFor()
  assert.deepEqual((await downloadedDraft()).charts, built.charts, 'Rejected import changed the active charts')
  await importDraft(built)
  await page.getByRole('status').filter({ hasText: 'Loaded 1 chart' }).waitFor()
  assert.deepEqual((await downloadedDraft()).charts, built.charts)
  checks.push('tab retention and draft import/export through public adapter APIs')

  // Keep each accepted release's frozen chart. Rebuilding this fixture during an upgrade
  // would hide a serialization break. The gallery fields/data remain the current ones.
  for (const version of contract.readableDraftVersions) {
    const legacy = JSON.parse(await fs.readFile(new URL(`./fixtures/graphic-walker-${version}.json`, import.meta.url), 'utf8'))
    assert.equal(legacy.graphicWalkerVersion, version)
    await importDraft({ ...built, graphicWalkerVersion: version, charts: [legacy.chart] })
    await page.getByRole('status').filter({ hasText: 'Loaded 1 chart' }).waitFor()
    await editor.locator('.mark-symbol path').first().waitFor()
    const restored = await downloadedDraft()
    assert.deepEqual(restored.charts[0].encodings.columns, legacy.chart.encodings.columns)
    assert.deepEqual(restored.charts[0].encodings.rows, legacy.chart.encodings.rows)
    assert.deepEqual(restored.charts[0].encodings.color, legacy.chart.encodings.color)
  }
  await importDraft(built)
  await page.getByRole('status').filter({ hasText: 'Loaded 1 chart' }).waitFor()
  checks.push(`saved charts from supported versions: ${contract.readableDraftVersions.join(', ')}`)

  const site = fixture.rows.find(row => !row.excluded)?.site
  await setSites(site)
  await page.waitForFunction(expected => document.querySelector('.chart-builder-intro strong')?.textContent !== `${expected.toLocaleString()} filters`, expected)
  const subset = await downloadedDraft()
  assert.ok(subset.context.filterIds.length > 0 && subset.context.filterIds.length < expected)
  assert.deepEqual(subset.charts, built.charts)
  await page.getByRole('button', { name: 'Theme: light', exact: true }).first().click()
  await settleLayout(page)
  const colours = await editor.locator('.App').evaluate(el => ({ editor: getComputedStyle(el).backgroundColor,
    gallery: getComputedStyle(document.querySelector('#chart-builder')).backgroundColor }))
  assert.equal(colours.editor, colours.gallery, 'Editor theme diverged from the gallery')
  await snapshot('builder-dark')
  for (const width of [390, 768]) {
    await page.setViewportSize({ width, height: 900 }); await settleLayout(page)
    const size = await page.evaluate(() => ({ width: innerWidth, document: document.documentElement.scrollWidth }))
    assert.ok(size.document <= size.width + 2, `Page overflows at ${width}px: ${JSON.stringify(size)}`)
    await snapshot(`builder-${width}`)
  }
  checks.push('gallery filters, theme and responsive layout')

  await page.setViewportSize({ width: 1500, height: 1050 }); await settleLayout(page)
  await setSites('__no_site__')
  await page.getByText('No filters match this selection.', { exact: false }).waitFor()
  const empty = await downloadedDraft()
  assert.equal(empty.context.filterIds.length, 0)
  assert.deepEqual(empty.charts, built.charts)
  await settings().click()
  await page.getByRole('button', { name: 'Interface', exact: true }).click()
  await page.getByRole('button', { name: 'Off', exact: true }).click()
  await page.getByRole('button', { name: 'Close display settings', exact: true }).click()
  await page.getByRole('button', { name: 'Enable chart builder', exact: true }).waitFor()
  requests.length = 0
  await page.reload({ waitUntil: 'networkidle' })
  await page.getByRole('button', { name: 'Enable chart builder', exact: true }).waitFor()
  assert.ok(!requests.some(url => /GraphicWalkerWorkspace/.test(url)))
  assert.deepEqual(errors, [])
  checks.push('empty selection, disabled reload and no browser exceptions')
  console.log(JSON.stringify({ version: contract.testedVersion, status: 'passed', checks }, null, 2))
} catch (error) {
  await snapshot('failure')
  console.error(JSON.stringify({ status: 'failed', completed: checks, browserErrors: errors, error: String(error) }, null, 2))
  process.exitCode = 1
} finally {
  if (output) await context.tracing.stop({ path: path.join(output, 'trace.zip') })
  await browser.close()
}
