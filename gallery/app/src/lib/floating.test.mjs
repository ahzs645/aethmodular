import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'

/**
 * docs/gallery-chart-standards.md "Floating layers": a popover, note or menu floats (portalled, fixed,
 * kept inside the window). The pattern that gets cut off is a body positioned absolutely just under or
 * over its trigger, inside a panel: the ⓘ notes were drawn that way and were clipped at a phone's edge.
 * Every such rule in styles.css is listed here with why it is safe; a new one fails until it floats.
 */
const css = fs.readFileSync(new URL('../styles.css', import.meta.url), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '')
const ALLOWED = {
  '.frame-menu': 'ChartFrame RefMenu always renders it .floating (portalled, fixed)',
  '.combobox-menu': 'Combobox always renders it .floating (portalled, fixed)',
  '.views-pop': 'the header\'s Views menu: the sticky header clips nothing',
  '.navmenu-list': 'the header\'s nav menus: the sticky header clips nothing',
  '.dash-pop-panel': 'the Dashboard header\'s panels: the sticky header clips nothing',
}
const anchored = /position:\s*absolute/.source

test('popovers float instead of sitting in place under their trigger', () => {
  const rules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map(([, sel, body]) => ({ sel: sel.trim(), body }))
  const inPlace = rules.filter(({ body }) => new RegExp(anchored).test(body) && /(top|bottom):\s*(calc\()?100%/.test(body))
  const unexpected = inPlace.filter(({ sel }) => !sel.split(',').every((s) => s.trim() in ALLOWED))
  assert.deepEqual(unexpected.map((r) => r.sel), [],
    'a popover positioned under its trigger inside a panel gets clipped: portal it (hooks/useFloatingMenu, ChartFrame useFloatingTip), or list it in ALLOWED with the reason it cannot be cut off')
})

test('the chart notes (ⓘ) are never positioned inside a panel', () => {
  const body = [...css.matchAll(/\.info-tip-body\s*\{([^{}]*)\}/g)].map((m) => m[1]).join('\n')
  assert.ok(body, '.info-tip-body rule found')
  assert.doesNotMatch(body, /position:\s*absolute/)
})
