import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

/**
 * docs/gallery-chart-standards.md "Points": a chart that draws filters as points draws them through
 * components/ScatterLayer (memoised dots, FocusLayer, PinRing, HitLayer), so every chart hovers, pins and
 * costs the same. These are the patterns that crept back in one chart at a time before it existed: each was
 * slow on a dense chart or left a highlight stuck on every chart. A new one fails here with the shared piece
 * to use instead; a real exception goes in its ALLOWED list with the reason.
 */
const src = new URL('..', import.meta.url).pathname
const files = []
const walk = (dir) => {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name)
    if (e.isDirectory()) walk(p)
    else if (/\.tsx?$/.test(e.name)) files.push(p)
  }
}
walk(src)
// code only: block comments and // line comments (not the // of a URL) removed
const code = (p) => fs.readFileSync(p, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|\s)\/\/.*$/gm, '$1')
const rel = (p) => path.relative(src, p)
const offenders = (re, allowed) => files.filter((p) => !(rel(p) in allowed) && re.test(code(p))).map(rel)

test('nearest-point hit testing lives in components/HitLayer only', () => {
  const ALLOWED = {
    'components/HitLayer.tsx': 'the shared one',
    'charts/calibration/AnalogLab.tsx': 'an anonymous point pool (no filter ids) with its own local hover',
  }
  assert.deepEqual(offenders(/\bDelaunay\b/, ALLOWED), [],
    'draw the points through components/ScatterLayer (or HitLayer for a non-filter layer) rather than a Delaunay of your own')
})

test('no chart reorders its marks to put the focused one on top', () => {
  assert.deepEqual(offenders(/\bfocusLast\b/, {}), [],
    'moving the hovered mark in the DOM redraws every dot and loses its mouseleave: ScatterLayer draws the focused filter again on top (FocusLayer)')
})

test('marks do not set the gallery hover from their own mouse handlers', () => {
  // the gallery's (lib/highlight, conventionally `hl`); a chart's own local or legend hover is its business
  const ALLOWED = {
    'components/GroupDrawer.tsx': 'table rows, not marks',
  }
  assert.deepEqual(offenders(/onMouse(Enter|Over)=\{[^\n]*\bhl\.setHover\(/, ALLOWED), [],
    'a handler per mark is a hit test per mark and a hover left behind when the mark is redrawn: use ScatterLayer (one pointer layer, nearest point)')
})

test('charts read the gallery hover only for their own filters', () => {
  const ALLOWED = {
    'lib/highlight.tsx': 'defines it',
    'pages/MeetingFollowupPage.tsx': 'feeds SpectrumPlot, whose traces follow every hover in the card',
  }
  assert.deepEqual(offenders(/\buseHighlight\(\)/, ALLOWED), [],
    'useHighlight() with no keys re-renders the chart on every hover in the gallery: useHighlightActions() to pin/open/brush, useHighlight(keys) to read a hover of your own filters (ScatterLayer does this for points)')
})

test('a hover stays in its card', () => {
  const frame = fs.readFileSync(path.join(src, 'components/ChartFrame.tsx'), 'utf8')
  assert.equal((frame.match(/<section className=[^\n]*\{\.\.\.cardAttrs\}/g) ?? []).length, 2,
    'both ChartFrame roots carry the card id: lib/highlight reads it off the element under the pointer')
  // the context only reaches hooks below the frame; a chart reading the hover in the component that renders its own
  // ChartFrame would see every card's hover
  const ALLOWED = {
    'pages/MeetingFollowupPage.tsx': 'SpectraPanel, which reads the hover, renders inside ThreePlots\' ChartFrame',
    'charts/correlation/Scatterplot.tsx': 'NO_KEYS: never reads a hover, it only sets one',
  }
  const above = files.filter((p) => !(rel(p) in ALLOWED) && /\buseHighlight\(/.test(code(p)) && /<ChartFrame\b/.test(code(p)) && !/\bhoverCard=/.test(code(p))).map(rel)
  assert.deepEqual(above, [], 'name the card: const hoverCard = useId(); useHighlight(keys, hoverCard); <ChartFrame hoverCard={hoverCard}>')
})

test('no :has() on the page root', () => {
  const css = fs.readFileSync(new URL('../styles.css', import.meta.url), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '')
  const selectors = [...css.matchAll(/([^{}]+)\{[^{}]*\}/g)].flatMap(([, sel]) => sel.split(',').map((s) => s.trim()))
  assert.deepEqual(selectors.filter((s) => /^(html|body|:root)[^\s>+~]*:has\(/.test(s)), [],
    'a :has() on html/body makes Chrome restyle the whole page whenever a node is added anywhere (a tooltip, a focus dot): set a class instead (html.ctx-open)')
})
