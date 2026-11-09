import test from 'node:test'
import assert from 'node:assert/strict'
import { addItem, addText, chartCount, decodeSection, encodeSection, fromWire, isSectionId, isText, itemHref, itemView, MAX_ITEMS, moveItem, newSection, patchItem, removeItem, sameSection, toWire } from './sections.ts'

const view = (tab = 'correlation', q = {}) => ({ h: { tab, ssites: 'ETAD', fits: 'ols' }, q })
const item = (chart, v = view()) => ({ chart, title: chart, view: v })

test('a chart joins once per set of settings, and the same chart at other settings is a second item', () => {
  let s = addItem(newSection('A'), item('scatter'))
  s = addItem(s, item('scatter'))
  assert.equal(s.items.length, 1)
  s = addItem(s, item('scatter', view('correlation', { 'scatter.colorBy': 'Season' })))
  assert.equal(s.items.length, 2)
  // key order in the address is not a difference
  s = addItem(s, item('scatter', { h: { fits: 'ols', ssites: 'ETAD', tab: 'correlation' }, q: {} }))
  assert.equal(s.items.length, 2)
})

test('a full section takes no more charts', () => {
  let s = newSection('A')
  for (let i = 0; i < MAX_ITEMS + 3; i++) s = addItem(s, item(`c${i}`))
  assert.equal(s.items.length, MAX_ITEMS)
})

test('moving stops at the ends, and removing and patching touch one item', () => {
  let s = ['a', 'b', 'c'].reduce((x, c) => addItem(x, item(c)), newSection('A'))
  const [a, b] = s.items
  assert.equal(moveItem(s, a.id, -1), s)
  s = moveItem(s, a.id, 1)
  assert.deepEqual(s.items.map((i) => i.chart), ['b', 'a', 'c'])
  s = patchItem(s, b.id, { wide: true })
  assert.deepEqual(s.items.map((i) => !!i.wide), [true, false, false])
  assert.deepEqual(removeItem(s, a.id).items.map((i) => i.chart), ['b', 'c'])
})

test('an item keeps what the chart shows, not how the page placed it', () => {
  const v = itemView({ h: { tab: 'weather' }, q: { view: 'single', chart: 'x', side: 'top', focus: 'x', toc: 'shown', embed: '1', embedchart: '1', v: 't', 'scatter.layout': 'Per group' } })
  assert.deepEqual(v.q, { 'scatter.layout': 'Per group' })
})

test('a section survives its link: order, notes, widths and every setting', async () => {
  let s = { ...newSection('Addis by season'), about: 'For Ann: ≥ µg/m³ — seasons' }
  s = addItem(s, item('run-crossplot', view('calibration', { 'calibration.preset': 'ocec440_airspec' })))
  s = addItem(s, item('scatter'))
  s = patchItem(s, s.items[0].id, { wide: true, note: 'Slope near 1, offset −2.2' })
  const back = await decodeSection(await encodeSection(s))
  assert.ok(sameSection(s, back))
  assert.equal(back.id, s.id)
  assert.equal(back.items[0].note, 'Slope near 1, offset −2.2')
  assert.equal(back.items[0].wide, true)
  assert.deepEqual(back.items[0].view, s.items[0].view)
})

test('a dozen charts with shared settings stay a short link', async () => {
  let s = newSection('Twelve')
  const h = { tab: 'correlation', net: 'both', ssites: 'CLST,CLTA', isites: 'ANTU1,LCON1', fits: 'ols,deming', x: 'EC (FTIR)', y: 'HIPS BC', from: '2023-01-01', to: '2024-12-31' }
  for (let i = 0; i < 12; i++) s = addItem(s, { chart: `chart-${i}`, title: `Chart ${i}`, view: { h, q: { 'scatter.layout': 'Per site', [`c${i}.bin`]: String(i) } } })
  const token = await encodeSection(s)
  assert.ok(token.length < 2000, `${token.length}`)
  assert.ok(token.length < JSON.stringify(toWire(s)).length * 0.6, `${token.length}`)
})

test('a link holds no more than a section: anything else in it is dropped or refused', () => {
  assert.throws(() => fromWire(null))
  assert.throws(() => fromWire({ n: 'x' }))
  assert.throws(() => fromWire({ n: 3, i: [] }))
  const s = fromWire({ id: 'evil', n: 'x'.repeat(500), i: [null, { c: 7 }, { c: 'ok', t: 9, h: { tab: 'map', bad: 3 }, q: 'nope', m: 5, w: 1 }] })
  assert.ok(isSectionId(s.id))
  assert.equal(s.name.length, 120)
  assert.deepEqual(s.items.map((i) => i.chart), ['ok'])
  assert.deepEqual(s.items[0].view, { h: { tab: 'map' }, q: {} })
  assert.equal(s.items[0].title, 'ok')
  assert.equal(s.items[0].note, undefined)
})

test('a link cannot overrun a section', () => {
  const s = fromWire({ n: 'x', i: Array.from({ length: 200 }, (_, i) => ({ c: `c${i}`, h: {} })) })
  assert.equal(s.items.length, MAX_ITEMS)
})

test('text blocks sit between charts, do not count as charts, and survive a link', async () => {
  let s = addItem(addItem(newSection('A'), item('a')), item('b'))
  s = addText(s, s.items[1].id)
  s = addText(s)
  s = patchItem(s, s.items[1].id, { title: 'Dry season', note: 'The offset **grows**.' })
  assert.deepEqual(s.items.map((i) => isText(i) ? 'text' : i.chart), ['a', 'text', 'b', 'text'])
  assert.equal(chartCount(s), 2)
  const back = await decodeSection(await encodeSection(s))
  // a block with nothing in it is not worth a link
  assert.deepEqual(back.items.map((i) => isText(i) ? i.title : i.chart), ['a', 'Dry season', 'b'])
  assert.equal(back.items[1].note, 'The offset **grows**.')
  assert.ok(isText(back.items[1]))
})

test('a set height survives a link, and an odd one is ignored', async () => {
  const s = patchItem(addItem(newSection('A'), item('a')), addItem(newSection('A'), item('a')).items[0].id, {})
  const tall = patchItem(s, s.items[0].id, { height: 'L' })
  assert.equal((await decodeSection(await encodeSection(tall))).items[0].height, 'L')
  assert.equal(fromWire({ n: 'x', i: [{ c: 'a', h: {}, z: 'XL' }] }).items[0].height, undefined)
})

test('a token is not taken for a section id', async () => {
  const token = await encodeSection(addItem(newSection('A'), item('scatter')))
  assert.equal(isSectionId(token), false)
  assert.equal(isSectionId(newSection().id), true)
})

test('a Dashboard tile opens maximised, and a chart in the one-chart view', () => {
  globalThis.window ??= { location: { pathname: '/g/' } }
  const tile = new URL(itemHref({ chart: 'tile:t3', view: { h: { tab: 'dashboard' }, q: { a: '1' } } }, true), 'http://x')
  assert.equal(tile.searchParams.get('dmax'), 't3')
  assert.equal(tile.searchParams.get('embedchart'), '1')
  assert.equal(tile.searchParams.get('chart'), null)
  const chart = new URL(itemHref({ chart: 'scatter', view: { h: { tab: 'correlation' }, q: {} } }, false), 'http://x')
  assert.equal(chart.searchParams.get('chart'), 'scatter')
  assert.equal(chart.searchParams.get('embed'), null)
  assert.equal(chart.hash, '#tab=correlation')
})
