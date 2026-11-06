import test from 'node:test'
import assert from 'node:assert/strict'
import { addItem, addText, newSection, patchItem } from './sections.ts'
import { deckSlides, deckText } from './sectionDeck.ts'

const chart = (title) => ({ chart: title, title, view: { h: { tab: 'calibration' }, q: {} } })

test('no slide or speaker note holds an em dash', () => {
  let s = { ...newSection('Addis — by season'), about: 'For Ann — draft' }
  s = addItem(s, chart('Crossplot — predicted vs reference'))
  s = patchItem(s, s.items[0].id, { note: 'Slope near 1 — offset −2.2' })
  s = addText(s)
  s = patchItem(s, s.items[1].id, { title: 'Dry — season', note: 'It grows — slowly' })
  // what goes on the slides: `item` is the chart itself, kept for its drawing
  const text = JSON.stringify(deckSlides(s, 'http://x/#tab=custom').map(({ item, ...slide }) => slide))
  assert.equal(text.includes('—'), false)
  assert.equal(deckText('Crossplot — predicted'), 'Crossplot: predicted')
})

test('a chart is a slide, with the reader\'s note as what is said over it', () => {
  let s = addItem(newSection('A'), chart('Crossplot'))
  s = patchItem(s, s.items[0].id, { note: 'Slope near 1' })
  const slides = deckSlides(s)
  assert.deepEqual(slides.map((x) => x.kind), ['title', 'chart'])
  assert.ok(slides[1].notes.startsWith('SAY: Slope near 1\nNOTES:'))
  assert.equal(slides[1].item, s.items[0])
  assert.ok(slides[0].notes.includes('Roadmap: Crossplot.'))
})

test('a heading alone divides, a heading with a body is a text slide, and an empty block is nothing', () => {
  let s = addText(addText(addText(newSection('A'))))
  s = patchItem(s, s.items[0].id, { title: 'Dry season' })
  s = patchItem(s, s.items[1].id, { title: 'Why', note: 'Because' })
  assert.deepEqual(deckSlides(s).map((x) => x.kind), ['title', 'divider', 'text'])
})
