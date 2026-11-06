import test from 'node:test'
import assert from 'node:assert/strict'
import { placePointLabels, textWidth } from './fitLabel.ts'

// Fixed font metrics for geometry tests; Playwright checks the actual SVG font.
globalThis.document = { createElement: () => ({ getContext: () => ({ measureText: text => ({ width: text.length * 6 }) }) }) }
const boxes = labels => labels.map(l => {
  const w = textWidth(l.text, 10.5, 600)
  return { left: l.anchor === 'end' ? l.x - w : l.x, right: l.anchor === 'end' ? l.x : l.x + w, top: l.y - 10.5, bottom: l.y + 2.625 }
})
const overlap = (a, b) => a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom
function check(labels, points, width, height) {
  const bs = boxes(labels)
  bs.forEach((b, i) => {
    assert.ok(b.left >= 0 && b.right <= width && b.top >= 0 && b.bottom <= height)
    bs.slice(i + 1).forEach(other => assert.ok(!overlap(b, other), 'names must not overlap'))
    points.forEach(p => assert.ok(!overlap(b, { left: p.x - 10, right: p.x + 10, top: p.y - 10, bottom: p.y + 10 }), 'names must clear the stars'))
  })
}

test('close and tied medians near the right edge retain every name with connectors', () => {
  const points = ['Delhi', 'Addis Ababa', 'Beijing', 'JPL'].map((text, i) => ({ key: text, text, x: 235 - i, y: i < 2 ? 30 : 30 + i * 30 }))
  const labels = placePointLabels(points, 240, 140)
  assert.equal(labels.length, 4)
  assert.ok(labels.some(l => l.connector))
  check(labels, points, 240, 140)
  assert.deepEqual(labels, placePointLabels([...points].reverse(), 240, 140), 'layout must not depend on input order')
})

test('eight identical point positions fit without moving the data', () => {
  const points = Array.from({ length: 8 }, (_, i) => ({ key: `${i}`, text: `Site ${i}`, x: 230, y: 150 }))
  const before = structuredClone(points)
  const labels = placePointLabels(points, 240, 300)
  assert.equal(labels.length, 8)
  check(labels, points, 240, 300)
  assert.deepEqual(points, before)
})

test('labels at the top, bottom and left edges stay inside the plot', () => {
  const points = [{ key: 'a', text: 'Top', x: 0, y: 0 }, { key: 'b', text: 'Bottom', x: 240, y: 100 }]
  const labels = placePointLabels(points, 240, 100)
  assert.equal(labels.length, 2)
  check(labels, points, 240, 100)
})

test('an isolated name keeps its preferred position without a connector', () => {
  const [label] = placePointLabels([{ key: 'a', text: 'Delhi', x: 100, y: 50 }], 240, 100)
  assert.equal(label.y, 41)
  assert.equal(label.connector, null)
})

test('an impossible box omits the direct label rather than drawing outside it', () => {
  assert.deepEqual(placePointLabels([{ key: 'a', text: 'Addis Ababa', x: 10, y: 10 }], 20, 20), [])
  assert.deepEqual(placePointLabels([], 240, 100), [])
})
