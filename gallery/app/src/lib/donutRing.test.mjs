import assert from 'node:assert/strict'
import { test } from 'node:test'
import { assignRing, ringSlots } from './donutRing.ts'

const crosses = (a, b, c, d) => {
  const o = (p, q, r) => Math.sign((q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x))
  return o(a, b, c) !== o(a, b, d) && o(c, d, a) !== o(c, d, b)
}

test('ring slots run along the top and bottom and down both sides', () => {
  const slots = ringSlots(1000, 500, { w: 100, h: 100 })
  assert.equal(slots.filter((s) => s.side === 'top').length, 10)
  assert.equal(slots.filter((s) => s.side === 'left').length, 3)
  assert.equal(slots.length, 26)
  for (const s of slots) assert.ok(s.x > 0 && s.x < 1000 && s.y > 0 && s.y < 500)
})

test('each point gets its own slot, nearest where order allows, and leaders do not cross', () => {
  const slots = ringSlots(1200, 600, { w: 100, h: 100 })
  const centre = { x: 600, y: 300 }
  const points = [{ x: 200, y: 200 }, { x: 900, y: 180 }, { x: 610, y: 420 }, { x: 300, y: 380 }, { x: 1000, y: 400 }, { x: 640, y: 150 }]
  const pick = assignRing(points, slots, centre)
  assert.equal(new Set(pick).size, points.length)
  for (let i = 0; i < points.length; i++)
    for (let j = i + 1; j < points.length; j++)
      assert.ok(!crosses(points[i], slots[pick[i]], points[j], slots[pick[j]]), `leaders ${i} and ${j} cross`)
  // a point near the top edge takes a top slot right above it
  assert.equal(slots[pick[5]].side, 'top')
  assert.ok(Math.abs(slots[pick[5]].x - 640) <= 60)
})

test('too few slots is an error, no points is no slots', () => {
  assert.deepEqual(assignRing([], [{ x: 0, y: 0 }], { x: 0, y: 0 }), [])
  assert.throws(() => assignRing([{ x: 0, y: 0 }, { x: 1, y: 1 }], [{ x: 0, y: 0 }], { x: 0, y: 0 }))
})
