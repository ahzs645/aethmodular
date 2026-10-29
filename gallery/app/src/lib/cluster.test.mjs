import test from 'node:test'
import assert from 'node:assert/strict'
import { averageLinkage, leafOrder, corrDistance } from './cluster.ts'

/** the original member-pair scan, as the reference */
function naive(dist) {
  let cl = dist.map((_, i) => ({ m: [i], node: { idx: i, height: 0 } }))
  while (cl.length > 1) {
    let best = Infinity, bi = 0, bj = 1
    for (let i = 0; i < cl.length; i++) for (let j = i + 1; j < cl.length; j++) {
      let sum = 0
      for (const a of cl[i].m) for (const b of cl[j].m) sum += dist[a][b]
      const avg = sum / (cl[i].m.length * cl[j].m.length)
      if (avg < best) { best = avg; bi = i; bj = j }
    }
    const merged = { m: [...cl[bi].m, ...cl[bj].m], node: { height: best, children: [cl[bi].node, cl[bj].node] } }
    cl = cl.flatMap((c, k) => (k === bi ? [merged] : k === bj ? [] : [c]))
  }
  return cl[0].node
}
const heights = (c) => (c.children ? [c.height, ...c.children.flatMap(heights)] : [])

test('average linkage matches the member-pair definition', () => {
  let seed = 7
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
  const pts = Array.from({ length: 30 }, () => [rnd(), rnd()])
  const dist = pts.map((p) => pts.map((q) => Math.hypot(p[0] - q[0], p[1] - q[1])))
  const a = averageLinkage(dist), b = naive(dist)
  assert.deepEqual(leafOrder(a), leafOrder(b))
  heights(a).forEach((h, i) => assert.ok(Math.abs(h - heights(b)[i]) < 1e-12))
})

test('two tight groups come out as neighbours, and corrDistance treats a gap as unrelated', () => {
  const r = [[1, 0.99, 0.1, 0.1], [0.99, 1, 0.1, null], [0.1, 0.1, 1, 0.98], [0.1, null, 0.98, 1]]
  const d = corrDistance(r)
  assert.equal(d[1][3], 1)
  assert.equal(d[2][2], 0)
  const order = leafOrder(averageLinkage(d))
  assert.equal(Math.abs(order.indexOf(0) - order.indexOf(1)), 1)
  assert.equal(Math.abs(order.indexOf(2) - order.indexOf(3)), 1)
})
