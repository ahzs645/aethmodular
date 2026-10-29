/**
 * Hierarchical clustering for the gallery's trees: the species network's dendrogram and the
 * Research tab's site × site similarity matrix. Pure, node-tested (cluster.test.mjs).
 */

/** A node of the merge tree; leaves carry the item index, internal nodes the merge height. */
export interface Clade {
  idx?: number
  height: number
  children?: Clade[]
}

/**
 * Agglomerative clustering with average linkage on a distance matrix.
 * Start with singletons; repeatedly merge the two clusters whose mean
 * pairwise distance is smallest; record that mean as the merge height.
 * The cluster-to-cluster means are updated in place (Lance–Williams), so a few
 * hundred items stay quick.
 */
export function averageLinkage(dist: number[][]): Clade {
  const n = dist.length
  if (!n) return { height: 0 }
  // d[a][b]: mean distance between the clusters now held in slots a and b (Lance–Williams, size-weighted)
  const d = dist.map((row) => row.slice())
  const size = dist.map(() => 1)
  const node: Clade[] = dist.map((_, i) => ({ idx: i, height: 0 }))
  // live slots in first-seen order; a merge keeps the first partner's slot, so leaf order stays stable
  let live = dist.map((_, i) => i)
  while (live.length > 1) {
    let best = Infinity, bi = 0, bj = 1
    for (let i = 0; i < live.length; i++) {
      for (let j = i + 1; j < live.length; j++) {
        const v = d[live[i]][live[j]]
        if (v < best) { best = v; bi = i; bj = j }
      }
    }
    const a = live[bi], b = live[bj], s = size[a] + size[b]
    for (const k of live) if (k !== a && k !== b) d[a][k] = d[k][a] = (d[a][k] * size[a] + d[b][k] * size[b]) / s
    node[a] = { height: best, children: [node[a], node[b]] }
    size[a] = s
    live = live.filter((k) => k !== b)
  }
  return node[live[0]]
}

/** The leaves of a tree, left to right: the order that puts each cluster's members next to each other. */
export function leafOrder(root: Clade): number[] {
  const out: number[] = []
  const walk = (c: Clade) => { if (c.children) c.children.forEach(walk); else if (c.idx != null) out.push(c.idx) }
  walk(root)
  return out
}

/** Distance 1 − r from a correlation matrix (a missing r counts as unrelated, distance 1). */
export const corrDistance = (r: (number | null)[][]): number[][] =>
  r.map((row, i) => row.map((v, j) => (i === j ? 0 : v == null || !Number.isFinite(v) ? 1 : 1 - v)))
