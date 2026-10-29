/**
 * Grouping map markers that would overlap at the current zoom, as PGMaps' cluster donuts do: a group draws
 * as one marker until the map is zoomed in far enough for its members to sit apart. Unlike MapLibre's own
 * clustering (a fixed pixel radius, counts only), a group forms where the markers' own discs collide, so a
 * map of differently sized donuts groups exactly the ones that hide each other.
 */

/** MapLibre's world size at zoom 0, in CSS pixels (512 px tiles) */
const TILE = 512
const MAX_LAT = 85.051129

/** world pixel position of a lon/lat at a zoom (Web Mercator) */
export function worldPixel(lon: number, lat: number, zoom: number): [number, number] {
  const size = TILE * 2 ** zoom
  const s = Math.sin((Math.max(-MAX_LAT, Math.min(MAX_LAT, lat)) * Math.PI) / 180)
  return [((lon + 180) / 360) * size, (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * size]
}

/** the lon/lat of a world pixel position at a zoom (the inverse of worldPixel) */
export function fromWorldPixel(x: number, y: number, zoom: number): [number, number] {
  const size = TILE * 2 ** zoom
  const n = Math.PI - (2 * Math.PI * y) / size
  return [(x / size) * 360 - 180, (180 / Math.PI) * Math.atan(Math.sinh(n))]
}

export interface MapGroup<T> {
  /** the items drawn as this marker: one for an ungrouped item */
  members: T[]
  /** where the marker sits: the members' centroid on screen */
  lon: number
  lat: number
}

/**
 * Merge, closest pair first, any two groups whose discs overlap at `zoom` (centre distance under the sum
 * of their radii plus `gap` pixels), until none do. `radiusOf` is a group's drawn radius, so a merged
 * group that grows can take in a further neighbour. Members keep the input order.
 */
export function groupOverlapping<T extends { lon: number; lat: number }>(
  items: readonly T[], zoom: number, radiusOf: (members: T[]) => number, gap = 0,
): MapGroup<T>[] {
  type G = { idx: number[]; x: number; y: number; r: number }
  const px = items.map((it) => worldPixel(it.lon, it.lat, zoom))
  const make = (idx: number[]): G => ({
    idx,
    x: idx.reduce((a, i) => a + px[i][0], 0) / idx.length,
    y: idx.reduce((a, i) => a + px[i][1], 0) / idx.length,
    r: radiusOf(idx.map((i) => items[i])),
  })
  const groups = items.map((_, i) => make([i]))
  for (;;) {
    let best: [number, number] | null = null, bestD = Infinity
    for (let a = 0; a < groups.length; a++) {
      for (let b = a + 1; b < groups.length; b++) {
        const d = Math.hypot(groups[a].x - groups[b].x, groups[a].y - groups[b].y)
        if (d < groups[a].r + groups[b].r + gap && d < bestD) { bestD = d; best = [a, b] }
      }
    }
    if (!best) break
    const [a, b] = best
    groups[a] = make([...groups[a].idx, ...groups[b].idx].sort((p, q) => p - q))
    groups.splice(b, 1)
  }
  return groups.map((g) => {
    const [lon, lat] = g.idx.length === 1 ? [items[g.idx[0]].lon, items[g.idx[0]].lat] : fromWorldPixel(g.x, g.y, zoom)
    return { members: g.idx.map((i) => items[i]), lon, lat }
  })
}
