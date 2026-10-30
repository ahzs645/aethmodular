import { useMemo, useRef } from 'react'
import * as d3 from 'd3'

/**
 * The pointer layer a dense scatter shares (Scatterplot's panels, Crossplot, the scatter matrix's cells): one
 * transparent rect over the plot and one Delaunay over the drawn points, so the nearest point within `hit` px
 * answers a hover, a click and a right-click. The dots under it take no pointer events. A handler on every dot made
 * each move a hit test over thousands of them, and a dot redrawn or moved under the pointer never sent its
 * mouseleave, which left the hover on every chart.
 *
 * The rect is filled transparent, so a click on empty plot reaches the gallery's background click (lib/highlight),
 * which lets go of the pin.
 */
export function HitLayer<T>({ items, xy, width, height, hit = 18, reach, follow = false, onHover, onLeave, onPick, onMenu }: {
  items: readonly T[]
  /** each item's position in plot pixels, in the same order; memoise it, as the index is rebuilt when it changes */
  xy: readonly (readonly [number, number])[]
  width: number
  height: number
  /** how far from a point the pointer still picks it, px */
  hit?: number
  /** extra reach for a large mark (its radius) */
  reach?: (t: T) => number
  /** call onHover on every move over a point (a tooltip that follows the pointer), not only when the point changes */
  follow?: boolean
  onHover: (e: React.MouseEvent<SVGRectElement>, t: T) => void
  /** once, when the pointer leaves the last point */
  onLeave: () => void
  onPick?: (t: T, e: React.MouseEvent<SVGRectElement>) => void
  onMenu?: (e: React.MouseEvent<SVGRectElement>, t: T) => void
}) {
  const delaunay = useMemo(() => d3.Delaunay.from(xy, (p) => p[0], (p) => p[1]), [xy])
  const current = useRef<T | null>(null)
  const nearest = (e: React.MouseEvent<SVGRectElement>): T | null => {
    if (items.length === 0) return null
    const [mx, my] = d3.pointer(e)
    const i = delaunay.find(mx, my)
    const t = items[i], p = xy[i]
    if (t === undefined || !p) return null
    return Math.hypot(p[0] - mx, p[1] - my) <= hit + (reach?.(t) ?? 0) ? t : null
  }
  const move = (e: React.MouseEvent<SVGRectElement>, t: T | null) => {
    const was = current.current
    current.current = t
    // the cursor follows the point under it without a re-render
    e.currentTarget.style.cursor = t !== null && onPick ? 'pointer' : ''
    if (t !== null) { if (t !== was || follow) onHover(e, t) }
    else if (was !== null) onLeave()
  }
  return (
    <rect width={width} height={height} fill="transparent"
      onMouseMove={(e) => move(e, nearest(e))}
      onMouseLeave={(e) => move(e, null)}
      onClick={(e) => {
        const t = nearest(e)
        if (t !== null && onPick) { e.stopPropagation(); onPick(t, e) }
        else move(e, null)
      }}
      onContextMenu={(e) => { const t = nearest(e); if (t !== null) onMenu?.(e, t) }}
    />
  )
}
