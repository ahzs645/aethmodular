import { useMemo, useRef, type ReactNode } from 'react'
import { HitLayer } from '@/components/HitLayer'
import type { TipBody, useTooltip } from '@/hooks/useTooltip'
import { useFilterEdits } from '@/lib/filterEdits'
import { FocusLayer, PinRing, useHighlightActions } from '@/lib/highlight'

/**
 * The points of a chart that links its filters across the gallery: every crossplot, timeline and strip of dots draws
 * them through this, so they all hover, pin, open and right-click the same way and cost the same.
 *
 *   - `dots`: the marks, memoised by the chart. A hover never redraws them: a focused filter dims them all with one
 *     class on their group (styles.css .scatter-dots.has-focus) and is drawn again on top (lib/highlight FocusLayer),
 *     with a dashed ring when it is the pinned one (PinRing).
 *   - one pointer layer finds the point nearest the pointer (components/HitLayer). The marks take no pointer events,
 *     so a mark redrawn or moved under the pointer cannot leave the hover behind.
 *   - by default a hover shows `tipOf` and lights the filter up on every chart, a click opens its record (pinning
 *     it), and a right-click opens the filter menu (lib/filterEdits). A chart passes its own handlers to differ.
 *
 * The scatter matrix (charts/correlation/Correlogram SplomCell) uses the same three pieces in its own arrangement:
 * its hover marks go on an overlay above thirty panels of dots, so a hover does not repaint them.
 */
export function ScatterLayer<T>({ items, xy, idOf, dots, width, height, focus, focusR = 6, tip, tipOf, onHover, onLeave, onPick, menu = true,
  hit, reach, follow, active = true, children }: {
  items: readonly T[]
  /** each item's place in plot pixels, in the same order; memoise it (the pointer index is rebuilt when it changes) */
  xy: readonly (readonly [number, number])[]
  /** the filter an item stands for, as charts pass it to the highlight (lib/highlight); undefined: not linked */
  idOf: (t: T) => string | null | undefined
  /** the marks, memoised by the chart; drawn in a group the focus dims */
  dots: ReactNode
  width: number
  height: number
  /** the focused filter's mark, drawn over the dimmed dots at `at` */
  focus: (t: T, at: readonly [number, number]) => ReactNode
  /** the focused mark's radius, for the pin ring around it */
  focusR?: number | ((t: T) => number)
  tip?: ReturnType<typeof useTooltip>
  /** the tooltip for a hovered item; null: none */
  tipOf?: (t: T) => TipBody | null
  /** replace the default hover (tooltip + gallery-wide highlight) */
  onHover?: (e: React.MouseEvent<SVGRectElement>, t: T) => void
  onLeave?: () => void
  /** what a click does; default: open the filter's record, which pins it */
  onPick?: (t: T) => void
  /** right-click a point for the filter menu (default on) */
  menu?: boolean
  /** how far from a point the pointer still picks it, px (HitLayer's default 18) */
  hit?: number
  /** extra reach for a large mark (its radius) */
  reach?: (t: T) => number
  /** the tooltip follows the pointer on every move, not only when the point changes */
  follow?: boolean
  /** false: no pointer layer, e.g. while a brush selects */
  active?: boolean
  /** drawn over the dots and the focused mark, under the pointer layer: fits, labels */
  children?: ReactNode
}) {
  const hl = useHighlightActions()
  const edits = useFilterEdits()
  const dotsRef = useRef<SVGGElement>(null)
  // the linked filters (a hover or pin of one of these re-renders only the focus layer) and where each item sits
  const keys = useMemo(() => new Set(items.flatMap((t) => { const k = idOf(t); return k ? [k] : [] })), [items]) // eslint-disable-line react-hooks/exhaustive-deps
  const index = useMemo(() => new Map(items.map((t, i) => [t, i])), [items])
  const radius = (t: T) => (typeof focusR === 'function' ? focusR(t) : focusR)
  const id = (t: T) => idOf(t) ?? undefined
  return (
    <>
      <g ref={dotsRef} className="scatter-dots" pointerEvents="none">{dots}</g>
      <g className="scatter-focus" pointerEvents="none">
        <FocusLayer keys={keys} items={items} idOf={id} groupRef={dotsRef} render={(t, pinned) => {
          const at = xy[index.get(t) ?? -1]
          if (!at) return null
          return <>{focus(t, at)}{pinned && <PinRing cx={at[0]} cy={at[1]} r={radius(t)} />}</>
        }} />
      </g>
      {children}
      {active && (
        <HitLayer items={items} xy={xy} width={width} height={height} hit={hit} reach={reach} follow={follow}
          onHover={onHover ?? ((e, t) => {
            const body = tipOf?.(t)
            if (tip && body) tip.show(e, body)
            hl.setHover(idOf(t) ?? null)
          })}
          onLeave={onLeave ?? (() => { tip?.hide(); hl.setHover(null) })}
          onPick={onPick ?? ((t) => { const k = idOf(t); if (k) hl.openSample(k) })}
          onMenu={menu ? (e, t) => { const k = idOf(t); if (k) edits.openMenu(e, { ids: [k], title: k }) } : undefined} />
      )}
    </>
  )
}
