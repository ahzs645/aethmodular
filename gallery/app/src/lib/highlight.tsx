import { createContext, useCallback, useContext, useEffect, useLayoutEffect, useMemo, useState, useSyncExternalStore, type ReactNode } from 'react'
import { INK, dotOpacity, type DotLevel } from '@/lib/theme'

/**
 * Cross-chart linking. Hovering a filter in one chart lights the same filter
 * up in every other chart on the page; clicking opens its detail drawer and
 * pins it, which also puts its record in the subset bar. This is the
 * "cross-graph interaction" the gallery's interactivity page describes, and
 * the thing a static notebook figure can never do: "which point in the
 * crossplot is that spike in the timeseries?"
 *
 * `selected` is the many-filter version of the same idea: a brushed region on
 * the scatterplot (or a brushed axis on the parallel coordinates) becomes a
 * set of ids, and every other chart on the tab dims what is not in it. It is
 * a highlight, not a filter — the statistics still use the whole subset.
 */
/**
 * What a chart knows about one physical filter when it opens the drawer. SPARTAN
 * filters in filters.json get the full drawer with this shown on top; filters the
 * export does not carry (IMPROVE test filters, Bishoftu, Adama) get a drawer built
 * from these fields alone.
 */
export interface FilterRecord {
  /** the id as the chart knows it, e.g. ETAD-0144-8, improve:1851923, J1269 */
  id: string
  /** the chart it came from, shown as the section title */
  source: string
  site?: string
  date?: string
  fields: [string, string | number | null | undefined][]
  note?: string
}

/**
 * The hovered filter changes on every mouse move, so it lives in a small store outside React state: a hover
 * re-renders only the charts that ask for it, and a chart that passes its own ids to useHighlight(keys) only
 * when the hovered filter is one of them (a Beijing point does not redraw the Addis panels).
 */
interface HoverStore { get: () => string | null; set: (id: string | null) => void; subscribe: (f: () => void) => () => void }
function createHoverStore(): HoverStore {
  let value: string | null = null
  const subs = new Set<() => void>()
  return {
    get: () => value,
    set: (id) => { if (id === value) return; value = id; subs.forEach((f) => f()) },
    subscribe: (f) => { subs.add(f); return () => { subs.delete(f) } },
  }
}

interface HighlightContext extends Omit<HighlightState, 'hoverId' | 'focusId'> { store: HoverStore }

interface HighlightState {
  hoverId: string | null
  pinnedId: string | null
  /** ids picked by a brush somewhere on the page; null = no brush active */
  selected: Set<string> | null
  setHover: (id: string | null) => void
  setPinned: (id: string | null) => void
  /** close the detail drawer and clear its pin and transient highlight */
  dismissSample: () => void
  setSelected: (ids: Set<string> | null) => void
  /** pin the sample and open its detail drawer */
  openSample: (id: string) => void
  /** open the drawer for any filter, with what the calling chart knows about it */
  openRecord: (rec: FilterRecord) => void
  /** the id that should currently read as highlighted (pinned wins) */
  focusId: string | null
}

const defaultStore = createHoverStore()
const Ctx = createContext<HighlightContext>({
  store: defaultStore,
  pinnedId: null,
  selected: null,
  setHover: defaultStore.set,
  setPinned: () => {},
  dismissSample: () => {},
  setSelected: () => {},
  openSample: () => {},
  openRecord: () => {},
})

export function HighlightProvider({ children, pinnedId, setPinned, openSample, onDismiss, openRecord = () => {} }: {
  children: ReactNode
  pinnedId: string | null
  setPinned: (id: string | null) => void
  openSample: (id: string) => void
  onDismiss: () => void
  openRecord?: (rec: FilterRecord) => void
}) {
  const store = useMemo(createHoverStore, [])
  const setHover = store.set
  const [selected, setSelectedState] = useState<Set<string> | null>(null)
  const dismissSample = useCallback(() => {
    setHover(null)
    onDismiss()
  }, [onDismiss])
  const pin = useCallback((id: string | null) => {
    if (id === null) dismissSample()
    else setPinned(id)
  }, [dismissSample, setPinned])
  // Closing from the drawer or subset bar also removes the hover highlight.
  useEffect(() => { if (pinnedId === null) setHover(null) }, [pinnedId])
  useEffect(() => {
    const onBackgroundClick = (e: MouseEvent) => {
      const target = e.target
      if (!(target instanceof Element)) return
      // Controls, the detail drawer and D3 brushes keep their own interactions.
      if (target.closest('aside, button, a, input, select, label, summary, [role="button"], [role="radio"], .frame-controls, .legend, .brush-layer, .brush')) return
      if (getComputedStyle(target).cursor === 'pointer') return
      const tag = target.tagName.toLowerCase()
      const plotBackground = tag === 'svg' || tag === 'g' || (tag === 'rect' && target.getAttribute('fill') === 'transparent')
      const pageBackground = ['html', 'body', 'main', 'section', 'div'].includes(tag) && !target.closest('svg')
      if (plotBackground || pageBackground) dismissSample()
    }
    // Bubble after point handlers, which can stop propagation for a hit layer.
    document.addEventListener('click', onBackgroundClick)
    return () => document.removeEventListener('click', onBackgroundClick)
  }, [dismissSample])
  const setSelected = (ids: Set<string> | null) => setSelectedState(ids && ids.size > 0 ? ids : null)
  // the value does not change on hover: only pins, brushes and the handlers below change it
  const value = useMemo<HighlightContext>(
    () => ({ store, pinnedId, selected, setHover, setPinned: pin, dismissSample, setSelected, openSample, openRecord }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [store, pinnedId, selected, pin, dismissSample, openSample, openRecord]
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

/**
 * The page's highlight state. With `keys` (the ids a chart draws, in the form it passes to setHover), `hoverId`
 * is the hovered filter only when it is one of them, and the chart re-renders only then: hovering elsewhere
 * leaves it alone. Without `keys` every hover re-renders the caller, as before.
 */
const NO_HOVER: ReadonlySet<string> = new Set()
/**
 * The pin, selection and handlers without the hover: a component that only acts (pins, opens, brushes) uses this,
 * so a hover anywhere in the gallery does not re-render it. `useHighlight()` with no keys re-renders on every hover.
 */
export const useHighlightActions = (): HighlightState => useHighlight(NO_HOVER)

export function useHighlight(keys?: ReadonlySet<string> | null): HighlightState {
  const c = useContext(Ctx)
  const read = () => { const h = c.store.get(); return !keys || (h !== null && keys.has(h)) ? h : null }
  const hoverId = useSyncExternalStore(c.store.subscribe, read, read)
  return useMemo(() => ({ ...c, hoverId, focusId: hoverId ?? c.pinnedId }), [c, hoverId])
}

/**
 * Mark styling for a point given the current focus and brush selection:
 * bigger + opaque when it is the one, dimmed when another is or when a brush
 * selection exists that it is not part of.
 */
/**
 * Marks in draw order with the focused one (hovered or pinned) last, so it paints over its
 * neighbours; SVG has no z-index. Each comes with its original index, for charts that place a
 * mark by its index (deterministic jitter). Unfocused, the order is unchanged.
 */
export function focusLast<T>(items: readonly T[], idOf: (t: T) => string | undefined, focusId: string | null): [T, number][] {
  const out = items.map((t, i): [T, number] => [t, i])
  if (!focusId) return out
  const k = items.findIndex((t) => idOf(t) === focusId)
  if (k < 0 || k === items.length - 1) return out
  const [f] = out.splice(k, 1)
  out.push(f)
  return out
}

type DotStyle = { r: number; opacity: number; stroke: string; strokeWidth: number; dot?: DotLevel }
export function focusStyle(id: string, focusId: string | null, { r, level = 'standard' }: { r: number; level?: DotLevel } = { r: 4 }, selected: Set<string> | null = null): DotStyle {
  const opacity = dotOpacity(level)
  if (focusId === id) return { r: r * 1.9, opacity: 1, stroke: INK.fit, strokeWidth: 2 }
  if (focusId) return { r, opacity: Math.min(opacity, 0.25), stroke: INK.panel, strokeWidth: 0.6 }
  if (selected) {
    if (selected.has(id)) return { r, opacity: Math.max(opacity, 0.9), stroke: INK.fit, strokeWidth: 0.8 }
    return { r, opacity: Math.min(opacity, 0.1), stroke: INK.panel, strokeWidth: 0.4 }
  }
  // `dot` is the data-dot tag lib/themeCss lifts on the dark page; leave it off a mark scaled by a dim
  return { r, opacity, ...dotRim(), dot: level }
}

/**
 * The rim a filled scatter dot wears when it is not focused (docs/gallery-chart-standards.md §5):
 * the panel colour, so overlapping dots stay apart and look the same in both themes. A rim in the
 * dot's own colour or a dark hex reads as a bright ring or vanishes on the dark panel. Hollow dots,
 * rims that encode something, and dots under ~2.5 px in dense clouds keep their own stroke.
 */
export const dotRim = () => ({ stroke: INK.panel, strokeWidth: 0.8 })

/**
 * The fill of a dot at an opacity level (lib/theme DOT_OPACITY), times `scale` (a legend's dim): the
 * light step, tagged data-dot so lib/themeCss lifts it on the dark page. A dimmed dot (scale < 1) is
 * left untagged, or the CSS step would undo the dim.
 */
export const dotFill = (level: DotLevel, scale = 1) => ({ fillOpacity: dotOpacity(level) * scale, 'data-dot': scale === 1 ? level : undefined })

/** SPARTAN ids carry a replicate suffix outside filters.json (ETAD-0144-8); the export keys on ETAD-0144. */
export { baseFilterId } from './filterId'

/**
 * The focused mark (hovered here or on another chart, or pinned) drawn over a chart's dots, which it dims with
 * one class on their group (styles.css .has-focus). The only part of the chart that follows the hover: a hover
 * then re-renders this one mark, not every dot, the axes or the fits. `keys` are the ids the chart draws.
 */
export function FocusLayer<T>({ keys, items, idOf, groupRef, render }: {
  keys: ReadonlySet<string>; items: readonly T[]; idOf: (t: T) => string | undefined
  groupRef: React.RefObject<SVGGElement | null>; render: (t: T) => ReactNode
}) {
  const { focusId } = useHighlight(keys)
  const focused = focusId ? items.find((t) => idOf(t) === focusId) ?? null : null
  useLayoutEffect(() => { groupRef.current?.classList.toggle('has-focus', focused !== null) }, [focused, groupRef])
  return focused === null ? null : <>{render(focused)}</>
}
