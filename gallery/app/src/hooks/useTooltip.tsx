import { estimatorText, useEstimator } from '@/lib/regressionPrefs'
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode, type RefObject } from 'react'
import { createPortal } from 'react-dom'

export type TipBody = ReactNode | string[]

/**
 * One tooltip per chart. Every chart used to carry its own hover state and
 * the same getBoundingClientRect arithmetic; this keeps that in one place.
 *
 *   const tip = useTooltip(wrapRef)
 *   <circle onMouseEnter={(e) => tip.show(e, ['ETAD-0035', 'n = 3'])} onMouseLeave={tip.hide} />
 *   {tip.node}
 *
 * A string[] body renders one line each with the first line bold, which is
 * the convention the charts already follow.
 *
 * Where the card is and what it says live in a small store that only the card
 * (TooltipCard) reads, so a hover moving from point to point re-renders the
 * card alone, never the chart that owns it: a scatter matrix re-rendered its
 * legend and every histogram on each mouse move before. `active` re-renders
 * the owner only when the card opens or closes.
 */
export function useTooltip(wrapRef: RefObject<Element>, { large = false, id }: { large?: boolean; id?: string } = {}) {
  // Viewport coordinates: the tooltip is portalled to <body> with position:fixed,
  // so a chart inside a scrolling box (overflow-x: auto) can no longer clip it.
  const method = useEstimator()
  const [store] = useState(createTipStore)

  const show = useCallback(
    (e: React.MouseEvent | React.FocusEvent, body: TipBody) => {
      if (!wrapRef.current) return
      // Touch browsers synthesize mouseenter on a tap and may never send
      // mouseleave. Details belong in the tapped record, rather than a sticky tip.
      const source = e.nativeEvent as MouseEvent & { sourceCapabilities?: { firesTouchEvents?: boolean } }
      if ('clientX' in e && (source.sourceCapabilities?.firesTouchEvents || window.matchMedia('(hover: none)').matches)) return
      // A non-modal record drawer keeps its charts live, but background plot
      // readouts must not float over the open record.
      if (document.querySelector('.drawer') && !wrapRef.current.closest('.drawer, [role="dialog"]')) return
      const point = 'clientX' in e ? { x: e.clientX, y: e.clientY } : (() => {
        const rect = e.currentTarget.getBoundingClientRect()
        return { x: rect.left + rect.width / 2, y: rect.bottom }
      })()
      store.set({ ...point, body: Array.isArray(body) ? body.map(l => estimatorText(l, method)) : body })
    },
    [wrapRef, method, store]
  )
  const hide = useCallback(() => store.set(null), [store])
  const active = useSyncExternalStore(store.subscribe, () => store.get() !== null, () => false)
  // one element for the life of the chart: the owner's re-renders pass it through untouched
  const node = useMemo(() => <TooltipCard store={store} hide={hide} large={large} id={id} />, [store, hide, large, id])
  return useMemo(() => ({ show, hide, node, active }), [show, hide, node, active])
}

type Tip = { x: number; y: number; body: TipBody } | null
type TipStore = ReturnType<typeof createTipStore>
function createTipStore() {
  let tip: Tip = null
  const listeners = new Set<() => void>()
  return {
    get: () => tip,
    set: (next: Tip) => { if (next === tip) return; tip = next; for (const l of listeners) l() },
    subscribe: (l: () => void) => { listeners.add(l); return () => { listeners.delete(l) } },
  }
}

/** the card itself: the one part that follows the pointer */
function TooltipCard({ store, hide, large, id }: { store: TipStore; hide: () => void; large: boolean; id?: string }) {
  const tip = useSyncExternalStore(store.subscribe, store.get, store.get)
  const open = tip !== null

  // a fixed tooltip would otherwise float in place while the page scrolls under it
  useEffect(() => {
    if (!open) return
    window.addEventListener('scroll', hide, { passive: true, capture: true })
    window.addEventListener('blur', hide)
    window.addEventListener('gallery:dismiss-tooltips', hide)
    document.addEventListener('pointerdown', hide, true)
    document.addEventListener('click', hide, true)
    return () => {
      window.removeEventListener('scroll', hide, { capture: true })
      window.removeEventListener('blur', hide)
      window.removeEventListener('gallery:dismiss-tooltips', hide)
      document.removeEventListener('pointerdown', hide, true)
      document.removeEventListener('click', hide, true)
    }
  }, [open, hide])

  // open towards whichever side has room, so a point near the right or bottom edge keeps its label on screen.
  // Large readouts are wider than either side of a narrow window, so they open towards the roomier side
  // and wrap to fit it rather than running off the other edge.
  const LARGE_MAX = 560, GAP = 14, MARGIN = 8
  const roomRight = tip ? window.innerWidth - tip.x - GAP - MARGIN : 0
  const roomLeft = tip ? tip.x - GAP - MARGIN : 0
  const flipX = tip ? (large ? roomRight < LARGE_MAX && roomLeft > roomRight : tip.x > window.innerWidth - 300) : false
  const maxWidth = large ? Math.max(160, Math.min(LARGE_MAX, flipX ? roomLeft : roomRight)) : undefined
  const flipY = tip ? tip.y > window.innerHeight - 160 : false
  // after layout, the card's real size: keep it inside the window, on whichever side of the pointer has room
  const boxRef = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const el = boxRef.current
    if (!tip || !el) return
    // measure at the window's left edge, where nothing squeezes it, then place it
    Object.assign(el.style, { left: '0px', top: '0px', right: 'auto', bottom: 'auto' })
    const r = el.getBoundingClientRect(), W = window.innerWidth, H = window.innerHeight
    let left = tip.x + GAP
    if (left + r.width > W - MARGIN) left = tip.x - GAP - r.width
    left = Math.max(MARGIN, Math.min(left, W - MARGIN - r.width))
    let top = tip.y + 10
    if (top + r.height > H - MARGIN) top = tip.y - 10 - r.height
    top = Math.max(MARGIN, Math.min(top, H - MARGIN - r.height))
    Object.assign(el.style, { left: `${left}px`, top: `${top}px`, right: 'auto', bottom: 'auto' })
  }, [tip])
  return tip
    ? createPortal(
        <div
          ref={boxRef}
          id={id}
          role="tooltip"
          className={large ? 'tooltip large' : 'tooltip'}
          style={{
            position: 'fixed',
            ...(maxWidth ? { maxWidth } : {}),
            ...(flipX ? { right: window.innerWidth - tip.x + 14 } : { left: tip.x + 14 }),
            ...(flipY ? { bottom: window.innerHeight - tip.y + 10 } : { top: tip.y + 10 }),
          }}
        >
          {Array.isArray(tip.body)
            ? tip.body.map((l, i) => <div key={i}>{i === 0 ? <strong>{l}</strong> : l}</div>)
            : tip.body}
        </div>,
        document.body
      )
    : null
}
