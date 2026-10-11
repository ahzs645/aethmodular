import { estimatorText, useEstimator } from '@/lib/regressionPrefs'
import { useCallback, useEffect, useState, type ReactNode, type RefObject } from 'react'
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
 */
export function useTooltip(wrapRef: RefObject<HTMLElement>, { large = false }: { large?: boolean } = {}) {
  // Viewport coordinates: the tooltip is portalled to <body> with position:fixed,
  // so a chart inside a scrolling box (overflow-x: auto) can no longer clip it.
  const method = useEstimator()
  const [tip, setTip] = useState<{ x: number; y: number; body: TipBody } | null>(null)

  const show = useCallback(
    (e: React.MouseEvent, body: TipBody) => {
      if (!wrapRef.current) return
      setTip({ x: e.clientX, y: e.clientY, body: Array.isArray(body) ? body.map(l => estimatorText(l, method)) : body })
    },
    [wrapRef, method]
  )
  const hide = useCallback(() => setTip(null), [])

  // a fixed tooltip would otherwise float in place while the page scrolls under it
  useEffect(() => {
    if (!tip) return
    window.addEventListener('scroll', hide, { passive: true, capture: true })
    return () => window.removeEventListener('scroll', hide, { capture: true })
  }, [tip, hide])

  // open towards whichever side has room, so a point near the right or bottom edge keeps its label on screen.
  // Large readouts are wider than either side of a narrow window, so they open towards the roomier side
  // and wrap to fit it rather than running off the other edge.
  const LARGE_MAX = 560, GAP = 14, MARGIN = 8
  const roomRight = tip ? window.innerWidth - tip.x - GAP - MARGIN : 0
  const roomLeft = tip ? tip.x - GAP - MARGIN : 0
  const flipX = tip ? (large ? roomRight < LARGE_MAX && roomLeft > roomRight : tip.x > window.innerWidth - 300) : false
  const maxWidth = large ? Math.max(160, Math.min(LARGE_MAX, flipX ? roomLeft : roomRight)) : undefined
  const flipY = tip ? tip.y > window.innerHeight - 160 : false
  const node = tip
    ? createPortal(
        <div
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

  return { show, hide, node, active: tip !== null }
}
