import { useCallback, useState, type ReactNode, type RefObject } from 'react'

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
export function useTooltip(wrapRef: RefObject<HTMLElement>) {
  const [tip, setTip] = useState<{ x: number; y: number; body: TipBody } | null>(null)

  const show = useCallback(
    (e: React.MouseEvent, body: TipBody) => {
      const el = wrapRef.current
      if (!el) return
      const b = el.getBoundingClientRect()
      setTip({ x: e.clientX - b.left, y: e.clientY - b.top, body })
    },
    [wrapRef]
  )
  const hide = useCallback(() => setTip(null), [])

  const node = tip ? (
    <div className="tooltip" style={{ left: tip.x + 14, top: tip.y + 10 }}>
      {Array.isArray(tip.body)
        ? tip.body.map((l, i) => <div key={i}>{i === 0 ? <strong>{l}</strong> : l}</div>)
        : tip.body}
    </div>
  ) : null

  return { show, hide, node, active: tip !== null }
}
