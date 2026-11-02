import { createContext, useCallback, useContext, useLayoutEffect, useMemo, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

/**
 * Controls a chart's inner layout offers (a panel group's Panel axes and hoisted fit grouping, components/PanelGroup)
 * drawn in the enclosing ChartFrame's own options row, with the chart's other options, rather than in a row of their
 * own above the panels. The frame provides a slot; a control portals into it, so it keeps the React context of where
 * it is declared (the panel group, the chart's options) while its DOM sits in `.frame-controls`.
 *
 * Outside a ChartFrame (a panel group in page markup of its own) there is no slot: draw the controls in place.
 */
export interface FrameSlot {
  /** the element the controls go into; null while the frame has not drawn it yet */
  el: HTMLElement | null
  /** announce controls to draw, so a frame with no options of its own draws its options row; returns the release */
  claim: () => () => void
}
const Ctx = createContext<FrameSlot | null>(null)

/**
 * For ChartFrame: the slot's provider, whether anything is claimed (draw the options row even with no `controls`),
 * and the slot element to render at the end of that row.
 */
export function useFrameSlot(): { value: FrameSlot; claimed: boolean; slot: ReactNode } {
  const [el, setEl] = useState<HTMLElement | null>(null)
  const [count, setCount] = useState(0)
  const claim = useCallback(() => { setCount((n) => n + 1); return () => setCount((n) => n - 1) }, [])
  const value = useMemo(() => ({ el, claim }), [el, claim])
  return { value, claimed: count > 0, slot: <span ref={setEl} className="frame-hoisted" /> }
}

export const FrameSlotProvider = ({ value, children }: { value: FrameSlot; children: ReactNode }) => <Ctx.Provider value={value}>{children}</Ctx.Provider>

/**
 * Draw `children` in the enclosing frame's options row (when `on` and inside a ChartFrame); returns null when there
 * is no frame, so the caller draws them in place.
 */
export function useInFrameControls(on: boolean, children: ReactNode): { inFrame: boolean; node: ReactNode } {
  const frame = useContext(Ctx)
  useLayoutEffect(() => (frame && on ? frame.claim() : undefined), [frame, on])
  if (!frame) return { inFrame: false, node: null }
  return { inFrame: true, node: on && frame.el ? createPortal(children, frame.el) : null }
}
