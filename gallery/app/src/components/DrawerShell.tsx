import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'

/** How far (px) a sheet has to be pulled down before letting go closes it. */
const DISMISS = 90

/**
 * The outer shell of the side drawers (a filter, a bin of filters, a record). On a wide window
 * it is the right-hand panel; on a phone the CSS turns it into a bottom sheet over a scrim, with
 * a grip that can be pulled down to close it. The scrim and grip are hidden on wide windows.
 */
export function DrawerShell({ label, onClose, children, portal = false }: {
  label: string; onClose: () => void; children: React.ReactNode
  /** A drawer opened inside a chart floats outside its clipping and single-chart layout. */
  portal?: boolean
}) {
  const ref = useRef<HTMLElement>(null)
  const close = useRef(onClose)
  close.current = onClose
  const start = useRef<number | null>(null)
  useEffect(() => {
    window.dispatchEvent(new Event('gallery:dismiss-tooltips'))
    const other = (e: Event) => { if ((e as CustomEvent).detail !== ref.current) close.current() }
    window.addEventListener('gallery:drawer-open', other)
    window.dispatchEvent(new CustomEvent('gallery:drawer-open', { detail: ref.current }))
    return () => window.removeEventListener('gallery:drawer-open', other)
  }, [label])
  useEffect(() => {
    if (!portal) return
    const previous = document.activeElement
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape' && !e.defaultPrevented) close.current() }
    window.addEventListener('keydown', key)
    ref.current?.querySelector<HTMLElement>('button')?.focus({ preventScroll: true })
    return () => {
      window.removeEventListener('keydown', key)
      if (previous instanceof HTMLElement && previous.isConnected) previous.focus({ preventScroll: true })
    }
  }, [portal])
  const pull = (dy: number) => {
    const el = ref.current
    if (!el) return
    el.style.transition = dy ? 'none' : ''
    el.style.transform = dy ? `translateY(${dy}px)` : ''
  }
  const body = <>
    <div className="sheet-scrim" onClick={onClose} aria-hidden />
    <aside ref={ref} className="drawer" role="dialog" aria-label={label}>
      <div className="sheet-grip" aria-hidden
        onPointerDown={(e) => { start.current = e.clientY; try { e.currentTarget.setPointerCapture(e.pointerId) } catch { /* a synthetic pointer */ } }}
        onPointerMove={(e) => { if (start.current !== null) pull(Math.max(0, e.clientY - start.current)) }}
        onPointerUp={(e) => { const dy = start.current === null ? 0 : e.clientY - start.current; start.current = null; pull(0); if (dy > DISMISS) onClose() }}
        onPointerCancel={() => { start.current = null; pull(0) }} />
      {children}
    </aside>
  </>
  return portal ? createPortal(body, document.body) : body
}

/** A wide table or plot scrolls independently of the drawer and its header. */
export function DrawerScroll({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="drawer-scroll" tabIndex={0} role="region" aria-label={label}>{children}</div>
}
