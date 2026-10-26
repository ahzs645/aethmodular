import { useEffect, useRef } from 'react'

/** How far (px) a sheet has to be pulled down before letting go closes it. */
const DISMISS = 90

/**
 * The outer shell of the side drawers (a filter, a bin of filters, a record). On a wide window
 * it is the right-hand panel; on a phone the CSS turns it into a bottom sheet over a scrim, with
 * a grip that can be pulled down to close it. The scrim and grip are hidden on wide windows.
 */
export function DrawerShell({ label, onClose, children }: { label: string; onClose: () => void; children: React.ReactNode }) {
  const ref = useRef<HTMLElement>(null)
  const start = useRef<number | null>(null)
  useEffect(() => {
    window.dispatchEvent(new Event('gallery:dismiss-tooltips'))
  }, [label])
  const pull = (dy: number) => {
    const el = ref.current
    if (!el) return
    el.style.transition = dy ? 'none' : ''
    el.style.transform = dy ? `translateY(${dy}px)` : ''
  }
  return <>
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
}

/** A wide table or plot scrolls independently of the drawer and its header. */
export function DrawerScroll({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="drawer-scroll" tabIndex={0} role="region" aria-label={label}>{children}</div>
}
