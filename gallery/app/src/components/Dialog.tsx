import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type MutableRefObject, type ReactNode, type Ref, type RefObject } from 'react'
import { createPortal } from 'react-dom'

/**
 * Popovers that open over a dialog and close on their own Escape: while one is in the page, the dialog leaves
 * Escape and Tab to it (a Combobox list, the export dialog's Colors popover). Escape then closes the popover
 * first, and the next Escape the dialog.
 */
export const NESTED_POPOVERS = '.combobox-menu, .export-color-popover'

const FOCUSABLE = 'button:not(:disabled), input:not(:disabled):not([type="hidden"]), select:not(:disabled), textarea:not(:disabled), a[href], [tabindex]:not([tabindex="-1"])'

/** the open dialogs, innermost last: only the innermost answers Escape and Tab */
const open: object[] = []

/**
 * The gallery's modal dialog, one for every caller: portalled to <body> over a backdrop (a click on the backdrop
 * or Escape closes it), focus moved in on open, Tab and Shift-Tab kept inside, focus given back to what had it
 * (the button that opened it) on close, role="dialog" with aria-modal. The look is the caller's: `className` on
 * the box (e.g. "export-dialog ratio-dialog"), `backdropClassName` on the backdrop; both read the theme's CSS
 * variables, so the dark theme needs nothing more.
 *
 * Escape is taken on the window in the capture phase and stopped there, so a drawer, a menu or a page shortcut
 * behind the dialog does not also act on it; a dialog opened from a dialog closes alone. While an element matching
 * `escapeGuard` is in the page (by default NESTED_POPOVERS) the dialog leaves the key to it.
 */
export function Dialog({
  onClose, onBackdrop, label, labelledBy, className, backdropClassName = 'export-backdrop', style, initialFocus,
  escapeGuard = NESTED_POPOVERS, modal = true, boxRef, children,
}: {
  /** Escape, and a click on the backdrop unless `onBackdrop` is given */
  onClose: () => void
  /** a click on the backdrop, when it should differ from Escape */
  onBackdrop?: () => void
  /** aria-label of the dialog (or name its heading with `labelledBy`) */
  label?: string
  labelledBy?: string
  className?: string
  backdropClassName?: string
  style?: CSSProperties
  /** what takes focus on open: a selector inside the dialog or a ref; the dialog box itself when absent, missing or disabled */
  initialFocus?: string | RefObject<HTMLElement | null>
  /** while an element matching this is in the page, Escape and Tab are left to it ('' never) */
  escapeGuard?: string
  /** false where something outside the dialog opens over it and must stay reachable (the sample drawer) */
  modal?: boolean
  boxRef?: Ref<HTMLDivElement>
  children: ReactNode
}) {
  const box = useRef<HTMLDivElement | null>(null)
  // read while rendering: an autoFocus child takes focus before any effect runs
  const [opener] = useState(() => (typeof document === 'undefined' ? null : document.activeElement as HTMLElement | null))
  const live = useRef({ onClose, escapeGuard })
  live.current = { onClose, escapeGuard }
  const setBox = useCallback((el: HTMLDivElement | null) => {
    box.current = el
    if (typeof boxRef === 'function') boxRef(el)
    else if (boxRef) (boxRef as MutableRefObject<HTMLDivElement | null>).current = el
  }, [boxRef])

  useLayoutEffect(() => {
    const el = box.current
    if (!el || el.contains(document.activeElement)) return
    const target = typeof initialFocus === 'string' ? el.querySelector<HTMLElement>(initialFocus) : initialFocus?.current
    target?.focus()
    // a disabled target (a Download button before its file is ready) takes no focus: the dialog holds it instead
    if (!el.contains(document.activeElement)) el.focus()
    // focus goes back to the opener only when it is still in the dialog (or nowhere): a choice that moved it elsewhere keeps it
    return () => {
      const active = document.activeElement
      if (opener?.isConnected && (!active || active === document.body || box.current?.contains(active))) opener.focus({ preventScroll: true })
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const me = {}
    open.push(me)
    const key = (e: KeyboardEvent) => {
      if ((e.key !== 'Escape' && e.key !== 'Tab') || e.defaultPrevented || e.isComposing || open[open.length - 1] !== me) return
      const { escapeGuard: guard, onClose: close } = live.current
      const el = box.current
      if (!el || (guard && document.querySelector(guard))) return
      if (e.key === 'Escape') { e.preventDefault(); e.stopImmediatePropagation(); close(); return }
      const active = document.activeElement
      const inside = !!active && el.contains(active)
      // focus in something opened over the dialog (the sample drawer) keeps its own Tab order
      if (!inside && active && active !== document.body) return
      const items = [...el.querySelectorAll<HTMLElement>(FOCUSABLE)].filter((x) => x.tabIndex >= 0 && x.getClientRects().length > 0)
      const first = items[0], last = items[items.length - 1]
      if (!first) { e.preventDefault(); el.focus(); return }
      if (e.shiftKey && (!inside || active === first || active === el)) { e.preventDefault(); last.focus() }
      else if (!e.shiftKey && (!inside || active === last)) { e.preventDefault(); first.focus() }
    }
    window.addEventListener('keydown', key, true)
    return () => { window.removeEventListener('keydown', key, true); open.splice(open.indexOf(me), 1) }
  }, [])

  return createPortal(
    <div className={backdropClassName} onMouseDown={(e) => { if (e.target === e.currentTarget) (onBackdrop ?? live.current.onClose)() }}>
      <div ref={setBox} className={className} role="dialog" aria-modal={modal || undefined} aria-label={label} aria-labelledby={labelledBy}
        tabIndex={-1} style={{ outline: 'none', ...style }}>
        {children}
      </div>
    </div>,
    document.body,
  )
}
