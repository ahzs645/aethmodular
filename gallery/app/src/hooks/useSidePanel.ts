import { useEffect, useState } from 'react'

/**
 * Height of the sticky app header. The side panels and the table of contents
 * sit just below it, and it changes whenever the tab row wraps: on resize, and
 * when a hidden tab (Census) joins the row because it is open.
 */
export function useHeaderHeight(fallback = 96) {
  const [h, setH] = useState(fallback)
  useEffect(() => {
    const el = document.querySelector('.app-header') as HTMLElement | null
    if (!el) return
    const measure = () => setH(el.offsetHeight)
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return h
}

const OPEN_KEY = 'gallery.filters.open'
const readOpen = () => { try { return localStorage.getItem(OPEN_KEY) === '1' } catch { return false } }
const writeOpen = (v: boolean) => { try { localStorage.setItem(OPEN_KEY, v ? '1' : '0') } catch { /* private window */ } }

const OPEN_EVENT = 'gallery-side-panel'

/**
 * Whether the left side panel is open. Filters and Fits share it: it is the same panel on
 * different tabs. Every caller sees one value, so a button elsewhere (the Dashboard's header)
 * can open the panel the dock draws.
 */
export function useSidePanelOpen(): [boolean, (v: boolean) => void] {
  const [open, setOpenState] = useState(readOpen)
  useEffect(() => {
    const sync = (e: Event) => setOpenState((e as CustomEvent<boolean>).detail)
    window.addEventListener(OPEN_EVENT, sync)
    return () => window.removeEventListener(OPEN_EVENT, sync)
  }, [])
  return [open, (v: boolean) => { writeOpen(v); window.dispatchEvent(new CustomEvent(OPEN_EVENT, { detail: v })) }]
}
