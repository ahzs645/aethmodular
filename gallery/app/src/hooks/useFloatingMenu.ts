import { useLayoutEffect, useState, type CSSProperties, type RefObject } from 'react'

/** Keep a portalled menu inside the visible viewport, including an open mobile keyboard. */
export function useFloatingMenu(open: boolean, trigger: RefObject<Element>, menu: RefObject<HTMLElement>, heightLimit = 300) {
  const [place, setPlace] = useState<CSSProperties | null>(null)
  useLayoutEffect(() => {
    if (!open) { setPlace(null); return }
    const update = () => {
      const button = trigger.current, list = menu.current
      if (!button || !list) return
      const r = button.getBoundingClientRect(), viewport = window.visualViewport
      const viewTop = viewport?.offsetTop ?? 0, viewLeft = viewport?.offsetLeft ?? 0
      const viewWidth = viewport?.width ?? window.innerWidth, viewHeight = viewport?.height ?? window.innerHeight
      const below = Math.max(0, viewTop + viewHeight - r.bottom - 8), above = Math.max(0, r.top - viewTop - 8)
      const up = below < Math.min(heightLimit, list.offsetHeight) && above > below
      const maxHeight = Math.max(60, Math.min(heightLimit, viewHeight - 16, up ? above : below))
      const height = Math.min(list.offsetHeight, maxHeight), width = Math.min(list.offsetWidth, viewWidth - 16)
      const left = Math.max(viewLeft + 8, Math.min(r.left, viewLeft + viewWidth - width - 8))
      const top = Math.max(viewTop + 8, Math.min(up ? r.top - height - 4 : r.bottom + 4, viewTop + viewHeight - height - 8))
      setPlace({ position: 'fixed', left, top, bottom: 'auto', minWidth: Math.min(r.width, viewWidth - 16), maxWidth: Math.min(420, viewWidth - 16), maxHeight })
    }
    update()
    const onScroll = (e: Event) => { if (!menu.current?.contains(e.target as Node)) update() }
    const viewport = window.visualViewport
    const resize = new ResizeObserver(update)
    if (menu.current) resize.observe(menu.current)
    if (trigger.current) resize.observe(trigger.current)
    window.addEventListener('scroll', onScroll, true)
    window.addEventListener('resize', update)
    viewport?.addEventListener('resize', update)
    viewport?.addEventListener('scroll', update)
    return () => {
      resize.disconnect()
      window.removeEventListener('scroll', onScroll, true)
      window.removeEventListener('resize', update)
      viewport?.removeEventListener('resize', update)
      viewport?.removeEventListener('scroll', update)
    }
  }, [open, trigger, menu, heightLimit])
  return place
}
