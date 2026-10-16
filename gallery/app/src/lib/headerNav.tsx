import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'
import { useSearchParam } from '@/lib/url'

/**
 * State the header shares with the page below it.
 * - The "On this tab" contents sidebar (components/PageToc): hidden unless the URL says
 *   toc=shown; its Show / Hide button sits in the header beside Views, offered only while the
 *   tab has a contents list.
 * - Navigation for the ⌘K search (components/CommandPalette): open a tab or meeting page and
 *   scroll to a chart once it has drawn.
 */
export interface ChartTarget { tab: string; meeting?: string; id?: string; title?: string }
interface HeaderNav {
  tocShown: boolean
  setTocShown: (v: boolean) => void
  /** the tab has a contents list (PageToc with more than one section) */
  tocPresent: boolean
  setTocPresent: (v: boolean) => void
  tabs: { key: string; label: string; title?: string }[]
  meetings: { key: string; label: string }[]
  goTo: (t: ChartTarget) => void
}
const Ctx = createContext<HeaderNav | null>(null)
export const useHeaderNav = () => useContext(Ctx)

export function HeaderNavProvider({ tabs, meetings, open, children }: {
  tabs: HeaderNav['tabs']; meetings: HeaderNav['meetings']
  /** switch to a tab (and meeting page) */
  open: (tab: string, meeting?: string) => void
  children: ReactNode
}) {
  const [toc, setToc] = useSearchParam<'shown' | 'hidden'>('toc', 'hidden', ['shown', 'hidden'])
  const [tocPresent, setTocPresent] = useState(false)
  const value = useMemo<HeaderNav>(() => ({
    tocShown: toc === 'shown', setTocShown: (v) => setToc(v ? 'shown' : 'hidden'), tocPresent, setTocPresent,
    tabs, meetings,
    goTo: (t) => { open(t.tab, t.meeting); if (t.id || t.title) scrollToChart(t) },
  }), [toc, setToc, tocPresent, tabs, meetings, open])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

/** the chart's section once its tab has drawn it (charts load their data, so keep looking for a while) */
function scrollToChart(t: ChartTarget) {
  // ChartFrame shows a title's head (before ' — ') as its heading
  const prefix = (t.title ?? '').replace(/^…\s*/, '').split(' — ')[0].split('…')[0].trim().slice(0, 48).toLowerCase()
  const find = () => {
    if (t.id) {
      const el = document.getElementById(t.id) ?? document.querySelector<HTMLElement>(`[id^="${CSS.escape(t.id)}-"]`)
      if (el) return el
    }
    if (!prefix) return null
    return [...document.querySelectorAll<HTMLElement>('section.panel')].find((s) => (s.querySelector('h2')?.textContent ?? '').trim().toLowerCase().includes(prefix)) ?? null
  }
  let tries = 0
  const tick = () => {
    const el = find()
    if (el) {
      const align = (behavior: ScrollBehavior) => {
        const top = (document.querySelector('.app-header') as HTMLElement | null)?.offsetHeight ?? 96
        window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - top - 16, behavior })
      }
      align('auto')
      // charts above it are still loading and push it down: follow it until the page settles,
      // unless the reader scrolls away first
      let last = window.scrollY
      const follow = (n: number) => window.setTimeout(() => {
        if (Math.abs(window.scrollY - last) > 4 && n > 0) return
        align('auto'); last = window.scrollY
        if (n < 12) follow(n + 1)
      }, 350)
      window.setTimeout(() => { last = window.scrollY; follow(0) }, 200)
      el.classList.add('panel-flash')
      window.setTimeout(() => el.classList.remove('panel-flash'), 1600)
    } else if (++tries < 40) window.setTimeout(tick, 150)
  }
  window.setTimeout(tick, 60)
}
