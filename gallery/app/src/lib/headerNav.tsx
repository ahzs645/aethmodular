import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'
import { useSearchParam } from '@/lib/url'

/**
 * State the header shares with the page below it.
 * - The "On this tab" contents sidebar (components/PageToc): hidden unless the URL says
 *   toc=shown; its Show / Hide button sits in the header beside Views, offered only while the
 *   tab has a contents list.
 * - Navigation for the ⌘K search (components/CommandPalette): open a tab or meeting page and
 *   scroll to a chart once it has drawn (or, one chart at a time, show it: SHOW_CHART_EVENT).
 */
export interface ChartTarget { tab: string; meeting?: string; id?: string; title?: string }
interface HeaderNav {
  tocShown: boolean
  setTocShown: (v: boolean) => void
  /** the tab has a contents list (PageToc with more than one section) */
  tocPresent: boolean
  setTocPresent: (v: boolean) => void
  /** `group`: the header menu's group the tab sits in (components/NavMenu), for the search's sections */
  tabs: { key: string; label: string; title?: string; group?: string }[]
  meetings: { key: string; label: string; title?: string }[]
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
    goTo: (t) => {
      // told before the tab switches, so the single-chart view (lib/singleView) can wait for the chart on the new page
      if (t.id || t.title) window.dispatchEvent(new CustomEvent<ChartTarget>(SHOW_CHART_EVENT, { detail: t }))
      open(t.tab, t.meeting)
      if (t.id || t.title) scrollToChart(t)
    },
  }), [toc, setToc, tocPresent, tabs, meetings, open])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

/**
 * Asks the single-chart view (lib/singleView) to bring a chart into view: it shows one chart at a time
 * and hides the rest, so scrolling to the chart's section would land on nothing. Its detail is the ChartTarget.
 */
export const SHOW_CHART_EVENT = 'gallery:show-chart'

/** the chart's section on the page as drawn now, by its id (a ChartFrame adds a method suffix), else by its title */
export function findChart(t: ChartTarget): HTMLElement | null {
  if (t.id) {
    const el = document.getElementById(t.id) ?? document.querySelector<HTMLElement>(`[id^="${CSS.escape(t.id)}-"]`)
    if (el) return el
  }
  // ChartFrame shows a title's head (before ' — ') as its heading
  const prefix = (t.title ?? '').replace(/^…\s*/, '').split(' — ')[0].split('…')[0].trim().slice(0, 48).toLowerCase()
  if (!prefix) return null
  // a frame's own title first: in the single-chart view the chart in view has a picker for its heading
  return [...document.querySelectorAll<HTMLElement>('section.panel')]
    .find((s) => `${s.dataset.frameTitle ?? ''}\n${s.querySelector('h2')?.textContent ?? ''}`.trim().toLowerCase().includes(prefix)) ?? null
}

/** the chart's section once its tab has drawn it (charts load their data, so keep looking for a while) */
function scrollToChart(t: ChartTarget) {
  let tries = 0
  const tick = () => {
    // one chart at a time: the single-chart view shows it instead (SHOW_CHART_EVENT)
    if (document.documentElement.dataset.single === '1') return
    const el = findChart(t)
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
      flashChart(el)
    } else if (++tries < 40) window.setTimeout(tick, 150)
  }
  window.setTimeout(tick, 60)
}

/** a brief ring around the chart the search opened (styles.css .panel-flash) */
export function flashChart(el: HTMLElement) {
  el.classList.remove('panel-flash')
  void el.offsetWidth
  el.classList.add('panel-flash')
  window.setTimeout(() => el.classList.remove('panel-flash'), 1600)
}
