import { createContext, useContext, useRef, useSyncExternalStore } from 'react'
import { currentView, isEmbedChart } from './views.ts'

/** What one chart's embed tells the section page around it (components/SectionFrame). */
export interface EmbedChartMessage {
  type: typeof EMBED_CHART_MESSAGE
  /** the chart it shows: until the page has drawn the one asked for, the page's first */
  active: string | null
  /** how far down the page the chart reaches, for a card that would clip it */
  height: number
  view: ReturnType<typeof currentView>
}
export const EMBED_CHART_MESSAGE = 'gallery:embed-chart'

/**
 * Inside a section's embed: mark the page so the styles leave only the chart, and report the chart in view and the
 * view around it, so the section keeps a setting changed in the chart and knows when it has drawn. Compared on a
 * timer because a page writes the address from many places; it costs a string comparison twice a second.
 */
export function watchEmbedChart() {
  document.documentElement.dataset.embedChart = '1'
  let last = ''
  window.setInterval(() => {
    // a chart's frame, or the Dashboard's maximised tile
    const el = document.querySelector<HTMLElement>('[data-single-active], .dash-max [data-tile-id]')
    const active = el?.dataset.frameKey ?? (el?.dataset.tileId ? `tile:${el.dataset.tileId}` : null)
    const height = el ? Math.ceil(el.getBoundingClientRect().bottom + window.scrollY) : 0
    const now = `${active}|${height}|${window.location.href}`
    if (now === last) return
    last = now
    window.parent.postMessage({ type: EMBED_CHART_MESSAGE, active, height, view: currentView() } satisfies EmbedChartMessage, window.location.origin)
  }, 500)
}

/** the chart a section's embed is for, read once: the address follows the chart afterwards */
const wanted = typeof window !== 'undefined' && isEmbedChart() ? new URL(window.location.href).searchParams.get('chart') : null
/** Past this the wanted chart has not appeared, perhaps because its key comes from what it draws: everything is drawn, so nothing can stay missing. */
const DRAW_ALL_AFTER_MS = 12000
let drawAll = wanted === null
const listeners = new Set<() => void>()
if (!drawAll) window.setTimeout(() => { drawAll = true; listeners.forEach((l) => l()) }, DRAW_ALL_AFTER_MS)
const subscribe = (l: () => void) => { listeners.add(l); return () => { listeners.delete(l) } }

/** inside a chart that is drawn: a frame in it is part of that chart */
export const EmbedDrawnContext = createContext(false)

/**
 * Whether a ChartFrame draws its chart. A section's embed runs one chart's whole page and shows that chart: the
 * others are hidden, and on Calibration they are 18 of 19 charts and 90 % of the page. So only the wanted chart
 * draws. Each still renders its frame, which is how the page finds the wanted one by key. A frame that has drawn
 * stays drawn (its key can change with its settings), and `active` is the frame the page is showing, however it
 * got there (a setting that swaps one frame for another).
 */
export function useEmbedDraw(key: string, active: boolean) {
  const all = useSyncExternalStore(subscribe, () => drawAll)
  const inside = useContext(EmbedDrawnContext)
  const drawn = useRef(false)
  if (all || inside || active || key === wanted) drawn.current = true
  return drawn.current
}
