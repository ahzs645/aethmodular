import { useSyncExternalStore } from 'react'

/**
 * While a chart's export dialog is open, the chart on the page is redrawn for the file:
 *   width        the chart's width in px (a journal column at ~8 pt text), or null for as on screen
 *   statsInside  the regression summary is left out of the drawing and handed over as data
 *                (a hidden [data-export-stats] group), so the export can set it inside the plot
 * Every chart reads it the same way (components/RegressionOverlay finds its own panel from the
 * DOM), so no chart needs to know about export. The single-chart view stops fitting the chart
 * to the window meanwhile (lib/singleView SingleFit), so the export is drawn at page size.
 * Cleared when the dialog closes; the width goes back with it.
 */
export interface ExportRender { panel: HTMLElement; width: number | null; statsInside: boolean }

let current: ExportRender | null = null
const listeners = new Set<() => void>()
let sized: { el: HTMLElement; width: string; maxWidth: string } | null = null

export function setExportRender(next: ExportRender | null) {
  if (sized) { sized.el.style.width = sized.width; sized.el.style.maxWidth = sized.maxWidth; sized = null }
  const chart = next?.width ? next.panel.querySelector<HTMLElement>('.frame-chart') : null
  if (chart && next?.width) {
    sized = { el: chart, width: chart.style.width, maxWidth: chart.style.maxWidth }
    chart.style.width = `${Math.round(next.width)}px`
    chart.style.maxWidth = `${Math.round(next.width)}px`
  }
  current = next
  for (const l of listeners) l()
}

export const useExportRender = () => useSyncExternalStore(
  (l) => { listeners.add(l); return () => { listeners.delete(l) } },
  () => current,
)

/**
 * After a redraw: resolves once the panel's DOM has been quiet for `quiet` ms and the chart has taken its new
 * size (a size observer, a re-render and a chart's own measuring can each take a frame, and a background tab
 * runs few frames). Narrowing, it waits for the chart to fit `width`; widening or back to the page's width,
 * for any change of size, or `grow` ms when nothing changes. Never longer than `max`: a chart with a minimum
 * width never fits a narrow column.
 */
export function settled(panel: HTMLElement, width: number | null = null, quiet = 180, grow = 600, max = 2500) {
  const widths = () => [...panel.querySelectorAll('.frame-chart svg')]
    .filter((s) => !s.parentElement?.closest('svg') && s.getClientRects().length)
    .map((s) => s.getBoundingClientRect().width)
  const before = widths()
  const shrink = width !== null && before.some((w) => w > width + 2)
  const ready = (t: number) => {
    const now = widths()
    return shrink ? now.every((w) => w <= width! + 2)
      : t > grow || now.length !== before.length || now.some((w, i) => Math.abs(w - before[i]) > 1)
  }
  return new Promise<void>((resolve) => {
    const start = performance.now()
    const check = () => { const t = performance.now() - start; if (t > max || ready(t)) done(); else timer = setTimeout(check, quiet) }
    let timer = setTimeout(check, quiet)
    const mo = new MutationObserver(() => {
      clearTimeout(timer)
      timer = setTimeout(check, performance.now() - start > max ? 0 : quiet)
    })
    mo.observe(panel, { subtree: true, childList: true, attributes: true, characterData: true })
    function done() { mo.disconnect(); clearTimeout(timer); resolve() }
  })
}
