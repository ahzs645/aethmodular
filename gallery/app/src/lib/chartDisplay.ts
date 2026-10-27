/** Viewer display settings, shared by default, with optional settings for a single chart. */
export type ControlsSide = 'top' | 'right' | 'left'
export interface ChartDisplay {
  controlsSide: ControlsSide
  density: 'compact' | 'readable'
  explain: boolean
  theme: 'light' | 'dark' | 'auto'
}
export interface ChartDisplayState {
  controlsSide: ControlsSide
  charts: Record<string, Partial<ChartDisplay>>
}
export const DEFAULT_CHART_DISPLAY: ChartDisplayState = { controlsSide: 'top', charts: {} }
export const DISPLAY_SIDES: ControlsSide[] = ['top', 'right', 'left']

export function readChartDisplay(value: unknown): ChartDisplayState {
  const raw = value && typeof value === 'object' ? value as Partial<ChartDisplayState> : {}
  const charts: ChartDisplayState['charts'] = {}
  if (raw.charts && typeof raw.charts === 'object' && !Array.isArray(raw.charts)) {
    for (const [id, entry] of Object.entries(raw.charts)) {
      if (!entry || typeof entry !== 'object') continue
      const clean: Partial<ChartDisplay> = {}
      if (DISPLAY_SIDES.includes(entry.controlsSide!)) clean.controlsSide = entry.controlsSide
      if (entry.density === 'compact' || entry.density === 'readable') clean.density = entry.density
      if (typeof entry.explain === 'boolean') clean.explain = entry.explain
      if (['light', 'dark', 'auto'].includes(entry.theme!)) clean.theme = entry.theme
      if (Object.keys(clean).length) Object.defineProperty(charts, id, { value: clean, enumerable: true, configurable: true, writable: true })
    }
  }
  return { controlsSide: DISPLAY_SIDES.includes(raw.controlsSide!) ? raw.controlsSide! : 'top', charts }
}

/** A shared change also clears that setting's exceptions, so “all charts” always means all. */
export function sharedChartDisplay(state: ChartDisplayState, patch: Partial<ChartDisplay>): ChartDisplayState {
  const charts: ChartDisplayState['charts'] = {}
  for (const [id, settings] of Object.entries(state.charts)) {
    const next = { ...settings }
    for (const key of Object.keys(patch) as (keyof ChartDisplay)[]) delete next[key]
    if (Object.keys(next).length) Object.defineProperty(charts, id, { value: next, enumerable: true, configurable: true, writable: true })
  }
  return { controlsSide: patch.controlsSide ?? state.controlsSide, charts }
}

export interface VisibleChart {
  key: string
  title: string
  rect: { left: number; right: number; top: number; bottom: number }
  focused?: boolean
}
/** Visible screen area, rather than the fraction of a potentially very tall chart. */
export function mostVisibleChart(charts: VisibleChart[], viewport: VisibleChart['rect']): VisibleChart | null {
  let best: VisibleChart | null = null, bestArea = 0
  for (const chart of charts) {
    const r = chart.rect
    const area = Math.max(0, Math.min(r.right, viewport.right) - Math.max(r.left, viewport.left))
      * Math.max(0, Math.min(r.bottom, viewport.bottom) - Math.max(r.top, viewport.top))
    if (area > 0 && (chart.focused && !best?.focused || !!chart.focused === !!best?.focused && area > bestArea)) {
      best = chart
      bestArea = area
    }
  }
  return best
}
