import { useSyncExternalStore } from 'react'
import { DEFAULT_CHART_DISPLAY, readChartDisplay, sharedChartDisplay, type ChartDisplay, type ChartDisplayState } from './chartDisplay'

const KEY = 'aeth-gallery:chart-display:v1'
function read(): ChartDisplayState {
  try {
    const saved = localStorage.getItem(KEY)
    if (saved) return readChartDisplay(JSON.parse(saved))
    // Retain the placement choices made with the old, per-chart layout button.
    const sides = JSON.parse(localStorage.getItem('aeth-gallery:controls-side') ?? '{}')
    return readChartDisplay({ charts: Object.fromEntries(Object.entries(sides).map(([id, controlsSide]) => [id, { controlsSide }])) })
  } catch { return DEFAULT_CHART_DISPLAY }
}
let state = read()
const listeners = new Set<() => void>()
const subscribe = (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener) } }
const snapshot = () => state
function update(next: ChartDisplayState) {
  state = next
  try { localStorage.setItem(KEY, JSON.stringify(state)) } catch { /* The choice still holds for this session. */ }
  listeners.forEach(listener => listener())
}
export const useChartDisplayPrefs = () => useSyncExternalStore(subscribe, snapshot)
export const setSharedChartDisplay = (patch: Partial<ChartDisplay>) => update(sharedChartDisplay(state, patch))
export const setSingleChartDisplay = (key: string, patch: Partial<ChartDisplay>) => update({ ...state, charts: { ...state.charts, [key]: { ...state.charts[key], ...patch } } })
export function resetChartDisplay(key: string) {
  const charts = { ...state.charts }
  delete charts[key]
  update({ ...state, charts })
}
if (typeof window !== 'undefined') window.addEventListener('storage', event => {
  if (event.key === KEY || event.key === null) { state = read(); listeners.forEach(listener => listener()) }
})
