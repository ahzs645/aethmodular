import { useSyncExternalStore } from 'react'
import { readLegendOrders, type SavedLegendOrder } from './legendOrder'

/** Legend orders by chart key (data-chart-key on the panel), remembered in this browser. */
const KEY = 'aeth-gallery:legend-order:v1'
function read() {
  try { return readLegendOrders(JSON.parse(localStorage.getItem(KEY) ?? '{}')) } catch { return new Map<string, SavedLegendOrder>() }
}
let state = read()
const listeners = new Set<() => void>()
const subscribe = (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener) } }
function changed() {
  try { localStorage.setItem(KEY, JSON.stringify(Object.fromEntries(state))) } catch { /* the choice still holds for this session */ }
  listeners.forEach((listener) => listener())
}

export const useLegendOrder = (chart: string | undefined) => useSyncExternalStore(subscribe, () => (chart ? state.get(chart) : undefined))
export function setLegendOrder(chart: string, saved: SavedLegendOrder) { state.set(chart, saved); changed() }
export function resetLegendOrder(chart: string) { state.delete(chart); changed() }

if (typeof window !== 'undefined') window.addEventListener('storage', (event) => {
  if (event.key === KEY || event.key === null) { state = read(); listeners.forEach((listener) => listener()) }
})
