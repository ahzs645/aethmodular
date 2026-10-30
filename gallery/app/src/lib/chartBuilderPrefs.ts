import { useSyncExternalStore } from 'react'

const KEY = 'aeth-gallery:chart-builder:v1'
const shareTabs = (import.meta.env.VITE_SHARE_TABS ?? '').split(',').map((tab: string) => tab.trim()).filter(Boolean)
export const CHART_BUILDER_AVAILABLE = !shareTabs.length || shareTabs.includes('builder')
const read = () => { try { return localStorage.getItem(KEY) === '1' } catch { return false } }
let enabled = read()
const listeners = new Set<() => void>()
const emit = () => listeners.forEach(listener => listener())
const subscribe = (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener) } }
export const useChartBuilderEnabled = () => useSyncExternalStore(subscribe, () => enabled)
export function setChartBuilderEnabled(value: boolean) {
  enabled = value
  try { localStorage.setItem(KEY, value ? '1' : '0') } catch { /* Works for this session when storage is blocked. */ }
  emit()
}
window.addEventListener('storage', event => {
  if (event.key === KEY || event.key === null) { enabled = read(); emit() }
})
