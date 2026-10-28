import { useSyncExternalStore } from 'react'

export type SpartanSource = 'bundled' | 'live'
const KEY = 'aeth-gallery:spartan-source:v1'
const read = (): SpartanSource => { try { return localStorage.getItem(KEY) === 'live' ? 'live' : 'bundled' } catch { return 'bundled' } }
let state = { source: read(), revision: 0 }
const listeners = new Set<() => void>()
const subscribe = (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener) } }
const snapshot = () => state
const emit = () => listeners.forEach(listener => listener())
export const useSpartanSource = () => useSyncExternalStore(subscribe, snapshot)
export function setSpartanSource(source: SpartanSource) {
  if (source === state.source) return
  state = { source, revision: state.revision + (source === 'live' ? 1 : 0) }
  try { localStorage.setItem(KEY, source) } catch { /* Session preference still works. */ }
  emit()
}
export function refreshSpartanSource() { state = { ...state, revision: state.revision + 1 }; emit() }
if (typeof window !== 'undefined') window.addEventListener('storage', event => {
  if (event.key === KEY || event.key === null) { state = { ...state, source: read() }; emit() }
})
