import { useSyncExternalStore } from 'react'
import { currentTheme, readThemePref, resolveTheme, setInkTheme, THEME_KEY, type ThemeName, type ThemePref } from './theme.ts'

/**
 * Light or dark. The page chrome follows `html[data-theme]` (styles.css), and so do the charts: their
 * marks are drawn in the light colours and lib/themeCss maps each one to its dark twin in CSS, so a
 * switch recolours the page in place (no remount, no lost scroll or hover). The few things CSS cannot
 * recolour (colour ramps, heat-cell labels, canvas rasters) subscribe with useTheme() and re-render.
 * The choice is a per-viewer convenience in localStorage; 'auto' follows the system setting while the
 * page is open.
 */
let pref: ThemePref = readThemePref()
const listeners = new Set<() => void>()
const emit = () => listeners.forEach((l) => l())

function apply(theme: ThemeName) {
  document.documentElement.dataset.theme = theme
  if (theme === currentTheme()) return
  setInkTheme(theme)
  emit()
}

export function setThemePref(next: ThemePref) {
  pref = next
  try { localStorage.setItem(THEME_KEY, next) } catch { /* the choice still holds for the session */ }
  apply(resolveTheme(next))
  emit()
}

try {
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (pref === 'auto') apply(resolveTheme('auto')); emit() })
} catch { /* no matchMedia (node tests) */ }

const subscribe = (l: () => void) => { listeners.add(l); return () => { listeners.delete(l) } }
export const useThemePref = () => useSyncExternalStore(subscribe, () => pref)
export const useTheme = () => useSyncExternalStore(subscribe, currentTheme)
/** A chart can follow the system even while the rest of the page has a fixed theme. */
export const useSystemDark = () => useSyncExternalStore(subscribe, () => resolveTheme('auto') === 'dark')
