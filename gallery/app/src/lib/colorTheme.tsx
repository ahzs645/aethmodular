import { useSyncExternalStore } from 'react'
import { currentTheme, readThemePref, resolveTheme, setInkTheme, THEME_KEY, type ThemeName, type ThemePref } from './theme.ts'

/**
 * Light or dark. The page chrome follows `html[data-theme]` (styles.css); the chart marks carry
 * their colours as SVG attributes so export can serialise them, so a theme change swaps the INK
 * values (lib/theme setInkTheme) and main.tsx remounts the app under the new theme. Everything a
 * chart shows lives in the URL, so a remount loses nothing a reader chose. Exports are always
 * drawn in the light theme (lib/export printColours). The choice is a per-viewer convenience in
 * localStorage; 'auto' follows the system setting while the page is open.
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
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (pref === 'auto') apply(resolveTheme('auto')) })
} catch { /* no matchMedia (node tests) */ }

const subscribe = (l: () => void) => { listeners.add(l); return () => { listeners.delete(l) } }
export const useThemePref = () => useSyncExternalStore(subscribe, () => pref)
export const useTheme = () => useSyncExternalStore(subscribe, currentTheme)
