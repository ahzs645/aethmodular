import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { CHART_DISPLAY_CSS } from './chartDisplayStyles'

/**
 * Viewer preferences. `explain` is the one that matters: off by default, the
 * interface shows titles, controls and marks only; on, every chart shows its
 * explanation and provenance and the subset bar shows its captions. Stored in
 * localStorage as a per-viewer convenience, never as app state. `density` is how the gallery
 * reads: compact (the default) or readable, which scales every text size up (styles.css
 * --text-scale, and chart text by its font-size attribute) for a projector or a tired reader.
 */
export type Density = 'compact' | 'readable'
/** `notes` is the flag behind the ⓘ beside chart titles (InfoTip, "Notes on this chart"): off by
 * default so titles stay clean; the Display and data menu's Interface section turns it on. */
interface Prefs {
  explain: boolean
  setExplain: (v: boolean) => void
  density: Density
  setDensity: (v: Density) => void
  notes: boolean
  setNotes: (v: boolean) => void
}

const Ctx = createContext<Prefs>({ explain: false, setExplain: () => {}, density: 'compact', setDensity: () => {}, notes: false, setNotes: () => {} })
/** Builds for readers outside the group (gallery/share/publish_pages.py, build_share.py) set
 * VITE_HIDE_EXPLAIN=1: the header toggle disappears and explanations stay off. Local dev keeps it. */
export const EXPLAIN_AVAILABLE = import.meta.env.VITE_HIDE_EXPLAIN !== '1'
const KEY = 'aeth-gallery:explain'
const DENSITY_KEY = 'aeth-gallery:density'
const NOTES_KEY = 'aeth-gallery:chart-notes'
const readNotes = (): boolean => { try { return localStorage.getItem(NOTES_KEY) === '1' } catch { return false } }
const readDensity = (): Density => { try { return localStorage.getItem(DENSITY_KEY) === 'readable' ? 'readable' : 'compact' } catch { return 'compact' } }

export function PrefsProvider({ children }: { children: ReactNode }) {
  const [explain, setExplainState] = useState<boolean>(() => {
    if (!EXPLAIN_AVAILABLE) return false
    try {
      return localStorage.getItem(KEY) === '1'
    } catch {
      return false
    }
  })
  useEffect(() => {
    document.documentElement.dataset.explain = explain ? '1' : '0'
    try {
      localStorage.setItem(KEY, explain ? '1' : '0')
    } catch {
      /* private window, blocked storage: the toggle still works for the session */
    }
  }, [explain])
  const [density, setDensity] = useState<Density>(readDensity)
  useEffect(() => {
    document.documentElement.dataset.density = density
    try { localStorage.setItem(DENSITY_KEY, density) } catch { /* the choice still holds for the session */ }
  }, [density])
  const [notes, setNotes] = useState<boolean>(readNotes)
  useEffect(() => {
    try { localStorage.setItem(NOTES_KEY, notes ? '1' : '0') } catch { /* the choice still holds for the session */ }
  }, [notes])
  const value = useMemo(() => ({ explain: EXPLAIN_AVAILABLE && explain, setExplain: setExplainState, density, setDensity, notes, setNotes }), [explain, density, notes])
  return <Ctx.Provider value={value}><style>{CHART_DISPLAY_CSS}</style>{children}</Ctx.Provider>
}

export const usePrefs = () => useContext(Ctx)
