import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

/**
 * Viewer preferences. `explain` is the one that matters: off by default, the
 * interface shows titles, controls and marks only; on, every chart shows its
 * explanation and provenance and the subset bar shows its captions. Stored in
 * localStorage as a per-viewer convenience, never as app state.
 */
interface Prefs {
  explain: boolean
  setExplain: (v: boolean) => void
}

const Ctx = createContext<Prefs>({ explain: false, setExplain: () => {} })
/** Builds for readers outside the group (gallery/share/publish_pages.py, build_share.py) set
 * VITE_HIDE_EXPLAIN=1: the header toggle disappears and explanations stay off. Local dev keeps it. */
export const EXPLAIN_AVAILABLE = import.meta.env.VITE_HIDE_EXPLAIN !== '1'
const KEY = 'aeth-gallery:explain'

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
  const value = useMemo(() => ({ explain: EXPLAIN_AVAILABLE && explain, setExplain: setExplainState }), [explain])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export const usePrefs = () => useContext(Ctx)
