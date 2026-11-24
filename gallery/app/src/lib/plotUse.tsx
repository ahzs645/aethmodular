import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { DEFAULT_PAST_TAU, type HipsScope } from './hipsLinearise'

/**
 * Which of the Plot tab's page-wide settings the open page reads.
 *
 * The tabs without the Samples panel (Meetings, Experimental, ML) mostly draw their own exports, so a MAC or HIPS
 * row offered on every one of their pages sat there doing nothing (24 Nov: Aethalometer loading k). Instead a page
 * says what it reads and App's PageSidebar offers only that:
 *
 *   useMac()                          lib/mac: registers 'mac' by itself (useMac(false): reads it without offering it)
 *   const hips = useHipsView()        registers 'hips'; { ceiling, scope, cut }, ceiling null = as measured
 *
 * A page that draws HIPS BC from its own export follows the MAC with `useMac().bc(v)` and the HIPS view with
 * `linearised(...)` below. Outside the provider (an embed of one chart, a test) nothing registers and the HIPS view
 * is as measured.
 */
export type PlotUse = 'mac' | 'hips'

export interface HipsView {
  /** the ceiling τ∞ HIPS is linearised with; null = as measured */
  ceiling: number | null
  scope: HipsScope
  /** the optical depth the `past` scope linearises from */
  cut: number
}
const AS_MEASURED: HipsView = { ceiling: null, scope: 'all', cut: DEFAULT_PAST_TAU }

const Reg = createContext<((u: PlotUse) => () => void) | null>(null)
const Used = createContext<Record<PlotUse, boolean>>({ mac: false, hips: false })
const Hips = createContext<HipsView>(AS_MEASURED)

export function PlotUseProvider({ hips, children }: { hips: HipsView; children: ReactNode }) {
  const [n, setN] = useState<Record<PlotUse, number>>({ mac: 0, hips: 0 })
  const add = useCallback((u: PlotUse) => {
    setN((c) => ({ ...c, [u]: c[u] + 1 }))
    return () => setN((c) => ({ ...c, [u]: c[u] - 1 }))
  }, [])
  const used = useMemo(() => ({ mac: n.mac > 0, hips: n.hips > 0 }), [n.mac > 0, n.hips > 0]) // eslint-disable-line react-hooks/exhaustive-deps
  const view = useMemo(() => hips, [hips.ceiling, hips.scope, hips.cut]) // eslint-disable-line react-hooks/exhaustive-deps
  return <Reg.Provider value={add}><Used.Provider value={used}><Hips.Provider value={view}>{children}</Hips.Provider></Used.Provider></Reg.Provider>
}

/** mark the open page as reading a Plot setting while the calling component is mounted (and `on`: a chart that
 * draws HIPS only under one of its own options says so) */
export function usePlotUse(u: PlotUse, on = true): void {
  const add = useContext(Reg)
  useEffect(() => (add && on ? add(u) : undefined), [add, u, on])
}

/** what the open page reads (App's PageSidebar) */
export const usePlotUsed = (): Record<PlotUse, boolean> => useContext(Used)

/** the gallery's HIPS view (Plot → HIPS), and the open page reads it (while `on`) */
export function useHipsView(on = true): HipsView {
  usePlotUse('hips', on)
  return useContext(Hips)
}

/**
 * One filter's HIPS value under the view: × 1 / (1 − τ/τ∞), τ = Fabs × V / 353 (lib/hipsLinearise opticalDepth).
 * `inScope` says whether the filter's site is covered (lib/hipsRun siteInScope); null where the filter is darker
 * than the ceiling can read, the value itself where the view leaves it as measured.
 */
export function linearised(value: number, tau: number | null | undefined, view: HipsView, inScope = true): number | null {
  if (view.ceiling == null || !inScope || tau == null || !Number.isFinite(tau)) return value
  if (view.scope === 'past' && tau < view.cut) return value
  const r = tau / view.ceiling
  return r < 1 ? value / (1 - r) : null
}
