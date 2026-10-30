/**
 * What the Components and Protocols pages read from the exports the gallery already has: calibration.json (the
 * explorer's summary fits at the rule k under each cross-validation protocol, and its site-grouped sweeps) and
 * calibration_runs.json (per-preset CV curves and the Addis per-filter readouts with HIPS Fabs). Nothing is refitted.
 */
import { useEffect, useState } from 'react'
import { loadDerived, loadJson } from '@/lib/data'
import type { Row } from '@/lib/table'
import type { CalibFile, CalibRunsFile } from '@/lib/types'

/** calibration.json through the gallery's cache key and relabelling (hooks/useGalleryData), so a download is shared */
export function useCalib() {
  const [state, setState] = useState<{ calib: CalibFile | null; error: string }>({ calib: null, error: '' })
  useEffect(() => {
    let live = true
    loadDerived<CalibFile>('gallery:calibration', async () =>
      JSON.parse(JSON.stringify(await loadJson('calibration.json')).replace(/\bPasadena\b/g, 'JPL')))
      .then((calib) => { if (live) setState({ calib, error: '' }) })
      .catch((e) => { if (live) setState({ calib: null, error: String(e instanceof Error ? e.message : e) }) })
    return () => { live = false }
  }, [])
  return state
}

/** the three explorer protocols (calibration.json `mo`), named as ftir_51 and the deck name them */
export const PROTOCOL: Record<string, string> = {
  site_heldout: 'A · site-held-out (site-grouped 5-fold, first major minimum)',
  app: 'B · Calibration app (interleaved 10-fold, within 5 %)',
  app_fmm: 'B2 · interleaved 10-fold, first major minimum',
}
export const PROTOCOL_ORDER = ['site_heldout', 'app', 'app_fmm']
/** ftir_21's colours: blue site-held-out, red Calibration app; B2 between them */
export const PROTOCOL_COLOR: Record<string, string> = {
  [PROTOCOL.site_heldout]: '#4e79a7', [PROTOCOL.app]: '#e15759', [PROTOCOL.app_fmm]: '#f28e2b',
}
/** the notebooks' `mode` values (ftir_21/22/23) to the same labels and colours */
export const protocolOf = (mode: unknown) => PROTOCOL[String(mode)] ?? String(mode)

/** the cohorts at their default size, as ftir_20–23 name their six setups */
export const COHORT_NAME: Record<string, string> = {
  pool: 'Entire IMPROVE network (13,010)', smoke: 'Biomass-smoke (906)', eth_shaped: 'Ethiopia-shaped smoke (300)',
  analogs: 'Spectral analogs (500)', ocec: 'Lowest-OC/EC (800)',
}
export const COHORT_ORDER = ['pool', 'smoke', 'eth_shaped', 'analogs', 'ocec']
export const SPECTRA_NAME: Record<string, string> = { raw: 'raw', airspec: 'spline baseline (AIRSpec df1=6)', deriv2: 'second derivative', neutral: 'neutral baseline' }

/**
 * The explorer's rule-k rows at Addis for every cohort at its default size (no size for pool and smoke), selected
 * on raw spectra, every training lot: one row per cohort × spectra × protocol. `ob`/`om` are the OLS intercept and
 * slope against HIPS Fabs/10 on the fixed 190 filters (ftir_21's estimator), `r2`/`rmse` on the same filters.
 */
export function ruleRows(calib: CalibFile): Row[] {
  return calib.grid.filter((r) => r.tg === 'addis' && r.lot === 'all' && r.el === 'all' && (r.sel ?? 'raw') === 'raw'
    && r.cut === (calib.default_cutoff[r.co] ?? null) && COHORT_NAME[r.co] && PROTOCOL[r.mo])
    .sort((a, b) => COHORT_ORDER.indexOf(a.co) - COHORT_ORDER.indexOf(b.co) || PROTOCOL_ORDER.indexOf(a.mo) - PROTOCOL_ORDER.indexOf(b.mo))
    .map((r) => ({ co: r.co, cohort: COHORT_NAME[r.co], spectra: SPECTRA_NAME[r.sp] ?? r.sp, sp: r.sp, mode: r.mo, protocol: PROTOCOL[r.mo],
      k: r.ak ?? r.k, intercept: r.ob, slope10: r.om, r2: r.r2, rmse: r.rmse, rmsecv: r.rmsecv ?? null }))
}

/** HIPS BC (Fabs / exportMac) mean and population variance on the Addis evaluation filters: the fixed 190 and all 239 */
export interface AddisMoments { fixed: { mean: number; variance: number; n: number } | null; all: { mean: number; variance: number; n: number } | null }
export function addisMoments(runs: CalibRunsFile | null, exportMac = 10): AddisMoments {
  const run = runs?.runs.find((r) => r.target === 'addis' && r.ref_kind === 'fabs')
  if (!run) return { fixed: null, all: null }
  const of = (keep: (i: number) => boolean) => {
    const x = run.eval.ref.map((f, i) => (keep(i) && Number.isFinite(f) ? f / exportMac : NaN)).filter(Number.isFinite)
    if (!x.length) return null
    const mean = x.reduce((s, v) => s + v, 0) / x.length
    return { mean, variance: x.reduce((s, v) => s + (v - mean) ** 2, 0) / x.length, n: x.length }
  }
  return { fixed: of((i) => run.eval.fixed[i]), all: of(() => true) }
}

/** mean HIPS Fabs per season on the 239 Addis filters (calibration_runs eval groups are Ethiopian seasons) */
export function seasonFabs(runs: CalibRunsFile | null): Map<string, number> {
  const run = runs?.runs.find((r) => r.target === 'addis' && r.ref_kind === 'fabs')
  const sums = new Map<string, [number, number]>()
  run?.eval.ref.forEach((f, i) => {
    const g = run.eval.group[i]
    if (!Number.isFinite(f) || !g) return
    const s = sums.get(g) ?? [0, 0]
    sums.set(g, [s[0] + f, s[1] + 1])
  })
  return new Map([...sums].map(([g, [s, n]]) => [g, s / n]))
}
