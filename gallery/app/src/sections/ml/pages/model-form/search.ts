/**
 * ftir_55 and ftir_51 05/06 redrawn from calibration.json (the explorer's summary fits at the rule k, which the
 * Calibration tab already loads): the best guardrail-passing score per cohort × spectra, the folded
 * specification plane, and the protocol and training-lot comparison. Nothing is refitted: each row's Deming
 * slope against Fabs/10 is re-expressed at the reader's MAC (slope × MAC/10, intercept unchanged, exact for Deming
 * with λ ∝ MAC², lib/mac), and the explorer's score |intercept| + 5·|slope − 1| and its slope guardrail are
 * recomputed from it.
 */
import { useEffect, useState } from 'react'
import { loadDerived, loadJson } from '@/lib/data'
import { COHORT_SHORT, PREPROCESSING, label } from '@/lib/labels'
import type { Row } from '@/lib/table'
import type { CalibFile, CalibRow } from '@/lib/types'

/** calibration.json through the gallery's cache key and relabelling (hooks/useGalleryData), so the Calibration tab's download is reused */
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

/** ftir_55's objective weight: one unit of slope costs as much as five µg/m³ of intercept */
export const W = 5
/** ftir_55's core guardrails (the explorer's batch defaults), and the slope box the Calibration tab ranks with */
export const GUARDS = {
  ftir55: { label: 'ftir_55: slope 0.7–1.3', box: [0.7, 1.3] as [number, number] },
  explorer: { label: 'Calibration tab: slope 0.85–1.18', box: [0.85, 1.18] as [number, number] },
} as const
export type GuardKey = keyof typeof GUARDS

export const COHORT_ORDER = ['pool', 'smoke', 'eth_shaped', 'analogs', 'ocec']
export const SPECTRA_ORDER = ['raw', 'deriv2', 'airspec']
export const cohortName = (co: string) => label(COHORT_SHORT, co)
export const spectraName = (sp: string) => label(PREPROCESSING, sp)

const ok = (v: number | null | undefined): v is number => v != null && Number.isFinite(v)
const slopeAt = (m: number | null, mac: number, exportMac: number) => (ok(m) ? m * (mac / exportMac) : null)
export const scoreOf = (slope: number | null, intercept: number | null) => (ok(slope) && ok(intercept) ? Math.abs(intercept) + W * Math.abs(slope - 1) : null)

/**
 * ftir_55 §2c from the grid: every Addis configuration scored on all its filters (evaluation lot all), every
 * protocol and training lot, at the rule k; best passing score, its fit, and the counts, per cohort × spectra.
 */
export function bestScoreCells(calib: CalibFile, mac: number, guard: GuardKey, floor = calib.heldout_floor): Row[] {
  const [lo, hi] = GUARDS[guard].box
  const out: Row[] = []
  for (const co of COHORT_ORDER) for (const sp of SPECTRA_ORDER) {
    const rows = calib.grid.filter((r) => r.tg === 'addis' && r.el === 'all' && r.co === co && r.sp === sp)
    if (!rows.length) continue
    let best: { r: CalibRow; s: number; m: number } | null = null
    let passing = 0
    for (const r of rows) {
      const m = slopeAt(r.dm, mac, calib.mac_value), s = scoreOf(m, r.db)
      if (m == null || s == null || !ok(r.ho) || r.ho < floor || m < lo || m > hi) continue
      passing++
      if (!best || s < best.s) best = { r, s, m }
    }
    out.push({
      cohort: cohortName(co), spectra: spectraName(sp), tried: rows.length, passing,
      best_score: best?.s ?? null, best_slope: best?.m ?? null, best_intercept: best?.r.db ?? null,
      best: best ? `cutoff ${best.r.cut ?? '—'}, ${best.r.sel && best.r.sel !== 'raw' ? `selected on ${spectraName(best.r.sel).toLowerCase()}, ` : ''}${best.r.mo}, lot ${best.r.lot}, k = ${best.r.k}` : 'none passes',
    })
  }
  return out
}

/** a point of the folded plane: |slope − 1| and |intercept| on the chosen evaluation set, with what it was */
export interface FoldedPoint { dx: number; dy: number; slope: number; intercept: number; floor: boolean; r: CalibRow }

/**
 * ftir_55 §2a, the 27 Aug deck's screen: site-held-out configurations (training lot all, evaluation lot all) with a
 * held-out score, folded onto |slope − 1| and |intercept|; `allPairs` uses the 239-filter all-pairs fit, else the fixed 190.
 */
export function foldedPlane(calib: CalibFile, mac: number, allPairs: boolean): FoldedPoint[] {
  return calib.grid.flatMap((r) => {
    if (r.tg !== 'addis' || r.mo !== 'site_heldout' || r.lot !== 'all' || r.el !== 'all' || !ok(r.ho)) return []
    const m = slopeAt(allPairs ? r.adm : r.dm, mac, calib.mac_value), b = allPairs ? r.adb : r.db
    if (m == null || !ok(b)) return []
    return [{ dx: Math.abs(m - 1), dy: Math.abs(b), slope: m, intercept: b, floor: r.ho >= calib.heldout_floor, r }]
  })
}

/** ftir_51's variants (§5 CV protocols, §6 training lot), and the other spectra of the same cohorts */
export const PROTOCOL_CONFIGS: { co: string; cut: number | null; sel: string; sp: string; label: string }[] = [
  { co: 'pool', cut: null, sel: 'raw', sp: 'raw', label: 'All IMPROVE × raw (network_raw)' },
  { co: 'ocec', cut: 800, sel: 'raw', sp: 'raw', label: 'Lowest OC/EC 800 × raw (ocec800_raw)' },
  { co: 'ocec', cut: 800, sel: 'raw', sp: 'deriv2', label: 'Lowest OC/EC 800 × 2nd derivative' },
  { co: 'ocec', cut: 800, sel: 'raw', sp: 'airspec', label: 'Lowest OC/EC 800 × spline (ocec800_corr)' },
  { co: 'analogs', cut: 500, sel: 'airspec', sp: 'raw', label: 'Analogs 500 (spline-selected) × raw (analogs_selcorr)' },
  { co: 'analogs', cut: 500, sel: 'airspec', sp: 'airspec', label: 'Analogs 500 (spline-selected) × spline' },
]
export const PROTOCOL_NAME: Record<string, string> = { site_heldout: 'A · site-grouped 5-fold', app: 'B · interleaved 10-fold, within 5 %', app_fmm: 'B2 · interleaved 10-fold, first minimum' }
export const LOT_NAME: Record<string, string> = { all: 'all lots', '248': 'lot 248 only', '251': 'lot 251 only' }

/** the fixed-190 Deming fits of PROTOCOL_CONFIGS under each protocol and training lot, slopes at the reader's MAC */
export function protocolRows(calib: CalibFile, mac: number): Row[] {
  return PROTOCOL_CONFIGS.flatMap((c) => calib.grid
    .filter((r) => r.tg === 'addis' && r.el === 'all' && r.co === c.co && r.cut === c.cut && (r.sel ?? 'raw') === c.sel && r.sp === c.sp && r.mo in PROTOCOL_NAME && r.lot in LOT_NAME)
    .map((r): Row => ({ configuration: c.label, protocol: PROTOCOL_NAME[r.mo], lot: LOT_NAME[r.lot], k: r.k, slope: slopeAt(r.dm, mac, calib.mac_value), intercept: r.db, heldout: r.ho })))
    .sort((a, b) => LOT_ORDER.indexOf(String(a.lot)) - LOT_ORDER.indexOf(String(b.lot))
      || PROTOCOL_ORDER.indexOf(String(a.protocol)) - PROTOCOL_ORDER.indexOf(String(b.protocol)))
}
// explicit orders: an object's integer-like keys ('248') would come before 'all'
const LOT_ORDER = [LOT_NAME.all, LOT_NAME['248'], LOT_NAME['251']]
const PROTOCOL_ORDER = [PROTOCOL_NAME.site_heldout, PROTOCOL_NAME.app, PROTOCOL_NAME.app_fmm]
