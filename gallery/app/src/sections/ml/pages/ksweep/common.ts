/**
 * What the k-sweep page's charts share: the baselines and IMPROVE subsets as the export names them, their
 * colours, and row lookups on public/data/ml/ksweep.json (gallery/data/ml_pages/ksweep.py).
 */
import { BANK_COLOR, BASELINE_COLOR, INK } from '@/lib/theme'
import { num, type Row } from '@/lib/table'
import type { SpectralExport } from '../../../research/pages/spectralCommon'
import { rows } from '../../common'
import { ANALYSIS_SEASON_COLOR } from '@/lib/seasons'
import { TABLEAU } from '@/lib/palettes'

export type Data = SpectralExport

/** the Addis groups every readout is split by (dry_feb calendar, as the workflow binned them) */
export const GROUPS = ['All Addis', 'Dry (Oct-Feb)', 'Belg (Mar-May)', 'Kiremt (Jun-Sep)'] as const
export const SEASON_COLOR: Readonly<Record<string, string>> = ANALYSIS_SEASON_COLOR

/** the IMPROVE-only subsets (fixed before any Addis readout), in the workflow's order */
export const SUBSET_COLOR: Record<string, string> = {
  all: TABLEAU.teal, nsK_top10: TABLEAU.red, nsK_bottom50: TABLEAU.brown, ocec_top10: TABLEAU.purple, smoke_cohort: TABLEAU.orange,
}
export const SUBSET_SHORT: Record<string, string> = {
  all: 'all held-out IMPROVE', nsK_top10: 'non-soil K ≥ p90', nsK_bottom50: 'non-soil K ≤ median', ocec_top10: 'OC/EC ≥ p90', smoke_cohort: 'phase-2 smoke cohort',
}

export interface Baseline { key: string; label: string; long: string; color: string }

/** the five baselines in the notebook's legend order, with the gallery's colours for them */
export function baselinesOf(data: Data): Baseline[] {
  return rows(data, 'baselines').sort((a, b) => (num(a.order) ?? 0) - (num(b.order) ?? 0)).map((r) => {
    const key = String(r.baseline)
    return { key, label: String(r.label), long: String(r.long), color: BASELINE_COLOR[key] ?? BANK_COLOR[key] ?? INK.axis }
  })
}

/** the k the frozen protocol picks (1-SE rule on site-grouped CV) for a baseline in a cohort; `set` is the analog season */
export function protocolK(data: Data, baseline: string, cohort: 'standard' | 'analog', set?: string): number | null {
  const r = rows(data, 'protocol_k').find((p) => p.baseline === baseline && p.cohort === cohort && (cohort === 'standard' || p.set === set))
  return num(r?.protocol_k)
}

/** one value of a table row by (baseline, set, k) */
export const at = (rs: Row[], baseline: string, set: string, k: number, col: string): number | null =>
  num(rs.find((r) => r.baseline === baseline && r.set === set && r.k === k)?.[col])

/**
 * The k-sweep panels' shared y range, as a panel group shape (components/PanelGroup `share={{ y: padRange }}`): the union of
 * the panels' ranges (each already holds its reference line) padded 6 % either side.
 */
export const padRange = ([lo, hi]: [number, number]): [number, number] => { const pad = (hi - lo || 1) * 0.06; return [lo - pad, hi + pad] }
