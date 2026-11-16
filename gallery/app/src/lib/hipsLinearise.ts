/**
 * Linearising HIPS on dark filters (the Meeting page "Linearising HIPS at Addis" (pages/HipsLinearisePage.tsx)). The IMPROVE curve of HIPS optical
 * depth against thermal EC on the filter is saturating, τ = aL / (1 + aL/τ∞); inverting it gives the optical depth a
 * proportional HIPS would have read, τ_lin = τ / (1 − τ/τ∞), which is research/ftir_hips_chem/scripts/
 * reference_closure.linearise. Because volume, deposit area and MAC cancel in the ratio, BC (Fabs ÷ MAC) is
 * linearised by the same factor. Everything here is a pure function of the per-filter optical depth and the ceiling,
 * so the page can follow any ceiling the reader picks.
 */
import { bisectRight, group } from 'd3'
import { deming, median } from './stats.ts'

/** deposit area of the 25 mm PTFE filter HIPS reads, cm² (hips_loading.DEPOSIT_AREA_CM2) */
export const DEPOSIT_AREA_CM2 = 3.53

/**
 * The IMPROVE ceilings the gallery offers (hips_loading.IMPROVE_CEILING and the 10 Nov closure): the central fit of
 * all sites 2005–2025, both ends of its 90 % interval, and the 2015–2025 fit. hipsLinearise.test.mjs fails if they
 * move away from the committed summary (hips_linearise/linearise.json).
 */
export const IMPROVE_CEILINGS = [
  { key: 'central', tau_inf: 2.2596, label: '2.26 (central)' },
  { key: 'lo', tau_inf: 1.8212, label: '1.82 (90 % low)' },
  { key: 'hi', tau_inf: 2.7014, label: '2.70 (90 % high)' },
  { key: 'era', tau_inf: 1.7092, label: '1.71 (2015–2025)' },
] as const
export const CENTRAL_CEILING: number = IMPROVE_CEILINGS[0].tau_inf

/** HIPS optical depth of a filter from Fabs (Mm⁻¹) and its sampled volume (m³): the laboratory's own HIPS τ */
export const opticalDepth = (fabs: number, volume: number): number => (fabs * volume) / (100 * DEPOSIT_AREA_CM2)

interface LoadingSample { id: string; site: string; date: string; loading: number; tau: number }
export type LoadingGroupSample = [site: string, date: string, previousLoading: number, loading: number, tau: number, id?: string]
export interface LoadingGroup { lower: number; upper: number | null; rows: LoadingGroupSample[] }

/** SPARTAN's nominal sampling cadence. Use the same loading edges as IMPROVE, with the prior sampled date at each site. */
export const SPARTAN_LOADING_MAX_GAP_DAYS = 9

export function previousLoadingGroups(samples: readonly LoadingSample[], edges: readonly number[]): LoadingGroup[] {
  const bins = new Map<number, LoadingGroupSample[]>()
  const eligible = samples.filter((s) => s.loading > 0 && Number.isFinite(s.loading) && Number.isFinite(s.tau) && Number.isFinite(Date.parse(s.date)))
  const sites = [...group(eligible, (s) => s.site)].sort(([a], [b]) => a.localeCompare(b))
  for (const [site, samplesAtSite] of sites) {
    const dates = [...group(samplesAtSite, (s) => s.date)].sort(([a], [b]) => a.localeCompare(b))
    for (let i = 1; i < dates.length; i++) {
      const [date, rows] = dates[i], [previousDate, previousRows] = dates[i - 1]
      const gapDays = (Date.parse(date) - Date.parse(previousDate)) / 86_400_000
      if (gapDays < 1 || gapDays > SPARTAN_LOADING_MAX_GAP_DAYS) continue
      // Co-located filters on one date contribute a single lag value, with no arbitrary order within that date.
      const lag = median(previousRows.map((s) => s.loading))
      const index = bisectRight(edges, lag) - 1
      if (index < 0) continue
      const members = bins.get(index) ?? []
      for (const row of rows) members.push([site, date, lag, row.loading, row.tau, row.id])
      bins.set(index, members)
    }
  }
  return [...bins].sort(([a], [b]) => a - b).map(([i, rows]) => ({ lower: edges[i], upper: edges[i + 1] ?? null, rows }))
}

/** the saturating IMPROVE response: optical depth for a proportional optical depth u = aL */
export const saturate = (u: number, ceiling: number): number => u / (1 + u / ceiling)

/** τ_lin = τ / (1 − τ/τ∞); NaN at or above the ceiling, where there is no finite answer */
export const linearise = (tau: number, ceiling: number): number => (tau < ceiling ? tau / (1 - tau / ceiling) : NaN)

/** τ_lin / τ = 1 / (1 − τ/τ∞): the factor that multiplies BC (and Fabs) */
export const correction = (tau: number, ceiling: number): number => (tau < ceiling ? 1 / (1 - tau / ceiling) : NaN)

/** dτ_lin/dτ = 1 / (1 − τ/τ∞)²: how much an absolute error in τ grows */
export const noiseGain = (tau: number, ceiling: number): number => (tau < ceiling ? 1 / (1 - tau / ceiling) ** 2 : NaN)

/** the exponent b in HIPS ∝ EC^b that the curve predicts on these filters: the mean of d ln τ / d ln u = 1 − τ/τ∞ */
export const predictedExponent = (taus: readonly number[], ceiling: number): number =>
  taus.length ? taus.reduce((s, t) => s + Math.min(1, Math.max(0, 1 - t / ceiling)), 0) / taus.length : NaN

export interface LinFilter { id: string; season: string; lot: string | null; tau: number; volume_m3: number; volume_known: boolean; hips_bc: number; ftir: number | null; pm25: number | null }

/** a filter's HIPS BC once linearised at this ceiling (µg/m³), NaN past the ceiling */
export const linearBc = (f: Pick<LinFilter, 'tau' | 'hips_bc'>, ceiling: number): number => f.hips_bc * correction(f.tau, ceiling)

/**
 * The Deming fit of FTIR EC on HIPS BC, as measured (ceiling null) or linearised at a ceiling, with the error ratio
 * λ the closure workflow uses (closeout LAMBDA_ADDIS). Filters past the ceiling drop out, as in the workflow.
 */
export function fitAgainstHips(filters: readonly LinFilter[], lambda: number, ceiling: number | null): { slope: number; intercept: number; n: number } | null {
  const xs: number[] = [], ys: number[] = []
  for (const f of filters) {
    if (f.ftir == null) continue
    const x = ceiling == null ? f.hips_bc : linearBc(f, ceiling)
    if (Number.isFinite(x)) { xs.push(x); ys.push(f.ftir) }
  }
  const d = deming(xs, ys, lambda)
  return d ? { ...d, n: xs.length } : null
}

/** median FTIR EC ÷ HIPS BC over a group of filters, as measured and linearised at a ceiling */
export function ratioMedians(filters: readonly LinFilter[], ceiling: number): { n: number; tau: number; measured: number; linearised: number } {
  const own = filters.filter((f) => f.ftir != null)
  return {
    n: own.length,
    tau: median(own.map((f) => f.tau)),
    measured: median(own.map((f) => f.ftir! / f.hips_bc)),
    linearised: median(own.map((f) => f.ftir! / linearBc(f, ceiling)).filter(Number.isFinite)),
  }
}
