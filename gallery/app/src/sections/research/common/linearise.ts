/**
 * Linearising HIPS on dark filters (the Research page "Linearising HIPS at Addis"). The IMPROVE curve of HIPS optical
 * depth against thermal EC on the filter is saturating, τ = aL / (1 + aL/τ∞); inverting it gives the optical depth a
 * proportional HIPS would have read, τ_lin = τ / (1 − τ/τ∞), which is research/ftir_hips_chem/scripts/
 * reference_closure.linearise. Because volume, deposit area and MAC cancel in the ratio, BC (Fabs ÷ MAC) is
 * linearised by the same factor. Everything here is a pure function of the per-filter optical depth and the ceiling,
 * so the page can follow any ceiling the reader picks.
 */
import { deming, median } from '../../../lib/stats.ts'

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
