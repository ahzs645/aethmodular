/**
 * Warren White's patterned-deposit model beside the gallery's saturating curve (Meeting page "Warren White's model and
 * ours", pages/HipsWarrenPage.tsx). Both say the same thing: HIPS reports −ln of a transmittance averaged over a deposit
 * that is not evenly thick, so thick places hide absorption. With x the mean absorption density on the filter (what
 * HIPS would report for an even deposit) and z the local thickness relative to the mean (mean 1),
 *
 *     τ(x) = −ln E[exp(−x z)]
 *
 *   even deposit      z = 1                                             τ = x (Beer–Lambert)
 *   screen pattern    z = α/ω on a share h of the area, 1/ω elsewhere   White et al. (2025), Eq. 15 and 18; ω = 1 + (α − 1)h
 *   saturating curve  z is a Poisson number (mean τ∞) of clumps, each of exponentially distributed thickness:
 *                     E[exp(−x z)] = exp(−x / (1 + x/τ∞)) exactly, so τ = x / (1 + x/τ∞)
 *
 * The early bend is the variance of z: τ ≈ x − var(z) x²/2. It is h(1 − h)(α − 1)²/ω² for a screen, at most (1 − h)/h,
 * and 2/τ∞ for the curve, so the two can be compared on one number.
 */
import { deming, median } from './stats.ts'
import type { LinFilter } from './hipsLinearise.ts'

/** IMPROVE's support screen: the share of its area that is holes (McDade, Dillner & Indresand 2009; White et al. 2025, Table 2) */
export const IMPROVE_HOLES = 0.638
/** the network-wide amplitude White et al. (2025) suggest for IMPROVE, from its agreement with CSN */
export const WHITE_ALPHA = 10
/** an amplitude large enough to draw "fully pixelated" (nothing between the dots) */
export const FULL_ALPHA = 1e9

const omega = (alpha: number, h: number) => 1 + (alpha - 1) * h
/** the optical depth HIPS reports for a two-zone deposit of mean density x */
export function twoZone(x: number, alpha: number, h: number = IMPROVE_HOLES): number {
  const w = omega(alpha, h)
  return -Math.log(h * Math.exp((-alpha * x) / w) + (1 - h) * Math.exp(-x / w))
}
/** the same screen, with each zone reading along the saturating curve instead of Beer–Lambert */
export function composite(x: number, alpha: number, ceiling: number, h: number = IMPROVE_HOLES): number {
  const w = omega(alpha, h), curve = (u: number) => u / (1 + u / ceiling)
  return -Math.log(h * Math.exp(-curve((alpha * x) / w)) + (1 - h) * Math.exp(-curve(x / w)))
}
/** the optical depth a fully pixelated deposit cannot pass: absorptance = h */
export const pixelationLimit = (h: number = IMPROVE_HOLES): number => -Math.log(1 - h)
/** variance of the local relative thickness of a two-zone deposit */
export const patternVariance = (alpha: number, h: number = IMPROVE_HOLES): number => (h * (1 - h) * (alpha - 1) ** 2) / omega(alpha, h) ** 2
/** …and of the deposit the saturating curve describes */
export const curveVariance = (ceiling: number): number => 2 / ceiling
/** the share of its area with nothing on it, in the deposit the saturating curve describes */
export const curveClearShare = (ceiling: number): number => Math.exp(-ceiling)

const TOP = 60, STEPS = 1500
/** x values a response is tabulated on: dense where filters are, sparse out to 60 */
const GRID = Float64Array.from({ length: STEPS + 1 }, (_, i) => TOP * (i / STEPS) ** 2)

/**
 * The inverse of an increasing response: the mean density x that reads τ, by interpolation in a table. NaN where the
 * response never gets that far (a filter darker than the deposit could be).
 */
export function inverse(response: (x: number) => number): (tau: number) => number {
  const y = new Float64Array(GRID.length)
  for (let i = 1; i < GRID.length; i++) y[i] = Math.max(y[i - 1], response(GRID[i]))
  const last = y[y.length - 1]
  return (tau) => {
    if (!(tau > 0)) return tau === 0 ? 0 : NaN
    if (tau >= last - 1e-9) return NaN
    let lo = 0, hi = y.length - 1
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (y[mid] <= tau) lo = mid; else hi = mid }
    const span = y[hi] - y[lo]
    return span > 0 ? GRID[lo] + ((tau - y[lo]) / span) * (GRID[hi] - GRID[lo]) : GRID[lo]
  }
}

export interface HipsModel {
  key: string; label: string; short: string
  /** the optical depth HIPS reports for a true density x */
  response: (x: number) => number
  /** the true density for a reported optical depth; NaN where there is none */
  invert: (tau: number) => number
  /** variance of the local relative thickness */
  variance: number
}
export const EVEN: HipsModel = { key: 'none', label: 'As measured', short: 'as measured', response: (x) => x, invert: (t) => t, variance: 0 }
export function screenModel(key: string, label: string, short: string, alpha: number, h: number = IMPROVE_HOLES): HipsModel {
  const response = (x: number) => twoZone(x, alpha, h)
  return { key, label, short, response, invert: inverse(response), variance: patternVariance(alpha, h) }
}
export function curveModel(key: string, label: string, short: string, ceiling: number): HipsModel {
  return { key, label, short, response: (x) => x / (1 + x / ceiling), invert: (t) => (t < ceiling ? t / (1 - t / ceiling) : NaN), variance: curveVariance(ceiling) }
}
export function compositeModel(key: string, label: string, short: string, alpha: number, ceiling: number, h: number = IMPROVE_HOLES): HipsModel {
  const response = (x: number) => composite(x, alpha, ceiling, h)
  // independent factors: 1 + var = (1 + var_screen)(1 + var_curve)
  return { key, label, short, response, invert: inverse(response), variance: (1 + patternVariance(alpha, h)) * (1 + curveVariance(ceiling)) - 1 }
}

/** the factor a model multiplies a filter's Fabs (or BC) by */
export const factorOf = (m: Pick<HipsModel, 'invert'>, tau: number): number => m.invert(tau) / tau

export interface ModelFit { slope: number; intercept: number; n: number; past: number; mac: number; factor: number }
/**
 * The Deming fit of FTIR EC (y) on HIPS BC corrected by a model (x), the MAC at which its slope is 1 (MAC ÷ slope, as
 * hipsLinearise.macForSlopeOne), the filters the model has no answer for, and the median factor.
 */
export function fitModel(filters: readonly Pick<LinFilter, 'tau' | 'hips_bc' | 'ftir'>[], lambda: number, m: Pick<HipsModel, 'invert'>, mac: number): ModelFit | null {
  const xs: number[] = [], ys: number[] = [], ks: number[] = []
  let past = 0
  for (const f of filters) {
    if (f.ftir == null) continue
    const k = factorOf(m, f.tau)
    if (!Number.isFinite(k)) { past++; continue }
    xs.push(f.hips_bc * k); ys.push(f.ftir); ks.push(k)
  }
  const d = xs.length >= 3 ? deming(xs, ys, lambda) : null
  return d ? { slope: d.slope, intercept: d.intercept, n: xs.length, past, mac: d.slope > 0 ? mac / d.slope : NaN, factor: median(ks) } : null
}

/**
 * The saturating curve's deposit as a list of local thicknesses (mean 1), for a picture: a Poisson number of exponential
 * clumps per place, from a fixed seed.
 */
export function clumpedDeposit(ceiling: number, places: number, seed = 7): number[] {
  let s = seed
  const rnd = () => { s = (s * 1664525 + 1013904223) % 4294967296; return (s + 0.5) / 4294967296 }
  const out = Array.from({ length: places }, () => {
    let n = 0, p = Math.exp(-ceiling), c = p
    const u = rnd()
    while (u > c && n < 60) { n++; p *= ceiling / n; c += p }
    let z = 0
    for (let i = 0; i < n; i++) z -= Math.log(rnd()) / ceiling
    return z
  })
  const m = out.reduce((a, b) => a + b, 0) / places
  return out.map((z) => z / m)
}
/** a screen's deposit across a strip: `dots` holes, thick over each and thin between */
export function screenDeposit(alpha: number, h: number, places: number, dots: number): number[] {
  const w = omega(alpha, h)
  return Array.from({ length: places }, (_, i) => {
    const phase = (((i + 0.5) / places) * dots) % 1
    return Math.abs(phase - 0.5) <= h / 2 ? alpha / w : 1 / w
  })
}
