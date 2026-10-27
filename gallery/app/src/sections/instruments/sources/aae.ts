/**
 * The Sources (AAE) page's arithmetic, kept apart from the charts so it can be tested
 * (aae.test.mjs) and so every chart classifies a filter the same way.
 *
 * The AAE here is the absorption Ångström exponent from absorption, BCc × the MA350 attenuation
 * cross-section, which is what gallery/data/export_aethalometer.py writes as "AAE UV-IR (aethalometer)".
 * An AAE taken from BCc itself is offset by ln(σ_short/σ_long)/ln(λ_short/λ_long), −1.016 for UV/IR
 * (research/ftir_hips_chem/scripts/optics.py, aae_from_columns): the source notebooks computed that one.
 */
import { colorsFor } from '../../../lib/groupings.ts'
import { regression, type RegressionStats } from '../../../lib/stats.ts'

/** MA350 attenuation cross-sections, m²/g (src/external/calibration.py; the exporter's SIGMA) */
export const MA350_SIGMA = { UV: 24.069, Blue: 19.070, Green: 17.028, Red: 14.091, IR: 10.120 } as const
/** MA350 channel centres, nm (research/ftir_hips_chem/scripts/config.py WAVELENGTHS_NM; meta.wavelengths_nm) */
export const MA350_NM = { UV: 375, Blue: 470, Green: 528, Red: 625, IR: 880 } as const
export type Channel = keyof typeof MA350_NM

/** the two exported AAE pairs */
export const AAE_PAIRS = {
  'UV-IR': { field: 'AAE UV-IR (aethalometer)', daily: 'AAE_UV_IR', short: 'UV', long: 'IR' },
  'Blue-IR': { field: 'AAE Blue-IR (aethalometer)', daily: 'AAE_Blue_IR', short: 'Blue', long: 'IR' },
} as const
export type AaePair = keyof typeof AAE_PAIRS
export const AAE_PAIR_KEYS = Object.keys(AAE_PAIRS) as AaePair[]

/** Source-region cut points, config.AAE_REGIONS (exported in meta.json as aae_regions). */
export interface AaeRegions { fossil_max: number; biomass_min: number }
export const DEFAULT_REGIONS: AaeRegions = { fossil_max: 0.9, biomass_min: 1.5 }
export const regionsOf = (meta: { aae_regions?: Partial<AaeRegions> } | null | undefined): AaeRegions => {
  const r = meta?.aae_regions
  return r && Number.isFinite(r.fossil_max) && Number.isFinite(r.biomass_min) ? { fossil_max: r.fossil_max!, biomass_min: r.biomass_min! } : DEFAULT_REGIONS
}

export const CLASSES = ['Fossil fuel', 'Mixed', 'Biomass'] as const
export type AaeClass = (typeof CLASSES)[number]
/** the ordered palette the built-in AAE grouping gets (lib/groupings colorsFor), so the page and the grouping agree */
export const CLASS_COLOR: Record<AaeClass, string> = Object.fromEntries(CLASSES.map((c, i) => [c, colorsFor(3)[i]])) as Record<AaeClass, string>

/** optics.classify_aae: fossil at or below fossil_max, biomass at or above biomass_min, mixed between; null without a value */
export function classify(v: unknown, r: AaeRegions = DEFAULT_REGIONS): AaeClass | null {
  if (typeof v !== 'number' || !Number.isFinite(v)) return null
  if (v <= r.fossil_max) return 'Fossil fuel'
  if (v >= r.biomass_min) return 'Biomass'
  return 'Mixed'
}

export interface ClassShares { n: number; shares: Record<AaeClass, number>; median: number | null; mean: number | null }
/** optics.aae_source_summary: each class's fraction of the finite values (0–1), with the median and mean */
export function classShares(values: readonly unknown[], r: AaeRegions = DEFAULT_REGIONS): ClassShares {
  const v = values.filter((x): x is number => typeof x === 'number' && Number.isFinite(x))
  const shares = { 'Fossil fuel': 0, Mixed: 0, Biomass: 0 } as Record<AaeClass, number>
  for (const x of v) shares[classify(x, r)!] += 1
  if (v.length) for (const c of CLASSES) shares[c] /= v.length
  const s = [...v].sort((a, b) => a - b)
  const median = !s.length ? null : s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2
  return { n: v.length, shares, median, mean: v.length ? v.reduce((a, b) => a + b, 0) / v.length : null }
}

/** optics.aae: −ln(b_short/b_long)/ln(λ_short/λ_long); null for a non-positive or missing channel */
export function aae(bShort: unknown, bLong: unknown, nmShort: number, nmLong: number): number | null {
  if (typeof bShort !== 'number' || typeof bLong !== 'number' || !(bShort > 0) || !(bLong > 0) || !Number.isFinite(bShort) || !Number.isFinite(bLong)) return null
  return -Math.log(bShort / bLong) / Math.log(nmShort / nmLong)
}

/**
 * How far an AAE taken from BCc sits from the absorption AAE, when the two channels' cross-sections
 * are in the ratio `ratio` (σ_short/σ_long): AAE_BCc = AAE_abs − ln(ratio)/ln(λ_long/λ_short).
 * With the MA350's own ratio (24.069/10.120 for UV/IR) this is −1.016.
 */
export const bccOffset = (ratio: number, nmShort: number, nmLong: number) => Math.log(ratio) / Math.log(nmShort / nmLong)
export const ma350Ratio = (short: Channel, long: Channel) => MA350_SIGMA[short] / MA350_SIGMA[long]

/**
 * The AAE a sample gets when its BCc is turned back into absorption with an assumed cross-section
 * ratio: what addis_01's "MAC sensitivity" did with its UV/IR MAC pairs. ratio 1 is BCc itself.
 */
export const aaeWithRatio = (aaeBcc: number, ratio: number, nmShort: number, nmLong: number) => aaeBcc - bccOffset(ratio, nmShort, nmLong)

/**
 * pandas rolling(window, min_periods).mean() over consecutive entries (the notebooks' 7-bin mean of
 * daily AAE): each output averages the finite values among this entry and the window − 1 before it.
 */
export function rollingMean(values: readonly (number | null | undefined)[], window = 7, minPeriods = 3): (number | null)[] {
  return values.map((_, i) => {
    let sum = 0, n = 0
    for (let j = Math.max(0, i - window + 1); j <= i; j++) {
      const v = values[j]
      if (typeof v === 'number' && Number.isFinite(v)) { sum += v; n++ }
    }
    return n >= minPeriods ? sum / n : null
  })
}

// ---------------------------------------------------------------- PMF dominance

/**
 * The two ways the notebooks called a PMF day's dominant source (dominant_source_comparison.ipynb):
 * the largest normalised GF fraction ("GF"), or the largest K_F contribution in µg/m³ ("K_F"). Each
 * comes with the dominant share of its own total, which the threshold notebooks sweep.
 */
export type Dominance = 'GF' | 'K_F'
export const DOMINANCE_LABEL: Record<Dominance, string> = { GF: 'GF (normalised fraction)', K_F: 'K_F (µg/m³ contribution)' }
export function dominantOf(values: Record<string, number | null | undefined>, labels: readonly string[]): { source: string; share: number } | null {
  let best: string | null = null, top = -Infinity, total = 0
  for (const l of labels) {
    const v = values[l]
    if (typeof v !== 'number' || !Number.isFinite(v)) continue
    total += v
    if (v > top) { top = v; best = l }
  }
  return best === null || !(total > 0) ? null : { source: best, share: top / total }
}

/** counts of (a, b) pairs, for an agreement matrix */
export function crosstab<A extends string, B extends string>(pairs: readonly { a: A | null; b: B | null }[]): Map<string, number> {
  const out = new Map<string, number>()
  for (const p of pairs) if (p.a != null && p.b != null) out.set(`${p.a}\u0000${p.b}`, (out.get(`${p.a}\u0000${p.b}`) ?? 0) + 1)
  return out
}
export const cell = (tab: Map<string, number>, a: string, b: string) => tab.get(`${a}\u0000${b}`) ?? 0

// ---------------------------------------------------------------- threshold sweeps

export interface SweepPair { x: number; y: number; t: number }
export interface SweepStep { threshold: number; low: RegressionStats | null; high: RegressionStats | null; nLow: number; nHigh: number }
/**
 * dominant_source_threshold_analysis.analyze_dominance_threshold_visual: at each threshold, the pairs
 * whose split value is at or below it ("low") and above it ("high"), each fitted when it has at least
 * three pairs (OLS, plus Deming for a 1:1 comparison).
 */
export function thresholdSweep(pairs: readonly SweepPair[], thresholds: readonly number[], lambda = 1): SweepStep[] {
  const ok = pairs.filter((p) => [p.x, p.y, p.t].every(Number.isFinite))
  const fit = (s: SweepPair[]) => (s.length >= 3 ? regression(s.map((p) => p.x), s.map((p) => p.y), { errorsInVariables: true, lambda }) : null)
  return thresholds.map((threshold) => {
    const low = ok.filter((p) => p.t <= threshold), high = ok.filter((p) => p.t > threshold)
    return { threshold, low: fit(low), high: fit(high), nLow: low.length, nHigh: high.length }
  })
}
