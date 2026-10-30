/**
 * ftir_47's blank-line audit as pure functions (node-tested in blankLines.test.mjs). SPARTAN reduces each filter
 * with one deployed calibration line, τ = ln((Intercept + Slope·R1) / T1), Fabs = 100·τ·A/V, fitted to blanks; a
 * line is keyed on (lot, Intercept, Slope), never on lot alone, because a lot is re-calibrated whenever the
 * instrument changes. The notebook refits each line's blanks (OLS of T1 on R1, linear and quadratic, `np.polyfit`),
 * pools a lot's blanks into one quadratic (the 8561843 construction it retires), and recomputes every sample's Fabs
 * under each construction. Polynomials here are lowest power first (numpy's are highest first).
 */
import { quadratic, quantile } from './stats.ts'

/** lines with fewer blanks get no refit (hips_lab.blank_lines, ftir_47 MIN_BLANKS) */
export const MIN_BLANKS = 5

export interface LineRef { lot: string; intercept: number; slope: number }
/** hips_lab._line_id: the deployed line's key */
export const lineKey = (l: LineRef) => `${l.lot}|I=${l.intercept.toFixed(3)}|S=${l.slope.toFixed(4)}`
/** "251 · 1416.2 − 2.783·R1" */
export const lineLabel = (l: LineRef) => `${l.lot} · ${l.intercept.toFixed(1)} ${l.slope < 0 ? '−' : '+'} ${Math.abs(l.slope).toFixed(3)}·R1`

export interface BlankPoint extends LineRef { R1: number; T1: number; date?: string | null }
/** a loaded sample: its shipped Fabs and dv = 100·A/V, so Fabs = τ·dv */
export interface SamplePoint extends LineRef { id: string; code: string; R1: number; T1: number; fabs: number; dv: number }

export type Poly = number[]
export const polyval = (c: Poly, x: number) => c.reduce((s, v, i) => s + v * x ** i, 0)

/** population standard deviation (np.std), the notebook's rms */
const std = (v: readonly number[]) => {
  if (!v.length) return NaN
  const m = v.reduce((a, b) => a + b, 0) / v.length
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length)
}

/** np.polyfit(x, y, 1), lowest power first */
export function linFit(xs: readonly number[], ys: readonly number[]): Poly | null {
  const n = xs.length
  if (n < 2) return null
  const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n
  let sxx = 0, sxy = 0
  for (let i = 0; i < n; i++) { sxx += (xs[i] - mx) ** 2; sxy += (xs[i] - mx) * (ys[i] - my) }
  if (sxx === 0) return null
  const b = sxy / sxx
  return [my - b * mx, b]
}

/** np.polyfit(x, y, 2), lowest power first; fitted on centred x for conditioning, then expanded */
export function quadFit(xs: readonly number[], ys: readonly number[]): Poly | null {
  if (xs.length < 3) return null
  const m = xs.reduce((a, b) => a + b, 0) / xs.length
  const f = quadratic(xs.map((x) => x - m), ys)
  if (!f) return null
  const [c0, c1, c2] = f.coef
  return [c0 - c1 * m + c2 * m * m, c1 - 2 * c2 * m, c2]
}

const rmsAbout = (c: Poly | null, R: readonly number[], T: readonly number[]) => (c ? std(T.map((t, i) => t - polyval(c, R[i]))) : null)

export interface BlankLine extends LineRef {
  key: string; n: number
  r1Min: number; r1Max: number; t1Min: number | null; t1Max: number | null
  /** first and last analysis date of its blanks, where known */
  from: string | null; to: string | null
  /** scatter of the blanks about the shipped line, and about the refits (null: unfit or not known) */
  rmsDeployed: number | null; rmsLin: number | null; rmsQuad: number | null
  lin: Poly | null; quad: Poly | null
}

const groupBy = <T>(items: readonly T[], keyOf: (t: T) => string) => {
  const m = new Map<string, T[]>()
  for (const t of items) { const k = keyOf(t); m.set(k, [...(m.get(k) ?? []), t]) }
  return m
}

/** ftir_47 §1 (the line ledger): one row per deployed line, refitted when it has `minBlanks` blanks */
export function fitBlankLines(blanks: readonly BlankPoint[], minBlanks = MIN_BLANKS): BlankLine[] {
  return [...groupBy(blanks, lineKey).entries()].map(([key, g]) => {
    const R = g.map((b) => b.R1), T = g.map((b) => b.T1), dates = g.map((b) => b.date).filter((d): d is string => !!d).sort()
    const { lot, intercept, slope } = g[0]
    const fit = g.length >= minBlanks
    const lin = fit ? linFit(R, T) : null, quad = fit ? quadFit(R, T) : null
    return { key, lot, intercept, slope, n: g.length, r1Min: Math.min(...R), r1Max: Math.max(...R), t1Min: Math.min(...T), t1Max: Math.max(...T),
      from: dates[0] ?? null, to: dates[dates.length - 1] ?? null,
      rmsDeployed: std(T.map((t, i) => t - intercept - slope * R[i])), lin, quad, rmsLin: rmsAbout(lin, R, T), rmsQuad: rmsAbout(quad, R, T) }
  }).sort((a, b) => a.lot.localeCompare(b.lot) || (a.from ?? '').localeCompare(b.from ?? '') || b.intercept - a.intercept)
}

/** calibration_runs.json hips.blanks (hips_lab.blank_lines over every SPARTAN FB/LB blank): numpy order reversed */
export interface NetworkBlankLine { lot: string; deployed_intercept: number; deployed_slope: number; n: number; lin: number[]; quad: number[]; r1_min: number; r1_max: number; rms_lin: number; rms_quad: number }
export function networkLines(record: Record<string, NetworkBlankLine> | null | undefined): BlankLine[] {
  return Object.entries(record ?? {}).map(([key, b]) => ({
    key, lot: String(b.lot), intercept: b.deployed_intercept, slope: b.deployed_slope, n: b.n,
    r1Min: b.r1_min, r1Max: b.r1_max, t1Min: null, t1Max: null, from: null, to: null,
    rmsDeployed: null, rmsLin: b.rms_lin, rmsQuad: b.rms_quad, lin: [...b.lin].reverse(), quad: [...b.quad].reverse(),
  }))
}

export interface LotPool {
  lot: string; n: number; nLines: number; nFitted: number
  r1Min: number; r1Max: number; lin: Poly | null; quad: Poly | null; rmsLin: number | null; rmsQuad: number | null
  /** ftir_47 §1: do the blank R1 ranges of the lot's fitted lines share a common interval (null: one line) */
  shareInterval: boolean | null
}

/** ftir_47 §1 (the lot ledger): every blank of a lot pooled into one linear and one quadratic */
export function poolLots(blanks: readonly BlankPoint[], lines: readonly BlankLine[], minBlanks = MIN_BLANKS): LotPool[] {
  return [...groupBy(blanks, (b) => b.lot).entries()].filter(([, g]) => g.length >= minBlanks).map(([lot, g]) => {
    const R = g.map((b) => b.R1), T = g.map((b) => b.T1)
    const lin = linFit(R, T), quad = quadFit(R, T)
    const fitted = lines.filter((l) => l.lot === lot && l.lin)
    return { lot, n: g.length, nLines: new Set(g.map(lineKey)).size, nFitted: fitted.length, r1Min: Math.min(...R), r1Max: Math.max(...R),
      lin, quad, rmsLin: rmsAbout(lin, R, T), rmsQuad: rmsAbout(quad, R, T),
      shareInterval: fitted.length > 1 ? Math.max(...fitted.map((l) => l.r1Min)) <= Math.min(...fitted.map((l) => l.r1Max)) : null }
  }).sort((a, b) => a.lot.localeCompare(b.lot))
}

export const CONSTRUCTIONS = ['deployed', 'line_lin', 'line_quad', 'lot_quad'] as const
export type Construction = (typeof CONSTRUCTIONS)[number]
export const CONSTRUCTION_LABEL: Record<Construction, string> = {
  deployed: 'deployed line', line_lin: 'per-line linear', line_quad: 'per-line quadratic', lot_quad: 'lot-pooled quadratic',
}

/** τ = ln(top / T1), null where either side is not positive */
export const tauFromTop = (top: number | null, T1: number) => (top !== null && top > 0 && T1 > 0 ? Math.log(top / T1) : null)

export interface Exposure {
  s: SamplePoint; line: BlankLine | null
  /** R1 below or above its own line's blank R1 range (null: no fitted line) */
  below: boolean | null; above: boolean | null
  /** Fabs recomputed under each construction (Mm⁻¹), and its change from the deployed line */
  fabs: Record<Construction, number | null>; dfabs: Record<Exclude<Construction, 'deployed'>, number | null>
}

/** ftir_47 §2: each sample's Fabs under the deployed line, its own line's refits and its lot's pooled quadratic */
export function exposeSamples(samples: readonly SamplePoint[], lines: readonly BlankLine[], lots: readonly LotPool[]): Exposure[] {
  const byKey = new Map(lines.map((l) => [l.key, l])), byLot = new Map(lots.map((l) => [l.lot, l]))
  return samples.map((s) => {
    const line = byKey.get(lineKey(s)) ?? null, fitted = line?.lin ? line : null, lot = byLot.get(s.lot)
    const at = (c: Poly | null | undefined) => { const t = tauFromTop(c ? polyval(c, s.R1) : null, s.T1); return t === null ? null : t * s.dv }
    const fabs = { deployed: at([s.intercept, s.slope]), line_lin: at(fitted?.lin), line_quad: at(fitted?.quad), lot_quad: at(lot?.quad) }
    const d = (v: number | null) => (v !== null && fabs.deployed !== null ? v - fabs.deployed : null)
    return { s, line: fitted, below: fitted ? s.R1 < fitted.r1Min : null, above: fitted ? s.R1 > fitted.r1Max : null,
      fabs, dfabs: { line_lin: d(fabs.line_lin), line_quad: d(fabs.line_quad), lot_quad: d(fabs.lot_quad) } }
  })
}

const absQ = (v: (number | null)[], p: number) => { const a = v.filter((x): x is number => x !== null && Number.isFinite(x)).map(Math.abs); return a.length ? quantile(a, p) : NaN }

export interface SiteExposure {
  code: string; n: number; nUnfit: number; nLines: number
  below: number; above: number; outside: number; medianFabs: number
  /** median and p90 |ΔFabs| against the deployed line (Mm⁻¹) */
  med: Record<Exclude<Construction, 'deployed'>, number>; p90: Record<Exclude<Construction, 'deployed'>, number>
}

/** ftir_47 §2's site table: samples on a fitted line only, as the notebook's `net` */
export function siteExposure(exps: readonly Exposure[]): SiteExposure[] {
  return [...groupBy(exps, (e) => e.s.code).entries()].map(([code, all]) => {
    const g = all.filter((e) => e.line)
    const share = (f: (e: Exposure) => boolean) => (g.length ? g.filter(f).length / g.length : NaN)
    const q = (p: number) => ({ line_lin: absQ(g.map((e) => e.dfabs.line_lin), p), line_quad: absQ(g.map((e) => e.dfabs.line_quad), p), lot_quad: absQ(g.map((e) => e.dfabs.lot_quad), p) })
    return { code, n: g.length, nUnfit: all.length - g.length, nLines: new Set(g.map((e) => e.line!.key)).size,
      below: share((e) => !!e.below), above: share((e) => !!e.above), outside: share((e) => !!e.below || !!e.above),
      medianFabs: g.length ? quantile(g.map((e) => e.s.fabs), 0.5) : NaN, med: q(0.5), p90: q(0.9) }
  })
}

export interface LineBreakdown { key: string; ref: LineRef; n: number; r1Min: number; r1Max: number; line: BlankLine | null; median: Record<Construction, number> }

/** ftir_47's Pasadena breakdown, for any site: per deployed line, the median Fabs each construction implies */
export function lineBreakdown(exps: readonly Exposure[]): LineBreakdown[] {
  return [...groupBy(exps, (e) => lineKey(e.s)).entries()].map(([key, g]) => {
    const med = (c: Construction) => { const v = g.map((e) => e.fabs[c]).filter((x): x is number => x !== null && Number.isFinite(x)); return v.length ? quantile(v, 0.5) : NaN }
    return { key, ref: { lot: g[0].s.lot, intercept: g[0].s.intercept, slope: g[0].s.slope }, n: g.length, r1Min: Math.min(...g.map((e) => e.s.R1)), r1Max: Math.max(...g.map((e) => e.s.R1)),
      line: g[0].line, median: { deployed: med('deployed'), line_lin: med('line_lin'), line_quad: med('line_quad'), lot_quad: med('lot_quad') } }
  }).sort((a, b) => b.n - a.n)
}

/** ftir_47's check: the deployed line reproduces the shipped Fabs (max |Δ| over the samples) */
export function recomputeCheck(exps: readonly Exposure[]): { n: number; maxDiff: number } {
  const d = exps.flatMap((e) => (e.fabs.deployed !== null ? [Math.abs(e.fabs.deployed - e.s.fabs)] : []))
  return { n: d.length, maxDiff: d.length ? Math.max(...d) : NaN }
}

/** warren_meeting_slides: the R1 the deployed line expects at a sample's T1, and how far R1 falls short of it (%) */
export function rSuppression(s: { R1: number; T1: number; intercept: number; slope: number }): { rExpected: number; pct: number } | null {
  if (!s.slope) return null
  const rExpected = (s.T1 - s.intercept) / s.slope
  return rExpected ? { rExpected, pct: (1 - s.R1 / rExpected) * 100 } : null
}
