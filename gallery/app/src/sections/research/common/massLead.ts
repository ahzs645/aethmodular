/**
 * ftir_46's mass lead as pure functions (node-tested in massLead.test.mjs): the calibration residual against
 * gravimetric PM2.5 mass at fixed prediction, per volume and per deposit, with a per-filter bootstrap.
 *
 * The MAC. The residual is r = f(X) − Fabs/MAC. Regress r on [f(X), mass, …]: f(X) is in the design, so its own
 * coefficient soaks up f(X) exactly and the mass coefficient is 0 − β/MAC, where β is the mass coefficient of
 * Fabs ~ f(X) + mass + … (Mm⁻¹ per µg/m³, which no MAC touches). So the partial slope at fixed f(X) is −β/MAC:
 * at MAC 10 the notebook's −0.0677 is −0.6767/10, and at any other MAC it is that × 10/MAC. Per deposit the
 * Fabs side is Fabs·V/(A·MAC), so its coefficient on mass per area carries the same 1/MAC. The raw slope (no f(X)
 * in the design) mixes f(X)'s own mass slope with Fabs's and does not scale; the page recomputes it at the MAC.
 */
import { quantile } from './stats.ts'

/** one filter of an explorer evaluation set, joined to the gallery row and the HIPS optics export */
export interface MassFilter {
  id: string
  /** the explorer's group (Addis: Ethiopian season; elsewhere DJF/MAM/JJA/SON) */
  group: string
  month: number | null
  /** HIPS Fabs (Mm⁻¹), the explorer's reference */
  fabs: number
  /** the calibration's prediction f(X) (µg/m³) */
  fx: number
  /** gravimetric PM2.5 (µg/m³) */
  mass: number
  volume: number | null
  area: number | null
  lot: string | null
}

export type Framing = 'volume' | 'deposit'

/** the columns a framing fits, at one MAC: y is the Fabs side, r the residual f(X) − Fabs/MAC, both in the framing's units */
export interface Framed { use: MassFilter[]; fx: number[]; m: number[]; y: number[]; r: number[] }

/**
 * Per volume: f(X), Fabs (Mm⁻¹) and mass in µg/m³, r = f(X) − Fabs/MAC. Per deposit (ftir_46 §5): the same as
 * loadings on the deposit, f(X)·V/A, Fabs·V/(A·MAC) and mass·V/A in µg/cm² (filters without volume and area
 * drop out). The coefficient on mass of y ~ f(X) + mass is then Mm⁻¹ per µg/m³ per volume (MAC-free), and µg of
 * HIPS-EC-equivalent per µg of mass per deposit (∝ 1/MAC).
 */
export function framed(filters: readonly MassFilter[], framing: Framing, mac: number): Framed {
  const use = framing === 'volume' ? [...filters] : filters.filter((f) => f.volume != null && f.area != null && f.volume > 0 && f.area > 0)
  const k = (f: MassFilter) => (framing === 'volume' ? 1 : f.volume! / f.area!)
  const fx = use.map((f) => f.fx * k(f)), m = use.map((f) => f.mass * k(f))
  const b = use.map((f) => (f.fabs / mac) * k(f))
  return { use, fx, m, y: framing === 'volume' ? use.map((f) => f.fabs) : b, r: fx.map((v, i) => v - b[i]) }
}

/** a seeded uniform generator (mulberry32), so an interval holds still while the page re-renders */
export function seeded(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** a short string as a seed, so each fit has its own fixed draws */
export const seedOf = (s: string) => [...s].reduce((h, c) => Math.imul(h ^ c.charCodeAt(0), 16777619) >>> 0, 2166136261)

/**
 * Least squares with an intercept on the rows `take` (all when omitted) of the columns X; the slope coefficients,
 * NaN for a column that is constant on those rows (a lot no resampled filter carries), as sklearn's minimum-norm
 * solve leaves the other coefficients unchanged. Null when the rest is singular.
 */
export function lstsq(X: readonly (readonly number[])[], y: readonly number[], take?: readonly number[]): number[] | null {
  const idx = take ?? y.map((_, i) => i)
  const n = idx.length
  const live = X.map((c) => idx.some((i) => c[i] !== c[idx[0]]))
  const cols = X.filter((_, j) => live[j])
  const p = cols.length + 1
  if (n <= p) return null
  // centred normal equations: the intercept drops out
  const mean = cols.map((c) => idx.reduce((s, i) => s + c[i], 0) / n)
  const my = idx.reduce((s, i) => s + y[i], 0) / n
  const A = Array.from({ length: p - 1 }, () => new Array<number>(p).fill(0))
  for (const i of idx) {
    const d = cols.map((c, j) => c[i] - mean[j]), dy = y[i] - my
    for (let a = 0; a < p - 1; a++) { for (let b = a; b < p - 1; b++) A[a][b] += d[a] * d[b]; A[a][p - 1] += d[a] * dy }
  }
  for (let a = 0; a < p - 1; a++) for (let b = 0; b < a; b++) A[a][b] = A[b][a]
  const k = p - 1
  for (let c = 0; c < k; c++) {
    let piv = c
    for (let r = c + 1; r < k; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r
    if (!(Math.abs(A[piv][c]) > 1e-12 * Math.max(1, Math.abs(A[c][c])))) return null
    ;[A[c], A[piv]] = [A[piv], A[c]]
    for (let r = 0; r < k; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let j = c; j <= k; j++) A[r][j] -= f * A[c][j] }
  }
  const solved = A.map((row, i) => row[k] / row[i])
  let s = 0
  return live.map((on) => (on ? solved[s++] : NaN))
}

/** a coefficient with its per-filter bootstrap 95 % interval (numpy's linear quantiles of the draws) */
export interface Estimate { est: number; lo: number; hi: number; n: number }

/** the coefficient of column `col` in y ~ X, resampling whole filters with replacement `draws` times (ftir_46 ols_coef) */
export function coefBoot(X: readonly (readonly number[])[], y: readonly number[], col: number, { draws = 2000, seed = 1 } = {}): Estimate | null {
  const n = y.length
  const fit = lstsq(X, y)
  if (!fit || !Number.isFinite(fit[col])) return null
  const rand = seeded(seed), boots: number[] = []
  const take = new Array<number>(n)
  for (let d = 0; d < draws; d++) {
    for (let i = 0; i < n; i++) take[i] = Math.floor(rand() * n)
    const b = lstsq(X, y, take)
    if (b && Number.isFinite(b[col])) boots.push(b[col])
  }
  return { est: fit[col], lo: quantile(boots, 0.025), hi: quantile(boots, 0.975), n }
}

/** an estimate times a constant (a negative one swaps the interval's ends) */
export const scaled = (e: Estimate | null, k: number): Estimate | null =>
  e && { ...e, est: e.est * k, lo: Math.min(e.lo * k, e.hi * k), hi: Math.max(e.lo * k, e.hi * k) }

/** x and y residualised on the base design: the added-variable plot's coordinates, whose OLS slope is the partial slope */
export function addedVariable(base: readonly (readonly number[])[], x: readonly number[], y: readonly number[]): { rx: number[]; ry: number[] } | null {
  const bx = lstsq(base, x), by = lstsq(base, y)
  if (!bx || !by) return null
  const resid = (v: readonly number[], c: number[]) => {
    const fitted = v.map((_, i) => base.reduce((s, col, j) => s + (Number.isFinite(c[j]) ? c[j] * col[i] : 0), 0))
    const off = v.reduce((s, t, i) => s + t - fitted[i], 0) / v.length
    return v.map((t, i) => t - fitted[i] - off)
  }
  return { rx: resid(x, bx), ry: resid(y, by) }
}

/** the annual cycle as two columns (ftir_46's sin/cos of the sampling month) */
export const seasonCols = (months: readonly number[]): number[][] =>
  [months.map((m) => Math.sin((2 * Math.PI * m) / 12)), months.map((m) => Math.cos((2 * Math.PI * m) / 12))]

/** one dummy column per level but the first (the first is the intercept's) */
export function dummies(levels: readonly string[]): number[][] {
  const kinds = [...new Set(levels)].sort()
  return kinds.slice(1).map((k) => levels.map((l) => (l === k ? 1 : 0)))
}

const iqr = (v: readonly number[]) => quantile(v, 0.75) - quantile(v, 0.25)
const corr = (a: readonly number[], b: readonly number[]) => {
  const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n
  let sab = 0, saa = 0, sbb = 0
  for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2 }
  return saa && sbb ? sab / Math.sqrt(saa * sbb) : NaN
}

/** ftir_46 §5's conditionings for the Addis residual's mass slope */
export type Conditioning = 'season' | 'full' | 'none'
export const CONDITIONING: Record<Conditioning, string> = {
  season: 'f(X) + season', full: 'f(X) + season + volume + lot', none: 'none (raw residual)',
}

/** a partial slope row of ftir_46 §5: the residual's slope on mass under a conditioning, × the mass IQR */
export interface PartialRow { framing: Framing; conditioning: Conditioning; slope: Estimate | null; slopeXIqr: number | null; n: number }

/**
 * ftir_46 §5 (cell 13) at one MAC: r ~ base + mass for base = f(X) + season, f(X) + season + volume + lot, and
 * nothing. Season is the month's sin/cos; filters without a month (or, for the full base, without volume or lot) drop out.
 */
export function partialSlopes(filters: readonly MassFilter[], framing: Framing, mac: number, opts: { draws?: number; seed?: number } = {}): PartialRow[] {
  return (['season', 'full', 'none'] as Conditioning[]).map((cond) => {
    const need = (f: MassFilter) => cond === 'none' || (f.month != null && (cond === 'season' || (f.volume != null && f.lot != null)))
    const F = framed(filters.filter(need), framing, mac)
    const base: number[][] = cond === 'none' ? [] : [F.fx, ...seasonCols(F.use.map((f) => f.month!))]
    if (cond === 'full') base.push(F.use.map((f) => f.volume!), ...dummies(F.use.map((f) => f.lot!)))
    const slope = F.use.length > base.length + 3 ? coefBoot([...base, F.m], F.r, base.length, { draws: opts.draws, seed: (opts.seed ?? 1) + seedOf(cond + framing) }) : null
    return { framing, conditioning: cond, slope, slopeXIqr: slope ? slope.est * iqr(F.m) : null, n: F.use.length }
  })
}

/** ftir_46 §7's per-site fit in one framing: the mass coefficient of y ~ f(X) + mass, and the residual's slopes */
export interface SiteFit {
  framing: Framing
  n: number
  corrMassFx: number
  /** coefficient on mass in Fabs ~ f(X) + mass (per volume, Mm⁻¹ per µg/m³) or its per-deposit form (µg EC-eq per µg) */
  massCoef: Estimate | null
  /** the residual's slope on mass at fixed f(X): −massCoef/MAC per volume, −massCoef per deposit (exact) */
  rPartial: Estimate | null
  /** the residual's raw slope on mass (no f(X) in the design) */
  rSlope: number | null
  /** the same fit inside a mass-per-area window (the Addis deposit-loading range), when ≥ 15 filters sit in it */
  inRange: Estimate | null
  nInRange: number
  massIqr: number
}

/** the mass per area of each filter with volume and area (µg/cm²): the window `siteFit` restricts to */
export const massPerArea = (f: MassFilter): number | null => (f.volume && f.area ? (f.mass * f.volume) / f.area : null)

/**
 * ftir_46 fit_rows for one site (or a pool) at one MAC. `range` is the Addis deposit-loading window [min, max] of
 * mass per area; filters without volume and area cannot be placed in it and stay out of the in-range fit.
 */
export function siteFit(filters: readonly MassFilter[], framing: Framing, mac: number, range: [number, number] | null,
  { draws = 2000, seed = 1 }: { draws?: number; seed?: number } = {}): SiteFit | null {
  const F = framed(filters, framing, mac)
  if (F.use.length < 6) return null
  const massCoef = coefBoot([F.fx, F.m], F.y, 1, { draws, seed })
  // r = f(X) − b with b = y/MAC per volume and b = y per deposit, so its mass coefficient at fixed f(X) is −(y's)
  const rPartial = scaled(massCoef, framing === 'volume' ? -1 / mac : -1)
  const raw = lstsq([F.m], F.r)
  const inside = range ? F.use.map((f) => { const a = massPerArea(f); return a != null && a >= range[0] && a <= range[1] }) : []
  const nInRange = inside.filter(Boolean).length
  let inRange: Estimate | null = null
  if (range && nInRange >= 15) {
    const pick = <T,>(v: readonly T[]) => v.filter((_, i) => inside[i])
    inRange = coefBoot([pick(F.fx), pick(F.m)], pick(F.y), 1, { draws, seed: seed + 7 })
  }
  return { framing, n: F.use.length, corrMassFx: corr(F.m, F.fx), massCoef, rPartial, rSlope: raw ? raw[0] : null, inRange, nInRange, massIqr: iqr(F.m) }
}
