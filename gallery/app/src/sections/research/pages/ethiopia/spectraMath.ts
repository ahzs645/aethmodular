/**
 * The spectra behind the Ethiopia calibrations page's June 01, 04, 06 [21] and 13 [9] figures, as pure functions
 * (node-tested in spectraMath.test.mjs against the notebooks' saved and printed numbers):
 *
 *  - Savitzky–Golay smoothing / differentiation exactly as scipy.signal.savgol_filter (mode 'interp', delta 1);
 *  - June 01's straight-line flatten and CH / C=O band areas of the five Adama PTFE spectra;
 *  - month splits of the ETAD spectra, pooled exactly from per-month means and standard deviations;
 *  - medians and quartiles of June 13's per-spectrum organic-band fractions.
 *
 * The export carries absorbance in mAbs (× 1000); these helpers do not care about the unit.
 */
import { median, quantile } from '../../common/stats.ts'

// ---------------------------------------------------------------- Savitzky–Golay

/** Solve A x = b (small, dense) by Gauss–Jordan with partial pivoting. */
function solve(A: number[][], b: number[]): number[] {
  const n = b.length, M = A.map((row, i) => [...row, b[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]]
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      const f = M[r][c] / M[c][c]
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]
    }
  }
  return M.map((row, i) => row[n] / row[i])
}

/**
 * Weights over a window of `window` samples that return the `deriv`-th derivative, at sample `at` of the window
 * (0 … window − 1), of the order-`order` least-squares polynomial through the window (sample spacing 1).
 */
export function savgolWeights(window: number, order: number, deriv: number, at: number): number[] {
  const c = (window - 1) / 2, terms = order + 1
  // the weights are the row u Mᵀ … : w_i = Σ_j g_j (i − c)^j with (VᵀV) g = d, d_j = ∂^deriv (t − c)^j at t = at
  const VtV = Array.from({ length: terms }, (_, a) => Array.from({ length: terms }, (_, b) => {
    let s = 0
    for (let i = 0; i < window; i++) s += (i - c) ** (a + b)
    return s
  }))
  const d = Array.from({ length: terms }, (_, j) => {
    if (j < deriv) return 0
    let f = 1
    for (let k = 0; k < deriv; k++) f *= j - k
    return f * (at - c) ** (j - deriv)
  })
  const g = solve(VtV, d)
  return Array.from({ length: window }, (_, i) => g.reduce((s, gj, j) => s + gj * (i - c) ** j, 0))
}

/** scipy.signal.savgol_filter(y, window, order, deriv) with its defaults (delta 1, mode 'interp'). */
export function savgol(y: ArrayLike<number>, window: number, order: number, deriv = 0): number[] {
  const n = y.length, h = (window - 1) / 2
  if (window % 2 !== 1 || window <= order || n < window) throw new Error(`savgol: window ${window}, order ${order}, ${n} samples`)
  const centre = savgolWeights(window, order, deriv, h)
  const out = new Array<number>(n)
  const apply = (w: number[], start: number) => { let s = 0; for (let k = 0; k < window; k++) s += w[k] * y[start + k]; return s }
  for (let i = h; i < n - h; i++) out[i] = apply(centre, i - h)
  // 'interp': the edge points come from the polynomial fitted to the first / last full window
  for (let i = 0; i < h; i++) {
    out[i] = apply(savgolWeights(window, order, deriv, i), 0)
    out[n - 1 - i] = apply(savgolWeights(window, order, deriv, window - 1 - i), n - window)
  }
  return out
}

// ---------------------------------------------------------------- June 01: Adama band areas

/** np.trapz(y, x) */
export function trapz(y: readonly number[], x: readonly number[]): number {
  let s = 0
  for (let i = 1; i < y.length; i++) s += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2
  return s
}

/** June 01 [9]'s crude scatter correction: subtract the straight line through the spectrum's two end points. */
export function flatten(y: readonly number[], wn: readonly number[]): number[] {
  const n = y.length, x0 = wn[0], x1 = wn[n - 1]
  return y.map((v, i) => v - (y[0] + (y[n - 1] - y[0]) * (wn[i] - x0) / (x1 - x0)))
}

/** The area of a flattened spectrum over [lo, hi] cm⁻¹, integrated towards increasing wavenumber (01's np.trapz on the reversed band). */
export function bandArea(flat: readonly number[], wn: readonly number[], lo: number, hi: number): number {
  const idx = wn.map((_, i) => i).filter((i) => wn[i] >= lo && wn[i] <= hi)
  const ys = idx.map((i) => flat[i]), xs = idx.map((i) => wn[i])
  if (xs.length > 1 && xs[0] > xs[xs.length - 1]) { ys.reverse(); xs.reverse() }
  return trapz(ys, xs)
}

/** 1 for the largest value (pandas rank(ascending=False) without ties) */
export const rankDesc = (v: readonly number[]) => v.map((x) => 1 + v.filter((y) => y > x).length)

// ---------------------------------------------------------------- June 04: month splits

/** a month split: group name → months (1–12); months in no group fall to OTHER, undated spectra to UNDATED */
export interface MonthSplit { key: string; label: string; groups: [string, number[]][] }
export const OTHER = 'other months', UNDATED = 'no sampling date'
export const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
export const ALL = 'all 319 spectra'

export const MONTH_SPLITS: MonthSplit[] = [
  { key: 'all', label: 'All spectra (04 fig 1)', groups: [] },
  { key: 'placeholder', label: "04's placeholder split (fig 2)", groups: [['high-biomass months (Oct–Feb)', [10, 11, 12, 1, 2]], ['diesel months (Jun–Aug)', [6, 7, 8]]] },
  { key: 'dry_feb', label: 'Dry · Belg · Kiremt (Feb in Dry)', groups: [['Dry (Oct–Feb)', [10, 11, 12, 1, 2]], ['Belg (Mar–May)', [3, 4, 5]], ['Kiremt (Jun–Sep)', [6, 7, 8, 9]]] },
  { key: 'belg_feb', label: 'Bega · Belg · Kiremt (Feb in Belg)', groups: [['Bega (Oct–Jan)', [10, 11, 12, 1]], ['Belg (Feb–May)', [2, 3, 4, 5]], ['Kiremt (Jun–Sep)', [6, 7, 8, 9]]] },
  { key: 'month', label: 'Each month', groups: MONTH_NAMES.map((m, i): [string, number[]] => [m, [i + 1]]) },
]

/** the group a spectrum's month falls in under a split ('all' puts every spectrum in one group) */
export function groupOf(month: number | null | undefined, split: MonthSplit): string {
  if (split.key === 'all') return ALL
  if (month == null || !Number.isFinite(month)) return UNDATED
  return split.groups.find(([, ms]) => ms.includes(month))?.[0] ?? OTHER
}

/**
 * Pool per-month means and standard deviations (ddof 0, as exported) into a group's mean and sd per channel,
 * exactly: mean = Σ nₘ μₘ / N, sd² = Σ nₘ (σₘ² + μₘ²) / N − mean². `parts` are [n, mean[], sd[]] per month.
 */
export function pool(parts: [number, ArrayLike<number>, ArrayLike<number>][]): { n: number; mean: number[]; sd: number[] } {
  const use = parts.filter(([n]) => n > 0)
  const N = use.reduce((a, [n]) => a + n, 0)
  const len = use[0]?.[1].length ?? 0
  const mean = new Array<number>(len).fill(0), sd = new Array<number>(len).fill(0)
  if (!N) return { n: 0, mean, sd }
  for (let j = 0; j < len; j++) {
    let s1 = 0, s2 = 0
    for (const [n, mu, sigma] of use) { s1 += n * mu[j]; s2 += n * (sigma[j] ** 2 + mu[j] ** 2) }
    mean[j] = s1 / N
    sd[j] = Math.sqrt(Math.max(0, s2 / N - mean[j] ** 2))
  }
  return { n: N, mean, sd }
}

// ---------------------------------------------------------------- June 13 [9]: organic-band profile

export const BANDS = ['CH', 'C=O', 'OH', 'aromatic'] as const
export const BAND_LABEL: Record<(typeof BANDS)[number], string> = {
  CH: 'CH (2800–3000 cm⁻¹)', 'C=O': 'C=O (1650–1780 cm⁻¹)', OH: 'OH (3050–3500 cm⁻¹)', aromatic: 'aromatic (1580–1620 cm⁻¹)',
}

/** median and quartiles of each band's fraction over a set of spectra */
export function profileStats(rows: readonly Record<string, unknown>[]) {
  return BANDS.map((b) => {
    const v = rows.map((r) => r[b]).filter((x): x is number => typeof x === 'number' && Number.isFinite(x))
    return { band: b, n: v.length, median: v.length ? median(v) : NaN, q25: v.length ? quantile(v, 0.25) : NaN, q75: v.length ? quantile(v, 0.75) : NaN }
  })
}
