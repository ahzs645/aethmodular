/**
 * Regroup the IMPROVE filters and refit the HIPS ceiling in the browser (gallery/data/export_hips_improve_groupings.py,
 * `hips_linearise/improve_groupings.json`). Pure functions, node-tested (hipsGroupings.test.mjs).
 *
 * The published ceiling (τ∞ 2.26) is τ = aL / (1 + aL/τ∞) fitted through the medians of groups of filters, grouped by
 * the previous sample's TOR EC loading at the same site, with hand-set edges and groups under 150 filters left out.
 * The export holds, for each way of grouping, the filters in fine bins and in each fine bin a histogram of loading and
 * of τ, so any grouping on the fine edges can be rebuilt: its medians are read off the merged histograms (log
 * interpolation within a 4 % bin), and the curve is fitted as the workflow fits it (least squares on log τ, a in
 * [0, 1], τ∞ in [0.2, 100]), here by profiling τ∞.
 */

export type GroupBasis = 'previous' | 'own' | 'tau'
export type EdgeRule = 'published' | 'count' | 'log'
export type Residual = 'log' | 'abs'

export interface ValueBins { lower: number; ratio: number; bins: number }
/** a sparse run: [first value bin, count, count, …] */
export type Run = number[]
export interface FineBin { n: number; L: Run; t: Run }
export interface BasisData { label: string; fine: number[]; published: number[] | null; bins: FineBin[] }
export interface SweepRow {
  basis: GroupBasis; rule: EdgeRule; groups: number; used: number; edges: number[]
  a_per_ug: number; tau_inf: number; tau_inf_lo: number; tau_inf_hi: number
}
export interface GroupingsFile {
  generated: string; source: string; samples: number; min_n: number; counts: number[]; n_boot: number
  values: { loading: ValueBins; tau: ValueBins }
  bases: Record<GroupBasis, BasisData>
  sweep: SweepRow[]
}

export const A_BOUNDS: [number, number] = [0, 1]
export const TAU_BOUNDS: [number, number] = [0.2, 100]

// ------------------------------------------------------------------ edges, as fine-edge indices
/** the positive fine edge nearest a value on a log scale, as its index */
function snapIndex(value: number, fine: readonly number[]): number {
  let best = -1, dist = Infinity
  fine.forEach((e, i) => { if (e > 0) { const d = Math.abs(Math.log(e) - Math.log(value)); if (d < dist) { dist = d; best = i } } })
  return best
}

/**
 * A grouping's edges as indices into the fine edges, first 0 and last the top (export_hips_improve_groupings.rule_edges):
 * the published edges; equal-count groups, each inner edge the top of the fine bin where the running share first
 * reaches k/n; or n log-spaced groups between the 0.5 % and 99.95 % points. Edges that coincide collapse.
 */
export function ruleEdges(rule: EdgeRule, n: number, basis: Pick<BasisData, 'fine' | 'published' | 'bins'>): number[] {
  const { fine } = basis, top = fine.length - 1
  if (rule === 'published' && basis.published) {
    const out = basis.published.map((e) => (e <= 0 ? 0 : Math.min(top, snapIndex(e, fine))))
    return [...new Set(out)].sort((a, b) => a - b)
  }
  const counts = basis.bins.map((b) => b.n)
  const total = counts.reduce((a, b) => a + b, 0)
  const cum: number[] = []
  counts.reduce((acc, c, i) => (cum[i] = acc + c, acc + c), 0)
  const at = (share: number) => { const i = cum.findIndex((c) => c / total >= share); return (i < 0 ? counts.length - 1 : i) + 1 }
  let inner: number[]
  if (rule === 'count') inner = Array.from({ length: n - 1 }, (_, k) => at((k + 1) / n))
  else {
    const lo = fine[at(0.005)], hi = fine[at(0.9995)]
    inner = n > 2 ? Array.from({ length: n - 1 }, (_, k) => snapIndex(lo * (hi / lo) ** (k / (n - 2)), fine)) : [snapIndex(Math.sqrt(lo * hi), fine)]
  }
  return [0, ...[...new Set(inner.filter((i) => i > 0 && i < top))].sort((a, b) => a - b), top]
}

// ------------------------------------------------------------------ medians from the histograms
function addRun(into: number[], run: Run) {
  if (!run.length) return
  const first = run[0]
  for (let j = 1; j < run.length; j++) into[first + j - 1] += run[j]
}
/** the median of a histogram on log bins, interpolated in log within its bin */
export function histMedian(counts: readonly number[], bins: ValueBins): number {
  const n = counts.reduce((a, b) => a + b, 0)
  if (!n) return NaN
  const half = n / 2
  let cum = 0
  for (let k = 0; k < counts.length; k++) {
    if (cum + counts[k] >= half && counts[k] > 0) {
      const frac = (half - cum) / counts[k]
      return bins.lower * bins.ratio ** (k + frac)
    }
    cum += counts[k]
  }
  return NaN
}

export interface Group { lower: number; upper: number; n: number; loading: number; tau: number }
/** the groups between consecutive edges (fine-edge indices), with their filter counts and medians; small groups dropped */
export function groupsOf(file: Pick<GroupingsFile, 'values'>, basis: BasisData, edges: readonly number[], minN: number): { groups: Group[]; dropped: number; droppedFilters: number } {
  const groups: Group[] = []
  let dropped = 0, droppedFilters = 0
  for (let j = 0; j + 1 < edges.length; j++) {
    const L = new Array(file.values.loading.bins).fill(0), t = new Array(file.values.tau.bins).fill(0)
    let n = 0
    for (let k = edges[j]; k < edges[j + 1]; k++) { const b = basis.bins[k]; if (!b) continue; n += b.n; addRun(L, b.L); addRun(t, b.t) }
    if (n === 0) continue
    if (n < minN) { dropped++; droppedFilters += n; continue }
    groups.push({ lower: basis.fine[edges[j]], upper: basis.fine[edges[j + 1]], n, loading: histMedian(L, file.values.loading), tau: histMedian(t, file.values.tau) })
  }
  return { groups, dropped, droppedFilters }
}

// ------------------------------------------------------------------ the fit
const curve = (L: number, a: number, tauInf: number) => { const u = a * L; return u / (1 + u / tauInf) }
function sse(points: readonly { loading: number; tau: number }[], a: number, tauInf: number, residual: Residual) {
  let s = 0
  for (const p of points) {
    const f = curve(p.loading, a, tauInf)
    const r = residual === 'log' ? Math.log(f) - Math.log(p.tau) : f - p.tau
    s += r * r
  }
  return Number.isFinite(s) ? s : Infinity
}
/** golden-section minimum of a unimodal function on [lo, hi] */
function golden(f: (x: number) => number, lo: number, hi: number, iters = 60): number {
  const g = (Math.sqrt(5) - 1) / 2
  let a = lo, b = hi, c = b - g * (b - a), d = a + g * (b - a), fc = f(c), fd = f(d)
  for (let i = 0; i < iters; i++) {
    if (fc < fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c) } else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d) }
  }
  return (a + b) / 2
}
/** the best a for a fixed ceiling, searched on log a */
function bestA(points: readonly { loading: number; tau: number }[], tauInf: number, residual: Residual): { a: number; sse: number } {
  const la = golden((v) => sse(points, Math.exp(v), tauInf, residual), Math.log(1e-5), Math.log(A_BOUNDS[1]))
  const a = Math.exp(la)
  return { a, sse: sse(points, a, tauInf, residual) }
}

export interface CeilingFit {
  a: number; tauInf: number; rms: number; atBound: boolean
  /** the fit's error at each ceiling, a refitted at each: where it is lowest is the ceiling */
  profile: { tauInf: number; rms: number }[]
}
/** τ∞ and a through the group medians, as the workflow's least squares, with the profile it is read from */
export function fitCeiling(points: readonly { loading: number; tau: number }[], residual: Residual = 'log', gridSize = 90): CeilingFit | null {
  const ok = points.filter((p) => p.loading > 0 && p.tau > 0)
  if (ok.length < 3) return null
  const [lo, hi] = TAU_BOUNDS.map(Math.log)
  const profile = Array.from({ length: gridSize }, (_, i) => {
    const tauInf = Math.exp(lo + ((hi - lo) * i) / (gridSize - 1))
    return { tauInf, rms: Math.sqrt(bestA(ok, tauInf, residual).sse / ok.length) }
  })
  const i = profile.reduce((best, p, k) => (p.rms < profile[best].rms ? k : best), 0)
  const l = Math.log(profile[Math.max(0, i - 1)].tauInf), h = Math.log(profile[Math.min(gridSize - 1, i + 1)].tauInf)
  const tauInf = Math.exp(golden((v) => bestA(ok, Math.exp(v), residual).sse, l, h, 50))
  const { a, sse: s } = bestA(ok, tauInf, residual)
  return { a, tauInf, rms: Math.sqrt(s / ok.length), atBound: tauInf > TAU_BOUNDS[1] * 0.98 || tauInf < TAU_BOUNDS[0] * 1.02, profile }
}

/** τ∞ with each group left out in turn: which groups the ceiling rests on */
export function dropOne(groups: readonly Group[], residual: Residual = 'log'): { group: Group; tauInf: number }[] {
  return groups.map((g, i) => ({ group: g, tauInf: fitCeiling(groups.filter((_, k) => k !== i), residual, 40)?.tauInf ?? NaN }))
}

/** the sweep row for a grouping, when it is one of the precomputed ones (log residuals, the export's minimum group) */
export function sweepRow(file: Pick<GroupingsFile, 'sweep' | 'min_n'>, basis: GroupBasis, rule: EdgeRule, n: number, minN: number, residual: Residual): SweepRow | undefined {
  if (minN !== file.min_n || residual !== 'log') return undefined
  return file.sweep.find((r) => r.basis === basis && r.rule === rule && (rule === 'published' || r.groups === n))
}
