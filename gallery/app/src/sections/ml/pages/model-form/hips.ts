/**
 * The closed-form half of ftir_08 (research/ftir_hips_chem/scripts/pls_transfer.py), for the model-form page's
 * HIPS-target and VIP charts: the transfer metrics on a set of filters, the component a CV curve picks, and the
 * VIP overlap and band summaries of two profiles. Pure (node-tested in hips.test.mjs); the fits themselves
 * (the PLS models, their VIP) are the notebook's.
 */

export interface TransferMetrics { n: number; slope: number; intercept: number; R2: number; RMSE: number; MAE: number; bias: number; median_bias: number }

const median = (v: number[]) => {
  const s = [...v].sort((a, b) => a - b), m = s.length >> 1
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

/**
 * pls_transfer.regression_metrics: observed on x, an OLS line of predicted on observed, R² of that line (the
 * squared correlation), and RMSE / MAE / bias against the 1:1 line. Null below three finite pairs.
 */
export function transferMetrics(observed: readonly number[], predicted: readonly number[]): TransferMetrics | null {
  const x: number[] = [], y: number[] = []
  observed.forEach((o, i) => { const p = predicted[i]; if (Number.isFinite(o) && Number.isFinite(p)) { x.push(o); y.push(p) } })
  const n = x.length
  if (n < 3) return null
  const mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n
  let sxx = 0, sxy = 0, syy = 0
  for (let i = 0; i < n; i++) { sxx += (x[i] - mx) ** 2; sxy += (x[i] - mx) * (y[i] - my); syy += (y[i] - my) ** 2 }
  const slope = sxy / sxx, intercept = my - slope * mx
  let ssRes = 0
  for (let i = 0; i < n; i++) ssRes += (y[i] - (slope * x[i] + intercept)) ** 2
  const d = y.map((v, i) => v - x[i])
  return {
    n, slope, intercept, R2: syy > 0 ? 1 - ssRes / syy : NaN,
    RMSE: Math.sqrt(d.reduce((a, v) => a + v * v, 0) / n),
    MAE: d.reduce((a, v) => a + Math.abs(v), 0) / n,
    bias: d.reduce((a, v) => a + v, 0) / n,
    median_bias: median(d),
  }
}

/** pls_transfer.select_components_cv's pick: the k with the lowest mean CV RMSE (the first, on a tie). */
export function chosenK(curve: readonly { k: number; rmse: number }[]): number | null {
  let best: { k: number; rmse: number } | null = null
  for (const c of curve) if (Number.isFinite(c.rmse) && (!best || c.rmse < best.rmse)) best = c
  return best?.k ?? null
}

/** ranks 1…n with ties averaged (scipy.stats.rankdata, as spearmanr uses) */
function ranks(v: readonly number[]): number[] {
  const order = v.map((_, i) => i).sort((a, b) => v[a] - v[b])
  const r = new Array<number>(v.length)
  for (let i = 0; i < order.length;) {
    let j = i
    while (j + 1 < order.length && v[order[j + 1]] === v[order[i]]) j++
    for (let t = i; t <= j; t++) r[order[t]] = (i + j) / 2 + 1
    i = j + 1
  }
  return r
}

function pearson(a: readonly number[], b: readonly number[]): number {
  const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n
  let sab = 0, saa = 0, sbb = 0
  for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2 }
  return sab / Math.sqrt(saa * sbb)
}

/** indices of the `n` largest values (numpy's argsort(...)[-n:]; a tie exactly at the cut is broken by index) */
function top(v: readonly number[], n: number): Set<number> {
  return new Set(v.map((_, i) => i).sort((a, b) => v[a] - v[b] || a - b).slice(-n))
}

export interface VipOverlap { spearman_r: number; important_a_n: number; important_b_n: number; important_jaccard: number; top_n_overlap: number; a_vip_mass_on_b_important: number }

/** pls_transfer.vip_overlap_summary of two VIP profiles on one grid */
export function vipOverlap(a: readonly number[], b: readonly number[], threshold = 1, topN = 200): VipOverlap {
  const ia = new Set<number>(), ib = new Set<number>()
  a.forEach((v, i) => { if (v >= threshold) ia.add(i) })
  b.forEach((v, i) => { if (v >= threshold) ib.add(i) })
  const both = [...ia].filter((i) => ib.has(i)).length
  const union = ia.size + ib.size - both
  const ta = top(a, topN), tb = top(b, topN)
  const mass = a.reduce((s, v) => s + v * v, 0)
  return {
    spearman_r: pearson(ranks(a), ranks(b)),
    important_a_n: ia.size, important_b_n: ib.size,
    important_jaccard: union ? both / union : NaN,
    top_n_overlap: [...ta].filter((i) => tb.has(i)).length,
    a_vip_mass_on_b_important: ib.size ? [...ib].reduce((s, i) => s + a[i] ** 2, 0) / mass : NaN,
  }
}

/** pls_transfer.summarize_vip_bands' share: the band's VIP² over the whole profile's, on low ≤ wn < high */
export function vipBandShare(wn: readonly number[], vip: readonly number[], low: number, high: number): number {
  let inBand = 0, total = 0
  wn.forEach((w, i) => { const v2 = vip[i] ** 2; total += v2; if (w >= low && w < high) inBand += v2 })
  return inBand / total
}

/** runs of consecutive points where both profiles reach VIP ≥ `threshold` (ftir_07 [5]'s shaded "VIP ≥ 1 in both") */
export function sharedRuns(a: readonly number[], b: readonly number[], threshold = 1): [number, number][] {
  const out: [number, number][] = []
  let start = -1
  for (let i = 0; i <= a.length; i++) {
    const on = i < a.length && a[i] >= threshold && b[i] >= threshold
    if (on && start < 0) start = i
    if (!on && start >= 0) { out.push([start, i - 1]); start = -1 }
  }
  return out
}
