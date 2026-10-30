/**
 * The empirical cumulative distribution, as the notebooks drew it (primary_tasks_notebook c11, the Fe/EC CDF:
 * `np.sort(v)` against `np.arange(1, n + 1) / n`): the share of values at or below each value. Pure, for the
 * Distribution tab's ECDF layout (charts/distribution/Histogram) and its node test (ecdf.test.mjs).
 */

/** the sorted finite values and, for each, the fraction of values at or below it (i / n, ties sharing the last) */
export function ecdf(values: readonly number[]): { x: number[]; p: number[] } {
  const x = values.filter(Number.isFinite).sort((a, b) => a - b)
  const n = x.length
  // a tied run takes the fraction of its last member, so F(v) is one number per value
  const p = x.map((_, i) => (i + 1) / n)
  for (let i = n - 2; i >= 0; i--) if (x[i] === x[i + 1]) p[i] = p[i + 1]
  return { x, p }
}

/** F(v): the fraction of the sorted values at or below v (0 below the first, 1 from the last) */
export function ecdfAt(sorted: readonly number[], v: number): number {
  if (!sorted.length || !Number.isFinite(v)) return NaN
  // the first index whose value is above v
  let lo = 0, hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (sorted[mid] <= v) lo = mid + 1
    else hi = mid
  }
  return lo / sorted.length
}

/**
 * The step path of an ECDF over [x0, x1], in data units: flat at 0 up to the first value, a riser at each value,
 * flat at 1 after the last. Points outside the range are pinned to its ends, so a clipped axis keeps the curve's
 * level where it leaves the frame.
 */
export function ecdfSteps(sorted: readonly number[], x0: number, x1: number): [number, number][] {
  const out: [number, number][] = [[x0, ecdfAt(sorted, x0)]]
  const n = sorted.length
  for (let i = 0; i < n; i++) {
    const v = sorted[i]
    if (v <= x0 || v > x1) continue
    if (i + 1 < n && sorted[i + 1] === v) continue
    out.push([v, out[out.length - 1][1]], [v, (i + 1) / n])
  }
  out.push([x1, out[out.length - 1][1]])
  return out
}
