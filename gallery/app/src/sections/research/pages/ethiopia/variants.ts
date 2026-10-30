/**
 * Paired and per-k summaries of the June 2026 per-spectrum ETAD EC (the export's `per_filter`, one column per
 * calibration, indexed by `calibrations`), as pure functions node-tested in spectra.test.mjs:
 *
 *  - 07 [11]: each variant against CAL-0, spectrum by spectrum, and its median ratio (np.nanmedian(v / CAL-0));
 *  - 09 [12] panel C: median, mean and share negative against the component count;
 *  - 10 [12] panel B: lot-251-only against both lots at k = 30, and 10's lot-248 effect (ratio of the medians).
 */
import { median } from '../../common/stats.ts'

type Value = string | number | boolean | null
type Row = Record<string, Value>
const num = (v: Value | undefined): number => (typeof v === 'number' && Number.isFinite(v) ? v : NaN)

/** calibration label → its per-spectrum EC (NaN where missing), in spectrum order */
export function ecByCalibration(perFilter: readonly Row[], index: readonly Row[]): Map<string, number[]> {
  return new Map(index.map((c) => [String(c.calibration), perFilter.map((f) => num(f[String(c.column)]))]))
}

const finite = (v: number[]) => v.filter(Number.isFinite)

/** np.nanmedian(v / ref) over the spectra where both are finite */
export function medianRatio(v: readonly number[], ref: readonly number[]): number {
  const r = finite(v.map((x, i) => x / ref[i]))
  return r.length ? median(r) : NaN
}

/** np.nanmedian(v) / np.nanmedian(ref): 10's lot-248 effect */
export function ratioOfMedians(v: readonly number[], ref: readonly number[]): number {
  return median(finite([...v])) / median(finite([...ref]))
}

/** the component count in a June calibration label ("all 906, k21 (first major min)" → 21) */
export const kOf = (label: string): number | null => { const m = /\bk(\d+)\b/.exec(label); return m ? Number(m[1]) : null }

export interface KSummary { k: number; calibration: string; median: number; mean: number; pctNegative: number; n: number }

/** 09 [11]/[12]: per calibration of a family, its median and mean over the spectra with an EC (pandas skips NaN) and
 * its % negative over all of them (09's (conc_df < 0).mean(): a spectrum without a volume, so without an EC, counts as not negative) */
export function kSummary(ec: Map<string, number[]>, labels: readonly string[]): KSummary[] {
  return labels.flatMap((calibration) => {
    const k = kOf(calibration), v = ec.get(calibration)
    if (k == null || !v) return []
    const f = finite(v)
    return [{ k, calibration, n: f.length, median: median(f), mean: f.reduce((a, b) => a + b, 0) / f.length, pctNegative: (100 * f.filter((x) => x < 0).length) / v.length }]
  }).sort((a, b) => a.k - b.k)
}
