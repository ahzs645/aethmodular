/**
 * The deployed HIPS calibration lines in the optics export, each with one colour shared by the HIPS envelope and
 * blank-line pages: lines in the order they were first used (earliest HIPS analysis date), coloured from SERIES.
 */
import type { Optics } from '../../common/optics'
import { lineKey, lineLabel, type LineRef } from '../../common/blankLines'
import { SERIES } from '../../common/ui'
import { INK } from '@/lib/theme'

/** the filter's deployed line, or null where the export has no line on the row */
export const refOf = (o: Optics): LineRef | null =>
  o.lot !== null && o.cal_intercept !== null && o.cal_slope !== null ? { lot: o.lot, intercept: o.cal_intercept, slope: o.cal_slope } : null

export const NO_LINE = '(no line on the row)'

export interface LinePalette { keys: string[]; color: (key: string | null) => string; label: (key: string | null) => string; ref: (key: string) => LineRef | undefined }

/** every line in the export, oldest first, plus any `extra` (e.g. network lines with no focus-site filter) */
export function linePalette(optics: readonly Optics[], extra: readonly LineRef[] = []): LinePalette {
  const first = new Map<string, { ref: LineRef; at: string }>()
  for (const o of optics) {
    const ref = refOf(o)
    if (!ref) continue
    const k = lineKey(ref), at = o.analysis_date ?? o.date ?? '9999'
    const cur = first.get(k)
    if (!cur || at < cur.at) first.set(k, { ref, at })
  }
  for (const ref of extra) if (!first.has(lineKey(ref))) first.set(lineKey(ref), { ref, at: '9999' })
  const keys = [...first.entries()].sort((a, b) => a[1].at.localeCompare(b[1].at) || a[0].localeCompare(b[0])).map(([k]) => k)
  return {
    keys,
    color: (k) => (k && keys.includes(k) ? SERIES[keys.indexOf(k) % SERIES.length] : INK.muted),
    label: (k) => (k && first.has(k) ? lineLabel(first.get(k)!.ref) : NO_LINE),
    ref: (k) => first.get(k)?.ref,
  }
}

/** an evenly sampled path of y = f(x) over [lo, hi] on a crossplot's scales */
export const curvePath = (f: (x: number) => number, lo: number, hi: number, x: (v: number) => number, y: (v: number) => number, steps = 48) =>
  Array.from({ length: steps + 1 }, (_, i) => lo + ((hi - lo) * i) / steps).map((v, i) => `${i ? 'L' : 'M'}${x(v)},${y(f(v))}`).join('')
