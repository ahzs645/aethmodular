import type * as d3 from 'd3'
import { crossPath } from '@/lib/exclusionMode'
import { INK } from '@/lib/theme'

/** one excluded filter on a SweepPlot: its x (a date or a number) and y in the chart's units */
export interface SweepMark { x: Date | number; y: number }

/**
 * Filters → excluded: Mark on a SweepPlot (components/SweepPlot `children`): the excluded filters as the muted × of
 * components/ScatterLayer, where they fall inside the axes, so they never stretch the range. SweepPlot has no marks
 * of its own, so these carry no hover; the chart's ExcludedNote says how many there are.
 */
export function SweepMarks({ marks, x, y }: { marks: readonly SweepMark[]; x: unknown; y: d3.ScaleContinuousNumeric<number, number> }) {
  if (!marks.length) return null
  const sx = x as ((v: Date | number) => number) & { domain: () => (Date | number)[] }
  const [xa, xb] = sx.domain().map(Number), [ya, yb] = y.domain()
  const inside = (v: number, a: number, b: number) => v >= Math.min(a, b) && v <= Math.max(a, b)
  const d = marks.filter((m) => Number.isFinite(m.y) && inside(+m.x, xa, xb) && inside(m.y, ya, yb)).map((m) => crossPath(sx(m.x), y(m.y))).join('')
  return d ? <path className="scatter-marks" d={d} fill="none" stroke={INK.muted} strokeWidth={1.3} strokeOpacity={0.85} strokeLinecap="round" pointerEvents="none" /> : null
}
