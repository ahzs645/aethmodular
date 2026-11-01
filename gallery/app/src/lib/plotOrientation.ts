import type { ScaleContinuousNumeric } from 'd3'

/** Preserve filter identity and provenance while changing only the displayed coordinate pair. */
export function orientPoints<T extends { x: number; y: number }>(points: T[], swapped: boolean): T[] {
  return swapped ? points.map(p => ({ ...p, x: p.y, y: p.x })) : points
}

/** Scales for layers expressed in the original axes, before OrientedPlotLayer transposes them. */
export function sourceScales<S extends ScaleContinuousNumeric<number, number>>(x: S, y: S, swapped: boolean) {
  return swapped ? { x: y.copy().range([...y.range()].reverse()), y: x.copy().range([...x.range()].reverse()) } : { x, y }
}

/** λ = variance(y error) / variance(x error): reversing the axes reverses that ratio. */
export function orientErrorRatio(lambda: number | null | undefined, swapped: boolean) {
  const value = lambda ?? 1
  return swapped ? 1 / value : value
}
