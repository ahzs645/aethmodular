import * as d3 from 'd3'

/**
 * One axis policy for every chart on the correlation tab, so the scatterplot,
 * the hexbin and the connected scatter agree about what "the same pair" looks
 * like. Chosen in the subset bar's axes row.
 *
 *  identity  a 1:1 line is drawn, the panel is squared and x and y share one
 *            span — otherwise "1:1" lies (PlotConfig.equal_axes)
 *  mode      Data           floor at min(0, data min): 0 is always visible,
 *                           negatives near the detection limit are kept
 *            From 0,0       both axes start at exactly 0; points below are
 *                           not drawn but stay in the statistics, and the
 *                           chart says how many
 *            Show intercept extend the axes so the fitted lines' y-intercepts
 *                           are on the canvas, and mark them
 */
export const AXES_MODES = ['Data', 'From 0,0', 'Show intercept'] as const
export type AxesMode = (typeof AXES_MODES)[number]

export interface AxesOpts {
  identity: boolean
  mode: AxesMode
  log: boolean
}

export const DEFAULT_AXES: AxesOpts = { identity: true, mode: 'From 0,0', log: false }

export interface PairDomain {
  x: [number, number]
  y: [number, number]
  /** number of points outside the domain and therefore not drawn */
  dropped: number
}

export function pairDomain(
  xs: number[],
  ys: number[],
  opts: AxesOpts,
  /** y values that must be on the canvas in 'Show intercept' mode (fit intercepts) */
  intercepts: number[] = []
): PairDomain {
  const extras = opts.mode === 'Show intercept' ? intercepts.filter(Number.isFinite) : []
  const lo = (v: number) => {
    if (opts.log) return Math.max(1e-4, v)
    if (opts.mode === 'From 0,0') return 0
    return Math.min(0, v)
  }
  const padOf = (a: number, b: number) => Math.max(1e-9, (b - a) * 0.06)

  let dom: PairDomain
  if (opts.mode === 'Show intercept' && !opts.log) {
    // A negative y observation should not create an equally large empty x
    // margin. Keep every observation and selected intercept, then equalise
    // the spans so the 1:1 line still has the same units per pixel.
    const xMin = Math.min(0, d3.min(xs) ?? 0)
    const yMin = Math.min(0, d3.min(ys) ?? 0, ...extras)
    const xMax = Math.max(0, d3.max(xs) ?? 1)
    const yMax = Math.max(0, d3.max(ys) ?? 1, ...extras)
    const pad = Math.max(1e-9, Math.max(xMax - xMin, yMax - yMin) * 0.02)
    const x: [number, number] = [xMin - pad, xMax + pad]
    const y: [number, number] = [yMin - pad, yMax + pad]
    if (opts.identity) {
      const span = Math.max(x[1] - x[0], y[1] - y[0])
      x[1] = x[0] + span
      y[1] = y[0] + span
    }
    dom = { x, y, dropped: 0 }
  } else if (opts.identity) {
    const minV = Math.min(d3.min(xs) ?? 0, d3.min(ys) ?? 0, ...extras)
    const maxV = Math.max(d3.max(xs) ?? 1, d3.max(ys) ?? 1)
    const pad = padOf(minV, maxV)
    const d: [number, number] = [lo(minV - pad), maxV + pad]
    dom = { x: d, y: d, dropped: 0 }
  } else {
    const [x0, x1] = d3.extent(xs) as [number, number]
    const [y0, y1] = d3.extent(ys) as [number, number]
    const yMin = Math.min(y0, ...extras)
    dom = {
      x: [lo(x0 - padOf(x0, x1)), x1 + padOf(x0, x1)],
      y: [lo(yMin - padOf(yMin, y1)), y1 + padOf(yMin, y1)],
      dropped: 0,
    }
  }
  let dropped = 0
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] < dom.x[0] || ys[i] < dom.y[0]) dropped++
  }
  dom.dropped = dropped
  return dom
}

export const inDomain = (d: PairDomain, x: number, y: number) => x >= d.x[0] && y >= d.y[0]
