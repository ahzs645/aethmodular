import { FONT, INK } from '@/lib/theme'

/**
 * The three kinds of reference line every chart draws, one style each
 * (docs/gallery-chart-standards.md §1). Positions are in pixels, so any scale
 * (linear, log, band) works:
 *
 *   <IdentityLine x1={x(d0)} y1={y(d0)} x2={x(d1)} y2={y(d1)} />    1:1 on a crossplot
 *   <RefLine orient="v" at={x(1)} from={0} to={ih} />               agreement: slope 1, intercept 0, zero
 *   <Marker orient="h" at={y(0.85)} from={0} to={iw} label="0.85" />  a data value to compare against
 *   <RefBand orient="v" at={[x(0.8), x(1.2)]} size={iw} length={ih} />  a range the data should fall in (slope box)
 */
const identity = { get stroke() { return INK.identity }, strokeWidth: 1.5, strokeDasharray: '5 4' }
const agreement = { get stroke() { return INK.axis }, strokeOpacity: 0.7, strokeWidth: 1 }
const marker = { strokeWidth: 1, strokeDasharray: '3 3' }

export function IdentityLine({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) {
  return <line x1={x1} y1={y1} x2={x2} y2={y2} {...identity} pointerEvents="none" data-identity="" />
}

type Span = { orient: 'v' | 'h'; at: number; from: number; to: number }
const ends = ({ orient, at, from, to }: Span) => orient === 'v' ? { x1: at, x2: at, y1: from, y2: to } : { x1: from, x2: to, y1: at, y2: at }

/** Where the data would sit if the methods agreed (slope 1, intercept 0, a zero difference). */
export function RefLine(props: Span) {
  return <line {...ends(props)} {...agreement} pointerEvents="none" />
}

/**
 * A value to compare against: a guardrail floor, the chosen k, a cutoff, a median.
 * `color` ties it to one series (a bank, a cohort); the label sits at the line's start.
 */
export function Marker({ label, color = INK.text, opacity = 0.75, labelSide = 'right', labelDy = 0, ...span }: Span & {
  label?: string; color?: string; opacity?: number
  /** vertical markers: which side of the line the label sits (left near the right edge) */
  labelSide?: 'left' | 'right'
  /** push the label down, so several markers' labels do not overlap */
  labelDy?: number
}) {
  const e = ends(span)
  return <g pointerEvents="none">
    <line {...e} {...marker} stroke={color} strokeOpacity={opacity} />
    {label && (span.orient === 'v'
      ? <text x={labelSide === 'left' ? e.x1 - 4 : e.x1 + 4} y={Math.min(e.y1, e.y2) + 10 + labelDy} textAnchor={labelSide === 'left' ? 'end' : 'start'} fontSize={10} fontFamily={FONT.mono} fill={color}>{label}</text>
      : <text x={Math.max(e.x1, e.x2) - 4} y={e.y1 - 4} textAnchor="end" fontSize={10} fontFamily={FONT.mono} fill={color}>{label}</text>)}
  </g>
}

/**
 * A range the data should fall in: the guardrail slope box (calib.slope_box), one style on every
 * calibration chart (docs/gallery-chart-standards.md §1 GuardrailBands). `orient="v"` shades the
 * x range between the two positions `at` over the plot height `length`; `orient="h"` a y range
 * across the plot width. Positions are pixels in either order and are clamped to [0, size], so a
 * box running past the axis stops at the plot edge. The accent is read at render, so it follows
 * the dark theme.
 */
export function RefBand({ orient, at, size, length, opacity = 0.06 }: {
  orient: 'v' | 'h'; at: [number, number]
  /** the plot's extent along the band's axis (iw for "v", ih for "h") */
  size: number
  /** the band's extent across (ih for "v", iw for "h") */
  length: number
  opacity?: number
}) {
  const lo = Math.max(0, Math.min(size, Math.min(at[0], at[1]))), hi = Math.max(0, Math.min(size, Math.max(at[0], at[1])))
  if (!(hi > lo)) return null
  const box = orient === 'v' ? { x: lo, y: 0, width: hi - lo, height: length } : { x: 0, y: lo, width: length, height: hi - lo }
  return <rect {...box} fill={INK.accent} fillOpacity={opacity} pointerEvents="none" data-guardrail="" />
}
