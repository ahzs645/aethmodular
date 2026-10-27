import { FONT, INK } from '@/lib/theme'

/**
 * The three kinds of reference line every chart draws, one style each
 * (docs/gallery-chart-standards.md §1). Positions are in pixels, so any scale
 * (linear, log, band) works:
 *
 *   <IdentityLine x1={x(d0)} y1={y(d0)} x2={x(d1)} y2={y(d1)} />    1:1 on a crossplot
 *   <RefLine orient="v" at={x(1)} from={0} to={ih} />               agreement: slope 1, intercept 0, zero
 *   <Marker orient="h" at={y(0.85)} from={0} to={iw} label="0.85" />  a data value to compare against
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
