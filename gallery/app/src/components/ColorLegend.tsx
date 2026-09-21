import * as d3 from 'd3'
import { INK, FONT } from '@/lib/theme'

/**
 * Continuous colour-ramp legend — react-graph-gallery's "color legend" for a
 * continuous scale (heatmap / choropleth / hexbin pages). One component so
 * every chart that colours by a number draws the same ramp with real tick
 * values, instead of a gradient div with the two end labels typed by hand.
 *
 * Accepts any d3 scale that maps a number to a colour (sequential, linear
 * with a colour range, or quantize); the ramp is sampled from the scale
 * itself, so the legend cannot disagree with the marks.
 */
export function ColorLegend({
  scale,
  label,
  width = 180,
  format,
  note,
  ticks = 4,
}: {
  scale: { domain(): number[]; (v: number): string }
  label?: string
  width?: number
  format?: (v: number) => string
  /** trailing caption, e.g. "white = 0" */
  note?: string
  ticks?: number
}) {
  const dom = scale.domain()
  const lo = dom[0]
  const hi = dom[dom.length - 1]
  const fmtV = format ?? ((v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2)))
  const n = 40
  const stops = d3.range(n + 1).map((i) => scale(lo + ((hi - lo) * i) / n))
  const x = d3.scaleLinear().domain([lo, hi]).range([0, width])
  const tickVals = x.ticks(ticks).filter((t) => t >= lo && t <= hi)
  const h = 10
  const svgH = h + 20
  // room for the end tick labels, which are centred on the ramp's edges
  const PAD = 18
  return (
    <span className="legend-item" style={{ gap: 8 }}>
      {label && <span>{label}</span>}
      <svg width={width + 2 * PAD} height={svgH} fontFamily={FONT.family}>
        <g transform={`translate(${PAD},0)`}>
          {stops.map((c, i) => (
            <rect key={i} x={(i * width) / (n + 1)} y={0} width={width / (n + 1) + 0.5} height={h} fill={c} />
          ))}
          <rect x={0} y={0} width={width} height={h} fill="none" stroke={INK.border} />
          {tickVals.map((t) => (
            <g key={t} transform={`translate(${x(t)},${h})`}>
              <line y2={3} stroke={INK.axis} />
              <text y={13} textAnchor="middle" fontSize={10} fill={INK.muted}>{fmtV(t)}</text>
            </g>
          ))}
        </g>
      </svg>
      {note && <span className="legend-note">{note}</span>}
    </span>
  )
}

/**
 * Bubble-size legend — three reference circles for a sqrt radius scale, the
 * way react-graph-gallery's bubble-plot / bubble-map pages draw it. Laid out
 * side by side with the value under each, so the labels never overprint.
 */
export function SizeLegend({
  scale,
  label,
  format,
}: {
  scale: d3.ScalePower<number, number>
  label?: string
  format?: (v: number) => string
}) {
  const [, hi] = scale.domain() as [number, number]
  if (!Number.isFinite(hi) || hi <= 0) return null
  const fmtV = format ?? ((v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2)))
  const vals = [hi / 16, hi / 4, hi].filter((v) => v > 0)
  const rMax = scale(hi)
  const gap = 10
  let cx = 0
  const items = vals.map((v) => {
    const r = scale(v)
    cx += r
    const item = { v, r, cx }
    cx += r + gap + 12
    return item
  })
  const w = cx + 4
  const h = rMax * 2 + 18
  return (
    <span className="legend-item" style={{ gap: 8 }}>
      {label && <span>{label}</span>}
      <svg width={w} height={h} style={{ overflow: 'visible' }} fontFamily={FONT.family}>
        {items.map((it) => (
          <g key={it.v}>
            <circle cx={it.cx} cy={rMax + 1} r={it.r} fill="none" stroke={INK.axis} strokeWidth={1} />
            <text x={it.cx} y={rMax * 2 + 14} textAnchor="middle" fontSize={10} fill={INK.muted}>{fmtV(it.v)}</text>
          </g>
        ))}
      </svg>
    </span>
  )
}
