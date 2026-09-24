import { useId, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { hexbin as d3Hexbin } from 'd3-hexbin'
import { ChartFrame, Empty, Note, Select } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { inDomain, pairDomain, type AxesOpts } from '@/lib/axes'
import { regression, fmt } from '@/lib/stats'
import { INK, MARGIN, RAMP_SEQUENTIAL, FONT, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

/**
 * 2D density (hexbin) — 16 figures in the estate, all of them on the large
 * IMPROVE pool where a plain scatter overplots into a solid blob. Same idea
 * here: at 900+ filters the dense core of a crossplot is unreadable as points.
 * Reads the same x/y pair and the same axes policy as the scatterplot above,
 * so the two frames line up. (Log axes are ignored here: hex bins on a log
 * scale bin the pixels, not the data.)
 */
export function DensityHexbin({ rows, meta, xField, yField, axes }: { rows: FilterRow[]; meta: MetaFile; xField: string; yField: string; axes: AxesOpts }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const clipId = useId()
  const MAX_SQUARE = 520

  const [radius, setRadius] = useState('14')
  const opts: AxesOpts = { ...axes, log: false }

  const points = useMemo(
    () =>
      rows
        .map((r) => [r[xField], r[yField]] as [unknown, unknown])
        .filter(
          (p): p is [number, number] =>
            typeof p[0] === 'number' && typeof p[1] === 'number' && Number.isFinite(p[0]) && Number.isFinite(p[1])
        ),
    [rows, xField, yField]
  )

  const availW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerW = Math.min(availW, MAX_SQUARE)
  const innerH = innerW
  const svgHeight = innerH + MARGIN.top + MARGIN.bottom

  const { x, y, stats, domain } = useMemo(() => {
    const xs = points.map((p) => p[0])
    const ys = points.map((p) => p[1])
    // both axes are measurements with their own error: the fit shown is Deming only
    const stats = regression(xs, ys, { errorsInVariables: true })
    const intercepts = stats && stats.demingIntercept !== null ? [stats.demingIntercept] : []
    const domain = pairDomain(xs, ys, opts, intercepts)
    return {
      x: d3.scaleLinear().domain(domain.x).range([0, innerW]).nice(),
      y: d3.scaleLinear().domain(domain.y).range([innerH, 0]).nice(),
      stats,
      domain,
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, innerW, innerH, opts.identity, opts.mode])

  const bins = useMemo(() => {
    const hb = d3Hexbin<[number, number]>()
      .x((d) => x(d[0]))
      .y((d) => y(d[1]))
      .radius(Number(radius))
      .extent([[0, 0], [innerW, innerH]])
    return hb(points.filter((p) => inDomain(domain, p[0], p[1]))).map((b) => Object.assign(b, { path: hb.hexagon() }))
  }, [points, x, y, radius, innerW, innerH, domain])

  const maxCount = d3.max(bins, (b) => b.length) ?? 1
  const color = d3.scaleQuantize<string>().domain([0, maxCount]).range(RAMP_SEQUENTIAL)
  const [d0, d1] = x.domain() as [number, number]
  const originAxes = opts.mode === 'Show intercept' && d0 < 0 && (y.domain()[0] as number) < 0

  return (
    <ChartFrame
      id="hexbin"
      title="2D density (hexbin) — where the crossplot is actually dense"
      subtitle="The same pair and the same axes as the scatterplot, binned. Overplotting hides that most filters sit in a small low-loading core; the hexbin shows the mass of the data rather than its outline."
      provenance="stands in for 16 hexbin+plot+scatter figures · react-graph-gallery.com/2d-density-plot"
      controls={
        <>
          <Select label="hex radius" value={radius} options={['8', '11', '14', '18', '24']} onChange={setRadius} />
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap centered">
        {points.length < 3 ? (
          <Empty>Fewer than 3 filters carry both fields in this subset.</Empty>
        ) : (
          <svg width={innerW + MARGIN.left + MARGIN.right} height={svgHeight} className="animated">
            <defs>
              <clipPath id={clipId}>
                <rect x={0} y={0} width={innerW} height={innerH} />
              </clipPath>
            </defs>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {originAxes ? (
                <>
                  <YAxis scale={y} x={x(0)} label={withUnit(yField, meta.field_units)} gridWidth={innerW} gridFrom={0} labelX={-50} />
                  <XAxis scale={x} y={y(0)} label={withUnit(xField, meta.field_units)} labelDy={innerH - y(0) + 40} />
                </>
              ) : (
                <>
                  <YAxis scale={y} x={0} label={withUnit(yField, meta.field_units)} gridWidth={innerW} />
                  <XAxis scale={x} y={innerH} label={withUnit(xField, meta.field_units)} />
                </>
              )}

              {bins.map((b, i) => (
                <path
                  key={i}
                  d={b.path}
                  transform={`translate(${b.x},${b.y})`}
                  fill={color(b.length)}
                  stroke="#fff"
                  strokeWidth={0.4}
                  onMouseEnter={(e) =>
                    tip.show(e, [`${b.length} filters`, `${xField} ≈ ${fmt(x.invert(b.x), 2)}`, `${yField} ≈ ${fmt(y.invert(b.y), 2)}`])
                  }
                  onMouseLeave={tip.hide}
                />
              ))}

              <g clipPath={`url(#${clipId})`} pointerEvents="none">
                {opts.identity && <line x1={x(d0)} y1={y(d0)} x2={x(d1)} y2={y(d1)} stroke={INK.identity} strokeWidth={1.5} strokeDasharray="5 4" />}
                {stats && (
                  <>
                    {stats.demingSlope !== null && (
                      <line
                        x1={x(d0)} y1={y(stats.demingSlope * d0 + stats.demingIntercept!)}
                        x2={x(d1)} y2={y(stats.demingSlope * d1 + stats.demingIntercept!)}
                        stroke={INK.deming} strokeWidth={2} strokeDasharray="7 3"
                      />
                    )}
                  </>
                )}
              </g>
              {stats && opts.mode === 'Show intercept' && (
                <g pointerEvents="none" fontFamily={FONT.mono} fontSize={10.5}>
                  {stats.demingIntercept !== null && (
                    <>
                      <circle cx={x(0)} cy={y(stats.demingIntercept)} r={4.5} fill="#fff" stroke={INK.deming} strokeWidth={2} />
                      <text x={x(0) + 8} y={y(stats.demingIntercept)} dy="0.35em" fill={INK.deming}>b = {fmt(stats.demingIntercept, 3)}</text>
                    </>
                  )}
                </g>
              )}

              {stats && (
                <g transform="translate(10,10)" fontFamily={FONT.mono} fontSize={11} pointerEvents="none">
                  <rect width={214} height={62} rx={5} fill="#fff" fillOpacity={0.93} stroke={INK.border} />
                  <text x={9} y={18} fill={INK.text}>n = {stats.n} in {bins.length} hexes</text>
                  <text x={9} y={33} fill={INK.text}>R² = {fmt(stats.r2, 4)}</text>
                  {stats.demingSlope !== null && stats.demingIntercept !== null && (
                    <text x={9} y={48} fill={INK.deming}>Deming y = {fmt(stats.demingSlope)}x {stats.demingIntercept < 0 ? '−' : '+'} {fmt(Math.abs(stats.demingIntercept))}</text>
                  )}
                </g>
              )}
            </g>
          </svg>
        )}
        <div className="legend">
          <ColorLegend scale={color} label="filters per hex" format={(v) => String(Math.round(v))} />
        </div>
        {domain.dropped > 0 && <Note>{domain.dropped} points below zero are outside the axes and not binned; they are still in the fit.</Note>}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
