import { useMemo } from 'react'
import type { ScaleLinear, ScaleTime, ScaleBand } from 'd3'
import { INK, FONT } from '@/lib/theme'

type AnyScale =
  | ScaleLinear<number, number>
  | ScaleTime<number, number>
  | ScaleBand<string>

function ticksOf(scale: AnyScale, count: number): { v: any; pos: number }[] {
  if ('bandwidth' in scale) {
    const s = scale as ScaleBand<string>
    return s.domain().map((d) => ({ v: d, pos: (s(d) ?? 0) + s.bandwidth() / 2 }))
  }
  const s = scale as ScaleLinear<number, number>
  return s.ticks(count).map((t) => ({ v: t, pos: s(t) }))
}

export function XAxis({
  scale,
  y,
  label,
  tickCount = 6,
  format,
  rotate = 0,
  labelDy,
}: {
  scale: AnyScale
  y: number
  label?: string
  tickCount?: number
  format?: (v: any) => string
  rotate?: number
  /** y (relative to the axis) for the title; defaults below the ticks */
  labelDy?: number
}) {
  const ticks = useMemo(() => ticksOf(scale, tickCount), [scale, tickCount])
  const range = scale.range() as [number, number]
  const x0 = Math.min(...range)
  const x1 = Math.max(...range)
  return (
    <g transform={`translate(0,${y})`} fontFamily={FONT.family}>
      <line x1={x0} x2={x1} stroke={INK.axis} strokeWidth={1} />
      {ticks.map(({ v, pos }, i) => (
        <g key={i} transform={`translate(${pos},0)`}>
          <line y2={5} stroke={INK.axis} />
          <text
            y={rotate ? 10 : 18}
            textAnchor={rotate ? 'end' : 'middle'}
            transform={rotate ? `rotate(${rotate})` : undefined}
            fontSize={11}
            fill={INK.muted}
          >
            {format ? format(v) : String(v)}
          </text>
        </g>
      ))}
      {label && (
        <text
          x={(x0 + x1) / 2}
          y={labelDy ?? (rotate ? 52 : 40)}
          textAnchor="middle"
          fontSize={12}
          fontWeight={600}
          fill={INK.text}
        >
          {label}
        </text>
      )}
    </g>
  )
}

export function YAxis({
  scale,
  x,
  label,
  tickCount = 6,
  format,
  gridWidth,
  gridFrom,
  labelX,
}: {
  scale: AnyScale
  x: number
  label?: string
  tickCount?: number
  format?: (v: any) => string
  gridWidth?: number
  /** where grid lines start; defaults to the axis position. Set when the axis is drawn at x=0 inside the plot. */
  gridFrom?: number
  /** where to put the rotated axis title; defaults to 50px left of the axis */
  labelX?: number
}) {
  const ticks = useMemo(() => ticksOf(scale, tickCount), [scale, tickCount])
  const range = scale.range() as [number, number]
  const y0 = Math.min(...range)
  const y1 = Math.max(...range)
  return (
    <g fontFamily={FONT.family}>
      <line x1={x} x2={x} y1={y0} y2={y1} stroke={INK.axis} strokeWidth={1} />
      {ticks.map(({ v, pos }, i) => (
        <g key={i} transform={`translate(0,${pos})`}>
          {gridWidth ? (
            <line x1={gridFrom ?? x} x2={(gridFrom ?? x) + gridWidth} stroke={INK.grid} strokeWidth={1} />
          ) : null}
          <line x1={x - 5} x2={x} stroke={INK.axis} />
          <text x={x - 9} dy="0.32em" textAnchor="end" fontSize={11} fill={INK.muted}>
            {format ? format(v) : String(v)}
          </text>
        </g>
      ))}
      {label && (
        <text
          transform={`translate(${labelX ?? x - 50},${(y0 + y1) / 2}) rotate(-90)`}
          textAnchor="middle"
          fontSize={12}
          fontWeight={600}
          fill={INK.text}
        >
          {label}
        </text>
      )}
    </g>
  )
}
