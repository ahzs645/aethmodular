import { estimatorText, useEstimator } from '@/lib/regressionPrefs'
import { chemText } from '@/lib/format'
import { useCallback, useId, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { AxisFieldPopover, useAxisFields, type AxisRole } from '@/components/AxisFieldPopover'
import type { ScaleLinear, ScaleTime, ScaleBand } from 'd3'
import { INK, FONT } from '@/lib/theme'
import { wrapCategory } from '@/components/CategoryAxis'
import { fieldHelp } from '@/lib/glossary'
import { useTooltip } from '@/hooks/useTooltip'

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

/** a tick label's width at the axis's 11 px (about 6.2 px a character) plus a gap */
const labelWidth = (text: string) => text.length * 6.2 + 12

/**
 * The scale's ticks, thinned until no two labels overlap and none runs off the axis: d3 is asked
 * for fewer ticks in turn, and if it still crowds, every second label is kept. Band axes and
 * rotated labels are left as they are.
 */
function fittedTicks(scale: AnyScale, count: number, format: ((v: any) => string) | undefined, rotate: number) {
  const all = ticksOf(scale, count)
  if ('bandwidth' in scale || rotate || all.length < 3) return all
  const text = (v: any) => (format ? format(v) : String(v))
  const fits = (t: { v: any; pos: number }[]) => t.every((a, i) => i === 0 || a.pos - t[i - 1].pos >= (labelWidth(text(a.v)) + labelWidth(text(t[i - 1].v))) / 2)
  let t = all, n = count
  while (!fits(t) && n > 2) {
    n = Math.min(n - 1, t.length - 1)
    t = ticksOf(scale, n)
  }
  while (!fits(t) && t.length > 2) t = t.filter((_, i) => i % 2 === 0)
  return t
}

export function XAxis({
  scale,
  y,
  label,
  tickCount = 6,
  format,
  rotate = 0,
  labelDy,
  wrapLabel = false,
  pick,
}: {
  scale: AnyScale
  y: number
  label?: string
  wrapLabel?: boolean
  /** the shared field this axis shows; its title then opens a field picker (AxisFieldPopover) */
  pick?: AxisRole
  tickCount?: number
  format?: (v: any) => string
  rotate?: number
  /** y (relative to the axis) for the title; defaults below the ticks */
  labelDy?: number
}) {
  const method = useEstimator()
  // as many ticks as fit: fewer when the labels (dates above all, 'Feb 2023') would touch at this width
  const ticks = useMemo(() => fittedTicks(scale, tickCount, format, rotate), [scale, tickCount, format, rotate])
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
        <AxisTitle pick={pick}>
          <text
            x={(x0 + x1) / 2}
            y={labelDy ?? (rotate ? 52 : 40)}
            textAnchor="middle"
            fontSize={12}
            fontWeight={600}
            fill={INK.text}
          >
            {wrapLabel ? wrapCategory(chemText(estimatorText(label, method)), Math.max(100, x1 - x0 - 8)).map((line, i) => <tspan key={i} x={(x0 + x1) / 2} dy={i ? 14 : 0}>{line}</tspan>) : chemText(estimatorText(label, method))}{pick && <tspan className="axis-pick-caret" dx={4} data-export="skip">▾</tspan>}
          </text>
        </AxisTitle>
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
  wrapLabel = false,
  pick,
}: {
  scale: AnyScale
  x: number
  label?: string
  wrapLabel?: boolean
  /** the shared field this axis shows; its title then opens a field picker (AxisFieldPopover) */
  pick?: AxisRole
  tickCount?: number
  format?: (v: any) => string
  gridWidth?: number
  /** where grid lines start; defaults to the axis position. Set when the axis is drawn at x=0 inside the plot. */
  gridFrom?: number
  /** where to put the rotated axis title; defaults to just past the widest tick label (at most 50px left of the axis) */
  labelX?: number
}) {
  const method = useEstimator()
  const ticks = useMemo(() => ticksOf(scale, tickCount), [scale, tickCount])
  const range = scale.range() as [number, number]
  const y0 = Math.min(...range)
  const y1 = Math.max(...range)
  const titleLines = wrapLabel && label ? wrapCategory(chemText(estimatorText(label, method)), Math.max(100, y1 - y0 - 8)) : []
  // the title sits the same distance from the tick labels as the x axis's title does from its own (about 13 px):
  // measured as drawn, so short labels ("0–30") do not leave it stranded and the readable text size is counted
  const tickBox = useRef<SVGGElement>(null)
  const [tickW, setTickW] = useState<number | null>(null)
  // measured again only when the tick text changes: getBBox forces a layout, and a chart re-renders on every hover
  const tickText = ticks.map(({ v }) => (format ? format(v) : String(v))).join('|')
  useLayoutEffect(() => {
    let w = 0
    try { for (const t of tickBox.current?.querySelectorAll('text') ?? []) w = Math.max(w, t.getBBox().width) } catch { /* not drawn (a hidden chart) */ }
    if (w > 0 && (tickW === null || Math.abs(w - tickW) > 0.5)) setTickW(w)
  }, [tickText]) // eslint-disable-line react-hooks/exhaustive-deps
  const spread = titleLines.length > 1 ? ((titleLines.length - 1) / 2) * 14 : 0
  // never further out than the 50 px the charts' left margins allow for
  const titleX = labelX ?? (tickW === null ? x - 50 : Math.max(x - 50, x - 9 - tickW - 15 - spread))
  return (
    <g fontFamily={FONT.family}>
      <line x1={x} x2={x} y1={y0} y2={y1} stroke={INK.axis} strokeWidth={1} />
      <g ref={tickBox}>
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
      </g>
      {label && (
        <AxisTitle pick={pick}>
          <text
            transform={`translate(${titleX},${(y0 + y1) / 2}) rotate(-90)`}
            textAnchor="middle"
            fontSize={12}
            fontWeight={600}
            fill={INK.text}
          >
            {wrapLabel ? titleLines.map((line, i) => <tspan key={i} x={0} y={(i - (titleLines.length - 1) / 2) * 14}>{line}</tspan>) : chemText(estimatorText(label, method))}{pick && <tspan className="axis-pick-caret" dx={4} data-export="skip">▾</tspan>}
          </text>
        </AxisTitle>
      )}
    </g>
  )
}

/** An axis title that, when the axis shows a shared field, opens the field picker on click or Enter. */
export function AxisTitle({ pick, children }: { pick?: AxisRole; children: ReactNode }) {
  const ctx = useAxisFields()
  const titleRef = useRef<SVGGElement>(null)
  const tipId = useId()
  const tip = useTooltip(titleRef, { id: tipId })
  const [anchor, setAnchor] = useState<Element | null>(null)
  const close = useCallback((restoreFocus = true) => {
    setAnchor(null)
    if (restoreFocus) (anchor as HTMLElement | SVGElement | null)?.focus({ preventScroll: true })
  }, [anchor])
  if (!pick || !ctx) return <>{children}</>
  const field = ctx.value[pick]
  const help = [
    ...(field === 'HIPS BC'
      ? [`MAC used: ${ctx.meta.mac_value} m²/g`]
      : chemText(fieldHelp(field, ctx.labelOf?.(field) ?? field)).split('\n')),
    'Click to choose what this axis shows',
  ]
  const open = (el: Element) => { tip.hide(); setAnchor(el) }
  return (
    <>
      <g ref={titleRef} className="axis-pick" role="button" tabIndex={0} aria-haspopup="dialog" aria-expanded={!!anchor} aria-label={`Change the ${pick === 'field' ? 'measurement' : `${pick} axis`}: ${field}`} aria-describedby={tip.active ? tipId : undefined}
        onMouseEnter={e => { if (!anchor) tip.show(e, help) }} onMouseLeave={tip.hide}
        onFocus={e => { if (!anchor && e.currentTarget.matches(':focus-visible')) tip.show(e, help) }} onBlur={tip.hide}
        onClick={(e) => open(e.currentTarget)} onKeyDown={(e) => {
          if (e.key === 'Escape') tip.hide()
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(e.currentTarget) }
        }}>
        {children}
      </g>
      {tip.node}
      {anchor && <AxisFieldPopover role={pick} anchor={anchor} onClose={close} />}
    </>
  )
}
