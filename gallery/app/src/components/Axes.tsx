import { estimatorText, useEstimator } from '@/lib/regressionPrefs'
import { chemText } from '@/lib/format'
import { useCallback, useId, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { AxisFieldPopover, useAxisFields, type AxisRole } from '@/components/AxisFieldPopover'
import type { ScaleLinear, ScaleTime, ScaleBand } from 'd3'
import { INK, FONT } from '@/lib/theme'
import { wrapCategory } from '@/components/CategoryAxis'
import { fieldHelp } from '@/lib/glossary'
import { useTooltip } from '@/hooks/useTooltip'
import { useSearchParam } from '@/lib/url'
import { textWidth } from '@/lib/fitLabel'

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

/** a tick label's width as drawn at the axis's 11 px, plus a gap */
const labelWidth = (text: string) => textWidth(text, 11) + 12

/**
 * The scale's ticks, thinned until no two labels overlap and none runs off the axis: d3 is asked
 * for fewer ticks in turn, and if it still crowds, every second label is kept. Band axes and
 * rotated labels are left as they are.
 */
function fittedTicks(scale: AnyScale, count: number, format: ((v: any) => string) | undefined, rotate: number) {
  let all = ticksOf(scale, count)
  // a log axis over two or more decades labels the decades only (0.1, 1, 10), not every minor tick between
  if ('base' in scale && typeof scale.base === 'function') {
    const b = (scale as unknown as { base(): number }).base()
    const decades = all.filter(({ v }) => { const e = Math.log(Math.abs(v)) / Math.log(b); return Math.abs(e - Math.round(e)) < 1e-9 })
    if (decades.length >= 3) all = decades
  }
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
  gridHeight,
  gridFrom,
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
  /** vertical grid lines at the ticks, this tall (the plot height); mirrors YAxis gridWidth. Draw the axis before the marks so the grid sits under them. */
  gridHeight?: number
  /** where grid lines start (their top, in the parent's y); defaults to gridHeight above the axis */
  gridFrom?: number
}) {
  const method = useEstimator()
  // as many ticks as fit: fewer when the labels (dates above all, 'Feb 2023') would touch at this width
  const ticks = useMemo(() => fittedTicks(scale, tickCount, format, rotate), [scale, tickCount, format, rotate])
  const range = scale.range() as [number, number]
  const x0 = Math.min(...range)
  const x1 = Math.max(...range)
  return (
    <g transform={`translate(0,${y})`} fontFamily={FONT.family}>
      {gridHeight ? ticks.map(({ pos }, i) => (
        <line key={`g${i}`} x1={pos} x2={pos} y1={(gridFrom ?? y - gridHeight) - y} y2={(gridFrom ?? y - gridHeight) - y + gridHeight} stroke={INK.grid} strokeWidth={1} />
      )) : null}
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
            {wrapLabel ? wrapCategory(chemText(estimatorText(label, method)), Math.max(100, x1 - x0 - 8), 12, 600).map((line, i) => <tspan key={i} x={(x0 + x1) / 2} dy={i ? 14 : 0}>{line}</tspan>) : chemText(estimatorText(label, method))}{pick && <tspan className="axis-pick-caret" dx={4} data-export="skip">▾</tspan>}
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
  const titleLines = wrapLabel && label ? wrapCategory(chemText(estimatorText(label, method)), Math.max(100, y1 - y0 - 8), 12, 600) : []
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

/** A curved two-way arrow in the axes' corner; uses the field pickers' shared swap behavior. */
export function AxisSwap({ x, y, onSwap }: { x: number; y: number; onSwap?: () => void }) {
  const ctx = useAxisFields()
  const ref = useRef<SVGGElement>(null)
  const tipId = useId()
  const tip = useTooltip(ref, { id: tipId })
  if (!onSwap && (!ctx || !ctx.value.x || !ctx.value.y || ctx.value.x === ctx.value.y
    || (ctx.rolesOf && (!ctx.rolesOf(ctx.value.y).includes('x') || !ctx.rolesOf(ctx.value.x).includes('y'))))) return null
  const label = 'Swap x and y axes'
  const swap = () => { tip.hide(); if (onSwap) onSwap(); else if (ctx) ctx.set('x', ctx.value.y) }
  return <>
    <g ref={ref} className="axis-swap" transform={`translate(${x},${y})`} data-export="skip"
      role="button" tabIndex={0} aria-label={label} aria-describedby={tip.active ? tipId : undefined}
      onMouseEnter={e => tip.show(e, [label])} onMouseLeave={tip.hide}
      onFocus={e => { if (e.currentTarget.matches(':focus-visible')) tip.show(e, [label]) }} onBlur={tip.hide}
      onClick={swap} onKeyDown={e => {
        if (e.key === 'Escape') tip.hide()
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); swap() }
      }}>
      <rect x={-16} y={-16} width={32} height={32} fill="transparent" />
      <path d="M-9-10v9A10 10 0 0 0 1 9h9M-13-6l4-4 4 4M6 5l4 4-4 4" transform="scale(0.65)" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" pointerEvents="none" />
    </g>
    {tip.node}
  </>
}

/** Editable pairs use their shared fields; fixed comparisons keep their orientation per plot/tile. */
export function useAxisSwap(key: string, useFields = false) {
  const ctx = useAxisFields()
  const [setting, setSetting] = useSearchParam<string>(`axes.swap.${key}`, '0', ['0', '1'])
  const shared = useFields && ctx && ctx.value.x && ctx.value.y && ctx.value.x !== ctx.value.y
    && (!ctx.rolesOf || (ctx.rolesOf(ctx.value.y).includes('x') && ctx.rolesOf(ctx.value.x).includes('y')))
  return {
    swapped: !shared && setting === '1',
    swap: () => { if (shared) ctx.set('x', ctx.value.y); else setSetting(setting === '1' ? '0' : '1') },
  }
}

/** Transpose saved geometry with its confidence bands; keep embedded text readable. */
export function OrientedPlotLayer({ swapped, width, height, children }: { swapped: boolean; width: number; height: number; children: ReactNode }) {
  const ref = useRef<SVGGElement>(null)
  useLayoutEffect(() => {
    const group = ref.current, matrix = group?.getCTM()
    if (!swapped || !group || !matrix) return
    // Work on the rendered text, including labels inside components such as Marker. Counter only
    // the reflection, at each label's own anchor; explicit SVG transforms survive every export format.
    const undo: (() => void)[] = []
    for (const text of group.querySelectorAll('text')) {
      const ctm = text.getCTM()
      if (!ctm) continue
      const relative = matrix.inverse().multiply(ctm)
      const linear = new DOMMatrix([relative.a, relative.b, relative.c, relative.d, 0, 0])
      const counter = linear.inverse().multiply(new DOMMatrix([0, -1, -1, 0, 0, 0])).multiply(linear)
      if (![counter.a, counter.b, counter.c, counter.d].every(Number.isFinite)) continue
      const x = text.x.baseVal.numberOfItems ? text.x.baseVal.getItem(0).value : 0
      const y = text.y.baseVal.numberOfItems ? text.y.baseVal.getItem(0).value : 0
      const original = text.getAttribute('transform')
      const anchored = new DOMMatrix().translate(x, y).multiply(counter).translate(-x, -y)
      const projected = new DOMMatrix([0, -1, -1, 0, width, height]).multiply(relative).multiply(anchored)
      const box = text.getBBox()
      const corners = [[box.x, box.y], [box.x + box.width, box.y], [box.x, box.y + box.height], [box.x + box.width, box.y + box.height]]
        .map(([px, py]) => new DOMPoint(px, py).matrixTransform(projected))
      const left = Math.min(...corners.map(p => p.x)), right = Math.max(...corners.map(p => p.x))
      const top = Math.min(...corners.map(p => p.y)), bottom = Math.max(...corners.map(p => p.y))
      // A label that used to sit at the top now sits at the right edge. Keep its upright text inside the plot.
      const dx = left < 0 ? -left : right > width ? width - right : 0
      const dy = top < 0 ? -top : bottom > height ? height - bottom : 0
      const shift = new DOMPoint(dx, dy).matrixTransform(new DOMMatrix([projected.a, projected.b, projected.c, projected.d, 0, 0]).inverse())
      const applied = `${original ?? ''} translate(${x},${y}) matrix(${counter.a} ${counter.b} ${counter.c} ${counter.d} 0 0) translate(${-x},${-y}) translate(${shift.x},${shift.y})`
      text.setAttribute('transform', applied)
      undo.push(() => { if (text.getAttribute('transform') === applied) {
        if (original === null) text.removeAttribute('transform'); else text.setAttribute('transform', original)
      } })
    }
    return () => undo.forEach(restore => restore())
  }, [swapped, width, height, children])
  return <g ref={ref} transform={swapped ? `matrix(0 -1 -1 0 ${width} ${height})` : undefined}>{children}</g>
}
