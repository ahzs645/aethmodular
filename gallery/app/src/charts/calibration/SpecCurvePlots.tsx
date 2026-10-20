import { useMemo, useState } from 'react'
import { robustDomain } from '@/lib/axes'
import * as d3 from 'd3'
import { XAxis, YAxis } from '@/components/Axes'
import { FONT, INK } from '@/lib/theme'
import { PREPROCESSING, label } from '@/lib/labels'
import { selectionPanels } from '@/lib/specification'
import type { SpecCurveFile, SpecMetricInfo } from './SpecCurve'

export type SpecLayout = 'combined' | 'stacked' | 'overlay'
const TOP_H = 250, ROW_H = 17, GROUP_H = 18, GAP = 24
const COLOURS = ['#2171b5', '#0f766e', '#8856a7', '#d95f0e', '#636363']

interface Props {
  file: SpecCurveFile & { value: number[] }; metric: SpecMetricInfo; shown: number[]; width: number; id: string; layout: SpecLayout
  highlighted: boolean[]; showDots: boolean; highlightLabel: string
  required: Set<number>; toggleChoice: (c: number) => void
  selectedIndices: Set<number>; describe: (i: number) => string[]
  showTip: (e: React.MouseEvent, lines: string[]) => void; hideTip: () => void
  select?: (i: number) => void
  /** strip colour for the share (0–1) of configurations using a choice (the chart's palette) */
  heat: (t: number) => string
  /**
   * what the panels split by: the selection spectra (where a search selects and models on the same
   * baseline, that baseline) or the model spectra (a search that crosses them, e.g. raw models of
   * spline-selected cohorts). Default 'sel'.
   */
  panelBy?: 'sel' | 'sp'
  /** the curve alone, without the choice strips (the guardrail editor's live preview), at this height */
  curveOnly?: { height: number }
}

export function SpecCurvePlots(props: Props) {
  const { file, metric, shown, width, layout, id, highlighted, showDots, highlightLabel } = props
  const by = props.panelBy ?? 'sel'
  const panels = useMemo(() => selectionPanels(shown, file[by], file.categories[by]), [shown, file, by])
  const colorOf = (key: string) => COLOURS[file.categories[by].indexOf(key) % COLOURS.length]
  const selectionName = (key: string) => by === 'sp' ? `${label(PREPROCESSING, key)} models` : `${label(PREPROCESSING, key)} selection`
  // The baseline is already identified by the panel and selection controls.
  // Keep original choice indices so clicking the remaining rows still filters
  // the matching saved masks, rather than the removed preprocessing masks.
  const groups = (file.groups ?? []).filter(g => g.title !== 'Spectral preprocessing')
  const choiceIndices = file.choices.map((_, c) => c).filter(c => !(file.groups ?? []).some(g => g.title === 'Spectral preprocessing' && c >= g.start && c < g.end))
  const topH = props.curveOnly?.height ?? TOP_H
  // the curve alone needs room for the y axis only, and sets its panels side by side where they fit
  const labelPad = props.curveOnly ? 64 : Math.ceil(Math.max(0, ...choiceIndices.map(c => file.choices[c].length * 6.3))) + 22
  const space = Math.max(240, width - labelPad - 20)
  const panelCount = layout === 'combined' ? 1 : panels.length
  const columns = props.curveOnly ? Math.max(1, Math.min(panelCount, Math.floor((space + 36) / 330))) : 1
  const plotW = (space - (columns - 1) * 36) / columns
  const rowY = (c: number) => choiceIndices.indexOf(c) * ROW_H + (groups.length ? (groups.findIndex(g => c >= g.start && c < g.end) + 1) * GROUP_H : 0)
  const stripH = choiceIndices.length * ROW_H + groups.length * GROUP_H
  const panelH = props.curveOnly ? 48 + topH + 46 : 48 + topH + GAP + stripH + 60
  const ys = shown.map(i => file.value[i])
  const [qLo, qHi] = robustDomain(ys, 0.003, 0.997, { fallback: metric.band }).domain
  const pad = 0.08 * Math.max(qHi - qLo, metric.band[1] - metric.band[0])
  // the intercept keeps its long-standing frame; other measures pad around the data and the target band
  const domain: [number, number] = metric.ref === 0 && metric.band[0] === -0.5 ? [qLo - 0.6, Math.max(2.6, qHi + 1.9)]
    : [Math.min(qLo, metric.band[0]) - pad, Math.max(qHi, metric.band[1]) + pad]
  const y = d3.scaleLinear().domain(domain).range([topH, 0])
  const panelList = layout === 'combined' ? [{ key: 'combined', indices: shown }] : panels
  const height = layout === 'overlay' ? topH + 150 : Math.ceil(panelList.length / columns) * panelH + (props.curveOnly ? 0 : 14)
  return <svg width={Math.max(width, labelPad + space + 20)} height={height} fontFamily={FONT.family} data-layout={layout}>
    {layout === 'overlay' ? <g transform="translate(74,46)">
      <YAxis scale={y} x={0} label={metric.axis} gridWidth={Math.max(240, width - 100)} tickCount={6} pick="y" />
      <rect x={0} y={y(metric.band[1])} width={Math.max(240, width - 100)} height={y(metric.band[0]) - y(metric.band[1])} fill={INK.good} fillOpacity={0.08} />
      <line x1={0} x2={Math.max(240, width - 100)} y1={y(metric.ref)} y2={y(metric.ref)} stroke={INK.text} />
      {panels.map(p => {
        const w = Math.max(240, width - 100), n = p.indices.length
        const x = d3.scaleLinear().domain([0, 100]).range([0, w])
        const rank = (j: number) => n === 1 ? 50 : j / (n - 1) * 100
        const samples = p.indices.map((i, j) => ({ i, j })).filter((_, j) => j % Math.max(1, Math.floor(n / (w * 2))) === 0 || j === n - 1)
        const line = d3.line<typeof samples[number]>().x(d => x(rank(d.j))).y(d => y(Math.max(domain[0], file.value[d.i])))
        return <g key={p.key} data-selection={p.key}>
          <path d={line(samples) ?? ''} fill="none" stroke={colorOf(p.key)} strokeWidth={1.8} />
          {showDots && p.indices.map((i, j) => highlighted[i] && <circle key={i} data-highlight="true" cx={x(rank(j))} cy={y(Math.max(domain[0], file.value[i]))} r={2.3} fill={INK.negative} />)}
        </g>
      })}
      <XAxis scale={d3.scaleLinear().domain([0, 100]).range([0, Math.max(240, width - 100)])} y={topH} label={`Percentile within each selection, ${metric.rank.replace('Configuration rank by', 'ordered by')}`} tickCount={6} format={v => `${v}%`} />
      <rect x={0} y={0} width={Math.max(240, width - 100)} height={topH} fill="transparent"
        onMouseMove={e => {
          const r = e.currentTarget.getBoundingClientRect(), t = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width))
          props.showTip(e, [`${(100 * t).toFixed(1)}th percentile within each selection`, ...panels.flatMap(p => {
            const i = p.indices[Math.round(t * (p.indices.length - 1))]
            return [selectionName(p.key), ...props.describe(i)]
          }), 'Percentiles compare distributions; configurations are not paired.'])
        }} onMouseLeave={props.hideTip} />
      {panels.map((p, k) => <g key={p.key} transform={`translate(${(k % 2) * Math.max(240, (width - 100) / 2)},${topH + 70 + Math.floor(k / 2) * 18})`}>
        <line x1={0} x2={14} stroke={colorOf(p.key)} strokeWidth={2} />
        <text x={20} dy="0.32em" fontSize={10.5} fill={INK.text}>{selectionName(p.key).slice(0, Math.max(20, Math.floor((width - 100) / 11) - 20))}{selectionName(p.key).length > Math.max(20, Math.floor((width - 100) / 11) - 20) ? '…' : ''} · n={p.indices.length.toLocaleString()}<title>{selectionName(p.key)}</title></text>
      </g>)}
    </g> : panelList.map((p, k) => <SpecPanel key={p.key} {...props} indices={p.indices}
      name={p.key === 'combined' ? 'All selected cohorts combined' : selectionName(p.key)}
      color={p.key === 'combined' ? INK.accent : colorOf(p.key)} xOffset={labelPad + (k % columns) * (plotW + 36)}
      yOffset={Math.floor(k / columns) * panelH + 42} plotW={plotW} y={y} labelPad={labelPad} labels={k % columns === 0} topH={topH}
      rowY={rowY} stripH={stripH} choiceIndices={choiceIndices} groups={groups} clipId={`${id}-panel-${k}`} />)}
    {/* the editor's preview states its rule in words below the curve, so it drops this footnote */}
    {!props.curveOnly && <text x={20} y={height - 6} fontSize={10.5} fill={INK.muted}>
      {showDots ? `Red dots: ${highlightLabel}` : 'Red dots hidden'}{layout === 'stacked' && ' · shared vertical scale; each panel sorted independently'}
    </text>}
  </svg>
}

function SpecPanel(props: Props & {
  indices: number[]; name: string; color: string; xOffset: number; yOffset: number; plotW: number
  y: d3.ScaleLinear<number, number>; labelPad: number; labels: boolean; rowY: (c: number) => number; stripH: number; clipId: string; topH: number
  choiceIndices: number[]; groups: NonNullable<SpecCurveFile['groups']>
}) {
  const { file, indices, plotW, y, labelPad, labels, rowY, stripH, highlighted, showDots, topH } = props
  const curveOnly = !!props.curveOnly
  const [cursor, setCursor] = useState<number | null>(null)
  const n = indices.length, x = d3.scaleLinear().domain([0, Math.max(1, n)]).range([0, plotW])
  const yLo = y.domain()[0], clamp = (i: number) => y(Math.max(yLo, file.value[i]))
  const samples = indices.map((i, j) => ({ i, j })).filter((_, j) => j % Math.max(1, Math.floor(n / (plotW * 2))) === 0 || j === n - 1)
  const line = d3.line<typeof samples[number]>().x(p => x(p.j + 0.5)).y(p => clamp(p.i))
  const area = d3.area<typeof samples[number]>().x(p => x(p.j + 0.5)).y0(y(props.metric.ref)).y1(p => clamp(p.i))
  const strips = useMemo(() => {
    if (curveOnly) return [] as string[]
    const cols = Math.max(1, Math.min(indices.length, Math.round(plotW)))
    return file.masks.map(mask => {
      const canvas = document.createElement('canvas'); canvas.width = cols; canvas.height = 1
      const ctx = canvas.getContext('2d'); if (!ctx) return ''
      const img = ctx.createImageData(cols, 1)
      for (let b = 0; b < cols; b++) {
        const j0 = Math.floor(b * indices.length / cols), j1 = Math.max(j0 + 1, Math.floor((b + 1) * indices.length / cols))
        let used = 0; for (let j = j0; j < j1; j++) if (mask[indices[j]] === '1') used++
        const rgb = used === 0 ? d3.rgb(255,255,255) : d3.rgb(props.heat(used / (j1 - j0)))
        img.data.set([rgb.r, rgb.g, rgb.b, 255], b * 4)
      }
      ctx.putImageData(img,0,0); return canvas.toDataURL()
    })
  }, [file, indices, plotW, curveOnly, props.heat])
  const locate = (e: React.MouseEvent<SVGRectElement>) => Math.min(n - 1, Math.max(0, Math.floor(x.invert(e.clientX - e.currentTarget.getBoundingClientRect().left))))
  const count = indices.filter(i => highlighted[i]).length
  return <g transform={`translate(${props.xOffset},${props.yOffset})`} data-selection={props.name}>
    <text x={0} y={-22} fontSize={11.5} fontWeight={600} fill={props.color}>{props.name.length > plotW / 6.1 ? props.name.slice(0, Math.max(10, Math.floor(plotW / 6.1) - 1)) + '…' : props.name}<title>{props.name}</title></text>
    <text x={0} y={-6} fontSize={10.5} fill={INK.muted}>{n.toLocaleString()} configurations · {count.toLocaleString()} match</text>
    <defs><clipPath id={props.clipId}><rect width={plotW} height={topH} /></clipPath></defs>
    {labels ? <YAxis scale={y} x={0} label={props.metric.axis} gridWidth={plotW} tickCount={6} pick="y" /> : y.ticks(6).map(v => <line key={v} x1={0} x2={plotW} y1={y(v)} y2={y(v)} stroke={INK.grid} />)}
    <rect x={0} y={y(props.metric.band[1])} width={plotW} height={y(props.metric.band[0]) - y(props.metric.band[1])} fill={INK.good} fillOpacity={0.08} />
    <g clipPath={`url(#${props.clipId})`}>
      <path d={area(samples) ?? ''} fill={props.color} fillOpacity={0.12} />
      <path d={line(samples) ?? ''} fill="none" stroke={props.color} strokeWidth={1.4} />
      {indices.map((i, j) => props.selectedIndices.has(i) && <circle key={`s${i}`} cx={x(j + 0.5)} cy={clamp(i)} r={3.4} fill="none" stroke={INK.text} strokeOpacity={0.7} />)}
      {showDots && indices.map((i, j) => highlighted[i] && <circle key={`h${i}`} data-highlight="true" cx={x(j + 0.5)} cy={clamp(i)} r={2.3} fill={INK.negative} />)}
    </g>
    <line x1={0} x2={plotW} y1={y(props.metric.ref)} y2={y(props.metric.ref)} stroke={INK.text} />
    {cursor !== null && cursor < n && <line x1={x(cursor + 0.5)} x2={x(cursor + 0.5)} y1={0} y2={(curveOnly ? topH : topH + GAP + stripH)} stroke={INK.text} strokeOpacity={0.5} pointerEvents="none" />}
    <rect x={0} y={0} width={plotW} height={topH} fill="transparent" style={{ cursor: props.select ? 'pointer' : 'default' }}
      onMouseMove={e => { const j = locate(e); setCursor(j); props.showTip(e, [`#${(j + 1).toLocaleString()} of ${n.toLocaleString()} · ${props.name}`, ...props.describe(indices[j])]) }}
      onMouseLeave={() => { setCursor(null); props.hideTip() }} onClick={e => props.select?.(indices[locate(e)])} />
    {curveOnly ? <XAxis scale={x} y={topH} label={props.metric.rank} tickCount={Math.max(2, Math.floor(plotW / 85))} format={v => d3.format(',')(v)} /> : <g transform={`translate(0,${topH + GAP})`}>
      {props.choiceIndices.map(c => <image key={c} href={strips[c]} x={0} y={rowY(c)} width={plotW} height={ROW_H - 1.5} preserveAspectRatio="none" style={{ imageRendering: 'pixelated' }} />)}
      {labels && props.groups.map(g => <text key={g.title} x={-labelPad + 4} y={rowY(g.start) - 5} fontSize={11.5} fontWeight={700} fill={INK.text}>{g.title}</text>)}
      {labels && props.choiceIndices.map(c => <g key={c} style={{cursor:'pointer'}} role="button" tabIndex={0} aria-label={file.choices[c]} aria-pressed={props.required.has(c)}
        onClick={() => props.toggleChoice(c)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); props.toggleChoice(c) } }}>
        {props.required.has(c) && <rect x={-labelPad + 4} y={rowY(c) + 1} width={labelPad - 8} height={ROW_H - 2} rx={3} fill={INK.accent} fillOpacity={0.12} />}
        <text x={-8} y={rowY(c) + ROW_H / 2} dy="0.32em" textAnchor="end" fontSize={11} fill={props.required.has(c) ? INK.accent : INK.text}>{file.choices[c]}</text>
        <title>Click to toggle this choice. Alternatives within the same setting are combined; different settings must all match.</title>
      </g>)}
      <XAxis scale={x} y={stripH} label={props.metric.rank} tickCount={Math.max(2, Math.floor(plotW / 85))} format={v => d3.format(',')(v)} />
    </g>}
  </g>
}
