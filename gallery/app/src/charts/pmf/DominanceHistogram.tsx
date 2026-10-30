import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { FacetGrid } from '@/components/FacetGrid'
import { Key } from '@/components/Key'
import { Marker } from '@/components/RefLines'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { belowAbove, dominantShare } from '@/lib/pmfCoverage'
import { PAPER_ORDER } from '@/lib/pmfPaper'
import { FONT, INK } from '@/lib/theme'
import type { PmfFile, PmfRow } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { FixedSetNote } from '@/lib/markedRows'

/** the notebooks' DOMINANCE_THRESHOLDS, in % */
const THRESHOLDS = [30, 40, 50, 60, 70]
/** matplotlib's ax.hist(bins=20): twenty equal bins from the smallest share to the largest */
const BINS = 20
const RULES = [
  { key: 'GF', title: 'GF (fraction) rule', axis: 'Dominant source, % of the normalised fractions', of: (r: PmfRow) => r.fraction },
  { key: 'K_F', title: 'K_F (concentration) rule', axis: 'Dominant source, % of the day\'s K_F µg/m³', of: (r: PmfRow) => r.ugm3 },
] as const
const COLOURS = ['One colour', 'By dominant source'] as const

/**
 * dominant_source_threshold_analysis c12: how dominant the dominant PMF factor is, by either rule. The threshold
 * analyses (Aethalometer › Sources, ThresholdSweep) cut at 30–70 %; this is how many days each cut leaves on either
 * side. Bars can be split by which source dominates, so a cut that removes "dominated" days says whose days those are.
 */
export function DominanceHistogram({ pmf, rows }: { pmf: PmfFile; rows: PmfRow[] }) {
  const [colour, setColour] = useSearchParam<(typeof COLOURS)[number]>('pmfdom.colour', 'One colour', COLOURS)
  const labels = useMemo(() => [...pmf.sources].sort((a, b) => PAPER_ORDER.indexOf(a.label) - PAPER_ORDER.indexOf(b.label)), [pmf.sources])
  const panels = useMemo(() => RULES.map((rule) => {
    const days = rows.flatMap((r) => { const d = dominantShare(rule.of(r), labels.map((s) => s.label)); return d ? [{ r, pct: 100 * d.share, source: d.source }] : [] })
    const v = days.map((d) => d.pct)
    return { ...rule, days, n: v.length, mean: d3.mean(v) ?? NaN, median: d3.median(v) ?? NaN, counts: belowAbove(v, THRESHOLDS) }
  }), [rows, labels])

  return (
    <ChartFrame id="pmf-dominance"
      title="How dominant is the dominant source — the dominant PMF factor's share of each day, by the GF and K_F rules"
      subtitle={`Each ${pmf.site_name} PMF day in view (${rows.length} of ${pmf.n}), binned as the notebook did (20 equal bins). GF takes the largest of the normalised source fractions; K_F the largest contribution over the day's K_F total in µg/m³. Dashed lines are the notebooks' dominance thresholds (30–70 %); the box lists the days at or below / above each, the split ThresholdSweep fits on.`}
      provenance="research/ftir_hips_chem/dominant_source_threshold_analysis.ipynb c12 (dominance distribution overview) · from pmf.json"
      controls={<Segmented label="Bars" value={colour} options={COLOURS} onChange={setColour} title="One colour, as the notebook; or each bar split by the source that dominates its days" />}>
      {rows.length < 3 ? <Empty>Too few PMF days in view ({rows.length}).</Empty> : <>
        <FacetGrid items={panels} minCell={300} maxCols={2} title={(p) => p.title}
          note={(p) => `n=${p.n} · mean ${p.mean.toFixed(1)} % · median ${p.median.toFixed(1)} %`}
          render={(p, w) => <DominancePanel days={p.days} counts={p.counts} width={w} axis={p.axis} stacked={colour === 'By dominant source'} sources={labels} />} />
        <Key items={colour === 'By dominant source' ? labels.map((s) => ({ label: s.label, color: s.color, kind: 'square' as const })) : [{ label: 'PMF days', color: INK.accent, kind: 'square' }, { label: 'dominance thresholds', color: INK.negative, kind: 'dash' }]} />
      </>}
      <FixedSetNote what="the PMF solution's days (the model ran on them)" />
    </ChartFrame>
  )
}

function DominancePanel({ days, counts, width, axis, stacked, sources }: {
  days: { r: PmfRow; pct: number; source: string }[]; counts: { threshold: number; below: number; above: number }[]
  width: number; axis: string; stacked: boolean; sources: { label: string; color: string }[]
}) {
  const ref = useRef<HTMLDivElement>(null)
  const tip = useTooltip(ref)
  useDimensions(ref)
  const m = { top: 12, right: 12, bottom: 46, left: 44 }
  const iw = Math.max(160, width - m.left - m.right), ih = 220
  const v = days.map((d) => d.pct)
  const [lo, hi] = [Math.min(...v, ...THRESHOLDS), Math.max(...v, ...THRESHOLDS)]
  const x = d3.scaleLinear().domain([Math.floor(lo / 5) * 5, Math.ceil(hi / 5) * 5]).range([0, iw])
  const [vmin, vmax] = d3.extent(v) as [number, number]
  const edges = d3.range(BINS + 1).map((i) => vmin + ((vmax - vmin) * i) / BINS)
  // each day in its bin; the last bin includes its right edge, as numpy's histogram
  const binOf = (p: number) => Math.min(BINS - 1, Math.max(0, Math.floor(((p - vmin) / (vmax - vmin || 1)) * BINS)))
  const bins = edges.slice(0, -1).map((x0, i) => ({ x0, x1: edges[i + 1], days: days.filter((d) => binOf(d.pct) === i) }))
  const y = d3.scaleLinear().domain([0, (d3.max(bins, (b) => b.days.length) ?? 1) * 1.1]).nice().range([ih, 0])
  const order = sources.map((s) => s.label)
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg width={width} height={ih + m.top + m.bottom} fontFamily={FONT.family}>
        <g transform={`translate(${m.left},${m.top})`}>
          <YAxis scale={y} x={0} label="PMF days" gridWidth={iw} tickCount={5} />
          <XAxis scale={x} y={ih} label={axis} tickCount={8} />
          {bins.map((b, i) => {
            const bx = x(b.x0) + 0.5, bw = Math.max(0, x(b.x1) - x(b.x0) - 1)
            const parts = stacked ? order.map((s) => ({ s, n: b.days.filter((d) => d.source === s).length })).filter((p) => p.n) : [{ s: '', n: b.days.length }]
            let acc = 0
            return <g key={i} onMouseMove={(e) => tip.show(e, [`${b.x0.toFixed(1)} – ${b.x1.toFixed(1)} %`, `${b.days.length} PMF days`,
              ...order.flatMap((s) => { const n = b.days.filter((d) => d.source === s).length; return n ? [`${s}: ${n}`] : [] })])} onMouseLeave={tip.hide}>
              {parts.map((p) => {
                const y0 = y(acc), y1 = y(acc + p.n)
                acc += p.n
                return <rect key={p.s} x={bx} width={bw} y={y1} height={Math.max(0, y0 - y1)} fill={stacked ? sources.find((s) => s.label === p.s)?.color : INK.accent} fillOpacity={0.75} stroke={INK.panel} strokeWidth={0.5} />
              })}
            </g>
          })}
          {THRESHOLDS.map((t) => <Marker key={t} orient="v" at={x(t)} from={0} to={ih} color={INK.negative} />)}
          <g transform={`translate(${iw - 106},4)`} fontSize={10} fontFamily={FONT.mono} fill={INK.text} pointerEvents="none">
            <rect x={-4} y={-2} width={104} height={14 + counts.length * 12} fill={INK.panel} fillOpacity={0.9} stroke={INK.grid} />
            <text y={9}>≤ / &gt; threshold</text>
            {counts.map((c, i) => <text key={c.threshold} y={21 + i * 12}>{c.threshold} %: {c.below}/{c.above}</text>)}
          </g>
        </g>
      </svg>
      {tip.node}
    </div>
  )
}
