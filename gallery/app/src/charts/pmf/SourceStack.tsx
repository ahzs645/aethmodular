import { useMemo, useRef } from 'react'
import { calendarShortName, seasonOnly } from '@/lib/seasons'
import { timeSeasonBands } from '@/components/SeasonBands'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { useChartOptions, type OptionContext } from '@/lib/chartOptions'
import { PAPER_COLORS, PAPER_ORDER, PAPER_SEASONS, paperSeasonBands, paperSeasonOf } from '@/lib/pmfPaper'
import { Legend, toggleIn, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { useHighlight } from '@/lib/highlight'
import { INK, MARGIN } from '@/lib/theme'
import type { MetaFile, PmfFile, PmfRow } from '@/lib/types'
import { useFitHeight } from '@/lib/chartFit'

const MODES = ['Relative (%)', 'Absolute (µg/m³)'] as const
const SHAPES = ['Stacked', 'Daily bars', 'Streamgraph'] as const
const CALENDARS = ['Season calendar', 'Paper (four seasons)'] as const
const PALETTES = ['Gallery', 'Paper'] as const
const DAY = 86_400_000

/**
 * Stacked area / streamgraph — PMF source contributions over time.
 *
 * This is the chart the estate is most missing: exactly one stacked-area cell
 * exists across 827 figures, and source apportionment is precisely the
 * question it answers. Fractions are the normalised relative contributions
 * (normalize_gf_fractions); the raw GF1-GF5 columns are PM2.5 mass fractions
 * summing to 0.03-0.46 and would be wrong to stack directly.
 *
 * The streamgraph is the same stack with a wiggle offset and inside-out
 * ordering — it trades a readable baseline for a readable *shape*, which is
 * the right trade when the question is "when does charcoal swell".
 */
export function SourceStack({ pmf, rows: subsetRows, meta }: { pmf: PmfFile; rows: PmfRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()
  const height = useFitHeight() ?? 440

  // the options and when each applies (lib/chartOptions): relative shares sum to 100 % every
  // day, so a streamgraph would only wobble the baseline; it is offered on the absolute scale
  const o = useChartOptions('sourcestack', [
    { key: 'mode', label: 'Scale', choices: MODES, default: 'Relative (%)' },
    { key: 'shape', label: 'Shape', choices: SHAPES.map((v) => ({ value: v, when: v === 'Streamgraph' ? (c: OptionContext) => c.mode !== 'Relative (%)' : undefined })), default: 'Stacked',
      title: 'Daily bars: one bar per sampling day on the date axis (the PMF paper\'s Figure 7). Streamgraph (absolute scale): wiggle offset, inside-out order — no baseline, better shape' },
    { key: 'calendar', label: 'Seasons', choices: CALENDARS.map((v) => ({ value: v, label: v === 'Season calendar' ? `Sidebar: ${calendarShortName(meta.season_convention)}` : v })), default: 'Paper (four seasons)', style: 'select',
      title: 'The strip along the top: the Filters sidebar\'s season calendar (it follows any change there), or the PMF paper\'s four seasons' },
    { key: 'palette', label: 'Colours', choices: PALETTES, default: 'Paper', title: 'Paper: the PMF paper\'s source colours and stacking order' },
    { key: 'count', node: () => <span className="control">{rows.length} filters · {pmf.site_name}</span> },
  ])
  const mode = o.values.mode as (typeof MODES)[number]
  const { hidden, setHidden, dim, hover, setHover } = useLegend()

  const paper = o.values.palette === 'Paper'
  const bars = o.values.shape === 'Daily bars'
  // the paper's order from the bottom (Fossil Fuel up to Charcoal) for its colours or its bars
  const ordered = paper || bars ? [...pmf.sources].sort((a, b) => PAPER_ORDER.indexOf(a.label) - PAPER_ORDER.indexOf(b.label)) : pmf.sources
  const labels = ordered.map((s) => s.label).filter((l) => !hidden.has(l))
  const colorOf = useMemo(() => new Map(pmf.sources.map((s) => [s.label, paper ? PAPER_COLORS[s.label] ?? s.color : s.color])), [pmf.sources, paper])
  const paperSeasons = o.values.calendar === 'Paper (four seasons)'
  // the export keys the dominant source (polluted_marine); readers see its name
  const labelOfKey = (k: string | null | undefined) => pmf.sources.find((s) => s.key === k)?.label ?? k ?? '—'

  // Season filtering and the active calendar are applied upstream by the
  // global subset bar, so every tab means the same thing by "Kiremt".
  const rows = useMemo(
    () => subsetRows.map((r) => ({ ...r, d: new Date(r.date + 'T00:00:00') })).sort((a, b) => +a.d - +b.d),
    [subsetRows]
  )

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const relative = mode === 'Relative (%)'
  const shape = o.values.shape as (typeof SHAPES)[number]

  const stacked = useMemo(() => {
    const src = relative ? 'fraction' : 'ugm3'
    const gen = d3
      .stack<(typeof rows)[number]>()
      .keys(labels)
      .value((r, k) => {
        const v = (r as any)[src][k]
        return typeof v === 'number' && Number.isFinite(v) ? v : 0
      })
    if (shape === 'Streamgraph') gen.offset(d3.stackOffsetWiggle).order(d3.stackOrderInsideOut)
    return gen(rows)
  }, [rows, labels, relative, shape])

  const x = d3.scaleTime().domain(d3.extent(rows, (r) => r.d) as [Date, Date]).range([0, innerW])
  /** the filter nearest the pointer: every x position in the stack is one filter */
  const nearestTo = (e: React.MouseEvent) => {
    const b = wrapRef.current!.getBoundingClientRect()
    const dt = x.invert(e.clientX - b.left - MARGIN.left)
    return rows.reduce((best, r) => (Math.abs(+r.d - +dt) < Math.abs(+best.d - +dt) ? r : best), rows[0])
  }
  const yLo = shape === 'Streamgraph' ? (d3.min(stacked, (l) => d3.min(l, (d) => d[0])) ?? 0) : 0
  const yHi = d3.max(stacked, (l) => d3.max(l, (d) => d[1])) ?? 1
  const y = d3.scaleLinear().domain([yLo, yHi * (shape === 'Streamgraph' ? 1 : 1.02)]).range([innerH, 0]).nice()

  const area = d3.area<d3.SeriesPoint<(typeof rows)[number]>>().x((d) => x(d.data.d)).y0((d) => y(d[0])).y1((d) => y(d[1])).curve(d3.curveMonotoneX)

  const seasonBands = useMemo(() => {
    if (!rows.length) return []
    return paperSeasons ? paperSeasonBands(x, innerW) : timeSeasonBands(x, meta.seasons, innerW)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, innerW, meta.seasons, paperSeasons])
  // the strip's seasons in order, each once, for its legend
  const stripSeasons = [...new Map(seasonBands.map((b) => [b.name, b])).values()]
  // a bar per sampling day: three-quarters of the typical spacing between samples, so gaps read as gaps
  const spacing = d3.median(rows.slice(1).map((r, i) => x(r.d) - x(rows[i].d))) ?? 6
  const barW = Math.max(1.5, Math.min(18, spacing * 0.75, x(new Date(+rows[0]?.d + 3 * DAY)) - x(rows[0]?.d ?? new Date())))
  const recordOf = (near: (typeof rows)[number]) => hl.openRecord({
    id: near.id,
    source: 'the PMF source stack',
    site: 'Addis Ababa',
    date: near.date,
    fields: [
      ['season', near.season],
      ...labels.map((l): [string, string] => [l, `${near.fraction[l] !== null ? (near.fraction[l]! * 100).toFixed(1) + ' %' : '—'}${near.ugm3[l] !== null ? ` · ${fmt(near.ugm3[l], 2)} µg/m³` : ''}`]),
      ['dominant source', near.dominant_source ? `${labelOfKey(near.dominant_source)} (${((near.dominant_fraction ?? 0) * 100).toFixed(0)} %)` : null],
    ],
  })
  const tipOf = (near: (typeof rows)[number]) => [
    `${near.date} · ${paperSeasons ? paperSeasonOf(near.date)?.name ?? '' : seasonOnly(String(near.season ?? ''))}`,
    ...labels.map((l) => {
      const f = near.fraction[l]
      const a = near.ugm3[l]
      return `${l}: ${f !== null ? (f * 100).toFixed(1) + ' %' : '—'}${a !== null ? ` (${fmt(a, 2)} µg/m³)` : ''}`
    }),
    `dominant: ${labelOfKey(near.dominant_source)} ${near.dominant_fraction !== null ? `(${(near.dominant_fraction * 100).toFixed(0)} %)` : ''}`,
    'click for the filter record',
  ]

  return (
    <ChartFrame
      id="sourcestack"
      title="Stacked area · Streamgraph — PMF source contributions over time"
      subtitle={`${pmf.n} ETAD filters with a PMF solution. Relative mode stacks the normalised source mix to 100 %; absolute mode stacks the K_Fn concentrations in µg/m³, so the height is total apportioned mass. Click a source in the legend to drop it from the stack.`}
      provenance="the estate has exactly 1 stacked-area figure in 827 · react-graph-gallery.com/stacked-area-plot · /streamchart"
      controls={o.controls()}
    >
      <div ref={wrapRef} className="chart-wrap">
        {!rows.length ? (
          <Empty>No PMF filters in this subset.</Empty>
        ) : (
          <svg width={width} height={height} className="animated">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {/* the season strip, solid as in the paper's figure, named where a band has room */}
              <g pointerEvents="none">
                {seasonBands.map((b, i) => <rect key={i} data-export-season={b.name} data-export-color={b.color} x={b.x} y={-14} width={b.w} height={11} fill={b.color} />)}
                {seasonBands.map((b, i) => {
                  // the full name where it fits, else its first word (Belg), else nothing (about 5 px a character at 9 px)
                  const full = seasonOnly(b.name), short = full.split(' (')[0]
                  const text = full.length * 5 + 8 <= b.w ? full : short.length * 5 + 8 <= b.w ? short : ''
                  return text && <text key={i} x={b.x + b.w / 2} y={-5.5} textAnchor="middle" fontSize={9} fill={INK.text}>{text}</text>
                })}
              </g>
              <YAxis
                scale={y}
                x={0}
                label={shape === 'Streamgraph' ? (relative ? 'share (baseline free)' : 'µg/m³ (baseline free)') : relative ? 'share of apportioned mass' : 'µg/m³'}
                gridWidth={innerW}
                format={(v: number) => (relative ? `${Math.round(v * 100)}%` : String(v))}
              />
              <XAxis scale={x as any} y={innerH} format={(d: Date) => d3.timeFormat('%b %Y')(d)} tickCount={7} />

              {bars && stacked.map((layer) => (
                <g key={String(layer.key)} fill={colorOf.get(String(layer.key)) ?? INK.muted} fillOpacity={dim(String(layer.key))}>
                  {layer.map((d) => <rect key={d.data.id} x={x(d.data.d) - barW / 2} width={barW} y={y(d[1])} height={Math.max(0, y(d[0]) - y(d[1]))}
                    style={{ cursor: 'pointer' }} onClick={() => recordOf(d.data)} onMouseMove={(e) => tip.show(e, tipOf(d.data))} onMouseLeave={tip.hide} />)}
                </g>
              ))}
              {!bars && stacked.map((layer) => (
                <path
                  key={String(layer.key)}
                  d={area(layer) ?? ''}
                  fill={colorOf.get(String(layer.key)) ?? INK.muted}
                  fillOpacity={0.88 * dim(String(layer.key))}
                  stroke={INK.panel}
                  strokeWidth={0.4}
                  style={{ transition: 'd 0.35s', cursor: 'pointer' }}
                  onClick={(e) => recordOf(nearestTo(e))}
                  onMouseMove={(e) => tip.show(e, tipOf(nearestTo(e)))}
                  onMouseLeave={tip.hide}
                />
              ))}
            </g>
          </svg>
        )}
        <Legend
          items={ordered.map((s) => ({ label: s.label, color: colorOf.get(s.label) ?? s.color, shape: 'square' as const }))}
          hidden={hidden}
          onToggle={(l) => setHidden((h) => toggleIn(h, l))}
          onHover={setHover}
          highlighted={hover}
          note={undefined}
        />
        {stripSeasons.length > 0 && <div className="legend">{stripSeasons.map((b) => <span key={b.name} className="legend-item" title={paperSeasons ? PAPER_SEASONS.find((p) => p.name === b.name)?.note + ' 2023' : undefined}><span className="swatch square" style={{ background: b.color }} />{seasonOnly(b.name)}</span>)}
          <span className="legend-note">{paperSeasons ? 'top strip: the PMF paper\'s seasons (its 2023 dates)' : `top strip: ${meta.season_convention === 'rain_rule' ? 'the rain rule, from this year\'s rain' : 'the selected season calendar'}`}</span></div>}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
