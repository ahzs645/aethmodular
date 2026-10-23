import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { useTooltip } from '@/hooks/useTooltip'
import { useDimensions } from '@/hooks/useGalleryData'
import { useFitHeight } from '@/lib/chartFit'
import { fmt } from '@/lib/stats'
import { useChartOptions } from '@/lib/chartOptions'
import { FONT, INK } from '@/lib/theme'
import type { RoseFile, WeatherSource } from '@/lib/weather'
import { NO_PMF } from '@/lib/pmfGroups'
import type { FilterRow, MetaFile } from '@/lib/types'

const SPEED_COLORS = ['#c6dbef', '#6baed6', '#08519c']
const SPLITS = ['Site', 'Season', 'PMF source'] as const
const MAX_PANELS = 16
const SIZE = 210

/**
 * Wind roses over the filters' sampling hours: 8 sectors, stacked by speed class, calm hours
 * in the middle, one outer-ring scale for every panel so roses compare directly
 * (meteorology_revisions R1). Split by site, by season at the chosen site, or (Addis) by each
 * filter's dominant PMF source, meteorology_source_interaction §2's question: do some source
 * types arrive from particular directions?
 */
export function WindRoses({ rows, meta, roses, source, site, pmfLabelOfDate, splits = SPLITS, id = 'weather-roses', title = 'Wind roses — where the air came from while the filters sampled' }: {
  rows: FilterRow[]; meta: MetaFile; roses: RoseFile | null; source: WeatherSource
  /** the splits on offer: the one-site view splits by season or PMF source, the comparison by site */
  splits?: readonly (typeof SPLITS)[number][]; id?: string; title?: string
  site: { name: string; code: string } | null
  pmfLabelOfDate: (d: string | null | undefined) => string | null
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const tip = useTooltip(wrapRef)
  const { width } = useDimensions(wrapRef)
  const fit = useFitHeight()
  // the options and when each applies (lib/chartOptions): the page says which splits it offers
  const o = useChartOptions(id, [
    { key: 'split', label: 'Split by', default: splits[0], choices: SPLITS.map((v) => ({ value: v, when: () => splits.includes(v) })),
      title: 'Site: one rose per site. Season and PMF source: the chosen site (PMF exists for Addis 2023 only).' },
  ])
  const split = o.values.split as (typeof SPLITS)[number]
  const table = roses ? (source === 'station' ? roses.station : roses.era5) : null

  const groups = useMemo(() => {
    if (!table) return []
    const one = site ? rows.filter((r) => r.site === site.name) : rows
    const keyed: [string, string, FilterRow[]][] = split === 'Site'
      ? d3.groups(rows, (r) => r.site).map(([k, rs]) => [k, meta.sites.find((s) => s.name === k)?.color ?? INK.muted, rs])
      : split === 'Season'
        ? d3.groups(one, (r) => r.season).map(([k, rs]) => [k, meta.seasons.find((s) => s.name === k)?.color ?? INK.muted, rs])
        : d3.groups(one.filter((r) => { const l = pmfLabelOfDate(r.date); return !!l && l !== NO_PMF }), (r) => pmfLabelOfDate(r.date)!).map(([k, rs]) => [k, INK.muted, rs])
    return keyed.map(([name, color, rs]) => {
      const counts = new Array(25).fill(0)
      let n = 0
      for (const r of rs) { const c = table[r.id]; if (c) { n++; c.forEach((v, i) => { counts[i] += v }) } }
      const hours = d3.sum(counts)
      const pct = counts.map((c) => (hours ? (100 * c) / hours : 0))
      return { name, color, n, hours, pct }
    }).filter((g) => g.hours > 0).sort((a, b) => b.n - a.n)
  }, [table, rows, split, site, meta, pmfLabelOfDate])
  const shown = groups.slice(0, MAX_PANELS)
  const rMax = d3.max(shown, (g) => d3.max(d3.range(8), (s) => g.pct[s * 3] + g.pct[s * 3 + 1] + g.pct[s * 3 + 2])) ?? 10
  // in a fitted tile the roses share it: the column count that gives the largest rose fitting both
  // the width and the height (a title line under each, 10 px gaps); elsewhere a fixed 210 px
  // on a page, roses grow to share the row (210 to 300 px each) rather than staying small
  const size = !shown.length ? SIZE : fit === null
    ? Math.round(Math.max(SIZE, Math.min(300, (width - (shown.length - 1) * 10) / shown.length)))
    : Math.round(Math.max(110, Math.min(280, largestRose(shown.length, width, fit))))
  const R = size / 2 - 26
  const r = d3.scaleSqrt().domain([0, rMax]).range([16, R]).nice()
  const arc = d3.arc<{ a: number; r0: number; r1: number }>().startAngle((d) => d.a - Math.PI / 8 + 0.03).endAngle((d) => d.a + Math.PI / 8 - 0.03).innerRadius((d) => d.r0).outerRadius((d) => d.r1)
  const label = (name: string) => name.replace(`${site?.name} · `, '')

  return (
    <ChartFrame id={id} title={title}
      subtitle={`Hours inside the filters' sampling windows, by direction (8 sectors) and speed; the number in the middle is calm (${roses?.calm ?? '< 0.5 m/s'}). ${source === 'station' ? 'Airport station winds (10 m, open site).' : 'ERA5 10 m winds are a 0.25° grid-box average and run lower than an airport mast.'} Every panel shares one radial scale.`}
      provenance="meteorology_source_interaction §2 · meteorology_revisions R1"
      controls={o.controls()}>
      <div ref={wrapRef} className="chart-wrap" style={{ display: 'flex', flexWrap: 'wrap', gap: 10, justifyContent: 'center' }}>
        {!roses ? <Empty>Loading the wind roses…</Empty> : shown.length === 0 ? <Empty>{split === 'PMF source' ? 'No filter here falls on a PMF day (ETAD 2023 only).' : 'No filters with wind data in this subset.'}</Empty> : shown.map((g) => (
          <svg key={g.name} width={size} height={size + 18} fontFamily={FONT.family}>
            <text x={size / 2} y={13} textAnchor="middle" fontSize={11.5} fontWeight={600} fill={g.color === INK.muted ? INK.text : g.color}>{label(g.name)}</text>
            <g transform={`translate(${size / 2},${size / 2 + 14})`}>
              {r.ticks(3).filter((t) => t > 0).map((t) => <g key={t}><circle r={r(t)} fill="none" stroke={INK.grid} /><text x={2} y={-r(t) - 2} fontSize={8.5} fill={INK.muted}>{t}%</text></g>)}
              {['N', 'E', 'S', 'W'].map((d, i) => <text key={d} x={Math.sin((i * Math.PI) / 2) * (R + 10)} y={-Math.cos((i * Math.PI) / 2) * (R + 10)} dy="0.35em" textAnchor="middle" fontSize={10} fill={INK.muted}>{d}</text>)}
              {d3.range(8).map((s) => {
                let acc = 0
                return [0, 1, 2].map((k) => {
                  const v = g.pct[s * 3 + k]
                  const r0 = r(acc), r1 = r(acc + v)
                  acc += v
                  return v > 0 ? <path key={`${s}-${k}`} d={arc({ a: (s * Math.PI) / 4, r0, r1 }) ?? ''} fill={SPEED_COLORS[k]} stroke={INK.panel} strokeWidth={0.5}
                    onMouseEnter={(e) => tip.show(e, [`${label(g.name)} · from ${roses.sectors[s]}`, `${roses.speed_classes[k]}: ${fmt(v, 1)} % of hours`,
                      `all speeds from ${roses.sectors[s]}: ${fmt(g.pct[s * 3] + g.pct[s * 3 + 1] + g.pct[s * 3 + 2], 1)} %`])}
                    onMouseLeave={tip.hide} /> : null
                })
              })}
              <circle r={14} fill={INK.panel} stroke={INK.grid} />
              <text dy="0.35em" textAnchor="middle" fontSize={9} fill={INK.muted} fontFamily={FONT.mono}>{fmt(g.pct[24], 0)}%</text>
              <text y={size / 2 - 4} textAnchor="middle" fontSize={9} fill={INK.muted}>{g.n} filters · {g.hours} h</text>
            </g>
          </svg>
        ))}
        {tip.node}
      </div>
      <div className="legend">
        {(roses?.speed_classes ?? []).map((c, i) => <span key={c} className="legend-item"><span className="swatch square" style={{ background: SPEED_COLORS[i] }} />{c}</span>)}
        <span className="legend-item">centre: calm hours</span>
      </div>
      {groups.length > MAX_PANELS && <Note>Showing the {MAX_PANELS} groups with the most filters of {groups.length}.</Note>}
    </ChartFrame>
  )
}

/** the largest rose side that lays n roses out in width × height (10 px gaps, 18 px title per rose) */
function largestRose(n: number, width: number, height: number) {
  let best = 0
  for (let cols = 1; cols <= n; cols++) {
    const rows = Math.ceil(n / cols)
    best = Math.max(best, Math.min((width - (cols - 1) * 10) / cols, (height - (rows - 1) * 10) / rows - 18))
  }
  return best
}
