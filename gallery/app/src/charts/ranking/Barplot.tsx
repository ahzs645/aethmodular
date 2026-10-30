import { useMemo, useRef } from 'react'
import { seasonOnly } from '@/lib/seasons'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useFitHeight } from '@/lib/chartFit'
import { useSearchParam, useTileSetting } from '@/lib/url'
import { GroupingPicker, useGroupings } from '@/components/GroupingDialog'
import { applyGrouping } from '@/lib/groupings'
import { ExcludedNote } from '@/lib/markedRows'

const STATS = ['Median', 'Mean'] as const
/** what splits a site's bars: its seasons, nothing (one bar per site) or a grouping's bands */
const SPLITS = ['Season', 'None', 'Grouping'] as const
type Split = (typeof SPLITS)[number]

/**
 * Grouped barplot with IQR whiskers — 70 bar figures in the estate.
 * Bars carry the median rather than the mean: these distributions are
 * right-skewed and a handful of high-loading filters drag the mean somewhere
 * no sample actually sits.
 */
export function Barplot({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const fitted = useFitHeight()

  const [stat, setStat] = useTileSetting<(typeof STATS)[number]>('barplot.stat', 'Median', STATS)
  const [splitParam, setSplit] = useTileSetting<Split>('barplot.split', 'Season', SPLITS)
  const [groupId, setGroupId] = useSearchParam<string>('barplot.group', '')
  const groupings = useGroupings()
  const groupingDef = groupings.find(groupId)
  // Grouping without a grouping picked draws as None until one is
  const split: Split = splitParam === 'Grouping' && !groupingDef ? 'None' : splitParam
  const measured = useMemo(() => rows.filter((r) => typeof r[field] === 'number' && Number.isFinite(r[field])), [rows, field])
  const grp = useMemo(() => (groupingDef ? applyGrouping(groupingDef, measured) : null), [groupingDef, measured])
  // the bars within a site, in order, with their colours: seasons, the grouping's bands, or the site itself
  const parts = split === 'Season' ? meta.seasons.map((s) => ({ name: s.name, color: s.color }))
    : split === 'Grouping' && grp ? grp.labels.map((l, i) => ({ name: l, color: grp.colors[i] }))
    : meta.sites.map((s) => ({ name: s.name, color: s.color }))
  const seasons = parts.map((p) => p.name)
  const partOf = (r: FilterRow) => (split === 'Season' ? r.season : split === 'Grouping' && grp ? grp.labelOf(r) : r.site)
  const lg = useLegend()

  const data = useMemo(() => {
    const out: { site: string; season: string; value: number; q1: number; q3: number; n: number; color: string }[] = []
    for (const s of meta.sites) {
      for (const season of seasons) {
        if (split === 'None' && season !== s.name) continue
        const vals = rows
          .filter((r) => r.site === s.name && partOf(r) === season)
          .map((r) => r[field])
          .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
          .sort(d3.ascending)
        if (vals.length < 3) continue
        out.push({
          site: s.name,
          season,
          value: stat === 'Mean' ? (d3.mean(vals) ?? 0) : (d3.quantile(vals, 0.5) ?? 0),
          // whiskers that belong to the statistic: the IQR around a median, a 95 % interval around a mean
          q1: stat === 'Mean' ? (d3.mean(vals) ?? 0) - 1.96 * (d3.deviation(vals) ?? 0) / Math.sqrt(vals.length) : d3.quantile(vals, 0.25) ?? 0,
          q3: stat === 'Mean' ? (d3.mean(vals) ?? 0) + 1.96 * (d3.deviation(vals) ?? 0) / Math.sqrt(vals.length) : d3.quantile(vals, 0.75) ?? 0,
          n: vals.length,
          color: parts.find((x) => x.name === season)?.color ?? s.color,
        })
      }
    }
    return out
  }, [rows, field, stat, meta, split, grp]) // eslint-disable-line react-hooks/exhaustive-deps

  const sites = [...new Set(data.map((d) => d.site))]
  const shown = data.filter((d) => lg.show(d.season))
  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  // site names tilt when they would run into each other, and the plot gives up the height they need
  const longest = d3.max(sites, (s) => s.length) ?? 0
  const tilt = sites.length > 0 && (innerW / sites.length) * 0.9 < longest * 6.6
  // the drop of a name tilted 40° (about 5.8 px a character at 11 px), less what the axis margin already holds
  const below = tilt ? Math.min(150, Math.max(0, longest * 5.8 * Math.sin((40 * Math.PI) / 180) - 18)) : 0
  // unfitted, the chart grows by the tilted labels instead of giving up plot height to them
  const height = fitted ?? 430 + below
  const innerH = height - MARGIN.top - MARGIN.bottom - below

  const x0 = d3.scaleBand<string>().domain(sites).range([0, innerW]).paddingInner(0.24)
  // each site's group holds only the seasons it has (under per-site calendars they differ by site)
  const x1For = (site: string) =>
    d3.scaleBand<string>().domain(seasons.filter((se) => lg.show(se) && shown.some((d) => d.site === site && d.season === se))).range([0, x0.bandwidth()]).padding(0.1)
  const x1BySite = new Map(sites.map((site) => [site, x1For(site)]))
  const yLo = Math.min(0, d3.min(shown, (d) => Math.min(d.value, d.q1)) ?? 0)
  const y = d3.scaleLinear().domain([yLo, (d3.max(shown, (d) => Math.max(d.value, d.q3)) ?? 1) * 1.08]).range([innerH, 0]).nice()

  return (
    <ChartFrame
      id="barplot"
      title={`Barplot — ${split === 'Season' ? 'site by season' : split === 'Grouping' && groupingDef ? `site by ${groupingDef.name}` : 'one bar per site'}, with the spread shown`}
      subtitle={`${stat} per site${split === 'Season' ? ' per selected season (each site on its own calendar unless shared Ethiopian bins are chosen)' : split === 'Grouping' && groupingDef ? ` per band of ${groupingDef.name}` : ''}, whiskered to ${stat === 'Mean' ? 'the 95 % confidence interval of the mean' : 'the interquartile range'}. A bar alone would imply a precision these n≈10–90 groups don't have, so the spread travels with it.`}
      provenance="stands in for 70 bar/barh figures · comparisons.summary_bars · react-graph-gallery.com/barplot"
      controls={
        <>
          <Segmented label="Statistic" value={stat} options={STATS} onChange={setStat} />
          <Segmented label="Split by" value={splitParam} options={SPLITS} onChange={setSplit} title="What divides each site's bars: its seasons, nothing, or a grouping's bands" />
          {splitParam === 'Grouping' && <GroupingPicker value={groupId} rows={measured} onChange={setGroupId} />}
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {data.length === 0 ? (
          <Empty>No {split === 'None' ? 'site' : split === 'Season' ? 'site-season group' : 'site-band group'} has 3 or more filters with {field}.</Empty>
        ) : (
          <svg width={width} height={height} className="animated">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={y} x={0} label={withUnit(field, meta.field_units)} gridWidth={innerW} pick="field" />
              <XAxis scale={x0} y={innerH} rotate={tilt ? -40 : 0} />
              {shown.map((d, i) => {
                const x1 = x1BySite.get(d.site)!
                const bx = (x0(d.site) ?? 0) + (x1(d.season) ?? 0)
                const bw = x1.bandwidth()
                return (
                  <g
                    key={i}
                    data-export-season={d.season} data-export-color={d.color}
                    opacity={lg.dim(d.season)}
                    onMouseEnter={(e) => tip.show(e, [split === 'None' ? d.site : `${d.site} · ${split === 'Season' ? seasonOnly(d.season) : d.season}`, `${stat.toLowerCase()} = ${fmt(d.value)}`, `${stat === 'Mean' ? '95 % CI of the mean' : 'IQR'} ${fmt(d.q1)} – ${fmt(d.q3)}`, `n = ${d.n}`])}
                    onMouseLeave={tip.hide}
                  >
                    <rect x={bx} y={Math.min(y(d.value), y(0))} width={bw} height={Math.abs(y(d.value) - y(0))} fill={d.color} fillOpacity={0.78} rx={2} />
                    <line x1={bx + bw / 2} x2={bx + bw / 2} y1={y(d.q1)} y2={y(d.q3)} stroke={INK.text} strokeWidth={1.3} />
                    <line x1={bx + bw / 2 - 4} x2={bx + bw / 2 + 4} y1={y(d.q3)} y2={y(d.q3)} stroke={INK.text} strokeWidth={1.3} />
                    <line x1={bx + bw / 2 - 4} x2={bx + bw / 2 + 4} y1={y(d.q1)} y2={y(d.q1)} stroke={INK.text} strokeWidth={1.3} />
                    {bw > 22 && !tilt && (
                      <text x={bx + bw / 2} y={innerH + 32} textAnchor="middle" fontSize={9} fill={INK.muted} fontFamily={FONT.mono}>
                        {d.n}
                      </text>
                    )}
                  </g>
                )
              })}
            </g>
          </svg>
        )}
        {split !== 'None' && <Legend items={parts.filter((p) => data.some((d) => d.season === p.name)).map((p) => ({ label: p.name, color: p.color, shape: 'square' as const }))} {...lg.props}
          note={tilt ? 'n per bar is in its tooltip' : 'small numbers under the site names are n per bar'} />}
        {tip.node}
      </div>
      <ExcludedNote fields={[field]} />
    </ChartFrame>
  )
}
