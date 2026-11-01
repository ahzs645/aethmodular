import { useJsonLoader, useLoaded } from '@/lib/data'
import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { FacetGrid } from '@/components/FacetGrid'
import { useGroupDomain } from '@/components/PanelGroup'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { monthlyMeans } from '@/lib/stats'
import { extentOf } from '@/lib/facets'
import { FONT, INK, MONTH_LETTERS, VARIABLE_COLOR } from '@/lib/theme'
import { loadDaily, type DailyWeather } from '@/lib/weather'
import { useChartOptions } from '@/lib/chartOptions'
import { usePanelAxes } from '@/lib/panelAxes'
import type { MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'

const MAX_SITES = 16
/** stacking more panels than this makes a long scroll: the grid takes over */
const MAX_STACKED = 6
const NONE: Record<string, DailyWeather | null> = {}
// the site name is the facet's title row, above the svg
const M = { top: 8, right: 30, bottom: 34, left: 34 }
/** a stacked panel above the last one: the season strip, no month letters (the bottom panel has them) */
const BOTTOM_STRIP_ONLY = 16
const TITLE_IN_SVG = 14
const LAYOUTS = ['Stacked', 'Grid'] as const
/** Panel axes (the panel group): precipitation from 0, temperature padded 2 °C, as each panel draws its own */
const SHARE = {
  y: ([, hi]: [number, number]): [number, number] => [0, hi || 1],
  y2: ([lo, hi]: [number, number]): [number, number] => [Math.floor(lo - 2), Math.ceil(hi + 2)],
}

type Month = { m: number; n: number; p: number | null; t: number | null; rh: number | null }
type Climate = { site: { name: string; code: string }; months: Month[] | null }

/**
 * Does each site's season calendar match its weather? Mean monthly precipitation (bars)
 * and temperature (line) from daily ERA5 over the site's sampling span, with the local
 * calendar as a strip under the months. The rain-based "dynamic seasons" of
 * addis_03_meteorology, as a check on the fixed month bins (docs/site-seasonality*.md).
 */
export function MonthlyClimate({ meta, sites, id = 'weather-climate', title = 'Monthly climate and the season calendar — one panel per site', large = false }: {
  meta: MetaFile; sites: { name: string; code: string }[]; id?: string; title?: string
  /** one big panel (the one-site view) instead of small multiples */
  large?: boolean
}) {
  const jsonLoader = useJsonLoader()
  const wrapRef = useRef<HTMLDivElement>(null)
  const W = large ? 620 : 250, H = large ? 240 : 150
  const tip = useTooltip(wrapRef)
  const shown = sites.slice(0, MAX_SITES)
  const key = shown.map((s) => s.code).join(',')
  // the last sites' climate stays up while the next ones load
  const data = useLoaded(key, () => Promise.all(shown.map((s) => loadDaily(s.code, jsonLoader).then((d) => [s.code, d] as const).catch(() => [s.code, null] as const)))
    .then((pairs): Record<string, DailyWeather | null> => Object.fromEntries(pairs)), { keep: true, deps: [jsonLoader] }).data ?? NONE

  const o = useChartOptions('climate', [
    { key: 'layout', label: 'Layout', default: 'Stacked', hidden: 'Grid',
      choices: LAYOUTS.map((value) => ({ value, when: (c) => value !== 'Stacked' || (c.n as number) <= MAX_STACKED })),
      when: (c) => !c.large && (c.n as number) >= 2,
      title: `Stacked: one site per row, so the months and the season strips line up down the page (up to ${MAX_STACKED} sites). Grid: small multiples side by side` },
  ], { n: shown.length, large })
  const stacked = o.values.layout === 'Stacked' && !large && shown.length >= 2
  // one precipitation axis and one temperature axis for every panel under Panel axes: Shared (the panel group below)
  const [pax] = usePanelAxes()
  const shared = pax === 'Shared' && shown.length >= 2

  const climate = useMemo((): Climate[] => shown.map((s) => {
    const d = data[s.code]
    if (!d) return { site: s, months: null }
    // lib/stats monthlyMeans, as the AERONET climatology: each month's mean over the days that have a value
    const p = monthlyMeans(d.date, d.p), t = monthlyMeans(d.date, d.t), rh = monthlyMeans(d.date, d.rh)
    const months = d3.range(1, 13).map((m) => {
      const i = m - 1, mp = p[i].mean
      // mean daily rate x days in the month: a monthly total comparable across sites
      return { m, n: d.date.filter((x) => Number(x.slice(5, 7)) === m).length, p: mp !== null ? mp * new Date(2023, m, 0).getDate() : null, t: t[i].mean, rh: rh[i].mean }
    })
    return { site: s, months }
  }), [data, key]) // eslint-disable-line react-hooks/exhaustive-deps

  const last = climate.length - 1

  return (
    <ChartFrame id={id} title={title} controls={o.controls()}
      subtitle={`Mean monthly precipitation (bars, mm) and temperature (line, °C) from daily ERA5 over each site's sampling span; the strip under the months is the site's local season calendar. Wet bars under a 'dry' season are where a calendar and the weather disagree.${shared ? ' All panels share one precipitation axis and one temperature axis, so heights compare across sites.' : shown.length >= 2 ? ' Each panel has its own axes: compare shapes, not heights, across sites.' : ''}`}
      provenance="addis_03_meteorology Task 3 (dynamic, rain-based seasons) for every site · ERA5 via Open-Meteo">
      <div ref={wrapRef} className="chart-wrap">
        {shown.length === 0 && <Empty>No SPARTAN sites in the current pick.</Empty>}
        <FacetGrid items={climate} minCell={large || stacked ? 300 : W} maxCell={stacked ? 760 : W + 60} maxCols={large || stacked ? 1 : 4}
          gap={stacked ? 6 : 12} center expandable={!large}
          title={({ site }) => site.name} label={({ site }) => `${site.name} monthly climate`}
          itemKey={({ site }) => site.code} share={SHARE}
          render={(c, pw, i, cell) => {
            // stacked: the month letters only under the bottom panel, the season strip under every one
            const letters = !stacked || i === last || cell.expanded
            const mb = letters ? M.bottom : BOTTOM_STRIP_ONLY
            const h = cell.expanded ? 300 : (cell.height ?? (stacked ? 118 : H - TITLE_IN_SVG)) - (M.bottom - mb)
            if (!c.months) return <div style={{ width: pw, height: h }} className="u">{data[c.site.code] === null ? 'no daily weather' : 'loading…'}</div>
            return <ClimatePanel site={c.site.name} months={c.months} meta={meta} width={pw} height={h} bottom={mb} letters={letters} tip={tip} />
          }} />
        {tip.node}
      </div>
      {sites.length > MAX_SITES && <Note>Showing the first {MAX_SITES} of {sites.length} sites; narrow the site pick to see the others.</Note>}
    </ChartFrame>
  )
}

/** One site's panel: its axes come from the panel group (Panel axes: Shared draws one range for every site). */
function ClimatePanel({ site, months, meta, width, height, bottom, letters, tip }: {
  site: string; months: Month[]; meta: MetaFile; width: number; height: number; bottom: number; letters: boolean
  tip: ReturnType<typeof useTooltip>
}) {
  const iw = width - M.left - M.right, ih = height - M.top - bottom
  const tv = months.map((m) => m.t).filter((v): v is number => v !== null)
  const ownP = extentOf([0, ...months.map((m) => m.p ?? 0)]), ownT = extentOf(tv)
  const gp = useGroupDomain('y', ownP), gt = useGroupDomain('y2', ownT)
  const x = d3.scaleBand<number>().domain(d3.range(1, 13)).range([0, iw]).padding(0.15)
  const yP = d3.scaleLinear().domain(gp?.domain ?? SHARE.y(ownP ?? [0, 1])).range([ih, 0]).nice()
  const yT = d3.scaleLinear().domain(gt?.domain ?? (ownT ? SHARE.y2(ownT) : [-2, 3])).range([ih, 0])
  const seasons = seasonsForSite(meta, site)
  const line = d3.line<Month>().defined((m) => m.t !== null).x((m) => (x(m.m) ?? 0) + x.bandwidth() / 2).y((m) => yT(m.t!))
  return (
    <svg width={width} height={height} fontFamily={FONT.family}>
      <g transform={`translate(${M.left},${M.top})`}>
        {yP.ticks(3).map((t) => <g key={t}><line x1={0} x2={iw} y1={yP(t)} y2={yP(t)} stroke={INK.grid} /><text x={-4} y={yP(t)} dy="0.32em" textAnchor="end" fontSize={9.5} fill={VARIABLE_COLOR.rain}>{t}</text></g>)}
        {yT.ticks(3).map((t) => <text key={t} x={iw + 4} y={yT(t)} dy="0.32em" fontSize={9.5} fill={VARIABLE_COLOR.temperature}>{t}°</text>)}
        {months.map((m) => (
          <g key={m.m}>
            {m.p !== null && <rect x={x(m.m)} width={x.bandwidth()} y={yP(m.p)} height={ih - yP(m.p)} fill={VARIABLE_COLOR.rain} fillOpacity={0.75} />}
            <rect x={x(m.m)} width={x.bandwidth()} height={ih + (letters ? 22 : 11)} fill="transparent"
              {...tip.bind(() => [`${site} · ${d3.timeFormat('%B')(new Date(2023, m.m - 1, 1))}`,
                `precipitation ${fmt(m.p, 0)} mm/month`, `temperature ${fmt(m.t, 1)} °C`, `RH ${fmt(m.rh, 0)} %`,
                `season: ${seasons.find((s) => s.months.includes(m.m))?.name.replace(`${site} · `, '') ?? '—'}`, `${m.n} days`])} />
          </g>
        ))}
        <path d={line(months) ?? ''} fill="none" stroke={VARIABLE_COLOR.temperature} strokeWidth={1.8} pointerEvents="none" />
        {months.map((m) => {
          const se = seasons.find((s) => s.months.includes(m.m))
          return <g key={`s${m.m}`} pointerEvents="none">
            <rect x={(x(m.m) ?? 0) - x.step() * x.paddingInner() / 2} width={x.step()} y={ih + 4} height={7} fill={se?.color ?? '#ddd'} />
            {letters && <text x={(x(m.m) ?? 0) + x.bandwidth() / 2} y={ih + 22} textAnchor="middle" fontSize={9.5} fill={INK.muted}>{MONTH_LETTERS[m.m - 1]}</text>}
          </g>
        })}
      </g>
    </svg>
  )
}
