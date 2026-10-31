import { useJsonLoader } from '@/lib/data'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { FacetGrid } from '@/components/FacetGrid'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { loadDaily, type DailyWeather } from '@/lib/weather'
import type { MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'

const MONTHS = 'JFMAMJJASOND'.split('')
const MAX_SITES = 16
// the site name is the facet's title row, above the svg
const M = { top: 8, right: 30, bottom: 34, left: 34 }
const TITLE_IN_SVG = 14

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
  const [data, setData] = useState<Record<string, DailyWeather | null>>({})
  const shown = sites.slice(0, MAX_SITES)
  const key = shown.map((s) => s.code).join(',')

  useEffect(() => {
    let live = true
    Promise.all(shown.map((s) => loadDaily(s.code, jsonLoader).then((d) => [s.code, d] as const).catch(() => [s.code, null] as const)))
      .then((pairs) => { if (live) setData(Object.fromEntries(pairs)) })
    return () => { live = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, jsonLoader])

  const climate = useMemo(() => shown.map((s) => {
    const d = data[s.code]
    if (!d) return { site: s, months: null }
    const months = d3.range(1, 13).map((m) => {
      const idx = d.date.map((x, i) => [Number(x.slice(5, 7)), i] as const).filter(([mm]) => mm === m).map(([, i]) => i)
      const p = idx.map((i) => d.p[i]).filter((v): v is number => v !== null)
      const t = idx.map((i) => d.t[i]).filter((v): v is number => v !== null)
      const rh = idx.map((i) => d.rh[i]).filter((v): v is number => v !== null)
      // mean daily rate x days in the month: a monthly total comparable across sites
      return { m, n: idx.length, p: p.length ? d3.mean(p)! * new Date(2023, m, 0).getDate() : null, t: d3.mean(t) ?? null, rh: d3.mean(rh) ?? null }
    })
    return { site: s, months }
  }), [data, key]) // eslint-disable-line react-hooks/exhaustive-deps


  return (
    <ChartFrame id={id} title={title}
      subtitle="Mean monthly precipitation (bars, mm) and temperature (line, °C) from daily ERA5 over each site's sampling span; the strip under the months is the site's local season calendar. Wet bars under a 'dry' season are where a calendar and the weather disagree."
      provenance="addis_03_meteorology Task 3 (dynamic, rain-based seasons) for every site · ERA5 via Open-Meteo">
      <div ref={wrapRef} className="chart-wrap">
        {shown.length === 0 && <Empty>No SPARTAN sites in the current pick.</Empty>}
        <FacetGrid items={climate} minCell={large ? 300 : W} maxCell={W + 60} maxCols={large ? 1 : 4} gap={12} center expandable={!large}
          title={({ site }) => site.name} label={({ site }) => `${site.name} monthly climate`}
          render={({ site, months }, pw, _, cell) => {
          const h = cell.expanded ? 300 : cell.height ?? H - TITLE_IN_SVG
          if (!months) return <div style={{ width: pw, height: h }} className="u">{data[site.code] === null ? 'no daily weather' : 'loading…'}</div>
          const iw = pw - M.left - M.right, ih = h - M.top - M.bottom
          const x = d3.scaleBand<number>().domain(d3.range(1, 13)).range([0, iw]).padding(0.15)
          const pMax = d3.max(months, (m) => m.p ?? 0) || 1
          const yP = d3.scaleLinear().domain([0, pMax]).range([ih, 0]).nice()
          const tv = months.map((m) => m.t).filter((v): v is number => v !== null)
          const yT = d3.scaleLinear().domain([Math.floor((d3.min(tv) ?? 0) - 2), Math.ceil((d3.max(tv) ?? 1) + 2)]).range([ih, 0])
          const seasons = seasonsForSite(meta, site.name)
          const line = d3.line<(typeof months)[number]>().defined((m) => m.t !== null).x((m) => (x(m.m) ?? 0) + x.bandwidth() / 2).y((m) => yT(m.t!))
          return (
            <svg width={pw} height={h} fontFamily={FONT.family}>
              <g transform={`translate(${M.left},${M.top})`}>
                {yP.ticks(3).map((t) => <g key={t}><line x1={0} x2={iw} y1={yP(t)} y2={yP(t)} stroke={INK.grid} /><text x={-4} y={yP(t)} dy="0.32em" textAnchor="end" fontSize={9.5} fill="#4292c6">{t}</text></g>)}
                {yT.ticks(3).map((t) => <text key={t} x={iw + 4} y={yT(t)} dy="0.32em" fontSize={9.5} fill="#c0392b">{t}°</text>)}
                {months.map((m) => (
                  <g key={m.m}>
                    {m.p !== null && <rect x={x(m.m)} width={x.bandwidth()} y={yP(m.p)} height={ih - yP(m.p)} fill="#4292c6" fillOpacity={0.75} />}
                    <rect x={x(m.m)} width={x.bandwidth()} height={ih + 22} fill="transparent"
                      onMouseEnter={(e) => tip.show(e, [`${site.name} · ${d3.timeFormat('%B')(new Date(2023, m.m - 1, 1))}`,
                        `precipitation ${fmt(m.p, 0)} mm/month`, `temperature ${fmt(m.t, 1)} °C`, `RH ${fmt(m.rh, 0)} %`,
                        `season: ${seasons.find((s) => s.months.includes(m.m))?.name.replace(`${site.name} · `, '') ?? '—'}`, `${m.n} days`])}
                      onMouseLeave={tip.hide} />
                  </g>
                ))}
                <path d={line(months) ?? ''} fill="none" stroke="#c0392b" strokeWidth={1.8} pointerEvents="none" />
                {months.map((m) => {
                  const se = seasons.find((s) => s.months.includes(m.m))
                  return <g key={`s${m.m}`} pointerEvents="none">
                    <rect x={(x(m.m) ?? 0) - x.step() * x.paddingInner() / 2} width={x.step()} y={ih + 4} height={7} fill={se?.color ?? '#ddd'} />
                    <text x={(x(m.m) ?? 0) + x.bandwidth() / 2} y={ih + 22} textAnchor="middle" fontSize={9.5} fill={INK.muted}>{MONTHS[m.m - 1]}</text>
                  </g>
                })}
              </g>
            </svg>
          )
        }} />
        {tip.node}
      </div>
      {sites.length > MAX_SITES && <Note>Showing the first {MAX_SITES} of {sites.length} sites; narrow the site pick to see the others.</Note>}
    </ChartFrame>
  )
}
