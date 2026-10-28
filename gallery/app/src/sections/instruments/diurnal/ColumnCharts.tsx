import { useMemo, useRef } from 'react'
import { useColourScheme } from '@/lib/colourSchemes'
import { ColorLegend } from '@/components/ColorLegend'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { HeatCell } from '@/components/HeatCell'
import { Key } from '@/components/Key'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/format'
import { FONT, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { SeasonMeta, SiteMeta } from '@/lib/types'
import { dates, HOURS, hourLabel, hourRuns, seasonsOf, unscale, utcLabel, type ColumnFile, type ColumnIndex } from './data'
import { cycleSeries, dayGroups, HourPlot, inGroup } from './HourCycle'

const PROVENANCE = 'notebooks/analysis/meteorology/aeronet_addis_diagnostics.ipynb and aeronet_addis_deep_dive.ipynb (Addis only), research/ftir_ec_phase3/AERONET_SPARTAN_2026-08-22.md (retrieval hours by city); here every SPARTAN site with a station'

/**
 * What a 24 h filter's day looks like to the sun photometer, site by site: the share of each site's
 * direct-sun retrievals in each local hour, the almucantar inversion hours (dots) that the absorption
 * products come from, against the 24 hours a filter samples.
 */
export function DaylightWindow({ index, sites, focus, onFocus }: {
  index: ColumnIndex; sites: SiteMeta[]; focus: string | null; onFocus: (code: string) => void
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const rows = sites.filter((s) => index.sites[s.code])
  // on a narrow screen the notes go and the cells shrink, so the grid fits without scrolling
  const narrow = width > 0 && width < 720
  const labelW = narrow ? 104 : 170, noteW = narrow ? 0 : 210, head = 26, rowH = 22
  const cell = Math.max(narrow ? 9 : 12, Math.min(30, Math.floor((width - labelW - noteW) / 24)))
  const w = labelW + 24 * cell + noteW
  const shares = rows.map((s) => { const c = index.sites[s.code].hour_counts.aod, t = d3.sum(c) || 1; return c.map((n) => n / t) })
  const max = d3.max(shares.flat()) ?? 1
  const colours = useColourScheme('diurnal-daylight', 'sequential')
  const colour = d3.scaleSequential(d3.interpolateRgbBasis(colours.name === 'Gallery' && !colours.reverse ? colours.stops.slice(1) : colours.stops)).domain([0, max])
  const y = (i: number) => head + (i + 1) * rowH
  return (
    <ChartFrame id="diurnal-daylight" title="What the photometer sees of a 24 h filter — retrieval hours by site"
      subtitle={`For each site in view, the share of its AERONET direct-sun retrievals (${index.level}) falling in each local standard hour; a dot marks the almucantar inversion hours (at least 2 % of the site's inversions), which carry AAOD and SSA. The top row is the filter: it samples all 24. The note gives the median number of hours with a retrieval on a day that has any. Click a site to look at it below.`}
      provenance={PROVENANCE}>
      <div ref={wrapRef} className="chart-wrap" style={{ overflowX: 'auto' }}>
        {rows.length === 0 ? <Empty>No site in view has an AERONET station. Pick SPARTAN sites in the Filters panel.</Empty> : (
          <svg width={w} height={y(rows.length) + 6} fontFamily={FONT.family}>
            {HOURS.filter((h) => h % 3 === 0).map((h) => <text key={h} x={labelW + h * cell + cell / 2} y={head - 10} textAnchor="middle" fontSize={10} fill={INK.muted}>{String(h).padStart(2, '0')}</text>)}
            <text x={labelW - 8} y={head + rowH / 2} dy="0.35em" textAnchor="end" fontSize={12} fill={INK.text}>24 h filter</text>
            {HOURS.map((h) => <HeatCell key={h} x={labelW + h * cell + 1} y={head + 1} width={cell - 2} height={rowH - 4} fill={INK.neutral} />)}
            {!narrow && <text x={labelW + 24 * cell + 8} y={head + rowH / 2} dy="0.35em" fontSize={11} fill={INK.muted}>samples every hour</text>}
            {rows.map((s, i) => {
              const info = index.sites[s.code], inv = new Set(info.inv_hours), on = s.code === focus
              return (
                <g key={s.code} transform={`translate(0,${y(i)})`} style={{ cursor: 'pointer' }} onClick={() => onFocus(s.code)}>
                  <rect x={0} width={w} height={rowH - 2} fill={on ? INK.tint : 'transparent'} />
                  <circle cx={10} cy={rowH / 2 - 1} r={5} fill={s.color} />
                  <text x={22} y={rowH / 2} dy="0.1em" fontSize={narrow ? 11 : 12} fontWeight={on ? 650 : 400} fill={INK.text}>{s.name}</text>
                  {HOURS.map((h) => {
                    const v = shares[i][h]
                    return <HeatCell key={h} x={labelW + h * cell + 1} y={1} width={cell - 2} height={rowH - 4} fill={v > 0 ? colour(v) : null}
                      onEnter={(e) => tip.show(e, [`${s.name} · ${hourLabel(h)} (${utcLabel(info.offset)})`, `${fmt(100 * v, 1)} % of ${info.retrievals.toLocaleString()} retrievals`,
                        `${info.hour_counts.inv[h]} inversions${inv.has(h) ? ' (an inversion hour)' : ''}`, `${fmt(info.hours_per_day, 0)} h with a retrieval on a typical day`])} onLeave={tip.hide}>
                      {inv.has(h) && <circle cx={labelW + h * cell + cell / 2} cy={rowH / 2 - 1} r={narrow ? 1.8 : 2.4} fill={INK.text} pointerEvents="none" />}
                    </HeatCell>
                  })}
                  {!narrow && <text x={labelW + 24 * cell + 8} y={rowH / 2} dy="0.1em" fontSize={11} fill={INK.muted}>
                    {fmt(info.hours_per_day, 0)} h/day · inversions {hourRuns(info.inv_hours) || '—'}
                  </text>}
                </g>
              )
            })}
          </svg>
        )}
        {tip.node}
      </div>
      {rows.length > 0 && <Key items={[{ label: 'inversion hour', color: INK.text, kind: 'dot' }]}>
        <ColorLegend scale={colour} scheme={colours.scheme} label="share of the site's retrievals in the hour" format={(v) => `${fmt(100 * v, 0)} %`} />
      </Key>}
    </ChartFrame>
  )
}

type ColumnQuantity = 'aod' | 'ae' | 'fmf' | 'share'
const QUANTITY: Record<ColumnQuantity, { label: string; axis: string; digits: number }> = {
  aod: { label: 'AOD 500', axis: 'AOD 500 nm', digits: 3 },
  ae: { label: 'Ångström 440–870', axis: 'Ångström exponent 440–870 nm', digits: 2 },
  fmf: { label: 'Fine fraction', axis: 'Fine-mode fraction 500 nm (SDA)', digits: 2 },
  share: { label: 'Retrievals', axis: 'Share of retrievals (%)', digits: 1 },
}

/** The column's own hour-of-day cycle at one site: AOD, Ångström exponent, fine fraction or retrieval share. */
export function ColumnCycle({ file, site, seasons, split }: { file: ColumnFile; site: SiteMeta; seasons: readonly SeasonMeta[]; split: boolean }) {
  const [q, setQ] = useSearchParam<ColumnQuantity>('dn.col', 'aod', Object.keys(QUANTITY))
  const t = file.aod
  const days = useMemo(() => dates(file.first, t), [file, t])
  const season = useMemo(() => seasonsOf(days, seasons), [days, seasons])
  const groups = dayGroups(split, seasons, season)
  const spec = QUANTITY[q]
  const series = useMemo(() => {
    if (q === 'share') return groups.map((g) => {
      const n = HOURS.map(() => 0)
      t.h.forEach((h, i) => { if (inGroup(g, season[i])) n[h] += t.n[i] })
      const total = d3.sum(n) || 1
      return { id: g.name, label: g.name, color: g.color, group: g.name, curve: 'monotone' as const,
        points: HOURS.map((h) => ({ x: h, y: n[h] ? (100 * n[h]) / total : 0, tip: [g.name, hourLabel(h), `${fmt((100 * n[h]) / total, 1)} % of ${total.toLocaleString()} retrievals`] })) }
    })
    const values = q === 'aod' ? unscale(t.aod500, file.scale.aod500) : q === 'ae' ? unscale(t.ae, file.scale.ae) : unscale(t.fmf, file.scale.fmf)
    return cycleSeries({ id: q, label: spec.label, hours: t.h, values, seasons: season, groups, digits: spec.digits })
  }, [q, t, file.scale, season, groups, spec])
  const any = series.some((s) => s.points.some((p) => p.y != null))
  return (
    <ChartFrame id="diurnal-column-cycle" title={`${site.name} · the column through the day — ${spec.label} by local hour`}
      subtitle={`Each day-hour with a retrieval at ${file.station.replace(/_/g, ' ')} (${file.first} to ${file.last}, ${file.days} days) is the mean of its retrievals; the line is the median of those day-hours, the band the interquartile range, at hours with at least 5. Shaded: the inversion hours (${hourRuns(file.inv_hours)}). Seasons follow the site's calendar in the Filters panel. Retrievals: each hour's share of the season's direct-sun retrievals.`}
      provenance={PROVENANCE}
      controls={<Segmented label="Show" value={q} options={Object.keys(QUANTITY) as ColumnQuantity[]} onChange={setQ} optionLabel={(k) => QUANTITY[k].label} />}>
      {any ? <HourPlot series={series} yLabel={spec.axis} offset={file.offset} shade={file.inv_hours} digits={spec.digits} include={q === 'share' ? [0] : undefined} />
        : <Empty>No hour has five day-hours of {spec.label} at this site.</Empty>}
      {split && <Key items={groups.map((g) => ({ label: g.name, color: g.color, kind: 'line' as const }))} />}
    </ChartFrame>
  )
}
