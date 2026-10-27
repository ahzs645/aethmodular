import { useMemo, useRef } from 'react'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { FacetGrid } from '@/components/FacetGrid'
import { Legend, useLegend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useSearchParam } from '@/lib/url'
import type { MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { CHANNELS, SCREEN_TEXT, channelColor, channelName, instColor, type Channel, type Screen, type Series } from './data'
import { MONTH_STAT_LABEL, MonthPanel, type MonthStat } from './MonthPanel'
import { summarize, type Day } from './stats'

const byMonth = (days: readonly Day[], get: (d: Day) => number | null) =>
  Array.from({ length: 12 }, (_, i) => summarize(days.filter((d) => d.month === i + 1).map(get)))

const PROVENANCE = 'addis_02_temporal_patterns_daily.ipynb Task 1 (monthly pattern by wavelength, monthly bars by season) for every site · gallery/data/export_aethalometer.py'

/**
 * The year at one site: each channel's daily BCc by calendar month, over every year in the span,
 * with the site's season calendar behind (addis_02_daily Task 1, which drew IR and UV for Addis).
 */
export function MonthlyPattern({ s, meta, screen }: { s: Series; meta: MetaFile; screen: Screen }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const [stat, setStat] = useSearchParam<MonthStat>('aeom', 'mean', ['mean', 'median'])
  const leg = useLegend(['Blue', 'Green', 'Red'])
  const series = useMemo(() => CHANNELS.map((c) => ({ key: c, label: channelName(c, meta), color: channelColor(c), months: byMonth(s.days, (d) => d.bc[c]) })), [s.days, meta])
  const years = new Set(s.days.map((d) => d.date.slice(0, 4)))
  return (
    <ChartFrame id="aeth-overview-monthly" title={`${s.inst.site} · black carbon by month — each wavelength over the site's season calendar`}
      subtitle={`Daily BCc of ${s.inst.label} grouped by calendar month (the day each bin's middle falls on), over ${s.days.length} days in ${years.size} year${years.size === 1 ? '' : 's'}: ${MONTH_STAT_LABEL[stat].toLowerCase()} per month. UV and IR shown first, as in the notebook; click a wavelength to add it. The strip under the months is the site's season calendar. Screen: ${SCREEN_TEXT[screen]}.`}
      provenance={PROVENANCE}
      controls={<Segmented label="Statistic" value={stat} options={['mean', 'median'] as MonthStat[]} onChange={setStat} optionLabel={(o) => MONTH_STAT_LABEL[o]} />}>
      <div ref={wrapRef} className="chart-wrap">
        {!s.days.length ? <Empty>No daily bins in this span.</Empty>
          : <MonthPanel series={series.filter((se) => leg.show(se.key)).map((se) => ({ ...se, opacity: leg.dim(se.key) }))} seasons={seasonsForSite(meta, s.inst.site)}
            stat={stat} width={Math.min(width, 900)} height={280} unit="BCc (µg/m³)" />}
        <Legend items={CHANNELS.map((c) => ({ label: c, color: channelColor(c), shape: 'line' as const, detail: `${meta.wavelengths_nm[c]} nm` }))} {...leg.props} />
      </div>
    </ChartFrame>
  )
}

/** The same, one channel, a panel per instrument, each on its own site's calendar. */
export function MonthlyBySite({ series, meta, screen, channel, setChannel }: {
  series: Series[]; meta: MetaFile; screen: Screen; channel: Channel; setChannel: (c: Channel) => void
}) {
  const [stat, setStat] = useSearchParam<MonthStat>('aeom', 'mean', ['mean', 'median'])
  const items = useMemo(() => series.filter((s) => s.days.length).map((s) => ({
    s, months: byMonth(s.days, (d) => d.bc[channel]),
  })), [series, channel])
  return (
    <ChartFrame id="aeth-overview-monthly-sites" title={`Black carbon by month — ${channel} BCc, a panel per instrument on its own season calendar`}
      subtitle={`Daily ${channelName(channel, meta)} BCc grouped by calendar month (the day each bin's middle falls on): ${MONTH_STAT_LABEL[stat].toLowerCase()}. Each panel has its own y-axis and its site's season calendar (the strip under the months), so read the shape of the year, not heights across panels. Screen: ${SCREEN_TEXT[screen]}.`}
      provenance={PROVENANCE}
      controls={<>
        <Segmented label="Wavelength" value={channel} options={CHANNELS} onChange={setChannel} />
        <Segmented label="Statistic" value={stat} options={['mean', 'median'] as MonthStat[]} onChange={setStat} optionLabel={(o) => MONTH_STAT_LABEL[o]} />
      </>}>
      <div className="chart-wrap">
        {!items.length ? <Empty>No daily bins in this span.</Empty> : (
          <FacetGrid items={items} minCell={300} maxCols={3} render={({ s, months }, w) => (
            <MonthPanel series={[{ key: s.inst.code, label: s.inst.label, color: instColor(meta, s.inst), months }]} seasons={seasonsForSite(meta, s.inst.site)}
              stat={stat} width={w} height={210} title={s.inst.code === s.inst.siteCode ? s.inst.site : s.inst.info.site} unit="µg/m³" />
          )} />
        )}
      </div>
    </ChartFrame>
  )
}
