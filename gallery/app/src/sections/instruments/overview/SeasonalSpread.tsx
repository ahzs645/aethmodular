import { useMemo } from 'react'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { IntervalStrip, type IntervalBlock, type IntervalMark } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { fmt } from '@/lib/format'
import { INK } from '@/lib/theme'
import { seasonOnly } from '@/lib/seasons'
import { useSearchParam } from '@/lib/url'
import type { MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { CHANNELS, SCREEN_TEXT, channelColor, channelName, instColor, type Channel, type Screen, type Series } from './data'
import { summarize, type Day, type Summary } from './stats'

/** the 10–90 % range faint, the interquartile range and median bold, the mean hollow */
function marks(s: Summary | null, color: string, k = 1): IntervalMark[] {
  if (!s) return []
  return [
    { mid: null, lo: s.q10 / k, hi: s.q90 / k, color, faint: true },
    { mid: s.median / k, lo: s.q25 / k, hi: s.q75 / k, color },
    { mid: s.mean / k, color, shape: 'hollow', r: 3 },
  ]
}
const statsTip = (title: string, s: Summary, digits = 2, unit = 'µg/m³') => [title, `n = ${s.n} days`,
  `mean ± SD ${fmt(s.mean, digits)} ± ${fmt(s.sd, digits)} ${unit}`, `median ${fmt(s.median, digits)} [IQR ${fmt(s.q25, digits)}–${fmt(s.q75, digits)}]`,
  `10–90 % ${fmt(s.q10, digits)}–${fmt(s.q90, digits)}`, `range ${fmt(s.min, digits)}–${fmt(s.max, digits)}`]

/** the site's seasons in calendar order, then every day */
function seasonGroups(days: readonly Day[], meta: MetaFile, site: string) {
  const seasons = seasonsForSite(meta, site)
  return [
    ...seasons.map((se) => ({ key: se.name, label: seasonOnly(se.name), color: se.color, days: days.filter((d) => d.season?.name === se.name) })),
    { key: 'all', label: 'All days', color: INK.text, days: [...days] },
  ].filter((g) => g.days.length)
}

const KEY = [
  { label: 'median and interquartile range', color: INK.text, kind: 'dot' as const },
  { label: 'mean', color: INK.text, kind: 'hollow' as const },
  { label: '10–90 % of days', color: INK.muted, kind: 'line' as const },
]
const PROVENANCE = 'addis_02_temporal_patterns_daily.ipynb Task 3 (seasonal statistics and boxplot); Multi_Site_Analysis.ipynb §3 and _FollowUp §7a (distribution by wavelength) · gallery/data/export_aethalometer.py'

/**
 * One site's daily BCc by season and wavelength (addis_02_daily Task 3's seasonal boxplot and
 * statistics; Multi_Site_Analysis §3 and _FollowUp 7a's boxplots by wavelength).
 */
export function SeasonalSpread({ s, meta, screen }: { s: Series; meta: MetaFile; screen: Screen }) {
  const blocks = useMemo<IntervalBlock[]>(() => seasonGroups(s.days, meta, s.inst.site).map((g) => ({
    key: g.key, title: g.label, color: g.color === INK.text ? undefined : g.color,
    rows: CHANNELS.map((c) => {
      const st = summarize(g.days.map((d) => d.bc[c]))
      return { key: c, label: channelName(c, meta), labelColor: channelColor(c), marks: [marks(st, channelColor(c))], note: st ? `n=${st.n}` : '', tip: st ? statsTip(`${g.label} · ${channelName(c, meta)}`, st) : undefined }
    }),
  })), [s.days, meta, s.inst.site])
  return (
    <ChartFrame id="aeth-overview-seasons" title={`${s.inst.site} · black carbon by season — each wavelength's daily distribution`}
      subtitle={`Daily BCc of ${s.inst.label} in each of the site's seasons and over all days: median and interquartile range, mean (open circle) and the 10–90 % range of days. Hover a row for the notebook's statistics (n, mean ± SD, median and IQR, range). Screen: ${SCREEN_TEXT[screen]}.`}
      provenance={PROVENANCE}>
      <div className="chart-wrap">
        {!s.days.length ? <Empty>No daily bins in this span.</Empty> : <IntervalStrip blocks={blocks} panels={[{ label: 'BCc (µg/m³)' }]} labelWidth={130} rowHeight={17} noteWidth={56} />}
        <Key items={KEY} />
      </div>
    </ChartFrame>
  )
}

type Scale = 'abs' | 'rel'
/** The sites side by side: one channel's daily distribution by season, a block per instrument. */
export function SeasonalSpreadBySite({ series, meta, screen, channel, setChannel }: {
  series: Series[]; meta: MetaFile; screen: Screen; channel: Channel; setChannel: (c: Channel) => void
}) {
  const [scale, setScale] = useSearchParam<Scale>('aeos', 'abs', ['abs', 'rel'])
  const blocks = useMemo<IntervalBlock[]>(() => series.filter((s) => s.days.length).map((s) => {
    const all = summarize(s.days.map((d) => d.bc[channel]))
    const k = scale === 'rel' && all ? all.median : 1
    return {
      key: s.inst.code, title: s.inst.code === s.inst.siteCode ? s.inst.site : s.inst.label, color: instColor(meta, s.inst),
      rows: seasonGroups(s.days, meta, s.inst.site).map((g) => {
        const st = summarize(g.days.map((d) => d.bc[channel]))
        return { key: g.key, label: g.label, labelColor: g.color === INK.text ? undefined : g.color, marks: [marks(st, instColor(meta, s.inst), k)], note: st ? `n=${st.n}` : '',
          tip: st ? statsTip(`${s.inst.site} · ${g.label} · ${channel}`, st) : undefined }
      }),
    }
  }), [series, meta, channel, scale])
  return (
    <ChartFrame id="aeth-overview-seasons-sites" title={`Black carbon by season — ${channel} BCc at each instrument, on its own season calendar`}
      subtitle={`Daily ${channelName(channel, meta)} BCc in each site's seasons and over all days: median and interquartile range, mean (open circle), 10–90 % range. ${scale === 'rel' ? "Divided by the instrument's median over all days, so the seasonal swing compares across sites whatever their level. " : ''}Screen: ${SCREEN_TEXT[screen]}.`}
      provenance={PROVENANCE}
      controls={<>
        <Segmented label="Wavelength" value={channel} options={CHANNELS} onChange={setChannel} />
        <Segmented label="Scale" value={scale} options={['abs', 'rel'] as Scale[]} onChange={setScale} optionLabel={(o) => (o === 'abs' ? 'µg/m³' : 'Relative to median')} />
      </>}>
      <div className="chart-wrap">
        {!blocks.length ? <Empty>No daily bins in this span.</Empty>
          : <IntervalStrip blocks={blocks} panels={[{ label: scale === 'rel' ? `${channel} BCc / instrument's median` : `${channel} BCc (µg/m³)`, reference: scale === 'rel' ? 1 : undefined }]} labelWidth={170} rowHeight={17} noteWidth={56} />}
        <Key items={KEY} />
      </div>
    </ChartFrame>
  )
}
