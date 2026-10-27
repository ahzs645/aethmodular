import { useMemo } from 'react'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { IntervalStrip, type IntervalBlock, type IntervalMark, type IntervalRow } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { fmt, signed } from '@/lib/format'
import { INK } from '@/lib/theme'
import { seasonOnly } from '@/lib/seasons'
import type { MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { CHANNELS, DOW, SCREEN_TEXT, channelName, instColor, type Channel, type Screen, type Series } from './data'
import { summarize, weekendEffect, type Day, type WeekendEffect } from './stats'

const weekdayColor = () => INK.accent
const weekendColor = () => INK.negative
const pText = (p: number) => (p < 0.001 ? 'p < 0.001' : `p = ${fmt(p, 3)}`)
const effectMark = (w: WeekendEffect | null, color: string): IntervalMark[] => (w ? [{ mid: w.change, lo: w.lo, hi: w.hi, color, shape: 'diamond', r: 5 }] : [])
const effectTip = (title: string, w: WeekendEffect, channel: string) => [title,
  `${channel}: Mon–Fri ${fmt(w.weekday.mean, 2)} µg/m³ (n=${w.weekday.n}), Sat–Sun ${fmt(w.weekend.mean, 2)} µg/m³ (n=${w.weekend.n})`,
  `weekend ${signed(w.change, 1)} % [95 % ${signed(w.lo, 1)} to ${signed(w.hi, 1)}], Welch ${pText(w.p)}`,
  'days are autocorrelated, so the interval and p are optimistic']

function groups(days: readonly Day[], meta: MetaFile, site: string) {
  return [
    ...seasonsForSite(meta, site).map((se) => ({ key: se.name, label: seasonOnly(se.name), color: se.color as string | undefined, days: days.filter((d) => d.season?.name === se.name) })),
    { key: 'all', label: 'All days', color: undefined, days: [...days] },
  ].filter((g) => g.days.length)
}
const PROVENANCE = 'addis_02_temporal_patterns_daily.ipynb Task 2 (weekend effect by season, day-of-week boxplots) for every site · gallery/data/export_aethalometer.py'

/**
 * Weekdays and weekends at one site (addis_02_daily Task 2): each day of the week's daily BCc by
 * season, and the weekend's mean against the weekdays' (the notebook's "reduction", with a Welch
 * test where it used Student's).
 */
export function WeeklyPattern({ s, meta, screen, channel, setChannel }: {
  s: Series; meta: MetaFile; screen: Screen; channel: Channel; setChannel: (c: Channel) => void
}) {
  const blocks = useMemo<IntervalBlock[]>(() => groups(s.days, meta, s.inst.site).map((g) => {
    const get = (d: Day) => d.bc[channel]
    const rows: IntervalRow[] = DOW.map((name, i) => {
      const st = summarize(g.days.filter((d) => d.dow === i).map(get))
      const c = i >= 5 ? weekendColor() : weekdayColor()
      return { key: name, label: name, labelColor: c, note: st ? `n=${st.n}` : '',
        marks: [st ? [{ mid: st.median, lo: st.q25, hi: st.q75, color: c }, { mid: st.mean, color: c, shape: 'hollow' as const, r: 3 }] : [], []],
        tip: st ? [`${g.label} · ${name}`, `n = ${st.n} days`, `median ${fmt(st.median, 2)} [IQR ${fmt(st.q25, 2)}–${fmt(st.q75, 2)}] µg/m³`, `mean ${fmt(st.mean, 2)} ± ${fmt(st.sd, 2)}`] : undefined }
    })
    const w = weekendEffect(g.days, get)
    rows.push({ key: 'effect', label: 'Weekend vs weekdays', strong: true, marks: [[], effectMark(w, INK.text)], note: w ? pText(w.p) : '',
      tip: w ? effectTip(`${g.label} · weekend vs weekdays`, w, channel) : undefined })
    return { key: g.key, title: g.label, color: g.color, rows }
  }), [s.days, meta, s.inst.site, channel])
  return (
    <ChartFrame id="aeth-overview-weekly" title={`${s.inst.site} · weekdays and weekends — ${channel} BCc by day of the week and season`}
      subtitle={`Left: daily ${channelName(channel, meta)} BCc of ${s.inst.label} on each day of the week (the day each bin's middle falls on), by season: median and interquartile range, mean (open circle). Right: the weekend's mean against the weekdays', as a percentage of the weekday mean, with a 95 % interval from the Welch standard error. The notebook reported the same difference as a reduction with Student's t-test; daily values are autocorrelated, so both the interval and p are optimistic. Screen: ${SCREEN_TEXT[screen]}.`}
      provenance={PROVENANCE}
      controls={<Segmented label="Wavelength" value={channel} options={CHANNELS} onChange={setChannel} />}>
      <div className="chart-wrap">
        {!blocks.length ? <Empty>No daily bins in this span.</Empty> : (
          <IntervalStrip blocks={blocks} labelWidth={150} rowHeight={16} noteWidth={74} minPanelWidth={220}
            panels={[{ label: `${channel} BCc (µg/m³)`, heading: 'By day of the week' }, { label: 'Weekend − weekdays (% of weekday mean)', reference: 0, heading: 'Weekend effect', format: (v) => `${v > 0 ? '+' : ''}${Math.round(v)}` }]} />
        )}
        <Key items={[{ label: 'Mon–Fri', color: weekdayColor() }, { label: 'Sat–Sun', color: weekendColor() }, { label: 'mean', color: INK.text, kind: 'hollow' }, { label: 'weekend effect, 95 % interval', color: INK.text, kind: 'diamond' }]} />
      </div>
    </ChartFrame>
  )
}

/** The weekend effect at every instrument, by season. */
export function WeeklyBySite({ series, meta, screen, channel, setChannel }: {
  series: Series[]; meta: MetaFile; screen: Screen; channel: Channel; setChannel: (c: Channel) => void
}) {
  const blocks = useMemo<IntervalBlock[]>(() => series.filter((s) => s.days.length).map((s) => ({
    key: s.inst.code, title: s.inst.code === s.inst.siteCode ? s.inst.site : s.inst.label, color: instColor(meta, s.inst),
    rows: groups(s.days, meta, s.inst.site).map((g) => {
      const w = weekendEffect(g.days, (d) => d.bc[channel])
      return { key: g.key, label: g.label, labelColor: g.color, marks: [effectMark(w, instColor(meta, s.inst))], note: w ? pText(w.p) : 'too few days',
        tip: w ? effectTip(`${s.inst.site} · ${g.label}`, w, channel) : undefined }
    }),
  })), [series, meta, channel])
  return (
    <ChartFrame id="aeth-overview-weekly-sites" title={`Weekends against weekdays — ${channel} BCc at each instrument, by season`}
      subtitle={`The weekend's mean daily ${channelName(channel, meta)} BCc against the weekdays' (Mon–Fri), as a percentage of the weekday mean, in each site's seasons and over all days, with a 95 % interval from the Welch standard error (optimistic: days are autocorrelated). Days are dated by the middle of their 24 h bin. Screen: ${SCREEN_TEXT[screen]}.`}
      provenance={PROVENANCE}
      controls={<Segmented label="Wavelength" value={channel} options={CHANNELS} onChange={setChannel} />}>
      <div className="chart-wrap">
        {!blocks.length ? <Empty>No daily bins in this span.</Empty>
          : <IntervalStrip blocks={blocks} labelWidth={170} rowHeight={18} noteWidth={74} panels={[{ label: 'Weekend − weekdays (% of weekday mean)', reference: 0, format: (v) => `${v > 0 ? '+' : ''}${Math.round(v)}` }]} />}
      </div>
    </ChartFrame>
  )
}
