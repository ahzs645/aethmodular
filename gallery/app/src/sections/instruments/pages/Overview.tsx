import { useEffect, useMemo, useState } from 'react'
import { Empty, Note, Segmented, Select } from '@/components/ChartFrame'
import { useSearchParam } from '@/lib/url'
import type { SectionPageProps } from '../../types'
import { CHANNELS, SCREEN_LABEL, useInstruments, useSeries, type Channel, type Screen, type Span } from '../overview/data'
import { DailyTimeline } from '../overview/DailyTimeline'
import { MonthlyBySite, MonthlyPattern } from '../overview/MonthlyPattern'
import { SeasonalSpread, SeasonalSpreadBySite } from '../overview/SeasonalSpread'
import { SpectralShape } from '../overview/SpectralShape'
import { WeeklyBySite, WeeklyPattern } from '../overview/WeeklyPattern'
import { Episodes } from '../overview/Episodes'
import { FilterMethods } from '../overview/FilterMethods'
import { Completeness } from '../overview/Completeness'
import { SitesTimeline } from '../overview/SitesTimeline'
import { SiteTable } from '../overview/SiteTable'

const SCREENS: Screen[] = ['none', 'neg', '3sd']
const SPANS: Span[] = ['filters', 'record']

/**
 * Sub-page 'overview' of the Aethalometer tab: the MA350 daily records themselves, driven by the
 * Filters panel's site pick like the Weather and AERONET tabs.
 *
 * One site: its daily BCc at five wavelengths over the filters (Multi_Site_Analysis, _Fixed,
 * _FollowUp), the year by month and season, the spectral shape, weekdays and weekends, high days
 * and day-to-day change (addis_02_temporal_patterns_daily, generalised from Addis to every site),
 * the filters three ways (addis_02.5) and how complete the record is. At Addis, either instrument.
 *
 * Several sites: the instruments side by side: a summary table, one time axis, completeness, the
 * year on each site's own calendar, seasonal spread, spectral shape and the weekend effect.
 *
 * Every chart reads the daily files of gallery/data/export_aethalometer.py (legacy aggregates, bins
 * labelled at their start: docs/matched-sample-audit-2026-09-10.md); the page's screen and span
 * apply to all of them.
 */
export function Overview({ rows, meta, picked, onPick }: SectionPageProps) {
  const instruments = useInstruments(meta)
  const [screen, setScreen] = useSearchParam<Screen>('aeoscr', 'neg', SCREENS)
  const [span, setSpan] = useSearchParam<Span>('aeospan', 'filters', SPANS)
  const [channel, setChannel] = useSearchParam<Channel>('aeoch', 'IR', CHANNELS)
  const [instPick, setInstPick] = useSearchParam<string>('aeoi', '')

  // the sites in view that have an instrument, in meta order
  const sites = useMemo(() => {
    const inView = new Set(rows.map((r) => r.site))
    return meta.sites.filter((s) => inView.has(s.name) && instruments?.some((i) => i.site === s.name))
  }, [rows, meta.sites, instruments])
  const one = sites.length === 1 ? sites[0] : null
  const atSite = useMemo(() => (instruments ?? []).filter((i) => sites.some((s) => s.name === i.site)), [instruments, sites])
  const inst = one ? atSite.find((i) => i.code === instPick) ?? atSite.find((i) => i.code === one.code) ?? atSite[0] : null
  const { series, loading, failed } = useSeries(one ? (inst ? [inst] : []) : atSite, meta, rows, screen, span)
  // completeness at one site compares its instruments (Addis has two)
  const { series: siteSeries } = useSeries(one ? atSite : [], meta, rows, screen, span)

  // narrowing the pick from the comparison table remembers the pick to go back to
  const [previous, setPrevious] = useState<string[] | null>(null)
  useEffect(() => { if (sites.length > 1) setPrevious(null) }, [sites.length])
  const openSite = onPick ? (code: string) => {
    setPrevious([...picked])
    onPick([code])
    window.scrollTo({ top: 0, behavior: 'smooth' })
  } : undefined

  if (!instruments) return <div className="panel"><Empty>Loading the aethalometer instruments…</Empty></div>
  if (!sites.length) {
    const names = [...new Set(instruments.map((i) => i.site))].join(', ')
    return <div className="panel"><p className="frame-empty">No site in view has an aethalometer. The MA350 daily records cover {names}: pick one or more of them in the Filters panel.</p></div>
  }

  const s = series[0]
  const dropped = series.flatMap((x) => CHANNELS.map((c) => x.dropped[c].neg + x.dropped[c].high)).reduce((a, b) => a + b, 0)
  const controls = (
    <div className="weather-head-controls">
      {one && atSite.length > 1 && <Select label="Instrument" value={inst?.code ?? ''} options={atSite.map((i) => i.code)} onChange={setInstPick}
        optionLabel={(c) => atSite.find((i) => i.code === c)?.label ?? c} title="The Addis filter site's MA350, or the second instrument at AAU (Jacros)" />}
      <Segmented label="Days" value={span} options={SPANS} onChange={setSpan} optionLabel={(o) => (o === 'filters' ? "Filters' span" : 'Whole record')}
        title="The days inside the span of the site's filters in view (±1 day; the date brush narrows it), or the instrument's whole daily file" />
      <Segmented label="Screen" value={screen} options={SCREENS} onChange={setScreen} optionLabel={(o) => SCREEN_LABEL[o]}
        title="Keep every daily value, drop negative BCc, or also drop values above the channel's mean + 3 SD as addis_02 did" />
      {previous && previous.length > 1 && onPick && <button type="button" className="btn" onClick={() => { onPick(previous); setPrevious(null) }}
        title="Put the Filters panel's pick back to the sites you were comparing">← Back to comparing {previous.length} sites</button>}
    </div>
  )
  const status = <>
    {loading && <Note>Loading the daily files…</Note>}
    {failed.length > 0 && <Note>No daily file could be read for {failed.join(', ')}.</Note>}
    {dropped > 0 && <Note>The screen dropped {dropped} daily channel values in this view (negative{screen === '3sd' ? ' or above mean + 3 SD' : ''}).{series.some((x) => x.inst.siteCode === 'ETAD') ? " At Addis the legacy files' UV channel goes negative on many days and spikes to hundreds of µg/m³ on others: read UV there with care (the addis_02 screen removes the worst)." : ''}</Note>}
  </>

  if (one && inst) {
    return (
      <>
        <div className="panel weather-head">
          <div>
            <h2 style={{ margin: 0 }}>{one.name} · the aethalometer's daily record</h2>
            <p className="subtitle" style={{ margin: '4px 0 0' }}>
              {inst.label}: {inst.info.bins} daily bins from {inst.info.first} to {inst.info.last} ({inst.info.file}), each labelled at its start;
              {s ? ` ${s.days.length} in the ${span === 'filters' ? "span of the site's filters in view" : 'whole record'}.` : ''} The legacy files are daily aggregates whose matching
              to the filters was never verified (processed_sites/README.md); their bins start at {Object.keys(inst.info.bin_start_hours).map((h) => `${h}:00`).join(' / ')}.
              {' '}Pick more sites in the Filters panel to compare them.
            </p>
          </div>
          {controls}
        </div>
        {status}
        {s && <>
          <DailyTimeline s={s} meta={meta} screen={screen} />
          <MonthlyPattern s={s} meta={meta} screen={screen} />
          <SeasonalSpread s={s} meta={meta} screen={screen} />
          <SpectralShape series={[s]} meta={meta} screen={screen} by="season" />
          <WeeklyPattern s={s} meta={meta} screen={screen} channel={channel} setChannel={setChannel} />
          <Episodes s={s} meta={meta} screen={screen} channel={channel} setChannel={setChannel} />
          <FilterMethods s={s} meta={meta} />
          <Completeness series={siteSeries} meta={meta} />
        </>}
      </>
    )
  }

  return (
    <>
      <div className="panel weather-head">
        <div>
          <h2 style={{ margin: 0 }}>The aethalometers · {sites.length} sites compared</h2>
          <p className="subtitle" style={{ margin: '4px 0 0' }}>
            The MA350 daily records at the sites in view ({atSite.length} instruments: at Addis also Jacros, the second instrument at AAU), each on its own
            season calendar. {openSite ? 'Click a site in the table (or pick just one in the Filters panel)' : 'Pick just one site in the Filters panel'} for its
            wavelengths, months, weekdays and filters.
          </p>
        </div>
        {controls}
      </div>
      {status}
      <SiteTable series={series} meta={meta} screen={screen} onSite={openSite} />
      <SitesTimeline series={series} meta={meta} screen={screen} channel={channel} setChannel={setChannel} />
      <Completeness series={series} meta={meta} />
      <MonthlyBySite series={series} meta={meta} screen={screen} channel={channel} setChannel={setChannel} />
      <SeasonalSpreadBySite series={series} meta={meta} screen={screen} channel={channel} setChannel={setChannel} />
      <SpectralShape series={series} meta={meta} screen={screen} by="site" />
      <WeeklyBySite series={series} meta={meta} screen={screen} channel={channel} setChannel={setChannel} />
    </>
  )
}
