import { useCallback, useEffect, useMemo, useState } from 'react'
import * as d3 from 'd3'
import { Segmented } from '@/components/ChartFrame'
import { AxisFieldsProvider, type AxisRole } from '@/components/AxisFieldPopover'
import { Scatterplot } from '@/charts/correlation/Scatterplot'
import { WeatherTimeline } from '@/charts/weather/WeatherTimeline'
import { MonthlyClimate } from '@/charts/weather/MonthlyClimate'
import { RainWashout } from '@/charts/weather/RainWashout'
import { HumidityAgreement } from '@/charts/weather/HumidityAgreement'
import { WindRoses } from '@/charts/weather/WindRoses'
import { WeatherCorrelations } from '@/charts/weather/WeatherCorrelations'
import { StationCheck } from '@/charts/weather/StationCheck'
import { SiteWeatherSummary } from '@/charts/weather/SiteWeatherSummary'
import { PwMoisture } from '@/charts/weather/PwMoisture'
import { useExport } from '@/lib/data'
import { useSearchParam } from '@/lib/url'
import { WEATHER_VARS, weatherField, type RoseFile, type StationCompare, type WeatherSource } from '@/lib/weather'
import type { AxesOpts } from '@/lib/axes'
import type { FilterRow, MetaFile } from '@/lib/types'

const ADDIS = 'ETAD'

/**
 * Weather and the filters, driven by the Filters panel's site pick (no picker of its own).
 *
 * One site picked: the analyses the Addis notebooks ran on the airport station
 * (addis_03_meteorology, notebooks/analysis/meteorology/*) for that site: the daily timeline
 * under the filters, its climate against its season calendar, rain washout, the measurement
 * against any weather variable, humidity and method agreement, wind roses by season or PMF
 * source, seasonal correlations, and at Addis the check against the station.
 *
 * Several sites picked: the comparison: which weather goes with the measurement at each site,
 * then climate, washout and wind roses side by side. A site in the comparison table narrows
 * the pick to that site, and the one-site view offers the way back.
 *
 * Every filter carries the weather averaged over its own sampling window
 * (gallery/data/export_weather.py); dates, seasons and hand edits apply as on every tab.
 */
export function WeatherPage({ rows, picked, onPick, meta, field, xField, yField, axes, setField, pmfLabelOfDate }: {
  rows: FilterRow[]
  /** the Filters panel's SPARTAN pick, and a way to set it (absent outside a SPARTAN network) */
  picked: Set<string>
  onPick?: (codes: string[]) => void
  meta: MetaFile; field: string; xField: string; yField: string; axes: AxesOpts
  setField: (f: string) => void
  pmfLabelOfDate: (d: string | null | undefined) => string | null
}) {
  const roses = useExport<RoseFile>('weather/roses.json').data
  const station = useExport<StationCompare>('weather/station_ETAD.json').data

  // the picked sites that have weather, in the Filters panel's order
  const wxRows = useMemo(() => rows.filter((r) => typeof r['Temperature (ERA5)'] === 'number'), [rows])
  const sites = useMemo(() => {
    const n = d3.rollup(wxRows, (v) => v.length, (r) => r.site)
    return meta.sites.filter((s) => n.has(s.name)).map((s) => ({ name: s.name, code: s.code, n: n.get(s.name)! }))
  }, [wxRows, meta.sites])
  const one = sites.length === 1 ? sites[0] : null
  const isAddis = one?.code === ADDIS

  const [sourceParam, setSource] = useSearchParam<string>('wsrc', 'ERA5', ['ERA5', 'station'])
  const source: WeatherSource = sourceParam === 'station' && isAddis ? 'station' : 'ERA5'
  const wx = (k: Parameters<typeof weatherField>[0]) => weatherField(k, source) ?? weatherField(k, 'ERA5')!

  // narrowing the pick from the comparison table remembers the pick to go back to
  const [previous, setPrevious] = useState<string[] | null>(null)
  useEffect(() => { if (sites.length > 1) setPrevious(null) }, [sites.length])
  const openSite = (code: string) => {
    if (!onPick) return
    setPrevious([...picked])
    onPick([code])
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  // the scatter's x is a weather variable of its own; its y is the tab's measurement
  const [wxX, setWxX] = useState(wx('range'))
  // switching source keeps the same quantity on x (Temperature (ERA5) -> Temperature (station))
  useEffect(() => setWxX((cur) => {
    const v = WEATHER_VARS.find((x) => x.era5 === cur || x.station === cur)
    return v ? weatherField(v.key, source) ?? v.era5 : cur
  }), [source])
  const axisValue = useMemo(() => ({ x: wxX, y: field, field }), [wxX, field])
  const setAxis = useCallback((role: AxisRole, f: string) => (role === 'x' ? setWxX(f) : setField(f)), [setField])
  const scatter = (title: string, id: string) => (
    <AxisFieldsProvider meta={meta} value={axisValue} set={setAxis}>
      <Scatterplot rows={wxRows.filter((r) => typeof r[wxX] === 'number')} meta={meta} xField={wxX} yField={field}
        axes={{ ...axes, identity: false, mode: 'Data' }} id={id} title={title} />
    </AxisFieldsProvider>
  )

  if (sites.length === 0) {
    return <div className="panel"><p className="frame-empty">No picked site has weather. Weather is pulled for SPARTAN sites (ERA5 over every filter's sampling window): pick one or more SPARTAN sites in the Filters panel.</p></div>
  }

  if (one) {
    const span = `${d3.min(wxRows, (r) => r.date)} to ${d3.max(wxRows, (r) => r.date)}`
    return (
      <>
        <div className="panel weather-head">
          <div>
            <h2 style={{ margin: 0 }}>{one.name} · weather and the filters</h2>
            <p className="subtitle" style={{ margin: '4px 0 0' }}>
              {one.n} filters, {span}, each with the weather averaged over its own sampling window{isAddis ? ' (ERA5, or the Bole airport station the Addis notebooks used)' : ' (ERA5)'}.
              Pick more sites in the Filters panel to compare them.
            </p>
          </div>
          <div className="weather-head-controls">
            {isAddis && <Segmented label="Weather" value={source} options={['ERA5', 'station'] as WeatherSource[]} onChange={setSource}
              optionLabel={(o) => (o === 'station' ? 'Airport station' : 'ERA5')}
              title="ERA5 reanalysis, or the Bole airport station (Dec 2022 – Sep 2024)" />}
            {previous && previous.length > 1 && onPick && <button type="button" className="btn" onClick={() => { onPick(previous); setPrevious(null) }}
              title="Put the Filters panel's pick back to the sites you were comparing">← Back to comparing {previous.length} sites</button>}
          </div>
        </div>
        <WeatherTimeline rows={wxRows} meta={meta} field={field} site={one} />
        <MonthlyClimate meta={meta} sites={[one]} large id="weather-site-climate" title={`${one.name} · monthly climate against its season calendar`} />
        <RainWashout rows={wxRows} meta={meta} field={field} prcpField={wx('prcp')} id="weather-site-washout" title={`${one.name} · rain washout`} />
        {scatter(`${one.name} · ${field} against the weather`, 'weather-site-scatter')}
        <HumidityAgreement rows={wxRows} meta={meta} xField={xField} yField={yField} rhField={wx('rh')} />
        <PwMoisture rows={wxRows} meta={meta} onePick />
        <WindRoses rows={wxRows} meta={meta} roses={roses} source={source} site={one} pmfLabelOfDate={pmfLabelOfDate}
          splits={isAddis ? ['Season', 'PMF source'] : ['Season']} id="weather-site-roses" title={`${one.name} · wind roses`} />
        <WeatherCorrelations rows={wxRows} meta={meta} source={source} site={one} />
        {isAddis && <StationCheck data={station} />}
      </>
    )
  }

  return (
    <>
      <div className="panel weather-head">
        <div>
          <h2 style={{ margin: 0 }}>Weather and the filters · {sites.length} sites compared</h2>
          <p className="subtitle" style={{ margin: '4px 0 0' }}>
            The sites picked in the Filters panel, on ERA5 averaged over each filter's sampling window. Click a site in the table
            {onPick ? ' (or pick just one in the Filters panel)' : ''} for its timeline, seasonal wind roses and correlations, and the Addis station check.
          </p>
        </div>
      </div>
      <SiteWeatherSummary rows={wxRows} meta={meta} field={field} current={null} onSite={openSite} />
      {scatter(`${field} against the weather — the picked sites`, 'weather-compare-scatter')}
      <MonthlyClimate meta={meta} sites={sites} id="weather-compare-climate" title="Monthly climate and the season calendar — the picked sites" />
      <RainWashout rows={wxRows} meta={meta} field={field} prcpField={weatherField('prcp', 'ERA5')!} id="weather-compare-washout" title="Rain washout — the picked sites" />
      <PwMoisture rows={wxRows} meta={meta} onePick={false} />
      <WindRoses rows={wxRows} meta={meta} roses={roses} source="ERA5" site={null} pmfLabelOfDate={pmfLabelOfDate}
        splits={['Site']} id="weather-compare-roses" title="Wind roses — the picked sites" />
    </>
  )
}
