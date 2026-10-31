import { useCallback, useEffect, useMemo, useState } from 'react'
import * as d3 from 'd3'
import { SidebarControls } from '@/lib/pageSidebar'
import { Segmented } from '@/components/ChartFrame'
import { AxisFieldsProvider, type AxisRole } from '@/components/AxisFieldPopover'
import { Scatterplot } from '@/charts/correlation/Scatterplot'
import { WeatherTimeline } from '@/charts/weather/WeatherTimeline'
import { MonthlyClimate } from '@/charts/weather/MonthlyClimate'
import { RainWashout } from '@/charts/weather/RainWashout'
import { HumidityAgreement } from '@/charts/weather/HumidityAgreement'
import { WindRoses } from '@/charts/weather/WindRoses'
import { PollutionRose } from '@/charts/weather/PollutionRose'
import { PolarPlot } from '@/charts/weather/PolarPlot'
import { WindSpeedConcentration } from '@/charts/weather/WindSpeedConcentration'
import { WeatherCorrelations } from '@/charts/weather/WeatherCorrelations'
import { StationCheck } from '@/charts/weather/StationCheck'
import { SiteWeatherSummary } from '@/charts/weather/SiteWeatherSummary'
import { PwMoisture } from '@/charts/weather/PwMoisture'
import { useExport } from '@/lib/data'
import { useSearchParam } from '@/lib/url'
import { useSiteFocus } from '@/lib/siteFocus'
import { WEATHER_VARS, weatherField, type RoseFile, type StationCompare, type WeatherSource } from '@/lib/weather'
import type { AxesOpts } from '@/lib/axes'
import type { FilterRow, MetaFile } from '@/lib/types'

const ADDIS = 'ETAD'

/**
 * Weather and the filters, driven by the Samples panel's site pick (no picker of its own).
 *
 * One site picked: the analyses the Addis notebooks ran on the airport station
 * (addis_03_meteorology, notebooks/analysis/meteorology/*) for that site: the daily timeline
 * under the filters, its climate against its season calendar, rain washout, the measurement
 * against any weather variable, humidity and method agreement, wind roses by season or PMF
 * source, the measurement by wind (pollution rose, polar plot, wind speed), seasonal
 * correlations, and at Addis the check against the station.
 *
 * Several sites picked: the comparison: which weather goes with the measurement at each site,
 * then climate, washout, wind roses and the measurement by wind side by side. The page tab's
 * Analysis choice (lib/siteFocus, `sfocus`) or a site in the comparison table opens one of them in the
 * one-site view without changing the pick, and the one-site view offers the way back.
 *
 * Every filter carries the weather averaged over its own sampling window
 * (gallery/data/export_weather.py); dates, seasons and hand edits apply as on every tab.
 */
export function WeatherPage({ rows, meta, field, xField, yField, axes, setField, pmfLabelOfDate }: {
  rows: FilterRow[]
  meta: MetaFile; field: string; xField: string; yField: string; axes: AxesOpts
  setField: (f: string) => void
  pmfLabelOfDate: (d: string | null | undefined) => string | null
}) {
  const roses = useExport<RoseFile>('weather/roses.json').data
  const station = useExport<StationCompare>('weather/station_ETAD.json').data

  // the picked sites that have weather, in the Samples panel's order
  const wxRows = useMemo(() => rows.filter((r) => typeof r['Temperature (ERA5)'] === 'number'), [rows])
  const sites = useMemo(() => {
    const n = d3.rollup(wxRows, (v) => v.length, (r) => r.site)
    return meta.sites.filter((s) => n.has(s.name)).map((s) => ({ name: s.name, code: s.code, n: n.get(s.name)! }))
  }, [wxRows, meta.sites])
  // several sites picked: the comparison, or one of them on its own (the page tab's analysis choice)
  const focus = useSiteFocus(sites)
  const one = focus.site ?? (sites.length === 1 ? sites[0] : null)
  const oneRows = useMemo(() => (focus.site ? wxRows.filter((r) => r.site === focus.site!.name) : wxRows), [wxRows, focus.site])
  const isAddis = one?.code === ADDIS

  const [sourceParam, setSource] = useSearchParam<string>('wsrc', 'ERA5', ['ERA5', 'station'])
  const source: WeatherSource = sourceParam === 'station' && isAddis ? 'station' : 'ERA5'
  const wx = (k: Parameters<typeof weatherField>[0]) => weatherField(k, source) ?? weatherField(k, 'ERA5')!

  // a site in the comparison table opens its one-site view; the pick stays, so going back is one click
  const openSite = (code: string) => {
    focus.set(code)
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
  // stable between renders: a new rows array or axes object would refit the scatter and bootstrap its bands again
  const scatterRows = useMemo(() => oneRows.filter((r) => typeof r[wxX] === 'number'), [oneRows, wxX])
  const scatterAxes = useMemo<AxesOpts>(() => ({ ...axes, identity: false, mode: 'Data' }), [axes])
  const scatter = (title: string, id: string) => (
    <AxisFieldsProvider meta={meta} value={axisValue} set={setAxis}>
      <Scatterplot rows={scatterRows} meta={meta} xField={wxX} yField={field} axes={scatterAxes} id={id} title={title} />
    </AxisFieldsProvider>
  )

  if (sites.length === 0) {
    return <div className="panel"><p className="frame-empty">No picked site has weather. Weather is pulled for SPARTAN sites (ERA5 over every filter's sampling window): pick one or more SPARTAN sites in the Samples panel.</p></div>
  }

  if (one) {
    const span = `${d3.min(oneRows, (r) => r.date)} to ${d3.max(oneRows, (r) => r.date)}`
    return (
      <>
        {isAddis && <SidebarControls title="Weather">
            <Segmented label="Weather" value={source} options={['ERA5', 'station'] as WeatherSource[]} onChange={setSource}
              optionLabel={(o) => (o === 'station' ? 'Airport station' : 'ERA5')}
              title="ERA5 reanalysis, or the Bole airport station (Dec 2022 – Sep 2024)" />
        </SidebarControls>}
        <div className="panel weather-head">
          <div>
            <h2 style={{ margin: 0 }}>{one.name} · weather and the filters</h2>
            <p className="subtitle" style={{ margin: '4px 0 0' }}>
              {one.n} filters, {span}, each with the weather averaged over its own sampling window{isAddis ? ' (ERA5, or the Bole airport station the Addis notebooks used)' : ' (ERA5)'}.
              {' '}{focus.site ? `One of the ${sites.length} picked sites: switch site or go back to the comparison under Page › analysis.` : 'Pick more sites in the Samples panel to compare them.'}
            </p>
          </div>
          <div className="weather-head-controls">
            {focus.site && <button type="button" className="btn" onClick={() => focus.set(null)}
              title="Back to the picked sites side by side (Page › analysis)">← Back to comparing {sites.length} sites</button>}
          </div>
        </div>
        <WeatherTimeline rows={oneRows} meta={meta} field={field} site={one} />
        <MonthlyClimate meta={meta} sites={[one]} large id="weather-site-climate" title={`${one.name} · monthly climate against its season calendar`} />
        <RainWashout rows={oneRows} meta={meta} field={field} prcpField={wx('prcp')} id="weather-site-washout" title={`${one.name} · rain washout`} />
        {scatter(`${one.name} · ${field} against the weather`, 'weather-site-scatter')}
        <HumidityAgreement rows={oneRows} meta={meta} xField={xField} yField={yField} rhField={wx('rh')} />
        <PwMoisture rows={oneRows} meta={meta} onePick />
        <WindRoses rows={oneRows} meta={meta} roses={roses} source={source} site={one} pmfLabelOfDate={pmfLabelOfDate}
          splits={isAddis ? ['Season', 'PMF source'] : ['Season']} id="weather-site-roses" title={`${one.name} · wind roses`} />
        <PollutionRose rows={oneRows} meta={meta} field={field} source={source} site={one} splits={['Season']}
          id="weather-site-pollution-rose" title={`${one.name} · pollution rose`} />
        <PolarPlot rows={oneRows} meta={meta} field={field} source={source} site={one} splits={['Season']}
          id="weather-site-polar" title={`${one.name} · polar plot by wind direction and speed`} />
        <WindSpeedConcentration rows={oneRows} meta={meta} field={field} wspdField={wx('wspd')}
          id="weather-site-wind-speed" title={`${one.name} · the measurement by wind speed`} />
        <WeatherCorrelations rows={oneRows} meta={meta} source={source} site={one} />
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
            The sites picked in the Samples panel, on ERA5 averaged over each filter's sampling window. Click a site in the table
            (or choose One site under Page › analysis) for its timeline, seasonal wind roses and correlations, and the Addis station check.
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
      <PollutionRose rows={wxRows} meta={meta} field={field} source="ERA5" site={null} splits={['Site']}
        id="weather-compare-pollution-rose" title="Pollution roses — the picked sites" />
      <PolarPlot rows={wxRows} meta={meta} field={field} source="ERA5" site={null} splits={['Site']}
        id="weather-compare-polar" title="Polar plots by wind direction and speed — the picked sites" />
      <WindSpeedConcentration rows={wxRows} meta={meta} field={field} wspdField={weatherField('wspd', 'ERA5')!}
        id="weather-compare-wind-speed" title="The measurement by wind speed — the picked sites" />
    </>
  )
}
