import { useCallback, useEffect, useMemo, useState } from 'react'
import * as d3 from 'd3'
import { Combobox } from '@/components/Combobox'
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
import { useExport } from '@/lib/data'
import { useSearchParam } from '@/lib/url'
import { WEATHER_VARS, weatherField, type RoseFile, type StationCompare, type WeatherSource } from '@/lib/weather'
import type { AxesOpts } from '@/lib/axes'
import type { FilterRow, MetaFile, SeasonMeta, SiteMeta } from '@/lib/types'

const ADDIS = 'ETAD'
const GREY = '#9aa5b1'   // lib/networks paints sites outside the pick grey

/**
 * Weather and the filters, in two parts.
 *
 * One site: any SPARTAN site (picked here, whatever the Filters panel holds), with the
 * analyses the Addis notebooks ran on the airport station (addis_03_meteorology,
 * notebooks/analysis/meteorology/*): the daily timeline under the filters, its climate
 * against its season calendar, rain washout, the measurement against any weather variable,
 * humidity and method agreement, wind roses by season or PMF source, seasonal correlations.
 *
 * Compare sites: the Filters panel's pick side by side. "Add to comparison" puts the open
 * site into that pick. Every filter carries the weather averaged over its own sampling
 * window (gallery/data/export_weather.py); dates, seasons and hand edits apply to both parts.
 */
export function WeatherPage({ rows, anySiteRows, allSites, seasonsOfSite, picked, onToggleSite, meta, field, xField, yField, axes, setField, pmfLabelOfDate }: {
  /** the Filters panel's pick (the comparison) */
  rows: FilterRow[]
  /** every site's filters under the same dates, seasons and hand edits (the one-site view) */
  anySiteRows: FilterRow[]
  allSites: SiteMeta[]
  seasonsOfSite: (site: string) => SeasonMeta[]
  picked: Set<string>
  onToggleSite: (code: string) => void
  meta: MetaFile; field: string; xField: string; yField: string; axes: AxesOpts
  setField: (f: string) => void
  pmfLabelOfDate: (d: string | null | undefined) => string | null
}) {
  const roses = useExport<RoseFile>('weather/roses.json').data
  const station = useExport<StationCompare>('weather/station_ETAD.json').data

  // every SPARTAN site with weather, Addis first (the notebooks' site), then by name
  const sites = useMemo(() => {
    const has = d3.rollup(anySiteRows.filter((r) => typeof r['Temperature (ERA5)'] === 'number'), (v) => v.length, (r) => r.site)
    return allSites.filter((s) => has.has(s.name))
      .map((s) => ({ ...s, color: s.color === GREY ? '#2171b5' : s.color, n: has.get(s.name)! }))
      .sort((a, b) => (a.code === ADDIS ? -1 : b.code === ADDIS ? 1 : a.name.localeCompare(b.name)))
  }, [anySiteRows, allSites])
  const [siteCode, setSiteCode] = useSearchParam<string>('wsite', ADDIS)
  const site = sites.find((s) => s.code === siteCode) ?? sites[0] ?? null
  const isAddis = site?.code === ADDIS
  const [sourceParam, setSource] = useSearchParam<string>('wsrc', 'ERA5', ['ERA5', 'station'])
  const source: WeatherSource = sourceParam === 'station' && isAddis ? 'station' : 'ERA5'
  const wx = (k: Parameters<typeof weatherField>[0]) => weatherField(k, source) ?? weatherField(k, 'ERA5')!

  const siteRows = useMemo(() => (site ? anySiteRows.filter((r) => r.site === site.name) : []), [anySiteRows, site])
  // the one-site charts get a meta of that site alone, with its own calendar
  const siteMeta = useMemo<MetaFile>(() => (site ? { ...meta, sites: [site], seasons: seasonsOfSite(site.name), season_site: site.name } : meta), [meta, site, seasonsOfSite])
  const span = siteRows.length ? `${d3.min(siteRows, (r) => r.date)} to ${d3.max(siteRows, (r) => r.date)}` : ''

  // the scatter's x is a weather variable of its own; its y is the tab's measurement
  const [wxX, setWxX] = useState(wx('range'))
  // switching source keeps the same quantity on x (Temperature (ERA5) -> Temperature (station))
  useEffect(() => setWxX((cur) => {
    const v = WEATHER_VARS.find((x) => x.era5 === cur || x.station === cur)
    return v ? weatherField(v.key, source) ?? v.era5 : cur
  }), [source])
  const axisValue = useMemo(() => ({ x: wxX, y: field, field }), [wxX, field])
  const setAxis = useCallback((role: AxisRole, f: string) => (role === 'x' ? setWxX(f) : setField(f)), [setField])

  const pickedWithWeather = useMemo(() => sites.filter((s) => picked.has(s.code)), [sites, picked])
  const compareRows = useMemo(() => rows.filter((r) => typeof r['Temperature (ERA5)'] === 'number'), [rows])

  if (!site) {
    return <div className="panel"><p className="frame-empty">Weather is pulled for SPARTAN sites only (ERA5 over every filter's sampling window). Switch the network to SPARTAN in the Filters panel.</p></div>
  }
  const inCompare = picked.has(site.code)

  return (
    <>
      <div className="panel weather-head">
        <div>
          <h2 style={{ margin: 0 }}>Weather at one site</h2>
          <p className="subtitle" style={{ margin: '4px 0 0' }}>
            {site.n} filters at <strong>{site.name}</strong>{span ? `, ${span}` : ''}, each with the weather averaged over its own sampling window
            {isAddis ? ' (ERA5, or the Bole airport station the Addis notebooks used)' : ' (ERA5)'}. Any SPARTAN site, whatever the Filters panel picks; its dates, seasons and exclusions still apply.
          </p>
        </div>
        <div className="weather-head-controls">
          <Combobox label="Site" value={site.code} onChange={setSiteCode} title="Every SPARTAN site with weather"
            options={sites.map((s) => ({ key: s.code, label: s.name, hint: String(s.n) }))} />
          {isAddis && <Segmented label="Weather" value={source} options={['ERA5', 'station'] as WeatherSource[]} onChange={setSource}
            optionLabel={(o) => (o === 'station' ? 'Airport station' : 'ERA5')}
            title="ERA5 reanalysis, or the Bole airport station (Dec 2022 – Sep 2024), at Addis" />}
          <button type="button" className={`btn ${inCompare ? 'on' : ''}`} onClick={() => onToggleSite(site.code)}
            title={inCompare ? 'Take this site out of the comparison below (the Filters panel pick)' : 'Add this site to the comparison below (the Filters panel pick)'}>
            {inCompare ? '✓ In the comparison' : '+ Add to comparison'}
          </button>
        </div>
      </div>

      <WeatherTimeline rows={siteRows} meta={siteMeta} field={field} site={site} />
      <MonthlyClimate meta={siteMeta} sites={[site]} large id="weather-site-climate" title={`${site.name} · monthly climate against its season calendar`} />
      <RainWashout rows={siteRows} meta={siteMeta} field={field} prcpField={wx('prcp')} id="weather-site-washout" title={`${site.name} · rain washout`} />
      <AxisFieldsProvider meta={siteMeta} value={axisValue} set={setAxis}>
        <Scatterplot rows={siteRows.filter((r) => typeof r[wxX] === 'number')} meta={siteMeta} xField={wxX} yField={field}
          axes={{ ...axes, identity: false, mode: 'Data' }} id="weather-site-scatter" title={`${site.name} · ${field} against the weather`} />
      </AxisFieldsProvider>
      <HumidityAgreement rows={siteRows} meta={siteMeta} xField={xField} yField={yField} rhField={wx('rh')} />
      <WindRoses rows={siteRows} meta={siteMeta} roses={roses} source={source} site={site} pmfLabelOfDate={pmfLabelOfDate}
        splits={isAddis ? ['Season', 'PMF source'] : ['Season']} id="weather-site-roses" title={`${site.name} · wind roses`} />
      <WeatherCorrelations rows={siteRows} meta={siteMeta} source={source} site={site} />
      {isAddis && <StationCheck data={station} />}

      <div className="panel weather-head weather-compare-head">
        <div>
          <h2 style={{ margin: 0 }}>Compare sites</h2>
          <p className="subtitle" style={{ margin: '4px 0 0' }}>
            {pickedWithWeather.length
              ? <>The {pickedWithWeather.length} site{pickedWithWeather.length === 1 ? '' : 's'} picked in the Filters panel ({pickedWithWeather.map((s) => s.name).join(', ')}), on ERA5. Add more with “Add to comparison” above or in the Filters panel.</>
              : <>No site with weather is picked in the Filters panel. Add the site above with “Add to comparison”, or pick SPARTAN sites in the Filters panel.</>}
          </p>
        </div>
      </div>
      {pickedWithWeather.length > 0 && <>
        <SiteWeatherSummary rows={compareRows} meta={meta} field={field} current={site.code} onSite={(c) => { setSiteCode(c); window.scrollTo({ top: 0, behavior: 'smooth' }) }} />
        <MonthlyClimate meta={meta} sites={pickedWithWeather} id="weather-compare-climate" title="Monthly climate and the season calendar — the picked sites" />
        <RainWashout rows={compareRows} meta={meta} field={field} prcpField={weatherField('prcp', 'ERA5')!} id="weather-compare-washout" title="Rain washout — the picked sites" />
        <WindRoses rows={compareRows} meta={meta} roses={roses} source="ERA5" site={null} pmfLabelOfDate={pmfLabelOfDate}
          splits={['Site']} id="weather-compare-roses" title="Wind roses — the picked sites" />
      </>}
    </>
  )
}
