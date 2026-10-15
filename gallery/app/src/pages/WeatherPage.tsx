import { useCallback, useEffect, useMemo, useState } from 'react'
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
import { useExport } from '@/lib/data'
import { useSearchParam } from '@/lib/url'
import { WEATHER_VARS, weatherField, type RoseFile, type StationCompare, type WeatherSource } from '@/lib/weather'
import type { AxesOpts } from '@/lib/axes'
import type { FilterRow, MetaFile } from '@/lib/types'

const ADDIS = 'ETAD'

/**
 * The weather analyses the Addis notebooks ran on the airport station (addis_03_meteorology,
 * notebooks/analysis/meteorology/*), on ERA5 for every SPARTAN site: each filter carries the
 * weather averaged over its own sampling window (gallery/data/export_weather.py). The site
 * and network pick, seasons and dates in the Filters panel apply as on the other tabs.
 */
export function WeatherPage({ rows, meta, field, xField, yField, axes, setField, pmfLabelOfDate }: {
  rows: FilterRow[]; meta: MetaFile; field: string; xField: string; yField: string; axes: AxesOpts
  setField: (f: string) => void
  pmfLabelOfDate: (d: string | null | undefined) => string | null
}) {
  const roses = useExport<RoseFile>('weather/roses.json').data
  const station = useExport<StationCompare>('weather/station_ETAD.json').data

  // sites in view that have weather, Addis first (the notebooks' site)
  const sites = useMemo(() => {
    const withWx = new Set(rows.filter((r) => typeof r['Temperature (ERA5)'] === 'number').map((r) => r.site))
    return meta.sites.filter((s) => withWx.has(s.name)).map((s) => ({ name: s.name, code: s.code }))
      .sort((a, b) => (a.code === ADDIS ? -1 : b.code === ADDIS ? 1 : a.name.localeCompare(b.name)))
  }, [rows, meta.sites])
  const [siteCode, setSiteCode] = useSearchParam<string>('wsite', ADDIS)
  const site = sites.find((s) => s.code === siteCode) ?? sites[0] ?? null
  const hasAddis = sites.some((s) => s.code === ADDIS)
  const [sourceParam, setSource] = useSearchParam<string>('wsrc', 'ERA5', ['ERA5', 'station'])
  const source: WeatherSource = sourceParam === 'station' && hasAddis ? 'station' : 'ERA5'
  const wx = (k: Parameters<typeof weatherField>[0]) => weatherField(k, source) ?? weatherField(k, 'ERA5')!

  // the scatter's x is a weather variable of its own; its y is the tab's measurement
  const [wxX, setWxX] = useState(wx('range'))
  // switching source keeps the same quantity on x (Temperature (ERA5) -> Temperature (station))
  useEffect(() => setWxX((cur) => {
    const v = WEATHER_VARS.find((x) => x.era5 === cur || x.station === cur)
    return v ? weatherField(v.key, source) ?? v.era5 : cur
  }), [source])
  const axisValue = useMemo(() => ({ x: wxX, y: field, field }), [wxX, field])
  const setAxis = useCallback((role: AxisRole, f: string) => (role === 'x' ? setWxX(f) : setField(f)), [setField])
  const scatterRows = useMemo(() => rows.filter((r) => typeof r[wxX] === 'number'), [rows, wxX])

  if (sites.length === 0) {
    return <div className="panel"><p className="frame-empty">Weather is pulled for SPARTAN sites only (ERA5 over every filter's sampling window). Pick SPARTAN sites in the Filters panel.</p></div>
  }
  const addisRows = rows.filter((r) => r.code === ADDIS || r.site === 'Addis Ababa')

  return (
    <>
      <div className="panel weather-head">
        <div>
          <h2 style={{ margin: 0 }}>Weather and the filters</h2>
          <p className="subtitle" style={{ margin: '4px 0 0' }}>
            The Addis meteorology analyses (rain washout, temperature and the inversion proxy, humidity and method agreement, wind roses, seasonal correlations), for every SPARTAN site.
            Each filter carries the weather averaged over its own sampling window, so the fields also appear in every other tab’s pickers, under “Weather”.
          </p>
        </div>
        <div className="weather-head-controls">
          <Combobox label="Site" value={site?.code ?? ''} onChange={setSiteCode} title="The site for the single-site charts (timeline, seasonal roses, correlations); the others draw every site in the pick"
            options={sites.map((s) => ({ key: s.code, label: s.name }))} />
          <Segmented label="Weather" value={source} options={(hasAddis ? ['ERA5', 'station'] : ['ERA5']) as WeatherSource[]} onChange={setSource}
            optionLabel={(o) => (o === 'station' ? 'Addis station' : 'ERA5')}
            title="ERA5 reanalysis at every site, or (Addis only, Dec 2022 – Sep 2024) the Bole airport station the notebooks used" />
        </div>
      </div>
      <WeatherTimeline rows={rows} meta={meta} field={field} site={site} />
      <RainWashout rows={source === 'station' ? addisRows : rows} meta={meta} field={field} prcpField={wx('prcp')} />
      <AxisFieldsProvider meta={meta} value={axisValue} set={setAxis}>
        <Scatterplot rows={source === 'station' ? scatterRows.filter((r) => r.site === 'Addis Ababa') : scatterRows} meta={meta} xField={wxX} yField={field} axes={{ ...axes, identity: false, mode: 'Data' }} />
      </AxisFieldsProvider>
      <HumidityAgreement rows={source === 'station' ? addisRows : rows} meta={meta} xField={xField} yField={yField} rhField={wx('rh')} />
      <WindRoses rows={source === 'station' ? addisRows : rows} meta={meta} roses={roses} source={source} site={site} pmfLabelOfDate={pmfLabelOfDate} />
      <WeatherCorrelations rows={rows} meta={meta} source={source} site={site} />
      <MonthlyClimate meta={meta} sites={sites} />
      {hasAddis && <StationCheck data={station} />}
    </>
  )
}
