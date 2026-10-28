import { useContext, useMemo, type ReactNode } from 'react'
import * as d3 from 'd3'
import { Combobox } from '@/components/Combobox'
import { Empty, FrameExtrasContext, Segmented } from '@/components/ChartFrame'
import { WeatherTimeline } from '@/charts/weather/WeatherTimeline'
import { MonthlyClimate } from '@/charts/weather/MonthlyClimate'
import { RainWashout } from '@/charts/weather/RainWashout'
import { WindRoses } from '@/charts/weather/WindRoses'
import { PollutionRose } from '@/charts/weather/PollutionRose'
import { PolarPlot } from '@/charts/weather/PolarPlot'
import { WindSpeedConcentration } from '@/charts/weather/WindSpeedConcentration'
import { WeatherCorrelations } from '@/charts/weather/WeatherCorrelations'
import { SiteWeatherSummary } from '@/charts/weather/SiteWeatherSummary'
import { RunCrossplot } from '@/charts/calibration/RunCrossplot'
import { CVCurve } from '@/charts/calibration/CVCurve'
import { useExport } from '@/lib/data'
import { useTileSetting } from '@/lib/url'
import { weatherField, type RoseFile, type WeatherSource } from '@/lib/weather'
import type { CalibRunsFile, FilterRow, MetaFile } from '@/lib/types'

/**
 * Dashboard tiles for the Weather and Calibration tabs' charts. Those pages work out their
 * charts' inputs themselves (the rows with weather, the one picked site, ERA5 or the station;
 * the calibration run from a preset and a target); these adapters do the same from the
 * dashboard's rows, so the charts themselves are used unchanged.
 */

const ADDIS = 'ETAD'

/** add a control to the tile's ⋯ menu, after the tile's own field pickers */
function WithControls({ controls, children }: { controls: ReactNode; children: ReactNode }) {
  const extras = useContext(FrameExtrasContext)
  if (!extras) return <>{children}</>
  return <FrameExtrasContext.Provider value={{ ...extras, controls: <>{extras.controls}{controls}</> }}>{children}</FrameExtrasContext.Provider>
}

/** the rows with weather, the sites they cover, the one site if there is one, and the weather source */
function useWeather(rows: FilterRow[], meta: MetaFile) {
  const wxRows = useMemo(() => rows.filter((r) => typeof r['Temperature (ERA5)'] === 'number'), [rows])
  const sites = useMemo(() => {
    const n = d3.rollup(wxRows, (v) => v.length, (r) => r.site)
    return meta.sites.filter((s) => n.has(s.name)).map((s) => ({ name: s.name, code: s.code, n: n.get(s.name)! }))
  }, [wxRows, meta.sites])
  const one = sites.length === 1 ? sites[0] : null
  // the Bole airport station is an alternative only at Addis
  const [src, setSrc] = useTileSetting<string>('wsrc', 'ERA5', ['ERA5', 'station'])
  const source: WeatherSource = src === 'station' && one?.code === ADDIS ? 'station' : 'ERA5'
  const wx = (k: Parameters<typeof weatherField>[0]) => weatherField(k, source) ?? weatherField(k, 'ERA5')!
  const sourcePicker = one?.code === ADDIS
    ? <Segmented label="Weather" value={source} options={['ERA5', 'station'] as WeatherSource[]} onChange={setSrc}
      optionLabel={(o) => (o === 'station' ? 'Airport station' : 'ERA5')} />
    : null
  return { wxRows, sites, one, source, wx, sourcePicker }
}

const noWeather = <Empty>No site in view has weather (ERA5 is pulled for the SPARTAN sites).</Empty>

export function WeatherTimelineTile({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const w = useWeather(rows, meta)
  if (!w.sites.length) return noWeather
  return <WithControls controls={w.sourcePicker}><WeatherTimeline rows={w.wxRows} meta={meta} field={field} site={w.one} /></WithControls>
}

export function MonthlyClimateTile({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const w = useWeather(rows, meta)
  if (!w.sites.length) return noWeather
  return <MonthlyClimate meta={meta} sites={w.sites} title="Monthly climate and the season calendar" />
}

export function RainWashoutTile({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const w = useWeather(rows, meta)
  if (!w.sites.length) return noWeather
  return <WithControls controls={w.sourcePicker}><RainWashout rows={w.wxRows} meta={meta} field={field} prcpField={w.wx('prcp')} /></WithControls>
}

export function WindRosesTile({ rows, meta, pmfLabelOfDate }: { rows: FilterRow[]; meta: MetaFile; pmfLabelOfDate?: (d: string | null | undefined) => string | null }) {
  const w = useWeather(rows, meta)
  const roses = useExport<RoseFile>('weather/roses.json').data
  if (!w.sites.length) return noWeather
  return <WithControls controls={w.sourcePicker}>
    <WindRoses rows={w.wxRows} meta={meta} roses={roses} source={w.one ? w.source : 'ERA5'} site={w.one} pmfLabelOfDate={pmfLabelOfDate ?? (() => null)} />
  </WithControls>
}

// the measurement by wind: one site splits by season, several by site; the tile's picker sets the field
export function PollutionRoseTile({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const w = useWeather(rows, meta)
  if (!w.sites.length) return noWeather
  return <WithControls controls={w.sourcePicker}>
    <PollutionRose rows={w.wxRows} meta={meta} field={field} pickField={false} source={w.source} site={w.one} splits={w.one ? ['Season'] : ['Site']} />
  </WithControls>
}

export function PolarPlotTile({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const w = useWeather(rows, meta)
  if (!w.sites.length) return noWeather
  return <WithControls controls={w.sourcePicker}>
    <PolarPlot rows={w.wxRows} meta={meta} field={field} pickField={false} source={w.source} site={w.one} splits={w.one ? ['Season'] : ['Site']} />
  </WithControls>
}

export function WindSpeedTile({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const w = useWeather(rows, meta)
  if (!w.sites.length) return noWeather
  return <WithControls controls={w.sourcePicker}><WindSpeedConcentration rows={w.wxRows} meta={meta} field={field} pickField={false} wspdField={w.wx('wspd')} /></WithControls>
}

export function WeatherCorrelationsTile({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const w = useWeather(rows, meta)
  if (!w.sites.length) return noWeather
  return <WithControls controls={w.sourcePicker}><WeatherCorrelations rows={w.wxRows} meta={meta} source={w.source} site={w.one} /></WithControls>
}

export function WeatherSummaryTile({ rows, meta, field, onOpenSite }: { rows: FilterRow[]; meta: MetaFile; field: string; onOpenSite?: (code: string) => void }) {
  const w = useWeather(rows, meta)
  if (!w.sites.length) return noWeather
  return <SiteWeatherSummary rows={w.wxRows} meta={meta} field={field} current={null} onSite={onOpenSite ?? (() => {})} />
}

/** what the calibration tiles need from App: the explorer's runs and the gallery's filters */
export interface CalibrationData {
  runs: CalibRunsFile
  lambda10: number
  knownIds: Set<string>
  galleryRows: FilterRow[]
  /** every filter's row before the view's options (its deployed SPARTAN EC) */
  sourceRows?: FilterRow[]
  galleryMeta: MetaFile
  /** the drawer's meta (exported units), as the Calibration tab passes it */
  meta: MetaFile
}

/** a preset and a target, picked in the tile's ⋯ menu (Addis and the Calibration tab's default preset to start) */
function useRun(cal: CalibrationData) {
  const presets = cal.runs.presets
  const [presetKey, setPreset] = useTileSetting<string>('preset', 'ocec800_airspec', presets.map((p) => p.key))
  const preset = presets.find((p) => p.key === presetKey) ?? presets[0]
  const targets = [...new Set(cal.runs.runs.filter((r) => r.preset === preset?.key).map((r) => r.target))]
  const [targetId, setTarget] = useTileSetting<string>('target', 'addis', targets)
  const target = targets.includes(targetId) ? targetId : targets[0]
  const run = cal.runs.runs.find((r) => r.preset === preset?.key && r.target === target) ?? null
  const picker = <>
    <Combobox label="Calibration" value={preset?.key ?? ''} onChange={setPreset} options={presets.map((p) => ({ key: p.key, label: p.label }))} />
    {targets.length > 1 && <Combobox label="Evaluated on" value={target ?? ''} onChange={setTarget}
      options={targets.map((t) => ({ key: t, label: cal.runs.targets[t]?.label ?? t }))} />}
  </>
  return { run, picker }
}

export function RunCrossplotTile({ cal }: { cal: CalibrationData }) {
  const { run, picker } = useRun(cal)
  if (!run) return <Empty>No calibration run for this preset.</Empty>
  return <WithControls controls={picker}>
    <RunCrossplot lambda10={cal.lambda10} run={run} runs={cal.runs} meta={cal.meta} knownIds={cal.knownIds} galleryRows={cal.galleryRows} sourceRows={cal.sourceRows} galleryMeta={cal.galleryMeta} />
  </WithControls>
}

export function CVCurveTile({ cal }: { cal: CalibrationData }) {
  const { run, picker } = useRun(cal)
  if (!run) return <Empty>No calibration run for this preset.</Empty>
  return <WithControls controls={picker}><CVCurve run={run} runs={cal.runs} /></WithControls>
}
