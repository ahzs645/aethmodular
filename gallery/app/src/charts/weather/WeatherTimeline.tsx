import { useJsonLoader } from '@/lib/data'
import { useCallback, useMemo } from 'react'
import { seasonOnly } from '@/lib/seasons'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { AxisTitle } from '@/components/Axes'
import type { AxisRole } from '@/components/AxisFieldPopover'
import { AxisChoices, type AxisOption } from '@/components/AxisChoices'
import { FieldSelect } from '@/components/FieldSelect'
import { Legend } from '@/components/Legend'
import { PanelPicker } from '@/components/PanelPicker'
import { DailyLanes, FieldLane, NO_POINTS, dailyDomain, dailyLine, dailyReadout, fieldDomain, fieldPoints, useSiteDaily, type Day } from '@/components/DailyLanes'
import type { StackPanel } from '@/components/StackedPanels'
import { useHighlightActions } from '@/lib/highlight'
import { fmt } from '@/lib/stats'
import { chemText } from '@/lib/format'
import { estimatorText, useEstimator } from '@/lib/regressionPrefs'
import { INK, withUnit } from '@/lib/theme'
import { loadDaily } from '@/lib/weather'
import type { FilterRow, MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { wrapCategory } from '@/components/CategoryAxis'
import { usePanelPick } from '@/lib/panelPick'
import { isPmfField, pickerName } from '@/lib/pmfGroups'

// each panel's title stands up beside its y axis, as on any plot: the unit on a line of its own
// ("Precip." over "(mm/day)"), and a name too long for the panel's height wrapped onto more lines.
// The left margin grows with the most lines any panel needs (TITLE_LINE each), past the tick labels (TICK_ROOM)
const TITLE_FONT = 11.5, TITLE_LINE = 13, TICK_ROOM = 44

/** the daily weather a panel can show (lib/weather DailyWeather), by panel key */
const WEATHER: Record<string, { label: string; axis: string }> = {
  temp: { label: 'Temperature', axis: 'Temperature (°C)' },
  prcp: { label: 'Precipitation', axis: 'Precip. (mm/day)' },
  rh: { label: 'Relative humidity', axis: 'RH (%)' },
  ws: { label: 'Wind speed', axis: 'Wind (m/s)' },
  blh: { label: 'Boundary-layer height', axis: 'BLH (m)' },
}
/** the panels, top to bottom: weather keys, 'field' (the page's measurement) and 'f:<field>' for any other measurement */
const DEFAULT_PANELS = ['temp', 'prcp', 'rh', 'blh', 'field']
const ADD = 'Add a panel…'
const weatherOption = (k: string) => `Weather · ${WEATHER[k].label}`

/**
 * The weather behind one site's filters: daily ERA5 temperature (min–max band and mean),
 * precipitation, relative humidity and boundary-layer height, above the chosen measurement
 * for each filter on the same time axis, with the site's seasons shaded. The view the Addis
 * notebooks built from the airport station (addis_03_meteorology), for any SPARTAN site.
 * The lanes, cursor and season shading are components/DailyLanes (shared with the AERONET timeline);
 * the choice and order of panels is lib/panelPick, kept in the link as `wt.panels`.
 */
export function WeatherTimeline({ rows, meta, field, site }: { rows: FilterRow[]; meta: MetaFile; field: string; site: { name: string; code: string } | null }) {
  const jsonLoader = useJsonLoader()
  // the brush only: the filter points read the hover themselves (components/ScatterLayer)
  const hl = useHighlightActions()
  const load = useCallback((code: string) => loadDaily(code, jsonLoader), [jsonLoader])
  const { daily, error, domain, days } = useSiteDaily(rows, site, load, (name) => `No daily weather exported for ${name}.`)

  // the panels in the URL, in order, keeping only those this page can draw
  const allPanels = useMemo(() => [...Object.keys(WEATHER), 'field', ...meta.fields.map((f) => `f:${f}`)], [meta])
  const pick = usePanelPick('wt', allPanels, DEFAULT_PANELS)
  const panels = pick.ids
  const fieldOf = (k: string) => (k === 'field' ? field : k.slice(2))
  const isField = (k: string) => k === 'field' || k.startsWith('f:')
  const panelLabel = (k: string) => (k in WEATHER ? WEATHER[k].label : fieldOf(k))
  // every panel's axis title swaps it for another weather variable or measurement (components/AxisChoices);
  // the page's own measurement panel keeps the shared picker, which also sets the sidebar's measurement
  const choices = useMemo<AxisOption[]>(() => [
    ...Object.keys(WEATHER).map((k) => ({ key: k, group: 'Daily weather (ERA5)', label: WEATHER[k].label })),
    ...meta.field_groups.flatMap((g) => g.fields.filter((f) => meta.fields.includes(f))
      .map((f) => ({ key: `f:${f}`, group: g.label, label: f, unit: meta.field_units[f] }))),
  ], [meta])
  const addPanel = (v: string) => {
    const w = Object.keys(WEATHER).find((k) => weatherOption(k) === v)
    const k = w ?? (v === field ? 'field' : `f:${v}`)
    if (v !== ADD && !panels.includes(k)) pick.add(k)
  }
  // each measurement panel's filters at this site
  const fieldKeys = panels.filter(isField).map(fieldOf).join('|')
  const ptsOf = useMemo(() => new Map(fieldKeys.split('|').filter(Boolean).map((f) => [f, fieldPoints(rows, site?.name, f)])), [rows, site, fieldKeys])
  // the PMF source shares of this site's filters (only where the PMF solution covers the site: lib/pmfGroups withPmf),
  // so hovering the boundary-layer height (or any weather panel) says which sources the filters of those days carry
  const pmfDays = useMemo(() => rows.filter((r) => r.site === site?.name).flatMap((r) => {
    const shares = Object.entries(r).filter((e): e is [string, number] => isPmfField(e[0]) && typeof e[1] === 'number').sort((a, b) => b[1] - a[1])
    return shares.length ? [{ t: new Date(r.date).getTime(), id: r.id, shares }] : []
  }).sort((a, b) => a.t - b.t), [rows, site])
  /** the sources of the filter on a day, or the nearest within 3 days, as tooltip lines (none where there is no PMF) */
  const sourcesAt = (t: Date, own?: string): string[] => {
    if (!pmfDays.length) return []
    const p = own ? pmfDays.find((q) => q.id === own) : pmfDays[d3.bisector((q: { t: number }) => q.t).center(pmfDays, t.getTime())]
    if (!p) return []
    const off = Math.round((p.t - t.getTime()) / 864e5)
    if (Math.abs(off) > 3) return []
    const when = own || !off ? '' : ` (${Math.abs(off)} d ${off < 0 ? 'before' : 'after'})`
    return [`PMF sources${own ? '' : ` · ${p.id}`}${when}:`, ...p.shares.filter(([, v]) => v >= 0.05).slice(0, 4).map(([k, v]) => `· ${pickerName(k)} ${Math.round(v * 100)} %`)]
  }

  const title = (k: string) => (k in WEATHER ? WEATHER[k].axis : withUnit(fieldOf(k), meta.field_units))
  // the left margin fits the most lines any panel's title needs at its height (in a tile the panels share the tile's height)
  const left = (hs: readonly number[]) => TICK_ROOM + 8 + TITLE_LINE * Math.max(1, ...panels.map((k, i) => axisLines(title(k), hs[i] - 4).length))
  const val = (arr: (number | null)[] | undefined, i: number) => (arr?.[i] ?? null)
  const seasons = useMemo(() => (site ? seasonsForSite(meta, site.name) : []), [meta, site])

  const draw = (k: string): Pick<StackPanel<d3.ScaleTime<number, number>>, 'y' | 'children' | 'pointer'> => {
    if (!daily) return { y: { domain: [0, 1] } }
    const line = (arr: (number | null)[], color: string) => ({ x, y }: { x: d3.ScaleTime<number, number>; y: d3.ScaleLinear<number, number> }) =>
      <path d={dailyLine(days, arr, x, y)} fill="none" stroke={color} strokeWidth={1.1} />
    if (k === 'temp') return { y: { domain: dailyDomain(days, [daily.tmin, daily.tmax]), nice: true }, children: ({ x, y }) => {
      const band = d3.area<Day>().defined(({ i }) => val(daily.tmin, i) !== null && val(daily.tmax, i) !== null)
        .x(({ t }) => x(t)).y0(({ i }) => y(daily.tmin[i]!)).y1(({ i }) => y(daily.tmax[i]!))
      return <>
        <path d={band(days) ?? ''} fill={INK.fit} opacity={0.12} />
        <path d={dailyLine(days, daily.t, x, y)} fill="none" stroke="#c0392b" strokeWidth={1.2} />
      </>
    } }
    if (k === 'prcp') return { y: { domain: dailyDomain(days, [daily.p], { lo: 0 }), nice: true }, children: ({ x, y, width, height }) =>
      days.map(({ t, i }) => val(daily.p, i) ? <rect key={i} x={x(t)} width={Math.max(1, width / days.length)} y={y(daily.p[i]!)} height={height - y(daily.p[i]!)} fill="#4292c6" /> : null) }
    if (k === 'rh') return { y: { domain: [0, 100] }, children: line(daily.rh, '#2c7fb8') }
    if (k === 'blh') return { y: { domain: dailyDomain(days, [daily.blh], { lo: 0 }), nice: true }, children: line(daily.blh, '#7b6fb0') }
    if (k === 'ws') return { y: { domain: dailyDomain(days, [daily.ws], { lo: 0 }), nice: true }, children: line(daily.ws, '#4d9078') }
    const f = fieldOf(k), pts = ptsOf.get(f) ?? NO_POINTS
    return { y: { domain: fieldDomain(pts), nice: true }, pointer: false, children: ({ x, y, width, height, tip }) =>
      <FieldLane pts={pts} field={f} site={site?.name} x={x} y={y} width={width} height={height} seasons={meta.seasons} selected={hl.selected} tip={tip}
        tipOf={(p) => { const di = daily.date.indexOf(String(p.row.date).slice(0, 10))
          return [p.row.id, `${p.row.date} · ${seasonOnly(String(p.row.season ?? ''))}`, `${f}: ${fmt(p.v)}`,
            ...(di >= 0 ? [`BLH ${fmt(daily.blh[di], 0)} m`] : []), ...sourcesAt(p.t, p.row.id), 'click for details · right-click to exclude'] }} /> }
  }
  const lanes: StackPanel<d3.ScaleTime<number, number>>[] = panels.map((k) => ({
    key: k, ...draw(k),
    title: ({ height, left }) => k === 'field'
      ? <PanelTitle lines={axisLines(title(k), height - 4)} left={left} height={height} pick="field" />
      : <AxisChoices options={choices} y={[k, (v) => pick.replace(k, v)]} noun="panels"><PanelTitle lines={axisLines(title(k), height - 4)} left={left} height={height} pick="y" /></AxisChoices>,
  }))
  const readout = dailyReadout(days, (d) => !daily ? [] : [d3.utcFormat('%d %b %Y')(d.t), `T ${fmt(daily.t[d.i], 1)} °C (${fmt(daily.tmin[d.i], 1)} – ${fmt(daily.tmax[d.i], 1)})`,
    `precip ${fmt(daily.p[d.i], 1)} mm`, `RH ${fmt(daily.rh[d.i], 0)} %`, `wind ${fmt(daily.ws[d.i], 1)} m/s`, `BLH ${fmt(daily.blh[d.i], 0)} m`, ...sourcesAt(d.t)])
  const shownWeather = new Set(panels.filter((k) => k in WEATHER))

  return (
    <ChartFrame id="weather-timeline" title={`${site ? `${site.name} · ` : ''}weather timeline`}
      subtitle="Daily ERA5 at the site (local standard day): temperature range and mean, precipitation, relative humidity and boundary-layer height, above the chosen measurement for each filter. Add a panel for wind or any other measurement the filters have; remove one with its ×, or click a panel's axis title to swap it for another. Seasons are the site's local calendar."
      provenance="addis_03_meteorology (station data) generalised to every SPARTAN site · ERA5 via Open-Meteo, Copernicus Climate Change Service"
      controls={<PanelPicker pick={pick} labelOf={panelLabel}
        adder={<FieldSelect label="Add" value={ADD} meta={meta} ratios={false} onChange={addPanel}
          extraOptions={Object.keys(WEATHER).filter((k) => !shownWeather.has(k)).map(weatherOption)} title="Add a weather or measurement panel to the timeline" />} />}>
      <div className="chart-wrap">
        {error ? <Empty>{error}</Empty> : !site || !domain ? <Empty>No filters at this site in the current subset.</Empty> : !daily ? <Empty>Loading the daily weather…</Empty> : !panels.length ? <Empty>No panels: add one from the options.</Empty> : (
          <DailyLanes domain={domain} lanes={lanes} seasons={seasons} readout={readout} left={left} top={10} gap={16} />
        )}
        <Legend items={[...seasons.map((s) => ({ label: s.name.replace(`${site?.name} · `, ''), color: s.color, shape: 'band' as const })),
          { label: 'daily mean temperature', color: '#c0392b', shape: 'line' }, { label: 'daily min–max', color: INK.fit, shape: 'band' }]} />
      </div>
    </ChartFrame>
  )
}

/** a title split for standing up beside a panel `room` px tall: the name wrapped to fit, then the unit "(…)" on its own line */
function axisLines(text: string, room: number): string[] {
  const unit = text.endsWith(')') ? text.lastIndexOf(' (') : -1
  // wrapCategory counts 6 px a character at 11 px; the title is a little larger
  const wrap = (t: string) => wrapCategory(t, room * 11 / TITLE_FONT)
  return unit > 0 ? [...wrap(text.slice(0, unit)), text.slice(unit + 1)] : wrap(text)
}

/** a panel's title, standing up beside its y axis, its lines left to right; clicking it opens the panel's picker as an axis title does */
function PanelTitle({ lines, left, height, pick }: { lines: string[]; left: number; height: number; pick: AxisRole }) {
  const method = useEstimator()
  // the lines fill the margin from its left edge, so the last ends just short of the tick labels
  const baseline = (i: number) => -left + 4 + TITLE_LINE * (i + 1) - 2
  return (
    <AxisTitle pick={pick}>
      <g fontSize={TITLE_FONT} fontWeight={600} fill={INK.text}>
        {lines.map((line, i) => <text key={i} transform={`translate(${baseline(i)},${height / 2}) rotate(-90)`} textAnchor="middle">
          {chemText(estimatorText(line, method))}{i === lines.length - 1 && <tspan className="axis-pick-caret" dx={3} data-export="skip">▾</tspan>}
        </text>)}
      </g>
    </AxisTitle>
  )
}
