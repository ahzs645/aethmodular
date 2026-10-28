import { useEffect, useMemo, useRef, useState } from 'react'
import { seasonOnly } from '@/lib/seasons'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { FieldSelect } from '@/components/FieldSelect'
import { XIcon } from '@/components/Icons'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusLast, focusStyle, useHighlight } from '@/lib/highlight'
import { fidProps } from '@/lib/filterEdits'
import { fmt } from '@/lib/stats'
import { INK, withUnit } from '@/lib/theme'
import { loadDaily, type DailyWeather } from '@/lib/weather'
import type { FilterRow, MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { useFitHeight } from '@/lib/chartFit'
import { useSearchParam } from '@/lib/url'

const M = { top: 8, right: 20, bottom: 30, left: 68 }
const PANEL = 92, GAP = 14

/** the daily weather a panel can show (lib/weather DailyWeather), by panel key */
const WEATHER: Record<string, { label: string; axis: string }> = {
  temp: { label: 'Temperature', axis: 'Temperature (°C)' },
  prcp: { label: 'Precipitation', axis: 'Precip. (mm/day)' },
  rh: { label: 'Relative humidity', axis: 'RH (%)' },
  ws: { label: 'Wind speed', axis: 'Wind (m/s)' },
  blh: { label: 'Boundary-layer height', axis: 'BLH (m)' },
}
/** the panels, top to bottom: weather keys, 'field' (the page's measurement) and 'f:<field>' for any other measurement */
const DEFAULT_PANELS = 'temp|prcp|rh|blh|field'
const ADD = 'Add a panel…'
const weatherOption = (k: string) => `Weather · ${WEATHER[k].label}`

/**
 * The weather behind one site's filters: daily ERA5 temperature (min–max band and mean),
 * precipitation, relative humidity and boundary-layer height, above the chosen measurement
 * for each filter on the same time axis, with the site's seasons shaded. The view the Addis
 * notebooks built from the airport station (addis_03_meteorology), for any SPARTAN site.
 */
export function WeatherTimeline({ rows, meta, field, site }: { rows: FilterRow[]; meta: MetaFile; field: string; site: { name: string; code: string } | null }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()
  const [daily, setDaily] = useState<DailyWeather | null>(null)
  const [error, setError] = useState('')
  const [panelParam, setPanelParam] = useSearchParam<string>('wt.panels', DEFAULT_PANELS)
  const [cursor, setCursor] = useState<number | null>(null)

  useEffect(() => {
    if (!site) return
    let live = true
    setDaily(null); setError('')
    loadDaily(site.code).then((d) => { if (live) setDaily(d) }).catch(() => { if (live) setError(`No daily weather exported for ${site.name}.`) })
    return () => { live = false }
  }, [site])

  // the panels in the URL, in order, keeping only those this page can draw
  const panels = useMemo(() => [...new Set(panelParam.split('|'))].filter((k) => k in WEATHER || k === 'field' || (k.startsWith('f:') && meta.fields.includes(k.slice(2)))), [panelParam, meta])
  const setPanels = (ks: string[]) => setPanelParam(ks.join('|'))
  const fieldOf = (k: string) => (k === 'field' ? field : k.slice(2))
  const isField = (k: string) => k === 'field' || k.startsWith('f:')
  const panelLabel = (k: string) => (k in WEATHER ? WEATHER[k].label : fieldOf(k))
  const addPanel = (v: string) => {
    const w = Object.keys(WEATHER).find((k) => weatherOption(k) === v)
    const k = w ?? (v === field ? 'field' : `f:${v}`)
    if (v !== ADD && !panels.includes(k)) setPanels([...panels, k])
  }
  // each measurement panel's filters at this site
  const fieldKeys = panels.filter(isField).map(fieldOf).join('|')
  const ptsOf = useMemo(() => new Map(fieldKeys.split('|').filter(Boolean).map((f) => [f, rows.filter((r) => r.site === site?.name && typeof r[f] === 'number')
    .map((r) => ({ row: r, t: new Date(r.date), v: r[f] as number }))])), [rows, site, fieldKeys])
  // the filters' own span (with a fortnight either side), so the date brush and site pick decide the window
  const domain = useMemo<[Date, Date] | null>(() => {
    const ds = rows.filter((r) => r.site === site?.name).map((r) => new Date(r.date).getTime())
    if (!ds.length) return null
    return [new Date(Math.min(...ds) - 14 * 864e5), new Date(Math.max(...ds) + 14 * 864e5)]
  }, [rows, site])
  const days = useMemo(() => {
    if (!daily || !domain) return []
    return daily.date.map((d, i) => ({ t: new Date(d), i })).filter((x) => x.t >= domain[0] && x.t <= domain[1])
  }, [daily, domain])

  const innerW = Math.max(240, width - M.left - M.right)
  // in a fitted dashboard tile the panels share its height
  const fit = useFitHeight()
  const panelH = fit === null ? PANEL : Math.max(40, (fit - M.top - M.bottom - (panels.length - 1) * GAP) / Math.max(1, panels.length))
  const height = M.top + M.bottom + panels.length * panelH + (panels.length - 1) * GAP
  const x = d3.scaleTime().domain(domain ?? [new Date(), new Date()]).range([0, innerW])
  const top = (k: string) => M.top + panels.indexOf(k as never) * (panelH + GAP)
  const val = (arr: (number | null)[] | undefined, i: number) => (arr?.[i] ?? null)
  const ext = (arrs: ((number | null)[] | undefined)[], zero = false) => {
    const vs = days.flatMap(({ i }) => arrs.map((a) => val(a, i))).filter((v): v is number => v !== null)
    const lo = zero ? 0 : (d3.min(vs) ?? 0), hi = d3.max(vs) ?? 1
    return d3.scaleLinear().domain([lo, hi === lo ? lo + 1 : hi]).range([panelH, 0]).nice()
  }
  const yT = ext([daily?.tmin, daily?.tmax]), yP = ext([daily?.p], true), yR = d3.scaleLinear().domain([0, 100]).range([panelH, 0])
  const yB = ext([daily?.blh], true), yW = ext([daily?.ws], true)
  const yF = (f: string) => { const pts = ptsOf.get(f) ?? []
    return d3.scaleLinear().domain([Math.min(0, d3.min(pts, (p) => p.v) ?? 0), (d3.max(pts, (p) => p.v) ?? 1) * 1.05]).range([panelH, 0]).nice() }
  const yOf = (k: string) => (k === 'temp' ? yT : k === 'prcp' ? yP : k === 'rh' ? yR : k === 'blh' ? yB : k === 'ws' ? yW : yF(fieldOf(k)))
  const line = (arr: (number | null)[] | undefined, y: d3.ScaleLinear<number, number>) =>
    d3.line<{ t: Date; i: number }>().defined(({ i }) => val(arr, i) !== null).x(({ t }) => x(t)).y(({ i }) => y(val(arr, i)!))(days) ?? ''
  const band = d3.area<{ t: Date; i: number }>().defined(({ i }) => val(daily?.tmin, i) !== null && val(daily?.tmax, i) !== null)
    .x(({ t }) => x(t)).y0(({ i }) => yT(daily!.tmin[i]!)).y1(({ i }) => yT(daily!.tmax[i]!))

  // season shading: one rect per run of days in the same local season
  const seasons = site ? seasonsForSite(meta, site.name) : []
  const runs = useMemo(() => {
    const out: { a: Date; b: Date; color: string; name: string }[] = []
    for (const { t } of days) {
      const se = seasons.find((s) => s.months.includes(t.getUTCMonth() + 1))
      const last = out[out.length - 1]
      if (se && last && last.name === se.name) last.b = t
      else if (se) out.push({ a: t, b: t, color: se.color, name: se.name })
    }
    return out
  }, [days, seasons])

  const at = cursor === null ? null : days[d3.bisector((d: { t: Date }) => d.t).center(days, x.invert(cursor))]
  const title = (k: string) => (k in WEATHER ? WEATHER[k].axis : withUnit(fieldOf(k), meta.field_units))
  const shownWeather = new Set(panels.filter((k) => k in WEATHER))

  return (
    <ChartFrame id="weather-timeline" title={`${site ? `${site.name} · ` : ''}weather timeline`}
      subtitle="Daily ERA5 at the site (local standard day): temperature range and mean, precipitation, relative humidity and boundary-layer height, above the chosen measurement for each filter. Add a panel for wind or any other measurement the filters have; remove one with its ×. Seasons are the site's local calendar."
      provenance="addis_03_meteorology (station data) generalised to every SPARTAN site · ERA5 via Open-Meteo, Copernicus Climate Change Service"
      controls={<>
        <span className="control" style={{ flexWrap: 'wrap', gap: 4 }}>Panels
          {panels.map((k) => <span key={k} className="chip" style={{ display: 'inline-flex', alignItems: 'center', gap: 2, cursor: 'default', padding: '1px 3px 1px 9px' }}>
            {panelLabel(k)}
            <button type="button" className="btn quiet x-btn" onClick={() => setPanels(panels.filter((q) => q !== k))} aria-label={`Remove the ${panelLabel(k)} panel`} title="Remove this panel"><XIcon /></button>
          </span>)}
        </span>
        <FieldSelect label="Add" value={ADD} meta={meta} ratios={false} onChange={addPanel}
          extraOptions={Object.keys(WEATHER).filter((k) => !shownWeather.has(k)).map(weatherOption)} title="Add a weather or measurement panel to the timeline" />
      </>}>
      <div ref={wrapRef} className="chart-wrap">
        {error ? <Empty>{error}</Empty> : !site || !domain ? <Empty>No filters at this site in the current subset.</Empty> : !daily ? <Empty>Loading the daily weather…</Empty> : !panels.length ? <Empty>No panels: add one from the options.</Empty> : (
          <svg width={width} height={height} onMouseLeave={() => { setCursor(null); tip.hide() }}>
            <g transform={`translate(${M.left},0)`}>
              {panels.map((k) => (
                <g key={k} transform={`translate(0,${top(k)})`}>
                  {runs.map((r, i) => <rect key={i} x={x(r.a)} width={Math.max(1, x(r.b) - x(r.a) + innerW / Math.max(1, days.length))} height={panelH} fill={r.color} opacity={0.1} />)}
                  <YAxis scale={yOf(k)} x={0} gridWidth={innerW} tickCount={3}
                    label={title(k)} labelX={-50} pick={k === 'field' ? 'field' : undefined} />
                  {k === 'temp' && <>
                    <path d={band(days) ?? ''} fill={INK.fit} opacity={0.12} />
                    <path d={line(daily.t, yT)} fill="none" stroke="#c0392b" strokeWidth={1.2} />
                  </>}
                  {k === 'prcp' && days.map(({ t, i }) => val(daily.p, i) ? <rect key={i} x={x(t)} width={Math.max(1, innerW / days.length)} y={yP(daily.p[i]!)} height={panelH - yP(daily.p[i]!)} fill="#4292c6" /> : null)}
                  {k === 'rh' && <path d={line(daily.rh, yR)} fill="none" stroke="#2c7fb8" strokeWidth={1.1} />}
                  {k === 'blh' && <path d={line(daily.blh, yB)} fill="none" stroke="#7b6fb0" strokeWidth={1.1} />}
                  {k === 'ws' && <path d={line(daily.ws, yW)} fill="none" stroke="#4d9078" strokeWidth={1.1} />}
                  {isField(k) && focusLast(ptsOf.get(fieldOf(k)) ?? [], (p) => p.row.id, hl.focusId).map(([p]) => {
                    const st = focusStyle(p.row.id, hl.focusId, { r: 3, level: 'sparse' }, hl.selected)
                    return <circle key={p.row.id} cx={x(p.t)} cy={yOf(k)(p.v)} r={st.r} fill={meta.seasons.find((s) => s.name === p.row.season)?.color ?? INK.text}
                      fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={st.strokeWidth} data-dot={st.dot} style={{ cursor: 'pointer' }} {...fidProps(p.row.id)}
                      onMouseEnter={(e) => { hl.setHover(p.row.id); tip.show(e, [p.row.id, `${p.row.date} · ${seasonOnly(String(p.row.season ?? ''))}`, `${fieldOf(k)}: ${fmt(p.v)}`, 'click for details · right-click to exclude']) }}
                      onMouseLeave={() => { hl.setHover(null); tip.hide() }} onClick={() => hl.openSample(p.row.id)} />
                  })}
                  {!isField(k) && <rect width={innerW} height={panelH} fill="transparent"
                    onMouseMove={(e) => {
                      const mx = d3.pointer(e)[0]
                      setCursor(mx)
                      const d = days[d3.bisector((q: { t: Date }) => q.t).center(days, x.invert(mx))]
                      if (d) tip.show(e, [d3.utcFormat('%d %b %Y')(d.t), `T ${fmt(daily.t[d.i], 1)} °C (${fmt(daily.tmin[d.i], 1)} – ${fmt(daily.tmax[d.i], 1)})`,
                        `precip ${fmt(daily.p[d.i], 1)} mm`, `RH ${fmt(daily.rh[d.i], 0)} %`, `wind ${fmt(daily.ws[d.i], 1)} m/s`, `BLH ${fmt(daily.blh[d.i], 0)} m`])
                    }} />}
                </g>
              ))}
              {at && <line x1={x(at.t)} x2={x(at.t)} y1={M.top} y2={height - M.bottom} stroke={INK.text} strokeOpacity={0.35} strokeDasharray="3 3" pointerEvents="none" />}
              <XAxis scale={x} y={height - M.bottom} tickCount={8} format={d3.utcFormat("%b %Y")} />
            </g>
          </svg>
        )}
        <div className="legend">
          {seasons.map((s) => <span key={s.name} className="legend-item"><span className="swatch band" style={{ background: s.color }} />{s.name.replace(`${site?.name} · `, '')}</span>)}
          <span className="legend-item"><span className="swatch line" style={{ background: '#c0392b' }} />daily mean temperature</span>
          <span className="legend-item"><span className="swatch band" style={{ background: INK.fit }} />daily min–max</span>
        </div>
        {tip.node}
      </div>
    </ChartFrame>
  )
}
