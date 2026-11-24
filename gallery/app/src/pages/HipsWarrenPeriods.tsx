import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { XAxis } from '@/components/Axes'
import { ChartFrame, Note } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { Segmented } from '@/components/Controls'
import { LoadState } from '@/components/ExportState'
import { HeatCell } from '@/components/HeatCell'
import { IntervalStrip } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { Marker } from '@/components/RefLines'
import { ScatterPlot, type ScatterPoint, type ScatterScales } from '@/components/ScatterPlot'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useColourScheme } from '@/lib/colourSchemes'
import { useExport } from '@/lib/data'
import { fmt } from '@/lib/format'
import { saturate } from '@/lib/hipsLinearise'
import { curveModel, fitModel, inverse, twoZone, WHITE_ALPHA } from '@/lib/hipsPattern'
import { RESEARCH_SERIES } from '@/lib/palettes'
import { FONT, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { pathOf, range, type Ceil, type Linearise } from '@/pages/HipsLinearisePage'
import { DataTable, MEASURED, Readouts } from '@/pages/hipsLineariseParts'
import { HALF, line, macText, pct } from '@/pages/HipsWarrenParts'

/**
 * How the IMPROVE groups behind the ceiling are made, and what they hold by period and season (24 Nov 2026): the groups
 * for two periods with the curve through each, every group in every year, the fitted bend in five-year windows, the same
 * under other groupings, and what each choice does at Addis. Data: gallery/data/export_hips_improve_periods.py.
 */
const FILE = 'hips_linearise/improve_periods.json'
const COMMAND = 'uv run python gallery/data/export_hips_improve_periods.py'
type Season = 'all' | 'rest' | 'fire'
type Era = 'old' | 'new'
interface Group { g: number; n: number; loading_ug: number; tau: number }
interface Fit { filters: number; sites: number; fabs_per_ec: number; a: number; tau_inf: number; rms_log: number; bend: number; bend_lo: number; bend_hi: number; groups: Group[] }
interface Cell extends Fit { season: Season; season_label: string; era: Era; era_label: string }
export interface Periods {
  n_boot: number; window: number; min_n: number; year_min_n: number; edges: number[]; area_cm2: number
  cells: Cell[]
  robust: { season: Season; era: Era; basis: string; label: string; filters: number; tau_inf: number; rms_log: number; bend: number; bend_lo: number; bend_hi: number }[]
  windows: { season: Season; from: number; to: number; filters: number; fabs_per_ec: number; tau_inf: number; bend: number; bend_lo: number; bend_hi: number; darkest: number }[]
  years: { season: Season; year: number; groups: [number, number, number, number][] }[]
  pooled: (Fit & { season: Season; label: string; models: { key: string; label: string; params: number[]; rms_log: number; aic: number }[] })[]
}
export const usePeriods = () => useExport<Periods>(FILE)
const SEASONS: readonly Season[] = ['all', 'rest', 'fire']
const SEASON_LABEL: Record<Season, string> = { all: 'All months', rest: 'November to June', fire: 'July to October' }
const SEASON_COLOR: Record<Season, string> = { all: INK.muted, rest: RESEARCH_SERIES[0], fire: RESEARCH_SERIES[3] }
const ERA_COLOR: Record<Era, string> = { old: RESEARCH_SERIES[4], new: RESEARCH_SERIES[1] }
const PROV = <><code>{FILE}</code> (<code>gallery/data/export_hips_improve_periods.py</code>); IMPROVE TOR EC and HIPS on standard 3.53 cm² deposits.</>
/** beside its pair on a desk, the card's full width on a phone */
const PLOT = { flex: '1 1 300px', minWidth: 0, maxWidth: HALF + 80 } as const
const useSeason = () => useSearchParam<Season>('wp.season', 'all', SEASONS)
const edgeText = (edges: number[], g: number) => (edges[g + 1] >= 1000 ? `over ${edges[g]}` : `${edges[g]}–${edges[g + 1]}`)
const perEc = (tau: number, loading: number, area: number) => (tau * 100 * area) / loading
/** a ceiling, or "none" when the bend is so slight that the fit runs to its bound */
const at = (c: number) => (c >= 98 ? 'none' : fmt(c, 2))
const ceilings = (r: { tau_inf: number; bend_lo: number; bend_hi: number }) => `τ∞ ${at(r.tau_inf)} [${fmt(1 / r.bend_hi, 2)}, ${r.bend_lo > 0.011 ? fmt(1 / r.bend_lo, 2) : 'none'}]`

export function PeriodCharts({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const { data: p, error } = usePeriods()
  const [season, setSeason] = useSeason()
  if (!p) return <ChartFrame id="wp-groups" title="The IMPROVE groups by period" provenance={PROV}><LoadState what="the groups by period" error={error} command={COMMAND} /></ChartFrame>
  return <>
    <GroupsByPeriod p={p} season={season} setSeason={setSeason} />
    <GroupYears p={p} season={season} setSeason={setSeason} />
    <BendOverTime p={p} ceil={ceil} />
    <OtherGroupings p={p} ceil={ceil} />
    <SeasonAtAddis p={p} data={data} ceil={ceil} />
  </>
}

const SeasonPick = ({ value, onChange }: { value: Season; onChange: (s: Season) => void }) =>
  <Segmented label="Months" value={value} options={SEASONS} onChange={onChange} optionLabel={(s) => SEASON_LABEL[s]} title="The months the groups are made from; the year map below follows" />

// ------------------------------------------------------------------ 1. the groups themselves
function GroupsByPeriod({ p, season, setSeason }: { p: Periods; season: Season; setSeason: (s: Season) => void }) {
  const cells = p.cells.filter((c) => c.season === season)
  const counts = useMemo<ScatterPoint[]>(() => cells.flatMap((c) => c.groups.map((g) => ({
    key: `${c.era}/${g.g}`, x: g.g + (c.era === 'old' ? 0.33 : 0.67), y: g.n, color: ERA_COLOR[c.era], r: 4, opacity: 0.95,
    tip: [`Group ${g.g + 1}: previous filter ${edgeText(p.edges, g.g)} µg`, c.era_label, `${g.n.toLocaleString()} filters`, `median this filter ${fmt(g.loading_ug, 1)} µg, τ ${fmt(g.tau, 3)}`],
  }))), [p, season]) // eslint-disable-line react-hooks/exhaustive-deps
  const stems = ({ x, y, iw, ih }: ScatterScales) => <g pointerEvents="none">
    {cells.flatMap((c) => c.groups.map((g) => <line key={`${c.era}/${g.g}`} x1={x(g.g + (c.era === 'old' ? 0.33 : 0.67))} x2={x(g.g + (c.era === 'old' ? 0.33 : 0.67))} y1={ih} y2={y(g.n)} stroke={ERA_COLOR[c.era]} strokeWidth={6} opacity={0.35} />))}
    <Marker orient="h" at={y(p.min_n)} from={0} to={iw} color={INK.text} label={`left out under ${p.min_n}`} />
  </g>
  const medians = useMemo<ScatterPoint[]>(() => cells.flatMap((c) => c.groups.map((g) => ({
    key: `${c.era}/${g.g}`, x: g.loading_ug, y: g.tau, color: ERA_COLOR[c.era], r: 3 + Math.sqrt(g.n) / 40, opacity: 0.9,
    tip: [`Group ${g.g + 1}, ${c.era_label}`, `${g.n.toLocaleString()} filters whose previous filter held ${edgeText(p.edges, g.g)} µg`, `median TOR EC ${fmt(g.loading_ug, 1)} µg, median τ ${fmt(g.tau, 3)}`, `absorption per EC ${fmt(perEc(g.tau, g.loading_ug, p.area_cm2), 1)} m²/g`],
  }))), [p, season]) // eslint-disable-line react-hooks/exhaustive-deps
  const top = 72
  const curves = ({ x, y, iw, ih }: ScatterScales) => <g pointerEvents="none">
    <clipPath id="wp-groups-clip"><rect width={iw} height={ih} /></clipPath>
    <g clipPath="url(#wp-groups-clip)">{cells.map((c) => <g key={c.era} fill="none" stroke={ERA_COLOR[c.era]}>
      <path d={pathOf([0, top].map((v) => [x(v), y(c.a * v)]))} strokeWidth={1.25} strokeDasharray="3 4" opacity={0.7} />
      <path d={pathOf(range(0, top, 0.5).map((v) => [x(v), y(saturate(c.a * v, c.tau_inf))]))} strokeWidth={2.25} />
    </g>)}</g>
  </g>
  const rest = p.cells.filter((c) => c.season === 'rest'), fire = p.cells.filter((c) => c.season === 'fire')
  return (
    <ChartFrame id="wp-groups" title="How the groups are made, and what each period puts in them — the two periods agree in November to June and part in July to October"
      subtitle={`Every filter goes into one of ${p.edges.length - 1} groups by the TOR EC on the previous filter at its site (left: edges in µg, the same for every period), so a filter's own TOR error cannot choose its group. Each group then gives one dot (right): the median TOR EC and the median HIPS optical depth of the filters in it. The curve goes through the dots; dashed is the straight line it starts on, so the gap between them is the bend.`}
      controls={<SeasonPick value={season} onChange={setSeason} />} provenance={PROV}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 24, justifyContent: 'center' }}>
        <div style={PLOT}><ScatterPlot title="Filters in each group" points={counts} xLabel="Previous filter's TOR EC (µg): the group's edges" yLabel="Filters (log scale)" yScale="log" xDomain={[0, p.edges.length - 1]} yDomain={[100, 60000]} nice={false}
          xTicks={17} xFormat={(v: number) => (v % 2 === 0 && v < p.edges.length - 1 ? String(p.edges[v]) : '')} yFormat={(v: number) => ([100, 1000, 10000].includes(v) ? v.toLocaleString() : '')} maxSide={HALF} outside="pin" note={false} under={stems} /></div>
        <div style={PLOT}><ScatterPlot title={`Group medians, ${season === 'all' ? 'all months' : SEASON_LABEL[season]}`} points={medians} xLabel="TOR EC on the filter (µg), group median" yLabel="HIPS optical depth τ, group median" xDomain={[0, top]} yDomain={[0, 1.5]} nice={false}
          maxSide={HALF} outside="pin" note={false} under={curves} /></div>
      </div>
      <Key items={cells.map((c) => ({ label: `${c.era_label}: ${ceilings(c)}, ${c.filters.toLocaleString()} filters`, color: ERA_COLOR[c.era], kind: 'dot' as const }))} />
      <Readouts items={[
        ...rest.map((c) => ({ label: `November to June, ${c.era_label}`, value: `τ∞ ${at(c.tau_inf)}`, color: ERA_COLOR[c.era], sub: ceilings(c) })),
        ...fire.map((c) => ({ label: `July to October, ${c.era_label}`, value: `τ∞ ${at(c.tau_inf)}`, color: ERA_COLOR[c.era], sub: ceilings(c) })),
      ]} />
      <Note explain>A lower ceiling is a stronger bend. The intervals come from resampling whole sites {p.n_boot} times. The two periods are before and after the HIPS rebuild of the 2017 sample year.
        The rebuild moved HIPS by 5 to 6 %, which scales a curve and cannot bend it. July to October is tested because it holds most of the western fire season; the data do not say by themselves that smoke is the cause.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 2. every group, every year
function GroupYears({ p, season, setSeason }: { p: Periods; season: Season; setSeason: (s: Season) => void }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const colours = useColourScheme('wp-years', 'sequential')
  const rows = p.years.filter((y) => y.season === season)
  const nG = p.edges.length - 1
  const m = { top: 12, right: 12, bottom: 48, left: 74 }
  const iw = Math.max(260, Math.min(920, width || 800) - m.left - m.right), ch = 24, ih = ch * nG
  const cw = iw / rows.length
  const scale = d3.scaleSequential(d3.interpolateRgbBasis(colours.stops)).domain([5, 16]).clamp(true)
  const x = d3.scaleLinear().domain([rows[0].year - 0.5, rows[rows.length - 1].year + 0.5]).range([0, iw])
  const dark = p.years.filter((y) => y.season === 'fire').flatMap((y) => y.groups.filter((g) => g[0] >= 13).map((g) => ({ year: y.year, v: perEc(g[3], g[2], p.area_cm2) })))
  const low = d3.least(dark.filter((d) => d.year >= 2008), (d) => d.v)
  const first = p.years.filter((y) => y.season === 'rest').flatMap((y) => y.groups.filter((g) => g[0] >= 13).map((g) => perEc(g[3], g[2], p.area_cm2)))
  return (
    <ChartFrame id="wp-years" title={`Every group in every year, with no fit — the darkest groups of July to October fall furthest, down to ${low ? fmt(low.v, 1) : '—'} m²/g in ${low?.year ?? '—'}`}
      subtitle={`Colour: HIPS absorption per TOR EC in each group (median τ × 100 × ${p.area_cm2} ÷ median µg). If HIPS were proportional to EC, each column would be one colour from bottom to top; darker groups (top) reading lower is the bend. A column changing colour as a whole is a change of level that year, in HIPS or in TOR, not a bend. Groups under ${p.year_min_n} filters are empty.`}
      controls={<SeasonPick value={season} onChange={setSeason} />} provenance={PROV}>
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={iw + m.left + m.right} height={ih + m.top + m.bottom} fontFamily={FONT.family} style={{ display: 'block', margin: '0 auto' }}>
          <g transform={`translate(${m.left},${m.top})`}>
            {rows.flatMap((r, i) => {
              const by = new Map(r.groups.map((g) => [g[0], g]))
              return d3.range(nG).map((g) => {
                const c = by.get(g), v = c ? perEc(c[3], c[2], p.area_cm2) : null
                return <g key={`${r.year}/${g}`} {...(c ? tip.bind([`${r.year}, ${SEASON_LABEL[season]}`, `previous filter ${edgeText(p.edges, g)} µg`, `${c[1].toLocaleString()} filters`, `median TOR EC ${fmt(c[2], 1)} µg, τ ${fmt(c[3], 3)}`, `${fmt(v!, 1)} m²/g`]) : {})}>
                  <HeatCell x={i * cw} y={ih - (g + 1) * ch} width={cw - 1} height={ch - 1} fill={v == null ? null : scale(v)} label={v != null && cw > 30 ? fmt(v, 0) : null} />
                </g>
              })
            })}
            <XAxis scale={x} y={ih} tickCount={cw > 30 ? rows.length : 6} label="Year" labelDy={38} format={(v) => String(v)} />
            {d3.range(nG).map((g) => <text key={g} x={-8} y={ih - (g + 0.5) * ch} dy="0.35em" textAnchor="end" fontSize={11} fontFamily={FONT.mono} fill={INK.muted}>{edgeText(p.edges, g)}</text>)}
            <text transform={`translate(${-62},${ih / 2}) rotate(-90)`} textAnchor="middle" fontSize={12} fontWeight={600} fill={INK.text}>Group: previous filter’s TOR EC (µg)</text>
          </g>
        </svg>
        {tip.node}
      </div>
      <div className="legend" style={{ justifyContent: 'center' }}><ColorLegend scale={scale} scheme={colours.scheme} label="HIPS absorption per TOR EC (m²/g)" width={220} format={(v) => fmt(v, 0)} /></div>
      <Note>In November to June the groups past 36 µg read {fmt(d3.min(first)!, 1)} to {fmt(d3.max(first)!, 1)} m²/g. The whole-column step between 2006 and 2008 and the paler columns from 2021 in July to October change whole columns, so they are level changes, not bends. Whether HIPS or TOR moved is not known.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 3. the bend over time
function BendOverTime({ p, ceil }: { p: Periods; ceil: Ceil }) {
  const mid = (w: Periods['windows'][number]) => (w.from + w.to) / 2
  const points = useMemo<ScatterPoint[]>(() => p.windows.map((w) => ({ key: `${w.season}/${w.from}`, x: mid(w), y: w.bend, color: SEASON_COLOR[w.season], r: 4, opacity: 0.95,
    tip: [`${w.from}–${w.to}, ${SEASON_LABEL[w.season]}`, ceilings(w), `${w.filters.toLocaleString()} filters`, `darkest group τ ${fmt(w.darkest, 2)}`, `absorption per EC before the bend ${fmt(w.fabs_per_ec, 1)} m²/g`] })), [p])
  const under = ({ x, y, iw }: ScatterScales) => <g pointerEvents="none">
    {SEASONS.map((s) => { const ws = p.windows.filter((w) => w.season === s); return <g key={s}>
      <path d={pathOf([...ws.map((w): [number, number] => [x(mid(w)), y(w.bend_lo)]), ...[...ws].reverse().map((w): [number, number] => [x(mid(w)), y(w.bend_hi)])]) + 'Z'} fill={SEASON_COLOR[s]} opacity={s === 'all' ? 0 : 0.13} />
      <path d={pathOf(ws.map((w) => [x(mid(w)), y(w.bend)]))} fill="none" stroke={SEASON_COLOR[s]} strokeWidth={s === 'all' ? 1.5 : 2.25} strokeDasharray={s === 'all' ? '5 4' : undefined} /></g> })}
    <Marker orient="h" at={y(1 / ceil.ceiling)} from={0} to={iw} color={MEASURED} label={`page τ∞ ${fmt(ceil.ceiling, 2)}`} labelSide="left" />
  </g>
  const last = p.windows.filter((w) => w.from === 2019)
  const f = last.find((w) => w.season === 'fire'), r = last.find((w) => w.season === 'rest')
  return (
    <ChartFrame id="wp-time" title="The bend over the years — it grows in July to October from about 2016 and eases in the last window; November to June rises less and stays near the page's value"
      subtitle={`The strength of the bend, 1/τ∞, in ${p.window}-year windows, at the window's middle year: 0 is a straight line, and higher means HIPS reads lower on dark filters. Shaded: 90 % intervals from resampling whole sites. Neighbouring windows share years, so they are not separate results. Single years are too noisy to fit, so the year map above shows them without one.`} provenance={PROV}>
      <ScatterPlot points={points} xLabel={`Middle of the ${p.window}-year window`} yLabel="Strength of the bend, 1/τ∞" xDomain={[2009.5, 2023.5]} yDomain={[0, 1.05]} nice={false} height={340} xFormat={(v: number) => String(v)} outside="pin" note={false} under={under} />
      <Key items={SEASONS.map((s) => ({ label: SEASON_LABEL[s], color: SEASON_COLOR[s], kind: s === 'all' ? 'dash' as const : 'line' as const }))} />
      {f && r && <Note>The rise is not only seasonal: in 2019–2023 the fit is {ceilings(f)} in July to October and {ceilings(r)} in November to June. The last windows also reach less far: their darkest groups are near τ 0.4 to 0.5 in July to October, so the curve leans on a short range.</Note>}
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 4. the same under other groupings
function OtherGroupings({ p, ceil }: { p: Periods; ceil: Ceil }) {
  const blocks = (['rest', 'fire'] as Season[]).map((s) => ({ key: s, title: SEASON_LABEL[s], color: SEASON_COLOR[s],
    rows: [...new Set(p.robust.map((r) => r.basis))].map((b) => {
      const rs = p.robust.filter((r) => r.season === s && r.basis === b)
      return { key: `${s}/${b}`, label: rs[0].label, strong: b === 'previous_ug',
        marks: [rs.map((r, i) => ({ mid: r.bend, lo: r.bend_lo, hi: r.bend_hi, color: ERA_COLOR[r.era], dy: i ? 4 : -4, faint: b === 'loading_ug' }))],
        tip: [rs[0].label, SEASON_LABEL[s], ...rs.map((r) => `${r.era === 'old' ? '2008–2016' : '2017–2025'}: ${ceilings(r)}, typical miss ${pct(r.rms_log)}`)] }
    }) }))
  return (
    <ChartFrame id="wp-othergroups" title="Grouping the filters another way does not remove the July to October bend, or make the two periods of November to June differ"
      subtitle="The strength of the bend 1/τ∞ when filters are grouped by the filter one, two or three samples before, by the next one, or by their own TOR EC, for the same two periods. Only the last sorts filters by the TOR error itself, which makes any bend look stronger, so it is drawn faintly." provenance={PROV}>
      <IntervalStrip blocks={blocks} panels={[{ label: 'Strength of the bend, 1/τ∞', domain: [0, 1], marker: { value: 1 / ceil.ceiling, label: `page ${fmt(ceil.ceiling, 2)}` } }]} labelWidth={200} rowHeight={28} />
      <Key items={[{ label: '2008–2016', color: ERA_COLOR.old, kind: 'dot' }, { label: '2017–2025', color: ERA_COLOR.new, kind: 'dot' }]} />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 5. what it does to the model and to Addis
function SeasonAtAddis({ p, data, ceil }: { p: Periods; data: Linearise; ceil: Ceil }) {
  const rows = useMemo(() => {
    const fit = (c: number) => fitModel(data.filters, data.lambda, curveModel('k', '', '', c), data.mac)
    const none = fitModel(data.filters, data.lambda, { invert: (t) => t }, data.mac)
    const ten = fitModel(data.filters, data.lambda, { invert: inverse((x) => twoZone(x, WHITE_ALPHA)) }, data.mac)
    const bySeason = p.pooled.filter((c) => c.season !== 'all').map((c) => ({ key: c.season as string, label: `${c.label}, 2008–2025`, c: c.tau_inf, text: ceilings(c), f: fit(c.tau_inf), lo: fit(1 / c.bend_hi), hi: c.bend_lo > 0.011 ? fit(1 / c.bend_lo) : null,
      models: c.models }))
    return { none, ten, bySeason, page: fit(ceil.ceiling) }
  }, [p, data, ceil.ceiling])
  const body = [
    { key: 'none', label: 'As measured', text: '—', f: rows.none, range: '' },
    { key: 'ten', label: `White 2025, α = ${WHITE_ALPHA}`, text: 'from CSN, not IMPROVE TOR', f: rows.ten, range: '' },
    ...rows.bySeason.map((r) => ({ key: r.key, label: r.label, text: r.text, f: r.f, range: r.lo && r.hi ? `${line(r.lo)} to ${line(r.hi)}; MAC ${macText(r.hi)} to ${macText(r.lo)}` : '' })),
    { key: 'page', label: 'The ceiling this page uses, all months', text: `τ∞ ${fmt(ceil.ceiling, 2)}`, f: rows.page, range: '' },
  ]
  const alpha = (s: string, k: string) => rows.bySeason.find((r) => r.key === s)!.models.find((m) => m.key === k)!
  return (
    <ChartFrame id="wp-season-addis" title="What the months do at Addis — the November to June curve agrees with Warren's α = 10 and leaves a negative intercept; the July to October one cannot be used there"
      subtitle="The Addis FTIR EC on HIPS BC fit after a correction fitted to IMPROVE in each season, over the standard deposits of 2008–2025. The last column is the same fit at the ends of the ceiling's interval." provenance={PROV}>
      <DataTable rows={body} rowKey={(r) => r.key} highlight={(r) => r.key === 'rest'} columns={[
        { key: 'l', label: 'Correction', value: (r) => r.label }, { key: 'c', label: 'Ceiling', value: (r) => r.text },
        { key: 'f', label: 'Addis fit', value: (r) => line(r.f) }, { key: 'm', label: 'MAC for slope 1', value: (r) => macText(r.f), num: true }, { key: 'r', label: 'Over the interval', value: (r) => r.range },
      ]} />
      <Readouts items={[
        { label: 'November to June: α that fits', value: fmt(alpha('rest', 'free').params[1], 1), sub: `Warren’s network value is ${WHITE_ALPHA}` },
        { label: 'Typical miss, α = 10', value: pct(alpha('rest', 'alpha10').rms_log, 1), sub: `our curve ${pct(alpha('rest', 'curve').rms_log, 1)}, November to June` },
        { label: 'July to October: α that fits', value: alpha('fire', 'free').params[1] > 1000 ? 'over 1,000' : fmt(alpha('fire', 'free').params[1], 1), sub: `α = 10 misses by ${pct(alpha('fire', 'alpha10').rms_log, 0)}` },
      ]} />
      <Note explain>So the screen pattern is enough for the November to June groups, and not for the July to October ones. The page's ceiling, {fmt(ceil.ceiling, 2)}, is fitted to both together. Addis's own filters are from
        another kind of sampler and another kind of air, so neither season of IMPROVE is the right one by itself. The July to October curve has a ceiling so near Addis's own filters that it over-corrects them to a flat line, so it cannot be a correction for Addis. The others put the Addis MAC at about 10 to 18 and the intercept between about −1.8 and 0.</Note>
    </ChartFrame>
  )
}
