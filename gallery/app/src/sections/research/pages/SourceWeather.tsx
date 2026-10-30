import { useMemo, useRef } from 'react'
import { useColourScheme } from '@/lib/colourSchemes'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Segmented, Select } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { Crossplot } from '@/components/Crossplot'
import { HeatCell } from '@/components/HeatCell'
import { IntervalStrip, type IntervalRow } from '@/components/IntervalStrip'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useExport } from '@/lib/data'
import { fmt } from '@/lib/format'
import { ExcludedNote, useMarkedRows } from '@/lib/markedRows'
import { FONT, INK } from '@/lib/theme'
import type { FilterRow, PmfFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import type { SectionPageProps } from '../../types'
import { useAddisMet } from '../common/met'
import { ols, quantile, spearman } from '../common/stats'
import { DataTable, Grid, num, point } from '../common/ui'
import { PAIRS } from './RhSweep'

const MET_NB = 'notebooks/analysis/meteorology/meteorology_source_interaction.ipynb'
const DT_NB = 'notebooks/analysis/meteorology/temperature_gradient_analysis.ipynb'
const FRIDAY_NB = 'notebooks/analysis/meteorology/friday_summary_consolidated.ipynb'
const RAIN = 'Precipitation (station)', RAIN_ERA5 = 'Precipitation (ERA5)'
const DT_WINDOW = 'Temperature range (station)', T_WINDOW = 'Temperature (station)'
const DT_DAY = 'Temperature range (station, sample day)', TMIN_DAY = 'Minimum temperature (station, sample day)', TAVG_DAY = 'Mean temperature (station, sample day)'
const GROUPS: Record<string, string[]> = { 'Biomass burning': ['Charcoal', 'Wood Burning'], 'Fossil fuel combustion': ['Fossil Fuel Combustion', 'Polluted Marine'], 'Sea salt': ['Sea Salt Mixed'] }
const DOMINANCE = [0.3, 0.4, 0.5, 0.6]

interface Joined { r: FilterRow; ug: Record<string, number>; total: number }

/** the PMF site's filters with a PMF day (by filter id, else by date), each factor as K_F in µg/m³ */
function joinPmf(rows: FilterRow[], pmf: PmfFile | null): Joined[] {
  if (!pmf) return []
  const byId = new Map(pmf.rows.map((p) => [p.id, p])), byDate = new Map(pmf.rows.map((p) => [p.date, p]))
  return rows.filter((r) => r.code === pmf.site).flatMap((r) => {
    const p = byId.get(r.id) ?? byDate.get(r.date?.slice(0, 10))
    if (!p) return []
    const ug = Object.fromEntries(pmf.sources.map((s) => [s.label, p.ugm3?.[s.label] ?? NaN]))
    const total = Object.values(ug).filter(Number.isFinite).reduce((a, b) => a + b, 0)
    return [{ r, ug, total }]
  })
}

/** a factor × weather Spearman matrix, drawn as heat cells */
function Matrix({ rows, cols, cell }: { rows: string[]; cols: string[]; cell: (r: string, c: string) => { rho: number | null; n: number } }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const labelW = 170, top = 120
  const cw = Math.max(34, Math.min(70, (width - labelW - 10) / Math.max(1, cols.length))), ch = 26
  const colours = useColourScheme('sw-matrix', 'diverging')
  const color = d3.scaleQuantize<string>().domain([-1, 1]).range(colours.stops)
  return (
    <div ref={ref} style={{ position: 'relative', overflowX: 'auto' }}>
      <svg width={labelW + cols.length * cw + 90} height={top + rows.length * ch + 6} fontFamily={FONT.family}>
        {cols.map((c, j) => <text key={c} transform={`translate(${labelW + j * cw + cw / 2},${top - 6}) rotate(-50)`} fontSize={11} fill={INK.text}>{c}</text>)}
        {rows.map((r, i) => <g key={r}>
          <text x={labelW - 6} y={top + i * ch + ch / 2} dy="0.35em" textAnchor="end" fontSize={11.5} fill={INK.text}>{r}</text>
          {cols.map((c, j) => { const v = cell(r, c)
            return <HeatCell key={c} x={labelW + j * cw} y={top + i * ch} width={cw - 2} height={ch - 2} fill={v.rho === null ? null : color(v.rho)} label={v.rho === null ? '' : fmt(v.rho, 2)}
              onEnter={(e) => tip.show(e, [`${r} × ${c}`, `Spearman ρ ${fmt(v.rho, 3)}`, `n ${v.n}`])} onLeave={tip.hide} /> })}
        </g>)}
      </svg>
      <ColorLegend scale={color} scheme={colours.scheme} label="Spearman ρ" format={(v) => fmt(v, 1)} />
      {tip.node}
    </div>
  )
}

/**
 * PMF factors against the weather and the method agreement by source (Addis): factor × meteorology Spearman
 * correlations, rain washout by factor, the temperature range against each factor, and HIPS / FTIR /
 * aethalometer slopes by dominant source and by dominance threshold, with factors as concentrations (K_F),
 * not the GF fractions the meeting set aside.
 */
export function SourceWeather({ rows: viewRows, meta: viewMeta }: SectionPageProps) {
  const { rows, meta } = useAddisMet(viewRows, viewMeta)
  const pmf = useExport<PmfFile>('pmf.json').data
  const [basis, setBasis] = useSearchParam<'conc' | 'share'>('sw.basis', 'conc', ['conc', 'share'])
  const [factorParam, setFactor] = useSearchParam<string>('sw.factor', 'Charcoal')
  const [grouping, setGrouping] = useSearchParam<'separate' | 'grouped'>('sw.group', 'grouped', ['separate', 'grouped'])
  const factors = useMemo(() => pmf?.sources.map((s) => s.label) ?? [], [pmf])
  const factorColor = (f: string) => pmf?.sources.find((s) => s.label === f)?.color ?? INK.muted
  const factor = factors.includes(factorParam) ? factorParam : factors[0] ?? ''
  const joined: Joined[] = useMemo(() => joinPmf(rows, pmf), [rows, pmf])
  // Filters → excluded: Mark (lib/markedRows): the excluded filters in view with a PMF day, joined the same way; × on
  // the crossplot, a count under the summaries
  const markedMet = useAddisMet(useMarkedRows(), viewMeta).rows
  const markedJoined: Joined[] = useMemo(() => joinPmf(markedMet, pmf), [markedMet, pmf])
  const value = (j: Joined, f: string) => { const v = j.ug[f]; return Number.isFinite(v) ? (basis === 'conc' ? v : j.total > 0 ? v / j.total : NaN) : NaN }
  const unit = basis === 'conc' ? 'µg/m³' : 'share of the PMF total'
  const weather = useMemo(() => meta.fields.filter((f) => /\((station|ERA5)(, sample day)?\)$|Precipitable water/.test(f) && joined.some((j) => num(j.r, f) !== null)), [meta.fields, joined])

  // before the early returns below: a hook must run on every render
  const tColours = useColourScheme('sw-dt', 'diverging', 'Red–yellow–blue')
  if (!pmf) return <ChartFrame id="sw-load" title="PMF sources and the weather" provenance={MET_NB}><Empty>Loading the PMF factors…</Empty></ChartFrame>
  if (joined.length < 10) return <ChartFrame id="sw-empty" title="PMF sources and the weather" provenance={MET_NB}><Empty>The PMF solution covers {pmf.site_name} only; pick it in the Filters panel ({joined.length} of its filters in view).</Empty></ChartFrame>

  // washout: the notebook's rain classes, on the window-mean station rain (ERA5 where the station has none)
  const rainField = joined.some((j) => num(j.r, RAIN) !== null) ? RAIN : RAIN_ERA5
  const classes = [{ k: 'Dry', lo: -Infinity, hi: 0.1 }, { k: 'Light (0.1–2)', lo: 0.1, hi: 2 }, { k: 'Moderate (2–10)', lo: 2, hi: 10 }, { k: 'Heavy (≥ 10)', lo: 10, hi: Infinity }]
  const rainRows: IntervalRow[] = classes.map((c) => {
    const g = joined.filter((j) => { const v = num(j.r, rainField); return v !== null && v >= c.lo && v < c.hi })
    return { key: c.k, label: c.k, note: `n=${g.length}`, marks: factors.map((f) => { const v = g.map((j) => value(j, f)).filter(Number.isFinite)
      return v.length ? [{ mid: quantile(v, 0.5), lo: quantile(v, 0.25), hi: quantile(v, 0.75), color: factorColor(f) }] : [] }) }
  })
  // ΔT against the chosen factor, coloured by the coldest hour: the notebooks' sample-day station values where the export has them
  const sampleDay = joined.some((j) => num(j.r, DT_DAY) !== null)
  const DT = sampleDay ? DT_DAY : DT_WINDOW, T = sampleDay ? TMIN_DAY : T_WINDOW
  const dtOf = (js: Joined[]) => js.flatMap((j) => { const x = num(j.r, DT), t = num(j.r, T), y = value(j, factor); return x !== null && Number.isFinite(y) ? [{ j, x, y, t }] : [] })
  const dtPts = dtOf(joined), dtMarked = dtOf(markedJoined)
  const tExtent = d3.extent(dtPts.map((p) => p.t).filter((v): v is number => v !== null)) as [number, number]
  const tColor = d3.scaleSequential(d3.interpolateRgbBasis(tColours.stops)).domain([tExtent[0] ?? 0, tExtent[1] ?? 1])
  const rhoDt = factors.map((f) => ({ f, dt: spearman(joined.map((j) => num(j.r, DT) ?? NaN), joined.map((j) => value(j, f))), t: spearman(joined.map((j) => num(j.r, sampleDay ? TAVG_DAY : T_WINDOW) ?? NaN), joined.map((j) => value(j, f))),
    tmin: sampleDay ? spearman(joined.map((j) => num(j.r, TMIN_DAY) ?? NaN), joined.map((j) => value(j, f))) : null }))
  // slopes by source: the dominant factor (or group) of each day, by K_F concentration
  const dominant = (j: Joined) => {
    if (grouping === 'separate') return factors.reduce((a, f) => ((j.ug[f] ?? -1) > (j.ug[a] ?? -1) ? f : a), factors[0])
    const sums = Object.entries(GROUPS).map(([g, fs]) => [g, fs.reduce((s, f) => s + (Number.isFinite(j.ug[f]) ? j.ug[f] : 0), 0)] as const)
    return sums.reduce((a, b) => (b[1] > a[1] ? b : a))[0]
  }
  const shareOf = (j: Joined, key: string) => {
    const fs = grouping === 'separate' ? [key] : GROUPS[key]
    return j.total > 0 ? fs.reduce((s, f) => s + (Number.isFinite(j.ug[f]) ? j.ug[f] : 0), 0) / j.total : NaN
  }
  const keys = grouping === 'separate' ? factors : Object.keys(GROUPS)
  const pairs = PAIRS.filter((p) => meta.fields.includes(p.x) && meta.fields.includes(p.y))
  const nPairsOut = markedJoined.filter((j) => pairs.some((p) => num(j.r, p.x) !== null && num(j.r, p.y) !== null)).length
  const fitOf = (js: Joined[], p: (typeof PAIRS)[number]) => ols(js.map((j) => num(j.r, p.x) ?? NaN), js.map((j) => num(j.r, p.y) ?? NaN))
  const bySource = keys.flatMap((k) => pairs.map((p) => ({ k, p, f: fitOf(joined.filter((j) => dominant(j) === k), p) })))
  const byDominance = keys.flatMap((k) => DOMINANCE.map((t) => { const js = joined.filter((j) => shareOf(j, k) >= t); return { k, t, n: js.length, f: pairs[0] ? fitOf(js, pairs[0]) : null } }))

  return (
    <>
      <ChartFrame id="sw-matrix" title={`PMF factors against the weather — Spearman ρ, ${pmf.site_name}`}
        subtitle={`Each factor's daily ${basis === 'conc' ? 'concentration (K_F, µg/m³)' : 'share of the PMF total (K_F / Σ K_F)'} against the weather over the same filter window (Addis station and ERA5 means, AERONET precipitable water), on the ${joined.length} filters that have a PMF day.`}
        provenance={`${MET_NB} · ${FRIDAY_NB} (K_F, not GF)`}
        controls={<Segmented label="Factor as" value={basis} options={['conc', 'share'] as const} onChange={setBasis} optionLabel={(o) => (o === 'conc' ? 'µg/m³' : 'Share')} />}>
        <Matrix rows={factors} cols={weather} cell={(f, w) => { const ok = joined.filter((j) => num(j.r, w) !== null && Number.isFinite(value(j, f)))
          return { rho: ok.length >= 10 ? spearman(ok.map((j) => num(j.r, w)!), ok.map((j) => value(j, f))) : null, n: ok.length } }} />
        <ExcludedNote count={markedJoined.length} />
      </ChartFrame>

      <ChartFrame id="sw-rain" title="Rain washout by source — factor levels in dry to heavy-rain windows"
        subtitle={`Median and interquartile range of each factor (${unit}) by rain class on ${rainField}. The notebook binned daily station totals (0.1, 2, 10 mm); here the class is the window's mean rain in mm/day, so heavy single days are diluted over a multi-day filter.`}
        provenance={`${MET_NB} (§1 precipitation washout by source)`}>
        <IntervalStrip blocks={[{ key: 'rain', rows: rainRows }]} panels={factors.map((f) => ({ label: f, heading: f }))} labelWidth={120} noteWidth={50} minPanelWidth={150} />
        <ExcludedNote count={markedJoined.filter((j) => num(j.r, rainField) !== null).length} />
      </ChartFrame>

      <ChartFrame id="sw-dt" title={`Diurnal temperature range against ${factor} — coloured by the mean temperature`}
        subtitle={`A narrow temperature range suggests a stable, poorly mixed boundary layer; cold mornings suggest heating with charcoal and wood. Points coloured from cold (blue) to warm (red) by ${T}. ${sampleDay ? 'Temperatures are the Addis station\'s day on the sample date, as the notebooks join them (T_max − T_min, T_min).' : 'The notebooks used the station day on the sample date; without that export the window means stand in.'}`}
        provenance={`${DT_NB} (§1–3) · ${MET_NB} (§3 temperature range)`}
        controls={<Select label="Factor" value={factor} options={factors} onChange={setFactor} />}>
        <Grid>
          {dtPts.length >= 5 ? <Crossplot points={dtPts.map((p) => point(p.j.r, p.x, p.y, p.t === null ? INK.muted : tColor(p.t), 'ΔT', factor, 'Research · PMF and weather', [`T ${fmt(p.t, 1)} °C`]))} sameScale={false}
            marked={dtMarked.map((p) => point(p.j.r, p.x, p.y, INK.muted, 'ΔT', factor, 'Research · PMF and weather', [`T ${fmt(p.t, 1)} °C`]))}
            title={`${factor} vs ΔT`} xLabel={`${DT} (°C)`} yLabel={`${factor} (${unit})`} fit={false} /> : <Empty>Too few filters with {DT}.</Empty>}
          <DataTable rows={rhoDt} rowKey={(r) => r.f} columns={[
            { key: 'f', label: 'Factor', value: (r) => <span style={{ color: factorColor(r.f), fontWeight: 600 }}>{r.f}</span> },
            { key: 'dt', label: 'ρ with ΔT', num: true, value: (r) => fmt(r.dt, 2) }, { key: 't', label: 'ρ with mean T', num: true, value: (r) => fmt(r.t, 2) },
            ...(sampleDay ? [{ key: 'tmin', label: 'ρ with T_min', num: true, value: (r: (typeof rhoDt)[number]) => fmt(r.tmin, 2) }] : [])]} />
        </Grid>
        <ExcludedNote count={dtMarked.length} drawn />
        {dtPts.length >= 5 && <ColorLegend scale={tColor as unknown as { domain(): number[]; (v: number): string }} scheme={tColours.scheme} label={`${T} (°C)`} format={(v) => fmt(v, 0)} />}
      </ChartFrame>

      <ChartFrame id="sw-slopes" title="Method agreement by source — OLS slopes on the days each source dominates"
        subtitle={`Each filter assigned to its largest factor${grouping === 'grouped' ? ' group (biomass = charcoal + wood; fossil = fossil fuel + polluted marine; sea salt)' : ''} by K_F concentration, then an OLS fit per pairing. Below: filters on which a source holds at least 30–60 % of the PMF total, and the HIPS vs FTIR fit there.`}
        provenance={`${FRIDAY_NB} (§3–5 grouping, cross-plots by source, dominance thresholds)`}
        controls={<Segmented label="Sources" value={grouping} options={['grouped', 'separate'] as const} onChange={setGrouping} />} exportable={false}>
        <DataTable rows={bySource} rowKey={(r) => `${r.k}-${r.p.key}`} columns={[
          { key: 'k', label: 'Dominant source', value: (r) => r.k }, { key: 'p', label: 'Pairing', value: (r) => r.p.label },
          { key: 'n', label: 'n', num: true, value: (r) => r.f?.n ?? '—' }, { key: 's', label: 'Slope', num: true, value: (r) => fmt(r.f?.slope, 3) },
          { key: 'i', label: 'Intercept', num: true, value: (r) => fmt(r.f?.intercept, 3) }, { key: 'r2', label: 'R²', num: true, value: (r) => fmt(r.f?.r2, 3) }]} />
        <p className="subtitle" style={{ margin: '14px 0 4px' }}>Dominance thresholds — {pairs[0]?.label ?? 'no pairing'}</p>
        <DataTable rows={byDominance} rowKey={(r) => `${r.k}-${r.t}`} columns={[
          { key: 'k', label: 'Source', value: (r) => r.k }, { key: 't', label: 'Share ≥', num: true, value: (r) => `${r.t * 100} %` }, { key: 'n', label: 'n', num: true, value: (r) => r.n },
          { key: 's', label: 'Slope', num: true, value: (r) => fmt(r.f?.slope, 3) }, { key: 'i', label: 'Intercept', num: true, value: (r) => fmt(r.f?.intercept, 3) }, { key: 'r2', label: 'R²', num: true, value: (r) => fmt(r.f?.r2, 3) }]} />
        <ExcludedNote count={nPairsOut} />
        <Note>Fits with a handful of filters are shown as they come out; read the n before the slope.</Note>
      </ChartFrame>
    </>
  )
}
