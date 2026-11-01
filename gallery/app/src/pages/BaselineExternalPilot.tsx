import { dataUrl, useExport } from '@/lib/data'
import { ExportState } from '@/components/ExportState'
import { useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, ExplainOnly, Note, Segmented } from '@/components/ChartFrame'
import { BaselinePicker } from '@/components/Pickers'
import { Crossplot } from '@/components/Crossplot'
import { ScatterPlot, type ScatterPoint } from '@/components/ScatterPlot'
import type { FitGrouping } from '@/lib/siteRegression'
import { Key } from '@/components/Key'
import { useBaselineStyle } from '@/lib/baselineRank'
import { INK, baselineColor } from '@/lib/theme'
import { fixed as fmt } from '@/lib/format'
import { type FilterRecord, useHighlightActions } from '@/lib/highlight'
import { useSearchParam } from '@/lib/url'
import { useRegressions } from '@/lib/regressionPrefs'
import { Combobox } from '@/components/Combobox'
import { ExcludedNote, FixedSetNote, useCountedFilter, useExclMode } from '@/lib/markedRows'

type Method = 'AIRSpec' | 'VIBES'
type Reference = 'ECTR' | 'ECTT'
type AdamaRow = {
  date: string
  FilterId_ptfe: string
  FilterId_quartz: string
  flag: string | null
  SampleAnalysisId_provisional: number
  ECTR: number
  ECTT: number
  AIRSpec_EC_ugm3: number
  VIBES_EC_ugm3: number
  AIRSpec_over_ECTR: number
  VIBES_over_ECTR: number
  AIRSpec_over_ECTT: number
  VIBES_over_ECTT: number
}
type MatchedRow = {
  bishoftu_filter_id: string
  addis_filter_id: string
  bishoftu_date: string
  addis_date: string
  Fabs_ratio: number
  bishoftu_AIRSpec_EC_ugm3: number
  addis_AIRSpec_EC_ugm3: number
  bishoftu_VIBES_EC_ugm3: number
  addis_VIBES_EC_ugm3: number
}
type SiteRow = {
  filter_id: string
  date: string
  lot: number | null
  Fabs: number | null
  AIRSpec_EC_ugm3: number | null
  VIBES_EC_ugm3: number | null
  AIRSpec_band1617: number
  AIRSpec_band2920: number
  VIBES_band1617: number
  VIBES_band2920: number
}
type Pilot = {
  background_rank?: number
  schema_version: number
  frozen_run_signature: string
  evidence_date: string
  adama: AdamaRow[]
  bishoftu: (Omit<SiteRow, 'filter_id' | 'lot'> & { ExternalFilterId: string; LotId: number })[]
  addis: (SiteRow & { sample_id: string })[]
  matched: MatchedRow[]
  mapping_ranges: { method: Method; reference: Reference; scope: string; min: number; max: number }[]
  limitations: { adama: string; bishoftu: string; matching: string }
}

const BASE = dataUrl('')
const METHODS: Method[] = ['AIRSpec', 'VIBES']
/** Archived runs keep the archived VIBES colour; the completed rank-70 pilot uses the current one. */
const colorsFor = (completed: boolean): Record<Method, string> => ({ AIRSpec: baselineColor('AIRSpec'), VIBES: baselineColor('VIBES', completed ? 70 : null) })

/**
 * `mac` is the reader's MAC (Filters → units; config.MAC_VALUE until changed) and `demingLambda` the
 * explorer's lambda* (calibration.json, at MAC 10) scaled to it by MAC² (lib/mac): passed in, not restated here.
 */
export function BaselineExternalPilot({ mac, demingLambda, completed = false }: { mac: number; demingLambda: number; completed?: boolean }) {
  const COLORS = colorsFor(completed)
  const hl = useHighlightActions()
  const { data, error } = useExport<Pilot>(completed ? 'rank70/baseline_external_pilot.json' : 'baseline_external_pilot.json', (d) =>
    d.schema_version !== 1 || (completed && d.background_rank !== 70) || d.adama.length !== 5 || d.bishoftu.length !== 26 || d.addis.length !== 253 || d.matched.length !== 5 ? 'unexpected pilot dataset' : null)
  const [reference, setReference] = useState<Reference>('ECTR')
  const [method, setMethod] = useState<Method>('AIRSpec')
  const [opticalMethod, setOpticalMethod] = useState<Method>('AIRSpec')
  const [spectralMethod, setSpectralMethod] = useState<Method>('AIRSpec')
  const [opticalSite, setOpticalSite] = useState('Both sites')
  const [spectralSite, setSpectralSite] = useState('Both sites')
  if (!data) return <ExportState title="External Ethiopia pilot" what="the Adama and Bishoftu pilot" error={error} command="uv run --no-sync python gallery/data/export_baseline_external_pilot.py" />
  const unflagged = data.adama.filter(r => !r.flag)
  const ratio = (r: AdamaRow, m: Method) => r[`${m}_over_${reference}`]
  const range = (m: Method) => data.mapping_ranges.find(r => r.method === m && r.reference === reference && r.scope === 'three_sampling_unflagged')!
  const med = (m: Method) => d3.median(unflagged, r => ratio(r, m)) ?? NaN
  const max = d3.max(data.matched.flatMap(r => [r[`addis_${method}_EC_ugm3`], r[`bishoftu_${method}_EC_ugm3`]])) ?? 1
  // the two corrections side by side in each date's band; a click opens the pair's record
  const adamaPoints: ScatterPoint[] = data.adama.flatMap((r, i) => METHODS.map((m, j) => ({ x: i + 0.5, dx: j ? 9 : -9, y: ratio(r, m), key: `${r.date}${m}`, r: 5.5, color: COLORS[m], opacity: 1,
    stroke: r.flag ? INK.text : INK.panel, strokeWidth: r.flag ? 2 : 1,
    tip: [`${r.date} ${m}: ${fmt(ratio(r, m), 3)}× ${reference}`, `PTFE ${r.FilterId_ptfe}, quartz ${r.FilterId_quartz}`, ...(r.flag ? [r.flag] : [])], open: () => hl.openRecord(adamaRecord(r)) })))
  // each matched pair a row: Addis and Bishoftu on the prediction axis, each opening its own filter's record
  const pairPoints: ScatterPoint[] = data.matched.flatMap((r, i) => (['addis', 'bishoftu'] as const).map((side) => ({ x: r[`${side}_${method}_EC_ugm3`], y: i + 0.5, key: `${r.bishoftu_filter_id}${side}`, r: 5, opacity: 1,
    color: side === 'addis' ? '#2C6E9E' : '#C49442', stroke: INK.panel, strokeWidth: 0.8,
    tip: side === 'addis' ? [`Addis ${fmt(r[`addis_${method}_EC_ugm3`], 3)} µg/m³; ${r.addis_date}`] : [`Bishoftu ${fmt(r[`bishoftu_${method}_EC_ugm3`], 3)} µg/m³; ${r.bishoftu_date}; Fabs ratio ${fmt(r.Fabs_ratio, 3)}`],
    open: () => hl.openRecord(pairSide(r, side)) })))
  const delta = d3.median(data.matched, r => r[`bishoftu_${method}_EC_ugm3`] - r[`addis_${method}_EC_ugm3`]) ?? NaN
  const siteRows = [
    ...data.addis.map(r => ({ ...r, site: 'Addis', id: r.filter_id })),
    ...data.bishoftu.map(r => ({ ...r, site: 'Bishoftu', id: r.ExternalFilterId, lot: r.LotId })),
  ]
  return <>
    {completed && <ExplainOnly><ChartFrame title="Rank-70 external application" exportable={false}><p>All 31 external spectra were corrected over 4000–500 cm⁻¹ with the same frozen rank-70 background. Predictions use the shared-window EC model. Date pairing, the provisional Adama ID map, exclusions and optical matching remain unchanged.</p></ChartFrame></ExplainOnly>}
    <ExplainOnly><ChartFrame title={completed ? "Completed rank-70 external pilot: Adama and Bishoftu" : "Archived external pilot: Adama and Bishoftu"} exportable={false}>
      <p style={{ fontSize: 14, lineHeight: 1.6 }}>{completed ? 'The completed rank-70 VIBES shared-input and spline models were applied' : 'The earlier frozen full-pool AIRSpec and VIBES models were applied'} to five Adama and 26 Bishoftu spectra without refitting. All 31 external baseline corrections converged. The Adama panel uses measured quartz EC on date-candidate co-samples; the Bishoftu panel compares spectra at matched optical loading because Bishoftu has no thermal EC measurement.</p>
      <Note>Adama's spectrum-to-PTFE ID map and sampler equivalence await confirmation. Bishoftu's HIPS Fabs ÷ MAC {mac} is an optical equivalent, not a thermal EC reference. These panels are diagnostic observations, not site accuracy rankings.</Note>
      <a href={BASE + (completed ? 'rank70/baseline_external_pilot.json' : 'baseline_external_pilot.json')} download>Download pilot rows and mapping ranges</a>
      <p style={{ fontSize: 11, overflowWrap: 'anywhere' }}>Frozen run: <code>{data.frozen_run_signature}</code> · Evidence date {data.evidence_date}</p>
    </ChartFrame></ExplainOnly>

    <ChartFrame title="Adama against quartz thermal EC" subtitle="Five date-candidate PTFE/quartz pairs. Ringed points have a sampler flag." controls={<Segmented label="Thermal convention" value={reference} options={['ECTR', 'ECTT']} optionLabel={(v) => (v === 'ECTR' ? 'TOR EC' : 'TOT EC')} onChange={(v) => setReference(v as Reference)} />}>
      <div className="chart-wrap" role="img" aria-label={`Adama frozen prediction divided by quartz ${reference} for five candidate dates`}>
        <ScatterPlot points={adamaPoints} xBands={data.adama.map(r => ({ key: r.date, label: r.flag ? `${r.date.slice(5)} · ${r.flag}` : r.date.slice(5) }))} yDomain={[0, 2]} nice={false} refY={[1]}
          height={170} margin={{ top: 14, right: 40, bottom: 46, left: 75 }} yFormat={(v) => (+v).toFixed(1)} density="sparse" note={false}
          xLabel={`2024 sampling date${data.adama.some(r => r.flag) ? ' · sampler flag' : ''}`} yLabel={`Prediction / quartz ${reference}`} />
        <Key items={METHODS.map(m => ({ label: m, color: COLORS[m] }))} />
      </div>
      <Note>Three sampling-unflagged pairs: provisional median AIRSpec {fmt(med('AIRSpec'), 3)}× and VIBES {fmt(med('VIBES'), 3)}× {reference}. Across all 120 possible spectrum-to-filter assignments, the medians range {fmt(range('AIRSpec').min, 3)}–{fmt(range('AIRSpec').max, 3)}× and {fmt(range('VIBES').min, 3)}–{fmt(range('VIBES').max, 3)}×. These are mapping sensitivities, not confidence intervals. The unflagged pairs are still unconfirmed co-samples.</Note>
      <FixedSetNote what="the five frozen Adama PTFE/quartz pairs" />
    </ChartFrame>

    <AllFilterScatter rows={siteRows} mode="optical" method={opticalMethod} onMethod={setOpticalMethod} site={opticalSite} onSite={setOpticalSite} mac={mac} demingLambda={demingLambda} />
    <AllFilterScatter rows={siteRows} mode="spectral" method={spectralMethod} onMethod={setSpectralMethod} site={spectralSite} onSite={setSpectralSite} mac={mac} demingLambda={demingLambda} />

    <ChartFrame title="Bishoftu versus Addis at near-equal HIPS Fabs" subtitle="Five unique lot-251, October–December pairs within 20% optical-loading caliper; dates differ by year." controls={<BaselinePicker as="select" label="Frozen correction" value={method} options={METHODS} backgroundRank={completed ? 70 : null} onChange={(v) => setMethod(v as Method)} />}>
      <div className="chart-wrap" role="img" aria-label={`${method} predicted EC for five Addis and Bishoftu optical-loading-matched pairs`}>
        <ScatterPlot points={pairPoints} yBands={data.matched.map(r => ({ key: r.bishoftu_filter_id, label: `${r.bishoftu_filter_id} / ${r.addis_filter_id}` }))} xDomain={[0, max * 1.16]} nice={false}
          height={data.matched.length * 39} margin={{ top: 14, right: 40, bottom: 46, left: 188 }} xTicks={5} xFormat={(v) => fmt(+v, 1)} density="sparse" note={false}
          xLabel="Frozen FTIR EC prediction (µg/m³)"
          under={({ x, y }) => <g pointerEvents="none">{data.matched.map((r, i) => <line key={r.bishoftu_filter_id} x1={x(r[`addis_${method}_EC_ugm3`])} x2={x(r[`bishoftu_${method}_EC_ugm3`])} y1={y(i + 0.5)} y2={y(i + 0.5)} stroke="#A7B1BA" strokeWidth={2} />)}</g>} />
        <Key items={[{ label: 'Addis', color: '#2C6E9E' }, { label: 'Bishoftu', color: '#C49442' }]} />
      </div>
      <Note>Median paired Bishoftu − Addis prediction: +{fmt(delta, 3)} µg/m³ ({method}); all five differences are positive. Matching uses HIPS Fabs to compare spectra at similar optical loading. It cannot determine which site has accurate EC. Across all 26 Bishoftu filters, median {method} prediction is {fmt(d3.median(data.bishoftu, r => r[`${method}_EC_ugm3`]) ?? NaN, 3)} µg/m³.</Note>
      <FixedSetNote what="the five frozen loading-matched Bishoftu/Addis pairs" />
    </ChartFrame>
  </>
}

type PlotRow = Omit<SiteRow, 'filter_id'> & { site: string; id: string }

/** Drawer records for the pilot's filters; Addis ids resolve to the full SPARTAN record, the rest show these fields. */
function siteRecord(r: PlotRow, source: string, mac: number): FilterRecord {
  return {
    id: r.id,
    source: `"${source}"`,
    site: r.site,
    date: r.date,
    fields: [
      ['HIPS lot', r.lot ?? null],
      ['HIPS Fabs (Mm⁻¹)', r.Fabs ?? null],
      [`HIPS Fabs ÷ MAC ${mac} (µg/m³)`, r.Fabs != null ? r.Fabs / mac : null],
      ['AIRSpec frozen FTIR EC (µg/m³)', r.AIRSpec_EC_ugm3],
      ['VIBES frozen FTIR EC (µg/m³)', r.VIBES_EC_ugm3],
      ['AIRSpec local 1617 / 2920 (absorbance)', `${r.AIRSpec_band1617?.toFixed(5) ?? '—'} / ${r.AIRSpec_band2920?.toFixed(5) ?? '—'}`],
      ['VIBES local 1617 / 2920 (absorbance)', `${r.VIBES_band1617?.toFixed(5) ?? '—'} / ${r.VIBES_band2920?.toFixed(5) ?? '—'}`],
    ],
    note: 'No thermal EC is measured for these filters; HIPS Fabs ÷ MAC is an optical equivalent.',
  }
}

function adamaRecord(r: AdamaRow): FilterRecord {
  return {
    id: r.FilterId_ptfe,
    source: 'the Adama quartz comparison',
    site: 'Adama',
    date: r.date,
    fields: [
      ['co-sampled quartz filter', r.FilterId_quartz],
      ['sampler flag', r.flag ?? 'none'],
      ['quartz TOR EC (µg/m³)', r.ECTR],
      ['quartz TOT EC (µg/m³)', r.ECTT],
      ['AIRSpec frozen FTIR EC (µg/m³)', r.AIRSpec_EC_ugm3],
      ['VIBES frozen FTIR EC (µg/m³)', r.VIBES_EC_ugm3],
      ['AIRSpec ÷ TOR', r.AIRSpec_over_ECTR],
      ['VIBES ÷ TOR', r.VIBES_over_ECTR],
    ],
    note: "Date-candidate co-sample: the spectrum-to-PTFE map and sampler equivalence are still unconfirmed.",
  }
}

function pairSide(r: MatchedRow, side: 'addis' | 'bishoftu'): FilterRecord {
  const other = side === 'addis' ? 'bishoftu' : 'addis'
  return {
    id: r[`${side}_filter_id`],
    source: 'the Bishoftu–Addis matched pairs',
    site: side === 'addis' ? 'Addis Ababa' : 'Bishoftu',
    date: r[`${side}_date`],
    fields: [
      ['matched with', `${r[`${other}_filter_id`]} (${r[`${other}_date`]})`],
      ['Fabs ratio, Bishoftu ÷ Addis', r.Fabs_ratio],
      ['AIRSpec frozen FTIR EC (µg/m³)', r[`${side}_AIRSpec_EC_ugm3`]],
      ['VIBES frozen FTIR EC (µg/m³)', r[`${side}_VIBES_EC_ugm3`]],
      [`partner AIRSpec / VIBES (µg/m³)`, `${r[`${other}_AIRSpec_EC_ugm3`].toFixed(3)} / ${r[`${other}_VIBES_EC_ugm3`].toFixed(3)}`],
    ],
    note: 'Matched on HIPS Fabs (optical loading), lot 251; the pair cannot say which site has accurate EC.',
  }
}

function AllFilterScatter({ rows, mode, method, onMethod, site, onSite, mac, demingLambda }: {
  rows: PlotRow[]
  mode: 'optical' | 'spectral'
  method: Method
  onMethod: (method: Method) => void
  site: string
  onSite: (site: string) => void
  mac: number
  demingLambda: number
}) {
  const { rank } = useBaselineStyle()
  const optical = mode === 'optical'
  const { grouping } = useRegressions()
  const [fitPreference, setFitPreference] = useSearchParam<string>(optical ? 'bopticalfits' : 'bspectralfits', 'default', ['default', 'none', 'pooled', 'site'])
  const chosenFit = fitPreference === 'default' ? (optical || grouping === 'site' ? grouping : 'none') : fitPreference
  // with one site shown, a fit per site is the pooled fit
  const oneSite = site !== 'Both sites'
  const fitGrouping = oneSite && chosenFit === 'site' ? 'pooled' : chosenFit
  // Optical: HIPS Fabs ÷ MAC puts x in µg/m³ like the FTIR EC prediction, so the
  // panel can carry a 1:1 line (equal axes; Deming only, both axes carry error).
  const allPoints = rows.flatMap(r => {
    const xv = optical ? (r.Fabs != null ? r.Fabs / mac : null) : r[`${method}_band2920`]
    const yv = optical ? r[`${method}_EC_ugm3`] : r[`${method}_band1617`]
    return xv != null && yv != null && Number.isFinite(xv) && Number.isFinite(yv)
      ? [{ ...r, xv, yv }] : []
  })
  // Filters → excluded (lib/markedRows useCountedFilter): the export's exclusions and every hand exclusion leave the
  // points and the live fits, as on every other chart; Include puts the export's back
  const isCounted = useCountedFilter()
  const inSite = allPoints.filter(r => site === 'Both sites' || r.site === site)
  const points = inSite.filter(r => isCounted(String(r.id)))
  // Filters → excluded: Mark draws them as × (never in a fit or count); otherwise they are hidden (the Crossplot and
  // the ScatterPlot each say how many)
  const markMode = useExclMode() === 'mark'
  const out = markMode ? inSite.filter(r => !isCounted(String(r.id))) : []
  const counts = Object.fromEntries(['Addis', 'Bishoftu'].map(s => [s, points.filter(r => r.site === s).length]))
  const title = optical ? 'Every filter with HIPS and frozen FTIR EC' : 'Every corrected spectrum: two local bands'
  // Keep axis scales fixed while the site selector changes, so a filtered
  // cloud can still be compared against the complete-cohort view.
  const xext = d3.extent(allPoints, r => r.xv)
  const yext = d3.extent(allPoints, r => r.yv)
  // the optical panel is the square 1:1 crossplot with one shared domain; the band panel a square ScatterPlot with its own axes
  const shared: [number, number] = [Math.min(0, xext[0] ?? 0, yext[0] ?? 0), Math.max(xext[1] ?? 1, yext[1] ?? 1) * 1.06]
  const xDomain = d3.scaleLinear().domain(optical ? shared : [Math.min(0, xext[0] ?? 0), (xext[1] ?? 1) * 1.06]).nice().domain() as [number, number]
  const yDomain = d3.scaleLinear().domain([Math.min(0, yext[0] ?? 0), (yext[1] ?? 1) * 1.08]).nice().domain() as [number, number]
  const note = optical
    ? <><p>{points.length} physical filters shown: {counts.Addis} Addis and {counts.Bishoftu} Bishoftu. The full frozen Addis target set has 253 spectra; 20 have no paired HIPS Fabs and cannot appear on this optical plot. No thermal EC is measured at either site in this plot.</p>
        <p>x is HIPS Fabs ÷ MAC {mac}, an optical BC equivalent, so the 1:1 line shows optical–FTIR agreement, not EC accuracy. Deming uses λ = {fmt(demingLambda, 2)} (HIPS uncertainty ÷ MAC against the AIRSpec IMPROVE cross-validation error, AGENTS.md), because both axes carry measurement error.</p></>
    : <p>{points.length} physical filters shown: {counts.Addis} Addis and {counts.Bishoftu} Bishoftu. All 253 Addis and 26 Bishoftu corrected spectra appear by default, including Addis records without HIPS Fabs. Band heights are local peak-minus-continuum descriptors, not chemical identifications. Spectral-band Deming fits assume equal error variances (λ = 1); band-specific uncertainties are unavailable.</p>
  const siteColor = (site: string) => (site === 'Bishoftu' ? '#C49442' : '#2C6E9E')
  const pointTip = (r: PlotRow & { xv: number; yv: number }) => `${r.site} ${r.id} · ${r.date} · lot ${r.lot ?? 'unknown'} · ${optical ? `Fabs ${fmt(r.Fabs!, 2)} Mm⁻¹ (÷ MAC ${mac} = ${fmt(r.xv, 3)} µg/m³); ${method} FTIR EC ${fmt(r.yv, 3)} µg/m³` : `${method} local 2920 ${fmt(r.xv, 5)}, local 1617 ${fmt(r.yv, 5)} absorbance`}`
  const crossPoint = (r: PlotRow & { xv: number; yv: number }) => ({ x: r.xv, y: r.yv, color: siteColor(r.site), id: r.id, rec: siteRecord(r, title, mac), tip: [pointTip(r)] })
  const controls = <>
    <BaselinePicker as="select" label="Correction" value={method} options={METHODS} backgroundRank={rank} onChange={(v) => onMethod(v as Method)} />
    <Segmented label="Site" value={site} options={['Both sites', 'Addis', 'Bishoftu']} onChange={onSite} />
    <Combobox label="Fit grouping" value={oneSite && fitPreference === 'site' ? 'pooled' : fitPreference} options={[
      { key: 'default', label: `Use gallery setting (${fitGrouping === 'site' ? 'per site' : fitGrouping === 'pooled' ? 'pooled' : 'no lines'})` },
      { key: 'none', label: 'No fitted lines' }, { key: 'pooled', label: oneSite ? `One fit (${site})` : 'One pooled fit' }, ...(oneSite ? [] : [{ key: 'site', label: 'Separate fit per site' }]),
    ]} onChange={setFitPreference} />
  </>
  const tipContent = <>{note}<p>Separate fits use only each site's paired points and cover that site's observed x range. Fit equations describe the displayed pairs; saved EC calibrations remain frozen.</p></>
  // the optical panel compares like with like (µg/m³ on both axes): the shared 1:1 crossplot, on a domain fixed
  // across the site selector so a filtered cloud still compares with the full one
  if (optical) return <ChartFrame id="baseline-optical-sites" title={title} tip={tipContent} controls={controls}>
    <div className="chart-wrap centered">
      <Crossplot title="" maxSide={460} domain={xDomain} errorRatio={demingLambda} grouping={fitGrouping as FitGrouping | 'none'}
        xLabel={`HIPS Fabs ÷ MAC ${mac} (µg/m³)`} yLabel="Frozen FTIR EC prediction (µg/m³)"
        points={points.map(crossPoint)} marked={out.map(crossPoint)} />
      <Key items={[{ label: `Addis (n ${counts.Addis})`, color: siteColor('Addis') }, { label: `Bishoftu (n ${counts.Bishoftu})`, color: siteColor('Bishoftu') }]} />
    </div>
    <ExcludedNote count={out.length} drawn />
  </ChartFrame>
  // every filter of the chosen sites: Filters → excluded acts in the plot (hidden, or × under Mark) and the live fits
  // (pooled or per site, λ = 1) refit the counted points, their summary above the axes
  const bandPoints: ScatterPoint[] = inSite.map(r => ({ x: r.xv, y: r.yv, key: r.site + r.id, id: String(r.id), rec: siteRecord(r, title, mac), site: r.site,
    color: siteColor(r.site), r: r.site === 'Bishoftu' ? 5 : 3.4, level: r.site === 'Bishoftu' ? 'sparse' : 'dense', tip: [pointTip(r)] }))
  return <ChartFrame id="baseline-spectral-sites" title={title} tip={tipContent} controls={controls}>
    <div className="chart-wrap centered" role="img" aria-label={`${title}: ${counts.Addis} Addis and ${counts.Bishoftu} Bishoftu points`}>
      <ScatterPlot points={bandPoints} xDomain={xDomain} yDomain={yDomain} maxSide={400} margin={{ top: 14, right: 18, bottom: 46, left: 72 }} labelX={-60}
        xFormat={(v) => fmt(+v, 3)} yFormat={(v) => fmt(+v, 3)} fit={fitGrouping as FitGrouping | 'none'} errorRatio={1}
        xLabel="2920 cm⁻¹ local height (corrected absorbance)" yLabel="1617 cm⁻¹ local height (corrected absorbance)" />
      <Key items={[{ label: `Addis (n ${counts.Addis})`, color: siteColor('Addis') }, { label: `Bishoftu (n ${counts.Bishoftu})`, color: siteColor('Bishoftu') }]} />
    </div>
  </ChartFrame>
}
