import { monthBlocks } from '@/lib/regressionCI'
import { IdentityLine } from '@/components/RefLines'
import { dataUrl, useExport } from '@/lib/data'
import { ExportState } from '@/components/ExportState'
import { RegressionLines, RegressionReadout } from '@/components/RegressionOverlay'
import { useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, ExplainOnly, Note, Segmented } from '@/components/ChartFrame'
import { BaselinePicker } from '@/components/Pickers'
import { Crossplot } from '@/components/Crossplot'
import type { FitGrouping } from '@/lib/siteRegression'
import { Key } from '@/components/Key'
import { useBaselineStyle } from '@/lib/baselineRank'
import { INK, baselineColor } from '@/lib/theme'
import { fixed as fmt } from '@/lib/format'
import { regression } from '@/lib/stats'
import { dotFill, dotRim, type FilterRecord, useHighlightActions } from '@/lib/highlight'
import { useHandEdits } from '@/lib/filterEdits'
import { useSearchParam } from '@/lib/url'
import { useRegressions } from '@/lib/regressionPrefs'
import { Combobox } from '@/components/Combobox'
import { ExcludedNote, FixedSetNote, useExclMode, useMarkReason } from '@/lib/markedRows'
import { crossPath, markTip } from '@/lib/exclusionMode'

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
  const y = d3.scaleLinear().domain([0, 2]).range([190, 20])
  const x = (i: number) => 105 + i * 130
  const unflagged = data.adama.filter(r => !r.flag)
  const ratio = (r: AdamaRow, m: Method) => r[`${m}_over_${reference}`]
  const range = (m: Method) => data.mapping_ranges.find(r => r.method === m && r.reference === reference && r.scope === 'three_sampling_unflagged')!
  const med = (m: Method) => d3.median(unflagged, r => ratio(r, m)) ?? NaN
  const max = d3.max(data.matched.flatMap(r => [r[`addis_${method}_EC_ugm3`], r[`bishoftu_${method}_EC_ugm3`]])) ?? 1
  const sx = d3.scaleLinear().domain([0, max * 1.16]).range([188, 700])
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
      <div className="chart-wrap" style={{ overflowX: 'auto' }}><svg viewBox="0 0 760 260" width="100%" style={{ minWidth: 620, maxWidth: 760, margin: '0 auto' }} role="img" aria-label={`Adama frozen prediction divided by quartz ${reference} for five candidate dates`}>
        {[0, .5, 1, 1.5, 2].map(v => <g key={v}><line x1={75} x2={720} y1={y(v)} y2={y(v)} stroke={v === 1 ? INK.axis : INK.grid} strokeDasharray={v === 1 ? '4 3' : undefined}/><text x={68} y={y(v) + 4} textAnchor="end" fontSize={11} fill={INK.axis}>{v.toFixed(1)}</text></g>)}
        {data.adama.map((r, i) => <g key={r.date}><text x={x(i)} y={215} textAnchor="middle" fontSize={11} fill={INK.text}>{r.date.slice(5)}</text>{r.flag && <text x={x(i)} y={232} textAnchor="middle" fontSize={10} fill={INK.negative}>{r.flag}</text>}{METHODS.map((m, j) => <circle key={m} cx={x(i) + (j ? 9 : -9)} cy={y(ratio(r, m))} r={5.5} fill={COLORS[m]} stroke={r.flag ? INK.text : INK.panel} strokeWidth={r.flag ? 2 : 1} style={{ cursor: 'pointer' }} onClick={() => hl.openRecord(adamaRecord(r))}><title>{`${r.date} ${m}: ${fmt(ratio(r, m), 3)}× ${reference}; PTFE ${r.FilterId_ptfe}, quartz ${r.FilterId_quartz}${r.flag ? `; ${r.flag}` : ''}`}</title></circle>)}</g>)}
        <text x={400} y={252} textAnchor="middle" fontSize={11} fill={INK.axis}>2024 sampling date · prediction / measured quartz {reference}</text>
        <circle cx={570} cy={20} r={4} fill={COLORS.AIRSpec}/><text x={580} y={24} fontSize={11}>AIRSpec</text><circle cx={645} cy={20} r={4} fill={COLORS.VIBES}/><text x={655} y={24} fontSize={11}>VIBES</text>
      </svg></div>
      <Note>Three sampling-unflagged pairs: provisional median AIRSpec {fmt(med('AIRSpec'), 3)}× and VIBES {fmt(med('VIBES'), 3)}× {reference}. Across all 120 possible spectrum-to-filter assignments, the medians range {fmt(range('AIRSpec').min, 3)}–{fmt(range('AIRSpec').max, 3)}× and {fmt(range('VIBES').min, 3)}–{fmt(range('VIBES').max, 3)}×. These are mapping sensitivities, not confidence intervals. The unflagged pairs are still unconfirmed co-samples.</Note>
      <FixedSetNote what="the five frozen Adama PTFE/quartz pairs" />
    </ChartFrame>

    <AllFilterScatter rows={siteRows} mode="optical" method={opticalMethod} onMethod={setOpticalMethod} site={opticalSite} onSite={setOpticalSite} mac={mac} demingLambda={demingLambda} />
    <AllFilterScatter rows={siteRows} mode="spectral" method={spectralMethod} onMethod={setSpectralMethod} site={spectralSite} onSite={setSpectralSite} mac={mac} demingLambda={demingLambda} />

    <ChartFrame title="Bishoftu versus Addis at near-equal HIPS Fabs" subtitle="Five unique lot-251, October–December pairs within 20% optical-loading caliper; dates differ by year." controls={<BaselinePicker as="select" label="Frozen correction" value={method} options={METHODS} backgroundRank={completed ? 70 : null} onChange={(v) => setMethod(v as Method)} />}>
      <div className="chart-wrap" style={{ overflowX: 'auto' }}><svg viewBox="0 0 760 270" width="100%" style={{ minWidth: 620, maxWidth: 760, margin: '0 auto' }} role="img" aria-label={`${method} predicted EC for five Addis and Bishoftu optical-loading-matched pairs`}>
        {data.matched.map((r, i) => { const a = r[`addis_${method}_EC_ugm3`], b = r[`bishoftu_${method}_EC_ugm3`], py = 35 + i * 39; return <g key={r.bishoftu_filter_id}><text x={180} y={py + 4} textAnchor="end" fontSize={11} fill={INK.text}>{r.bishoftu_filter_id} / {r.addis_filter_id}</text><line x1={sx(a)} x2={sx(b)} y1={py} y2={py} stroke="#A7B1BA" strokeWidth={2}/><circle cx={sx(a)} cy={py} r={5} fill="#2C6E9E" style={{ cursor: 'pointer' }} onClick={() => hl.openRecord(pairSide(r, 'addis'))}><title>{`Addis ${fmt(a, 3)} µg/m³; ${r.addis_date}`}</title></circle><circle cx={sx(b)} cy={py} r={5} fill="#C49442" style={{ cursor: 'pointer' }} onClick={() => hl.openRecord(pairSide(r, 'bishoftu'))}><title>{`Bishoftu ${fmt(b, 3)} µg/m³; ${r.bishoftu_date}; Fabs ratio ${fmt(r.Fabs_ratio, 3)}`}</title></circle></g> })}
        <line x1={188} x2={700} y1={230} y2={230} stroke={INK.axis}/>{sx.ticks(5).map(t => <g key={t}><line x1={sx(t)} x2={sx(t)} y1={230} y2={234} stroke={INK.axis}/><text x={sx(t)} y={248} textAnchor="middle" fontSize={10} fill={INK.axis}>{fmt(t, 1)}</text></g>)}<text x={450} y={267} textAnchor="middle" fontSize={11} fill={INK.axis}>Frozen FTIR EC prediction (µg/m³)</text>
        <circle cx={550} cy={15} r={4} fill="#2C6E9E"/><text x={560} y={19} fontSize={11}>Addis</text><circle cx={625} cy={15} r={4} fill="#C49442"/><text x={635} y={19} fontSize={11}>Bishoftu</text>
      </svg></div>
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
  const hl = useHighlightActions()
  const optical = mode === 'optical'
  const { methods, showCI, grouping } = useRegressions()
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
  // filters excluded by hand on any page are left out of the points and the live fits
  const edits = useHandEdits()
  const inSite = allPoints.filter(r => site === 'Both sites' || r.site === site)
  const points = inSite.filter(r => !edits.isOut(String(r.id)))
  // Filters → excluded: Mark draws them as × (never in a fit or count); otherwise a note says they are hidden
  const markMode = useExclMode() === 'mark'
  const reasonOf = useMarkReason()
  const out = markMode ? inSite.filter(r => edits.isOut(String(r.id))) : []
  const handNote = markMode ? null : edits.note(inSite.length - points.length, { refit: true })
  const frozen = <FixedSetNote what="the frozen pilot cohort, 253 Addis and 26 Bishoftu spectra" hand />
  const counts = Object.fromEntries(['Addis', 'Bishoftu'].map(s => [s, points.filter(r => r.site === s).length]))
  const title = optical ? 'Every filter with HIPS and frozen FTIR EC' : 'Every corrected spectrum: two local bands'
  // Keep axis scales fixed while the site selector changes, so a filtered
  // cloud can still be compared against the complete-cohort view.
  const xext = d3.extent(allPoints, r => r.xv)
  const yext = d3.extent(allPoints, r => r.yv)
  // the optical panel is square with one shared domain; the band panel keeps its own axes
  // both panels are square plot areas with the key to the right, like every crossplot in the gallery
  const W = 800
  const plot = optical ? { l: 80, r: 480, t: 30, b: 430 } : { l: 95, r: 495, t: 30, b: 430 }
  const shared: [number, number] = [Math.min(0, xext[0] ?? 0, yext[0] ?? 0), Math.max(xext[1] ?? 1, yext[1] ?? 1) * 1.06]
  const x = d3.scaleLinear().domain(optical ? shared : [Math.min(0, xext[0] ?? 0), (xext[1] ?? 1) * 1.06]).nice().range([plot.l, plot.r])
  const y = d3.scaleLinear().domain(optical ? shared : [Math.min(0, yext[0] ?? 0), (yext[1] ?? 1) * 1.08]).nice().range([plot.b, plot.t])
  if (optical) y.domain(x.domain())
  const fitGroups = fitGrouping === 'none' ? [] : (fitGrouping === 'site' ? ['Addis', 'Bishoftu'] : ['All selected sites']).flatMap(name => {
    const members = fitGrouping === 'site' ? points.filter(r => r.site === name) : points
    const stats = regression(members.map(r => r.xv), members.map(r => r.yv), {
      errorsInVariables: true, lambda: optical ? demingLambda : 1, bootstrapGroups: monthBlocks(members),
    })
    if (!stats) return []
    return [{ name, stats, color: fitGrouping === 'site' ? name === 'Addis' ? '#2C6E9E' : '#C49442' : undefined,
      domain: fitGrouping === 'site' ? d3.extent(members, r => r.xv) as [number, number] : x.domain() }]
  })
  const readoutGap = 70 + methods.length * (showCI ? 45 : 15)
  const H = Math.max(500, 94 + fitGroups.length * readoutGap)
  const [d0, d1] = x.domain()
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
      <Crossplot title="" maxSide={460} domain={x.domain() as [number, number]} errorRatio={demingLambda} grouping={fitGrouping as FitGrouping | 'none'}
        xLabel={`HIPS Fabs ÷ MAC ${mac} (µg/m³)`} yLabel="Frozen FTIR EC prediction (µg/m³)"
        points={points.map(crossPoint)} marked={out.map(crossPoint)} />
      <Key items={[{ label: `Addis (n ${counts.Addis})`, color: siteColor('Addis') }, { label: `Bishoftu (n ${counts.Bishoftu})`, color: siteColor('Bishoftu') }]} />
    </div>
    <ExcludedNote count={out.length} drawn />
    {frozen}
  </ChartFrame>
  const legendX = plot.r + 30
  const legendY = plot.t + 4
  return <ChartFrame id="baseline-spectral-sites" title={title} tip={tipContent} controls={controls}>
    <div className="chart-wrap" style={{ overflowX: 'auto' }}><svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 620, maxWidth: W, margin: '0 auto' }} role="img" aria-label={`${title}: ${counts.Addis} Addis and ${counts.Bishoftu} Bishoftu points`}>
      {y.ticks(5).map(v => <g key={'y' + v}><line x1={plot.l} x2={plot.r} y1={y(v)} y2={y(v)} stroke={INK.grid}/><text x={plot.l - 8} y={y(v) + 4} textAnchor="end" fontSize={10} fill={INK.axis}>{optical ? fmt(v, 1) : fmt(v, 3)}</text></g>)}
      {x.ticks(optical ? 5 : 6).map(v => <g key={'x' + v}>{optical && <line x1={x(v)} x2={x(v)} y1={plot.t} y2={plot.b} stroke={INK.grid}/>}<line x1={x(v)} x2={x(v)} y1={plot.b} y2={plot.b + 4} stroke={INK.axis}/><text x={x(v)} y={plot.b + 21} textAnchor="middle" fontSize={10} fill={INK.axis}>{optical ? fmt(v, 1) : fmt(v, 3)}</text></g>)}
      <line x1={plot.l} x2={plot.r} y1={plot.b} y2={plot.b} stroke={INK.axis}/>
      {optical && <line x1={plot.l} x2={plot.l} y1={plot.t} y2={plot.b} stroke={INK.axis}/>}
      {optical && <IdentityLine x1={x(d0)} y1={y(d0)} x2={x(d1)} y2={y(d1)} />}
      {/* Filters → excluded: Mark — the filters excluded by hand as muted ×, drawn only (lib/exclusionMode) */}
      {out.length > 0 && <path d={out.map(r => crossPath(x(r.xv), y(r.yv))).join('')} fill="none" stroke={INK.muted} strokeWidth={1.3} strokeOpacity={0.85} strokeLinecap="round" />}
      {out.map(r => <rect key={'x' + r.site + r.id} x={x(r.xv) - 5} y={y(r.yv) - 5} width={10} height={10} fill="transparent" style={{ cursor: 'pointer' }} onClick={() => hl.openRecord(siteRecord(r, title, mac))}>
        <title>{markTip({ id: String(r.id), site: r.site, date: r.date }, reasonOf(String(r.id)) ?? 'by hand').join('\n')}</title></rect>)}
      {points.map(r => <circle key={r.site + r.id} cx={x(r.xv)} cy={y(r.yv)} r={r.site === 'Bishoftu' ? 5 : 3.4} fill={r.site === 'Bishoftu' ? '#C49442' : '#2C6E9E'} {...dotFill(r.site === 'Bishoftu' ? 'sparse' : 'dense')} {...dotRim()} style={{ cursor: 'pointer' }} onClick={() => hl.openRecord(siteRecord(r, title, mac))}><title>{pointTip(r)}</title></circle>)}
      <g transform={`translate(${plot.l},${plot.t})`}>{fitGroups.map(group => <g key={group.name} data-fit-site={group.name}>
        <RegressionLines stats={group.stats} color={group.color} x={v => x(v) - plot.l} y={v => y(v) - plot.t} domain={group.domain} width={plot.r - plot.l} height={plot.b - plot.t}/>
      </g>)}</g>
      <text x={(plot.l + plot.r) / 2} y={plot.b + 49} textAnchor="middle" fontSize={12} fill={INK.axis}>{optical ? `HIPS Fabs ÷ MAC ${mac} (µg/m³)` : '2920 cm⁻¹ local height (corrected absorbance)'}</text>
      <text transform={`translate(19 ${(plot.t + plot.b) / 2}) rotate(-90)`} textAnchor="middle" fontSize={12} fill={INK.axis}>{optical ? 'Frozen FTIR EC prediction (µg/m³)' : '1617 cm⁻¹ local height (corrected absorbance)'}</text>
      <g fontSize={11} fill={INK.text}>
        <circle cx={legendX} cy={legendY} r={4} fill="#2C6E9E"/><text x={legendX + 10} y={legendY + 4}>Addis</text>
        <circle cx={legendX} cy={legendY + 20} r={4} fill="#C49442"/><text x={legendX + 10} y={legendY + 24}>Bishoftu</text>
      </g>
      {methods.length > 0 && fitGroups.map((group, i) => <RegressionReadout key={group.name} stats={group.stats} title={group.name} color={group.color} x={legendX} y={legendY + 50 + i * readoutGap} width={250} />)}
    </svg></div>
    {handNote && <Note>{handNote}</Note>}
    <ExcludedNote count={out.length} drawn />
    {frozen}
  </ChartFrame>
}
