import { useEffect, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, ExplainOnly, Note } from '@/components/ChartFrame'
import { INK } from '@/lib/theme'
import { regression } from '@/lib/stats'
import { useHighlight, type FilterRecord } from '@/lib/highlight'

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

const BASE = `${import.meta.env.BASE_URL}data/`
const METHODS: Method[] = ['AIRSpec', 'VIBES']
const COLORS: Record<Method, string> = { AIRSpec: '#2C6E9E', VIBES: '#7A4FA3' }
const fmt = (x: number, n = 2) => x.toFixed(n)

/**
 * `mac` is config.MAC_VALUE (via meta.json) and `demingLambda` the explorer's
 * lambda* at that MAC (via calibration.json): passed in, not restated here.
 */
export function BaselineExternalPilot({ mac, demingLambda }: { mac: number; demingLambda: number }) {
  const hl = useHighlight()
  const [data, setData] = useState<Pilot | null>(null)
  const [error, setError] = useState('')
  const [reference, setReference] = useState<Reference>('ECTR')
  const [method, setMethod] = useState<Method>('AIRSpec')
  const [opticalMethod, setOpticalMethod] = useState<Method>('AIRSpec')
  const [spectralMethod, setSpectralMethod] = useState<Method>('AIRSpec')
  const [opticalSite, setOpticalSite] = useState('Both sites')
  const [spectralSite, setSpectralSite] = useState('Both sites')
  useEffect(() => {
    const controller = new AbortController()
    fetch(BASE + 'baseline_external_pilot.json', { signal: controller.signal })
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then((d: Pilot) => {
        if (d.schema_version !== 1 || d.adama.length !== 5 || d.bishoftu.length !== 26 || d.addis.length !== 253 || d.matched.length !== 5) throw new Error('Unexpected pilot dataset')
        setData(d)
      })
      .catch(e => { if (e.name !== 'AbortError') setError(String(e)) })
    return () => controller.abort()
  }, [])
  if (error) return <ChartFrame title="External Ethiopia pilot" exportable={false}><Empty>{error}. Run the frozen pilot and gallery export.</Empty></ChartFrame>
  if (!data) return <p>Loading the Adama and Bishoftu pilot…</p>
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
    <ExplainOnly><ChartFrame title="External Ethiopia pilot: Adama and Bishoftu" exportable={false}>
      <p style={{ fontSize: 14, lineHeight: 1.6 }}>The frozen full-pool AIRSpec and VIBES models were applied to five Adama and 26 Bishoftu spectra without refitting. All 31 external baseline corrections converged. The Adama panel uses measured quartz EC on date-candidate co-samples; the Bishoftu panel compares spectra at matched optical loading because Bishoftu has no thermal EC measurement.</p>
      <Note>Adama's spectrum-to-PTFE ID map and sampler equivalence await confirmation. Bishoftu's HIPS/MAC-10 is an optical equivalent, not a thermal EC reference. These panels are diagnostic observations, not site accuracy rankings.</Note>
      <a href={BASE + 'baseline_external_pilot.json'} download>Download pilot rows and mapping ranges</a>
      <p style={{ fontSize: 11, overflowWrap: 'anywhere' }}>Frozen run: <code>{data.frozen_run_signature}</code> · Evidence date {data.evidence_date}</p>
    </ChartFrame></ExplainOnly>

    <ChartFrame title="Adama against quartz thermal EC" subtitle="Five date-candidate PTFE/quartz pairs. Ringed points have a sampler flag." controls={<label className="control">Thermal convention<select className="select" value={reference} onChange={e => setReference(e.target.value as Reference)}><option value="ECTR">TOR EC</option><option value="ECTT">TOT EC</option></select></label>}>
      <div className="chart-wrap" style={{ overflowX: 'auto' }}><svg viewBox="0 0 760 260" width="100%" style={{ minWidth: 620, maxWidth: 760, margin: '0 auto' }} role="img" aria-label={`Adama frozen prediction divided by quartz ${reference} for five candidate dates`}>
        {[0, .5, 1, 1.5, 2].map(v => <g key={v}><line x1={75} x2={720} y1={y(v)} y2={y(v)} stroke={v === 1 ? '#4C5965' : '#E8EBEF'} strokeDasharray={v === 1 ? '4 3' : undefined}/><text x={68} y={y(v) + 4} textAnchor="end" fontSize={11} fill="#5B6470">{v.toFixed(1)}</text></g>)}
        {data.adama.map((r, i) => <g key={r.date}><text x={x(i)} y={215} textAnchor="middle" fontSize={11} fill="#1F2933">{r.date.slice(5)}</text>{r.flag && <text x={x(i)} y={232} textAnchor="middle" fontSize={10} fill="#B2182B">{r.flag}</text>}{METHODS.map((m, j) => <circle key={m} cx={x(i) + (j ? 9 : -9)} cy={y(ratio(r, m))} r={5.5} fill={COLORS[m]} stroke={r.flag ? '#1F2933' : 'white'} strokeWidth={r.flag ? 2 : 1} style={{ cursor: 'pointer' }} onClick={() => hl.openRecord(adamaRecord(r))}><title>{`${r.date} ${m}: ${fmt(ratio(r, m), 3)}× ${reference}; PTFE ${r.FilterId_ptfe}, quartz ${r.FilterId_quartz}${r.flag ? `; ${r.flag}` : ''}`}</title></circle>)}</g>)}
        <text x={400} y={252} textAnchor="middle" fontSize={11} fill="#5B6470">2024 sampling date · prediction / measured quartz {reference}</text>
        <circle cx={570} cy={20} r={4} fill={COLORS.AIRSpec}/><text x={580} y={24} fontSize={11}>AIRSpec</text><circle cx={645} cy={20} r={4} fill={COLORS.VIBES}/><text x={655} y={24} fontSize={11}>VIBES</text>
      </svg></div>
      <Note>Three sampling-unflagged pairs: provisional median AIRSpec {fmt(med('AIRSpec'), 3)}× and VIBES {fmt(med('VIBES'), 3)}× {reference}. Across all 120 possible spectrum-to-filter assignments, the medians range {fmt(range('AIRSpec').min, 3)}–{fmt(range('AIRSpec').max, 3)}× and {fmt(range('VIBES').min, 3)}–{fmt(range('VIBES').max, 3)}×. These are mapping sensitivities, not confidence intervals. The unflagged pairs are still unconfirmed co-samples.</Note>
    </ChartFrame>

    <AllFilterScatter rows={siteRows} mode="optical" method={opticalMethod} onMethod={setOpticalMethod} site={opticalSite} onSite={setOpticalSite} mac={mac} demingLambda={demingLambda} />
    <AllFilterScatter rows={siteRows} mode="spectral" method={spectralMethod} onMethod={setSpectralMethod} site={spectralSite} onSite={setSpectralSite} mac={mac} demingLambda={demingLambda} />

    <ChartFrame title="Bishoftu versus Addis at near-equal HIPS Fabs" subtitle="Five unique lot-251, October–December pairs within 20% optical-loading caliper; dates differ by year." controls={<label className="control">Frozen correction<select className="select" value={method} onChange={e => setMethod(e.target.value as Method)}>{METHODS.map(m => <option key={m}>{m}</option>)}</select></label>}>
      <div className="chart-wrap" style={{ overflowX: 'auto' }}><svg viewBox="0 0 760 270" width="100%" style={{ minWidth: 620, maxWidth: 760, margin: '0 auto' }} role="img" aria-label={`${method} predicted EC for five Addis and Bishoftu optical-loading-matched pairs`}>
        {data.matched.map((r, i) => { const a = r[`addis_${method}_EC_ugm3`], b = r[`bishoftu_${method}_EC_ugm3`], py = 35 + i * 39; return <g key={r.bishoftu_filter_id}><text x={180} y={py + 4} textAnchor="end" fontSize={11} fill="#1F2933">{r.bishoftu_filter_id} / {r.addis_filter_id}</text><line x1={sx(a)} x2={sx(b)} y1={py} y2={py} stroke="#A7B1BA" strokeWidth={2}/><circle cx={sx(a)} cy={py} r={5} fill="#2C6E9E" style={{ cursor: 'pointer' }} onClick={() => hl.openRecord(pairSide(r, 'addis'))}><title>{`Addis ${fmt(a, 3)} µg/m³; ${r.addis_date}`}</title></circle><circle cx={sx(b)} cy={py} r={5} fill="#C49442" style={{ cursor: 'pointer' }} onClick={() => hl.openRecord(pairSide(r, 'bishoftu'))}><title>{`Bishoftu ${fmt(b, 3)} µg/m³; ${r.bishoftu_date}; Fabs ratio ${fmt(r.Fabs_ratio, 3)}`}</title></circle></g> })}
        <line x1={188} x2={700} y1={230} y2={230} stroke="#5B6470"/>{sx.ticks(5).map(t => <g key={t}><line x1={sx(t)} x2={sx(t)} y1={230} y2={234} stroke="#5B6470"/><text x={sx(t)} y={248} textAnchor="middle" fontSize={10} fill="#5B6470">{fmt(t, 1)}</text></g>)}<text x={450} y={267} textAnchor="middle" fontSize={11} fill="#5B6470">Frozen FTIR EC prediction (µg/m³)</text>
        <circle cx={550} cy={15} r={4} fill="#2C6E9E"/><text x={560} y={19} fontSize={11}>Addis</text><circle cx={625} cy={15} r={4} fill="#C49442"/><text x={635} y={19} fontSize={11}>Bishoftu</text>
      </svg></div>
      <Note>Median paired Bishoftu − Addis prediction: +{fmt(delta, 3)} µg/m³ ({method}); all five differences are positive. Matching uses HIPS Fabs to compare spectra at similar optical loading. It cannot determine which site has accurate EC. Across all 26 Bishoftu filters, median {method} prediction is {fmt(d3.median(data.bishoftu, r => r[`${method}_EC_ugm3`]) ?? NaN, 3)} µg/m³.</Note>
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
  const hl = useHighlight()
  const optical = mode === 'optical'
  // Optical: HIPS Fabs ÷ MAC puts x in µg/m³ like the FTIR EC prediction, so the
  // panel can carry a 1:1 line (equal axes; Deming only, both axes carry error).
  const allPoints = rows.flatMap(r => {
    const xv = optical ? (r.Fabs != null ? r.Fabs / mac : null) : r[`${method}_band2920`]
    const yv = optical ? r[`${method}_EC_ugm3`] : r[`${method}_band1617`]
    return xv != null && yv != null && Number.isFinite(xv) && Number.isFinite(yv)
      ? [{ ...r, xv, yv }] : []
  })
  const points = allPoints.filter(r => site === 'Both sites' || r.site === site)
  const counts = Object.fromEntries(['Addis', 'Bishoftu'].map(s => [s, points.filter(r => r.site === s).length]))
  const title = optical ? 'Every filter with HIPS and frozen FTIR EC' : 'Every corrected spectrum: two local bands'
  // Keep axis scales fixed while the site selector changes, so a filtered
  // cloud can still be compared against the complete-cohort view.
  const xext = d3.extent(allPoints, r => r.xv)
  const yext = d3.extent(allPoints, r => r.yv)
  // the optical panel is square with one shared domain; the band panel keeps its own axes
  // both panels are square plot areas with the key to the right, like every crossplot in the gallery
  const W = 720, H = 500
  const plot = optical ? { l: 80, r: 480, t: 30, b: 430 } : { l: 95, r: 495, t: 30, b: 430 }
  const shared: [number, number] = [Math.min(0, xext[0] ?? 0, yext[0] ?? 0), Math.max(xext[1] ?? 1, yext[1] ?? 1) * 1.06]
  const x = d3.scaleLinear().domain(optical ? shared : [Math.min(0, xext[0] ?? 0), (xext[1] ?? 1) * 1.06]).nice().range([plot.l, plot.r])
  const y = d3.scaleLinear().domain(optical ? shared : [Math.min(0, yext[0] ?? 0), (yext[1] ?? 1) * 1.08]).nice().range([plot.b, plot.t])
  if (optical) y.domain(x.domain())
  const st = optical ? regression(points.map(r => r.xv), points.map(r => r.yv), { errorsInVariables: true, lambda: demingLambda }) : null
  const [d0, d1] = x.domain()
  const fitLine = (m: number, b: number) => {
    // clip the fitted line to the square so it never leaves the plot
    const xs = [d0, d1, (d0 - b) / m, (d1 - b) / m].filter(v => v >= d0 && v <= d1 && m * v + b >= d0 - 1e-9 && m * v + b <= d1 + 1e-9).sort((a, c) => a - c)
    return xs.length >= 2 ? { x1: x(xs[0]), y1: y(m * xs[0] + b), x2: x(xs[xs.length - 1]), y2: y(m * xs[xs.length - 1] + b) } : null
  }
  const dem = st && st.demingSlope !== null && st.demingIntercept !== null ? fitLine(st.demingSlope, st.demingIntercept) : null
  const note = optical
    ? <><p>{points.length} physical filters shown: {counts.Addis} Addis and {counts.Bishoftu} Bishoftu. The full frozen Addis target set has 253 spectra; 20 have no paired HIPS Fabs and cannot appear on this optical plot. No thermal EC is measured at either site in this plot.</p>
        <p>x is HIPS Fabs ÷ MAC {mac}, an optical BC equivalent, so the 1:1 line shows optical–FTIR agreement, not EC accuracy. Deming uses λ = {fmt(demingLambda, 2)} (HIPS uncertainty ÷ MAC against the AIRSpec IMPROVE cross-validation error, AGENTS.md), because both axes carry measurement error.</p></>
    : <p>{points.length} physical filters shown: {counts.Addis} Addis and {counts.Bishoftu} Bishoftu. All 253 Addis and 26 Bishoftu corrected spectra appear by default, including Addis records without HIPS Fabs. Band heights are local peak-minus-continuum descriptors, not chemical identifications.</p>
  const legendX = plot.r + 30
  const legendY = plot.t + 4
  return <ChartFrame title={title} tip={note} controls={<>
    <label className="control">Correction<select className="select" value={method} onChange={e => onMethod(e.target.value as Method)}>{METHODS.map(m => <option key={m}>{m}</option>)}</select></label>
    <label className="control">Site<select className="select" value={site} onChange={e => onSite(e.target.value)}>{['Both sites', 'Addis', 'Bishoftu'].map(s => <option key={s}>{s}</option>)}</select></label>
  </>}>
    <div className="chart-wrap" style={{ overflowX: 'auto' }}><svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 620, maxWidth: W, margin: '0 auto' }} role="img" aria-label={`${title}: ${counts.Addis} Addis and ${counts.Bishoftu} Bishoftu points`}>
      {y.ticks(5).map(v => <g key={'y' + v}><line x1={plot.l} x2={plot.r} y1={y(v)} y2={y(v)} stroke="#E8EBEF"/><text x={plot.l - 8} y={y(v) + 4} textAnchor="end" fontSize={10} fill="#5B6470">{optical ? fmt(v, 1) : fmt(v, 3)}</text></g>)}
      {x.ticks(optical ? 5 : 6).map(v => <g key={'x' + v}>{optical && <line x1={x(v)} x2={x(v)} y1={plot.t} y2={plot.b} stroke="#E8EBEF"/>}<line x1={x(v)} x2={x(v)} y1={plot.b} y2={plot.b + 4} stroke="#5B6470"/><text x={x(v)} y={plot.b + 21} textAnchor="middle" fontSize={10} fill="#5B6470">{optical ? fmt(v, 1) : fmt(v, 3)}</text></g>)}
      <line x1={plot.l} x2={plot.r} y1={plot.b} y2={plot.b} stroke="#5B6470"/>
      {optical && <line x1={plot.l} x2={plot.l} y1={plot.t} y2={plot.b} stroke="#5B6470"/>}
      {optical && <line x1={x(d0)} y1={y(d0)} x2={x(d1)} y2={y(d1)} stroke={INK.identity} strokeWidth={1.2} strokeDasharray="5 4"/>}
      {points.map(r => <circle key={r.site + r.id} cx={x(r.xv)} cy={y(r.yv)} r={r.site === 'Bishoftu' ? 5 : 3.4} fill={r.site === 'Bishoftu' ? '#C49442' : '#2C6E9E'} fillOpacity={r.site === 'Bishoftu' ? .9 : .45} stroke={r.site === 'Bishoftu' ? '#755A24' : 'none'} strokeWidth={.7} style={{ cursor: 'pointer' }} onClick={() => hl.openRecord(siteRecord(r, title, mac))}><title>{`${r.site} ${r.id} · ${r.date} · lot ${r.lot ?? 'unknown'} · ${optical ? `Fabs ${fmt(r.Fabs!, 2)} Mm⁻¹ (÷ MAC ${mac} = ${fmt(r.xv, 3)} µg/m³); ${method} FTIR EC ${fmt(r.yv, 3)} µg/m³` : `${method} local 2920 ${fmt(r.xv, 5)}, local 1617 ${fmt(r.yv, 5)} absorbance`}`}</title></circle>)}
      {dem && <line {...dem} stroke={INK.deming} strokeWidth={1.6} strokeDasharray="7 4"/>}
      <text x={(plot.l + plot.r) / 2} y={plot.b + 49} textAnchor="middle" fontSize={12} fill="#5B6470">{optical ? `HIPS Fabs ÷ MAC ${mac} (µg/m³)` : '2920 cm⁻¹ local height (corrected absorbance)'}</text>
      <text transform={`translate(19 ${(plot.t + plot.b) / 2}) rotate(-90)`} textAnchor="middle" fontSize={12} fill="#5B6470">{optical ? 'Frozen FTIR EC prediction (µg/m³)' : '1617 cm⁻¹ local height (corrected absorbance)'}</text>
      <g fontSize={11} fill={INK.text}>
        <circle cx={legendX} cy={legendY} r={4} fill="#2C6E9E"/><text x={legendX + 10} y={legendY + 4}>Addis</text>
        <circle cx={legendX} cy={legendY + 20} r={4} fill="#C49442"/><text x={legendX + 10} y={legendY + 24}>Bishoftu</text>
      </g>
      {optical && st && <g fontSize={11} fill={INK.text} transform={`translate(${legendX - 4},${legendY + 52})`} fontFamily="var(--mono)">
        <line x1={0} x2={22} y1={-4} y2={-4} stroke={INK.identity} strokeWidth={1.2} strokeDasharray="5 4"/><text x={30} y={0}>1:1</text>
        <line x1={0} x2={22} y1={16} y2={16} stroke={INK.deming} strokeWidth={1.6} strokeDasharray="7 4"/><text x={30} y={20} fill={INK.deming}>{`Deming  y = ${fmt(st.demingSlope!, 3)}x ${st.demingIntercept! < 0 ? '−' : '+'} ${fmt(Math.abs(st.demingIntercept!), 3)}`}</text>
        <text x={30} y={40} fill={INK.muted}>{`λ = ${fmt(demingLambda, 2)}`}</text>
        <text x={30} y={58} fill={INK.muted}>{`n = ${st.n} · R² = ${fmt(st.r2, 3)}`}</text>
      </g>}
    </svg></div>
  </ChartFrame>
}
