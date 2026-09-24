import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, ExplainOnly, Note, Select } from '@/components/ChartFrame'
import { PageToc } from '@/components/PageToc'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { FONT, INK } from '@/lib/theme'
import { useHighlight, type FilterRecord } from '@/lib/highlight'
import { baselineMetrics, pairedAbsoluteErrorDifference, retainedWavenumber } from '@/lib/baselineComparison'
import { BaselineResearchDiagnostics, svgLegendItem, type SvgLegendState } from '@/pages/BaselineResearchDiagnostics'
import { useLegend } from '@/components/Legend'
import { BaselineExternalPilot } from '@/pages/BaselineExternalPilot'
import { useSearchParam as useComparisonParam } from '@/lib/url'
import type { BaselineCohort, BaselineData, BaselineGroup, BaselineMethod, BaselinePair, BaselineSpectrum, HistoricalMask } from '@/lib/baselineComparison'

const METHODS: BaselineMethod[] = ['AIRSpec', 'VIBES']
const COLOR = { AIRSpec: '#2C6E9E', VIBES: '#7A4FA3' }
const BASE = `${import.meta.env.BASE_URL}data/`
const f = (x: number | null | undefined, digits = 3) => x == null || !Number.isFinite(x) ? '—' : x.toFixed(digits)
const cohortName = (x: BaselineCohort) => x === 'full_pool' ? 'Full pool' : 'Restricted lowest-OC/EC'
const maskName = (x: string) => ({ full: 'No selection mask', no_co2: 'Exclude 1800–2500', no_co2_max3600: 'Also exclude >3600', no_co2_max3500: 'Also exclude >3500' }[x] ?? x)

export function BaselineComparisonPage({ mac, demingLambda }: { mac: number; demingLambda: number }) {
  const [data, setData] = useState<BaselineData | null>(null)
  const [error, setError] = useState('')
  const [cohort, setCohort] = useComparisonParam<BaselineCohort>('bcohort', 'full_pool', ['full_pool','locked800'])
  const [site, setSite] = useComparisonParam<string>('bsite', 'All sites')
  const [band, setBand] = useComparisonParam<string>('bband', 'All loadings', ['All loadings','Q1','Q2','Q3','Q4'])
  const [sample, setSample] = useComparisonParam<string>('bsample', 'improve:1970697')
  useEffect(() => {
    const controller = new AbortController()
    fetch(BASE + 'baseline_comparison.json', { signal: controller.signal })
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then((d: BaselineData) => { if (d.schema_version !== 1) throw new Error('Unsupported data version'); setData(d) })
      .catch(e => { if (e.name !== 'AbortError') setError(String(e)) })
    return () => controller.abort()
  }, [])
  const all = useMemo(() => data?.pairs.filter(r => r.cohort === cohort) ?? [], [data, cohort])
  const rows = useMemo(() => all.filter(r => (site === 'All sites' || r.Site === site) && (band === 'All loadings' || r.loading_band === band)), [all, site, band])
  const sites = useMemo(() => ['All sites', ...Array.from(new Set(all.map(r => r.Site))).sort()], [all])
  if (error) return <ChartFrame title="AIRSpec / VIBES" exportable={false}><Empty>Could not load the comparison: {error}. Run <code>uv run --locked --no-sync python gallery/data/export_baseline_comparison.py</code> and reload.</Empty></ChartFrame>
  if (!data) return <p>Loading the frozen AIRSpec / VIBES comparison…</p>
  const subset = site !== 'All sites' || band !== 'All loadings'
  const interval = data.groups.find(g => g.cohort === cohort && g.dimension === 'overall')!
  const spectrum = data.spectra.find(r => r.sample_id === sample) ?? data.spectra[0]
  const masks = data.historical_masks.filter(r => ['full','no_co2','no_co2_max3600','no_co2_max3500'].includes(r.mask))
  return <PageToc>
    <ExplainOnly><ChartFrame title="AIRSpec / VIBES comparison" exportable={false}>
      <p style={{ fontSize: 14, lineHeight: 1.6 }}>Frozen results from {data.evidence_date}: <strong>{data.run.n_cases.toLocaleString()} correction cases</strong>, {data.run.n_failed} final failures. AIRSpec fits segmented splines. VIBES learns background patterns from {data.run.n_background_blanks} independent blanks (rank {data.run.background_rank}). Each method has its own fitted EC calibration.</p>
      <p style={{ fontSize: 14, lineHeight: 1.6 }}><strong>Spectral-cut clarification:</strong> the earlier 1800–2500 cm⁻¹ exclusion and optional &gt;3600 / &gt;3500 cuts changed <strong>analog selection only</strong>. Those earlier PLS fits still used all 2,002 channels. The original Colab run uses the full pool or lowest-OC/EC membership; the separate notebook follow-up below refits site-restricted masked analog selections. Both correction methods and EC fits use approximately 1426–3998 cm⁻¹.</p>
      <Note>VIBES improves held-out blank removal. The original pooled EC comparison does not establish an accuracy gain. The new exploratory notebook fits and its provenance appear below; neither analysis clips predictions or introduces new exclusions.</Note>
    </ChartFrame></ExplainOnly>
    <ChartFrame title="Paired EC performance" exportable={false} tip={<p><strong>Complete-cohort reference, {cohortName(cohort)}:</strong> ΔRMSE {f(interval.delta_RMSE)} µg/filter, paired 95% interval {f(interval.delta_RMSE_ci_low)} to {f(interval.delta_RMSE_ci_high)}. {subset && 'This interval does not describe the selected display subset.'} Both pooled intervals include zero. Different test populations prevent ranking the two calibration cohorts by these errors.</p>} controls={<>
      <label className="control">Calibration cohort<select className="select" value={cohort} onChange={e => { setCohort(e.target.value as BaselineCohort); setSite('All sites'); setBand('All loadings') }}><option value="full_pool">Full pool</option><option value="locked800">Restricted lowest-OC/EC</option></select></label>
      <Select label="Test site" value={site} options={sites} onChange={setSite} />
      <Select label="Training-defined loading band" value={band} options={['All loadings','Q1','Q2','Q3','Q4']} onChange={setBand} />
      <button className="btn quiet" onClick={() => { setSite('All sites'); setBand('All loadings') }}>Reset subset</button>
    </>}>
      <p style={{ fontSize: 13 }}><strong>{rows.length.toLocaleString()} paired test filters</strong> shown out of {all.length.toLocaleString()}, across {new Set(rows.map(r => r.Site)).size} sites. {subset ? 'Exploratory display subset. The saved model and cohort remain unchanged.' : 'Complete frozen test cohort.'}</p>
      <div style={{ overflowX: 'auto' }}><table className="census-table"><thead><tr><th>Method</th><th>PLS k</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>Predictive R²</th></tr></thead><tbody>{METHODS.map(method => {
        const m = baselineMetrics(rows, method)
        return <tr key={method}><td style={{ color: COLOR[method], fontWeight: 600 }}>{method}</td><td>{data.scores.find(s => s.method === method && s.cohort === cohort)?.k}</td><td>{f(m?.rmse)}</td><td>{f(m?.mae)}</td><td>{f(m?.bias)}</td><td>{f(m?.r2)}</td></tr>
      })}</tbody></table></div>
      <Note>Errors and bias: µg/filter. Bias = prediction − TOR reference. Predictive R² = 1 − SSE/SST. Training-only 5-fold site-grouped CV, first major minimum. Test sites do not overlap training sites.</Note>
    </ChartFrame>
    <ResidualPlot rows={rows} onInspect={setSample} available={new Set(data.spectra.map(r => r.sample_id))} cohort={cohort} />
    <ErrorDifferencePlot rows={rows} cohort={cohort} />
    <LoadingPlot groups={data.groups.filter(g => g.cohort === cohort && g.dimension === 'loading_band')} />
    <SpectraPlot spectrum={spectrum} data={data} onSample={setSample} masks={masks} />
    <RecoveryPlot data={data} />
    <BaselineResearchDiagnostics data={data} rows={rows} cohort={cohort} sample={spectrum.sample_id} />
    <BaselineExternalPilot mac={mac} demingLambda={demingLambda} />
    <ChartFrame title="Earlier spectral cuts changed analog membership" exportable={false}>
      <ExplainOnly><Note>Historical AIRSpec analog-selection experiment, 10 September. All Addis target group, 500 selected unique filters. This is a separate experiment from the paired AIRSpec/VIBES benchmark above.</Note></ExplainOnly>
      <div style={{ overflowX: 'auto' }}><table className="census-table"><thead><tr><th>Selection mask</th><th>Matching channels</th><th>Replaced vs no mask / 500</th><th>PLS channels</th></tr></thead><tbody>{masks.map(m => <tr key={m.mask}><td>{maskName(m.mask)}</td><td>{m.n_channels}</td><td>{m.changed_vs_full}</td><td>2002</td></tr>)}</tbody></table></div>
      <ExplainOnly><Note>The primary mask excluded 1800–2500 inclusive. Upper-cut variants retained values ≤3600 or ≤3500. The 1850-start sensitivity variants are retained in the downloadable data. This historical table has no paired masked EC outcome; the executed notebook panel above reports new site-restricted analog selections on a common test population.</Note></ExplainOnly>
    </ChartFrame>
    <ExplainOnly><ChartFrame title="Sample eligibility and provenance" exportable={false}>
      <p style={{ fontSize: 13 }}>IMPROVE: {data.exclusions.pool_eligible.toLocaleString()} eligible physical calibration filters out of {data.exclusions.pool_total.toLocaleString()}. Each contributing scan must be finite, with verified sample purpose, a site and positive finite TOR EC. Raw repeat scans are averaged by physical filter before correction.</p>
      <table className="census-table"><thead><tr><th>Frozen IMPROVE decision</th><th>Physical filters</th></tr></thead><tbody>{data.exclusions.pool_reasons.map(r => <tr key={r.reason}><td>{r.reason}</td><td>{r.n.toLocaleString()}</td></tr>)}</tbody></table>
      <Note>Restricted selection: {data.exclusions.locked_eligible} of the original 800 qualify. Omitted: {data.exclusions.locked_omitted.map(r => `${r.n} ${r.reason.toLowerCase()}`).join(', ')}.</Note>
      <p style={{ fontSize: 13 }}>Addis: canonical sample registry applied, {data.exclusions.addis_registry_excluded} flagged. {data.exclusions.addis_roles.map(r => `${r.n} ${r.role.split('_').join(' ')}`).join(', ')}. The six outside-scope filters are PM10. This run did not use the older fixed 190-filter subset or aethalometer threshold cuts. {data.exclusions.registry_unchanged ? 'The current registry matches the frozen registry.' : 'The registry has changed since this run.'}</p>
      <Note>The 253 Addis spectra have no independent thermal EC target in this experiment. HIPS/MAC is an optical equivalent, not a thermal reference.</Note>
      <p style={{ fontSize: 13, display: 'flex', gap: 18, flexWrap: 'wrap' }}><a href={BASE+'baseline_pool_ledger.csv'} download>IMPROVE inclusion ledger</a><a href={BASE+'baseline_addis_ledger.csv'} download>Addis inclusion ledger</a><a href={BASE+'baseline_paired_predictions.csv'} download>Paired test predictions</a><a href={BASE+'baseline_comparison.json'} download>Comparison data and source hashes</a><a href={BASE+'baseline_applicability.json'} download>Training spectral distance data</a></p>
      <details><summary style={{ cursor: 'pointer', fontSize: 12 }}>Source files and SHA-256 provenance</summary><p style={{ fontSize: 11, overflowWrap: 'anywhere' }}>Run signature: {data.run.signature}</p>{Object.entries(data.source_sha256).map(([name, hash]) => <p key={name} style={{ fontSize: 11, overflowWrap: 'anywhere' }}><code>{name}</code><br />{hash}</p>)}</details>
    </ChartFrame></ExplainOnly>
  </PageToc>
}

/** One paired test filter as a drawer record: TOR, both predictions and their errors. */
function pairRecord(r: BaselinePair, source: string): FilterRecord {
  return {
    id: r.sample_id,
    source,
    site: r.Site,
    date: r.date,
    fields: [
      ['measured TOR EC (µg/filter)', r.y],
      ['AIRSpec prediction (µg/filter)', r.AIRSpec],
      ['VIBES prediction (µg/filter)', r.VIBES],
      ['AIRSpec error (µg/filter)', r.AIRSpec - r.y],
      ['VIBES error (µg/filter)', r.VIBES - r.y],
      ['closer to TOR', Math.abs(r.VIBES - r.y) < Math.abs(r.AIRSpec - r.y) ? 'VIBES' : Math.abs(r.VIBES - r.y) > Math.abs(r.AIRSpec - r.y) ? 'AIRSpec' : 'tie'],
      ['loading band', r.loading_band],
      ['HIPS lot', r.lot],
      ['calibration cohort', cohortName(r.cohort as BaselineCohort)],
    ],
    note: 'IMPROVE test filter: TOR is the thermal-optical reference for this held-out test, not an Addis measurement.',
  }
}

/** Legends live in the SVG so chart downloads retain the method identities. */
function MethodLegend({ width, y, items, lg }: { width: number; y: number; items: { label: string; color: string }[]; lg: SvgLegendState }) {
  const step = Math.min(175,(width-48)/items.length)
  return <g transform={`translate(24,${y})`} fontFamily={FONT.family} fontSize={11}>{items.map((m,i)=>{ const { hit, ...p } = svgLegendItem(lg, m.label, step); return <g key={m.label} transform={`translate(${i*step},0)`} {...p}>{hit}<line x2={12} y1={-3} y2={-3} stroke={m.color} strokeWidth={2}/><text x={18} fill={INK.text}>{m.label}</text></g> })}</g>
}

function ResidualPlot({ rows, available, onInspect, cohort }: { rows: BaselinePair[]; available: Set<string>; onInspect: (s: string) => void; cohort: BaselineCohort }) {
  const ref = useRef<HTMLDivElement>(null); const { width } = useDimensions(ref); const tip = useTooltip(ref); const hl = useHighlight()
  // One panel per method on shared axes: overlaid at this density the two purples/blues merge into one cloud.
  const w = Math.max(300, width), side = w >= 760, gap = 36
  // each method panel is square, like every crossplot in the gallery
  const pw = Math.min(460, side ? (w - 70 - gap - 20) / 2 : w - 90), ih = pw, ph = ih + 62
  const x = d3.scaleLinear().domain(d3.extent(rows, r => r.y) as [number,number]).range([0,pw]).nice()
  const max = d3.max(rows, r => Math.max(Math.abs(r.AIRSpec-r.y),Math.abs(r.VIBES-r.y))) || 1
  const y = d3.scaleLinear().domain([-max*1.05,max*1.05]).range([ih,0]).nice()
  const h = side ? ph + 10 : 2 * ph + 10
  return <ChartFrame title="Test errors by EC loading" subtitle="One panel per method on shared axes. Hover for each physical filter and both predictions. Ringed points have stored spectra in the 12-case investigation. Click one to select its spectrum below." provenance="Frozen paired test predictions. No regression or new fit.">
    <Note>{cohortName(cohort)}, current display subset. All negative predictions remain in the errors.</Note>
    <div ref={ref} className="chart-wrap">{!rows.length ? <Empty>No test filters match both selections.</Empty> : <svg width={w} height={h} role="img" aria-label="Prediction errors by TOR EC mass, one panel each for AIRSpec and VIBES">{METHODS.map((method, k) => {
      const ox = 70 + (side ? k * (pw + gap) : 0), oy = 26 + (side ? 0 : k * ph)
      const mae = d3.mean(rows, r => Math.abs(r[method]-r.y))
      return <g key={method} transform={`translate(${ox},${oy})`}>
        <text y={-10} fontSize={12} fontWeight={600} fill={COLOR[method]} fontFamily={FONT.family}>{method}<tspan fill={INK.muted} fontWeight={400}>{`  ·  MAE ${f(mae)} µg/filter`}</tspan></text>
        <YAxis scale={y} x={0} gridWidth={pw} label={k === 0 || !side ? 'Prediction − TOR (µg/filter)' : undefined}/><XAxis scale={x} y={ih} label="Measured TOR EC (µg/filter)"/>
        <line x2={pw} y1={y(0)} y2={y(0)} stroke={INK.fit} strokeDasharray="4 3"/>
        {rows.map(r => <circle key={r.sample_id} cx={x(r.y)} cy={y(r[method]-r.y)} r={available.has(r.sample_id)?4:2.5} fill={COLOR[method]} fillOpacity={0.45} stroke={available.has(r.sample_id)?INK.text:'none'} style={{cursor:'pointer'}} onClick={()=>{ hl.openRecord(pairRecord(r, 'the AIRSpec / VIBES error plot')); if (available.has(r.sample_id)) onInspect(r.sample_id) }} onMouseEnter={e=>tip.show(e,[`${r.Site} / ${r.sample_id}`,`${r.date} · ${r.loading_band}`,`TOR ${f(r.y)}, AIRSpec ${f(r.AIRSpec)}, VIBES ${f(r.VIBES)} µg/filter`,`${method} error ${f(r[method]-r.y)}`,available.has(r.sample_id)?'Click for details and the stored spectrum':'Click for details (spectrum not in the 12-case inspection)'])} onMouseLeave={tip.hide}/>)}
      </g>
    })}</svg>}{tip.node}</div>
  </ChartFrame>
}
function ErrorDifferencePlot({ rows, cohort }: { rows: BaselinePair[]; cohort: BaselineCohort }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const lg = useLegend()
  const side = (b: d3.Bin<number, number>) => b.x1! <= 0 ? 'VIBES closer (<0)' : 'AIRSpec closer (>0)'
  const [binCount, setBinCount] = useComparisonParam<string>('bbins', '40', ['10','20','40','80'])
  const values = rows.map(pairedAbsoluteErrorDifference)
  const bound = d3.max(values, Math.abs) || 1
  const thresholds = d3.range(1, Number(binCount)).map(i => (i - Number(binCount) / 2) * 2 * bound / Number(binCount))
  const bins = d3.bin().domain([-bound, bound]).thresholds(thresholds)(values)
  const w = Math.max(300, width), iw = w - 100, ih = 245
  const x = d3.scaleLinear().domain([-bound, bound]).range([0, iw])
  const y = d3.scaleLinear().domain([0, d3.max(bins.filter(b => lg.show(side(b))), b => b.length) || 1]).nice().range([ih, 0])
  const better = values.filter(v => v < 0).length, worse = values.filter(v => v > 0).length
  return <ChartFrame title="Which method is closer for each filter?" tip={<p>The counts describe individual filters; they do not establish a statistically significant gain. Changing bins changes the display only. Inspired by the <a href="https://www.react-graph-gallery.com/histogram" target="_blank" rel="noreferrer">React Graph Gallery histogram</a>.</p>} controls={<Select label="Histogram bins" value={binCount} options={['10','20','40','80']} onChange={setBinCount}/>}>
    <Note>{cohortName(cohort)}, current display subset. Each filter contributes one value: |VIBES − TOR| − |AIRSpec − TOR|. Negative favors VIBES; positive favors AIRSpec.</Note>
    <div ref={ref} className="chart-wrap">{!rows.length ? <Empty>No test filters match both selections.</Empty> : <svg width={w} height={360} role="img" aria-label="Histogram of paired absolute error differences, VIBES minus AIRSpec"><g transform="translate(70,30)">
      <YAxis scale={y} x={0} gridWidth={iw} label="Physical filters" format={v=>Number.isInteger(v)?String(v):''}/>
      {bins.map((b,i)=>lg.show(side(b))&&<rect key={i} x={x(b.x0!)+.5} y={y(b.length)} width={Math.max(0,x(b.x1!)-x(b.x0!)-1)} height={ih-y(b.length)} fill={b.x1!<=0?COLOR.VIBES:COLOR.AIRSpec} fillOpacity={.8*lg.dim(side(b))} onMouseEnter={e=>tip.show(e,[`${f(b.x0)} to ${f(b.x1)} µg/filter`,`${b.length} filters (${f(100*b.length/rows.length,1)}%)`,`Lower bound included; upper bound ${i===bins.length-1?'included':'excluded'}.`])} onMouseLeave={tip.hide}><title>{`${f(b.x0)} to ${f(b.x1)}: ${b.length} filters`}</title></rect>)}
      <line x1={x(0)} x2={x(0)} y2={ih} stroke={INK.text} strokeDasharray="4 3"/>
      <XAxis scale={x} y={ih} label="Δ absolute error (µg/filter)"/>
    </g><MethodLegend width={w} y={348} items={[{label:'VIBES closer (<0)',color:COLOR.VIBES},{label:'AIRSpec closer (>0)',color:COLOR.AIRSpec}]} lg={lg}/></svg>}{tip.node}</div>
    <Note>{rows.length.toLocaleString()} paired filters: VIBES closer {better.toLocaleString()}, AIRSpec closer {worse.toLocaleString()}, exact ties {values.length-better-worse}. Mean difference (ΔMAE): {f(d3.mean(values))}; median: {f(d3.median(values))} µg/filter. All tails are shown. Exact ties fall in the bin starting at zero.</Note>
  </ChartFrame>
}
function LoadingPlot({ groups }: { groups: BaselineGroup[] }) {
  const ref=useRef<HTMLDivElement>(null);const {width}=useDimensions(ref)
  const w=Math.max(300,width),iw=w-165,ih=180
  const bounds=groups.flatMap(g=>[g.delta_RMSE_ci_low??g.delta_RMSE,g.delta_RMSE_ci_high??g.delta_RMSE,0])
  const x=d3.scaleLinear().domain(d3.extent(bounds) as [number,number]).nice().range([0,iw])
  return <ChartFrame title="Loading-band differences" tip={<p>Full-pool Q3 = (3.739, 6.994] µg/filter. Restricted Q3 = (2.915, 7.881]. Boundaries come from each cohort's training labels. They do not establish a prediction-time switching rule.</p>} subtitle="Points show VIBES minus AIRSpec RMSE. Negative favors VIBES. Whiskers are paired 95% site-cluster intervals, conditional on the frozen models.">
    <div ref={ref} className="chart-wrap"><svg width={w} height={265} role="img" aria-label="Loading quartile paired RMSE differences and confidence intervals"><g transform="translate(130,20)" fontFamily={FONT.family}><line x1={x(0)} x2={x(0)} y2={ih} stroke={INK.identity} strokeDasharray="4 3"/>{groups.map((g,i)=><g key={g.group} transform={`translate(0,${20+i*42})`}><text x={-12} textAnchor="end" dy="0.3em" fontSize={12}>{g.group} (n={g.n})</text><line x1={x(g.delta_RMSE_ci_low??g.delta_RMSE)} x2={x(g.delta_RMSE_ci_high??g.delta_RMSE)} stroke={COLOR.VIBES} strokeWidth={2}/><circle cx={x(g.delta_RMSE)} r={5} fill={COLOR.VIBES}><title>{`${g.group}: ΔRMSE ${f(g.delta_RMSE)}, interval ${f(g.delta_RMSE_ci_low)} to ${f(g.delta_RMSE_ci_high)}`}</title></circle></g>)}<XAxis scale={x} y={ih} label="VIBES − AIRSpec RMSE (µg/filter)"/></g></svg></div>
  </ChartFrame>
}
function SpectraPlot({ spectrum:s, data, onSample, masks }: { spectrum: BaselineSpectrum; data: BaselineData; onSample:(x:string)=>void; masks:HistoricalMask[] }) {
  const ref=useRef<HTMLDivElement>(null);const {width}=useDimensions(ref);const lg=useLegend()
  const [view,setView]=useComparisonParam<string>('bview', 'Corrected spectra', ['Corrected spectra','Raw and baselines','VIBES − AIRSpec']);const [mask,setMask]=useComparisonParam<string>('bmask', 'no_co2_max3600', ['full','no_co2','no_co2_max3600','no_co2_max3500'])
  const [regions,setRegions]=useComparisonParam<string>('bregions', 'Shade excluded regions', ['Shade excluded regions','Hide excluded regions'])
  const hidden=regions==='Hide excluded regions'
  const chosen=masks.find(m=>m.mask===mask)!
  const visible=data.wn.map(wn=>!hidden||retainedWavenumber(wn,chosen))
  const w=Math.max(300,width),iw=w-100,ih=270
  const x=d3.scaleLinear().domain([4000,1425]).range([0,iw])
  const difference=view==='VIBES − AIRSpec'
  const series=difference?[{label:'VIBES − AIRSpec',values:s.vibes.map((v,i)=>v-s.airspec[i]),color:COLOR.VIBES}]:view==='Corrected spectra'?[{label:'AIRSpec',values:s.airspec,color:COLOR.AIRSpec},{label:'VIBES',values:s.vibes,color:COLOR.VIBES}]:[{label:'Raw',values:s.raw,color:INK.muted},{label:'AIRSpec baseline',values:s.raw.map((v,i)=>v-s.airspec[i]),color:COLOR.AIRSpec},{label:'VIBES baseline',values:s.raw.map((v,i)=>v-s.vibes[i]),color:COLOR.VIBES}]
  const drawn=series.filter(r=>lg.show(r.label))
  const ext=d3.extent((drawn.length?drawn:series).flatMap(r=>r.values.filter((_,i)=>visible[i]))) as [number,number]
  if(difference){const bound=Math.max(Math.abs(ext[0]),Math.abs(ext[1]));ext[0]=-bound;ext[1]=bound}
  const pad=(ext[1]-ext[0])*.06||0.01
  const y=d3.scaleLinear().domain([ext[0]-pad,ext[1]+pad]).nice().range([ih,0])
  const line=d3.line<number>().defined((v,i)=>visible[i]&&Number.isFinite(v)).x((_,i)=>x(data.wn[i])).y(v=>y(v))
  const shade=(lo:number,hi:number)=><g key={lo}>{hidden?<><line x1={x(lo)} x2={x(lo)} y2={ih} stroke={INK.axis} strokeDasharray="3 3"/><line x1={x(hi)} x2={x(hi)} y2={ih} stroke={INK.axis} strokeDasharray="3 3"/><text x={(x(lo)+x(hi))/2} y={15} textAnchor="middle" fontFamily={FONT.family} fontSize={11} fill={INK.muted}>Hidden</text></>:<rect x={x(hi)} width={x(lo)-x(hi)} height={ih} fill="#c49442" fillOpacity={0.18}/>}</g>
  return <ChartFrame id="baseline-spectrum" title="One physical filter, both corrections" controls={<>
    <label className="control">Inspection case<select className="select" value={s.sample_id} onChange={e=>onSample(e.target.value)}>{data.spectra.map(r=><option key={r.sample_id} value={r.sample_id}>{r.site} / {r.sample_id} ({r.direction})</option>)}</select></label>
    <Select label="Spectrum view" value={view} options={['Corrected spectra','Raw and baselines','VIBES − AIRSpec']} onChange={setView}/>
    <label className="control">Earlier selection mask<select className="select" value={mask} onChange={e=>setMask(e.target.value)}>{masks.map(m=><option key={m.mask} value={m.mask}>{maskName(m.mask)}</option>)}</select></label>
    <Select label="Excluded regions" value={regions} options={['Shade excluded regions','Hide excluded regions']} onChange={setRegions}/>
  </>}>
    <Note><strong>Display only:</strong> {hidden?`${visible.filter(Boolean).length.toLocaleString()} of ${data.wn.length.toLocaleString()} channels shown. Excluded curves are hidden, with gaps preserving wavenumber spacing. The vertical axis uses visible values.`:'Amber marks regions excluded from earlier analog matching; all spectral points are shown.'} Saved predictions, correction and model-fitting grids remain unchanged.</Note>
    {difference&&<Note>Pointwise corrected absorbance: VIBES − AIRSpec. Zero means the corrections agree at that channel; a difference does not establish which is chemically correct.</Note>}
    <div ref={ref} className="chart-wrap"><svg width={w} height={385} role="img" aria-label={`Single-filter ${view.toLowerCase()}, ${regions.toLowerCase()}`}><g transform="translate(72,20)"><YAxis scale={y} x={0} gridWidth={iw} label={difference?'Δ absorbance':'Absorbance'}/>{chosen.co2_low!==null&&shade(chosen.co2_low,chosen.co2_high!)}{chosen.upper!==null&&shade(chosen.upper,4000)}{difference&&<line x2={iw} y1={y(0)} y2={y(0)} stroke={INK.fit} strokeDasharray="4 3"/>}{drawn.map(r=><path key={r.label} d={line(r.values)??''} stroke={r.color} strokeWidth={1.7} strokeOpacity={lg.dim(r.label)} fill="none"/>)}<XAxis scale={x} y={ih} label="Wavenumber (cm⁻¹)" tickCount={7}/></g><MethodLegend width={w} y={350} items={series.map(r=>({label:r.label,color:r.color}))} lg={lg}/><text x={24} y={375} fontFamily={FONT.family} fontSize={10} fill={INK.muted}>{`${maskName(mask)} · ${hidden?'hidden':'shaded'} · display only`}</text></svg></div>
    <Note>{s.site}, {s.sample_id}, {s.date}. TOR {f(s.truth)}, AIRSpec {f(s.AIRSpec_prediction)}, VIBES {f(s.VIBES_prediction)} µg/filter. All values refer to the full-pool fit, independent of the cohort/display filters above.</Note>
    <Note>These 12 cases were selected post hoc as three worse and three better cases at BRIS1 and CACR1. They are individual spectra, not representative averages. Both severe negative-prediction cases remain in evaluation.</Note>
  </ChartFrame>
}
function RecoveryPlot({data}:{data:BaselineData}) {
  const ref=useRef<HTMLDivElement>(null);const {width}=useDimensions(ref);const lg=useLegend();const [outcome,setOutcome]=useComparisonParam<string>('boutcome', 'Blank residual RMS', ['Blank residual RMS','Synthetic recovery RMSE'])
  const blank=outcome==='Blank residual RMS'
  const categories=blank?['etad','improve']:['0.01','0.05','0.15']
  const values=(m:BaselineMethod,c:string)=>blank?(data.blanks.find(r=>r.method===m&&r.source===c)?.median_rms??0)*1e4:(data.injections.find(r=>r.method===m&&String(r.amplitude)===c)?.median_recovery_rmse??0)*1e3
  const w=Math.max(300,width),iw=w-95,ih=245
  const x=d3.scaleBand().domain(categories).range([0,iw]).padding(.35)
  const y=d3.scaleLinear().domain([0,(d3.max(categories.flatMap(c=>METHODS.filter(lg.show).map(m=>values(m,c))))??1)*1.2]).nice().range([ih,0])
  return <ChartFrame title="Blank removal and known-addition recovery" controls={<Select label="Spectroscopy outcome" value={outcome} options={['Blank residual RMS','Synthetic recovery RMSE']} onChange={setOutcome}/> }>
    <div ref={ref} className="chart-wrap"><svg width={w} height={335} role="img" aria-label={outcome}><g transform="translate(70,20)"><YAxis scale={y} x={0} gridWidth={iw} label={blank?'Median RMS (10⁻⁴ absorbance)':'Median RMSE (10⁻³ absorbance)'}/><XAxis scale={x} y={ih} label={blank?'Held-out blanks':'Injected amplitude (absorbance)'} format={v=>blank?(v==='etad'?'ETAD (n=9)':'IMPROVE (n=126)'):v}/>{categories.flatMap(c=>METHODS.map((m,i)=>lg.show(m)&&<g key={c+m} opacity={lg.dim(m)}><rect x={x(c)!+i*x.bandwidth()/2} y={y(values(m,c))} width={x.bandwidth()/2} height={ih-y(values(m,c))} fill={COLOR[m]}/><text x={x(c)!+(i+.5)*x.bandwidth()/2} y={y(values(m,c))-6} textAnchor="middle" fontFamily={FONT.family} fontSize={11}>{f(values(m,c),2)}</text></g>))}</g><MethodLegend width={w} y={330} items={METHODS.map(m=>({label:m,color:COLOR[m]}))} lg={lg}/></svg></div>
    <Note>{blank?'Held-out blanks are separate from background training. Smaller residual RMS supports background removal, not independent ambient chemical accuracy.':'The same nine independent parent blanks recur at three amplitudes. Recovery uses corrected(blank + injection) − corrected(blank). VIBES improves larger additions but worsens the weakest. These are not 27 independent parent blanks.'}</Note>
  </ChartFrame>
}
