import { useRef } from 'react'
import { useExport } from '@/lib/data'
import { ExportState } from '@/components/ExportState'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Select, Toggle } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useSearchParam } from '@/lib/url'
import { FONT, INK, BASELINE_COLOR } from '@/lib/theme'
import { fmt } from '@/lib/format'

interface Score { cohort: string; method: string; RMSE: number; predictive_R2: number; k: number; n_train: number; n_test: number }
interface Interval { cohort: string; method: string; delta_rmse: number; ci_low: number; ci_high: number; n_sites: number }
interface Spectrum { n: number; wn: number[]; q: number[][] }
interface Evidence {
  schema_version: number; source: string; n_cases: number; recovered_rows: number[]
  scores: Score[]; intervals: Interval[]; labels: Record<string, string>
  blank_median_rms: Record<string, number>
  injections: { method: string; amplitude: number; signal_area_recovery: number; relative_rms_error: number }[]
  spectra: Record<string, Record<string, Spectrum>>
}
// benchmark variants by run: rank 70 is the current VIBES, rank 30 the archived full-range fit
const COLORS: Record<string, string> = { AIRSpec: BASELINE_COLOR.AIRSpec, VIBES: BASELINE_COLOR['VIBES-full-r70'], uncapped_full: BASELINE_COLOR['VIBES-full-r70'], capped_shared: BASELINE_COLOR['VIBES-full'], capped_full: BASELINE_COLOR['VIBES-full'], archived_truncated: '#666666' }
const NAMES: Record<string, string> = { AIRSpec: 'Spline · shared input', VIBES: 'Rank 70 · shared input', uncapped_full: 'Rank 70 · full input', capped_shared: 'Rank 30 · shared input', capped_full: 'Rank 30 · full input', archived_truncated: 'Archived truncated fit' }
const SPECTRUM_NAMES: Record<string, string> = { AIRSpec: 'Spline baseline', uncapped_full: 'VIBES rank 70 · completed', capped_full: 'VIBES rank 30 · preliminary' }
const f = (v: number) => fmt(v, 3)

/** Completed rank-70 benchmark; archived results remain separate. */
export function FullRangeVibesResults() {
  const { data, error } = useExport<Evidence>('vibes_fullrange_results.json', (d) => d.schema_version === 1 ? null : 'unsupported full-range results')
  const [cohort, setCohort] = useSearchParam<string>('vfcohort', 'full_pool', ['full_pool', 'locked800'])
  const [extras, setExtras] = useSearchParam<string>('vfextras', 'hide', ['hide', 'show'])
  if (!data) return <ExportState title="Completed full-range results" what="the completed full-range results" error={error} command="uv run --no-sync python gallery/data/export_vibes_fullrange_results.py" />
  const methods = extras === 'show' ? ['AIRSpec', 'VIBES', 'uncapped_full', 'capped_shared', 'capped_full', 'archived_truncated'] : ['AIRSpec', 'VIBES', 'uncapped_full']
  const scores = methods.map(m => data.scores.find(s => s.cohort === cohort && s.method === m)!)
  const intervals = methods.filter(m => m !== 'AIRSpec').map(m => data.intervals.find(s => s.cohort === cohort && s.method === m)!)
  return <>
    <ChartFrame id="completed-vibes-benchmark" title="Completed VIBES · EC prediction benchmark" exportable={false}
      controls={<><Select label="Calibration cohort" value={cohort} options={['full_pool','locked800']} optionLabel={v=>v==='full_pool'?'Full calibration pool':'Restricted lowest OC/EC'} onChange={setCohort} />
        <Toggle label="Include preliminary and archived variants" checked={extras === 'show'} onChange={v => setExtras(v ? 'show' : 'hide')} /></>}>
      <p>Same {scores[0].n_train.toLocaleString()} calibration filters and {scores[0].n_test.toLocaleString()} held-out IMPROVE filters for every variant. PLS factors are selected using calibration-site cross-validation only.</p>
      <Note>VIBES rank 70 is corrected over 4000–500 cm⁻¹. “Shared input” uses its corrected 4000–1425 channels in the EC model; “full input” uses 4000–500. These source-domain results do not establish EC accuracy at Addis. Seasonal comparisons and the specification grid use completed rank-70 models by default.</Note>
    </ChartFrame>
    <Performance scores={scores} source={data.source} />
    <Intervals rows={intervals} />
    <AggregateSpectra data={data} />
    <BlankMedians data={data} />
    <InjectionRecovery data={data} />
  </>
}

function Performance({ scores, source }: { scores: Score[]; source: string }) {
  const ref = useRef<HTMLDivElement>(null), { width } = useDimensions(ref), tip = useTooltip(ref)
  const w = Math.max(320, width), left = Math.min(210, w * .4), iw = w-left-90, ih = scores.length*46
  const x = d3.scaleLinear().domain([0, d3.max(scores, s => s.RMSE)! * 1.1]).range([0, iw])
  const y = d3.scaleBand().domain(scores.map(s => s.method)).range([0, ih]).padding(.25)
  return <ChartFrame id="completed-vibes-rmse" title="Held-out EC error · lower is better" provenance={source}>
    <div ref={ref} className="chart-wrap"><svg width={w} height={ih+75} role="img" aria-label="Completed VIBES held-out EC RMSE" fontFamily={FONT.family}>
      <g transform={`translate(${left},15)`}>{scores.map(s => <g key={s.method}>
        <text x={-10} y={y(s.method)!+y.bandwidth()/2} textAnchor="end" dy=".35em" fontSize={11}>{NAMES[s.method]}</text>
        <rect x={0} y={y(s.method)} width={x(s.RMSE)} height={y.bandwidth()} fill={COLORS[s.method]} fillOpacity={.8}
          onMouseEnter={e => tip.show(e, [NAMES[s.method], `RMSE ${f(s.RMSE)} µg/filter`, `Predictive R² ${f(s.predictive_R2)}`, `PLS factors K=${s.k}`])} onMouseLeave={tip.hide} />
        <text x={x(s.RMSE)+7} y={y(s.method)!+y.bandwidth()/2} dy=".35em" fontSize={11}>{f(s.RMSE)}</text>
      </g>)}<XAxis scale={x} y={ih} label="Held-out TOR EC RMSE (µg/filter)" /></g>
    </svg>{tip.node}</div>
    <div style={{ overflowX: 'auto' }}><table className="census-table"><thead><tr><th>EC model</th><th>PLS factors K</th><th>RMSE (µg/filter)</th><th>Predictive R²</th></tr></thead><tbody>{scores.map(s => <tr key={s.method}><td>{NAMES[s.method]}</td><td>{s.k}</td><td>{f(s.RMSE)}</td><td>{f(s.predictive_R2)}</td></tr>)}</tbody></table></div>
  </ChartFrame>
}

function Intervals({ rows }: { rows: Interval[] }) {
  const ref = useRef<HTMLDivElement>(null), { width } = useDimensions(ref), tip = useTooltip(ref)
  const w = Math.max(320, width), left = Math.min(210, w*.4), iw = w-left-25, ih = rows.length*44
  const values = rows.flatMap(r => [r.ci_low, r.ci_high, 0]), extent = d3.extent(values) as [number, number]
  const x = d3.scaleLinear().domain(extent).nice().range([0, iw])
  return <ChartFrame id="completed-vibes-intervals" title="Paired uncertainty · VIBES minus spline RMSE">
    <div ref={ref} className="chart-wrap"><svg width={w} height={ih+78} role="img" aria-label="Paired site-bootstrap RMSE intervals" fontFamily={FONT.family}><g transform={`translate(${left},20)`}>
      <line x1={x(0)} x2={x(0)} y2={ih} stroke={INK.identity} strokeDasharray="5 4" />
      {rows.map((r, i) => <g key={r.method} transform={`translate(0,${22+i*44})`} onMouseEnter={e => tip.show(e,[NAMES[r.method], `ΔRMSE ${f(r.delta_rmse)} µg/filter`, `95% interval ${f(r.ci_low)} to ${f(r.ci_high)}`, `${r.n_sites} held-out sites; 2,000 paired resamples`])} onMouseLeave={tip.hide}>
        <text x={-10} textAnchor="end" dy=".35em" fontSize={11}>{NAMES[r.method]}</text><line x1={x(r.ci_low)} x2={x(r.ci_high)} stroke={COLORS[r.method]} strokeWidth={2} /><circle cx={x(r.delta_rmse)} r={5} fill={COLORS[r.method]} />
      </g>)}<XAxis scale={x} y={ih} label="RMSE difference (µg/filter); negative favours VIBES" /></g></svg>{tip.node}</div>
    <Note>Intervals resample {rows[0].n_sites} held-out sites and remain conditional on these fitted models. An interval crossing zero does not establish a gain. Cohorts have different held-out populations and cannot be ranked by their errors alone.</Note>
  </ChartFrame>
}

function AggregateSpectra({ data }: { data: Evidence }) {
  const ref = useRef<HTMLDivElement>(null), { width } = useDimensions(ref)
  const [population, setPopulation] = useSearchParam<string>('vfpopulation', 'target', ['target', 'calibration'])
  const [range, setRange] = useSearchParam<string>('vfrange', 'Full', ['Full', 'Shared'])
  const legend = useLegend([SPECTRUM_NAMES.capped_full])
  const series = Object.entries(data.spectra[population]).map(([method, s]) => ({ method, ...s }))
  const visible = series.filter(s => legend.show(SPECTRUM_NAMES[s.method]))
  const lo = range === 'Full' ? 500 : 1425, w = Math.max(320, width), iw = w-88, ih = 245
  const ext = d3.extent(visible.flatMap(s => s.q.flatMap(q => q.filter((_, i) => s.wn[i]>=lo))))
  const x = d3.scaleLinear().domain([4000, lo]).range([0, iw]), y = d3.scaleLinear().domain([Math.min(0, ext[0]??0), ext[1]??.01]).nice().range([ih, 0])
  return <ChartFrame id="completed-vibes-spectra" title="Completed correction · paired spectral overlay" controls={<>
    <Select label="Spectral population" value={population} options={['target','calibration']} optionLabel={v=>v==='target'?'Addis test-set spectra':'IMPROVE source spectra'} onChange={setPopulation} />
    <Select label="Spectral range" value={range} options={['Full', 'Shared']} onChange={setRange} /></>}>
    <div ref={ref} className="chart-wrap">{!visible.length ? <Empty>Select a baseline in the legend to display spectra.</Empty> : <svg width={w} height={380} role="img" aria-label="Median corrected spectra with interquartile ranges" fontFamily={FONT.family}><g transform="translate(64,15)">
      <YAxis scale={y} x={0} gridWidth={iw} label="Corrected absorbance" format={v=>Number(v).toPrecision(2)} /><XAxis scale={x} y={ih} label="Wavenumber (cm⁻¹)" />
      {visible.map(s => <g key={s.method}>
        <path d={d3.area<number>().defined((_, i)=>s.wn[i]>=lo).x((_,i)=>x(s.wn[i])).y0((_,i)=>y(s.q[0][i])).y1((_,i)=>y(s.q[2][i]))(s.q[1])??''} fill={COLORS[s.method]} fillOpacity={.12} />
        <path aria-label={SPECTRUM_NAMES[s.method]} d={d3.line<number>().defined((_, i)=>s.wn[i]>=lo).x((_,i)=>x(s.wn[i])).y(v=>y(v))(s.q[1])??''} fill="none" stroke={COLORS[s.method]} strokeWidth={1.6} />
      </g>)}
      {visible.map((s,i)=><g key={s.method} transform={`translate(0,${ih+60+i*18})`}><line x2={18} stroke={COLORS[s.method]} strokeWidth={2}/><text x={25} dy=".35em" fontSize={11}>{SPECTRUM_NAMES[s.method]}</text></g>)}
    </g></svg>}<Legend items={series.map(s=>({label:SPECTRUM_NAMES[s.method],color:COLORS[s.method],shape:'line' as const}))} {...legend.props}/></div>
    <Note>Same {series[0].n.toLocaleString()} physical filters for each method; median and IQR of 8-channel display means. Spline spectra end near 1425 cm⁻¹. These broad features do not identify a chemical species, and the export ends at 500 rather than 400 cm⁻¹.</Note>
  </ChartFrame>
}

function BlankMedians({ data }: { data: Evidence }) {
  const ref = useRef<HTMLDivElement>(null), { width } = useDimensions(ref)
  const methods = ['AIRSpec','capped_shared','VIBES','capped_full','uncapped_full']
  const w=Math.max(320,width),left=Math.min(210,w*.4),iw=w-left-80,ih=methods.length*38
  const x=d3.scaleLog().domain([1e-6,1e-3]).range([0,iw])
  return <ChartFrame id="completed-vibes-blanks" title="Held-out blanks · residual medians">
    <div ref={ref} className="chart-wrap"><svg width={w} height={ih+78} role="img" aria-label="Median residual RMS on 135 held-out blanks"><g transform={`translate(${left},20)`} fontFamily={FONT.family}>
      {methods.map((m,i)=><g key={m} transform={`translate(0,${19+i*38})`}><text x={-10} textAnchor="end" dy=".35em" fontSize={11}>{NAMES[m]}</text><circle cx={x(data.blank_median_rms[m])} r={5} fill={COLORS[m]}/><text x={x(data.blank_median_rms[m])+9} dy=".35em" fontSize={10}>{data.blank_median_rms[m].toExponential(2)}</text></g>)}
      <XAxis scale={x as any} y={ih} label="Median residual RMS absorbance (log scale)" format={v=>Math.abs(Math.log10(Number(v))-Math.round(Math.log10(Number(v))))<1e-8?Number(v).toExponential(0):''} />
    </g></svg></div>
    <Note>All 135 held-out blanks remain included, including the two recovered fits. Shared-window comparisons use identical channels. Smaller blank residuals alone do not establish better EC prediction.</Note>
  </ChartFrame>
}

function InjectionRecovery({ data }: { data: Evidence }) {
  const ref=useRef<HTMLDivElement>(null),{width}=useDimensions(ref),legend=useLegend()
  const methods=['AIRSpec','capped_shared','VIBES'],shown=methods.filter(m=>legend.show(NAMES[m]))
  const w=Math.max(320,width),iw=w-88,ih=210
  const x=d3.scaleLinear().domain([0,.16]).range([0,iw]),y=d3.scaleLinear().domain([.75,1.03]).range([ih,0])
  return <ChartFrame id="completed-vibes-injections" title="Known-signal preservation · synthetic additions">
    <div ref={ref} className="chart-wrap">{!shown.length?<Empty>Select a method in the legend.</Empty>:<svg width={w} height={345} role="img" aria-label="Synthetic spectral area recovery" fontFamily={FONT.family}><g transform="translate(64,20)">
      <YAxis scale={y} x={0} gridWidth={iw} label="Recovered / added spectral area"/><XAxis scale={x} y={ih} label="Synthetic addition amplitude (absorbance)"/>
      <line x2={iw} y1={y(1)} y2={y(1)} stroke={INK.identity} strokeDasharray="5 4"/>
      {shown.map(m=>{const rows=data.injections.filter(r=>r.method===m);return <g key={m}><path d={d3.line<typeof rows[number]>().x(r=>x(r.amplitude)).y(r=>y(r.signal_area_recovery))(rows)??''} stroke={COLORS[m]} fill="none" strokeWidth={1.5}/>{rows.map(r=><circle key={r.amplitude} cx={x(r.amplitude)} cy={y(r.signal_area_recovery)} r={4} fill={COLORS[m]}/>)}</g>})}
      {shown.map((m,i)=><g key={m} transform={`translate(0,${ih+60+i*18})`}><line x2={18} stroke={COLORS[m]} strokeWidth={2}/><text x={25} dy=".35em" fontSize={11}>{NAMES[m]}</text></g>)}
    </g></svg>}<Legend items={methods.map(m=>({label:NAMES[m],color:COLORS[m],shape:'line' as const}))} {...legend.props}/></div>
    <Note>Medians across nine Addis blank parents at each of three synthetic amplitudes (27 additions), evaluated on the shared window. This is a numerical stress test, not chemical standards or independent EC validation.</Note>
  </ChartFrame>
}
