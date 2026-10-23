import { useRef } from 'react'
import { RefLine } from '@/components/RefLines'
import { useExport } from '@/lib/data'
import { ExportState } from '@/components/ExportState'
import * as d3 from 'd3'
import { ChartFrame, Empty, ExplainOnly, Note, Segmented, Toggle } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis } from '@/components/Axes'
import { SpectrumPlot } from '@/components/SpectrumPlot'
import { SweepPlot } from '@/components/SweepPlot'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useSearchParam } from '@/lib/url'
import { usePageChoice } from '@/lib/pageChoices'
import { SidebarControls } from '@/lib/pageSidebar'
import { FONT, BASELINE_COLOR } from '@/lib/theme'
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
const COLORS: Record<string, string> = { get AIRSpec() { return BASELINE_COLOR.AIRSpec }, get VIBES() { return BASELINE_COLOR['VIBES-full-r70'] }, get uncapped_full() { return BASELINE_COLOR['VIBES-full-r70'] }, get capped_shared() { return BASELINE_COLOR['VIBES-full'] }, get capped_full() { return BASELINE_COLOR['VIBES-full'] }, archived_truncated: '#666666' }
const NAMES: Record<string, string> = { AIRSpec: 'Spline · shared input', VIBES: 'Rank 70 · shared input', uncapped_full: 'Rank 70 · full input', capped_shared: 'Rank 30 · shared input', capped_full: 'Rank 30 · full input', archived_truncated: 'Archived truncated fit' }
const SPECTRUM_NAMES: Record<string, string> = { AIRSpec: 'Spline baseline', uncapped_full: 'VIBES rank 70 · completed', capped_full: 'VIBES rank 30 · preliminary' }
const f = (v: number) => fmt(v, 3)

/** Completed rank-70 benchmark; archived results remain separate. */
export function FullRangeVibesResults() {
  const { data, error } = useExport<Evidence>('vibes_fullrange_results.json', (d) => d.schema_version === 1 ? null : 'unsupported full-range results')
  // the page's calibration cohort (pages/BaselineComparisonPage PageChoices, shown in the sidebar)
  const { value: cohort } = usePageChoice('cohort', ['full_pool', 'locked800'])
  const [extras, setExtras] = useSearchParam<string>('vfextras', 'hide', ['hide', 'show'])
  if (!data) return <ExportState title="Completed full-range results" what="the completed full-range results" error={error} command="uv run --no-sync python gallery/data/export_vibes_fullrange_results.py" />
  const methods = extras === 'show' ? ['AIRSpec', 'VIBES', 'uncapped_full', 'capped_shared', 'capped_full', 'archived_truncated'] : ['AIRSpec', 'VIBES', 'uncapped_full']
  // a variant this export lacks for the cohort is left out, not a crash
  const scores = methods.flatMap(m => data.scores.filter(s => s.cohort === cohort && s.method === m).slice(0, 1))
  const intervals = methods.filter(m => m !== 'AIRSpec').flatMap(m => data.intervals.filter(s => s.cohort === cohort && s.method === m).slice(0, 1))
  return <>
    <SidebarControls title="Benchmark"><Toggle label="Include preliminary and archived variants" checked={extras === 'show'} onChange={v => setExtras(v ? 'show' : 'hide')} /></SidebarControls>
    {/* what the benchmark compares is explanation; its numbers are in the charts below */}
    <ExplainOnly><ChartFrame id="completed-vibes-benchmark" title="Completed VIBES · EC prediction benchmark" exportable={false}>
      <p>Same {scores[0]?.n_train.toLocaleString() ?? '–'} calibration filters and {scores[0]?.n_test.toLocaleString() ?? '–'} held-out IMPROVE filters for every variant. PLS factors are selected using calibration-site cross-validation only.</p>
      <Note>VIBES rank 70 is corrected over 4000–500 cm⁻¹. “Shared input” uses its corrected 4000–1425 channels in the EC model; “full input” uses 4000–500. These source-domain results do not establish EC accuracy at Addis. Seasonal comparisons and the specification grid use completed rank-70 models by default.</Note>
    </ChartFrame></ExplainOnly>
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
      <RefLine orient="v" at={x(0)} from={0} to={ih} />
      {rows.map((r, i) => <g key={r.method} transform={`translate(0,${22+i*44})`} onMouseEnter={e => tip.show(e,[NAMES[r.method], `ΔRMSE ${f(r.delta_rmse)} µg/filter`, `95% interval ${f(r.ci_low)} to ${f(r.ci_high)}`, `${r.n_sites} held-out sites; 2,000 paired resamples`])} onMouseLeave={tip.hide}>
        <text x={-10} textAnchor="end" dy=".35em" fontSize={11}>{NAMES[r.method]}</text><line x1={x(r.ci_low)} x2={x(r.ci_high)} stroke={COLORS[r.method]} strokeWidth={2} /><circle cx={x(r.delta_rmse)} r={5} fill={COLORS[r.method]} />
      </g>)}<XAxis scale={x} y={ih} label="RMSE difference (µg/filter); negative favours VIBES" /></g></svg>{tip.node}</div>
    <Note>Intervals resample {rows[0].n_sites} held-out sites and remain conditional on these fitted models. An interval crossing zero does not establish a gain. Cohorts have different held-out populations and cannot be ranked by their errors alone.</Note>
  </ChartFrame>
}

function AggregateSpectra({ data }: { data: Evidence }) {
  const [population, setPopulation] = useSearchParam<string>('vfpopulation', 'target', ['target', 'calibration'])
  const [range, setRange] = useSearchParam<string>('vfrange', 'Full', ['Full', 'Shared'])
  const legend = useLegend([SPECTRUM_NAMES.capped_full])
  const series = Object.entries(data.spectra[population]).map(([method, s]) => ({ method, ...s }))
  const visible = series.filter(s => legend.show(SPECTRUM_NAMES[s.method]))
  const lo = range === 'Full' ? 500 : 1425
  const ext = d3.extent(visible.flatMap(s => s.q.flatMap(q => q.filter((_, i) => s.wn[i]>=lo))))
  return <ChartFrame id="completed-vibes-spectra" title="Completed correction · paired spectral overlay" controls={<>
    <Segmented label="Spectral population" value={population} options={['target','calibration']} optionLabel={v=>v==='target'?'Addis test-set spectra':'IMPROVE source spectra'} onChange={setPopulation} />
    <Segmented label="Spectral range" value={range} options={['Full', 'Shared']} onChange={setRange} /></>}>
    <div className="chart-wrap">{!visible.length ? <Empty>Select a baseline in the legend to display spectra.</Empty> :
      <SpectrumPlot height={300} window={[4000, lo]} yDomain={[Math.min(0, ext[0]??0), ext[1]??.01]} yLabel="Corrected absorbance" yFormat={v=>Number(v).toPrecision(2)}
        hover="readout" dim={g=>legend.dim(g)}
        series={visible.map(s => ({ id: s.method, label: SPECTRUM_NAMES[s.method], group: SPECTRUM_NAMES[s.method], wn: s.wn, y: s.q[1], color: COLORS[s.method] }))}
        bands={visible.map(s => ({ id: s.method, group: SPECTRUM_NAMES[s.method], wn: s.wn, lo: s.q[0], hi: s.q[2], color: COLORS[s.method], opacity: .12 }))} />}
      <Legend items={series.map(s=>({label:SPECTRUM_NAMES[s.method],color:COLORS[s.method],shape:'line' as const}))} {...legend.props}/></div>
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
  const legend=useLegend()
  const methods=['AIRSpec','capped_shared','VIBES'],shown=methods.filter(m=>legend.show(NAMES[m]))
  return <ChartFrame id="completed-vibes-injections" title="Known-signal preservation · synthetic additions">
    <div className="chart-wrap">{!shown.length?<Empty>Select a method in the legend.</Empty>:
      <SweepPlot height={268} xDomain={[0,.16]} yDomain={[.75,1.03]} reference={1} dim={g=>legend.dim(g)}
        xLabel="Synthetic addition amplitude (absorbance)" yLabel="Recovered / added spectral area"
        series={shown.map(m=>({ id: m, label: NAMES[m], group: NAMES[m], color: COLORS[m], width: 1.5, marks: true, interval: 'none' as const,
          points: data.injections.filter(r=>r.method===m).map(r=>({ x: r.amplitude, y: r.signal_area_recovery })) }))} />}
      <Legend items={methods.map(m=>({label:NAMES[m],color:COLORS[m],shape:'line' as const}))} {...legend.props}/></div>
    <Note>Medians across nine Addis blank parents at each of three synthetic amplitudes (27 additions), evaluated on the shared window. This is a numerical stress test, not chemical standards or independent EC validation.</Note>
  </ChartFrame>
}
