import { useMemo, useRef } from 'react'
import { RefLine } from '@/components/RefLines'
import { dataUrl, useExport } from '@/lib/data'
import { ExportState } from '@/components/ExportState'
import { IntervalStrip } from '@/components/IntervalStrip'
import * as d3 from 'd3'
import { ChartFrame, Empty, ExplainOnly, Note, Segmented, Select, Toggle } from '@/components/ChartFrame'
import { PageToc } from '@/components/PageToc'
import { SidebarControls } from '@/lib/pageSidebar'
import { PageChoiceControls, PageChoices, usePageChoice, type PageChoiceDef } from '@/lib/pageChoices'
import { FullRangeVibesStatus } from '@/components/FullRangeVibesStatus'
import { FullRangeVibesResults } from '@/components/FullRangeVibesResults'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { BASELINE_COLOR, FONT, INK } from '@/lib/theme'
import { fmt as f } from '@/lib/format'
import { baselineName } from '@/lib/labels'
import { BaselineRankProvider, useBaselineColors } from '@/lib/baselineRank'
import { type FilterRecord, useHighlightActions } from '@/lib/highlight'
import { baselineMetrics, pairedAbsoluteErrorDifference, retainedWavenumber } from '@/lib/baselineComparison'
import { BaselineResearchDiagnostics, RegionContribution, SiteBandPattern, BlankSpread } from '@/pages/BaselineResearchDiagnostics'
import { Legend, useLegend } from '@/components/Legend'
import { BaselineFollowup } from '@/pages/BaselineFollowup'
import { CompletedVibesDiagnostics } from '@/pages/BaselineResearchDiagnostics'
import { BaselineExternalPilot } from '@/pages/BaselineExternalPilot'
import { useSearchParam as useComparisonParam } from '@/lib/url'
import { useFilterSpectrum, traceVersionLabel } from '@/lib/filterSpectrum'
import type { BaselineCohort, BaselineData, BaselineGroup, BaselineMethod, BaselinePair, BaselineSpectrum, HistoricalMask } from '@/lib/baselineComparison'
import { FixedSetNote } from '@/lib/markedRows'
import { FacetGrid } from '@/components/FacetGrid'
import { ScatterPlot, type ScatterPoint } from '@/components/ScatterPlot'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'

const METHODS: BaselineMethod[] = ['AIRSpec', 'VIBES']
const BASE = dataUrl('')
const cohortName = (x: BaselineCohort) => x === 'full_pool' ? 'Full pool' : 'Restricted lowest-OC/EC'
const maskName = (x: string) => ({ full: 'No selection mask', no_co2: 'Exclude 1800–2500', no_co2_max3600: 'Also exclude >3600', no_co2_max3500: 'Also exclude >3500' }[x] ?? x)

export function BaselineComparisonPage({ mac, demingLambda }: { mac: number; demingLambda: number }) {
  const [cohort, setCohort] = useComparisonParam<BaselineCohort>('bcohort', 'full_pool', ['full_pool','locked800'])
  const [sample, setSample] = useComparisonParam<string>(ARCHIVED_KEYS.sample, 'improve:1970697')
  const [archiveMode, setArchiveMode] = useComparisonParam<string>('barchive', 'hide', ['hide','show'])
  const showArchived=archiveMode==='show'
  const { data, error } = useExport<BaselineData>(showArchived ? 'baseline_comparison.json' : 'baseline_paired_r70.json',
    (d) => d.schema_version !== 1 || (!showArchived && d.run.background_rank !== 70) ? 'unsupported data version' : null)
  const { all, rows, sites, site, setSite, band, setBand } = usePairedSubset(data, cohort, ARCHIVED_KEYS)
  if (!data) return <ExportState title="Baselines" what="the baseline comparison" error={error} command="uv run --locked --no-sync python gallery/data/export_baseline_comparison.py" />
  const subset = site !== 'All sites' || band !== 'All loadings'
  const interval = data.groups.find(g => g.cohort === cohort && g.dimension === 'overall')!
  const spectrum = data.spectra.find(r => r.sample_id === sample) ?? data.spectra[0]
  const masks = data.historical_masks.filter(r => ['full','no_co2','no_co2_max3600','no_co2_max3500'].includes(r.mask))
  // one calibration cohort for the benchmark and the paired diagnostics, set in the sidebar
  const cohortChoice: PageChoiceDef = { key: 'cohort', urlKey: 'r70cohort', label: 'Calibration cohort', default: 'full_pool', options: ['full_pool', 'locked800'], legacy: ['vfcohort'],
    render: (v, set) => <Segmented label="Calibration cohort" value={v} options={['full_pool','locked800']} optionLabel={o => cohortName(o as BaselineCohort)} onChange={set} /> }
  // the version switch lives in the sidebar; what the versions are is explanation (Explain on)
  const archiveControl=<>
    <SidebarControls title="Baselines"><Toggle label="Show archived VIBES fit (4000–1425)" checked={showArchived} onChange={v=>setArchiveMode(v?'show':'hide')}/></SidebarControls>
    <ExplainOnly><ChartFrame title="Baseline versions" exportable={false}>
      <p>Completed rank-70 VIBES results appear below. The earlier fit restricted to 4000–1425 cm⁻¹ and its scores are archived.</p>
      <p><a href="#tab=meeting">Open per-filter overlays and completed rank-70 seasonal comparisons</a></p>
    </ChartFrame></ExplainOnly>
  </>
  if(!showArchived) return <BaselineRankProvider rank={70}><PageChoices panel={false} defs={[cohortChoice]}><PageToc>
    {archiveControl}
    <SidebarControls><PageChoiceControls /></SidebarControls>
    <ExplainOnly><FullRangeVibesStatus /></ExplainOnly>
    <FullRangeVibesResults />
    <PairedViews file="baseline_paired_r70.json" rank={70} keys={COMPLETED_KEYS} />
    <BaselineFollowup completed />
    <CompletedVibesDiagnostics />
    <BaselineExternalPilot completed mac={mac} demingLambda={demingLambda} />
    <SpectraPlot key="current" spectrum={spectrum} data={data} onSample={setSample} masks={masks} archivedBenchmark={false}/>
  </PageToc></PageChoices></BaselineRankProvider>
  return <BaselineRankProvider rank={data.run.background_rank}><PageChoices panel={false} defs={[cohortChoice]}><PageToc>
    {archiveControl}
    <SidebarControls><PageChoiceControls /></SidebarControls>
    <ExplainOnly><FullRangeVibesStatus linkToMeeting /></ExplainOnly>
    <FullRangeVibesResults />
    <ChartFrame title="Archived comparison · VIBES fit 4000–1425 cm⁻¹" exportable={false}>
      <ExplainOnly><p>This page preserves the earlier truncated-grid benchmark and its notebook follow-up. VIBES was fitted after restricting both samples and blanks to approximately 4000–1425 cm⁻¹. Its results do not describe the newer full-range VIBES fit cropped afterwards.</p></ExplainOnly>
      <p><a href="#tab=meeting">Open the current full-range VIBES comparisons</a></p>
    </ChartFrame>
    <ExplainOnly><ChartFrame title="Spline / VIBES comparison" exportable={false}>
      <p style={{ fontSize: 14, lineHeight: 1.6 }}>Frozen results from {data.evidence_date}: <strong>{data.run.n_cases.toLocaleString()} correction cases</strong>, {data.run.n_failed} final failures. The spline baseline (AIRSpec) fits segmented splines. VIBES learns background patterns from {data.run.n_background_blanks} independent blanks (rank {data.run.background_rank}). Each method has its own fitted EC calibration.</p>
      <p style={{ fontSize: 14, lineHeight: 1.6 }}><strong>Spectral-cut clarification:</strong> the earlier 1800–2500 cm⁻¹ exclusion and optional &gt;3600 / &gt;3500 cuts changed <strong>analog selection only</strong>. Those earlier PLS fits still used all 2,002 channels. The original Colab run uses the full pool or lowest-OC/EC membership; the separate notebook follow-up below refits site-restricted masked analog selections. Both correction methods and EC fits use approximately 1426–3998 cm⁻¹.</p>
      <Note>VIBES improves held-out blank removal. The original pooled EC comparison does not establish an accuracy gain. The new exploratory notebook fits and its provenance appear below; neither analysis clips predictions or introduces new exclusions.</Note>
    </ChartFrame></ExplainOnly>
    <ChartFrame title="Paired EC performance" exportable={false} tip={<p><strong>Complete-cohort reference, {cohortName(cohort)}:</strong> ΔRMSE {f(interval.delta_RMSE)} µg/filter, paired 95% interval {f(interval.delta_RMSE_ci_low)} to {f(interval.delta_RMSE_ci_high)}. {subset && 'This interval does not describe the selected display subset.'} Both pooled intervals include zero. Different test populations prevent ranking the two calibration cohorts by these errors.</p>} controls={<>
      <Segmented label="Calibration cohort" value={cohort} options={['full_pool', 'locked800']} optionLabel={v => cohortName(v as BaselineCohort)} onChange={v => { setCohort(v as BaselineCohort); setSite('All sites'); setBand('All loadings') }} />
      <Select label="Test site" value={site} options={sites} onChange={setSite} />
      <Select label="Training-defined loading band" value={band} options={['All loadings','Q1','Q2','Q3','Q4']} onChange={setBand} />
      <button type="button" className="btn quiet" onClick={() => { setSite('All sites'); setBand('All loadings') }}>Reset subset</button>
    </>}>
      <p style={{ fontSize: 13 }}><strong>{rows.length.toLocaleString()} paired test filters</strong> shown out of {all.length.toLocaleString()}, across {new Set(rows.map(r => r.Site)).size} sites. {subset ? 'Exploratory display subset. The saved model and cohort remain unchanged.' : 'Complete frozen test cohort.'}</p>
      <div style={{ overflowX: 'auto' }}><PairedMetrics data={data} rows={rows} cohort={cohort} /></div>
      <Note>Errors and bias: µg/filter. Bias = prediction − TOR reference. Predictive R² = 1 − SSE/SST. Training-only 5-fold site-grouped CV, first major minimum. Test sites do not overlap training sites.</Note>
      <FixedSetNote what="the frozen paired IMPROVE test filters of this benchmark" />
    </ChartFrame>
    <PairedPlots data={data} rows={rows} cohort={cohort} onInspect={setSample} />
    <SpectraPlot spectrum={spectrum} data={data} onSample={setSample} masks={masks} />
    <RecoveryPlot data={data} />
    <BaselineResearchDiagnostics data={data} rows={rows} cohort={cohort} sample={spectrum.sample_id} />
    <BaselineExternalPilot mac={mac} demingLambda={demingLambda} />
    <ChartFrame title="Earlier spectral cuts changed analog membership" exportable={false}>
      <ExplainOnly><Note>Historical AIRSpec analog-selection experiment, 10 September. All Addis target group, 500 selected unique filters. This is a separate experiment from the paired AIRSpec/VIBES benchmark above.</Note></ExplainOnly>
      <div style={{ overflowX: 'auto' }}><table className="census-table"><thead><tr><th>Selection mask</th><th>Matching channels</th><th>Replaced vs no mask / 500</th><th>PLS channels</th></tr></thead><tbody>{masks.map(m => <tr key={m.mask}><td>{maskName(m.mask)}</td><td>{m.n_channels}</td><td>{m.changed_vs_full}</td><td>2002</td></tr>)}</tbody></table></div>
      <ExplainOnly><Note>The primary mask excluded 1800–2500 inclusive. Upper-cut variants retained values ≤3600 or ≤3500. The 1850-start sensitivity variants are retained in the downloadable data. This historical table has no paired masked EC outcome; the executed notebook panel above reports new site-restricted analog selections on a common test population.</Note></ExplainOnly>
      <FixedSetNote what="the 10 September analog-selection experiment's 500 filters" />
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
  </PageToc></PageChoices></BaselineRankProvider>
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
      ['Spline prediction (µg/filter)', r.AIRSpec],
      ['VIBES prediction (µg/filter)', r.VIBES],
      ['Spline error (µg/filter)', r.AIRSpec - r.y],
      ['VIBES error (µg/filter)', r.VIBES - r.y],
      ['closer to TOR', Math.abs(r.VIBES - r.y) < Math.abs(r.AIRSpec - r.y) ? 'VIBES' : Math.abs(r.VIBES - r.y) > Math.abs(r.AIRSpec - r.y) ? 'Spline' : 'tie'],
      ['loading band', r.loading_band],
      ['HIPS lot', r.lot],
      ['calibration cohort', cohortName(r.cohort as BaselineCohort)],
    ],
    note: 'IMPROVE test filter: TOR is the thermal-optical reference for this held-out test, not an Addis measurement.',
  }
}

function ResidualPlot({ rows, available, onInspect, cohort }: { rows: BaselinePair[]; available: Set<string>; onInspect: (s: string) => void; cohort: BaselineCohort }) {
  const COLOR = useBaselineColors()
  const hl = useHighlightActions()
  // One square panel per method on shared axes (components/FacetGrid of ScatterPlots): overlaid at this density the
  // two purples/blues merge into one cloud
  const xDomain = d3.extent(rows, r => r.y) as [number, number]
  const max = d3.max(rows, r => Math.max(Math.abs(r.AIRSpec-r.y),Math.abs(r.VIBES-r.y))) || 1
  const yDomain: [number, number] = [-max*1.05, max*1.05]
  const points = useMemo(() => Object.fromEntries(METHODS.map(method => [method, rows.map((r): ScatterPoint => {
    const has = available.has(r.sample_id)
    return { x: r.y, y: r[method]-r.y, key: r.sample_id, color: COLOR[method], r: has ? 4 : 2.5, ...(has ? { stroke: INK.text, strokeWidth: 1 } : {}),
      tip: [`${r.Site} / ${r.sample_id}`,`${r.date} · ${r.loading_band}`,`TOR ${f(r.y)}, AIRSpec ${f(r.AIRSpec)}, VIBES ${f(r.VIBES)} µg/filter`,`${method} error ${f(r[method]-r.y)}`, has ? 'stored spectrum: a click also selects it below' : 'spectrum not in the 12-case inspection'],
      open: () => { hl.openRecord(pairRecord(r, 'the AIRSpec / VIBES error plot')); if (has) onInspect(r.sample_id) } }
  })])) as Record<BaselineMethod, ScatterPoint[]>, [rows, available, COLOR, hl, onInspect])
  return <ChartFrame title="Test errors by EC loading" subtitle="One panel per method on shared axes. Hover for each physical filter and both predictions. Ringed points have stored spectra in the 12-case investigation. Click one to select its spectrum below." provenance="Frozen paired test predictions. No regression or new fit.">
    <Note>{cohortName(cohort)}, current display subset. All negative predictions remain in the errors.</Note>
    <div className="chart-wrap" role="img" aria-label="Prediction errors by TOR EC mass, one panel each for AIRSpec and VIBES">{!rows.length ? <Empty>No test filters match both selections.</Empty> :
      <FacetGrid items={METHODS} minCell={340} maxCols={2} maxCell={540} gap={36}
        title={(method) => <span style={{ color: COLOR[method] }}>{baselineName(method, { short: true })}</span>}
        note={(method) => `MAE ${f(d3.mean(rows, r => Math.abs(r[method]-r.y)))} µg/filter`}
        render={(method) => <ScatterPlot points={points[method]} xDomain={xDomain} yDomain={yDomain} refY={[0]} density="dense" maxSide={460} note={false}
          xLabel="Measured TOR EC (µg/filter)" yLabel="Prediction − TOR (µg/filter)" />} />}</div>
    <FixedSetNote what="the frozen paired IMPROVE test filters of this benchmark" />
  </ChartFrame>
}
function ErrorDifferencePlot({ rows, cohort }: { rows: BaselinePair[]; cohort: BaselineCohort }) {
  const COLOR = useBaselineColors()
  const ref = useRef<HTMLDivElement>(null)
  const lg = useLegend()
  const side = (b: d3.Bin<number, number>) => b.x1! <= 0 ? 'VIBES closer (<0)' : 'Spline closer (>0)'
  const [binCount, setBinCount] = useComparisonParam<string>('bbins', '40', ['10','20','40','80'])
  const values = rows.map(pairedAbsoluteErrorDifference)
  const bound = d3.max(values, Math.abs) || 1
  const thresholds = d3.range(1, Number(binCount)).map(i => (i - Number(binCount) / 2) * 2 * bound / Number(binCount))
  const bins = d3.bin().domain([-bound, bound]).thresholds(thresholds)(values)
  const yDomain = d3.scaleLinear().domain([0, d3.max(bins.filter(b => lg.show(side(b))), b => b.length) || 1]).nice().domain() as [number, number]
  const better = values.filter(v => v < 0).length, worse = values.filter(v => v > 0).length
  // one bar per bin, coloured by which method it favours (components/SweepPlot bar series across the bins)
  const series: SweepSeries[] = [['VIBES closer (<0)', COLOR.VIBES], ['Spline closer (>0)', COLOR.AIRSpec]].filter(([label]) => lg.show(label)).map(([label, color]) => ({
    id: label, label, color, group: label, bar: true, fill: .8,
    points: bins.flatMap((b, i) => side(b) !== label ? [] : [{ x: (b.x0! + b.x1!) / 2, x0: b.x0!, x1: b.x1!, y: b.length,
      tip: [`${f(b.x0)} to ${f(b.x1)} µg/filter`,`${b.length} filters (${f(100*b.length/rows.length,1)}%)`,`Lower bound included; upper bound ${i===bins.length-1?'included':'excluded'}.`] }]) }))
  return <ChartFrame title="Which method is closer for each filter?" tip={<p>The counts describe individual filters; they do not establish a statistically significant gain. Changing bins changes the display only. Inspired by the <a href="https://www.react-graph-gallery.com/histogram" target="_blank" rel="noreferrer">React Graph Gallery histogram</a>.</p>} controls={<Select label="Histogram bins" value={binCount} options={['10','20','40','80']} onChange={setBinCount}/>}>
    <Note>{cohortName(cohort)}, current display subset. Each filter contributes one value: |VIBES − TOR| − |Spline − TOR|. Negative favors VIBES; positive favors the spline baseline.</Note>
    <div ref={ref} className="chart-wrap" role="img" aria-label="Histogram of paired absolute error differences, VIBES minus AIRSpec">{!rows.length ? <Empty>No test filters match both selections.</Empty> : <>
      <SweepPlot series={series} xDomain={[-bound, bound]} yDomain={yDomain} height={303} dim={lg.dim} xLabel="Δ absolute error (µg/filter)" yLabel="Physical filters"
        yFormat={v=>Number.isInteger(v)?String(v):''} children={({ x, ih }) => <RefLine orient="v" at={(x as d3.ScaleLinear<number, number>)(0)} from={0} to={ih} />} />
      <Legend items={[{label:'VIBES closer (<0)',color:COLOR.VIBES,shape:'square'},{label:'Spline closer (>0)',color:COLOR.AIRSpec,shape:'square'}]} {...lg.props}/>
    </>}</div>
    <Note>{rows.length.toLocaleString()} paired filters: VIBES closer {better.toLocaleString()}, spline closer {worse.toLocaleString()}, exact ties {values.length-better-worse}. Mean difference (ΔMAE): {f(d3.mean(values))}; median: {f(d3.median(values))} µg/filter. All tails are shown. Exact ties fall in the bin starting at zero.</Note>
    <FixedSetNote what="the frozen paired IMPROVE test filters of this benchmark" />
  </ChartFrame>
}
function LoadingPlot({ groups }: { groups: BaselineGroup[] }) {
  const COLOR = useBaselineColors()
  return <ChartFrame title="Loading-band differences" tip={<p>Full-pool Q3 = (3.739, 6.994] µg/filter. Restricted Q3 = (2.915, 7.881]. Boundaries come from each cohort's training labels. They do not establish a prediction-time switching rule.</p>} subtitle="Points show VIBES minus AIRSpec RMSE. Negative favors VIBES. Whiskers are paired 95% site-cluster intervals, conditional on the frozen models.">
    <div className="chart-wrap"><IntervalStrip labelWidth={130} rowHeight={30} panels={[{ label: 'VIBES − AIRSpec RMSE (µg/filter)', reference: 0 }]}
      blocks={[{ key: 'bands', rows: groups.map(g => ({ key: g.group, label: `${g.group} (n=${g.n})`,
        tip: [g.group, `ΔRMSE ${f(g.delta_RMSE)} µg/filter`, `interval ${f(g.delta_RMSE_ci_low)} to ${f(g.delta_RMSE_ci_high)}`],
        marks: [[{ lo: g.delta_RMSE_ci_low ?? g.delta_RMSE, hi: g.delta_RMSE_ci_high ?? g.delta_RMSE, mid: g.delta_RMSE, color: COLOR.VIBES, r: 5 }]] })) }]} /></div>
    <FixedSetNote what="the frozen paired IMPROVE test filters of this benchmark" />
  </ChartFrame>
}
/** one spectrum panel's height: the plot, its axis and the caption line (stacked panels name their method there) */
const PANEL_H = 360
function SpectraPlot({ spectrum:s, data, onSample, masks, archivedBenchmark=true }: { spectrum: BaselineSpectrum; data: BaselineData; onSample:(x:string)=>void; masks:HistoricalMask[]; archivedBenchmark?:boolean }) {
  const COLOR = useBaselineColors()
  const ref=useRef<HTMLDivElement>(null);const {width}=useDimensions(ref);const lg=useLegend()
  const [view,setView]=useComparisonParam<string>('bview', 'Corrected spectra', archivedBenchmark?['Corrected spectra','Raw and baselines','VIBES − AIRSpec']:['Corrected spectra']);const [mask,setMask]=useComparisonParam<string>('bmask', archivedBenchmark?'no_co2_max3600':'full', ['full','no_co2','no_co2_max3600','no_co2_max3500'])
  const [regions,setRegions]=useComparisonParam<string>('bregions', 'Shade excluded regions', ['Shade excluded regions','Hide excluded regions'])
  const [range,setRange]=useComparisonParam<string>('brange', archivedBenchmark?'Shared (4000–1425)':'Full (4000–500)', ['Shared (4000–1425)','Full (4000–500)'])
  const [layout,setLayout]=useComparisonParam<string>('bspectrumlayout', 'Overlay', ['Overlay','One method at a time','Stacked'])
  const [method,setMethod]=useComparisonParam<string>('bspectrummethod', 'Spline baseline')
  const fullRange=range==='Full (4000–500)'
  const [vibesRank,setVibesRank]=useComparisonParam<'30'|'70'>('bvibesRank','70',['30','70'])
  const fullSpectrum=useFilterSpectrum((!archivedBenchmark||fullRange)&&view==='Corrected spectra'?s.sample_id:null,archivedBenchmark?['VIBES']:['AIRSpec','VIBES'],vibesRank)
  const fullTrace=fullSpectrum?.series.VIBES
  const hidden=regions==='Hide excluded regions'
  const chosen=masks.find(m=>m.mask===mask)!
  const visible=(wn:number)=>wn>=(fullRange?500:1425)&&wn<=4000&&(!hidden||retainedWavenumber(wn,chosen))
  const w=Math.max(300,width),iw=w-100,ih=270
  const x=d3.scaleLinear().domain([4000,fullRange?500:1425]).range([0,iw])
  const difference=view==='VIBES − AIRSpec'
  const archived=difference?[{label:'VIBES − spline',values:s.vibes.map((v,i)=>v-s.airspec[i]),color:COLOR.VIBES}]:view==='Corrected spectra'?[{label:'Spline baseline',values:s.airspec,color:COLOR.AIRSpec},{label:'VIBES',values:s.vibes,color:COLOR.VIBES}]:[{label:'Raw',values:s.raw,color:INK.muted},{label:'Spline baseline curve',values:s.raw.map((v,i)=>v-s.airspec[i]),color:COLOR.AIRSpec},{label:'VIBES baseline',values:s.raw.map((v,i)=>v-s.vibes[i]),color:COLOR.VIBES}]
  const currentAir=fullSpectrum?.series.AIRSpec
  const currentSeries=[...(currentAir?[{label:'Spline baseline',values:Array.from(currentAir.y),wn:currentAir.wn,color:COLOR.AIRSpec}]:[]),...(fullTrace?[{label:'Full-range VIBES',values:Array.from(fullTrace.y),wn:fullTrace.wn,color:BASELINE_COLOR['VIBES-full-r70']}]:[])]
  const series=archivedBenchmark?[...archived.map(r=>({...r,label:r.label==='VIBES'?'Archived VIBES':r.label,wn:data.wn})),...(fullRange&&fullTrace&&view==='Corrected spectra'?[{label:'Full-range VIBES',values:Array.from(fullTrace.y),wn:fullTrace.wn,color:BASELINE_COLOR['VIBES-full-r70']}]:[])]:currentSeries
  const active=series.some(r=>r.label===method)?method:series[0]?.label??'Spline baseline'
  const drawn=series.filter(r=>lg.show(r.label)&&(layout!=='One method at a time'||r.label===active))
  // Keep a shared vertical scale when switching or stacking methods.
  const extent=d3.extent(series.flatMap(r=>r.values.filter((v,i)=>visible(r.wn[i])&&Number.isFinite(v))))
  const ext:[number,number]=[extent[0]??0,extent[1]??0.01]
  if(difference){const bound=Math.max(Math.abs(ext[0]),Math.abs(ext[1]));ext[0]=-bound;ext[1]=bound}
  const pad=(ext[1]-ext[0])*.06||0.01
  const y=d3.scaleLinear().domain([ext[0]-pad,ext[1]+pad]).nice().range([ih,0])
  const line=(r:typeof series[number])=>d3.line<number>().defined((v,i)=>visible(r.wn[i])&&Number.isFinite(v)).x((_,i)=>x(r.wn[i])).y(v=>y(v))(r.values)??''
  const shade=(lo:number,hi:number)=><g key={lo}>{hidden?<><line x1={x(lo)} x2={x(lo)} y2={ih} stroke={INK.axis} strokeDasharray="3 3"/><line x1={x(hi)} x2={x(hi)} y2={ih} stroke={INK.axis} strokeDasharray="3 3"/><text x={(x(lo)+x(hi))/2} y={15} textAnchor="middle" fontFamily={FONT.family} fontSize={11} fill={INK.muted}>Hidden</text></>:<rect x={x(hi)} width={x(lo)-x(hi)} height={ih} fill="#c49442" fillOpacity={0.18}/>}</g>
  return <ChartFrame id="baseline-spectrum" title={archivedBenchmark?'One physical filter, both corrections':'One physical filter, spline and full-range VIBES'} controls={<>
    <Segmented label="VIBES run" value={vibesRank} options={['70','30']} optionLabel={v=>v==='70'?'Completed · rank 70':'Archived · preliminary rank 30'} onChange={v=>setVibesRank(v as '30'|'70')}/>
    <Select label="Inspection case" value={s.sample_id} options={data.spectra.map(r=>r.sample_id)} optionLabel={id=>{const r=data.spectra.find(x=>x.sample_id===id)!; return `${r.site} / ${r.sample_id}${archivedBenchmark?` (${r.direction})`:''}`}} onChange={onSample}/>
    {archivedBenchmark&&<Segmented label="Spectrum view" value={view} options={['Corrected spectra','Raw and baselines','VIBES − AIRSpec']} optionLabel={v=>v==='VIBES − AIRSpec'?'VIBES − spline':v} onChange={setView}/>}
    <Segmented label="Spectrum range" value={range} options={['Shared (4000–1425)','Full (4000–500)']} onChange={setRange}/>
    {/* one curve (the VIBES − AIRSpec difference) has no layout to choose */}
    {series.length>1&&<Segmented label="Spectrum layout" value={layout} options={['Overlay','One method at a time','Stacked']} onChange={setLayout}/>}
    {layout==='One method at a time'&&series.length>0&&<Select label="Display method" value={active} options={series.map(r=>r.label)} onChange={setMethod}/>}
    <Select label="Earlier selection mask" value={mask} options={masks.map(m=>m.mask)} optionLabel={maskName} onChange={setMask}/>
    {/* the full mask excludes nothing, so there is nothing to shade or hide */}
    {mask!=='full'&&<Segmented label="Excluded regions" value={regions} options={['Shade excluded regions','Hide excluded regions']} optionLabel={v=>v.replace(' excluded regions','')} onChange={setRegions}/>}
  </>}>
    <Note><strong>Display only:</strong> {hidden?'Excluded curves are hidden, with gaps preserving wavenumber spacing. The vertical axis uses visible values.':'Amber marks regions excluded from earlier analog matching; all available spectral points are shown.'} Saved predictions, correction and model-fitting grids remain unchanged.</Note>
    {archivedBenchmark&&fullRange&&<Note>The archived raw spectrum and paired corrections end near 1425 cm⁻¹. {view==='Corrected spectra'?<>Orange shows the selected full-range VIBES correction (4000–500 cm⁻¹; 8-channel display means{fullTrace?`; ${traceVersionLabel(fullTrace)}`:''}). The prediction values below still refer to the archived fits.{!fullTrace&&' Full-range trace is loading or unavailable for this filter.'}</>:<>Choose Corrected spectra to display the current full-range VIBES trace. Raw/baseline and difference values below 1425 cm⁻¹ are unavailable.</>}</Note>}
    {!archivedBenchmark&&<Note>VIBES is corrected on 4000–500 cm⁻¹{fullTrace?` (${traceVersionLabel(fullTrace)})`:''}; the spline ends near 1425 cm⁻¹. Traces are 8-channel display means. Selecting the shared window crops the display of the full-range fit.{!fullTrace&&' Full-range VIBES is loading or unavailable; no archived trace is substituted.'}</Note>}
    {difference&&<Note>Pointwise corrected absorbance: VIBES − spline baseline. Zero means the corrections agree at that channel; a difference does not establish which is chemically correct.</Note>}
    <div ref={ref} className="chart-wrap"><svg width={w} height={PANEL_H*(layout==='Stacked'?Math.max(1,drawn.length):1)} role="img" aria-label={`Single-filter ${view.toLowerCase()}, ${range}, ${layout}`}>
      {(layout==='Stacked'&&drawn.length?drawn.map(r=>[r]):[drawn]).map((panel,k)=><g key={k} transform={`translate(0,${k*PANEL_H})`} data-spectrum-panel={layout==='Stacked'?panel[0]?.label??'comparison':'comparison'}>
        <g transform="translate(72,20)"><YAxis scale={y} x={0} gridWidth={iw} label={difference?'Δ absorbance':'Absorbance'}/>{chosen.co2_low!==null&&shade(chosen.co2_low,chosen.co2_high!)}{chosen.upper!==null&&shade(chosen.upper,4000)}{difference&&<line x2={iw} y1={y(0)} y2={y(0)} stroke={INK.fit} strokeDasharray="4 3"/>}{panel.map(r=><path key={r.label} aria-label={r.label} d={line(r)} stroke={r.color} strokeWidth={1.7} strokeOpacity={lg.dim(r.label)} fill="none"/>)}<XAxis scale={x} y={ih} label="Wavenumber (cm⁻¹)" tickCount={7}/></g>
        <text x={24} y={350} fontFamily={FONT.family} fontSize={10} fill={INK.muted}>{layout==='Stacked'&&panel[0]&&<tspan fill={panel[0].color} fontWeight={600}>{`${panel[0].label} · `}</tspan>}{`${range} · VIBES ${vibesRank==='70'?'completed rank 70':'preliminary rank 30'} · ${maskName(mask)} · display only`}</text>
      </g>)}
    </svg><Legend items={series.map(r=>({label:r.label,color:r.color,shape:'line' as const}))} {...lg.props}/></div>
    <Note>{s.site}, {s.sample_id}, {s.date}. TOR {f(s.truth)} µg/filter.{archivedBenchmark&&<> AIRSpec {f(s.AIRSpec_prediction)}, archived VIBES {f(s.VIBES_prediction)} µg/filter. Predictions refer to the archived full-pool benchmark.</>}</Note>
    <Note>These 12 cases were selected post hoc as three worse and three better cases at BRIS1 and CACR1 in the archived benchmark. They are individual spectra, not representative averages or a ranking of current full-range performance.</Note>
  </ChartFrame>
}
/** Where a paired view keeps its subset in the link: the archived page and the completed rank-70 views each their own. */
interface PairedKeys { site: string; band: string; sample: string }
const ARCHIVED_KEYS: PairedKeys = { site: 'bsite', band: 'bband', sample: 'bsample' }
const COMPLETED_KEYS: PairedKeys = { site: 'r70site', band: 'r70band', sample: 'r70sample' }
const LOADING_BANDS = ['All loadings','Q1','Q2','Q3','Q4']

/** One cohort's paired test filters and the site / loading-band subset on display, kept in the link under `keys`. */
function usePairedSubset(data: BaselineData | null | undefined, cohort: BaselineCohort, keys: PairedKeys) {
  const [site, setSite] = useComparisonParam<string>(keys.site, 'All sites')
  const [band, setBand] = useComparisonParam<string>(keys.band, 'All loadings', LOADING_BANDS)
  const all = useMemo(() => data?.pairs.filter(r => r.cohort === cohort) ?? [], [data, cohort])
  const rows = useMemo(() => all.filter(r => (site === 'All sites' || r.Site === site) && (band === 'All loadings' || r.loading_band === band)), [all, site, band])
  const sites = useMemo(() => ['All sites', ...Array.from(new Set(all.map(r => r.Site))).sort()], [all])
  return { all, rows, sites, site, setSite, band, setBand }
}

/** RMSE, MAE, bias and R² of the displayed subset per method; `rank` names the completed run's baselines in full. */
function PairedMetrics({ data, rows, cohort, rank }: { data: BaselineData; rows: BaselinePair[]; cohort: BaselineCohort; rank?: number }) {
  const COLOR = useBaselineColors()
  return <table className="census-table"><thead><tr><th>{rank ? 'Baseline' : 'Method'}</th><th>{rank ? 'PLS factors' : 'PLS k'}</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>Predictive R²</th></tr></thead><tbody>{METHODS.map(method => {
    const m = baselineMetrics(rows, method)
    return <tr key={method}>{rank ? <td>{baselineName(method, { backgroundRank: rank })}</td> : <td style={{ color: COLOR[method], fontWeight: 600 }}>{baselineName(method, { short: true })}</td>}<td>{data.scores.find(s => s.method === method && s.cohort === cohort)?.k}</td><td>{f(m?.rmse)}</td><td>{f(m?.mae)}</td><td>{f(m?.bias)}</td><td>{f(m?.r2)}</td></tr>
  })}</tbody></table>
}

/** The three paired charts both benchmarks draw on their displayed subset: errors by loading, which method is closer, loading bands. */
function PairedPlots({ data, rows, cohort, onInspect }: { data: BaselineData; rows: BaselinePair[]; cohort: BaselineCohort; onInspect: (s: string) => void }) {
  const available = useMemo(() => new Set(data.spectra.map(r => r.sample_id)), [data])
  return <>
    <ResidualPlot rows={rows} onInspect={onInspect} available={available} cohort={cohort} />
    <ErrorDifferencePlot rows={rows} cohort={cohort} />
    <LoadingPlot groups={data.groups.filter(g => g.cohort === cohort && g.dimension === 'loading_band')} />
  </>
}

/** Current shared-window predictions (the completed rank-70 run, the only one with these views); archived benchmark remains unchanged. */
function PairedViews({ file, rank, keys }: { file: string; rank: number; keys: PairedKeys }) {
  // the same file as the page above; the shared cache downloads it once
  const { data, error } = useExport<BaselineData>(file,
    (d) => d.run.background_rank !== rank || d.pairs.length !== 2464 ? 'unexpected current paired export' : null)
  const { value: cohort } = usePageChoice('cohort', ['full_pool','locked800']) as { value: BaselineCohort }
  const { all, rows, sites, site, setSite, band, setBand } = usePairedSubset(data, cohort, keys)
  const [sample, setSample] = useComparisonParam<string>(keys.sample, 'improve:1970697')
  if (!data) return <ExportState title="Rank-70 paired diagnostics" what="the rank-70 paired diagnostics" error={error} />
  const selected = data.spectra.find(r => r.sample_id === sample) ?? data.spectra[0]
  const interval = data.groups.find(r => r.cohort === cohort && r.dimension === 'overall')!
  return <>
    <SidebarControls title="Paired diagnostics">
      <Select label="Test site" value={sites.includes(site) ? site : 'All sites'} options={sites} onChange={setSite} />
      <Select label="Training-defined loading band" value={band} options={LOADING_BANDS} onChange={setBand} />
      <Select label="Decomposition inspection case" value={selected.sample_id} options={data.spectra.map(r=>r.sample_id)} onChange={setSample} />
    </SidebarControls>
    <ChartFrame id="rank70-paired" title="Completed rank 70 · paired EC diagnostics" exportable={false}>
      <ExplainOnly><p>VIBES correction: 4000–500 cm⁻¹, rank {rank}. Both EC models use the shared 4000–1425 channels.</p></ExplainOnly>
      <p className="chart-note" style={{ marginTop: 0 }}>{cohortName(cohort)} · {rows.length.toLocaleString()} of {all.length.toLocaleString()} paired test filters shown</p>
      <PairedMetrics data={data} rows={rows} cohort={cohort} rank={rank} />
      <Note>Complete-cohort ΔRMSE: {f(interval.delta_RMSE)} µg/filter; paired site interval {f(interval.delta_RMSE_ci_low)} to {f(interval.delta_RMSE_ci_high)}. This interval conditions on the saved fits and does not describe a selected display subset. The previously inspected test set makes these diagnostics exploratory.</Note>
      <p><a href={BASE+file.replace(/\.json$/, '_predictions.csv')} download>Rank-{rank} paired predictions</a> · <a href={BASE+file} download>Current diagnostics and source hashes</a></p>
      <FixedSetNote what="the frozen paired IMPROVE test filters of this benchmark" />
    </ChartFrame>
    <PairedPlots data={data} rows={rows} cohort={cohort} onInspect={setSample} />
    <RegionContribution data={data} sample={selected.sample_id} />
    <SiteBandPattern rows={rows} cohort={cohort} />
    <BlankSpread data={data} />
  </>
}
function RecoveryPlot({data}:{data:BaselineData}) {
  const COLOR = useBaselineColors()
  const ref=useRef<HTMLDivElement>(null);const lg=useLegend();const [outcome,setOutcome]=useComparisonParam<string>('boutcome', 'Blank residual RMS', ['Blank residual RMS','Synthetic recovery RMSE'])
  const blank=outcome==='Blank residual RMS'
  const categories=blank?['etad','improve']:['0.01','0.05','0.15']
  const values=(m:BaselineMethod,c:string)=>blank?(data.blanks.find(r=>r.method===m&&r.source===c)?.median_rms??0)*1e4:(data.injections.find(r=>r.method===m&&String(r.amplitude)===c)?.median_recovery_rmse??0)*1e3
  const yDomain=d3.scaleLinear().domain([0,(d3.max(categories.flatMap(c=>METHODS.filter(lg.show).map(m=>values(m,c))))??1)*1.2]).nice().domain() as [number,number]
  const unit=blank?'Median RMS (10⁻⁴ absorbance)':'Median RMSE (10⁻³ absorbance)'
  // two bars per category, labelled with their value (components/SweepPlot bar series on a point scale)
  const series:SweepSeries[]=METHODS.filter(lg.show).map(m=>({id:m,label:m,color:COLOR[m],group:m,bar:true,fill:1,valueLabel:v=>f(v,2),
    points:categories.map(c=>({x:c,y:values(m,c),tip:[m,`${c}: ${f(values(m,c),2)}`,unit]}))}))
  return <ChartFrame title="Blank removal and known-addition recovery" controls={<Segmented label="Spectroscopy outcome" value={outcome} options={['Blank residual RMS','Synthetic recovery RMSE']} onChange={setOutcome}/> }>
    <div ref={ref} className="chart-wrap" role="img" aria-label={outcome}>
      <SweepPlot series={series} xScale="point" yDomain={yDomain} height={303} barWidth={.65} dim={lg.dim} yLabel={unit} xLabel={blank?'Held-out blanks':'Injected amplitude (absorbance)'}
        xFormat={v=>blank?(v==='etad'?'ETAD (n=9)':'IMPROVE (n=126)'):String(v)}/>
      <Legend items={METHODS.map(m=>({label:m,color:COLOR[m],shape:'square' as const}))} {...lg.props}/>
    </div>
    <Note>{blank?'Held-out blanks are separate from background training. Smaller residual RMS supports background removal, not independent ambient chemical accuracy.':'The same nine independent parent blanks recur at three amplitudes. Recovery uses corrected(blank + injection) − corrected(blank). VIBES improves larger additions but worsens the weakest. These are not 27 independent parent blanks.'}</Note>
  </ChartFrame>
}
