import { useRef, useState } from 'react'
import { Marker } from '@/components/RefLines'
import { dataUrl, useExport } from '@/lib/data'
import { ExportState } from '@/components/ExportState'
import * as d3 from 'd3'
import { ChartFrame, ExplainOnly, Note, Select } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { FONT, INK, baselineColor } from '@/lib/theme'
import { fmt as f } from '@/lib/format'
import type { BaselineMethod } from '@/lib/baselineComparison'

const BASE = dataUrl('')
/** Baseline colours follow the run's rank; the per-filter router is its own series. */
const methodColor = (method: string, rank?: number | null) => method === 'routed' ? '#4D7C68' : baselineColor(method, rank)
const MODEL_ORDER = ['full_pool','locked800','analog_full','analog_exclude_1800_2500',
  'analog_exclude_1800_2500_above_3600','analog_exclude_1800_2500_above_3500']
const MODEL_NAME: Record<string,string> = {
  full_pool:'Full pool', locked800:'Lowest OC/EC', analog_full:'Analog · no cut',
  analog_exclude_1800_2500:'Analog · 1800–2500',
  analog_exclude_1800_2500_above_3600:'Analog · + >3600',
  analog_exclude_1800_2500_above_3500:'Analog · + >3500',
}

interface SelectionMetric {
  model:string; method:BaselineMethod; n_train:number; n_train_sites:number; components:number; n:number
  RMSE:number; MAE:number; bias:number; predictive_R2:number
  delta_RMSE_vs_full_AIRSpec:number; delta_RMSE_CI_low:number; delta_RMSE_CI_high:number
}
interface RoutingMetric {
  population:'training_site_nested'|'prior_inspected_test'; method:'AIRSpec'|'VIBES'|'routed'
  n:number; n_sites:number; selected_VIBES:number|null; RMSE:number; MAE:number
  bias:number; predictive_R2:number; delta_RMSE_vs_AIRSpec:number
  delta_RMSE_CI_low:number; delta_RMSE_CI_high:number
}
interface FollowupData {
  schema_version:number; source_run_signature:string; scope:string; background_rank?:number
  common_test_n:number; common_test_sites:number; train_n:number; train_sites:number
  selection_metrics:SelectionMetric[]
  selection_overlap:Record<string,{replaced_vs_unmasked:number;n_training_sites:number}>
  restricted_sensitivity:{method:BaselineMethod;in_historical_membership:boolean;n:number;RMSE:number;MAE:number;bias:number}[]
  routing_metrics:RoutingMetric[]
  routing_extreme:{sample_id:string;Site:string;y:number;VIBES:number;choice:string}[]
  addis_readiness:{status:string;n_addis_targets:number;n_independent_thermal_matches:number}
  source_sha256:Record<string,string>
}

export function BaselineFollowup({ completed = false }: { completed?: boolean } = {}) {
  const prefix = completed ? 'baseline_followup_r70_' : 'baseline_followup_'
  const {data,error}=useExport<FollowupData>(completed ? 'baseline_followup_r70.json' : 'baseline_followup.json', v=>
    v.schema_version!==1||v.scope!=='new_exploratory_followup_on_saved_corrections'
      ||(completed && v.background_rank !== 70)||v.common_test_n!==2327||v.selection_metrics.length!==12||v.routing_metrics.length!==6
      ? 'unexpected notebook follow-up export' : null)
  if(!data) return <ExportState title="Notebook follow-up" what="the executed notebook results" error={error}/>
  return <>
    {completed && <ChartFrame title="Completed rank-70 calibration selection and routing" exportable={false}><p>VIBES is corrected on 4000–500 cm⁻¹ with background rank 70. These EC models use the shared 4000–1425 channels, the original site split and the same training-only routing protocol. Outcomes are exploratory because this IMPROVE test set was previously inspected.</p></ChartFrame>}
    <ExplainOnly><ChartFrame id="baseline-followup" title="Executed notebook follow-up" exportable={false}>
      <p>{completed ? 'The common-test and routing analyses below were rerun through the migration CLI on the completed rank-70 corrections.' : 'The common-test and routing analyses below were run separately through the Jupyter CLI on the earlier Colab corrections.'} They refit EC calibrations; the original 12,808 corrected spectra and outer site split remain frozen. The original test outcomes had already been inspected, so these comparisons are exploratory.</p>
      <p style={{display:'flex',gap:16,flexWrap:'wrap',fontSize:13}}>
        {!completed && <a href={BASE+'baseline_followup_executed.ipynb'} download>Executed notebook</a>}
        <a href={BASE+prefix+'common_test_metrics.csv'} download>Selection metrics</a>
        <a href={BASE+prefix+'common_test_predictions.csv'} download>All common-test predictions</a>
        <a href={BASE+prefix+'routing_metrics.csv'} download>Routing metrics</a>
        <a href={BASE+prefix+'routing_prior_test_predictions.csv'} download>Routing predictions</a>
        <a href={BASE+prefix+'analog_membership.csv'} download>Analog membership</a>
      </p>
      <Note>All selection candidates share {data.common_test_n.toLocaleString()} IMPROVE filters across {data.common_test_sites} test sites. Addis has {data.addis_readiness.n_independent_thermal_matches} independently matched thermal EC references here, so no Addis EC accuracy is reported.</Note>
      <details><summary style={{cursor:'pointer',fontSize:12}}>Notebook sources and SHA-256</summary><p style={{fontSize:11}}>Original run signature: {data.source_run_signature}</p>{Object.entries(data.source_sha256).map(([name,hash])=><p key={name} style={{fontSize:11,overflowWrap:'anywhere'}}><code>{name}</code><br/>{hash}</p>)}</details>
    </ChartFrame></ExplainOnly>
    <SelectionChart data={data}/>
    <RestrictedCheck data={data}/>
    <RoutingChart data={data}/>
  </>
}

function SelectionChart({data}:{data:FollowupData}) {
  const [method,setMethod]=useState<BaselineMethod>('AIRSpec')
  const ref=useRef<HTMLDivElement>(null);const {width}=useDimensions(ref);const tip=useTooltip(ref)
  // a model row this export lacks is left out rather than failing the page
  const rows=MODEL_ORDER.flatMap(model=>data.selection_metrics.filter(r=>r.model===model&&r.method===method).slice(0,1))
  const baseline=data.selection_metrics.find(r=>r.model==='full_pool'&&r.method==='AIRSpec')
  const w=Math.max(width,620),left=195,right=66,iw=w-left-right,ih=276
  const x=d3.scaleLinear().domain([0,Math.max(...rows.map(r=>r.RMSE))*1.13]).nice().range([0,iw])
  const y=d3.scaleBand().domain(MODEL_ORDER).range([0,ih]).padding(.26)
  return <ChartFrame title="Common-test calibration selection" tip={<p>The 500-filter analog memberships use AIRSpec spectral similarity to Addis targets, with masks applied to ranking only; every PLS fit retains 2,002 channels. The lowest-OC/EC membership uses historical carbon labels. All intervals are exploratory site-cluster intervals conditional on the fits.</p>} controls={<Select label="Correction method" value={method} options={['AIRSpec','VIBES']} onChange={v=>setMethod(v as BaselineMethod)}/>} subtitle="All fits scored against the same 2,327 IMPROVE TOR-labelled outer-site filters; lower RMSE is better." provenance={data.background_rank === 70 ? 'Rank-70 migration CLI · newly fitted EC models on completed corrections' : 'Executed CLI notebook · EC fits on archived corrections'}>
    <div ref={ref} className="chart-wrap"><svg width={w} height={ih+100} role="img" aria-label={`${method} calibration selection RMSE on the common test`}><g transform={`translate(${left},23)`} fontFamily={FONT.family}>
      {baseline&&<Marker orient="v" at={x(baseline.RMSE)} from={0} to={ih} />}
      {rows.map(r=><g key={r.model} transform={`translate(0,${y(r.model)})`}>
        <text x={-11} y={y.bandwidth()/2+4} textAnchor="end" fontSize={11} fill={INK.text}>{MODEL_NAME[r.model]}</text>
        <rect y={2} width={x(r.RMSE)} height={Math.max(2,y.bandwidth()-4)} rx={2} fill={methodColor(method, data.background_rank)} fillOpacity={r.model==='full_pool'?1:.77} onMouseEnter={e=>tip.show(e,[`${MODEL_NAME[r.model]} · ${method}`,`RMSE ${f(r.RMSE)}; MAE ${f(r.MAE)} µg/filter`,`${r.n_train} training filters / ${r.n_train_sites} sites; PLS k=${r.components}`,`ΔRMSE vs full-pool AIRSpec ${f(r.delta_RMSE_vs_full_AIRSpec)} [${f(r.delta_RMSE_CI_low)}, ${f(r.delta_RMSE_CI_high)}]`])} onMouseLeave={tip.hide}></rect>
        <text x={x(r.RMSE)+6} y={y.bandwidth()/2+4} fontSize={10} fill={INK.text}>{f(r.RMSE)}</text>
      </g>)}
      <XAxis scale={x} y={ih} label="RMSE vs TOR EC (µg/filter)"/>
      <g transform={`translate(0,${ih+56})`} fontSize={10} fill={INK.text}>
        <rect width={11} height={11} fill={methodColor(method, data.background_rank)}/><text x={17} y={10}>{`${method} RMSE per calibration selection`}</text>
        <line x1={230} x2={254} y1={5.5} y2={5.5} stroke={INK.identity} strokeWidth={1.5} strokeDasharray="4 3"/><text x={260} y={10}>{`Full-pool AIRSpec reference, RMSE ${f(baseline?.RMSE ?? NaN)}`}</text>
      </g>
    </g></svg>{tip.node}</div>
  </ChartFrame>
}

function RestrictedCheck({data}:{data:FollowupData}) {
  const rows=data.restricted_sensitivity
  return <ChartFrame title="Lowest-OC/EC transfer check" exportable={false} subtitle="The historical low-OC/EC calibration is scored both inside and outside its membership on the same outer sites.">
    <div style={{overflowX:'auto'}}><table className="census-table"><thead><tr><th>Method</th><th>Test membership</th><th>Filters</th><th>RMSE</th><th>MAE</th><th>Bias</th></tr></thead><tbody>{rows.map(r=><tr key={r.method+String(r.in_historical_membership)}><td>{r.method}</td><td>{r.in_historical_membership?'Inside':'Outside'}</td><td>{r.n.toLocaleString()}</td><td>{f(r.RMSE)}</td><td>{f(r.MAE)}</td><td>{f(r.bias)}</td></tr>)}</tbody></table></div>
    <Note>The prior restricted-cohort score covered only the 137 members. The 2,190 other test filters expose the cost of applying that fit broadly; this does not identify the physical cause.</Note>
  </ChartFrame>
}

function RoutingChart({data}:{data:FollowupData}) {
  const ref=useRef<HTMLDivElement>(null);const {width}=useDimensions(ref);const tip=useTooltip(ref)
  const w=Math.max(width,620),left=73,right=20,iw=w-left-right,ih=235
  const populations=['training_site_nested','prior_inspected_test'] as const
  const names:Record<string,string>={training_site_nested:'Training-site folds',prior_inspected_test:'Previously inspected test'}
  const methods=['AIRSpec','VIBES','routed'] as const
  const x=d3.scaleBand().domain(populations).range([0,iw]).padding(.25)
  const inner=d3.scaleBand().domain(methods).range([0,x.bandwidth()]).padding(.08)
  const y=d3.scaleLinear().domain([0,(d3.max(data.routing_metrics,r=>r.RMSE)||1)*1.12]).nice().range([ih,0])
  const routed=data.routing_metrics.find(r=>r.population==='prior_inspected_test'&&r.method==='routed')
  const extreme=data.routing_extreme[0]
  return <ChartFrame title="Spectrum-only method routing" subtitle="A training-only spectral classifier chooses AIRSpec or VIBES per filter. Both base methods are retained for comparison." provenance={data.background_rank === 70 ? 'Rank-70 migration CLI · grouped training-site folds refit EC calibration and router' : 'Executed CLI notebook · grouped training-site folds refit EC calibration and router'}>
    <div ref={ref} className="chart-wrap"><svg width={w} height={ih+112} role="img" aria-label="AIRSpec, VIBES and routed EC error across training-site folds and previously inspected test sites"><g transform={`translate(${left},20)`} fontFamily={FONT.family}>
      <YAxis scale={y} x={0} gridWidth={iw} label="RMSE (µg/filter)" labelX={-47}/>
      {data.routing_metrics.map(r=><rect key={r.population+r.method} x={x(r.population)!+inner(r.method)!} y={y(r.RMSE)} width={inner.bandwidth()} height={ih-y(r.RMSE)} fill={methodColor(r.method, data.background_rank)} onMouseEnter={e=>tip.show(e,[`${names[r.population]} · ${r.method}`,`RMSE ${f(r.RMSE)}; MAE ${f(r.MAE)} µg/filter`,`${r.n.toLocaleString()} filters / ${r.n_sites} sites`,r.method==='routed'?`VIBES chosen for ${r.selected_VIBES?.toLocaleString()} filters`:undefined].filter(Boolean) as string[])} onMouseLeave={tip.hide}></rect>)}
      <XAxis scale={x} y={ih} format={v=>names[v]} label="Evaluation population"/>
      {methods.map((method,i)=><g key={method} transform={`translate(${i*125},${ih+67})`}><rect width={11} height={11} fill={methodColor(method, data.background_rank)}/><text x={17} y={10} fontSize={10} fill={INK.text}>{method==='routed'?'Routed':method}</text></g>)}
    </g></svg>{tip.node}</div>
    {routed&&<Note>Routing versus AIRSpec on the previously inspected test: ΔRMSE {f(routed.delta_RMSE_vs_AIRSpec)} µg/filter; paired 95% site interval {f(routed.delta_RMSE_CI_low)} to {f(routed.delta_RMSE_CI_high)}. {routed.delta_RMSE_CI_low <= 0 && routed.delta_RMSE_CI_high >= 0 ? 'The interval includes zero.' : 'The interval excludes zero; this remains an exploratory comparison.'} The large training-fold VIBES RMSE includes {extreme.sample_id} at {extreme.Site} (VIBES {f(extreme.VIBES)} vs TOR {f(extreme.y)}); that filter was not excluded.</Note>}
  </ChartFrame>
}
