import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, ExplainOnly, Note, Select } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { FONT, INK } from '@/lib/theme'
import { useHighlight } from '@/lib/highlight'
import { useLegend } from '@/components/Legend'
import { BaselineFollowup } from '@/pages/BaselineFollowup'
import type { ApplicabilityData, ApplicabilityRow, BaselineCohort, BaselineContribution, BaselineData, BaselineMethod, BaselinePair } from '@/lib/baselineComparison'

const METHODS: BaselineMethod[] = ['AIRSpec','VIBES']
const COLOR = { AIRSpec:'#2C6E9E', VIBES:'#7A4FA3' }
const f = (x:number|null|undefined,n=2) => x == null || !Number.isFinite(x) ? '—' : x.toFixed(n)
const BANDS = ['1425_1799','1800_2499','2500_2999','3000_4000'] as const
const bandName = (b:string) => b.replace('_','–')
const metricKey = (m:string) => m === 'PCA score distance' ? 't2_train_percentile' : 'q_train_percentile'

export function BaselineResearchDiagnostics({ data, rows, cohort, sample }: { data:BaselineData; rows:BaselinePair[]; cohort:BaselineCohort; sample:string }) {
  const [app,setApp] = useState<ApplicabilityData|null>(null)
  const [error,setError] = useState('')
  useEffect(()=>{
    const controller = new AbortController()
    fetch(`${import.meta.env.BASE_URL}data/baseline_applicability.json`,{signal:controller.signal})
      .then(r=>{if(!r.ok)throw new Error(`HTTP ${r.status}`);return r.json()})
      .then((v:ApplicabilityData)=>{if(v.schema_version!==1||v.scope!=='full_pool'||v.rows.length!==5160)throw new Error('Unexpected distance export');setApp(v)})
      .catch(e=>{if(e.name!=='AbortError')setError(String(e))})
    return ()=>controller.abort()
  },[])
  return <>
    <ExplainOnly><ChartFrame id="baseline-research" title="Research diagnostics" exportable={false}>
      <p>The executed notebook follow-up compares calibration selections on one IMPROVE test population and checks a spectrum-only routing rule. The other views inspect the original frozen corrections and predictions. All follow-up outcomes are exploratory because the original test set was already examined. The distance view adds a training-only PCA fit; it is not a validated operating cutoff.</p>
      <p><a href="https://amt.copernicus.org/articles/9/441/2016/" target="_blank" rel="noreferrer">Site-transfer study</a> · <a href="https://escholarship.org/uc/item/3ms682c5" target="_blank" rel="noreferrer">Multilevel EC study</a> · <a href="https://amt.copernicus.org/articles/9/2615/2016/" target="_blank" rel="noreferrer">Baseline-correction study</a></p>
    </ChartFrame></ExplainOnly>
    <BaselineFollowup/>
    {app ? <ApplicabilityViews data={app}/> : <ChartFrame title="Training spectral space" exportable={false}><Empty>{error||'Loading training-only spectral distances…'}</Empty></ChartFrame>}
    <RegionContribution data={data} sample={sample}/>
    <SiteBandPattern rows={rows} cohort={cohort}/>
    <BlankSpread data={data}/>
  </>
}

function ApplicabilityViews({data}:{data:ApplicabilityData}) {
  const [metric,setMetric]=useState('PCA score distance')
  const key=metricKey(metric)
  const ref=useRef<HTMLDivElement>(null);const {width}=useDimensions(ref)
  const tip=useTooltip(ref);const lg=useLegend()
  const w=Math.max(300,width),iw=w-105,ih=225
  const x=d3.scaleLinear().domain([0,100]).range([0,iw])
  const test=data.rows.filter(r=>r.kind==='test')
  const addis=data.rows.filter(r=>r.kind==='addis')
  const bins=[0,20,40,60,80].flatMap(low=>METHODS.map(method=>{
    const values=test.filter(r=>r.method===method && r[key]>=low && (low===80 ? r[key]<=100:r[key]<low+20))
    return {method,low,n:values.length,mae:d3.mean(values,r=>r.abs_error!)??0}
  }))
  const y=d3.scaleLinear().domain([0,(d3.max(bins.filter(b=>lg.show(b.method)),b=>b.mae)||1)*1.15]).nice().range([ih,0])
  const cx=(low:number)=>x(low+10)
  const high=(rows:ApplicabilityRow[],method:BaselineMethod)=>{
    const group=rows.filter(r=>r.method===method)
    return group.filter(r=>r[key]>95).length/group.length*100
  }
  return <>
    <ChartFrame title="Distance from training spectra" controls={<Select label="Spectral diagnostic" value={metric} options={['PCA score distance','Unmodeled spectral variation']} onChange={setMetric}/>}
      subtitle="Mean held-out absolute EC error by percentile of the training-only spectral distribution. Each method uses its own PCA fit.">
      <Note>Full-pool held-out IMPROVE filters only; 2,327 physical test filters per method. {metric==='PCA score distance'?'Score distance (T²) from an eight-component PCA fit. This is not the saved PLS score space.':'Residual spectral RMS (Q) after eight PCA components.'} Values are percentiles of the corresponding training spectra.</Note>
      <div ref={ref} className="chart-wrap"><svg width={w} height={310} role="img" aria-label="Mean absolute held-out EC error by training spectral distance percentile"><g transform="translate(70,20)"><YAxis scale={y} x={0} gridWidth={iw} label="Mean |error| (µg/filter)"/><rect x={x(95)} width={x(100)-x(95)} height={ih} fill="#c49442" fillOpacity={.12}/>{METHODS.filter(lg.show).map(method=>{
        const m=bins.filter(b=>b.method===method)
        return <g key={method} opacity={lg.dim(method)}><path d={d3.line<(typeof m)[number]>().x(b=>cx(b.low)).y(b=>y(b.mae))(m)??''} stroke={COLOR[method]} strokeWidth={2} fill="none"/>{m.map(b=><circle key={b.low} cx={cx(b.low)} cy={y(b.mae)} r={4.5} fill={COLOR[method]} onMouseEnter={e=>tip.show(e,[method,`Training percentile ${b.low}–${b.low+20}`,`${b.n} filters; mean |error| ${f(b.mae,3)} µg/filter`])} onMouseLeave={tip.hide}><title>{`${method}, ${b.low}–${b.low+20}: n=${b.n}, MAE ${f(b.mae,3)}`}</title></circle>)}</g>
      })}<XAxis scale={x} y={ih} label="Training spectral percentile"/></g><Legend width={w} y={300} items={METHODS.map(m=>({label:m,color:COLOR[m]}))} lg={lg}/></svg>{tip.node}</div>
      <Note>This is an exploratory binning of already inspected test outcomes. Mean errors can rise with EC loading or site composition. No cutoff has been selected or validated.</Note>
    </ChartFrame>
    <ChartFrame title="How often are spectra beyond the training 95th percentile?" tip={<><p>Full-pool models, {data.counts.test.toLocaleString()} test and {data.counts.addis} Addis physical filters. Addis has no thermal EC reference here; its bars describe spectral placement only.</p><p>Different baseline corrections produce different spectral spaces. A low percentile does not prove accuracy; a high percentile is a prompt to inspect, not grounds for excluding a filter.</p></>} subtitle="Proportion of each population with a spectral diagnostic above its own method's training 95th percentile.">
      <div className="chart-wrap"><svg width={w} height={255} role="img" aria-label="Percent of held-out IMPROVE and Addis spectra above training 95th percentile"><g transform="translate(70,20)">{['test','addis'].flatMap((kind,i)=>METHODS.map((method,j)=>{
        const v=high(kind==='test'?test:addis,method)
        const yy=15+i*96+j*31
        return <g key={kind+method} fontFamily={FONT.family} fontSize={11}><text x={-10} y={yy+13} textAnchor="end" fill={INK.text}>{kind==='test'?'IMPROVE':'Addis'}</text><rect x={0} y={yy} width={iw} height={16} fill={INK.grid}/><rect x={0} y={yy} width={iw*v/100} height={16} fill={COLOR[method]}/><text x={Math.min(iw-32,iw*v/100+5)} y={yy+13} fill={INK.text}>{f(v,1)}% · {method}</text></g>
      }))}</g></svg></div>
    </ChartFrame>
  </>
}

function RegionContribution({data,sample}:{data:BaselineData;sample:string}) {
  const ref=useRef<HTMLDivElement>(null);const {width}=useDimensions(ref);const lg=useLegend()
  const row=data.contributions.find(r=>r.cohort==='full_pool'&&r.sample_id===sample) as BaselineContribution|undefined
  if(!row)return null
  const entries=BANDS.map(b=>({name:bandName(b),spectra:row[`spectra_${b}`],coefficients:row[`coefficients_${b}`]}))
  const values=entries.flatMap(r=>[...(lg.show('Corrected spectra')?[r.spectra]:[]),...(lg.show('EC coefficients')?[r.coefficients]:[])]);const bound=Math.max(...values.map(Math.abs),1)
  const w=Math.max(300,width),iw=w-135,ih=240
  const x=d3.scaleBand().domain(entries.map(r=>r.name)).range([0,iw]).padding(.22)
  const y=d3.scaleLinear().domain([-bound*1.08,bound*1.08]).nice().range([ih,0])
  return <ChartFrame title="What moves this prediction?" subtitle="An exact algebraic split of VIBES minus AIRSpec EC prediction for the selected inspected case. Terms depend on both frozen models.">
    <ExplainOnly><Note>{sample}. Each interval has a change from corrected spectra and a change from fitted EC coefficients. A separate intercept and centering term contributes {f(row.intercept_and_centering_delta,3)} µg/filter.</Note></ExplainOnly>
    <div ref={ref} className="chart-wrap"><svg width={w} height={325} role="img" aria-label="Prediction difference decomposed by spectral region and model coefficient"><g transform="translate(95,20)"><YAxis scale={y} x={0} gridWidth={iw} label="Δ prediction (µg/filter)"/><line x2={iw} y1={y(0)} y2={y(0)} stroke={INK.text}/>{entries.flatMap(e=>[['spectra',COLOR.VIBES,'Corrected spectra'],['coefficients',COLOR.AIRSpec,'EC coefficients']].map(([key,color,label],i)=>{
      const v=e[key as 'spectra'|'coefficients'];const xx=x(e.name)!+i*x.bandwidth()/2
      if(!lg.show(label))return null
      return <rect key={e.name+key} x={xx} y={Math.min(y(v),y(0))} width={x.bandwidth()/2} height={Math.abs(y(v)-y(0))} fill={color} opacity={lg.dim(label)}><title>{`${e.name} ${key}: ${f(v,3)} µg/filter`}</title></rect>
    }))}<XAxis scale={x} y={ih} label="Wavenumber (cm⁻¹)"/></g><Legend width={w} y={315} items={[{label:'Corrected spectra',color:COLOR.VIBES},{label:'EC coefficients',color:COLOR.AIRSpec}]} lg={lg}/></svg></div>
    <ExplainOnly><Note>All terms plus rounding sum to {f(row.prediction_delta,3)} µg/filter. The symmetric split explains saved model arithmetic; it does not identify a chemical cause. The 12 inspection cases were selected after reviewing errors.</Note></ExplainOnly>
  </ChartFrame>
}

function SiteBandPattern({rows,cohort}:{rows:BaselinePair[];cohort:BaselineCohort}) {
  const ref=useRef<HTMLDivElement>(null);const {width}=useDimensions(ref)
  const tip=useTooltip(ref)
  const sites=Array.from(new Set(rows.map(r=>r.Site))).sort()
  const bands=['Q1','Q2','Q3','Q4']
  const cells=sites.flatMap(site=>bands.map(band=>{
    const group=rows.filter(r=>r.Site===site&&r.loading_band===band)
    return {site,band,n:group.length,delta:d3.mean(group,r=>Math.abs(r.VIBES-r.y)-Math.abs(r.AIRSpec-r.y))??null}
  }))
  const bound=d3.quantile(cells.filter(c=>c.n>=5).map(c=>Math.abs(c.delta!)).sort(d3.ascending),.9)||1
  const w=Math.max(300,width),iw=w-115,ih=sites.length*22
  const x=d3.scaleBand().domain(bands).range([0,iw]).padding(.07)
  const y=d3.scaleBand().domain(sites).range([0,ih]).padding(.06)
  const color=d3.scaleDiverging([-bound,0,bound],d3.interpolatePuOr).clamp(true)
  return <ChartFrame title="Site and loading pattern" subtitle="Mean VIBES minus AIRSpec absolute EC error in the selected frozen test population. This screens for heterogeneity; it is not a trained model-routing rule.">
    <ExplainOnly><Note>{cohort==='full_pool'?'Full pool':'Restricted lowest-OC/EC'}, current display subset ({rows.length} paired physical filters). Colors show ΔMAE in µg/filter: purple favors VIBES, orange favors AIRSpec. Cells with fewer than five filters are gray; blank cells have none.</Note></ExplainOnly>
    <div ref={ref} className="chart-wrap"><svg width={w} height={ih+118} role="img" aria-label="Site by training loading quartile heatmap of paired absolute error difference"><g transform="translate(77,25)" fontFamily={FONT.family}>{cells.map(c=>c.n?<rect key={c.site+c.band} x={x(c.band)} y={y(c.site)} width={x.bandwidth()} height={y.bandwidth()} fill={c.n<5?INK.grid:color(c.delta!)} stroke="white" onMouseEnter={e=>tip.show(e,[`${c.site} / ${c.band}`,`${c.n} physical filters`,`ΔMAE ${f(c.delta,3)} µg/filter`,c.n<5?'Shown gray: fewer than five filters':''])} onMouseLeave={tip.hide}><title>{`${c.site} ${c.band}, n=${c.n}, ΔMAE=${f(c.delta,3)}`}</title></rect>:null)}{sites.map(s=><text key={s} x={-8} y={y(s)!+y.bandwidth()/2+4} textAnchor="end" fontSize={10} fill={INK.text}>{s}</text>)}<XAxis scale={x} y={ih} label="Training-defined EC loading quartile"/></g><ColorKey x={77} y={ih+92} bound={bound} color={color}/></svg>{tip.node}</div>
    <ExplainOnly><Note>Color saturates beyond ±{f(bound,2)} µg/filter, the 90th percentile of displayed cell magnitudes; hover gives exact values and counts. The quartiles use measured TOR EC, so they cannot route an Addis prediction without independent EC. The separate CLI notebook evaluates an exploratory spectrum-only router; this heatmap is not that model.</Note></ExplainOnly>
  </ChartFrame>
}

function BlankSpread({data}:{data:BaselineData}) {
  const ref=useRef<HTMLDivElement>(null);const {width}=useDimensions(ref)
  const [source,setSource]=useState('ETAD')
  const points=data.blank_details.filter(r=>r.source===source.toLowerCase())
  const hl=useHighlight()
  // one blank, every correction: the drawer shows all methods side by side, not just the dot clicked
  const openBlank=(id:string)=>hl.openRecord({id,source:'the held-out blank spread',site:source,fields:data.blank_details.filter(r=>r.sample_id===id).map(r=>[`${r.method} residual RMS (10⁻⁴ absorbance)`,r.rms_from_zero*1e4]),note:'A field blank: residual after baseline correction, where smaller is closer to zero absorbance. It tests background removal, not EC accuracy.'})
  const w=Math.max(300,width),iw=w-105,ih=215
  const x=d3.scaleBand().domain(METHODS).range([0,iw]).padding(.45)
  const ymax=d3.max(points,p=>p.rms_from_zero*1e4)||1
  const y=d3.scaleLinear().domain([0,ymax*1.12]).nice().range([ih,0])
  return <ChartFrame title="Held-out blank spread" controls={<Select label="Blank source" value={source} options={['ETAD','IMPROVE']} onChange={setSource}/>} subtitle="Every paired held-out blank, beyond the medians shown above.">
    <Note>{source}: {new Set(points.map(r=>r.sample_id)).size} physical field blanks. Residual RMS after correction; smaller is closer to zero absorbance.</Note>
    <div ref={ref} className="chart-wrap"><svg width={w} height={300} role="img" aria-label="Individual held-out blank residual RMS by correction method"><g transform="translate(75,20)"><YAxis scale={y} x={0} gridWidth={iw} label="RMS (10⁻⁴ absorbance)"/>{points.map((p,i)=><circle key={p.sample_id+p.method} cx={x(p.method)!+x.bandwidth()/2+((i%9)-4)*2.4} cy={y(p.rms_from_zero*1e4)} r={source==='ETAD'?4:2.7} fill={COLOR[p.method]} fillOpacity={.55} style={{cursor:'pointer'}} onClick={()=>openBlank(p.sample_id)}><title>{`${p.sample_id} ${p.method}: ${f(p.rms_from_zero,7)} absorbance`}</title></circle>)}<XAxis scale={x} y={ih} label="Correction method"/></g></svg></div>
    <Note>Field blanks can carry artifacts or residual material. This assesses background removal, not ambient EC accuracy. The nine ETAD blanks are independent parents; synthetic injections reuse parents at multiple amplitudes.</Note>
  </ChartFrame>
}

/** What an in-SVG legend needs from `useLegend`: click hides a series, hover or focus dims the others. */
export type SvgLegendState={hidden:Set<string>;hover:string|null;toggle:(label:string)=>void;setHover:(label:string|null)=>void}
/** Props for one reactive in-SVG legend entry; hidden entries fade to .35, the others to .3 while one is hovered. */
export function svgLegendItem(lg:SvgLegendState,label:string,hitWidth:number) {
  const off=lg.hidden.has(label)
  return {role:'button',tabIndex:0,'aria-pressed':!off,'aria-label':`${label}: ${off?'show':'hide'}`,style:{cursor:'pointer'},opacity:off?.35:lg.hover&&lg.hover!==label?.3:1,
    onClick:()=>lg.toggle(label),onKeyDown:(e:KeyboardEvent)=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();lg.toggle(label)}},
    onMouseEnter:()=>lg.setHover(label),onMouseLeave:()=>lg.setHover(null),onFocus:()=>lg.setHover(label),onBlur:()=>lg.setHover(null),
    hit:<rect x={-3} y={-13} width={Math.max(20,hitWidth-6)} height={17} fill="transparent"/>}
}

function Legend({width,y,items,lg}:{width:number;y:number;items:{label:string;color:string}[];lg:SvgLegendState}) {
  const step=Math.min(175,(width-50)/items.length)
  return <g transform={`translate(25,${y})`} fontFamily={FONT.family} fontSize={10}>{items.map((item,i)=>{const {hit,...p}=svgLegendItem(lg,item.label,step);return <g key={item.label} transform={`translate(${i*step},0)`} {...p}>{hit}<line x2={12} y1={-3} y2={-3} stroke={item.color} strokeWidth={3}/><text x={17} fill={INK.text}>{item.label}</text></g>})}</g>
}

/** The heatmap's key: VIBES-closer purple through zero to AIRSpec-closer orange, plus the gray n<5 swatch. */
function ColorKey({x,y,bound,color}:{x:number;y:number;bound:number;color:(v:number)=>string}) {
  const stops=d3.range(0,1.0001,.1).map(t=>-bound+2*bound*t)
  const kw=280
  return <g transform={`translate(${x},${y})`} fontFamily={FONT.family} fontSize={10} fill={INK.text}>
    <defs><linearGradient id="site-band-key">{stops.map((v,i)=><stop key={i} offset={`${i*10}%`} stopColor={color(v)}/>)}</linearGradient></defs>
    <text x={0} y={-4}>ΔMAE, µg/filter</text>
    <rect x={0} y={0} width={kw} height={9} fill="url(#site-band-key)"/>
    <text x={0} y={21} fill={INK.muted}>{`−${f(bound,2)} VIBES closer`}</text>
    <text x={kw} y={21} textAnchor="end" fill={INK.muted}>{`AIRSpec closer +${f(bound,2)}`}</text>
    <rect x={kw+24} y={0} width={14} height={9} fill={INK.grid}/><text x={kw+43} y={8} fill={INK.muted}>fewer than 5 filters</text>
  </g>
}
