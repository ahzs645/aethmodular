import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { FeatureCollection } from 'geojson'
import { ChartFrame, ExplainOnly, Note, Segmented, Select, Toggle } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis } from '@/components/Axes'
import { SweepPlot } from '@/components/SweepPlot'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { loadJson } from '@/lib/data'
import { useSearchParam } from '@/lib/url'
import { ADDIS_ALL, FONT, IMPROVE_GREY, INK } from '@/lib/theme'
import { f2, pct, signed } from '@/lib/format'
import { Forest, Scatter, num, placeLabels, rowsOf, runLine, sourceOf, useXp, type Cols, type XpBase } from './shared'

/**
 * Adama TOR study (experiments_20261022.ipynb §8): the five Adama quartz/PTFE pilot pairs against IMPROVE
 * (the TOR network the calibration learned from), the SPARTAN network and SPARTAN Addis; where a pilot would
 * need to sit to test the Addis problem; how precise ~10 pairs can be; and the draft pass contract for the
 * coming Addis city pairs, written before any new outcome is read.
 */

interface NetRow { population: string; site: string; name: string; filters: number; ec_ug: number; ec_ugm3: number; pm_ug: number | null; ocec: number | null; distance: number; volume: number; urban: boolean; lat: number | null; lon: number | null }
interface Contract { status: string; written: string; reference: string; 'frozen models': string[]; 'pass rules': { rule: string; test: string; limit: number | number[]; unit: string; note?: string }[]; 'low-signal handling': string; eligibility: string; 'representativeness check': string; precision: string; caveat: string }
interface TorData extends XpBase {
  contract: Contract
  network: NetRow[]
  points: Cols
  neighbour_sites: { site: string; name: string; Addis: number; Adama: number; 'IMPROVE share of filters': number; lat: number | null; lon: number | null }[]
  precision: { pairs: number; population: string; 'half width (relative bias)': number; 'pool size': number }[]
  pilot: { model: string; reference: string; pairs: string; n: number; 'mean relative bias': number; 'passes |bias| ≤ 20 %': boolean; 'share of pairs within tolerance': number }[]
  adama_pairs: Record<string, any>[]
  adama_latlon: [number, number]
  target: { axis: string; 'Addis p5': number; 'Addis p25': number; 'Addis median': number; 'Addis p75': number; 'Addis p95': number; 'Adama median': number; 'Adama inside Addis 25-75': number; 'Adama inside Addis 5-95': number; 'Adama percentile in Addis': number; 'Adama percentile in IMPROVE': number; 'Adama percentile in SPARTAN network': number; 'IMPROVE median': number; 'SPARTAN network median': number }[]
  volume: { target: string; 'EC µg/filter': number; "volume at Adama's median EC (m³)": number; 'volume at SPARTAN Addis median EC (m³)': number; 'Adama volume now (m³)': number; 'hours at the Adama flow rate': number }[]
  seasons: { season: string; 'Addis filters': number; 'published ÷ HIPS (median)': number; 'published ÷ HIPS p25': number; 'published ÷ HIPS p75': number; 'Deming slope': number; 'Deming intercept': number; 'Adama pairs': number }[]
}
const source = sourceOf('TorStudy')
const ADAMA = '#1b7837'
const SPARTAN = '#7c3aed'
/** "12th", "1st"; below 1 reads "below the 1st" */
const nth = (v: number | undefined) => { if (v == null || !Number.isFinite(v)) return '—'; if (v < 1) return 'below the 1st'; const n = Math.round(v), t = n % 100
  return `${n}${t >= 11 && t <= 13 ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] ?? 'th'}` }
const POP_COLOR: Record<string, string> = { 'IMPROVE (TOR)': IMPROVE_GREY, 'SPARTAN network': SPARTAN, 'SPARTAN Addis': ADDIS_ALL, 'Adama pilot (TOR)': ADAMA }
const POP_LABEL: Record<string, string> = { 'IMPROVE (TOR)': 'IMPROVE sites (TOR)', 'SPARTAN network': 'other SPARTAN sites (FTIR)', 'SPARTAN Addis': 'SPARTAN Addis (FTIR)', 'Adama pilot (TOR)': 'Adama pilot (TOR)' }
const AXIS_INFO: Record<string, { label: string; short: string; unit: string; log: boolean; about: string }> = {
  'EC µg/filter': { label: 'EC per filter', short: 'EC / filter', unit: 'µg', log: true, about: 'EC mass on the filter (concentration × sampled volume): what the spectrum actually sees' },
  'EC µg/m³': { label: 'EC concentration', short: 'EC / m³', unit: 'µg/m³', log: true, about: 'EC in the air sampled' },
  'PM2.5 µg/filter': { label: 'Weighed PM2.5 per filter', short: 'PM2.5 / filter', unit: 'µg', log: true, about: 'gravimetric mass on the filter (none for Adama)' },
  'OC/EC': { label: 'OC/EC', short: 'OC/EC', unit: 'ratio', log: true, about: 'organic over elemental carbon: composition, independent of loading' },
  'distance percentile': { label: 'Spectral distance', short: 'Distance', unit: 'percentile in IMPROVE', log: false, about: 'how unlike the calibration set the spectrum is (larger of T² and Q, percentile among held-out IMPROVE filters)' },
  'volume m³': { label: 'Sampled volume', short: 'Volume', unit: 'm³', log: true, about: 'air drawn through the filter (the Adama sampler runs at a low flow)' },
}

export function TorStudy() {
  const { data, state } = useXp<TorData>('tor', 'Adama TOR study')
  if (!data) return <>{state}</>
  return <>
    <Findings d={data} />
    <PilotPosition d={data} />
    <LoadingPlane d={data} />
    <SeasonPanel d={data} />
    <WhereItSits d={data} />
    <NetworkScatter d={data} />
    <NetworkMap d={data} />
    <ContractCard d={data} />
    <NearestSites d={data} />
    <Precision d={data} />
    <PilotPairs d={data} />
  </>
}

function Findings({ d }: { d: TorData }) {
  const site = (pop: string) => d.network.find((r) => r.population === pop)
  const addis = site('SPARTAN Addis'), adama = site('Adama pilot (TOR)')
  const T = Object.fromEntries(d.target.map((t) => [t.axis, t]))
  const hw = (n: number) => d.precision.find((p) => p.pairs === n && p.population.startsWith('Addis'))?.['half width (relative bias)']
  const topA = [...d.neighbour_sites].sort((a, b) => b.Addis - a.Addis).slice(0, 3)
  const med = d.volume.find((v) => v.target === 'Addis median')
  const worst = [...d.seasons].sort((a, b) => b['published ÷ HIPS (median)'] - a['published ÷ HIPS (median)'])
  const adamaSeason = d.seasons.find((s) => s['Adama pairs'] > 0)?.season
  return (
    <ExplainOnly>
      <ChartFrame id="xd-findings" title="Adama TOR study — short answers" exportable={false}>
        <ol style={{ margin: '4px 0 0 18px', lineHeight: 1.55 }}>
          <li><strong>Lighter than SPARTAN Addis.</strong> Median EC per filter {f2(adama?.ec_ug)} µg for Adama (TOR) against {f2(addis?.ec_ug)} µg at SPARTAN Addis (FTIR): Adama sits at Addis's {nth(T['EC µg/filter']?.['Adama percentile in Addis'])} percentile and IMPROVE's {nth(T['EC µg/filter']?.['Adama percentile in IMPROVE'])}. Most of that is volume: {f2(adama?.volume)} m³ per filter against {f2(addis?.volume)}.</li>
          <li><strong>A different aerosol.</strong> Adama's TOR OC/EC is {f2(T['OC/EC']?.['Adama median'])} against {f2(T['OC/EC']?.['Addis median'])} at SPARTAN Addis (FTIR) and {f2(T['OC/EC']?.['IMPROVE median'])} in IMPROVE (TOR): above every SPARTAN Addis filter ({nth(T['OC/EC']?.['Adama percentile in Addis'])} percentile).</li>
          <li><strong>Similar spectral distance.</strong> Median distance percentile {Math.round(adama?.distance ?? NaN)} for Adama and {Math.round(addis?.distance ?? NaN)} for SPARTAN Addis, so similar novelty does not mean similar conditions.</li>
          <li><strong>One season only.</strong> All five pairs are from {adamaSeason}. At SPARTAN Addis the seasons behave differently (published FTIR ÷ HIPS {d.seasons.map((s) => `${s.season} ${f2(s['published ÷ HIPS (median)'])}`).join(', ')}; highest in {worst[0].season}), so a one-season pilot cannot speak for the others.</li>
          <li><strong>Where it would need to be:</strong> at Adama's EC concentration, reaching SPARTAN Addis's median EC per filter takes about {Math.round(med?.["volume at Adama's median EC (m³)"] ?? NaN)} m³, {f2((med?.["volume at Adama's median EC (m³)"] ?? NaN) / (med?.['Adama volume now (m³)'] ?? NaN))}× today's volume; plus Dry or Belg sampling and lower OC/EC.</li>
          <li><strong>Precision:</strong> with 10 pairs a mean bias on Addis-like filters is known to about ±{pct(hw(10))} (±{pct(hw(36))} with 36). Addis's nearest TOR sites: {topA.map((s) => `${s.name} (${pct(s.Addis)})`).join(', ')}.</li>
        </ol>
        <Note>{runLine(d)}. The Adama pairs are date candidates with unconfirmed IDs; EC for Adama is TOR (ECTR) on the quartz filter's own volume, for SPARTAN it is the published FTIR product (no TOR exists there).</Note>
      </ChartFrame>
    </ExplainOnly>
  )
}

function PilotPosition({ d }: { d: TorData }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const pts = useMemo(() => rowsOf(d.points), [d])
  const ada = pts.filter((p) => p.population === 'Adama pilot (TOR)')
  const rows = d.target
  const labelW = 190, rowH = 46, statusW = 150, pw = Math.max(260, width - labelW - statusW - 30)
  const verdict = (t: TorData['target'][number]) => t['Adama inside Addis 25-75'] >= 0.6 ? { txt: 'inside Addis\'s middle half', c: INK.good }
    : t['Adama inside Addis 5-95'] >= 0.6 ? { txt: 'inside Addis\'s range, off-centre', c: '#b45309' } : { txt: 'outside Addis\'s range', c: INK.negative }
  return (
    <ChartFrame id="xd-position" source={source('PilotPosition')}
      title="Is the pilot where it needs to be? — Adama against SPARTAN Addis, axis by axis"
      subtitle="To test the Addis problem, a pilot must sit where SPARTAN Addis sits. Orange bar: SPARTAN Addis's filters (5–95 % light, 25–75 % dark, median tick); green diamonds: the five Adama pairs; grey and purple ticks: the IMPROVE and SPARTAN-network medians. The verdict on the right counts the Adama pairs inside Addis's middle half or its 5–95 % range.">
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={labelW + pw + statusW + 20} height={rows.length * rowH + 8} fontFamily={FONT.family}>
          {rows.map((t, i) => {
            const info = AXIS_INFO[t.axis], cy = i * rowH + 20
            const vals = ada.map((p) => num(p[t.axis])).filter((v) => Number.isFinite(v) && (!info.log || v > 0))
            const ext = [t['Addis p5'], t['Addis p95'], t['IMPROVE median'], t['SPARTAN network median'], ...vals].filter((v) => Number.isFinite(v) && (!info.log || v > 0))
            const x = (info.log ? d3.scaleLog() : d3.scaleLinear()).domain(d3.extent(ext) as [number, number]).range([0, pw]).nice()
            const v = verdict(t)
            return <g key={t.axis}>
              <text x={labelW - 10} y={cy} dy="0.32em" textAnchor="end" fontSize={11.5} fontWeight={600} fill={INK.text}>{info.label}</text>
              <text x={labelW - 10} y={cy + 13} dy="0.32em" textAnchor="end" fontSize={9.5} fill={INK.muted}>{info.unit}</text>
              <g transform={`translate(${labelW},0)`}>
                <rect x={x(t['Addis p5'])} y={cy - 8} width={x(t['Addis p95']) - x(t['Addis p5'])} height={16} fill={ADDIS_ALL} fillOpacity={0.25} />
                <rect x={x(t['Addis p25'])} y={cy - 8} width={x(t['Addis p75']) - x(t['Addis p25'])} height={16} fill={ADDIS_ALL} fillOpacity={0.6} />
                <line x1={x(t['Addis median'])} x2={x(t['Addis median'])} y1={cy - 10} y2={cy + 10} stroke="#7a5c00" strokeWidth={2} />
                {Number.isFinite(t['IMPROVE median']) && <line x1={x(t['IMPROVE median'])} x2={x(t['IMPROVE median'])} y1={cy - 13} y2={cy + 13} stroke={IMPROVE_GREY} strokeWidth={2} strokeDasharray="2 2" />}
                {Number.isFinite(t['SPARTAN network median']) && <line x1={x(t['SPARTAN network median'])} x2={x(t['SPARTAN network median'])} y1={cy - 13} y2={cy + 13} stroke={SPARTAN} strokeWidth={2} strokeDasharray="2 2" />}
                {vals.map((val, j) => <path key={j} d={d3.symbol(d3.symbolDiamond, 80)() ?? ''} transform={`translate(${x(val)},${cy})`} fill={ADAMA} stroke="#fff" strokeWidth={1}
                  onMouseEnter={(e) => tip.show(e, [`Adama pair ${j + 1}`, `${info.label}: ${f2(val)} ${info.unit}`, `Addis 25–75 %: ${f2(t['Addis p25'])}–${f2(t['Addis p75'])}`])} onMouseLeave={tip.hide} />)}
                <XAxis scale={x} y={cy + 12} tickCount={5} format={(tv) => f2(+tv)} />
              </g>
              <text x={labelW + pw + 14} y={cy} dy="0.32em" fontSize={10.5} fontWeight={600} fill={v.c}>{v.txt}</text>
              <text x={labelW + pw + 14} y={cy + 13} dy="0.32em" fontSize={9.5} fill={INK.muted}>Adama at Addis's {nth(t['Adama percentile in Addis'])} pct</text>
            </g>
          })}
        </svg>
        {tip.node}
      </div>
      <Legend items={[{ label: 'SPARTAN Addis filters (5–95 %, 25–75 %)', color: ADDIS_ALL, shape: 'band' }, { label: 'Adama pilot pairs', color: ADAMA }, { label: 'IMPROVE median', color: IMPROVE_GREY, shape: 'dashed' }, { label: 'SPARTAN-network median', color: SPARTAN, shape: 'dashed' }]} />
      <Note>Season is the axis this chart cannot show: see "Season" below. Adama EC is TOR, SPARTAN EC is the published FTIR product, so EC and OC/EC compare two methods; spectral distance compares like with like.</Note>
    </ChartFrame>
  )
}

function LoadingPlane({ d }: { d: TorData }) {
  const pts = useMemo(() => rowsOf(d.points), [d])
  const lg = useLegend()
  const [improveDots, setImproveDots] = useState(false)
  const pops = ['IMPROVE (TOR)', 'SPARTAN network', 'SPARTAN Addis', 'Adama pilot (TOR)']
  const rng = d3.randomLcg(7)
  const sample = (rows: Record<string, any>[], n: number) => rows.length <= n ? rows : rows.filter(() => rng() < n / rows.length)
  const shown = pops.filter((p) => lg.show(p) && (p !== 'IMPROVE (TOR)' || improveDots))
  const plot = shown.flatMap((pop) => sample(pts.filter((p) => p.population === pop), pop === 'Adama pilot (TOR)' ? 10 : 1500).map((p, i) => ({
    x: num(p['EC µg/m³']), y: num(p['volume m³']), color: POP_COLOR[pop], r: pop === 'Adama pilot (TOR)' ? 7 : pop === 'SPARTAN Addis' ? 3 : 2, key: `${pop}${i}`,
    tip: [`${POP_LABEL[pop]} · ${p.site} ${p.id}`, `EC ${f2(p['EC µg/m³'])} µg/m³ × ${f2(p['volume m³'])} m³ = ${f2(p['EC µg/filter'])} µg per filter`] })))
  const T = d.target.find((t) => t.axis === 'EC µg/filter')!
  const med = d.volume.find((v) => v.target === 'Addis median')!
  const adama = d.network.find((r) => r.population === 'Adama pilot (TOR)')!
  const xD: [number, number] = [0.02, 50], yD: [number, number] = [0.5, 60]
  return (
    <ChartFrame id="xd-plane" source={source('LoadingPlane')}
      title="Loading = concentration × volume — why the pilot filters are light, and what would fix it"
      subtitle="Each filter by EC concentration (across) and sampled volume (up); grey diagonals are equal EC per filter. The orange band is SPARTAN Addis's middle half of EC per filter: a pilot filter must land in it to load the spectrum as SPARTAN Addis does. The arrow shows how much more air the Adama sampler would need at Adama's own EC concentration."
      controls={<Toggle label="IMPROVE filters" checked={improveDots} onChange={setImproveDots} title="Draw a sample of IMPROVE filters (EC from TOR)" />}>
      <Scatter points={plot} xLabel="EC concentration (µg/m³)" yLabel="sampled volume (m³)" xLog yLog height={440} xDomain={xD} yDomain={yD}
        under={({ x, y }) => {
          // the drawn (niced) axis ranges, so the diagonals run corner to corner and their labels spread out
          const [x0, x1] = x.domain(), [y0, y1] = y.domain()
          const line = (ug: number) => { const a = Math.max(x0, ug / y1), b = Math.min(x1, ug / y0); return { x1: x(a), y1: y(ug / a), x2: x(b), y2: y(ug / b) } }
          const band = (lo: number, hi: number) => { const xs = d3.range(0, 61).map((i) => x0 * (x1 / x0) ** (i / 60))
            const up = xs.map((cx) => [x(cx), y(Math.min(y1, Math.max(y0, hi / cx)))]), dn = xs.map((cx) => [x(cx), y(Math.min(y1, Math.max(y0, lo / cx)))]).reverse()
            return `M${[...up, ...dn].map((pt) => pt.join(',')).join('L')}Z` }
          return <g>
            <path d={band(T['Addis p25'], T['Addis p75'])} fill={ADDIS_ALL} fillOpacity={0.14} />
            {[1, 3, 10, 30, 100].map((ug) => { const l = line(ug); return <g key={ug}><line {...l} stroke={INK.identity} strokeDasharray="3 3" />
              {/* along its own diagonal, near the top end, so neighbouring labels never collide */}
              <text transform={`translate(${l.x1 + (l.x2 - l.x1) * 0.06},${l.y1 + (l.y2 - l.y1) * 0.06}) rotate(${(Math.atan2(l.y2 - l.y1, l.x2 - l.x1) * 180) / Math.PI})`} dy={-3}
                fontSize={9.5} fill={INK.muted} stroke="#fff" strokeWidth={3} paintOrder="stroke">{ug} µg/filter</text></g> })}
          </g>
        }}
        over={({ x, y }) => {
          const x0 = x(adama.ec_ugm3), y0 = y(adama.volume), y1 = y(Math.min(yD[1], med["volume at Adama's median EC (m³)"]))
          return <g pointerEvents="none">
            <defs><marker id="xd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill={ADAMA} /></marker></defs>
            <line x1={x0} y1={y0 - 8} x2={x0} y2={y1} stroke={ADAMA} strokeWidth={2} markerEnd="url(#xd-arrow)" />
            <text x={x0 + 6} y={(y0 + y1) / 2} fontSize={10.5} fontWeight={600} fill={ADAMA} stroke="#fff" strokeWidth={3} paintOrder="stroke">≈ {Math.round(med["volume at Adama's median EC (m³)"])} m³ for Addis's median loading</text>
          </g>
        }} />
      <Legend {...lg.props} items={pops.map((p) => ({ label: POP_LABEL[p], color: POP_COLOR[p] }))} />
      <table className="census-table" style={{ marginTop: 6 }}>
        <thead><tr><th>Target EC per filter</th><th>µg</th><th>Volume at Adama's EC</th><th>× today's Adama volume</th><th>Hours at the Adama flow</th><th>Volume at SPARTAN Addis's EC</th></tr></thead>
        <tbody>{d.volume.map((v) => <tr key={v.target}><td>{v.target}</td><td>{f2(v['EC µg/filter'])}</td><td>{f2(v["volume at Adama's median EC (m³)"])} m³</td>
          <td>{f2(v["volume at Adama's median EC (m³)"] / v['Adama volume now (m³)'])}×</td><td>{Math.round(v['hours at the Adama flow rate'])} h</td><td>{f2(v['volume at SPARTAN Addis median EC (m³)'])} m³</td></tr>)}</tbody>
      </table>
      <Note>At SPARTAN Addis's own median EC concentration (if the Addis city site resembles it), the volumes in the last column apply. Longer sampling at the same flow reaches the same loading; a 24 h sample at the Adama flow is about {f2(adama.volume)} m³.</Note>
    </ChartFrame>
  )
}

function SeasonPanel({ d }: { d: TorData }) {
  const rows = d.seasons.map((s) => ({
    key: s.season, label: `${s.season}${s['Adama pairs'] ? `  ← all ${s['Adama pairs']} Adama pairs` : ''}`, color: s['Adama pairs'] ? ADAMA : ADDIS_ALL, bold: s['Adama pairs'] > 0,
    values: { ratio: { v: s['published ÷ HIPS (median)'], lo: s['published ÷ HIPS p25'], hi: s['published ÷ HIPS p75'] }, slope: { v: s['Deming slope'] }, intercept: { v: s['Deming intercept'] } },
    tip: [s.season, `${s['Addis filters']} Addis filters`, `published FTIR ÷ HIPS: median ${f2(s['published ÷ HIPS (median)'])} (25–75 % ${f2(s['published ÷ HIPS p25'])}–${f2(s['published ÷ HIPS p75'])})`, `Deming slope ${f2(s['Deming slope'])}, intercept ${signed(s['Deming intercept'])}`, `${s['Adama pairs']} Adama pairs`],
  }))
  return (
    <ChartFrame id="xd-season" source={source('SeasonPanel')}
      title="Season — when the Addis disagreement happens, and when the pilot was sampled"
      subtitle="SPARTAN Addis by season (site calendar, February in Belg): published FTIR EC ÷ HIPS BC (median with 25–75 %), and the Deming fit against HIPS. The seasons differ, and the Adama pilot was sampled entirely in one of them (green). Pairs meant to test the calibration at Addis need to cover every season, or at least the one the question is about.">
      <Forest rows={rows} labelWidth={230} panels={[{ key: 'ratio', label: 'published FTIR ÷ HIPS BC', ref: 1, fmt: f2 }, { key: 'slope', label: 'Deming slope vs HIPS', ref: 1, fmt: f2 }, { key: 'intercept', label: 'Deming intercept (µg/m³)', ref: 0, fmt: (v) => signed(v) }]} />
    </ChartFrame>
  )
}

function ContractCard({ d }: { d: TorData }) {
  const c = d.contract
  const lim = (r: Contract['pass rules'][number]) => Array.isArray(r.limit) ? `${r.limit[0]} – ${r.limit[1]}` : r.unit === 'fraction of TOR' ? `≤ ${Math.round((r.limit as number) * 100)} %` : r.unit === 'share of pairs' ? `≥ ${Math.round((r.limit as number) * 100)} % of pairs` : `≤ ${r.limit} ${r.unit}`
  const pilotFor = (model: string) => d.pilot.find((p) => p.model === model && p.reference === 'ECTR' && p.pairs === 'three unflagged')
  const models = [...new Set(d.pilot.map((p) => p.model))]
  return (
    <ChartFrame id="xd-contract" source={source('ContractCard')} exportable={false}
      title="The pass contract — written before the Addis city pairs are read"
      subtitle="What agreement with TOR would count as evidence that the frozen calibration works at Addis. Values are proposals to agree with Ann and Satoshi; once agreed, the file is kept as the record and the pairs are scored against it unchanged.">
      <div style={{ display: 'inline-block', padding: '2px 8px', borderRadius: 4, background: '#fff4e5', color: '#b45309', fontSize: 11, fontWeight: 700, marginBottom: 8 }}>{c.status}</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 14 }}>
        <div>
          <div className="row-label">Reference</div><p style={{ margin: '2px 0 10px', fontSize: 12.5 }}>{c.reference}</p>
          <div className="row-label">Frozen models</div>
          <ol style={{ margin: '2px 0 10px 18px', fontSize: 12.5 }}>{c['frozen models'].map((m) => <li key={m}>{m}</li>)}</ol>
          <div className="row-label">Eligibility (from metadata, before TOR is opened)</div><p style={{ margin: '2px 0 10px', fontSize: 12.5 }}>{c.eligibility}</p>
          <div className="row-label">Low-signal handling</div><p style={{ margin: '2px 0 10px', fontSize: 12.5 }}>{c['low-signal handling']}</p>
        </div>
        <div>
          <table className="census-table">
            <thead><tr><th>Pass rule</th><th>Test</th><th>Limit</th>{models.map((m) => <th key={m} style={{ fontSize: 10 }}>Adama pilot<br />{m}</th>)}</tr></thead>
            <tbody>{c['pass rules'].map((r) => <tr key={r.rule}>
              <td><strong>{r.rule}</strong>{r.note && <div style={{ fontSize: 10.5, color: INK.muted }}>{r.note}</div>}</td><td style={{ fontSize: 11.5 }}>{r.test}</td><td>{lim(r)}</td>
              {models.map((m) => { const p = pilotFor(m)
                const cellTxt = !p ? '—' : r.rule === 'mean bias' ? `${signed(p['mean relative bias'] * 100, 0)} % ${p['passes |bias| ≤ 20 %'] ? '✓' : '✗'}`
                  : r.rule === 'per-pair agreement' ? `${pct(p['share of pairs within tolerance'])} ${p['share of pairs within tolerance'] >= 0.8 ? '✓' : '✗'}` : 'too few pairs'
                const good = cellTxt.includes('✓'), bad = cellTxt.includes('✗')
                return <td key={m} style={{ fontSize: 11.5, color: good ? INK.good : bad ? INK.negative : INK.muted }}>{cellTxt}</td> })}
            </tr>)}</tbody>
          </table>
          <p style={{ fontSize: 11.5, color: INK.muted, margin: '6px 0' }}>Pilot columns: the three unflagged Adama pairs against ECTR. {c.caveat}.</p>
          <div className="row-label">Before scoring</div><p style={{ margin: '2px 0 0', fontSize: 12.5 }}>{c['representativeness check']}. {c.precision}.</p>
        </div>
      </div>
    </ChartFrame>
  )
}

function NetworkMap({ d }: { d: TorData }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const [world, setWorld] = useState<FeatureCollection | null>(null)
  useEffect(() => { let live = true; loadJson<FeatureCollection>('world.geojson').then((w) => { if (live) setWorld(w) }).catch(() => {}); return () => { live = false } }, [])
  const [of, setOf] = useSearchParam<'Addis' | 'Adama'>('xd-map.of', 'Addis', ['Addis', 'Adama'])
  const [region, setRegion] = useSearchParam<'World' | 'United States'>('xd-map.region', 'World', ['World', 'United States'])
  const [arcs, setArcs] = useState(true)
  const sites = d.neighbour_sites.filter((s) => s.lat != null && s.lon != null)
  const net = Object.fromEntries(d.network.map((r) => [r.site, r]))
  const addis = net.ETAD, adama = net.ADAMA
  const origin: [number, number] = of === 'Addis' ? [addis.lon!, addis.lat!] : [d.adama_latlon[1], d.adama_latlon[0]]
  const h = Math.max(320, Math.min(520, width * 0.5))
  const us: GeoJSON.Feature = { type: 'Feature', properties: {}, geometry: { type: 'MultiPoint', coordinates: sites.map((s) => [s.lon!, s.lat!]) } }
  const projection = region === 'World'
    ? d3.geoNaturalEarth1().fitExtent([[10, 10], [Math.max(60, width - 10), h - 10]], { type: 'Sphere' })
    : d3.geoAlbersUsa().fitExtent([[20, 20], [Math.max(80, width - 20), h - 20]], us)
  const path = d3.geoPath(projection)
  const maxShare = d3.max(sites, (s) => s[of]) ?? 0.01
  const colour = d3.scaleSequential((t: number) => d3.interpolateYlOrRd(0.15 + 0.85 * t)).domain([0, maxShare])
  const rN = d3.scaleSqrt().domain([0, d3.max(d.network, (r) => r.filters) ?? 1]).range([1.5, 9])
  const top = [...sites].sort((a, b) => b[of] - a[of]).slice(0, 12)
  const proj = (lon: number, lat: number) => projection([lon, lat])
  return (
    <ChartFrame id="xd-map" source={source('NetworkMap')}
      title="The TOR network the calibration learned from — and where Addis borrows from"
      subtitle={`Every IMPROVE site in the calibration set (size = filters), coloured by its share of ${of}'s nearest spectral neighbours (the 500 most similar IMPROVE spectra per filter, averaged over ${of === 'Addis' ? 'the Addis evaluation filters' : 'the five Adama pilot filters'}). Arcs join ${of} to its twelve most-borrowed TOR sites. SPARTAN Addis and the Adama pilot are marked in Ethiopia.`}
      controls={<><Segmented label="Neighbours of" value={of} options={['Addis', 'Adama'] as const} onChange={setOf} /><Segmented label="Region" value={region} options={['World', 'United States'] as const} onChange={setRegion} />
        {region === 'World' && <Toggle label="Arcs" checked={arcs} onChange={setArcs} />}</>}>
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={width} height={h} fontFamily={FONT.family} style={{ display: 'block' }}>
          {region === 'World' && <path d={path({ type: 'Sphere' }) ?? ''} fill="#f3f7fb" stroke={INK.border} />}
          {world && <g>{world.features.map((f, i) => <path key={i} d={path(f) ?? ''} fill="#e9ecef" stroke="#fff" strokeWidth={0.5} />)}</g>}
          {region === 'World' && arcs && top.map((s) => <path key={`a${s.site}`} d={path({ type: 'LineString', coordinates: [origin, [s.lon!, s.lat!]] }) ?? ''}
            fill="none" stroke={colour(s[of])} strokeOpacity={0.75} strokeWidth={0.8 + 6 * s[of] / maxShare} />)}
          {[...sites].sort((a, b) => a[of] - b[of]).map((s) => { const p = proj(s.lon!, s.lat!); if (!p) return null
            const n = net[s.site]
            return <circle key={s.site} cx={p[0]} cy={p[1]} r={rN(n?.filters ?? 1)} fill={s[of] > 0 ? colour(s[of]) : '#fff'} stroke={s[of] > 0 ? '#7a2d0b' : IMPROVE_GREY} strokeWidth={0.6}
              onMouseEnter={(e) => tip.show(e, [`${s.name} (${s.site})${n?.urban ? ' · urban' : ''}`, `${n?.filters ?? '?'} calibration filters (${pct(s['IMPROVE share of filters'])} of IMPROVE)`,
                `share of Addis neighbours ${pct(s.Addis)} · of Adama ${pct(s.Adama)}`, `median TOR EC ${f2(n?.ec_ug)} µg/filter, ${f2(n?.ec_ugm3)} µg/m³`])} onMouseLeave={tip.hide} /> })}
          {region === 'World' && [[addis, 'SPARTAN Addis', ADDIS_ALL], [adama, 'Adama pilot', ADAMA]].map(([r, label, c]) => { const row = r as NetRow
            const p = proj(row.lon!, row.lat!); if (!p) return null
            return <g key={label as string} onMouseEnter={(e) => tip.show(e, [label as string, `${row.filters} filters · median EC ${f2(row.ec_ug)} µg/filter`, `median distance percentile ${Math.round(row.distance)}`])} onMouseLeave={tip.hide}>
              <path d={d3.symbol(label === 'Adama pilot' ? d3.symbolDiamond : d3.symbolStar, 160)() ?? ''} transform={`translate(${p[0]},${p[1]})`} fill={c as string} stroke={INK.text} strokeWidth={1} />
              <text x={p[0] + 9} y={p[1] + (label === 'Adama pilot' ? 12 : -6)} fontSize={10.5} fontWeight={700} fill={INK.text} stroke="#fff" strokeWidth={3} paintOrder="stroke">{label as string}</text>
            </g> })}
        </svg>
        {tip.node}
      </div>
      <div className="legend" style={{ justifyContent: 'center' }}>
        <ColorLegend scale={colour} label={`share of ${of}'s nearest neighbours`} format={(v) => pct(v)} />
        <Legend items={[{ label: 'SPARTAN Addis', color: ADDIS_ALL }, { label: 'Adama pilot (TOR)', color: ADAMA }, { label: 'IMPROVE site never chosen', color: IMPROVE_GREY, shape: 'ring' }]} />
      </div>
      <Note>Adama is placed at the pilot sampler's recorded position. Neighbours use the spline spectra and the 600–3600 cm⁻¹ similarity region without CO₂ and PTFE bands.</Note>
    </ChartFrame>
  )
}

const VARS: { key: string; label: string; log: boolean; fmt: (v: number) => string }[] = [
  { key: 'EC µg/filter', label: 'EC per filter (µg)', log: true, fmt: (v) => f2(v) },
  { key: 'EC µg/m³', label: 'EC concentration (µg/m³)', log: true, fmt: (v) => f2(v) },
  { key: 'OC/EC', label: 'OC/EC', log: true, fmt: (v) => f2(v) },
  { key: 'PM2.5 µg/filter', label: 'weighed PM2.5 per filter (µg)', log: true, fmt: (v) => String(Math.round(v)) },
  { key: 'distance percentile', label: 'spectral distance (percentile in IMPROVE)', log: false, fmt: (v) => String(Math.round(v)) },
  { key: 'volume m³', label: 'sampled volume (m³)', log: true, fmt: (v) => f2(v) },
]

function WhereItSits({ d }: { d: TorData }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const pts = useMemo(() => rowsOf(d.points), [d])
  const imp = pts.filter((p) => p.population === 'IMPROVE (TOR)'), add = pts.filter((p) => p.population === 'SPARTAN Addis'), ada = pts.filter((p) => p.population === 'Adama pilot (TOR)')
  const net = pts.filter((p) => p.population === 'SPARTAN network')
  const sites = d.network.filter((r) => r.population === 'IMPROVE (TOR)')
  const siteVal: Record<string, keyof NetRow> = { 'EC µg/filter': 'ec_ug', 'EC µg/m³': 'ec_ugm3', 'OC/EC': 'ocec', 'PM2.5 µg/filter': 'pm_ug', 'distance percentile': 'distance', 'volume m³': 'volume' }
  const labelW = 210, rowH = 74, pw = Math.max(300, width - labelW - 30)
  const jitter = (i: number) => ((Math.sin(i * 12.9898) * 43758.5453) % 1 + 1) % 1
  return (
    <ChartFrame id="xd-where" source={source('WhereItSits')}
      title="Every population on the same axes — IMPROVE, the SPARTAN network, SPARTAN Addis and Adama"
      subtitle="Grey band: IMPROVE calibration filters (5–95 % light, 25–75 % dark, median line; axes cover IMPROVE's 1st–99th percentiles), grey ticks each IMPROVE site's median. Purple band below it: the other SPARTAN sites' filters. Orange dots: SPARTAN Addis filters (EC from FTIR). Green diamonds: the five Adama pilot pairs (EC and OC from TOR). A pilot that is to test the calibration at Addis should overlap the orange, not only the grey.">
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={labelW + pw + 20} height={VARS.length * rowH + 10} fontFamily={FONT.family}>
          {VARS.map((v, vi) => {
            const q = (arr: number[], p: number) => d3.quantile(arr.slice().sort(d3.ascending), p)!
            const iv = imp.map((p) => num(p[v.key])).filter((z) => Number.isFinite(z) && (!v.log || z > 0))
            const nv = net.map((p) => num(p[v.key])).filter((z) => Number.isFinite(z) && (!v.log || z > 0))
            // IMPROVE's 1st–99th percentiles (its near-zero tail would squash a log axis) plus every Addis and Adama value
            const ext = [...add, ...ada].map((p) => num(p[v.key])).filter((x) => Number.isFinite(x) && (!v.log || x > 0))
            const all = [...(iv.length ? [q(iv, 0.01), q(iv, 0.99)] : []), ...ext]
            if (!all.length) return null
            const x = (v.log ? d3.scaleLog() : d3.scaleLinear()).domain(d3.extent(all) as [number, number]).range([0, pw]).nice().clamp(true)
            const top = vi * rowH + 6, mid = top + 22, midN = mid + 14
            return <g key={v.key}>
              <text x={labelW - 10} y={mid} dy="0.32em" textAnchor="end" fontSize={11} fill={INK.text}>{v.label}</text>
              <g transform={`translate(${labelW},0)`}>
                {iv.length > 0 && <>
                  <rect x={x(q(iv, 0.05))} y={mid - 9} width={x(q(iv, 0.95)) - x(q(iv, 0.05))} height={18} fill="#e5e7eb" />
                  <rect x={x(q(iv, 0.25))} y={mid - 9} width={x(q(iv, 0.75)) - x(q(iv, 0.25))} height={18} fill="#cbd2d9" />
                  <line x1={x(q(iv, 0.5))} x2={x(q(iv, 0.5))} y1={mid - 9} y2={mid + 9} stroke={INK.axis} strokeWidth={2} />
                </>}
                {nv.length > 0 && <>
                  <rect x={x(q(nv, 0.05))} y={midN - 4} width={x(q(nv, 0.95)) - x(q(nv, 0.05))} height={8} fill={SPARTAN} fillOpacity={0.18} />
                  <rect x={x(q(nv, 0.25))} y={midN - 4} width={x(q(nv, 0.75)) - x(q(nv, 0.25))} height={8} fill={SPARTAN} fillOpacity={0.45} />
                  <line x1={x(q(nv, 0.5))} x2={x(q(nv, 0.5))} y1={midN - 5} y2={midN + 5} stroke={SPARTAN} strokeWidth={2} />
                </>}
                {sites.map((s) => { const val = s[siteVal[v.key]] as number; return Number.isFinite(val) && (!v.log || val > 0) ? <line key={s.site} x1={x(val)} x2={x(val)} y1={mid + 11} y2={mid + 16} stroke={IMPROVE_GREY} strokeWidth={0.8} /> : null })}
                {add.map((p, i) => { const val = num(p[v.key]); return Number.isFinite(val) && (!v.log || val > 0) ? <circle key={i} cx={x(val)} cy={mid - 6 + jitter(i) * 12} r={2} fill={ADDIS_ALL} fillOpacity={0.55} /> : null })}
                {ada.map((p, i) => { const val = num(p[v.key]); return Number.isFinite(val) && (!v.log || val > 0)
                  ? <path key={i} d={d3.symbol(d3.symbolDiamond, 70)() ?? ''} transform={`translate(${x(val)},${mid})`} fill={ADAMA} stroke="#fff" strokeWidth={1}
                    onMouseEnter={(e) => tip.show(e, [`Adama ${p.id}`, `${v.label}: ${v.fmt(val)}`])} onMouseLeave={tip.hide} /> : null })}
                <XAxis scale={x} y={midN + 8} tickCount={6} format={(t) => v.fmt(+t)} />
              </g>
            </g>
          })}
        </svg>
        {tip.node}
      </div>
      <Legend items={[{ label: 'IMPROVE filters (band) and site medians (ticks)', color: '#cbd2d9', shape: 'square' }, { label: 'other SPARTAN sites (band)', color: SPARTAN, shape: 'square' }, { label: 'SPARTAN Addis filters', color: ADDIS_ALL }, { label: 'Adama pilot pairs', color: ADAMA }]} />
      <Note>Adama has no weighed PM2.5 in the pair records. EC per filter = concentration × the filter's own sampled volume; SPARTAN Addis EC is the published FTIR value (no TOR exists there).</Note>
    </ChartFrame>
  )
}

const SITE_AXES = ['EC µg/filter', 'EC µg/m³', 'OC/EC', 'PM2.5 µg/filter', 'distance percentile', 'volume m³'] as const
const SITE_KEY: Record<string, keyof NetRow> = { 'EC µg/filter': 'ec_ug', 'EC µg/m³': 'ec_ugm3', 'OC/EC': 'ocec', 'PM2.5 µg/filter': 'pm_ug', 'distance percentile': 'distance', 'volume m³': 'volume' }

function NetworkScatter({ d }: { d: TorData }) {
  const [xk, setXk] = useSearchParam<typeof SITE_AXES[number]>('xd-net.x', 'EC µg/filter', SITE_AXES)
  const [yk, setYk] = useSearchParam<typeof SITE_AXES[number]>('xd-net.y', 'distance percentile', SITE_AXES)
  const [labels, setLabels] = useState(true)
  const lg = useLegend()
  const order = ['IMPROVE (TOR)', 'SPARTAN network', 'SPARTAN Addis', 'Adama pilot (TOR)']
  const X = AXIS_INFO[xk], Y = AXIS_INFO[yk]
  const rows = [...d.network].sort((a, b) => order.indexOf(a.population) - order.indexOf(b.population)).filter((r) => lg.show(r.population))
  const pts = rows.map((r) => {
    const big = r.population === 'SPARTAN Addis' || r.population === 'Adama pilot (TOR)'
    return { x: r[SITE_KEY[xk]] as number, y: r[SITE_KEY[yk]] as number, key: `${r.population}${r.site}`, color: POP_COLOR[r.population] ?? INK.accent,
      r: big ? 8 : r.population === 'IMPROVE (TOR)' ? 2.5 + Math.sqrt(r.filters) / 4 : 4.5, hollow: r.population === 'IMPROVE (TOR)' && r.urban,
      tip: [`${r.name} (${r.site})`, POP_LABEL[r.population], `${r.filters} filters`, `${X.label}: ${f2(r[SITE_KEY[xk]])} ${X.unit}`, `${Y.label}: ${f2(r[SITE_KEY[yk]])} ${Y.unit}`, ...(r.urban ? ['urban IMPROVE site'] : [])] }
  })
  const named = rows.filter((r) => r.population !== 'IMPROVE (TOR)')
  const opt = (k: string) => `${AXIS_INFO[k].label} (${AXIS_INFO[k].unit})`
  return (
    <ChartFrame id="xd-net" source={source('NetworkScatter')}
      title="Sites compared — every IMPROVE and SPARTAN site, SPARTAN Addis and Adama, on two chosen axes"
      subtitle="One point per site: its median over all its filters. Pick what goes across and up. Grey = IMPROVE calibration sites (size = filters; hollow = Ann's six urban sites), purple = other SPARTAN sites, orange = SPARTAN Addis, green = the Adama pilot."
      controls={<>
        <Select label="Across" value={xk} options={[...SITE_AXES]} onChange={(v) => setXk(v as typeof xk)} optionLabel={opt} />
        <Select label="Up" value={yk} options={[...SITE_AXES]} onChange={(v) => setYk(v as typeof yk)} optionLabel={opt} />
        <Toggle label="Name SPARTAN sites" checked={labels} onChange={setLabels} />
      </>}>
      <Scatter points={pts} xLabel={opt(xk)} yLabel={opt(yk)} xLog={X.log} yLog={Y.log} height={460} yDomain={yk === 'distance percentile' ? [0, 100] : undefined} xDomain={xk === 'distance percentile' ? [0, 100] : undefined}
        over={({ x, y, iw, ih }) => {
          if (!labels) return null
          // Addis and Adama first so they get the best spots; every label placed where it overlaps nothing
          const items = [...named].sort((a, b) => +(a.population === 'SPARTAN network') - +(b.population === 'SPARTAN network')).flatMap((r) => {
            const vx = r[SITE_KEY[xk]] as number, vy = r[SITE_KEY[yk]] as number
            if (!Number.isFinite(vx) || !Number.isFinite(vy) || (X.log && vx <= 0) || (Y.log && vy <= 0)) return []
            const big = r.population !== 'SPARTAN network'
            return [{ key: r.site, x: x(vx), y: y(vy), r: big ? 8 : 4.5, text: big ? (r.population === 'SPARTAN Addis' ? 'SPARTAN Addis' : 'Adama pilot') : r.site, bold: big, optional: !big }]
          })
          return <g pointerEvents="none">{placeLabels(items, iw, ih).map((l) => <g key={l.key}>
            {l.leader && <line x1={l.ax} y1={l.ay} x2={l.lx} y2={l.ly} stroke={INK.axis} strokeWidth={0.6} />}
            <text x={l.tx} y={l.ty} dy="0.32em" textAnchor={l.anchor} fontSize={l.bold ? 11 : 9} fontWeight={l.bold ? 700 : 400} fill={INK.text} stroke="#fff" strokeWidth={3} strokeLinejoin="round" paintOrder="stroke">{l.text}</text>
          </g>)}</g>
        }} />
      <Legend {...lg.props} items={order.map((p) => ({ label: POP_LABEL[p], color: POP_COLOR[p] }))} note="click to show or hide" />
      <div style={{ fontSize: 11.5, color: INK.muted, display: 'grid', gap: 2, marginTop: 4 }}>
        <div><strong style={{ color: INK.text }}>Across · {X.label}:</strong> {X.about}.</div>
        <div><strong style={{ color: INK.text }}>Up · {Y.label}:</strong> {Y.about}.</div>
        <div>EC and OC are TOR for IMPROVE and Adama, the published FTIR product for SPARTAN sites.</div>
      </div>
    </ChartFrame>
  )
}

function NearestSites({ d }: { d: TorData }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const rows = [...d.neighbour_sites].sort((a, b) => b.Addis - a.Addis).slice(0, 18)
  const labelW = 190, rowH = 20, pw = Math.max(260, width - labelW - 40)
  const x = d3.scaleLinear().domain([0, d3.max(rows, (r) => Math.max(r.Addis, r.Adama, r['IMPROVE share of filters'])) ?? 0.1]).nice().range([0, pw])
  return (
    <ChartFrame id="xd-nearest" source={source('NearestSites')}
      title="Nearest TOR sites — which IMPROVE sites Addis and Adama borrow their calibration from"
      subtitle="Share of the 500 nearest IMPROVE spectra per filter that come from each site, for Addis (orange) and Adama (green), beside the site's share of all calibration filters (grey tick). A site far right of its tick is over-represented among the neighbours.">
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={labelW + pw + 30} height={rows.length * rowH + 44} fontFamily={FONT.family}>
          {rows.map((r, i) => { const cy = i * rowH + rowH / 2
            return <g key={r.site} onMouseEnter={(e) => tip.show(e, [`${r.name} (${r.site})`, `Addis ${pct(r.Addis)} · Adama ${pct(r.Adama)}`, `${pct(r['IMPROVE share of filters'])} of calibration filters`])} onMouseLeave={tip.hide}>
              <rect x={0} y={cy - rowH / 2} width={labelW + pw} height={rowH} fill={i % 2 ? 'transparent' : '#f8f9fb'} />
              <text x={labelW - 8} y={cy} dy="0.32em" textAnchor="end" fontSize={10.5} fill={INK.text}>{r.name.length > 26 ? `${r.name.slice(0, 25)}…` : r.name}</text>
              <rect x={labelW} y={cy - 6} width={x(r.Addis)} height={5} fill={ADDIS_ALL} />
              <rect x={labelW} y={cy + 1} width={x(r.Adama)} height={5} fill={ADAMA} />
              <line x1={labelW + x(r['IMPROVE share of filters'])} x2={labelW + x(r['IMPROVE share of filters'])} y1={cy - 8} y2={cy + 8} stroke={INK.text} strokeWidth={1.5} />
            </g> })}
          <g transform={`translate(${labelW},${rows.length * rowH + 4})`}><XAxis scale={x} y={0} label="share of nearest neighbours" tickCount={5} labelDy={28} format={(v) => pct(+v)} /></g>
        </svg>
        {tip.node}
      </div>
      <Legend items={[{ label: 'Addis', color: ADDIS_ALL, shape: 'square' }, { label: 'Adama pilot', color: ADAMA, shape: 'square' }, { label: "site's share of calibration filters", color: INK.text, shape: 'line' }]} />
    </ChartFrame>
  )
}

function Precision({ d }: { d: TorData }) {
  const pops = [...new Set(d.precision.map((p) => p.population))]
  const color: Record<string, string> = { [pops[0]]: ADDIS_ALL, [pops[1]]: IMPROVE_GREY }
  const series = pops.map((p) => ({ id: p, label: p, color: color[p], points: d.precision.filter((r) => r.population === p).map((r) => ({ x: r.pairs, y: r['half width (relative bias)'], tip: [p, `${r.pairs} pairs: mean bias known to ± ${pct(r['half width (relative bias)'])}`, `resampled from ${r['pool size'].toLocaleString()} filters`] })) }))
  return (
    <ChartFrame id="xd-precision" source={source('Precision')}
      title="How many pairs? — how precisely a mean bias can be known"
      subtitle="Half-width of the 95 % interval of a mean relative bias (FTIR − TOR) ÷ TOR, resampling IMPROVE's own out-of-site errors: on Addis-like filters (heavy and spectrally distant, both above IMPROVE's 60th percentile) and on all filters. The dashed line is the draft ±20 % pass rule; markers: the 5 Adama pairs, the ~10 Addis city pairs, the freeze's 36-pair seasonal design.">
      <SweepPlot series={series} xLabel="number of pairs" yLabel="± half width of the 95 % interval" height={300} yFormat={(v) => pct(v)} xScale="log"
        yMarkers={[{ at: 0.2, label: '±20 % rule' }]} xMarkers={[{ at: 5, label: 'Adama 5' }, { at: 10, label: 'Addis city ~10' }, { at: 36, label: 'freeze 36' }]} xFormat={(v) => String(Math.round(+v))} />
      <Legend items={pops.map((p) => ({ label: p, color: color[p], shape: 'line' as const }))} />
      <Note>This is sampling precision only: it assumes the new pairs behave like IMPROVE's Addis-like filters. Sampler, flow and site differences add to it.</Note>
    </ChartFrame>
  )
}

function PilotPairs({ d }: { d: TorData }) {
  const models = ['frozen AIRSpec', 'frozen VIBES', 'this notebook']
  const col: Record<string, string> = { 'frozen AIRSpec': 'airspec_frozen_ugm3', 'frozen VIBES': 'vibes_frozen_ugm3' }
  const thisCol = Object.keys(d.adama_pairs[0] ?? {}).find((k) => k.startsWith('FTIR ')) ?? ''
  const [model, setModel] = useSearchParam<string>('xd-pilot.model', 'frozen AIRSpec', models)
  const [refk, setRefk] = useSearchParam<'ECTR' | 'ECTT'>('xd-pilot.ref', 'ECTR', ['ECTR', 'ECTT'])
  const key = model === 'this notebook' ? thisCol : col[model]
  const pts = d.adama_pairs.map((p) => ({ x: num(p[refk === 'ECTR' ? 'ec_tr_ugm3' : 'ec_tt_ugm3']), y: num(p[key]), color: ADAMA, r: 6, hollow: !!p.flag, key: String(p.ptfe_id),
    tip: [`Adama ${p.ptfe_id} / quartz ${p.quartz_id}`, String(p.date), `${refk} ${f2(p[refk === 'ECTR' ? 'ec_tr_ugm3' : 'ec_tt_ugm3'])} µg/m³ · FTIR ${f2(p[key])} µg/m³`, `distance percentile ${Math.round(num(p['distance percentile']))}`, p.flag ? `flag: ${p.flag}` : 'unflagged'] }))
  const hi = Math.max(1, ...pts.flatMap((p) => [p.x, p.y]).filter(Number.isFinite)) * 1.15
  return (
    <ChartFrame id="xd-pilot" source={source('PilotPairs')}
      title="The Adama pilot pairs — FTIR against TOR, with the draft tolerance"
      subtitle="The five Adama quartz/PTFE date-candidate pairs (hollow: flagged for a start offset or a volume mismatch). Shaded: the draft per-pair tolerance |FTIR − TOR| ≤ max(0.5 µg/m³, 30 % of TOR). A pilot, not the test: identities and sampler equivalence are unconfirmed."
      controls={<><Segmented label="FTIR" value={model} options={models} onChange={setModel} /><Segmented label="TOR" value={refk} options={['ECTR', 'ECTT'] as const} onChange={setRefk} /></>}>
      <Scatter points={pts} xLabel={`TOR EC, ${refk} (µg/m³)`} yLabel={`FTIR EC, ${model} (µg/m³)`} identity height={340} xDomain={[0, hi]} yDomain={[0, hi]}
        under={({ x, y }) => { const xs = d3.range(0, hi + 0.01, hi / 60)
          const area = d3.area<number>().x((v) => x(v)).y0((v) => y(Math.max(0, v - Math.max(0.5, 0.3 * v)))).y1((v) => y(v + Math.max(0.5, 0.3 * v)))(xs)
          return <path d={area ?? ''} fill={INK.good} fillOpacity={0.1} /> }} />
      <table className="census-table" style={{ marginTop: 8 }}>
        <thead><tr><th>Model</th><th>Reference</th><th>Pairs</th><th>Mean relative bias</th><th>|bias| ≤ 20 %</th><th>Within tolerance</th></tr></thead>
        <tbody>{d.pilot.filter((p) => p.reference === refk).map((p) => <tr key={`${p.model}${p.pairs}`}><td>{p.model}</td><td>{p.reference}</td><td>{p.pairs} ({p.n})</td>
          <td>{signed(p['mean relative bias'] * 100, 0)} %</td><td style={{ color: p['passes |bias| ≤ 20 %'] ? INK.good : INK.negative }}>{p['passes |bias| ≤ 20 %'] ? 'yes' : 'no'}</td><td>{pct(p['share of pairs within tolerance'])}</td></tr>)}</tbody>
      </table>
    </ChartFrame>
  )
}
