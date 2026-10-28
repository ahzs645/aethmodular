import { useMemo, useRef } from 'react'
import { useColourScheme } from '@/lib/colourSchemes'
import * as d3 from 'd3'
import { MapLineLayer, MapMarker, MarkerContent, type LngLatPoint } from '@pgmaps/geo-toolkit/map'
import { greatCircleArc } from '@pgmaps/geo-toolkit/calculations'
import { GeoMap, type GeoMapHandle } from '@/components/GeoMap'
import { ChartFrame, ExplainOnly, Note, Segmented, Toggle } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis } from '@/components/Axes'
import { SweepPlot } from '@/components/SweepPlot'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useSearchParam } from '@/lib/url'
import { ADDIS_ALL, FONT, IMPROVE_GREY, INK, themed } from '@/lib/theme'
import { dotFill } from '@/lib/highlight'
import { coef, coefSigned, f2, pct, signed } from '@/lib/format'
import { Crossplot } from '@/components/Crossplot'
import { RangeRows, type RangeRow, type RangeSeries } from '@/components/RangeRows'
import { AxisChoices } from '@/components/AxisChoices'
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
const PLANE_FIELDS = ['EC µg/m³', 'volume m³', 'EC µg/filter', 'PM2.5 µg/filter', 'OC/EC', 'distance percentile']
const nth = (v: number | undefined) => { if (v == null || !Number.isFinite(v)) return '—'; if (v < 1) return 'below the 1st'; const n = Math.round(v), t = n % 100
  return `${n}${t >= 11 && t <= 13 ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] ?? 'th'}` }
const POP_COLOR: Record<string, string> = { 'IMPROVE (TOR)': IMPROVE_GREY, 'SPARTAN network': SPARTAN, 'SPARTAN Addis': ADDIS_ALL, 'Adama pilot (TOR)': ADAMA }
const POP_LABEL: Record<string, string> = { 'IMPROVE (TOR)': 'IMPROVE sites (TOR)', 'SPARTAN network': 'other SPARTAN sites (FTIR)', 'SPARTAN Addis': 'SPARTAN Addis (FTIR)', 'Adama pilot (TOR)': 'Adama pilot (TOR)' }
const AXIS_INFO: Record<string, { label: string; short: string; unit: string; log: boolean }> = {
  'EC µg/filter': { label: 'EC per filter', short: 'EC / filter', unit: 'µg', log: true },
  'EC µg/m³': { label: 'EC concentration', short: 'EC / m³', unit: 'µg/m³', log: true },
  'PM2.5 µg/filter': { label: 'Weighed PM2.5 per filter', short: 'PM2.5 / filter', unit: 'µg', log: true },
  'OC/EC': { label: 'OC/EC', short: 'OC/EC', unit: 'ratio', log: true },
  'distance percentile': { label: 'Spectral distance', short: 'Distance', unit: 'percentile in IMPROVE', log: false },
  'volume m³': { label: 'Sampled volume', short: 'Volume', unit: 'm³', log: true },
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
  const pts = useMemo(() => rowsOf(d.points), [d])
  const ada = pts.filter((p) => p.population === 'Adama pilot (TOR)')
  const verdict = (t: TorData['target'][number]) => t['Adama inside Addis 25-75'] >= 0.6 ? { text: 'inside Addis\'s middle half', color: INK.good }
    : t['Adama inside Addis 5-95'] >= 0.6 ? { text: 'inside Addis\'s range, off-centre', color: INK.warn } : { text: 'outside Addis\'s range', color: INK.negative }
  const series: RangeSeries[] = [
    { key: 'addis', label: 'SPARTAN Addis filters (5–95 %, 25–75 %, median)', color: ADDIS_ALL, kind: 'band' },
    { key: 'adama', label: 'Adama pilot pairs', color: ADAMA, kind: 'points', symbol: 'diamond' },
    { key: 'improve', label: 'IMPROVE median', color: IMPROVE_GREY, kind: 'tick' },
    { key: 'network', label: 'SPARTAN-network median', color: SPARTAN, kind: 'tick' },
  ]
  const rows: RangeRow[] = d.target.map((t) => {
    const info = AXIS_INFO[t.axis]
    return {
      key: t.axis, label: info.label, unit: info.unit, log: info.log,
      band: { lo: t['Addis p5'], q1: t['Addis p25'], mid: t['Addis median'], q3: t['Addis p75'], hi: t['Addis p95'] },
      ticks: { improve: t['IMPROVE median'], network: t['SPARTAN network median'] },
      points: { adama: ada.map((p) => num(p[t.axis])) },
      verdict: { ...verdict(t), detail: `Adama at Addis's ${nth(t['Adama percentile in Addis'])} pct` },
    }
  })
  return (
    <ChartFrame id="xd-position" source={source('PilotPosition')} provenance={runLine(d)}
      title="Is the pilot where it needs to be? — Adama against SPARTAN Addis, axis by axis"
      subtitle="To test the Addis problem, a pilot must sit where SPARTAN Addis sits. Each row is one axis on its own scale: SPARTAN Addis's filters as a band (5–95 % light, 25–75 % dark, median tick), the five Adama pairs as diamonds, and the IMPROVE and SPARTAN-network medians as dashed ticks. The verdict on the right counts the Adama pairs inside Addis's middle half or its 5–95 % range. Click a legend item to hide it; each axis rescales."
      tip={<>Season is the one axis this chart cannot show: see "Season" below. Adama EC is TOR and SPARTAN EC is the published FTIR product, so the EC and OC/EC rows compare two methods; spectral distance compares like with like.</>}>
      <RangeRows rows={rows} series={series}
        pointTip={(r, _s, v, j) => {
          const t = d.target.find((x) => x.axis === r.key)!
          return [`Adama pair ${j + 1}`, `${r.label}: ${f2(v)} ${r.unit ?? ''}`, `Addis 25–75 %: ${f2(t['Addis p25'])}–${f2(t['Addis p75'])}`]
        }} />
    </ChartFrame>
  )
}

/**
 * What the loading plane measures. EC is the study's axis (TOR at IMPROVE and Adama, published FTIR at SPARTAN); HIPS BC
 * and weighed PM2.5 put the same filters on the plane by absorption and by mass. Neither is measured on the Adama pilot.
 */
type PlaneMeasure = 'EC' | 'HIPS' | 'PM2.5'
const HIPS_COL = 'HIPS BC µg/m³'
const PLANE_MEASURE: Record<PlaneMeasure, { label: string; short: string; conc: (p: Record<string, any>) => number; ug: (p: Record<string, any>) => number; xD: [number, number]; diagonals: number[] }> = {
  EC: { label: 'EC', short: 'EC', conc: (p) => num(p['EC µg/m³']), ug: (p) => num(p['EC µg/filter']), xD: [0.02, 50], diagonals: [1, 3, 10, 30, 100] },
  HIPS: { label: 'HIPS BC (MAC 10)', short: 'HIPS BC', conc: (p) => num(p[HIPS_COL]), ug: (p) => num(p[HIPS_COL]) * num(p['volume m³']), xD: [0.02, 50], diagonals: [1, 3, 10, 30, 100] },
  'PM2.5': { label: 'weighed PM2.5', short: 'PM2.5', conc: (p) => num(p['PM2.5 µg/filter']) / num(p['volume m³']), ug: (p) => num(p['PM2.5 µg/filter']), xD: [0.5, 500], diagonals: [10, 30, 100, 300, 1000, 3000] },
}

function LoadingPlane({ d }: { d: TorData }) {
  const pts = useMemo(() => rowsOf(d.points), [d])
  const lg = useLegend()
  const [arrowParam, setArrow] = useSearchParam<'on' | 'off'>('xd-plane.arrow', 'off', ['on', 'off'])
  const [level, setLevel] = useSearchParam<'filter' | 'site'>('xd-plane.level', 'filter', ['filter', 'site'])
  // HIPS only where the run exported it (runs before 2026-10-28 did not)
  const hipsPops = useMemo(() => new Set(pts.filter((p) => Number.isFinite(num(p[HIPS_COL]))).map((p) => p.population as string)), [pts])
  const measures: PlaneMeasure[] = hipsPops.size ? ['EC', 'HIPS', 'PM2.5'] : ['EC', 'PM2.5']
  const [measureParam, setMeasure] = useSearchParam<PlaneMeasure>('xd-plane.measure', 'EC', ['EC', 'HIPS', 'PM2.5'])
  const measure = measures.includes(measureParam) ? measureParam : 'EC'
  const M = PLANE_MEASURE[measure]
  const ec = measure === 'EC'
  const arrow = ec && arrowParam === 'on'
  const pops = ['IMPROVE (TOR)', 'SPARTAN network', 'SPARTAN Addis', 'Adama pilot (TOR)']
  const rng = d3.randomLcg(7)
  const sample = (rows: Record<string, any>[], n: number) => rows.length <= n ? rows : rows.filter(() => rng() < n / rows.length)
  // the legend names each population by its label (POP_LABEL): hide and dim by that, not the population key
  const shown = pops.filter((p) => lg.show(POP_LABEL[p]))
  const dim = (pop: string) => lg.dim(POP_LABEL[pop])
  const has = (p: Record<string, any>) => Number.isFinite(M.conc(p)) && Number.isFinite(M.ug(p))
  // where each population has the measure at all (Adama has no HIPS or weighed PM2.5)
  const missing = pops.filter((pop) => !pts.some((p) => p.population === pop && has(p)))
  const netOf = new Map(d.network.map((r) => [`${r.population}${r.site}`, r]))
  // one point per site: its median concentration and median volume (EC: the same medians as "Sites compared"; else from its filters here)
  const siteRows = ec ? d.network : d3.groups(pts.filter(has), (p) => p.population as string, (p) => p.site as string).flatMap(([population, sites]) => sites.map(([site, rows]) => {
    const r = netOf.get(`${population}${site}`)
    return { population, site, name: r?.name ?? site, urban: r?.urban ?? false, filters: rows.length, ec_ugm3: d3.median(rows, M.conc)!, volume: d3.median(rows, (p) => num(p['volume m³']))!, ec_ug: d3.median(rows, M.ug)! }
  }))
  const sitePlot = siteRows.filter((r) => shown.includes(r.population)).sort((a, b) => pops.indexOf(a.population) - pops.indexOf(b.population)).map((r) => ({
    x: r.ec_ugm3, y: r.volume, color: POP_COLOR[r.population], key: `${r.population}${r.site}`, hollow: r.population === 'IMPROVE (TOR)' && r.urban, dim: dim(r.population),
    r: r.population === 'SPARTAN Addis' || r.population === 'Adama pilot (TOR)' ? 8 : r.population === 'IMPROVE (TOR)' ? 2.5 + Math.sqrt(r.filters) / 4 : 4.5,
    tip: [`${r.name} (${r.site})`, POP_LABEL[r.population], `${r.filters} filters, medians:`, `${M.short} ${f2(r.ec_ugm3)} µg/m³ · volume ${f2(r.volume)} m³ · ${f2(r.ec_ug)} µg per filter`] }))
  const filterPlot = shown.flatMap((pop) => sample(pts.filter((p) => p.population === pop && (ec || has(p))), pop === 'Adama pilot (TOR)' ? 10 : 1500).map((p, i) => {
    const id = pop === 'IMPROVE (TOR)' ? `improve:${p.id}` : String(p.id)
    return { x: M.conc(p), y: num(p['volume m³']), color: POP_COLOR[pop], dim: dim(pop), r: pop === 'Adama pilot (TOR)' ? 7 : pop === 'SPARTAN Addis' ? 3 : 2, key: `${pop}${i}`, id,
      tip: [`${POP_LABEL[pop]} · ${p.site} ${p.id}`, `${M.short} ${f2(M.conc(p))} µg/m³ × ${f2(p['volume m³'])} m³ = ${f2(M.ug(p))} µg per filter`],
      rec: { id, source: 'Adama TOR study · loading plane', site: String(p.site), fields: [['population', POP_LABEL[pop]] as [string, string], ...[...PLANE_FIELDS, ...(hipsPops.size ? [HIPS_COL] : [])].map((k) => [k, p[k]] as [string, number | null])] } }
  }))
  const plot = level === 'site' ? sitePlot : filterPlot
  // the target band: SPARTAN Addis's middle half per filter (EC: the notebook's own percentiles)
  const T = d.target.find((t) => t.axis === 'EC µg/filter')!
  const addisUg = pts.filter((p) => p.population === 'SPARTAN Addis').map(M.ug).filter(Number.isFinite).sort(d3.ascending)
  const band: [number, number] = ec ? [T['Addis p25'], T['Addis p75']] : [d3.quantileSorted(addisUg, 0.25) ?? NaN, d3.quantileSorted(addisUg, 0.75) ?? NaN]
  const med = d.volume.find((v) => v.target === 'Addis median')!
  const adama = d.network.find((r) => r.population === 'Adama pilot (TOR)')!
  const xD = M.xD, yD: [number, number] = [0.5, 60]
  return (
    <ChartFrame id="xd-plane" source={source('LoadingPlane')} provenance={runLine(d)}
      title={`Loading = concentration × volume — ${ec ? 'why the pilot filters are light, and what would fix it' : `the same plane by ${M.label}`}`}
      subtitle={`Each filter by ${M.label} concentration (across) and sampled volume (up); grey diagonals are equal ${M.short} per filter. The orange band is SPARTAN Addis's middle half of ${M.short} per filter${ec ? ': a pilot filter must land in it to load the spectrum as SPARTAN Addis does' : ''}. Measure puts the same filters on the plane by EC, by HIPS absorption (BC at MAC 10) or by weighed PM2.5 mass. Points: Site medians shows one point per site instead (its median concentration and volume; hollow IMPROVE circles are urban sites). Click a legend item to hide that population, or point at one to fade the others. Volume needed (EC only) draws an arrow for how much more air the Adama sampler would need at Adama's own EC concentration. Click a filter to open its record; right-click to exclude it.`}
      controls={<>
        <Segmented label="Measure" value={measure} options={measures} onChange={setMeasure} optionLabel={(o) => ({ EC: 'EC', HIPS: 'HIPS BC', 'PM2.5': 'PM2.5' }[o])} />
        <Segmented label="Points" value={level} options={['filter', 'site'] as const} onChange={setLevel} optionLabel={(o) => (o === 'filter' ? 'Each filter' : 'Site medians')} />
        {ec && <Toggle label="Volume needed" checked={arrow} onChange={(v) => setArrow(v ? 'on' : 'off')} title="Arrow from the Adama pilot up to the volume that would load a filter like SPARTAN Addis's median, at Adama's own EC concentration" />}
      </>}>
      <Scatter points={plot} xLabel={`${M.label} concentration (µg/m³)`} yLabel="sampled volume (m³)" xLog yLog maxSide={420} xDomain={xD} yDomain={yD}
        under={({ x, y }) => {
          // the drawn (niced) axis ranges, so the diagonals run corner to corner and their labels spread out
          const [x0, x1] = x.domain(), [y0, y1] = y.domain()
          const line = (ug: number) => { const a = Math.max(x0, ug / y1), b = Math.min(x1, ug / y0); return { x1: x(a), y1: y(ug / a), x2: x(b), y2: y(ug / b) } }
          const area = (lo: number, hi: number) => { const xs = d3.range(0, 61).map((i) => x0 * (x1 / x0) ** (i / 60))
            const up = xs.map((cx) => [x(cx), y(Math.min(y1, Math.max(y0, hi / cx)))]), dn = xs.map((cx) => [x(cx), y(Math.min(y1, Math.max(y0, lo / cx)))]).reverse()
            return `M${[...up, ...dn].map((pt) => pt.join(',')).join('L')}Z` }
          return <g>
            {Number.isFinite(band[0]) && Number.isFinite(band[1]) && <path d={area(band[0], band[1])} fill={ADDIS_ALL} fillOpacity={0.14} />}
            {M.diagonals.map((ug) => { const l = line(ug); return <g key={ug}><line {...l} stroke={INK.identity} strokeDasharray="3 3" />
              {/* along its own diagonal, near the top end, so neighbouring labels never collide */}
              <text transform={`translate(${l.x1 + (l.x2 - l.x1) * 0.06},${l.y1 + (l.y2 - l.y1) * 0.06}) rotate(${(Math.atan2(l.y2 - l.y1, l.x2 - l.x1) * 180) / Math.PI})`} dy={-3}
                fontSize={9.5} fill={INK.muted} stroke={INK.panel} strokeWidth={3} paintOrder="stroke">{ug} µg/filter</text></g> })}
          </g>
        }}
        over={!arrow ? undefined : ({ x, y }) => {
          const x0 = x(adama.ec_ugm3), y0 = y(adama.volume), y1 = y(Math.min(yD[1], med["volume at Adama's median EC (m³)"]))
          return <g pointerEvents="none">
            <defs><marker id="xd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill={ADAMA} /></marker></defs>
            <line x1={x0} y1={y0 - 8} x2={x0} y2={y1} stroke={ADAMA} strokeWidth={2} markerEnd="url(#xd-arrow)" />
            <text x={x0 + 6} y={(y0 + y1) / 2} fontSize={10.5} fontWeight={600} fill={ADAMA} stroke={INK.panel} strokeWidth={3} paintOrder="stroke">≈ {Math.round(med["volume at Adama's median EC (m³)"])} m³ needed ({f2(med["volume at Adama's median EC (m³)"] / adama.volume)}× today)</text>
          </g>
        }} />
      <Legend {...lg.props} items={pops.map((p) => ({ label: POP_LABEL[p], color: POP_COLOR[p] }))} />
      {!ec && missing.length > 0 && <Note>No {M.label} in this run for {missing.map((p) => POP_LABEL[p]).join(', ')}, so {missing.length > 1 ? 'they are' : 'it is'} not on this plane; the table below is EC per filter.</Note>}
      <table className="census-table" style={{ marginTop: 6 }}>
        <thead><tr><th>Target EC per filter</th><th>µg</th><th>Volume at Adama's EC</th><th>× today's Adama volume</th><th>Hours at the Adama flow</th><th>Volume at SPARTAN Addis's EC</th></tr></thead>
        <tbody>{d.volume.map((v) => <tr key={v.target}><td>{v.target}</td><td>{f2(v['EC µg/filter'])}</td><td>{f2(v["volume at Adama's median EC (m³)"])} m³</td>
          <td>{f2(v["volume at Adama's median EC (m³)"] / v['Adama volume now (m³)'])}×</td><td>{Math.round(v['hours at the Adama flow rate'])} h</td><td>{f2(v['volume at SPARTAN Addis median EC (m³)'])} m³</td></tr>)}</tbody>
      </table>
      <ExplainOnly><Note>At SPARTAN Addis's own median EC concentration (if the Addis city site resembles it), the volumes in the last column apply. Longer sampling at the same flow reaches the same loading; a 24 h sample at the Adama flow is about {f2(adama.volume)} m³.</Note></ExplainOnly>
    </ChartFrame>
  )
}

function SeasonPanel({ d }: { d: TorData }) {
  const rows = d.seasons.map((s) => ({
    key: s.season, label: `${s.season}${s['Adama pairs'] ? `  ← all ${s['Adama pairs']} Adama pairs` : ''}`, color: s['Adama pairs'] ? ADAMA : ADDIS_ALL, bold: s['Adama pairs'] > 0,
    values: { ratio: { v: s['published ÷ HIPS (median)'], lo: s['published ÷ HIPS p25'], hi: s['published ÷ HIPS p75'] }, slope: { v: s['Deming slope'] }, intercept: { v: s['Deming intercept'] } },
    tip: [s.season, `${s['Addis filters']} Addis filters`, `published FTIR ÷ HIPS: median ${f2(s['published ÷ HIPS (median)'])} (25–75 % ${f2(s['published ÷ HIPS p25'])}–${f2(s['published ÷ HIPS p75'])})`, `Deming slope ${coef(s['Deming slope'])}, intercept ${coefSigned(s['Deming intercept'])} µg/m³`, `${s['Adama pairs']} Adama pairs`],
  }))
  return (
    <ChartFrame id="xd-season" source={source('SeasonPanel')} provenance={runLine(d)}
      title="Season — when the Addis disagreement happens, and when the pilot was sampled"
      subtitle="SPARTAN Addis by season (site calendar, February in Belg): published FTIR EC ÷ HIPS BC (median with 25–75 %), and the Deming fit against HIPS. The seasons differ, and the Adama pilot was sampled entirely in one of them (green). Pairs meant to test the calibration at Addis need to cover every season, or at least the one the question is about.">
      <Forest rows={rows} labelWidth={230} panels={[{ key: 'ratio', label: 'published FTIR ÷ HIPS BC', ref: 1, fmt: f2 }, { key: 'slope', label: 'Deming slope vs HIPS', ref: 1, fmt: coef }, { key: 'intercept', label: 'Deming intercept (µg/m³)', ref: 0, fmt: coefSigned }]} />
    </ChartFrame>
  )
}

function ContractCard({ d }: { d: TorData }) {
  const c = d.contract
  const lim = (r: Contract['pass rules'][number]) => Array.isArray(r.limit) ? `${r.limit[0]} – ${r.limit[1]}` : r.unit === 'fraction of TOR' ? `≤ ${Math.round((r.limit as number) * 100)} %` : r.unit === 'share of pairs' ? `≥ ${Math.round((r.limit as number) * 100)} % of pairs` : `≤ ${r.limit} ${r.unit}`
  const pilotFor = (model: string) => d.pilot.find((p) => p.model === model && p.reference === 'ECTR' && p.pairs === 'three unflagged')
  const models = [...new Set(d.pilot.map((p) => p.model))]
  return (
    <ExplainOnly><ChartFrame id="xd-contract" source={source('ContractCard')} exportable={false} provenance={runLine(d)}
      title="The pass contract — written before the Addis city pairs are read"
      subtitle="What agreement with TOR would count as evidence that the frozen calibration works at Addis. Values are proposals to agree with Ann and Satoshi; once agreed, the file is kept as the record and the pairs are scored against it unchanged.">
      <div style={{ display: 'inline-block', padding: '2px 8px', borderRadius: 4, background: themed('#fff4e5', '#3a2a1c'), color: INK.warn, fontSize: 11, fontWeight: 700, marginBottom: 8 }}>{c.status}</div>
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
    </ChartFrame></ExplainOnly>
  )
}

function NetworkMap({ d }: { d: TorData }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const geo = useRef<GeoMapHandle>(null)
  const [of, setOf] = useSearchParam<'Addis' | 'Adama'>('xd-map.of', 'Addis', ['Addis', 'Adama'])
  const [region, setRegion] = useSearchParam<'World' | 'United States'>('xd-map.region', 'World', ['World', 'United States'])
  const [arcsParam, setArcs] = useSearchParam<'on' | 'off'>('xd-map.arcs', 'on', ['on', 'off'])
  const [projection, setProjection] = useSearchParam<'Flat' | 'Globe'>('xd-map.proj', 'Flat', ['Flat', 'Globe'])
  const arcs = arcsParam === 'on'
  const sites = d.neighbour_sites.filter((s) => s.lat != null && s.lon != null)
  const net = Object.fromEntries(d.network.map((r) => [r.site, r]))
  const addis = net.ETAD, adama = net.ADAMA
  const origin: [number, number] = of === 'Addis' ? [addis.lon!, addis.lat!] : [d.adama_latlon[1], d.adama_latlon[0]]
  const h = Math.max(320, Math.min(520, width * 0.5))
  const maxShare = d3.max(sites, (s) => s[of]) ?? 0.01
  const colours = useColourScheme('xd-map', 'sequential', 'Reds')
  const ramp = d3.interpolateRgbBasis(colours.stops)
  const colour = d3.scaleSequential((t: number) => ramp(0.15 + 0.85 * t)).domain([0, maxShare])
  const rN = d3.scaleSqrt().domain([0, d3.max(d.network, (r) => r.filters) ?? 1]).range([1.5, 9])
  const top = [...sites].sort((a, b) => b[of] - a[of]).slice(0, 12)
  // the world frames Ethiopia and the IMPROVE network together; the US view the contiguous states
  // (Fit sites takes in Alaska, Hawaii and the Virgin Islands)
  const frame = useMemo<LngLatPoint[]>(() => region === 'World' ? [...sites.map((s) => [s.lon!, s.lat!] as const), origin] : [[-125, 24], [-66, 50]],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [region, sites.length])
  // great-circle arcs (geo-toolkit), coloured and weighted by the share; drawn as a MapLibre line layer
  const arcLines = useMemo<GeoJSON.FeatureCollection>(() => ({ type: 'FeatureCollection', features: top.map((s) => ({
    type: 'Feature', properties: { id: s.site, c: colour(s[of]), w: 0.8 + 6 * s[of] / maxShare },
    geometry: { type: 'LineString', coordinates: greatCircleArc(origin, [s.lon!, s.lat!], { steps: 48 }) },
  })) }),
  // eslint-disable-next-line react-hooks/exhaustive-deps
  [of, colours.stops.join(), sites.length])
  return (
    <ChartFrame id="xd-map" source={source('NetworkMap')} provenance={runLine(d)}
      title="The TOR network the calibration learned from — and where Addis borrows from"
      subtitle={`Every IMPROVE site in the calibration set (size = filters), coloured by its share of ${of}'s nearest spectral neighbours (the 500 most similar IMPROVE spectra per filter, averaged over ${of === 'Addis' ? 'the Addis evaluation filters' : 'the five Adama pilot filters'}). Great-circle arcs join ${of} to its twelve most-borrowed TOR sites. SPARTAN Addis and the Adama pilot are marked in Ethiopia.`}
      controls={<><Segmented label="Neighbours of" value={of} options={['Addis', 'Adama'] as const} onChange={setOf} /><Segmented label="Region" value={region} options={['World', 'United States'] as const} onChange={setRegion} />
        <Segmented label="Projection" value={projection} options={['Flat', 'Globe'] as const} onChange={setProjection} />
        <Toggle label="Arcs" checked={arcs} onChange={(v) => setArcs(v ? 'on' : 'off')} />
        <button type="button" className="btn quiet" onClick={() => geo.current?.fit(sites.map((s) => [s.lon!, s.lat!]), { pad: 0.08 })}>Fit sites</button></>}>
      <div ref={ref} style={{ position: 'relative' }}>
        <GeoMap ref={geo} height={h} globe={projection === 'Globe'} frame={frame} frameKey={region} frameOptions={{ pad: 0.05, maxZoom: 5 }}
          label={`IMPROVE calibration sites coloured by their share of ${of}'s nearest neighbours`}>
          <MapLineLayer data={arcLines} visible={arcs} interactive={false} color={['get', 'c']} width={['get', 'w']} opacity={0.75} />
          {[...sites].sort((a, b) => a[of] - b[of]).map((s) => {
            const n = net[s.site], r = rN(n?.filters ?? 1)
            return <MapMarker key={s.site} longitude={s.lon!} latitude={s.lat!}>
              <MarkerContent className="bubble-marker" style={{ cursor: 'default' }}>
                <svg width={2 * r + 4} height={2 * r + 4} viewBox={`${-r - 2} ${-r - 2} ${2 * r + 4} ${2 * r + 4}`}
                  onMouseEnter={(e) => tip.show(e, [`${s.name} (${s.site})${n?.urban ? ' · urban' : ''}`, `${n?.filters ?? '?'} calibration filters (${pct(s['IMPROVE share of filters'])} of IMPROVE)`,
                    `share of Addis neighbours ${pct(s.Addis)} · of Adama ${pct(s.Adama)}`, `median TOR EC ${f2(n?.ec_ug)} µg/filter, ${f2(n?.ec_ugm3)} µg/m³`])} onMouseLeave={tip.hide}>
                  <circle r={r} fill={s[of] > 0 ? colour(s[of]) : INK.panel} stroke={s[of] > 0 ? '#7a2d0b' : IMPROVE_GREY} strokeWidth={0.6} />
                </svg>
              </MarkerContent>
            </MapMarker>
          })}
          {([[addis, 'SPARTAN Addis', ADDIS_ALL], [adama, 'Adama pilot', ADAMA]] as const).map(([row, label, c]) => {
            const lon = label === 'Adama pilot' ? d.adama_latlon[1] : row.lon!, lat = label === 'Adama pilot' ? d.adama_latlon[0] : row.lat!
            return <MapMarker key={label} longitude={lon} latitude={lat}>
              <MarkerContent className="bubble-marker" style={{ cursor: 'default' }}>
                <svg width={22} height={22} viewBox="-11 -11 22 22" onMouseEnter={(e) => tip.show(e, [label, `${row.filters} filters · median EC ${f2(row.ec_ug)} µg/filter`, `median distance percentile ${Math.round(row.distance)}`])} onMouseLeave={tip.hide}>
                  <path d={d3.symbol(label === 'Adama pilot' ? d3.symbolDiamond : d3.symbolStar, 160)() ?? ''} fill={c} stroke={INK.text} strokeWidth={1} />
                </svg>
                <span data-export="text" className="map-marker-label" style={label === 'Adama pilot' ? { top: 'calc(50% + 9px)' } : { bottom: 'calc(50% + 9px)' }}>{label}</span>
              </MarkerContent>
            </MapMarker>
          })}
        </GeoMap>
        {tip.node}
      </div>
      <div className="legend" style={{ justifyContent: 'center' }}>
        <ColorLegend scale={colour} scheme={colours.scheme} label={`share of ${of}'s nearest neighbours`} format={(v) => pct(v)} />
        <Legend items={[{ label: 'SPARTAN Addis', color: ADDIS_ALL }, { label: 'Adama pilot (TOR)', color: ADAMA }, { label: 'IMPROVE site never chosen', color: IMPROVE_GREY, shape: 'ring' }]} />
      </div>
      <Note>Adama is placed at the pilot sampler's recorded position. Neighbours use the spline spectra and the 600–3600 cm⁻¹ similarity region without CO₂ and PTFE bands. Drag to pan, scroll to zoom; Globe shows the arcs as the routes they are.</Note>
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
    <ChartFrame id="xd-where" source={source('WhereItSits')} provenance={runLine(d)}
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
                  <rect x={x(q(iv, 0.05))} y={mid - 9} width={x(q(iv, 0.95)) - x(q(iv, 0.05))} height={18} fill={themed('#e5e7eb', '#2c333c')} />
                  <rect x={x(q(iv, 0.25))} y={mid - 9} width={x(q(iv, 0.75)) - x(q(iv, 0.25))} height={18} fill={themed('#cbd2d9', '#46505c')} />
                  <line x1={x(q(iv, 0.5))} x2={x(q(iv, 0.5))} y1={mid - 9} y2={mid + 9} stroke={INK.axis} strokeWidth={2} />
                </>}
                {nv.length > 0 && <>
                  <rect x={x(q(nv, 0.05))} y={midN - 4} width={x(q(nv, 0.95)) - x(q(nv, 0.05))} height={8} fill={SPARTAN} fillOpacity={0.18} />
                  <rect x={x(q(nv, 0.25))} y={midN - 4} width={x(q(nv, 0.75)) - x(q(nv, 0.25))} height={8} fill={SPARTAN} fillOpacity={0.45} />
                  <line x1={x(q(nv, 0.5))} x2={x(q(nv, 0.5))} y1={midN - 5} y2={midN + 5} stroke={SPARTAN} strokeWidth={2} />
                </>}
                {sites.map((s) => { const val = s[siteVal[v.key]] as number; return Number.isFinite(val) && (!v.log || val > 0) ? <line key={s.site} x1={x(val)} x2={x(val)} y1={mid + 11} y2={mid + 16} stroke={IMPROVE_GREY} strokeWidth={0.8} /> : null })}
                {add.map((p, i) => { const val = num(p[v.key]); return Number.isFinite(val) && (!v.log || val > 0) ? <circle key={i} cx={x(val)} cy={mid - 6 + jitter(i) * 12} r={2} fill={ADDIS_ALL} {...dotFill('dense')} /> : null })}
                {ada.map((p, i) => { const val = num(p[v.key]); return Number.isFinite(val) && (!v.log || val > 0)
                  ? <path key={i} d={d3.symbol(d3.symbolDiamond, 70)() ?? ''} transform={`translate(${x(val)},${mid})`} fill={ADAMA} stroke={INK.panel} strokeWidth={1}
                    onMouseEnter={(e) => tip.show(e, [`Adama ${p.id}`, `${v.label}: ${v.fmt(val)}`])} onMouseLeave={tip.hide} /> : null })}
                <XAxis scale={x} y={midN + 8} tickCount={6} format={(t) => v.fmt(+t)} />
              </g>
            </g>
          })}
        </svg>
        {tip.node}
      </div>
      <Legend items={[{ label: 'IMPROVE filters (band) and site medians (ticks)', color: themed('#cbd2d9', '#46505c'), shape: 'square' }, { label: 'other SPARTAN sites (band)', color: SPARTAN, shape: 'square' }, { label: 'SPARTAN Addis filters', color: ADDIS_ALL }, { label: 'Adama pilot pairs', color: ADAMA }]} />
      <Note>Adama has no weighed PM2.5 in the pair records. EC per filter = concentration × the filter's own sampled volume; SPARTAN Addis EC is the published FTIR value (no TOR exists there).</Note>
    </ChartFrame>
  )
}

const SITE_AXES = ['EC µg/filter', 'EC µg/m³', 'OC/EC', 'PM2.5 µg/filter', 'distance percentile', 'volume m³'] as const
const SITE_GROUP: Record<string, string> = { 'EC µg/filter': 'On the filter', 'PM2.5 µg/filter': 'On the filter', 'volume m³': 'On the filter', 'EC µg/m³': 'In the air', 'OC/EC': 'In the air', 'distance percentile': 'Spectrum' }
const SITE_KEY: Record<string, keyof NetRow> = { 'EC µg/filter': 'ec_ug', 'EC µg/m³': 'ec_ugm3', 'OC/EC': 'ocec', 'PM2.5 µg/filter': 'pm_ug', 'distance percentile': 'distance', 'volume m³': 'volume' }

function NetworkScatter({ d }: { d: TorData }) {
  const [xk, setXk] = useSearchParam<typeof SITE_AXES[number]>('xd-net.x', 'EC µg/filter', SITE_AXES)
  const [yk, setYk] = useSearchParam<typeof SITE_AXES[number]>('xd-net.y', 'distance percentile', SITE_AXES)
  const [labelsParam, setLabels] = useSearchParam<'on' | 'off'>('xd-net.labels', 'on', ['on', 'off'])
  const labels = labelsParam === 'on'
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
  const choices = useMemo(() => SITE_AXES.map((k) => ({ key: k, label: AXIS_INFO[k].label, unit: AXIS_INFO[k].unit, group: SITE_GROUP[k] })), [])
  return (
    <ChartFrame id="xd-net" source={source('NetworkScatter')} provenance={runLine(d)}
      title="Sites compared — every IMPROVE and SPARTAN site, SPARTAN Addis and Adama, on two chosen axes"
      subtitle="One point per site: its median over all its filters. Click either axis title to pick what goes across and up. Grey = IMPROVE calibration sites (size = filters; hollow = Ann's six urban sites), purple = other SPARTAN sites, orange = SPARTAN Addis, green = the Adama pilot."
      controls={<Toggle label="Name SPARTAN sites" checked={labels} onChange={(v) => setLabels(v ? 'on' : 'off')} />}>
      <AxisChoices noun="site medians" options={choices} x={[xk, (v) => setXk(v as typeof xk)]} y={[yk, (v) => setYk(v as typeof yk)]}>
      <Scatter points={pts} pick xLabel={opt(xk)} yLabel={opt(yk)} xLog={X.log} yLog={Y.log} maxSide={440} yDomain={yk === 'distance percentile' ? [0, 100] : undefined} xDomain={xk === 'distance percentile' ? [0, 100] : undefined}
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
            <text x={l.tx} y={l.ty} dy="0.32em" textAnchor={l.anchor} fontSize={l.bold ? 11 : 9} fontWeight={l.bold ? 700 : 400} fill={INK.text} stroke={INK.panel} strokeWidth={3} strokeLinejoin="round" paintOrder="stroke">{l.text}</text>
          </g>)}</g>
        }} />
      </AxisChoices>
      <Legend {...lg.props} items={order.map((p) => ({ label: POP_LABEL[p], color: POP_COLOR[p] }))} note="click to show or hide" />
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
    <ChartFrame id="xd-nearest" source={source('NearestSites')} provenance={runLine(d)}
      title="Nearest TOR sites — which IMPROVE sites Addis and Adama borrow their calibration from"
      subtitle="Share of the 500 nearest IMPROVE spectra per filter that come from each site, for Addis (orange) and Adama (green), beside the site's share of all calibration filters (grey tick). A site far right of its tick is over-represented among the neighbours.">
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={labelW + pw + 30} height={rows.length * rowH + 44} fontFamily={FONT.family}>
          {rows.map((r, i) => { const cy = i * rowH + rowH / 2
            return <g key={r.site} onMouseEnter={(e) => tip.show(e, [`${r.name} (${r.site})`, `Addis ${pct(r.Addis)} · Adama ${pct(r.Adama)}`, `${pct(r['IMPROVE share of filters'])} of calibration filters`])} onMouseLeave={tip.hide}>
              <rect x={0} y={cy - rowH / 2} width={labelW + pw} height={rowH} fill={i % 2 ? 'transparent' : INK.zebra} />
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
    <ChartFrame id="xd-precision" source={source('Precision')} provenance={runLine(d)}
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
  const pts = d.adama_pairs.map((p) => ({ x: num(p[refk === 'ECTR' ? 'ec_tr_ugm3' : 'ec_tt_ugm3']), y: num(p[key]), color: ADAMA, hollow: !!p.flag,
    tip: [`Adama ${p.ptfe_id} / quartz ${p.quartz_id}`, String(p.date), `${refk} ${f2(p[refk === 'ECTR' ? 'ec_tr_ugm3' : 'ec_tt_ugm3'])} µg/m³ · FTIR ${f2(p[key])} µg/m³`, `distance percentile ${Math.round(num(p['distance percentile']))}`, p.flag ? `flag: ${p.flag}` : 'unflagged'] }))
  return (
    <ChartFrame id="xd-pilot" source={source('PilotPairs')} provenance={runLine(d)}
      title="The Adama pilot pairs — FTIR against TOR, with the draft tolerance"
      subtitle="The five Adama quartz/PTFE date-candidate pairs (hollow: flagged for a start offset or a volume mismatch). Shaded: the draft per-pair tolerance |FTIR − TOR| ≤ max(0.5 µg/m³, 30 % of TOR). No line is fitted to five pairs; the table scores them. A pilot, not the test: identities and sampler equivalence are unconfirmed."
      controls={<><Segmented label="FTIR" value={model} options={models} onChange={setModel} /><Segmented label="TOR" value={refk} options={['ECTR', 'ECTT'] as const} onChange={setRefk} /></>}>
      <Crossplot points={pts} title="" xLabel={`TOR EC, ${refk} (µg/m³)`} yLabel={`FTIR EC, ${model} (µg/m³)`} fit={false} maxSide={400}
        background={(x, y) => { const [lo, hi] = x.domain(), xs = d3.range(lo, hi + (hi - lo) / 120, (hi - lo) / 60)
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
