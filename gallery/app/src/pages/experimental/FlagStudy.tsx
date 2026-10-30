import { useFitHeight } from '@/lib/chartFit'
import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { useExport } from '@/lib/data'
import { ExportState } from '@/components/ExportState'
import { ChartFrame, ExplainOnly, Note, Segmented, Select, Toggle } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { Legend, useLegend } from '@/components/Legend'
import { ColorLegend } from '@/components/ColorLegend'
import { HeatCell } from '@/components/HeatCell'
import { SweepPlot } from '@/components/SweepPlot'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useColourScheme } from '@/lib/colourSchemes'
import { useSearchParam } from '@/lib/url'
import { ADDIS_ALL, FONT, IMPROVE_GREY, INK, inkOn, themed } from '@/lib/theme'
import { dotFill, dotRim } from '@/lib/highlight'
import { useTheme } from '@/lib/colorTheme'
import { f2, f3, pct, signed } from '@/lib/format'
import { Forest, placeLabels } from './shared'
import { FixedSetNote } from '@/lib/markedRows'

/**
 * Experimental · tests 1–2: what the decision workbook's "impacted" flag actually sees
 * (research/ftir_hips_chem/decision_workbook_20261022.ipynb, Section 6b), exported
 * by gallery/data/export_experimental.py; nothing is refitted in the browser.
 */

const FLAG_PATH = 'experimental/flag_detection.json'
type Cols = Record<string, (number | string | boolean | null)[]>
interface Rung {
  rung: string; inputs: string; 'at SPARTAN': boolean; AUC: number; 'AUC lo': number; 'AUC hi': number
  'gain over loading': number; 'gain lo': number; 'gain hi': number; 'average precision': number
}
interface Within { 'loading decile': number; score: string; n: number; 'impacted share': number; 'median PM2.5 µg': number; AUC: number }
interface CoverageCurve { 'kept share': number[]; 'RMSE kept': number[]; 'bias kept': number[]; 'impacted among kept': number[]; 'impacted let through': number[] }
interface SiteKeep { site: string; filters: number; 'kept share': number; 'impacted share': number; 'RMSE all': number; 'RMSE kept': number | null; 'missed failures': number; urban: boolean }
interface GridCell { 'loading bin': number; 'distance bin': number; 'loading range': string; 'distance range': string; 'IMPROVE filters': number; 'impacted share': number | null; 'caught share': number | null; 'Addis filters': number }
interface FlagData {
  schema_version: number
  generated: string
  source: string
  settings: { MAIN: string; ABS_TOL: number; REL_TOL: number; FLAG_RATE: number; BEST: string; N_FOLDS: number; k_selected: Record<string, number>; bundle: string }
  counts: { improve: number; impacted: number; flagged: number; missed: number; addis: number; network: number; network_sites: number }
  categories: string[]
  ladder: Rung[]
  within: Within[]
  coverage: Record<string, CoverageCurve>
  site_keep: SiteKeep[]
  missed_summary: Record<string, number | string>[]
  missed_top: Record<string, number | string | boolean>[]
  map_improve: Cols
  map_spartan: Cols
  grid: GridCell[]
}

export function FlagStudy() {
  const { data, error } = useExport<FlagData>(FLAG_PATH, (d) => d.schema_version === 1 ? null : 'unsupported data version')
  if (!data) return <ExportState title="What does the flag see?" what="the flag study" error={error} command="uv run --no-sync python gallery/data/export_experimental.py" />
  return (
    <>
      <Findings d={data} />
      <BlindSpotMap d={data} />
      <Ladder d={data} />
      <WithinLoading d={data} />
      <Coverage d={data} />
      <WhoGetsThrough d={data} />
      <SiteKeepChart d={data} />
      <SpartanSites d={data} />
    </>
  )
}

// ------------------------------------------------------------------ helpers
const LOADING = 'Loading: weighed PM2.5'
/** cells with fewer IMPROVE filters than this are left uncoloured on the map (a rate from a dozen filters is noise) */
const MIN_CELL = 30
const DISTANCE = 'Distance: T² + Q'
const CAT_COLOR: Record<string, string> = { get fine() { return themed('#d5d9de', '#4b545f') }, 'false alarm': '#6baed6', get 'caught failure'() { return INK.good }, get 'missed failure'() { return INK.negative } }
/** viridis from dark to light rungs; in the dark theme its darkest purples would vanish, so the ramp starts later */
/** the ladder's rungs in order, from one palette (Viridis by default) that the ladder and the coverage curves share
 *  (`xflag-rungs`, changed from either chart's options); the dark theme's palette steps are its own (lib/colourSchemes) */
const useRungColours = () => {
  const colours = useColourScheme('xflag-rungs', 'sequential', 'Viridis')
  const ramp = d3.interpolateRgbBasis(colours.stops)
  return { picker: colours.picker, color: (i: number, n: number) => ramp(0.08 + 0.8 * (i / Math.max(1, n - 1))) }
}
const col = <T,>(c: Cols, k: string) => c[k] as T[]
const pct100 = (v: unknown) => v == null || !Number.isFinite(+v) ? '—' : String(Math.round(+v))
const median = (v: (number | null)[]) => d3.median(v.filter((x): x is number => x != null && Number.isFinite(x)))
const source = (fn: string) => `gallery/app/src/pages/experimental/FlagStudy.tsx › ${fn}`
const runNote = (d: FlagData) => `Workbook run ${d.source.split('/').pop()} · impacted = |FTIR − TOR| > ${d.settings.ABS_TOL} µg and > ${Math.round(d.settings.REL_TOL * 100)} % of TOR · ${d.settings.MAIN} PLS, k ${d.settings.k_selected?.[d.settings.MAIN]} · ${d.counts.improve.toLocaleString()} IMPROVE filters, held-out sites (${d.settings.N_FOLDS} folds)`

/** Addis and SPARTAN rows of the map, as objects. */
function spartanRows(d: FlagData) {
  const c = d.map_spartan, n = (c.filter_id ?? []).length
  return d3.range(n).map((i) => ({
    population: c.population[i] as string, id: String(c.filter_id[i]), code: c.code[i] as string, site: c.site[i] as string,
    date: c.date[i] as string, season: (c.season[i] as string) || '', load: c['loading percentile'][i] as number | null,
    dist: c['distance percentile'][i] as number | null, pm: c['PM2.5 µg'][i] as number | null, risk: c.risk[i] as number,
    flagged: !!c.flagged[i], beyond: !!c['beyond IMPROVE loading'][i],
  }))
}

// ------------------------------------------------------------------ 0. findings
function Findings({ d }: { d: FlagData }) {
  const L = Object.fromEntries(d.ladder.map((r) => [r.rung, r]))
  const best = L[d.settings.BEST], load = L[LOADING], dist = L[DISTANCE]
  const hips = d.ladder.find((r) => !r['at SPARTAN'])
  const withinDist = d.within.filter((w) => w.score === DISTANCE).map((w) => w.AUC)
  const S = Object.fromEntries(d.missed_summary.map((r) => [r.category as string, r]))
  const miss = S['missed failure'], caught = S['caught failure']
  const rowsS = spartanRows(d)
  const addis = rowsS.filter((r) => r.population === 'Addis')
  const siteFlag = d3.groups(rowsS, (r) => r.population === 'Addis' ? 'Addis' : r.code).filter(([, rs]) => rs.length >= 10)
    .map(([k, rs]) => ({ k, f: d3.mean(rs, (r) => +r.flagged)! })).sort((a, b) => b.f - a.f)
  const addisRank = siteFlag.findIndex((s) => s.k === 'Addis') + 1
  const caughtShare = (d.counts.impacted - d.counts.missed) / d.counts.impacted
  const distAdds = dist && load && dist['gain lo'] > 0
  return (
    <ExplainOnly>
      <ChartFrame id="xflag-findings" title="What the flag sees — short answers" exportable={false}>
        <ol style={{ margin: '4px 0 0 18px', lineHeight: 1.55 }}>
          <li><strong>Measured loading alone</strong> (weighed PM2.5 µg per filter) separates impacted IMPROVE filters with AUC {f2(load?.AUC)}; the spectral distance alone (T² + Q) reaches {f2(dist?.AUC)}{dist ? ` (${signed(dist['gain over loading'])} vs loading, 95 % ${signed(dist['gain lo'])} to ${signed(dist['gain hi'])})` : ''}. {distAdds ? 'Distance carries information beyond a scale.' : 'Distance on its own is no better than a scale.'}</li>
          <li><strong>Within one loading decile</strong> the distance's AUC has median {f2(median(withinDist))} (range {f2(d3.min(withinDist))}–{f2(d3.max(withinDist))}). {(median(withinDist) ?? 0.5) < 0.6 ? 'Among filters of similar loading it barely ranks errors.' : 'It still ranks errors among filters of similar loading.'}</li>
          <li><strong>The best flag usable at SPARTAN</strong> is "{d.settings.BEST}": AUC {f2(best?.AUC)}, {signed(best?.['gain over loading'])} over loading (95 % {signed(best?.['gain lo'])} to {signed(best?.['gain hi'])}).{hips ? ` Adding HIPS lifts it to ${f2(hips.AUC)}, but HIPS is half of the FTIR–HIPS disagreement at SPARTAN, so that rung is circular there.` : ''}</li>
          <li><strong>Flagging {pct(d.settings.FLAG_RATE)}</strong> of IMPROVE catches {pct(caughtShare)} of impacted filters and lets {d.counts.missed.toLocaleString()} through. The missed ones sit at median loading percentile {pct100(miss?.['loading percentile (median)'])} and distance percentile {pct100(miss?.['distance percentile (median)'])} (caught: {pct100(caught?.['loading percentile (median)'])} and {pct100(caught?.['distance percentile (median)'])}); {pct(miss?.['under-predicted share'])} of them are under-predicted.</li>
          <li><strong>Addis on the same map:</strong> median loading percentile {pct100(median(addis.map((r) => r.load)))}, distance percentile {pct100(median(addis.map((r) => r.dist)))}; {pct(d3.mean(addis, (r) => +r.beyond))} are heavier than any IMPROVE filter and {pct(d3.mean(addis, (r) => +r.flagged))} would be flagged, rank {addisRank} of {siteFlag.length} SPARTAN sites (most flagged: {siteFlag.slice(0, 3).map((s) => `${s.k} ${pct(s.f)}`).join(', ')}). Whatever goes wrong at Addis, spectrum, loading and XRF do not make it look like IMPROVE's failures.</li>
        </ol>
        <Note>{runNote(d)}. These are statements about held-out IMPROVE sites against TOR; at SPARTAN there is no TOR, so the Addis rows say where Addis sits, not how wrong it is.</Note>
      </ChartFrame>
    </ExplainOnly>
  )
}

// ------------------------------------------------------------------ 1. the blind-spot map
type Overlay = 'Addis filters' | 'SPARTAN sites' | 'One SPARTAN site' | 'None'
function BlindSpotMap({ d }: { d: FlagData }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const [cellMetric, setCellMetric] = useSearchParam<'Failure rate' | 'Catch rate'>('xflag-map.cell', 'Failure rate', ['Failure rate', 'Catch rate'])
  const [show, setShow] = useSearchParam<'Failures' | 'Failures + alarms' | 'All' | 'None'>('xflag-map.show', 'Failures', ['Failures', 'Failures + alarms', 'All', 'None'])
  const [overlay, setOverlay] = useSearchParam<Overlay>('xflag-map.overlay', 'Addis filters', ['Addis filters', 'SPARTAN sites', 'One SPARTAN site', 'None'])
  const spartan = useMemo(() => spartanRows(d), [d])
  const codes = useMemo(() => [...new Set(spartan.filter((r) => r.population === 'network').map((r) => r.code))].sort(), [spartan])
  const [site, setSite] = useSearchParam<string>('xflag-map.site', codes[0] ?? '', codes)
  const [countsParam, setCounts] = useSearchParam<'on' | 'off'>('xflag-map.counts', 'on', ['on', 'off'])
  const counts = countsParam === 'on'
  const colours = useColourScheme('xflag-map', 'sequential')
  const lg = useLegend()

  const m = { top: 14, right: 16, bottom: 50, left: 62 }
  const beyondW = 34
  // the square fits the card (a phone included), beside the 'heavier than any IMPROVE filter' strip; in the
  // single-chart view or a tile it also fits the height it is given (lib/chartFit), or the page scrolls
  const fit = useFitHeight()
  const size = Math.max(200, Math.min(width - m.left - m.right - beyondW, fit === null ? 620 : fit - m.top - m.bottom))
  const x = d3.scaleLinear().domain([0, 100]).range([0, size])
  const y = d3.scaleLinear().domain([0, 100]).range([size, 0])
  const metricKey = cellMetric === 'Failure rate' ? 'impacted share' : 'caught share'
  const vals = d.grid.filter((g) => g['IMPROVE filters'] >= MIN_CELL).map((g) => g[metricKey]).filter((v): v is number => v != null)
  // the background uses only the light half of the ramp: it is context, the SPARTAN marks are the subject
  const ramp = d3.interpolateRgbBasis(colours.stops)
  const colour = d3.scaleSequential((t: number) => ramp(0.04 + 0.5 * t)).domain([0, Math.max(0.05, d3.max(vals) ?? 1)])
  const step = size / 5

  const mi = d.map_improve, cats = d.categories
  const n = (mi.filter_id ?? []).length
  const visibleCats = show === 'All' ? cats : show === 'Failures + alarms' ? ['false alarm', 'caught failure', 'missed failure'] : show === 'Failures' ? ['caught failure', 'missed failure'] : []
  const pts = d3.range(n).filter((i) => {
    const cat = cats[mi.category[i] as number]
    return visibleCats.includes(cat) && lg.show(cat) && mi['loading percentile'][i] != null && mi['distance percentile'][i] != null
  }).sort((a, b) => (mi.category[a] as number) - (mi.category[b] as number))

  const lx = (r: { load: number | null; beyond: boolean }) => r.beyond ? size + beyondW / 2 : x(r.load ?? 0)
  let over: typeof spartan = []
  if (overlay === 'Addis filters') over = spartan.filter((r) => r.population === 'Addis')
  if (overlay === 'One SPARTAN site') over = spartan.filter((r) => r.population === 'network' && r.code === site)
  const siteMedians = overlay !== 'SPARTAN sites' ? [] : d3.groups(spartan.filter((r) => r.load != null && r.dist != null), (r) => r.code)
    .map(([code, rs]) => ({ code, site: rs[0].site, n: rs.length, load: d3.median(rs, (r) => r.load!)!, dist: d3.median(rs, (r) => r.dist!)!,
      flagged: d3.mean(rs, (r) => +r.flagged)!, beyond: d3.mean(rs, (r) => +r.beyond)! }))
  const rSite = d3.scaleSqrt().domain([0, d3.max(siteMedians, (s) => s.n) ?? 1]).range([3, 13])
  const siteColours = useColourScheme('xflag-map.sites', 'sequential', 'Reds')
  const siteRamp = d3.interpolateRgbBasis(siteColours.stops)
  const flagColour = d3.scaleSequential((t: number) => siteRamp(0.2 + 0.8 * t)).domain([0, Math.max(0.2, d3.max(siteMedians, (s) => s.flagged) ?? 1)])
  // biggest first, so small sites are drawn on top; labels placed around each circle where they collide with nothing
  const sitesDrawn = [...siteMedians].sort((a, b) => b.n - a.n)
  const labels = placeLabels(sitesDrawn.map((s) => ({ key: s.code, x: x(s.load), y: y(s.dist), r: rSite(s.n), text: s.code === 'ETAD' ? 'Addis (ETAD)' : s.code, bold: s.code === 'ETAD' })), size + beyondW, size)

  return (
    <ChartFrame id="xflag-map" source={source('BlindSpotMap')} provenance={runNote(d)}
      title="Blind-spot map — where the flag looks and where failures hide"
      subtitle={`Every held-out IMPROVE filter placed by its loading (weighed PM2.5 µg, percentile in IMPROVE) and its spectral distance from the calibration (the larger of its T² and Q percentiles). Cells: share of filters there that are impacted, or share of those failures that the "${d.settings.BEST}" flag catches at a ${pct(d.settings.FLAG_RATE)} flag rate. Red dots are the failures it lets through. Rings are SPARTAN filters on the same scales; the strip on the right holds filters heavier than any IMPROVE filter.`}
      tip={<>Distances for SPARTAN come from the PLS fitted on all of IMPROVE, compared with the held-out-site distances of IMPROVE filters (the yardstick for a new site). A spectrum-only flag cannot see a failure that sits where most filters are fine; if SPARTAN rings fall among red dots in pale-green cells, a spectral warning will stay quiet there.</>}
      controls={<>
        <Segmented label="Cells" value={cellMetric} options={['Failure rate', 'Catch rate'] as const} onChange={setCellMetric} />
        <Segmented label="IMPROVE dots" value={show} options={['Failures', 'Failures + alarms', 'All', 'None'] as const} onChange={setShow} />
        <Select label="SPARTAN" value={overlay} options={['Addis filters', 'SPARTAN sites', 'One SPARTAN site', 'None']} onChange={(v) => setOverlay(v as Overlay)} />
        {overlay === 'One SPARTAN site' && <Select label="Site" value={site} options={codes} onChange={setSite} optionLabel={(c) => `${spartan.find((r) => r.code === c)?.site ?? c} (${c})`} />}
        {overlay === 'Addis filters' && <Toggle label="Count Addis filters per cell" checked={counts} onChange={(v) => setCounts(v ? 'on' : 'off')}
          title={`Print in each cell how many of the ${d.counts.addis} Addis filters fall there`} />}
      </>}>
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={size + beyondW + m.left + m.right} height={size + m.top + m.bottom} fontFamily={FONT.family} style={{ display: 'block', margin: '0 auto' }}>
          <g transform={`translate(${m.left},${m.top})`}>
            {d.grid.map((g) => {
              const v = g['IMPROVE filters'] >= MIN_CELL ? g[metricKey] : null
              return <HeatCell key={`${g['loading bin']}-${g['distance bin']}`} x={g['loading bin'] * step} y={(4 - g['distance bin']) * step} width={step} height={step}
                fill={v == null ? null : colour(v)} label={null}
                onEnter={(e) => tip.show(e, [`loading ${g['loading range']} · distance ${g['distance range']} percentile`, `${g['IMPROVE filters'].toLocaleString()} IMPROVE filters`,
                  `impacted ${pct(g['impacted share'])}`, `flag catches ${pct(g['caught share'])} of them`, `${g['Addis filters']} Addis filters here`])} onLeave={tip.hide} />
            })}
            {d.grid.map((g) => {
              const v = g['IMPROVE filters'] >= MIN_CELL ? g[metricKey] : null, cx = g['loading bin'] * step + step / 2, cy = (4 - g['distance bin']) * step
              return <g key={`t${g['loading bin']}-${g['distance bin']}`} pointerEvents="none">
                <text x={cx} y={cy + 13} textAnchor="middle" fontSize={9} fill={v == null ? INK.muted : inkOn(colour(v))} fillOpacity={v == null ? 1 : 0.75}>{v == null ? (g['IMPROVE filters'] ? `${g['IMPROVE filters']} filters` : '') : pct(v)}</text>
                {counts && overlay === 'Addis filters' && g['Addis filters'] > 0 &&
                  <text x={cx} y={cy + step - 6} textAnchor="middle" fontSize={10} fontWeight={600} fill={ADDIS_ALL} stroke={INK.panel} strokeWidth={2.5} paintOrder="stroke">{g['Addis filters']} Addis</text>}
              </g>
            })}
            <rect x={size} y={0} width={beyondW} height={size} fill={INK.empty} />
            <text transform={`translate(${size + beyondW / 2},${size / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill={INK.muted}>heavier than any IMPROVE filter</text>
            <XAxis scale={x} y={size} label="Loading percentile in IMPROVE (weighed PM2.5 per filter)" labelDy={38} tickCount={6} />
            <YAxis scale={y} x={0} label="Spectral distance percentile (max of T², Q)" tickCount={6} labelX={-46} />
            <g>
              {pts.map((i) => {
                const cat = cats[mi.category[i] as number]
                return <circle key={i} cx={x(mi['loading percentile'][i] as number)} cy={y(mi['distance percentile'][i] as number)} r={cat === 'fine' ? 1.6 : 2.4}
                  fill={CAT_COLOR[cat]} {...dotFill(cat === 'fine' ? 'dense' : 'sparse')} opacity={lg.dim(cat)}
                  onMouseEnter={(e) => tip.show(e, [`IMPROVE ${mi.site[i]} · ${mi.filter_id[i]}`, cat, `FTIR − TOR ${signed(mi['error µg'][i] as number)} µg`, `flag risk ${f2(mi.risk[i] as number)}`])} onMouseLeave={tip.hide} />
              })}
            </g>
            {over.filter((r) => r.dist != null && (r.load != null || r.beyond)).map((r) => (
              <circle key={r.id} cx={lx(r)} cy={y(r.dist!)} r={4} fill={r.population === 'Addis' ? ADDIS_ALL : '#7c3aed'} {...dotFill('dense')}
                {...dotRim()}
                onMouseEnter={(e) => tip.show(e, [`${r.site} · ${r.id}`, `${r.date}${r.season ? ` · ${r.season}` : ''}`, `PM2.5 ${f2(r.pm)} µg per filter${r.beyond ? ' (beyond IMPROVE)' : ''}`,
                  `loading pct ${f2(r.load)}, distance pct ${f2(r.dist)}`, `flag risk ${f2(r.risk)}${r.flagged ? ' · flagged' : ''}`])} onMouseLeave={tip.hide} />
            ))}
            {sitesDrawn.map((s) => (
              <circle key={s.code} cx={x(s.load)} cy={y(s.dist)} r={rSite(s.n)} fill={flagColour(s.flagged)} fillOpacity={0.95}
                stroke={s.code === 'ETAD' ? INK.text : INK.panel} strokeWidth={s.code === 'ETAD' ? 2.5 : 1.5}
                onMouseEnter={(e) => tip.show(e, [`${s.site} (${s.code})`, `${s.n} filters · median loading pct ${f2(s.load)}, distance pct ${f2(s.dist)}`, `${pct(s.flagged)} flagged · ${pct(s.beyond)} heavier than any IMPROVE filter`])} onMouseLeave={tip.hide} />
            ))}
            {labels.map((l) => <g key={l.key} pointerEvents="none">
              {l.leader && <line x1={l.ax} y1={l.ay} x2={l.lx} y2={l.ly} stroke={INK.axis} strokeWidth={0.6} />}
              <text x={l.tx} y={l.ty} dy="0.32em" textAnchor={l.anchor} fontSize={l.bold ? 11 : 9.5} fontWeight={l.bold ? 700 : 500}
                fill={INK.text} stroke={INK.panel} strokeWidth={3} strokeLinejoin="round" paintOrder="stroke">{l.text}</text>
            </g>)}
          </g>
        </svg>
        {tip.node}
      </div>
      {/* .legend: the PNG/SVG/PDF export redraws its labels and can move it (lib/export HTML_TEXT) */}
      <div className="legend" style={{ gap: 24, justifyContent: 'center' }}>
        <ColorLegend scale={colour} label={cellMetric === 'Failure rate' ? 'cells: share of IMPROVE filters impacted' : 'cells: share of failures the flag catches'} format={(v) => pct(v)} scheme={colours.scheme} />
        {overlay === 'SPARTAN sites' && <ColorLegend scale={flagColour} scheme={siteColours.scheme} label="sites: share of filters flagged (size = filters, black ring = Addis)" format={(v) => pct(v)} />}
        <Legend {...lg.props} items={[...cats.filter((c) => visibleCats.includes(c)).map((c) => ({ label: c, color: CAT_COLOR[c], detail: (col<number>(mi, 'category').filter((v) => cats[v] === c).length).toLocaleString() })),
          ...(overlay === 'None' || overlay === 'SPARTAN sites' ? [] : [{ label: overlay === 'Addis filters' ? 'Addis filters' : `${site} filters`, color: overlay === 'Addis filters' ? ADDIS_ALL : '#7c3aed' }])]} />
      </div>
      <FixedSetNote what="the workbook's held-out IMPROVE filters and every SPARTAN PM2.5 filter it scored" />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 2. ablation ladder
function Ladder({ d }: { d: FlagData }) {
  const rungs = useRungColours()
  const rows = d.ladder.map((r, i) => {
    const c = rungs.color(i, d.ladder.length), best = r.rung === d.settings.BEST
    return {
      key: r.rung, label: `${r.rung}${best ? ' ★' : ''}`, color: c, hollow: !r['at SPARTAN'], bold: best,
      note: i === 0 ? 'baseline' : r['gain lo'] > 0 ? 'adds' : 'n.s.',
      values: { auc: { v: r.AUC, lo: r['AUC lo'], hi: r['AUC hi'] }, gain: { v: r['gain over loading'], lo: r['gain lo'], hi: r['gain hi'] } },
      tip: [r.rung, `inputs: ${r.inputs}`, `AUC ${f3(r.AUC)} (95 % ${f3(r['AUC lo'])} to ${f3(r['AUC hi'])})`,
        `gain over loading ${signed(r['gain over loading'], 3)} (${signed(r['gain lo'], 3)} to ${signed(r['gain hi'], 3)})`, `average precision ${f3(r['average precision'])}`, r['at SPARTAN'] ? 'usable at SPARTAN' : 'needs HIPS: circular at SPARTAN'],
    }
  })
  return (
    <ChartFrame id="xflag-ladder" source={source('Ladder')} provenance={runNote(d)}
      title="Ablation ladder — what each input adds to the flag"
      subtitle={`The flag (gradient boosting, held-out IMPROVE sites) refitted on growing sets of inputs. Left: how well it separates impacted filters (AUC, 0.5 = coin toss; the line is loading alone). Right: the gain over measured loading alone on the same filters. Whiskers are 95 % intervals from resampling held-out sites; a rung earns its inputs only if its gain interval clears zero, marked "adds" on the right.`}
      controls={rungs.picker}
      tip={<>Hollow marks need HIPS, which is half of the FTIR–HIPS disagreement at SPARTAN. "Spectrum size" is the mean raw absorbance above 1425 cm⁻¹ and the C–H band area: a loading proxy read off the spectrum itself. Band features are the Section 5 set (Si–O, PTFE residue, carbonyl, C–H, O–H/N–H and their ratios).</>}>
      <Forest rows={rows} labelWidth={200} noteWidth={56}
        panels={[{ key: 'auc', label: 'AUC, held-out sites', ref: d.ladder[0]?.AUC, fmt: f2 }, { key: 'gain', label: 'AUC gain over loading alone', ref: 0, fmt: (v) => signed(v) }]} />
      <Note>★ = the best rung usable at SPARTAN, used for the blind-spot map and the missed-failure tables.</Note>
      <FixedSetNote what="the workbook's held-out IMPROVE filters and every SPARTAN PM2.5 filter it scored" />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 3. distance at fixed loading
function WithinLoading({ d }: { d: FlagData }) {
  const lg = useLegend()
  const scores = [...new Set(d.within.map((w) => w.score))]
  const color: Record<string, string> = { [LOADING]: IMPROVE_GREY, [DISTANCE]: '#7c3aed' }
  const series = scores.filter((s) => lg.show(s)).map((s) => ({
    id: s, label: s, group: s, color: color[s] ?? INK.good,
    points: d.within.filter((w) => w.score === s).map((w) => ({ x: w['loading decile'], y: w.AUC,
      tip: [s, `loading decile ${w['loading decile']} (median ${Math.round(w['median PM2.5 µg'])} µg PM2.5)`, `AUC ${f3(w.AUC)}`, `${w.n.toLocaleString()} filters, ${pct(w['impacted share'])} impacted`] })),
  }))
  const deciles = d.within.filter((w) => w.score === scores[0])
  return (
    <ChartFrame id="xflag-within" source={source('WithinLoading')} provenance={runNote(d)}
      title="Distance at fixed loading — does T²/Q rank errors among filters of similar loading?"
      subtitle="IMPROVE split into ten loading deciles (weighed PM2.5 µg per filter). Inside each decile, the AUC of each score for telling impacted from fine filters. A score that only tracks loading falls to 0.5 here (dashed line); one that sees something else stays above it."
      tip={<>Loading alone still varies a little within a decile, so its AUC is not exactly 0.5. The table below the chart shows how the base rate of impacted filters changes with loading.</>}>
      <SweepPlot series={series} xDomain={[0.5, 10.5]} yLabel="AUC inside the decile" xLabel="Loading decile (1 = lightest)" reference={0.5} height={300} dim={lg.dim}
        xFormat={(v) => String(Math.round(+v))} yFormat={(v) => f2(v)} />
      <Legend {...lg.props} items={scores.map((s) => ({ label: s, color: color[s] ?? INK.good, shape: 'line' as const }))} />
      <table className="census-table" style={{ marginTop: 6 }}>
        <thead><tr><th>Loading decile</th>{deciles.map((w) => <th key={w['loading decile']}>{w['loading decile']}</th>)}</tr></thead>
        <tbody>
          <tr><td>Median PM2.5 µg</td>{deciles.map((w) => <td key={w['loading decile']}>{Math.round(w['median PM2.5 µg'])}</td>)}</tr>
          <tr><td>Impacted share</td>{deciles.map((w) => <td key={w['loading decile']}>{pct(w['impacted share'])}</td>)}</tr>
        </tbody>
      </table>
      <FixedSetNote what="the workbook's held-out IMPROVE filters and every SPARTAN PM2.5 filter it scored" />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 4. error against coverage
const METRICS = ['RMSE kept', 'impacted let through', 'impacted among kept', 'bias kept'] as const
type Metric = typeof METRICS[number]
const METRIC_LABEL: Record<Metric, string> = { 'RMSE kept': 'RMSE among kept filters (µg)', 'impacted let through': 'share of impacted filters kept', 'impacted among kept': 'impacted share among kept', 'bias kept': 'mean error among kept (µg)' }
function Coverage({ d }: { d: FlagData }) {
  useTheme()
  const scores = Object.keys(d.coverage)
  const rungs = d.ladder.map((r) => r.rung)
  const keepShown = new Set([LOADING, DISTANCE, d.settings.BEST, 'oracle (true error)', 'random'])
  const lg = useLegend(scores.filter((s) => !keepShown.has(s)))
  const [metric, setMetric] = useSearchParam<Metric>('xflag-cover.metric', 'impacted let through', METRICS)
  const rungColours = useRungColours()
  const colorOf = (s: string) => s === 'oracle (true error)' ? INK.fit : s === 'random' ? INK.identity : rungColours.color(rungs.indexOf(s), rungs.length)
  const series = scores.filter((s) => lg.show(s)).map((s) => {
    const c = d.coverage[s]
    return { id: s, label: s, group: s, color: colorOf(s), dash: s === 'oracle (true error)' || s === 'random' ? '5 4' : undefined, marks: false,
      width: s === d.settings.BEST ? 2.6 : 1.6,
      points: c['kept share'].map((k, i) => ({ x: k, y: c[metric][i], tip: [s, `keep ${pct(k)} of filters`, `RMSE ${f2(c['RMSE kept'][i])} µg · bias ${signed(c['bias kept'][i])} µg`,
        `${pct(c['impacted let through'][i])} of impacted kept · ${pct(c['impacted among kept'][i])} of kept are impacted`] })) }
  })
  const fmt = metric === 'RMSE kept' || metric === 'bias kept' ? (v: number) => f2(v) : (v: number) => pct(v)
  return (
    <ChartFrame id="xflag-coverage" source={source('Coverage')} provenance={runNote(d)}
      title="Error against coverage — what is left after setting flagged filters aside"
      subtitle="Each score ranks the held-out IMPROVE filters from least to most risky; keep the lowest-risk share and read the error left among them. Reading right to left, a useful flag bends toward the oracle (keeps by the true error, the best any flag could do); one that tracks the random line only throws data away."
      tip={<>Lower error among kept filters is selective retention, not a better EC estimator: flagged filters still need an answer. "Missed failures" is the share of impacted filters still kept at that coverage. RMSE is in µg per filter, so it rewards setting heavy filters aside: on RMSE, loading alone looks best even though it is the weakest at finding impacted filters.</>}
      controls={<>{rungColours.picker}<Segmented label="Read" value={metric} options={METRICS} onChange={setMetric} optionLabel={(m) => ({ 'RMSE kept': 'RMSE', 'impacted let through': 'Missed failures', 'impacted among kept': 'Failure rate kept', 'bias kept': 'Bias' }[m])} /></>}>
      <SweepPlot series={series} xDomain={[1, 0.3]} yLabel={METRIC_LABEL[metric]} xLabel="Share of filters kept" height={340} dim={lg.dim} yFormat={fmt} xFormat={(v) => pct(+v)}
        xMarkers={[{ at: 1 - d.settings.FLAG_RATE, label: `flag ${pct(d.settings.FLAG_RATE)}` }]} reference={metric === 'bias kept' ? 0 : undefined} />
      <Legend {...lg.props} items={scores.map((s) => ({ label: s, color: colorOf(s), shape: s === 'oracle (true error)' || s === 'random' ? 'dashed' as const : 'line' as const }))} note="click to show or hide" />
      <FixedSetNote what="the workbook's held-out IMPROVE filters and every SPARTAN PM2.5 filter it scored" />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 5. who gets through
const PROFILE: { key: string; label: string; kind: 'pct100' | 'share' | 'ug' }[] = [
  { key: 'loading percentile (median)', label: 'Loading pct', kind: 'pct100' },
  { key: 'distance percentile (median)', label: 'Distance pct', kind: 'pct100' },
  { key: 'OC/BC percentile (median)', label: 'OC/BC pct', kind: 'pct100' },
  { key: 'under-predicted share', label: 'Under-predicted', kind: 'share' },
  { key: 'urban share', label: 'Urban sites', kind: 'share' },
  { key: 'smoke share (nsK top 5 %)', label: 'Smoke (nsK top 5 %)', kind: 'share' },
  { key: 'TOR EC µg (median)', label: 'TOR EC µg', kind: 'ug' },
  { key: '|error| µg (median)', label: '|error| µg', kind: 'ug' },
]
function WhoGetsThrough({ d }: { d: FlagData }) {
  const [list, setList] = useSearchParam<'on' | 'off'>('xflag-missed.list', 'off', ['on', 'off'])
  const open = list === 'on'
  const ugMax = Object.fromEntries(PROFILE.filter((p) => p.kind === 'ug').map((p) => [p.key, d3.max(d.missed_summary, (r) => r[p.key] as number) ?? 1]))
  const bar = (p: typeof PROFILE[number], v: number, color: string) => {
    const frac = p.kind === 'pct100' ? v / 100 : p.kind === 'share' ? v : v / ugMax[p.key]
    const text = p.kind === 'pct100' ? Math.round(v).toString() : p.kind === 'share' ? pct(v) : f2(v)
    return <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 96 }}>
      <div style={{ flex: '0 0 56px', height: 9, background: INK.empty, borderRadius: 2 }}><div style={{ width: `${Math.max(0, Math.min(1, frac)) * 100}%`, height: '100%', background: color, borderRadius: 2 }} /></div>
      <span style={{ fontVariantNumeric: 'tabular-nums', fontSize: 11 }}>{text}</span>
    </div>
  }
  const top = d.missed_top
  return (
    <ChartFrame id="xflag-missed" source={source('WhoGetsThrough')} provenance={runNote(d)}
      title="Who gets through — the failures the flag misses, beside those it catches"
      subtitle={`Held-out IMPROVE filters split by the "${d.settings.BEST}" flag at a ${pct(d.settings.FLAG_RATE)} flag rate: impacted and flagged (caught), impacted but kept (missed), flagged but fine (false alarm), and the rest. Percentiles are within IMPROVE (50 = typical).`}
      tip={<>OC/BC here is TOR OC over HIPS BC, as Addis is measured; smoke = non-soil potassium (K − 0.6 Fe) in IMPROVE's top 5 %. Urban = Ann's six urban IMPROVE sites.</>}
      controls={<Toggle label={`List the ${top.length} largest missed failures`} checked={open} onChange={(v) => setList(v ? 'on' : 'off')} />}>
      <div style={{ overflowX: 'auto' }}>
        <table className="census-table">
          <thead><tr><th>Group</th><th>Filters</th>{PROFILE.map((p) => <th key={p.key}>{p.label}</th>)}</tr></thead>
          <tbody>{d.missed_summary.map((r) => { const c = CAT_COLOR[r.category as string]
            return <tr key={r.category as string}><td style={{ color: c === CAT_COLOR.fine ? INK.muted : c, fontWeight: 600 }}>{r.category}</td><td>{(r.filters as number).toLocaleString()} <span style={{ color: INK.muted }}>({pct(r.share)})</span></td>
              {PROFILE.map((p) => <td key={p.key}>{bar(p, r[p.key] as number, c === CAT_COLOR.fine ? IMPROVE_GREY : c)}</td>)}</tr> })}</tbody>
        </table>
      </div>
      {open && <div style={{ overflowX: 'auto', marginTop: 10 }}><table className="census-table">
        <thead><tr><th>Site</th><th>Filter</th><th>Date</th><th>TOR EC µg</th><th>FTIR EC µg</th><th>Error µg</th><th>Loading pct</th><th>Distance pct</th><th>Flag risk</th></tr></thead>
        <tbody>{top.map((r) => <tr key={String(r.filter_id)}><td>{r.site}{r.urban ? ' (urban)' : ''}</td><td>{r.filter_id}</td><td>{r.date}</td><td>{f2(r['TOR EC µg'])}</td><td>{f2(r['FTIR EC µg'])}</td>
          <td style={{ color: (r['error µg'] as number) < 0 ? INK.positive : INK.negative }}>{signed(r['error µg'] as number)}</td><td>{Math.round(r['loading percentile'] as number)}</td><td>{Math.round(r['distance percentile'] as number)}</td><td>{f2(r.risk)}</td></tr>)}</tbody>
      </table></div>}
      <FixedSetNote what="the workbook's held-out IMPROVE filters and every SPARTAN PM2.5 filter it scored" />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 6. per held-out site
function SiteKeepChart({ d }: { d: FlagData }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const [which, setWhich] = useSearchParam<'Most flagged' | 'Most missed' | 'All sites'>('xflag-sites.show', 'Most flagged', ['Most flagged', 'Most missed', 'All sites'])
  const sorted = [...d.site_keep].sort(which === 'Most missed' ? (a, b) => b['missed failures'] - a['missed failures'] : (a, b) => a['kept share'] - b['kept share'])
  const rows = which === 'All sites' ? sorted : sorted.slice(0, 25)
  const rowH = 15, m = { top: 8, right: 120, bottom: 44, left: 70 }
  const iw = Math.max(260, width - m.left - m.right), ih = rows.length * rowH
  const x = d3.scaleLinear().domain([0, d3.max(rows, (r) => Math.max(r['RMSE all'], r['RMSE kept'] ?? 0)) ?? 1]).nice().range([0, iw])
  return (
    <ChartFrame id="xflag-sites" source={source('SiteKeepChart')} provenance={runNote(d)}
      title="Per held-out site — how much each site loses to the flag, and what it buys"
      subtitle={`Each IMPROVE site (held out when its filters were predicted): RMSE against TOR over all its filters (grey) and over the filters the flag keeps (coloured), with the share kept on the right. A flag that empties a few sites can look good overall while telling those sites nothing.`}
      controls={<Segmented label="Sites" value={which} options={['Most flagged', 'Most missed', 'All sites'] as const} onChange={setWhich} />}>
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={iw + m.left + m.right} height={ih + m.top + m.bottom} fontFamily={FONT.family}>
          <g transform={`translate(${m.left},${m.top})`}>
            <XAxis scale={x} y={ih} label="RMSE against TOR (µg per filter)" labelDy={36} tickCount={6} />
            {rows.map((r, i) => {
              const cy = i * rowH + rowH / 2, kept = r['RMSE kept']
              return <g key={r.site} onMouseEnter={(e) => tip.show(e, [`${r.site}${r.urban ? ' (urban)' : ''}`, `${r.filters} filters, ${pct(r['impacted share'])} impacted`, `kept ${pct(r['kept share'])}`,
                `RMSE ${f2(r['RMSE all'])} → ${f2(kept)} µg`, `${r['missed failures']} missed failures`])} onMouseLeave={tip.hide}>
                <rect x={-m.left} y={cy - rowH / 2} width={iw + m.left + m.right} height={rowH} fill={i % 2 ? 'transparent' : INK.zebra} />
                <text x={-6} y={cy} dy="0.32em" textAnchor="end" fontSize={10} fill={r.urban ? INK.negative : INK.text}>{r.site}</text>
                {kept != null && <line x1={x(r['RMSE all'])} x2={x(kept)} y1={cy} y2={cy} stroke={INK.axis} strokeWidth={1} />}
                <circle cx={x(r['RMSE all'])} cy={cy} r={3.5} fill={IMPROVE_GREY} />
                {kept != null && <circle cx={x(kept)} cy={cy} r={3.5} fill={INK.good} />}
                <text x={iw + 8} y={cy} dy="0.32em" fontSize={10} fill={INK.muted}>{pct(r['kept share'])} kept · {r['missed failures']} missed</text>
              </g>
            })}
          </g>
        </svg>
        {tip.node}
      </div>
      <Legend items={[{ label: 'all filters', color: IMPROVE_GREY }, { label: 'filters the flag keeps', color: INK.good }, { label: 'urban site (name in red)', color: INK.negative, shape: 'square' }]} />
      <FixedSetNote what="the workbook's held-out IMPROVE filters and every SPARTAN PM2.5 filter it scored" />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 7. SPARTAN sites
function SpartanSites({ d }: { d: FlagData }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const [sortBy, setSortBy] = useSearchParam<'Flagged' | 'Loading' | 'Distance'>('xflag-spartan.sort', 'Flagged', ['Flagged', 'Loading', 'Distance'])
  const sites = useMemo(() => d3.groups(spartanRows(d), (r) => r.code).map(([code, rs]) => ({
    code, site: rs[0].site, n: rs.length, flagged: d3.mean(rs, (r) => +r.flagged)!, beyond: d3.mean(rs, (r) => +r.beyond)!,
    load: d3.median(rs.filter((r) => r.load != null), (r) => r.load!) ?? null, dist: d3.median(rs.filter((r) => r.dist != null), (r) => r.dist!) ?? null,
    risk: d3.median(rs, (r) => r.risk)!,
  })).filter((s) => s.n >= 10), [d])
  const key = sortBy === 'Flagged' ? 'flagged' : sortBy === 'Loading' ? 'load' : 'dist'
  const rows = [...sites].sort((a, b) => ((b[key] as number) ?? -1) - ((a[key] as number) ?? -1))
  const rowH = 17, m = { top: 26, right: 16, bottom: 10, left: 170 }
  const colW = Math.max(110, (Math.max(520, width) - m.left - m.right) / 3)
  const cols: { key: 'flagged' | 'load' | 'dist'; label: string; dom: [number, number]; fmt: (v: number) => string; extra?: (s: typeof rows[number]) => number }[] = [
    { key: 'flagged', label: `flagged (risk ≥ the IMPROVE ${pct(d.settings.FLAG_RATE)} cut)`, dom: [0, Math.max(0.2, d3.max(rows, (r) => r.flagged) ?? 1)], fmt: pct },
    { key: 'load', label: 'median loading pct (bar beyond: heavier than any IMPROVE)', dom: [0, 100], fmt: (v) => String(Math.round(v)), extra: (s) => s.beyond },
    { key: 'dist', label: 'median distance pct', dom: [0, 100], fmt: (v) => String(Math.round(v)) },
  ]
  return (
    <ChartFrame id="xflag-spartan" source={source('SpartanSites')} provenance={runNote(d)}
      title="SPARTAN sites through the same flag — which ones it would mark"
      subtitle={`The best SPARTAN-usable flag ("${d.settings.BEST}"), refitted on all of IMPROVE, applied to every SPARTAN PM2.5 filter (sites with at least 10). Flagged share uses the risk cut that flags ${pct(d.settings.FLAG_RATE)} of IMPROVE; loading and distance are percentiles in IMPROVE, as on the map. There is no TOR here: a high share means "looks like IMPROVE's failures", not "is wrong".`}
      controls={<Segmented label="Sort by" value={sortBy} options={['Flagged', 'Loading', 'Distance'] as const} onChange={setSortBy} />}>
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={m.left + 3 * colW + m.right} height={m.top + rows.length * rowH + m.bottom} fontFamily={FONT.family}>
          {cols.map((c, ci) => <text key={c.key} x={m.left + ci * colW} y={14} fontSize={10} fill={INK.muted}>{c.label}</text>)}
          <g transform={`translate(0,${m.top})`}>
            {rows.map((s, i) => {
              const cy = i * rowH + rowH / 2, addis = s.code === 'ETAD'
              return <g key={s.code} onMouseEnter={(e) => tip.show(e, [`${s.site} (${s.code.replace('*', '')})`, `${s.n} filters · ${pct(s.flagged)} flagged · median risk ${f2(s.risk)}`,
                `median loading pct ${f2(s.load)} · ${pct(s.beyond)} heavier than any IMPROVE filter`, `median distance pct ${f2(s.dist)}`])} onMouseLeave={tip.hide}>
                <rect x={0} y={cy - rowH / 2} width={m.left + 3 * colW} height={rowH} fill={i % 2 ? 'transparent' : INK.zebra} />
                <text x={m.left - 8} y={cy} dy="0.32em" textAnchor="end" fontSize={10.5} fontWeight={addis ? 700 : 400} fill={addis ? ADDIS_ALL : INK.text}>{s.site.length > 26 ? `${s.site.slice(0, 25)}…` : s.site}</text>
                {cols.map((c, ci) => {
                  const v = s[c.key] as number | null, sc = d3.scaleLinear().domain(c.dom).range([0, colW - 46])
                  const ex = c.extra?.(s) ?? 0
                  return <g key={c.key} transform={`translate(${m.left + ci * colW},0)`}>
                    <rect x={0} y={cy - 4} width={colW - 46} height={8} fill={INK.empty} />
                    {v != null && <rect x={0} y={cy - 4} width={sc(Math.min(c.dom[1], v))} height={8} fill={addis ? ADDIS_ALL : c.key === 'flagged' ? INK.negative : '#7c3aed'} fillOpacity={0.75} />}
                    {ex > 0 && <rect x={colW - 46} y={cy - 4} width={Math.max(2, ex * 30)} height={8} fill={INK.text} fillOpacity={0.55} />}
                    <text x={colW - 46 + (ex > 0 ? Math.max(2, ex * 30) + 3 : 3)} y={cy} dy="0.32em" fontSize={9.5} fill={INK.muted}>{v == null ? '—' : c.fmt(v)}</text>
                  </g>
                })}
              </g>
            })}
          </g>
        </svg>
        {tip.node}
      </div>
      <Note>A dash under loading means the site has no weighed PM2.5 for most filters: the flag still scores those filters, it just cannot use their mass.</Note>
      <FixedSetNote what="the workbook's held-out IMPROVE filters and every SPARTAN PM2.5 filter it scored" />
    </ChartFrame>
  )
}
