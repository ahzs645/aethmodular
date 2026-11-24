import { useMemo } from 'react'
import { ChartFrame, ExplainOnly, Note, Segmented } from '@/components/ChartFrame'
import { Select } from '@/components/Controls'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { FindingsList } from '@/components/Findings'
import { ScrubField } from '@/components/ScrubField'
import { SweepPlot, histogramSeries, type SweepMarker, type SweepSeries } from '@/components/SweepPlot'
import { f2, f3, fmt } from '@/lib/format'
import { twoZone } from '@/lib/hipsPattern'
import { opticalDepth } from '@/lib/hipsLinearise'
import { siteInScope } from '@/lib/hipsRun'
import { useMac } from '@/lib/mac'
import { linearised, useHipsView } from '@/lib/plotUse'
import { SidebarControls } from '@/lib/pageSidebar'
import { pearson } from '@/lib/stats'
import { INK } from '@/lib/theme'
import { RESEARCH_SERIES } from '@/lib/palettes'
import { useSearchParam } from '@/lib/url'
import { sourceOf, useCounted, useXp, type XpBase } from './shared'

/**
 * Kyan Shlipak's manual loading compensation for the MA350 aethalometer (EC_BC deck; src/data/processors/site_corrections.py),
 * with k free to move. The export (gallery/data/export_aeth_loading_k.py) evaluates BCc(k) = BC1 / (1 − k·ATN1) on a grid of k
 * from the raw minutes; this page interpolates between grid points, so every chart follows the k chosen in Page options.
 */

interface Hist { lo: number; step: number; counts: number[]; n: number }
type Num = number | null
interface ChannelData {
  corr: Num[]; left_out: number[]; step: Num[]; step_q25: Num[]; step_q75: Num[]; advances_used: number
  hourly: Num[][]; hourly_instrument: Num[]; corr_instrument: Num
  k_median: number; k_q25: number; k_q75: number
  atn: Hist; atn_over3: number; k_hist: Hist; den_instrument: Hist
  filters: { bc_k: Num[][]; bc_instrument: Num[]; bc1: Num[]; bc2: Num[]; atn: Num[] }
}
interface SiteData {
  label: string; filter_site: string; minutes: number; span: [string, string]; advances: number; flow_ratio: Hist | null
  channels: Record<string, ChannelData>
  filters: { id: string[]; date: string[]; hours: number[]; ec: Num[]; hips: Num[]; volume?: Num[] }
}
interface LoadingData extends XpBase {
  settings: { k_grid: number[]; atn_min: number; floor: number; kyan: { median: number; bc_atn: number; residual: number }; channels: string[]; hips_mac: number }
  sites: Record<string, SiteData>
}

const source = sourceOf('AethLoadingStudy')
const COMMAND = 'uv run python gallery/data/export_aeth_loading_k.py'
const MODES = ['instrument', 'median', 'bcatn', 'step', 'residual', 'custom'] as const
type Mode = (typeof MODES)[number]
const MODE_LABEL: Record<Mode, string> = {
  instrument: 'The instrument’s own K, minute by minute',
  median: 'Median of the instrument’s K (ATN1 > 3)',
  bcatn: 'k that leaves BC uncorrelated with ATN1',
  step: 'k that removes the tape-advance step',
  residual: 'k that leaves FTIR EC residuals uncorrelated with ATN1',
  custom: 'Your own k',
}
const SHORT: Record<Mode, string> = { instrument: 'instrument K', median: 'median K', bcatn: 'BC–ATN', step: 'tape step', residual: 'FTIR residual', custom: 'your k' }
const SITE_LABEL: Record<string, string> = { 'ETAD-JACROS': 'Addis Ababa', USPA: 'Pasadena' }

// ---------------------------------------------------------------------------------------------- numbers
const finite = (v: Num | undefined): v is number => v != null && Number.isFinite(v)
/** linear interpolation of `vals` on `grid` at k; null where either neighbour is missing */
function interp(grid: number[], vals: Num[], k: number): number | null {
  if (!Number.isFinite(k)) return null
  if (k <= grid[0]) return finite(vals[0]) ? vals[0] : null
  for (let i = 1; i < grid.length; i++) {
    if (k <= grid[i]) {
      const a = vals[i - 1], b = vals[i]
      if (!finite(a) || !finite(b)) return null
      const t = (k - grid[i - 1]) / (grid[i] - grid[i - 1])
      return a + t * (b - a)
    }
  }
  const last = vals[vals.length - 1]
  return finite(last) ? last : null
}
/** the k where |v| is smallest, refined by a parabola through the best grid point and its neighbours */
function argminAbs(grid: number[], vals: Num[]): number | null {
  let best = -1
  vals.forEach((v, i) => { if (finite(v) && (best < 0 || Math.abs(v) < Math.abs(vals[best] as number))) best = i })
  if (best < 0) return null
  // a sign change next to the minimum: the root by linear interpolation
  for (const j of [best - 1, best + 1]) {
    const a = vals[Math.min(best, j)], b = vals[Math.max(best, j)]
    if (j >= 0 && j < vals.length && finite(a) && finite(b) && a * b < 0) {
      const i0 = Math.min(best, j)
      return grid[i0] + (grid[i0 + 1] - grid[i0]) * (a / (a - b))
    }
  }
  return grid[best]
}
function ols(xs: number[], ys: number[]) {
  const n = xs.length
  if (n < 3) return null
  const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n
  let sxx = 0, sxy = 0, syy = 0
  for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; sxx += dx * dx; sxy += dx * dy; syy += dy * dy }
  const slope = sxy / sxx, intercept = my - slope * mx, r = sxy / Math.sqrt(sxx * syy)
  return { n, slope, intercept, r, r2: r * r, resid: ys.map((y, i) => y - (intercept + slope * xs[i])) }
}

// ---------------------------------------------------------------------------------------------- shared state
function useLoading(d: LoadingData) {
  const siteKeys = Object.keys(d.sites)
  const [siteKey, setSite] = useSearchParam<string>('xk.site', siteKeys[0] ?? 'ETAD-JACROS', siteKeys)
  const [ch, setCh] = useSearchParam<string>('xk.ch', 'IR', d.settings.channels)
  const [mode, setMode] = useSearchParam<Mode>('xk.mode', 'bcatn', MODES)
  const [kText, setK] = useSearchParam<string>('xk.k', '0.012')
  const [yKey, setY] = useSearchParam<'ec' | 'hips'>('xk.y', 'ec', ['ec', 'hips'])
  const site = d.sites[siteKey] ?? d.sites[siteKeys[0]]
  const cd = site.channels[ch] ?? site.channels.IR
  const grid = d.settings.k_grid
  const ids = site.filters.id
  const idx = useMemo(() => ids.map((_, i) => i), [ids])
  const { counted, nOut } = useCounted(idx, (i) => ids[i])
  // FTIR residuals against each filter's mean ATN1, per grid k, over the counted filters
  const residualCorr = useMemo(() => grid.map((_, g) => {
    const xs: number[] = [], ys: number[] = [], atn: number[] = []
    for (const i of counted) {
      const bc = cd.filters.bc_k[i]?.[g], ec = site.filters.ec[i], a = cd.filters.atn[i]
      if (finite(bc) && finite(ec) && finite(a)) { xs.push(bc); ys.push(ec); atn.push(a) }
    }
    const f = ols(xs, ys)
    return f ? pearson(f.resid, atn) : null
  }), [grid, counted, cd, site])
  const presets: Record<Mode, number> = {
    instrument: NaN,
    median: cd.k_median,
    bcatn: argminAbs(grid, cd.corr) ?? NaN,
    step: argminAbs(grid, cd.step.map((v) => (finite(v) ? v - 1 : null))) ?? NaN,
    residual: argminAbs(grid, residualCorr) ?? NaN,
    custom: Math.min(0.02, Math.max(0, Number(kText) || 0)),
  }
  const k = presets[mode]
  /** a filter's aethalometer BC under a mode (the instrument's own BCc for 'instrument') */
  const bcOf = (i: number, m: Mode = mode) => m === 'instrument' ? cd.filters.bc_instrument[i] : interp(grid, cd.filters.bc_k[i], presets[m])
  return { d, site, siteKey, setSite, ch, setCh, mode, setMode, k, setK, kText, yKey, setY, cd, grid, counted, nOut, residualCorr, presets, bcOf, siteKeys }
}
type Loading = ReturnType<typeof useLoading>

function Options({ m }: { m: Loading }) {
  return <SidebarControls title="Loading k" tabNote={m.mode === 'instrument' ? 'instrument K' : `k ${fmt(m.k, 4)}`}>
    <div role="group" aria-label="Loading k" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <Segmented label="Site" value={m.siteKey} options={m.siteKeys} optionLabel={(s) => SITE_LABEL[s] ?? s} onChange={m.setSite} />
      <Segmented label="Channel" value={m.ch} options={m.d.settings.channels} onChange={m.setCh} />
      <Select label="k" value={m.mode} options={[...MODES]} optionLabel={(o) => MODE_LABEL[o as Mode]} onChange={(v) => m.setMode(v as Mode)}
        title="Which loading constant every chart on this page uses" />
      {m.mode === 'custom' && <span className="control" title="k in BCc = BC1 / (1 − k·ATN1), between 0 and 0.02">k <ScrubField label="Loading constant k" value={m.presets.custom} min={0} max={0.02} step={0.0001} width={90}
        format={(v) => fmt(v, 4)} onChange={(v) => m.setK(String(+v.toFixed(5)))} /></span>}
    </div>
  </SidebarControls>
}

export function AethLoadingStudy() {
  const { data, state } = useXp<LoadingData>('aeth_loading_k', 'Aethalometer loading k')
  if (!data) return <>{state ?? <Note>Run <code>{COMMAND}</code>.</Note>}</>
  return <Page d={data} />
}

function Page({ d }: { d: LoadingData }) {
  const m = useLoading(d)
  return <>
    <Options m={m} />
    <Explain m={m} />
    <Criteria m={m} />
    <Denominator m={m} />
    <Cross m={m} />
    <Table m={m} />
    <TimeSeries m={m} />
    <Diurnal m={m} />
    <SameAsHips m={m} />
  </>
}

const runLine = (m: Loading) => `export_aeth_loading_k.py · ${m.site.label} · ${m.site.minutes.toLocaleString()} minutes, ${m.site.span[0]} to ${m.site.span[1]} · ${m.ch}`
const kLabel = (m: Loading) => m.mode === 'instrument' ? 'the instrument’s own K' : `k ${fmt(m.k, 4)} (${SHORT[m.mode]})`

// ---------------------------------------------------------------------------------------------- explainer
function Explain({ m }: { m: Loading }) {
  const p = m.presets, cd = m.cd
  const flow = m.site.flow_ratio
  const flowMedian = useMemo(() => {
    if (!flow) return null
    let acc = 0
    for (let i = 0; i < flow.counts.length; i++) { acc += flow.counts[i]; if (acc >= flow.n / 2) return flow.lo + (i + 0.5) * flow.step }
    return null
  }, [flow])
  return (
    <ExplainOnly>
      <ChartFrame id="xk-findings" title="How the MA350 corrects for loading, and what Kyan changed" exportable={false}>
        <FindingsList note={<>{runLine(m)}. Kyan Shlipak’s numbers for Jacros IR (EC_BC deck): median K {m.d.settings.kyan.median}, BC–ATN {m.d.settings.kyan.bc_atn}, FTIR residual {m.d.settings.kyan.residual}. The tape-advance check is added here; it is not in his deck.</>}>
          <li><strong>The tape darkens.</strong> ATN = 100·ln(I₀/I). BC comes from how fast ATN rises: b<sub>ATN</sub> = (A/Q)·ΔATN/Δt, BC1 = b<sub>ATN</sub>/σ.</li>
          <li><strong>The loading effect.</strong> On a dark spot the same air raises ATN less, so BC1 reads low as ATN1 climbs, then jumps back up when the tape advances to a clean spot.</li>
          <li><strong>The MA350’s fix.</strong> BCc = BC1 / (1 − k·ATN1). The instrument solves k every minute from a second spot that takes less of the flow (target 75:25, Flow1/Flow2 = 3).</li>
          <li><strong>Why that fails here.</strong> {flowMedian != null ? <>The median Flow1/Flow2 at {SITE_LABEL[m.siteKey] ?? m.siteKey} is {f2(flowMedian)}. </> : null}When the two spots load almost alike, the live K is noise: some minutes get a negative denominator (BC changes sign) or one above 1 (BC shrinks).</li>
          <li><strong>Kyan’s fix: one k for the whole record.</strong> Either the median of the instrument’s K ({fmt(p.median, 4)} here), the k that leaves BC uncorrelated with ATN1 ({fmt(p.bcatn, 4)}, no FTIR used), or the k that leaves FTIR EC residuals uncorrelated with ATN1 ({fmt(p.residual, 4)}). Below ATN1 {m.d.settings.atn_min} the minute keeps BC1.</li>
          <li><strong>A check that needs no other instrument.</strong> At a tape advance the air does not change, so the right k makes BC the same just before and just after. Here that k is {fmt(p.step, 4)}, from {cd.advances_used.toLocaleString()} advances.</li>
          <li><strong>Same form as the HIPS correction, with one difference.</strong> HIPS linearised is 1/(1 − τ/τ∞). The aethalometer can find its own k because one spot keeps loading under the same air; HIPS reads each filter once, so its ceiling needs an outside reference.</li>
        </FindingsList>
      </ChartFrame>
    </ExplainOnly>
  )
}

// ---------------------------------------------------------------------------------------------- criteria vs k
function presetMarkers(m: Loading, skip: Mode[] = []): SweepMarker[] {
  const colors: Partial<Record<Mode, string>> = { median: RESEARCH_SERIES[1], bcatn: RESEARCH_SERIES[0], step: RESEARCH_SERIES[2], residual: RESEARCH_SERIES[3] }
  const out: SweepMarker[] = (['median', 'bcatn', 'step', 'residual'] as Mode[]).filter((k) => !skip.includes(k) && Number.isFinite(m.presets[k]))
    .map((k) => ({ at: m.presets[k], label: SHORT[k], color: colors[k] }))
  if (m.mode === 'custom') out.push({ at: m.k, label: 'your k', color: INK.accent })
  return out
}

function Criteria({ m }: { m: Loading }) {
  const g = m.grid, cd = m.cd
  const line = (id: string, label: string, color: string, ys: Num[], lo?: Num[], hi?: Num[]): SweepSeries => ({
    id, label, color, marks: false, points: g.map((k, i) => ({ x: k, y: ys[i], lo: lo?.[i], hi: hi?.[i], tip: [`k ${fmt(k, 4)}`, `${label}: ${fmt(ys[i], 3)}`] })),
  })
  const markers = presetMarkers(m)
  return (
    <ChartFrame id="xk-criteria" source={source('Criteria')} provenance={runLine(m)}
      title="Three ways to choose k, and where each one lands"
      subtitle={`Left: correlation of corrected BC with ATN1 over every minute (Kyan’s test; the instrument’s own BCc gives ${fmt(cd.corr_instrument, 3)}). Middle: BC just after a tape advance over just before (5–15 min each side, median and interquartile band); 1 means the loading effect is gone. Right: correlation of the FTIR EC residuals (OLS of EC on BC) with each filter’s mean ATN1, over ${m.counted.length} filters${m.nOut ? `, ${m.nOut} left out by Filters → excluded` : ''}. The first two use no FTIR.`}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
        <div style={{ flex: '1 1 260px' }}><SweepPlot series={[line('corr', 'r(BC, ATN1)', RESEARCH_SERIES[0], cd.corr)]} xLabel="k" yLabel="r of BC with ATN1 (minutes)" reference={0} xMarkers={markers} height={230} xFormat={(v) => fmt(v, 3)} /></div>
        <div style={{ flex: '1 1 260px' }}><SweepPlot series={[{ ...line('step', 'after ÷ before', RESEARCH_SERIES[2], cd.step, cd.step_q25, cd.step_q75), interval: 'band' }]} xLabel="k" yLabel="BC after ÷ before a tape advance" reference={1} xMarkers={markers} height={230} xFormat={(v) => fmt(v, 3)} /></div>
        <div style={{ flex: '1 1 260px' }}><SweepPlot series={[line('resid', 'r(residual, ATN1)', RESEARCH_SERIES[3], m.residualCorr)]} xLabel="k" yLabel="r of FTIR residual with ATN1 (filters)" reference={0} xMarkers={markers} height={230} xFormat={(v) => fmt(v, 3)} /></div>
      </div>
      <Note>Each vertical line is one way of choosing k; the chosen one drives every other chart (Page options). {m.mode === 'instrument' ? null : <>At k {fmt(m.k, 4)} the correction triples BC once ATN1 reaches {fmt(2 / (3 * m.k), 0)} (1 − k·ATN1 = ⅓).</>}</Note>
    </ChartFrame>
  )
}

// ---------------------------------------------------------------------------------------------- denominators
function Denominator({ m }: { m: Loading }) {
  const cd = m.cd, floor = m.d.settings.floor, atnMin = m.d.settings.atn_min
  // with one k the denominator is a linear map of ATN1, so each ATN1 bin becomes one denominator bin of width k·step
  // (reversed: darker spots, smaller denominators). Minutes at ATN1 ≤ 3 keep BC1 (denominator 1) and are counted apart.
  const ours = useMemo(() => {
    if (m.mode === 'instrument' || !(m.k > 0)) return null
    const { lo, step, counts } = cd.atn
    let total = 0, triple = 0, out = 0, kept = 0
    const binsCounts: number[] = [], binsEdges: number[] = []
    counts.forEach((c, i) => {
      const a0 = lo + i * step, a1 = a0 + step, mid = (a0 + a1) / 2
      total += c
      if (mid <= atnMin) { kept += c; return }
      const den = 1 - m.k * mid
      if (den <= floor) out += c
      if (den < 1 / 3) triple += c
      binsEdges.push(1 - m.k * a0); binsCounts.push(c)
    })
    if (!binsCounts.length) return null
    binsEdges.push(1 - m.k * (lo + counts.length * step))
    return { edges: [...binsEdges].reverse(), counts: [...binsCounts].reverse(), total, triple, out, kept }
  }, [cd, m.k, m.mode, floor, atnMin])
  const inst = cd.den_instrument
  const instShare = useMemo(() => {
    let neg = 0, over1 = 0, triple = 0
    inst.counts.forEach((c, i) => { const v = inst.lo + (i + 0.5) * inst.step; if (v < 0) neg += c; if (v > 1) over1 += c; if (v >= 0 && v < 1 / 3) triple += c })
    return { neg: neg / inst.n, over1: over1 / inst.n, triple: triple / inst.n }
  }, [inst])
  // densities, so bins of different widths compare: share of minutes per 0.01 of the denominator
  const density = (counts: number[], width: number, total: number) => counts.map((c) => c * (0.01 / width) / total)
  const series = [
    histogramSeries({ id: 'inst', label: 'instrument K, minute by minute', color: INK.identity, fill: 0.25,
      edges: Array.from({ length: inst.counts.length + 1 }, (_, i) => inst.lo + i * inst.step), counts: density(inst.counts, inst.step, inst.n) }),
    ...(ours ? [histogramSeries({ id: 'ours', label: kLabel(m), color: INK.accent, fill: 0.4, edges: ours.edges,
      counts: density(ours.counts, m.k * cd.atn.step, ours.total) })] : []),
  ]
  const pct = (v: number) => `${(100 * v).toFixed(1)} %`
  return (
    <ChartFrame id="xk-den" source={source('Denominator')} provenance={runLine(m)}
      title="The denominator 1 − k·ATN1, minute by minute"
      subtitle={`Kyan’s rules for a sane correction: it should not more than triple BC (denominator under ⅓), shrink it (over 1) or flip its sign (under 0). With the instrument’s own K: ${pct(instShare.neg)} of minutes flip sign, ${pct(instShare.over1)} shrink and ${pct(instShare.triple)} more than triple.${ours ? ` With ${kLabel(m)}: ${pct(ours.triple / ours.total)} more than triple and ${pct(ours.out / ours.total)} are past a factor of 10 and left out; ${pct(ours.kept / ours.total)} sit at ATN1 ≤ ${atnMin}, keep BC1 and are not drawn.` : ''}`}>
      <SweepPlot series={series} xLabel="1 − k·ATN1 (1 = no correction)" yLabel="share of minutes per 0.01" xDomain={[-1, 2]} xMarkers={[{ at: 0, label: 'sign flips' }, { at: 1 / 3, label: '× 3' }, { at: 1, label: 'none' }]} height={240} />
    </ChartFrame>
  )
}

// ---------------------------------------------------------------------------------------------- crossplot
function Cross({ m }: { m: Loading }) {
  const f = m.site.filters
  // HIPS BC follows the gallery's MAC and HIPS view (Plot tab), offered while this chart shows HIPS
  const onHips = m.yKey === 'hips'
  const mac = useMac(onHips)
  const view = useHipsView(onHips)
  const from = m.d.settings.hips_mac
  const lin = onHips && view.ceiling != null && siteInScope(m.site.filter_site, view.scope)
  const hipsOf = (i: number): number | null => {
    const v = f.hips[i], V = f.volume?.[i]
    if (!finite(v)) return null
    if (!lin) return mac.bc(v, from)
    // τ from Fabs (= BC × the export's MAC) and the sampled volume; a filter without a volume drops out under the view
    const tau = finite(V) ? opticalDepth(v * from, V) : null
    const l = tau == null ? null : linearised(v, tau, view)
    return l == null ? null : mac.bc(l, from)
  }
  const dropped = lin ? m.counted.filter((i) => finite(f.hips[i]) && hipsOf(i) == null).length : 0
  const yName = m.yKey === 'ec' ? 'FTIR EC (µg/m³)' : `HIPS BC${lin ? ', linearised' : ''} at MAC ${fmt(mac.mac, 1)} (µg/m³)`
  const points: CrossPoint[] = m.counted.flatMap((i) => {
    const x = m.bcOf(i), y = m.yKey === 'ec' ? f.ec[i] : hipsOf(i)
    if (!finite(x) || !finite(y)) return []
    return [{ x, y, color: m.yKey === 'ec' ? INK.accent : RESEARCH_SERIES[1], id: f.id[i], site: m.site.filter_site,
      tip: [`${f.id[i]} · ${f.date[i]}`, `aethalometer ${m.ch} BC: ${f2(x)} µg/m³ (${kLabel(m)})`, `${yName}: ${f2(y)}`, `mean ATN1 ${fmt(m.cd.filters.atn[i], 1)}`] }]
  })
  const fit = ols(points.map((p) => p.x), points.map((p) => p.y))
  return (
    <ChartFrame id="xk-cross" source={source('Cross')} provenance={runLine(m)}
      title={m.yKey === 'ec' ? 'FTIR EC against aethalometer BC on each filter’s window' : 'HIPS BC against aethalometer BC on each filter’s window'}
      subtitle={`Aethalometer ${m.ch} BC averaged over each filter’s own sampling window (half its minutes needed), with ${kLabel(m)}. ${fit ? `OLS, as in Kyan’s deck: ${f3(fit.slope)}x ${fit.intercept >= 0 ? '+' : '−'} ${f2(Math.abs(fit.intercept))}, r ${f3(fit.r)}, n ${fit.n}.` : ''} The fits drawn follow the gallery’s estimator settings. The aethalometer’s cross-section is not a filter MAC, so a slope off 1 is a scale, not an error.${onHips ? ` HIPS BC is Fabs ÷ MAC ${fmt(mac.mac, 1)} (Plot → MAC)${lin ? `, linearised at τ∞ ${fmt(view.ceiling, 2)}${view.scope === 'past' ? ` from τ ${fmt(view.cut, 2)}` : ''} with τ = Fabs × sampled volume ÷ 353 (Plot → HIPS)${dropped ? `; ${dropped} filter${dropped === 1 ? '' : 's'} without a volume or past the ceiling drop out` : ''}` : view.ceiling != null ? `; ${SITE_LABEL[m.siteKey] ?? m.siteKey} is outside the HIPS view’s sites, so as measured` : ', as measured (Plot → HIPS linearises it)'}.` : ''}`}
      controls={<Segmented label="y" value={m.yKey} options={['ec', 'hips'] as const} optionLabel={(o) => (o === 'ec' ? 'FTIR EC' : 'HIPS BC')} onChange={m.setY} />}>
      <Crossplot points={points} title={`${SITE_LABEL[m.siteKey] ?? m.siteKey} · ${m.ch}`} xLabel={`Aethalometer ${m.ch} BC (µg/m³)`} yLabel={yName} maxSide={440} />
    </ChartFrame>
  )
}

// ---------------------------------------------------------------------------------------------- table, as in Kyan's slide 9
function Table({ m }: { m: Loading }) {
  const f = m.site.filters, cd = m.cd
  const rows: { key: string; label: string; k: number | null; bc: (i: number) => Num; stepK?: number }[] = [
    { key: 'instrument', label: 'Instrument BCc (live K)', k: null, bc: (i) => cd.filters.bc_instrument[i] },
    { key: 'bc1', label: 'BC1, no correction', k: 0, bc: (i) => cd.filters.bc1[i] },
    { key: 'bc2', label: 'BC2, no correction', k: null, bc: (i) => cd.filters.bc2[i] },
    ...(['median', 'bcatn', 'step', 'residual'] as Mode[]).map((md) => ({ key: md, label: MODE_LABEL[md], k: m.presets[md], bc: (i: number) => m.bcOf(i, md) })),
    ...(m.mode === 'custom' ? [{ key: 'custom', label: 'Your k', k: m.presets.custom, bc: (i: number) => m.bcOf(i, 'custom') }] : []),
  ]
  const stats = rows.map((r) => {
    const xs: number[] = [], ys: number[] = [], atn: number[] = []
    for (const i of m.counted) {
      const x = r.bc(i), y = f.ec[i], a = cd.filters.atn[i]
      if (finite(x) && finite(y) && finite(a)) { xs.push(x); ys.push(y); atn.push(a) }
    }
    const fit = ols(xs, ys)
    return { ...r, fit, residR: fit ? pearson(fit.resid, atn) : null,
      corr: r.k == null ? (r.key === 'instrument' ? cd.corr_instrument : null) : interp(m.grid, cd.corr, r.k),
      step: r.k == null ? null : interp(m.grid, cd.step, r.k) }
  })
  const bold = (key: string) => key === m.mode || (m.mode === 'instrument' && key === 'instrument')
  return (
    <ChartFrame id="xk-table" source={source('Table')} provenance={runLine(m)}
      title="Every choice of k on the same filters"
      subtitle={`The layout of Kyan’s results table (slide 9), with the tape-advance step added. FTIR EC on aethalometer ${m.ch} BC, OLS, ${m.counted.length} filters with both. The row the page uses is in bold.`}>
      <div className="closeout-table-scroll" role="region" aria-label="k choices" tabIndex={0}>
        <table className="census-table" data-export="text">
          <thead><tr><th>BC used</th><th>k</th><th>Slope (EC/BC)</th><th>Intercept (µg/m³)</th><th>r</th><th>R²</th><th>r(residual, ATN1)</th><th>r(BC, ATN1), minutes</th><th>Tape step</th></tr></thead>
          <tbody>{stats.map((s) => <tr key={s.key} style={bold(s.key) ? { fontWeight: 600 } : undefined}>
            <td>{s.label}</td><td>{s.k == null ? '—' : fmt(s.k, 4)}</td>
            <td>{s.fit ? f2(s.fit.slope) : '—'}</td><td>{s.fit ? f2(s.fit.intercept) : '—'}</td><td>{s.fit ? f2(s.fit.r) : '—'}</td><td>{s.fit ? f2(s.fit.r2) : '—'}</td>
            <td>{finite(s.residR) ? f2(s.residR) : '—'}</td><td>{finite(s.corr) ? f3(s.corr) : '—'}</td><td>{finite(s.step) ? f3(s.step) : '—'}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <Note>Kyan’s Jacros IR table: instrument BCc 0.60x + 0.77 (r 0.83), BC1 1.03x + 0.25, BC2 0.94x + 0.31, BCc with his k 0.65x + 0.14 (r 0.94). His daily means used 9am–9am days; this page uses each filter’s own window, so expect small differences.</Note>
    </ChartFrame>
  )
}

// ---------------------------------------------------------------------------------------------- time series and diurnal
function TimeSeries({ m }: { m: Loading }) {
  const f = m.site.filters, cd = m.cd
  const pts = (get: (i: number) => Num) => m.counted.map((i) => ({ x: new Date(f.date[i]), y: get(i), id: f.id[i], tip: [f.id[i], f.date[i], fmt(get(i), 2)] }))
  const series: SweepSeries[] = [
    { id: 'inst', label: 'instrument BCc', color: INK.identity, points: pts((i) => cd.filters.bc_instrument[i]), marks: false, width: 1.2 },
    ...(m.mode !== 'instrument' ? [{ id: 'k', label: `BC, ${kLabel(m)}`, color: INK.accent, points: pts((i) => m.bcOf(i)), marks: false, width: 1.6 }] : []),
    { id: 'ec', label: 'FTIR EC', color: INK.negative, points: pts((i) => f.ec[i]), marks: true, line: false, r: 2.5 },
  ]
  return (
    <ChartFrame id="xk-time" source={source('TimeSeries')} provenance={runLine(m)}
      title="Filter by filter through the record"
      subtitle={`Aethalometer ${m.ch} BC over each filter’s window, as the instrument reported it and with ${kLabel(m)}, beside FTIR EC. Kyan’s point: the diurnal shape hardly moves, but the filter-day levels do.`}>
      <SweepPlot series={series} xScale="time" xLabel="filter start" yLabel="µg/m³" height={260} />
    </ChartFrame>
  )
}

function Diurnal({ m }: { m: Loading }) {
  const cd = m.cd
  const hours = Array.from({ length: 24 }, (_, h) => h)
  const atK = (h: number) => {
    if (m.mode === 'instrument') return null
    return interp(m.grid, m.grid.map((_, g) => cd.hourly[g]?.[h] ?? null), m.k)
  }
  const series: SweepSeries[] = [
    { id: 'inst', label: 'instrument BCc', color: INK.identity, points: hours.map((h) => ({ x: h, y: cd.hourly_instrument[h] })), width: 1.4 },
    { id: 'bc1', label: 'BC1 (k 0)', color: RESEARCH_SERIES[1], dash: '4 3', points: hours.map((h) => ({ x: h, y: cd.hourly[0]?.[h] ?? null })), width: 1.2 },
    ...(m.mode !== 'instrument' ? [{ id: 'k', label: kLabel(m), color: INK.accent, points: hours.map((h) => ({ x: h, y: atK(h) })), width: 1.8 }] : []),
  ]
  return (
    <ChartFrame id="xk-diurnal" source={source('Diurnal')} provenance={runLine(m)}
      title="Hour of day: the loading correction scales the day, it doesn’t reshape it"
      subtitle={`Mean ${m.ch} BC by local hour over the whole record.`}>
      <SweepPlot series={series} xLabel="hour of day (local)" yLabel="µg/m³" xDomain={[0, 23]} height={240} />
    </ChartFrame>
  )
}

// ---------------------------------------------------------------------------------------------- same form as the HIPS correction
function SameAsHips({ m }: { m: Loading }) {
  const cd = m.cd
  // the darkest ATN1 worth drawing: the 99th percentile of the minutes
  const atnTop = useMemo(() => {
    let acc = 0
    for (let i = 0; i < cd.atn.counts.length; i++) { acc += cd.atn.counts[i]; if (acc >= 0.99 * cd.atn.n) return cd.atn.lo + (i + 1) * cd.atn.step }
    return 100
  }, [cd])
  const T = 1.6
  const xs = Array.from({ length: 81 }, (_, i) => (i / 80) * T)
  // instantaneous response of HIPS: d(reported)/d(true) at a reported τ. Curve: (1 − τ/τ∞)². Screen: numerically.
  const curve = (tinf: number) => xs.map((t) => ({ x: t, y: t < tinf ? (1 - t / tinf) ** 2 : null }))
  const screen = (alpha: number) => {
    const pts: { x: number; y: number }[] = []
    for (let x = 0; x < 12; x += 0.02) {
      const t = twoZone(x, alpha), dt = (twoZone(x + 1e-4, alpha) - t) / 1e-4
      if (t <= T) pts.push({ x: t, y: dt })
    }
    return pts
  }
  const k = m.mode === 'instrument' ? m.presets.median : m.k
  const series: SweepSeries[] = [
    { id: 'aeth', label: `aethalometer, 1 − k·ATN1 (k ${fmt(k, 4)}), x = ATN1/100`, color: INK.accent, width: 2,
      points: xs.filter((t) => t * 100 <= atnTop).map((t) => ({ x: t, y: t * 100 > m.d.settings.atn_min ? 1 - k * 100 * t : 1 })) },
    { id: 'curve', label: 'HIPS, our curve τ∞ 2.26: (1 − τ/τ∞)²', color: RESEARCH_SERIES[4], width: 2, points: curve(2.2596) },
    { id: 'a10', label: 'HIPS, White’s screen α 10', color: RESEARCH_SERIES[0], width: 1.6, dash: '5 3', points: screen(10) },
  ]
  return (
    <ChartFrame id="xk-hips" source={source('SameAsHips')} provenance={runLine(m)}
      title="The same idea as the HIPS correction: how much of each new bit of soot still registers"
      subtitle={`Both instruments read less of each new increment as the deposit darkens. For the aethalometer that share is 1 − k·ATN1, up to the 99th percentile of ATN1 here (${fmt(atnTop, 0)}). For HIPS it is the slope of the reported optical depth against the true one. The x axes are not the same quantity: ATN includes the tape’s own multiple scattering, and HIPS removes scattering with its reflectance reading, so compare shapes, not values.${m.mode === 'instrument' ? ' The aethalometer line uses the median K while the page is on the instrument’s own K.' : ''}`}>
      <SweepPlot series={series} xLabel="darkness: ATN1/100 (aethalometer), reported τ (HIPS)" yLabel="share of a new increment registered" yDomain={[0, 1.05]} xDomain={[0, T]} height={260} />
    </ChartFrame>
  )
}
