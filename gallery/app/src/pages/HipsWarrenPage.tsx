import { useMemo, useRef, type ReactNode } from 'react'
import * as d3 from 'd3'
import { XAxis, YAxis } from '@/components/Axes'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { Segmented } from '@/components/Controls'
import { LoadState } from '@/components/ExportState'
import { HeatCell } from '@/components/HeatCell'
import { IntervalStrip, type IntervalMark } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { PageToc } from '@/components/PageToc'
import { IdentityLine, Marker, RefLine } from '@/components/RefLines'
import { ScatterPlot, type ScatterPoint, type ScatterScales } from '@/components/ScatterPlot'
import { ScrubField } from '@/components/ScrubField'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useColourScheme } from '@/lib/colourSchemes'
import { useExport } from '@/lib/data'
import { fmt, signed } from '@/lib/format'
import {
  clumpedDeposit, compositeModel, curveClearShare, curveVariance, factorOf, fitModel, IMPROVE_HOLES, inverse, patternVariance,
  screenDeposit, twoZone, WHITE_ALPHA, type HipsModel, type ModelFit,
} from '@/lib/hipsPattern'
import { SitePicker } from '@/lib/siteFocus'
import { median, quantile } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { BendFits, BendInFabs, CutTable, DEFAULT_CUT } from '@/pages/HipsBendChecks'
import { AgainstHips, Bend, CeilingOptions, COMMAND, pathOf, PROVENANCE, range, SIDE, useCeiling, WORKFLOW, type Ceil, type Linearise } from '@/pages/HipsLinearisePage'
import { DataTable, MEASURED, Readouts, type Column } from '@/pages/hipsLineariseParts'
import { PeriodCharts, usePeriods } from '@/pages/HipsWarrenPeriods'
import { RouteCharts, useCsn } from '@/pages/HipsWarrenRoutes'
import {
  ADDIS, C, HALF, line, macText, ModelOptions, pct, PRIVATE_BUILD, siteOption, ToChart, useDrain, useModels, useSiteSet, useWarren, WARREN_COMMAND, WARREN_FILE,
  type Drain, type Models, type SiteSet, type Warren,
} from '@/pages/HipsWarrenParts'

/**
 * Warren White's patterned-deposit model beside the gallery's saturating curve: whether ours is an extreme version of his,
 * what either does to Addis and the network, how far SPARTAN's unknown support screen matters, and the answers to what
 * Ann asked on 19 Nov 2026, each with the chart that carries it. Model in lib/hipsPattern.ts; data from
 * gallery/data/export_hips_warren.py. Account: docs/hips-warren-model-2026-11-24.md.
 */
const Part = ({ children }: { children: string }) => <h2 className="meeting-section-title">{children}</h2>
const PAPER = <>White et al. (2025, JAWMA 75, 209–232), Eq. 15 and 18; h = {IMPROVE_HOLES} from McDade, Dillner and Indresand (2009)</>
const COLORS = C

export function HipsWarrenPage() {
  const { data, error } = useExport<Linearise>('hips_linearise/linearise.json')
  const { data: w, error: wError } = useWarren()
  const drain = useDrain()
  const ceil = useCeiling(data)
  const models = useModels(ceil, drain)
  if (error || !data || !w) return (
    <ChartFrame id="wp-missing" title="Warren White's model and ours" provenance={WORKFLOW}>
      <LoadState what="the linearisation and White-model exports" error={error ?? wError} command={error ? COMMAND : WARREN_COMMAND} />
    </ChartFrame>
  )
  return <PageToc><Body data={data} w={w} drain={drain} ceil={ceil} models={models} /></PageToc>
}

interface Props { data: Linearise; w: Warren; drain: Drain | null; ceil: Ceil; models: Models }

function Body(p: Props) {
  const site = useSiteSet(p.data)
  return <>
    <ModelOptions m={p.models} />
    <CeilingOptions ceil={p.ceil} />
    <Part>1. What we can say, and how firmly</Part>
    <Claims {...p} />
    <Part>2. How Warren got there, and how we got there</Part>
    <RouteCharts data={p.data} w={p.w} ceil={p.ceil} />
    <Part>3. Why a dark filter reads low: the same explanation for both models</Part>
    <Picture {...p} />
    <Curves {...p} />
    <Part>4. The evidence on IMPROVE, inside Warren's trusted samples</Part>
    <Bend data={p.data} ceil={p.ceil} improveDefault="filters" />
    <BendFits data={p.data} />
    <BendInFabs data={p.data} ceil={p.ceil} />
    <TrustedSamples w={p.w} ceil={p.ceil} />
    <PeriodCharts data={p.data} ceil={p.ceil} />
    {PRIVATE_BUILD && p.drain && <DrainDiscs drain={p.drain} ceil={p.ceil} />}
    <Part>5. Applied: what each correction does to a site</Part>
    <AppliedSite {...p} site={site} />
    <SiteStrip {...p} site={site} />
    <Part>6. Our screen is not IMPROVE's</Part>
    <Mesh {...p} site={site} />
    <Split {...p} />
    <Geometry data={p.data} />
    <Part>7. Where to start correcting, and the filters that need none</Part>
    <Factors {...p} site={site} />
    <AgainstHips data={p.data} ceil={p.ceil} cutDefault={String(DEFAULT_CUT)} />
    <CutTable data={p.data} ceil={p.ceil} />
    <Part>8. What Ann asked on 19 November, and what is left</Part>
    <Asks drain={p.drain} />
    <Questions drain={p.drain} />
  </>
}

// ------------------------------------------------------------------ 1. claims
type Status = 'Firm' | 'Size open' | 'Not yet'
const STATUS_COLOR: Record<Status, string> = { Firm: '#4f9d69', 'Size open': '#d08c2c', 'Not yet': '#b5446e' }
interface Claim { claim: string; status: Status; why: ReactNode; breaks: string; chart: [string, string] }

function Claims({ data, w, drain, ceil }: Props) {
  const { data: periods } = usePeriods()
  const { data: csn } = useCsn()
  const pairs = csn?.sets[0].fits
  const inSeason = (s: string) => periods?.pooled.find((c) => c.season === s)
  const missOf = (s: string, k: string) => pct(inSeason(s)?.models.find((m) => m.key === k)?.rms_log ?? NaN, 1)
  const own = data.filters.filter((f) => f.ftir != null)
  const past = own.filter((f) => f.tau > w.limit_tau).length
  const rural = w.sets.find((s) => s.key === 'rural')!, urban = w.sets.find((s) => s.key === 'urban')!
  const ten = useMemo(() => fitModel(data.filters, data.lambda, { invert: inverse((x) => twoZone(x, WHITE_ALPHA)) }, data.mac), [data])
  const raw = fitModel(data.filters, data.lambda, { invert: (t) => t }, data.mac)
  const curve = fitModel(data.filters, data.lambda, { invert: (t) => (t < ceil.ceiling ? t / (1 - t / ceil.ceiling) : NaN) }, data.mac)
  const rows: Claim[] = [
    { claim: 'HIPS reads low on dark filters', status: 'Firm', chart: ['hl-bend', 'the bend'],
      why: <>Published by White et al. (2025), who call reported Fabs “only a lower bound”. Every IMPROVE fit bends, inside their samples too.</>,
      breaks: 'A linear HIPS response on uniform soot deposits at the same darkness' },
    { claim: 'It bends against CSN too, with no TOR EC in the values and no FTIR', status: 'Firm', chart: ['wp-csn', 'Warren’s test'],
      why: <>The same air at about a tenth of the darkness: IMPROVE reads {csn ? `${fmt(1 / csn.sets[0].groups[csn.sets[0].groups.length - 2].x * csn.sets[0].groups[csn.sets[0].groups.length - 2].tau, 2)} of CSN at τ ${fmt(csn.sets[0].groups[csn.sets[0].groups.length - 2].tau, 1)}` : '…'}. Our curve fitted there gives τ∞ {fmt(pairs?.curve.p[1], 2)} ({fmt(pairs?.curve1.p[0], 2)} if the networks may not differ by a constant), beside 2.26 from TOR.</>,
      breaks: 'CSN reading high on dark days for another reason (inlet, deposit area)' },
    { claim: 'Addis is in the range where it matters', status: 'Firm', chart: ['wp-factors', 'where to start'],
      why: <>{past} of {own.length} Addis filters ({pct(past / own.length)}) are past Warren's limit τ {fmt(w.limit_tau, 2)}, against {pct(rural.past_limit, 1)} of rural and {pct(urban.past_limit)} of urban IMPROVE filters in the 2010–2020 set he used.</>,
      breaks: 'A wrong deposit area or volume at Addis' },
    { claim: 'Addis HIPS as measured cannot set a level or an intercept', status: 'Firm', chart: ['wp-applied', 'applied'],
      why: <>Every correction raises the −6 intercept; by how much differs, and the largest overshoot zero.</>,
      breaks: 'Thermal EC on the Addis filters agreeing with Fabs ÷ 10' },
    { claim: 'Our curve and Warren’s model are the same kind of correction', status: 'Firm', chart: ['wp-curves', 'the curves'],
      why: <>Both are Beer–Lambert averaged over an uneven deposit. Ours is the same mathematics with a more uneven deposit: thickness variance {fmt(curveVariance(ceil.ceiling), 2)} against {fmt(patternVariance(WHITE_ALPHA), 2)} for α = 10, and at most {fmt((1 - IMPROVE_HOLES) / IMPROVE_HOLES, 2)} for any pattern IMPROVE's screen can make.</>,
      breaks: 'Nothing; it is algebra' },
    { claim: 'Whether the screen alone explains the IMPROVE bend depends on the months', status: 'Size open', chart: ['wp-groups', 'the groups by period'],
      why: <>November to June, α = 10 misses the group medians by {missOf('rest', 'alpha10')} and our curve by {missOf('rest', 'curve')}: the screen is nearly enough. July to October, α = 10 misses by {missOf('fire', 'alpha10')}; all months, up to {fmt(Math.max(...w.fits.find((f) => f.key === 'alpha10')!.residual_pct.map(Math.abs)), 0)} % in a group.</>,
      breaks: 'The bend going away on uniform deposits (drain discs), or the July to October bend having another cause' },
    { claim: 'The Addis intercept goes to about zero', status: 'Size open', chart: ['wp-applied', 'applied'],
      why: <>{line(raw)} as measured; {line(ten)} with Warren’s published α; {line(curve)} with our curve. It holds only for corrections about this large.</>,
      breaks: 'A much weaker pattern on SPARTAN’s screen' },
    { claim: 'The Addis MAC is 16 to 18 (by IMPROVE months it is about 10 to 18)', status: 'Not yet', chart: ['wp-season-addis', 'the months at Addis'],
      why: <>{macText(ten)} with Warren’s α, {macText(curve)} with ours, about 13 to 18 with our curve on Warren’s CSN pairs, about 10 with the November to June IMPROVE curve, and about 6 to 19 depending on SPARTAN’s screen. White (2016) adds up to a factor of 2 in HIPS’s own scale, so a high HIPS MAC is not yet dirty combustion.</>,
      breaks: 'Thermal EC at Addis; SPARTAN’s hole fraction and amplitude' },
  ]
  const columns: Column<Claim>[] = [
    { key: 'claim', label: 'Claim', value: (r) => <strong>{r.claim}</strong> },
    { key: 'status', label: 'Status', value: (r) => <span style={{ color: STATUS_COLOR[r.status], fontWeight: 700, whiteSpace: 'nowrap' }}>{r.status}</span> },
    { key: 'why', label: 'Rests on', value: (r) => r.why },
    { key: 'breaks', label: 'What would break it', value: (r) => r.breaks },
    { key: 'chart', label: 'Chart', value: (r) => <ToChart id={r.chart[0]}>{r.chart[1]}</ToChart> },
  ]
  return (
    <ChartFrame id="wp-claims" title="What we can say, and how firmly — HIPS reading low is firm; how much, and so the Addis MAC, is not"
      subtitle="Each claim with what it rests on and the result that would overturn it. Firm: more than one independent line of evidence, or algebra. Size open: the direction is agreed, the amount is not. Not yet: depends on something not measured."
      provenance={<>{PROVENANCE} <code>{WARREN_FILE}</code>; {PAPER}.</>}>
      <DataTable rows={rows} columns={columns} rowKey={(r) => r.claim} />
      {PRIVATE_BUILD && drain && <Note>This build also shows the Phoenix drain-disc pairs (part 4). They are unpublished UC Davis data: ask Warren and Ann before showing or sharing them.</Note>}
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 2. a picture of the deposits
const PLACES = 180
function Picture({ ceil, models }: Props) {
  const [xText, setX] = useSearchParam<string>('wp.x', '1.4')
  const x = Math.min(4, Math.max(0.05, Number(xText) || 1.4))
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const rows = useMemo(() => {
    const clumps = clumpedDeposit(ceil.ceiling, PLACES)
    return [
      { key: 'even', name: 'Even deposit (what reported Fabs assumes)', z: Array<number>(PLACES).fill(1), read: x, color: COLORS.none },
      { key: 'screen', name: `Screen pattern (Warren, α = ${WHITE_ALPHA})`, z: screenDeposit(WHITE_ALPHA, IMPROVE_HOLES, PLACES, 9), read: twoZone(x, WHITE_ALPHA), color: COLORS.alpha10 },
      { key: 'clumps', name: `Clumped deposit (exactly our curve, τ∞ ${fmt(ceil.ceiling, 2)})`, z: clumps, read: x / (1 + x / ceil.ceiling), color: COLORS.curve },
      ...(models.key === 'custom' ? [{ key: 'own', name: models.model.label, z: screenDeposit(models.alpha, models.h, PLACES, 9), read: models.model.response(x), color: COLORS.custom }] : []),
    ]
  }, [x, ceil.ceiling, models])
  const left = 4, right = 170, rowH = 104, barTop = 22, barMax = 70
  const W = Math.max(320, width || 700), iw = W - left - right
  const zMax = 3.2
  return (
    <ChartFrame id="wp-picture" title="Why a dark filter reads low — the same soot spread unevenly lets light through the thin places, so HIPS sees less of it"
      subtitle="Three filters in cross-section, all holding the same absorbing material. Bars hang down by the local thickness. HIPS measures the light that gets through, averaged over the filter, and reports −ln of it; where the deposit is thick, extra soot stops almost no extra light, and where it is thin, light passes. Drag the amount."
      provenance={<>Schematic from the models: {PAPER}; the clumped deposit is a Poisson number (mean τ∞) of exponentially thick clumps per place, whose average transmittance is exactly the saturating curve.</>}
      controls={<span className="control" title="True absorption depth: what HIPS would report if the deposit were even">True optical depth <ScrubField label="True optical depth" value={x} min={0.05} max={4} step={0.05} width={84} format={(v) => fmt(v, 2)} onChange={(v) => setX(String(+v.toFixed(2)))} /></span>}>
      <div ref={ref}>
        <svg width={W} height={rows.length * rowH + 4} fontFamily={FONT.family}>
          {rows.map((r, k) => <g key={r.key} transform={`translate(${left},${k * rowH})`}>
            <text y={13} fontSize={12.5} fontWeight={600} fill={INK.text}>{r.name}</text>
            <line x1={0} x2={iw} y1={barTop} y2={barTop} stroke={INK.axis} />
            {r.z.map((z, i) => <rect key={i} x={(i * iw) / PLACES} width={iw / PLACES + 0.3} y={barTop} height={Math.max(0, (Math.min(z, zMax) / zMax) * barMax)} fill={r.color}
              fillOpacity={0.15 + 0.85 * (1 - Math.exp(-x * z))} />)}
            <g transform={`translate(${iw + 16},${barTop + 4})`}>
              <text y={12} fontSize={11} fill={INK.muted}>HIPS reports</text>
              <text y={34} fontSize={20} fontWeight={700} fill={INK.text} fontFamily={FONT.mono}>{fmt(r.read, 2)}</text>
              <text y={52} fontSize={11.5} fill={r.key === 'even' ? INK.muted : r.color}>{r.key === 'even' ? `all of ${fmt(x, 2)}` : `${fmt(100 * (1 - r.read / x), 0)} % low (× ${fmt(x / r.read, 2)})`}</text>
            </g>
          </g>)}
        </svg>
      </div>
      <Key items={[{ label: 'Colour depth: share of the light stopped there', color: INK.muted, kind: 'square' }]} />
      <Note explain>Nothing about a single particle changes: the loss is geometry. It is also why the loss depends on the filter's darkness and not on its site, which is why
        the correction goes by τ. The clumps are a picture of how uneven the deposit would have to be to make our curve; they are not a claim that soot sits in clumps.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 2b. the curves, and how each fits IMPROVE
function Curves({ w, ceil, models, drain }: Props) {
  const list = models.models.filter((m) => m.key !== 'custom' || models.key === 'custom')
  const lines = ({ x, y, iw, ih }: ScatterScales) => {
    const xs = range(0, 3, 0.02)
    return <g pointerEvents="none">
      <clipPath id="wp-curves-clip"><rect width={iw} height={ih} /></clipPath>
      <g clipPath="url(#wp-curves-clip)">
        <path d={`${pathOf(xs.map((v) => [x(v), y(v / (1 + v / 2.7014))]))}L${xs.slice().reverse().map((v) => `${x(v)},${y(v / (1 + v / 1.8212))}`).join('L')}Z`} fill={COLORS.curve} opacity={0.12} />
        {list.map((m) => <path key={m.key} d={pathOf(xs.map((v) => [x(v), y(m.response(v))]))} fill="none" stroke={COLORS[m.key]} strokeWidth={m.key === models.key ? 3 : 1.75}
          strokeDasharray={m.key === 'none' ? '6 4' : m.key === 'full' ? '2 3' : undefined} />)}
        <Marker orient="h" at={y(w.limit_tau)} from={0} to={iw} color={COLORS.full} label={`IMPROVE screen limit τ ${fmt(w.limit_tau, 2)}`} labelSide="left" />
        <Marker orient="h" at={y(DEFAULT_CUT)} from={0} to={iw} color={INK.muted} label={`τ ${DEFAULT_CUT}`} />
      </g>
    </g>
  }
  const shown: { key: string; name: string; color: string }[] = [
    { key: 'curve', name: 'Saturating curve', color: COLORS.curve }, { key: 'alpha10', name: `White, α = ${WHITE_ALPHA}`, color: COLORS.alpha10 },
    { key: 'full', name: 'White, fully pixelated', color: COLORS.full }, { key: 'free_h', name: 'White, α and screen free', color: COLORS.custom },
  ]
  const residuals = useMemo<ScatterPoint[]>(() => shown.flatMap((s) => { // eslint-disable-line react-hooks/exhaustive-deps
    const f = w.fits.find((z) => z.key === s.key)!
    return f.residual_pct.map((v, i) => ({ key: `${s.key}/${i}`, x: w.medians[i].loading_ug, y: v, color: s.color, r: 3.5, opacity: 0.9,
      tip: [f.label, `IMPROVE group at ${fmt(w.medians[i].loading_ug, 1)} µg TOR EC, ${w.medians[i].n.toLocaleString()} filters`, `measured τ ${fmt(w.medians[i].tau, 3)}, ${signed(v, 1)} % from the fit`] }))
  }), [w])
  const resLines = ({ x, y }: ScatterScales) => <g pointerEvents="none">{shown.map((s) => {
    const f = w.fits.find((z) => z.key === s.key)!
    return <path key={s.key} d={pathOf(f.residual_pct.map((v, i) => [x(w.medians[i].loading_ug), y(v)]))} fill="none" stroke={s.color} strokeWidth={1.25} opacity={0.6} />
  })}</g>
  const fits = w.fits, best = Math.min(...fits.map((f) => f.aic))
  const freeH = fits.find((f) => f.key === 'free_h')!
  return (
    <ChartFrame id="wp-curves" title="Ours is the same kind of curve as Warren's, bent further, with every month in, than IMPROVE's own screen can bend it — early on it is the more extreme, past τ 1 it is the gentler"
      subtitle={`Left: the optical depth HIPS reports against the true one, for each model; the shaded band is our ceiling's 90 % interval. Right: how far the IMPROVE group medians (TOR EC, filters of the standard ${fmt(3.53, 2)} cm² area) sit from each model once its absorption per EC is fitted. The thickness variance is one number for the early bend of any of them: reported τ ≈ true − variance × true² ÷ 2.`}
      provenance={<><code>{WARREN_FILE}</code> (<code>gallery/data/export_hips_warren.py</code>); {PAPER}.</>}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 24, justifyContent: 'center' }}>
        <div style={{ flex: '1 1 300px', minWidth: 0, maxWidth: HALF + 80 }}><ScatterPlot title="Reported against true" points={[]} xLabel="True optical depth (even deposit)" yLabel="Optical depth HIPS reports" xDomain={[0, 3]} yDomain={[0, 2]} nice={false} maxSide={HALF} note={false} under={lines} /></div>
        <div style={{ flex: '1 1 300px', minWidth: 0, maxWidth: HALF + 80 }}><ScatterPlot title="IMPROVE medians, from each fitted model" points={residuals} xLabel="TOR EC on the filter (µg)" yLabel="Measured over fitted (%)" xScale="log" xDomain={[1, 60]} yDomain={[-16, 12]} nice={false}
          maxSide={HALF} outside="pin" note={false} refY={[0]} under={resLines} /></div>
      </div>
      <Key items={[...list.map((m) => ({ label: m.label, color: COLORS[m.key], kind: m.key === 'none' || m.key === 'full' ? 'dash' as const : 'line' as const })), { label: 'White, α and screen both free (right)', color: COLORS.custom, kind: 'dot' as const }]} />
      <Readouts items={[
        ...(PRIVATE_BUILD && drain ? [{ label: `Variance, α = ${fmt(drain.alpha.value, 1)}`, value: fmt(patternVariance(drain.alpha.value), 2), color: COLORS.drain, sub: 'Phoenix drain discs' }] : []),
        { label: `Variance, α = ${WHITE_ALPHA}`, value: fmt(patternVariance(WHITE_ALPHA), 2), color: COLORS.alpha10, sub: "Warren's network value" },
        { label: 'Most IMPROVE’s screen can make', value: fmt((1 - IMPROVE_HOLES) / IMPROVE_HOLES, 2), color: COLORS.full, sub: 'fully pixelated' },
        { label: 'Variance, our curve', value: fmt(curveVariance(ceil.ceiling), 2), color: COLORS.curve, sub: `${pct(curveClearShare(ceil.ceiling))} of its area clear` },
        { label: 'Screen the TOR data would need', value: `holes ${fmt(freeH.params[2], 2)}`, sub: `α ${fmt(freeH.params[1], 1)}; IMPROVE's is ${IMPROVE_HOLES}` },
      ]} />
      <DataTable rows={fits} rowKey={(f) => f.key} highlight={(f) => f.key === 'curve'} columns={[
        { key: 'l', label: 'Fitted to the 17 IMPROVE medians', value: (f) => f.label },
        { key: 'free', label: 'Free', value: (f) => f.free },
        { key: 'rms', label: 'Typical miss', value: (f) => pct(f.rms_log, 1), num: true },
        { key: 'aic', label: 'ΔAIC', value: (f) => fmt(f.aic - best, 1), num: true },
      ]} />
      <Note explain>Three answers. Same kind: yes, both average Beer–Lambert over an uneven deposit, and our formula is exactly the average over a clumped one.
        More extreme: yes at the start with every month in, because no pattern on IMPROVE's screen is uneven enough to bend as early as those data do (in November to June alone it nearly is); letting the screen float, the fit
        wants fewer holes than IMPROVE has. Not simply more extreme: Warren's pattern runs into a wall at τ {fmt(w.limit_tau, 2)} when nothing is between the dots, and ours does not
        until {fmt(ceil.ceiling, 2)}, so on the darkest filters his upper bound corrects more than we do. ΔAIC under about 2 cannot be told apart, over 10 is clearly worse.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 3. trusted samples
function TrustedSamples({ w, ceil }: { w: Warren; ceil: Ceil }) {
  const masked = w.areas.filter((a) => a.area_cm2 !== 3.53)
  const blocks = [{ key: 'sets', rows: w.sets.map((s) => ({
    key: s.key, label: s.label, strong: s.key === 'published', note: s.filters.toLocaleString(), labelColor: undefined,
    marks: [[{ mid: s.tau_inf, lo: s.lo, hi: s.hi, color: s.key === 'published' ? MEASURED : INK.text }] as IntervalMark[]],
    tip: [s.label.trim(), s.why, `τ∞ ${fmt(s.tau_inf, 2)} [${fmt(s.lo, 2)}, ${fmt(s.hi, 2)}]`, `${s.filters.toLocaleString()} filters, ${s.sites} sites, ${s.groups} groups`, `${pct(s.past_cut, 1)} past τ ${w.cut}, ${pct(s.past_limit, 2)} past τ ${fmt(w.limit_tau, 2)}`],
  })) }]
  const old = w.sets.find((s) => s.key === 'old')!, now = w.sets.find((s) => s.key === 'new')!
  return (
    <ChartFrame id="wp-range" title="Inside Warren's trusted samples the ceiling hardly moves — what moves it is the period, and that is mostly July to October (next charts)"
      subtitle={`τ∞ refitted on the samples White et al. (2016, 2025) use, with 90 % intervals from resampling whole sites ${w.n_boot} times. Trusted means 2003 on, no masked filters, the standard 3.53 cm² deposit, and sites with at least 1,000 samples; they cut nothing at the dark end. Number of filters on the right.`}
      provenance={<><code>{WARREN_FILE}</code>; deposit areas from the laboratory's HIPS table.</>}>
      <IntervalStrip blocks={blocks} panels={[{ label: 'Ceiling τ∞', domain: [1, 4.5], marker: { value: ceil.ceiling, label: `page ${fmt(ceil.ceiling, 2)}` } }]} labelWidth={280} rowHeight={26} noteWidth={66} />
      <Note>{masked.map((a) => `${a.filters.toLocaleString()} filters of ${a.area_cm2} cm²`).join(' and ')} were in the published set and treated as 3.53; without them τ∞ is {fmt(w.sets[1].tau_inf, 2)}, not {fmt(w.sets[0].tau_inf, 2)}.
        {' '}Before the 2017 HIPS rebuild {fmt(old.tau_inf, 2)} [{fmt(old.lo, 2)}, {fmt(old.hi, 2)}], after it {fmt(now.tau_inf, 2)} [{fmt(now.lo, 2)}, {fmt(now.hi, 2)}]. White (2025) puts the rebuilt instrument 5 to 6 % higher, which alone does not make that difference.</Note>
      <Note explain>Clog flags, which White (2025) also applies, are not in the table this reads. TOR's analysers changed in the same years; the carbon changes Zhang et al. (2025)
        document are CSN's, so IMPROVE's own need checking before the period difference is put on either instrument. The period difference is the July to October filters: see the groups by period.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 3b. drain discs (research build only)
function DrainDiscs({ drain, ceil }: { drain: Drain; ceil: Ceil }) {
  const ratio = (p: Drain['pairs'][number]) => p.r
  const points = useMemo<ScatterPoint[]>(() => drain.pairs.filter((p) => p.t1 < 2).map((p) => ({ key: p.d, x: p.t1, y: ratio(p), color: p.drain ? COLORS.drain : INK.muted, r: p.drain ? 4 : 2, opacity: p.drain ? 0.9 : 0.35,
    tip: [p.d, p.drain ? 'drain disc in the collocated sampler' : 'same holder in both samplers', `routine τ ${fmt(p.t1, 2)}, collocated ${fmt(p.t5, 2)} (× ${fmt(ratio(p), 2)})`,
      ...(p.mass != null ? [`PM2.5 mass × ${fmt(p.mass, 2)}`] : []), ...(p.ec != null ? [`TOR EC × ${fmt(p.ec, 2)}`] : [])] })), [drain])
  const alpha = drain.alpha.value
  const curves = useMemo(() => {
    const ts = range(0.15, 1.95, 0.02)
    const of = (inv: (t: number) => number) => ts.map((t) => [t, inv(t) / t] as [number, number])
    return [
      { key: 'fit', color: COLORS.drain, dash: '5 3', label: `White, α = ${fmt(alpha, 1)} [${fmt(drain.alpha.ci[0], 1)}, ${fmt(drain.alpha.ci[1], 1)}] (fits these pairs)`, pts: of(inverse((x) => twoZone(x, alpha))) },
      { key: 'ten', color: COLORS.alpha10, dash: undefined, label: `White, α = ${WHITE_ALPHA} (published)`, pts: of(inverse((x) => twoZone(x, WHITE_ALPHA))) },
      { key: 'curve', color: COLORS.curve, dash: undefined, label: `Our curve, τ∞ ${fmt(ceil.ceiling, 2)}`, pts: of((t) => (t < ceil.ceiling ? t / (1 - t / ceil.ceiling) : NaN)) },
    ]
  }, [alpha, drain, ceil.ceiling])
  const winters = useMemo(() => {
    const by = d3.group(drain.pairs.filter((p) => { const m = +p.d.slice(5, 7); return (m >= 10 || m <= 3) && Math.sqrt(p.t1 * p.t5) >= 1 }), (p) => +p.d.slice(0, 4) - (+p.d.slice(5, 7) <= 3 ? 1 : 0))
    return [...by].sort(([a], [b]) => a - b).map(([y, ps]) => ({ y, n: ps.length, drain: ps.some((p) => p.drain), r: ps.map(ratio), mass: ps.flatMap((p) => p.mass ?? []), ec: ps.flatMap((p) => p.ec ?? []) }))
  }, [drain])
  return (
    <ChartFrame id="wp-drain" title={`The direct test at Phoenix — with a drain disc the dark filters read × ${fmt(drain.dark.ratio, 2)} the routine ones; α = ${WHITE_ALPHA} expects × ${fmt(inverse((x) => twoZone(x, WHITE_ALPHA))(1.4) / 1.4, 1)} at τ 1.4`}
      subtitle={`Two IMPROVE samplers side by side. For part of the record one had a porous disc between filter and screen, meant to give an even deposit (White et al. 2025, section 5); window inferred as ${drain.window[0]} to ${drain.window[1]}. Left: collocated over routine Fabs (so a difference in flow drops out) against the routine filter's optical depth. Right: the same ratio for dark winter pairs (both about τ 1 or more), winter by winter, with the interquartile range, beside PM2.5 mass and TOR EC. If the disc's deposit is even, a correct model passes through the coloured dots.`}
      provenance={<>Private: <code>research/hips_drain_disc/pairs.json</code>, not in the public build or the repository. FED Fabs; filters tagged in UC Davis comments. {drain.note}</>}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 24, justifyContent: 'center' }}>
        <div style={{ flex: '1 1 300px', minWidth: 0, maxWidth: HALF + 80 }}><ScatterPlot title="Every pair" points={points} xLabel="Optical depth, routine filter" yLabel="Fabs, collocated ÷ routine" xDomain={[0, 2]} yDomain={[0.6, 2.8]} nice={false} maxSide={HALF} outside="pin" note={false} refY={[1]}
          under={({ x, y, iw, ih }) => <g pointerEvents="none"><clipPath id="wp-drain-clip"><rect width={iw} height={ih} /></clipPath><g clipPath="url(#wp-drain-clip)">
            {curves.map((c) => <path key={c.key} d={pathOf(c.pts.map(([a, b]) => [x(a), y(b)]))} fill="none" stroke={c.color} strokeWidth={2} strokeDasharray={c.dash} />)}</g></g>} /></div>
        <div style={{ flex: '1 1 380px', minWidth: 320 }}><IntervalStrip labelWidth={96} rowHeight={25} noteWidth={40} panels={[{ label: 'Fabs, collocated ÷ routine, dark winter pairs', domain: [0.8, 1.35], reference: 1 }]}
          blocks={[{ key: 'w', rows: winters.map((w) => ({ key: String(w.y), label: `${w.y}–${String(w.y + 1).slice(2)}`, strong: w.drain, note: `n ${w.n}`,
            marks: [[
              { mid: median(w.mass), color: INK.muted, shape: 'hollow' as const, r: 3, dy: -6, faint: true }, { mid: median(w.ec), color: INK.muted, shape: 'diamond' as const, r: 3, dy: 6, faint: true },
              { mid: median(w.r), lo: quantile(w.r, 0.25), hi: quantile(w.r, 0.75), color: w.drain ? COLORS.drain : INK.text },
            ]], tip: [`Winter ${w.y}–${w.y + 1}${w.drain ? ', drain discs' : ''}`, `Fabs × ${fmt(median(w.r), 3)}`, `mass × ${fmt(median(w.mass), 3)}, TOR EC × ${fmt(median(w.ec), 3)}`, `${w.n} pairs`] })) }]} /></div>
      </div>
      <Key items={[{ label: 'Drain disc in one sampler', color: COLORS.drain, kind: 'dot' }, { label: 'Same holder in both', color: INK.muted, kind: 'dot' }, ...curves.map((c) => ({ label: c.label, color: c.color, kind: c.dash ? 'dash' as const : 'line' as const })),
        { label: 'PM2.5 mass', color: INK.muted, kind: 'hollow' }, { label: 'TOR EC', color: INK.muted, kind: 'diamond' }]} />
      <Readouts items={[
        { label: 'Drain disc, τ 1 or more', value: `× ${fmt(drain.dark.ratio, 2)}`, color: COLORS.drain, sub: `[${fmt(drain.dark.ci[0], 2)}, ${fmt(drain.dark.ci[1], 2)}], ${drain.dark.n} pairs` },
        { label: 'All other dates, τ 1 or more', value: `× ${fmt(drain.dark.other_ratio, 2)}`, sub: `${drain.dark.other_n} pairs` },
        ...drain.ceilings.filter((c) => c.dates.startsWith('Drain')).map((c) => ({ label: `Bend against TOR, ${c.sampler.toLowerCase()}`, value: `τ∞ ${fmt(c.tau_inf, 1)}`, sub: `[${fmt(c.ci[0], 1)}, ${fmt(c.ci[1], 1)}], same days` })),
        { label: 'HIPS error from the pairs', value: `${pct(drain.hips_relative_error.without_drain, 1)}`, sub: `${pct(drain.hips_relative_error.all_pairs, 1)} with the disc months in, as the 20 Nov error model had it` },
      ]} />
      <Note explain>Two readings, and these pairs cannot choose between them. The discs may not make the deposit fully even, so the difference is a lower bound on the pattern. Or the discs do, and then
        whatever bend the disc filters still show against TOR is another loading effect, or composition. One site and one winter.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 4. a site under each correction
function AppliedSite({ data, models, site }: Props & { site: SiteSet }) {
  const { filters, lambda } = site
  const rows = useMemo(() => models.models.map((m) => ({ m, fit: fitModel(filters, lambda, m, data.mac) })), [models.models, filters, lambda, data.mac])
  const chosen = models.model, color = COLORS[models.key]
  const points = useMemo<ScatterPoint[]>(() => filters.flatMap((f) => {
    const k = factorOf(chosen, f.tau)
    const tip = (what: string, x: number) => [`${f.id} · ${what}`, `HIPS BC ${fmt(x, 2)} µg/m³ · FTIR EC ${fmt(f.ftir, 2)}`, `τ ${fmt(f.tau, 2)}${Number.isFinite(k) ? `, factor ${fmt(k, 2)}` : ', past what this model can read'}`]
    return [
      { id: f.id, key: `${f.id}/m`, x: f.hips_bc, y: f.ftir!, color: COLORS.none, r: 2.5, opacity: 0.5, tip: tip('as measured', f.hips_bc) },
      ...(models.key !== 'none' && Number.isFinite(k) ? [{ id: f.id, key: `${f.id}/c`, x: f.hips_bc * k, y: f.ftir!, color, r: 3.5, hollow: true, tip: tip(chosen.short, f.hips_bc * k) }] : []),
    ]
  }), [filters, chosen, models.key, color])
  // one square range: the bulk of the corrected values (past it a point sits on the edge), never less than the FTIR range
  const limit = Math.max(2, Math.ceil(Math.max(quantile(points.map((p) => p.x), 0.985), ...points.map((p) => p.y)) / 2) * 2)
  const drawn = rows.filter((r) => r.fit && (r.m.key === 'none' || r.m.key === models.key))
  const under = ({ x, y, iw, ih }: ScatterScales) => <g pointerEvents="none">
    <clipPath id="wp-applied-clip"><rect width={iw} height={ih} /></clipPath>
    <RefLine orient="h" at={y(0)} from={0} to={iw} />
    <IdentityLine x1={x(0)} y1={y(0)} x2={x(limit)} y2={y(limit)} />
    <g clipPath="url(#wp-applied-clip)">{drawn.map((r) => <line key={r.m.key} x1={x(0)} x2={x(limit)} y1={y(r.fit!.intercept)} y2={y(r.fit!.intercept + r.fit!.slope * limit)}
      stroke={COLORS[r.m.key]} strokeWidth={2} strokeDasharray={r.m.key === 'none' ? '6 4' : undefined} />)}</g>
  </g>
  type Row = (typeof rows)[number]
  const columns: Column<Row>[] = [
    { key: 'm', label: 'Correction', value: (r) => <><span style={{ display: 'inline-block', width: 9, height: 9, borderRadius: 9, background: COLORS[r.m.key], marginRight: 7 }} />{r.m.label}</> },
    { key: 'v', label: 'Thickness variance', value: (r) => fmt(r.m.variance, 2), num: true },
    { key: 'k', label: 'Median factor', value: (r) => (r.fit ? `× ${fmt(r.fit.factor, 2)}` : '—'), num: true },
    { key: 'fit', label: 'FTIR EC on HIPS BC', value: (r) => line(r.fit), num: true },
    { key: 'mac', label: 'MAC for slope 1', value: (r) => macText(r.fit), num: true },
    { key: 'past', label: 'Filters with no answer', value: (r) => (r.fit?.past ? `${r.fit.past} of ${r.fit.past + r.fit.n}` : '0'), num: true },
  ]
  const ten = rows.find((r) => r.m.key === 'alpha10')?.fit, none = rows.find((r) => r.m.key === 'none')?.fit
  return (
    <ChartFrame id="wp-applied" title={`Applied to ${site.name} — ${ten && none ? `Warren's published correction takes the intercept from ${signed(none.intercept, 1)} to ${signed(ten.intercept, 1)}, with no FTIR or TOR data behind it` : 'each correction, the same filters'}`}
      subtitle={`FTIR EC (y) against HIPS BC = Fabs ÷ ${data.mac} (x): grey as measured, coloured rings after the correction chosen in Page options (${chosen.label}). Deming, λ ${fmt(lambda, 2)}. The MAC for slope 1 is MAC ÷ slope; the intercept does not depend on it. A filter darker than the modelled deposit could ever read has no answer and drops out.`}
      provenance={<>{PROVENANCE} {PAPER}.</>}
      controls={<>
        <SitePicker as="select" {...site.pick.picker} optionLabel={siteOption(site.sites)} title="The gallery's page site (psite): the other one-site charts on this page follow it" />
        <Segmented label="Correction" value={models.key} options={models.models.map((m) => m.key)} onChange={models.setKey} optionLabel={(k) => models.models.find((m) => m.key === k)?.short ?? k} />
      </>}>
      {filters.length < 5 ? <Empty>Fewer than five filters with FTIR EC and HIPS at this site.</Empty> : <>
        <ScatterPlot points={points} xLabel="HIPS BC (µg/m³)" yLabel="FTIR EC (µg/m³)" xDomain={[0, limit]} yDomain={[0, limit]} nice={false} maxSide={SIDE} outside="pin" note={false} under={under} />
        <Key items={[{ label: 'As measured', color: COLORS.none, kind: 'dot' }, ...(models.key !== 'none' ? [{ label: chosen.label, color, kind: 'hollow' as const }] : []), { label: 'Deming fits', color: INK.text, kind: 'line' }, { label: '1:1', color: INK.identity, kind: 'dash' }]} />
        <DataTable rows={rows} columns={columns} rowKey={(r) => r.m.key} highlight={(r) => r.m.key === models.key} />
        <Note explain>Warren's α = {WHITE_ALPHA} was set by making IMPROVE agree with CSN in four US cities, so nothing at Addis was tuned to it. That it removes most of the intercept is the best
          independent support for the meeting's reading. It is IMPROVE's screen at IMPROVE's flow, though; part 6 asks what happens if SPARTAN's is different.</Note>
      </>}
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 4b. every site
function SiteStrip({ data, models, site }: Props & { site: SiteSet }) {
  const others = useMemo(() => models.models.filter((m) => m.key !== 'none' && m.key !== 'full' && (m.key !== 'custom' || models.key === 'custom')), [models.models, models.key])
  const rows = useMemo(() => site.sites.flatMap((s) => {
    const addis = s.code === ADDIS
    const fs = (addis ? data.filters : site.bySite.get(s.code)?.filters ?? []).filter((f) => f.ftir != null)
    if (fs.length < 20) return []
    const lambda = addis ? data.lambda : 1
    return [{ code: s.code, name: s.name.split(',')[0], n: fs.length, tau: median(fs.map((f) => f.tau)), raw: fitModel(fs, lambda, { invert: (t: number) => t }, data.mac),
      by: others.map((m) => ({ m, fit: fitModel(fs, lambda, m, data.mac) })) }]
  }).sort((a, b) => b.tau - a.tau), [site.sites, site.bySite, data, others])
  if (!site.loaded) return null
  return (
    <ChartFrame id="wp-sites" title="Applied to the network — the corrections agree on light sites and part on dark ones, where Addis is the extreme"
      subtitle="SPARTAN sites with 20 or more filters, darkest first. Left: the median filter's factor under each correction. Right: the intercept of FTIR EC on HIPS BC, as measured (grey ring) and under the correction chosen in Page options. Hover for the fits. Addis uses the closure's filters and λ, the others the public rows and λ 1."
      provenance={PROVENANCE}>
      <IntervalStrip labelWidth={150} rowHeight={22} noteWidth={48}
        panels={[{ label: 'Median factor', domain: [1, 2.1], reference: 1 }, { label: 'Intercept (µg/m³)', domain: [Math.floor(Math.min(-2, ...rows.map((r) => r.raw?.intercept ?? 0))), 2], reference: 0 }]}
        blocks={[{ key: 'sites', rows: rows.map((r) => {
          const c = r.by.find((b) => b.m.key === models.key)?.fit
          return { key: r.code, label: `${r.name} · τ ${fmt(r.tau, 2)}`, strong: r.code === site.code, note: `n ${r.n}`, onClick: () => site.pick.picker.onChange(r.code),
            marks: [r.by.map(({ m, fit }) => ({ mid: fit?.factor, color: COLORS[m.key as keyof typeof COLORS], r: m.key === models.key ? 5 : 3.5 })),
              [{ mid: r.raw?.intercept, color: COLORS.none, shape: 'hollow' as const }, ...(c ? [{ mid: c.intercept, color: COLORS[models.key] }] : [])]],
            tip: [`${r.name}, median τ ${fmt(r.tau, 2)}, ${r.n} filters`, `as measured ${line(r.raw)}, MAC for slope 1 ${macText(r.raw)}`, ...r.by.map(({ m, fit }) => `${m.short}: ${line(fit)}, MAC ${macText(fit)}`)] }
        }) }]} />
      <Key items={[...others.map((m) => ({ label: m.label, color: COLORS[m.key], kind: 'dot' as const })), { label: 'Intercept as measured', color: COLORS.none, kind: 'hollow' as const }]} />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 5. the screen
const HS = d3.range(0.16, 0.961, 0.025), ALPHAS = d3.range(0, 26).map((i) => 1.5 * (60 / 1.5) ** (i / 25))
function Mesh({ data, models, site }: Props & { site: SiteSet }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const colours = useColourScheme('wp-mesh', 'diverging')
  const { filters, lambda } = site
  const cells = useMemo(() => HS.flatMap((h, i) => ALPHAS.map((alpha, j) => ({ h, alpha, i, j, fit: fitModel(filters, lambda, { invert: inverse((x) => twoZone(x, alpha, h)) }, data.mac) }))), [filters, lambda, data.mac])
  const raw = useMemo(() => fitModel(filters, lambda, { invert: (t) => t }, data.mac), [filters, lambda, data.mac])
  const span = Math.max(1, Math.ceil(Math.abs(raw?.intercept ?? 6)))
  const scale = d3.scaleSequential(d3.interpolateRgbBasis(colours.stops)).domain([-span, span]).clamp(true)
  const m = { top: 12, right: 16, bottom: 46, left: 62 }
  const side = Math.max(220, Math.min(SIDE, (width || SIDE) - m.left - m.right))
  const x = d3.scaleLinear().domain([HS[0] - 0.0125, HS[HS.length - 1] + 0.0125]).range([0, side])
  const step = Math.log(ALPHAS[1] / ALPHAS[0])
  const y = d3.scaleLog().domain([ALPHAS[0] * Math.exp(-step / 2), ALPHAS[ALPHAS.length - 1] * Math.exp(step / 2)]).range([side, 0])
  const cw = side / HS.length, ch = side / ALPHAS.length
  const usable = (f: ModelFit | null) => !!f && f.past <= 0.05 * (f.past + f.n)
  const near = cells.filter((c) => usable(c.fit) && Math.abs(c.fit!.intercept) < 0.5)
  const macs = near.map((c) => c.fit!.mac).sort((a, b) => a - b)
  const marks = [{ h: IMPROVE_HOLES, alpha: WHITE_ALPHA, label: `IMPROVE, α ${WHITE_ALPHA}` }, ...(models.key === 'custom' ? [{ h: models.h, alpha: models.alpha, label: 'yours' }] : [])]
  return (
    <ChartFrame id="wp-mesh" title={`Our screen is not IMPROVE's, and it matters — many screens would take ${site.name}'s intercept to zero, at very different MACs`}
      subtitle={`${site.name}'s intercept (FTIR EC on HIPS BC) after correcting with a screen of any hole share and any amplitude α. Outlined cells leave less than 0.5 µg/m³. Only the share of holes and α enter the model: hole size does not, except through α, since small close holes and low flow blur the pattern. Empty cells are screens under which more than 5 % of the filters could not be as dark as they are. Click a cell to use that screen on the page.`}
      provenance={<>{PROVENANCE} {PAPER}.</>}
      controls={<SitePicker as="select" {...site.pick.picker} optionLabel={siteOption(site.sites)} title="The gallery's page site (psite)" />}>
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={side + m.left + m.right} height={side + m.top + m.bottom} fontFamily={FONT.family} style={{ display: 'block', margin: '0 auto' }}>
          <g transform={`translate(${m.left},${m.top})`}>
            {cells.map((c) => {
              const ok = usable(c.fit), hit = ok && Math.abs(c.fit!.intercept) < 0.5
              return <g key={`${c.i}/${c.j}`} style={{ cursor: 'pointer' }} onClick={() => models.setScreen(c.h, c.alpha)} {...tip.bind([`Holes ${fmt(c.h, 2)}, α ${fmt(c.alpha, 1)}`,
                c.fit ? line(c.fit) : 'no fit', `MAC for slope 1: ${macText(c.fit)}`, `median factor × ${fmt(c.fit?.factor, 2)}`, ...(c.fit?.past ? [`${c.fit.past} filters darker than this deposit could be`] : [])])}>
                <HeatCell x={x(c.h) - cw / 2} y={y(c.alpha) - ch / 2} width={cw + 0.4} height={ch + 0.4} fill={ok ? scale(c.fit!.intercept) : null} outline={hit} />
              </g>
            })}
            {marks.map((k) => <g key={k.label} pointerEvents="none" transform={`translate(${x(k.h)},${y(k.alpha)})`}>
              <circle r={6} fill="none" stroke={INK.text} strokeWidth={2} /><circle r={1.5} fill={INK.text} />
              <text x={10} y={4} fontSize={11} fontWeight={600} fill={INK.text} stroke={INK.panel} strokeWidth={3} paintOrder="stroke">{k.label}</text></g>)}
            <g pointerEvents="none" transform={`translate(${x(0.283)},${side - 5})`}><path d="M-5,-3L0,5L5,-3Z" fill={INK.text} /><text x={8} y={2} fontSize={11} fontWeight={600} fill={INK.text} stroke={INK.panel} strokeWidth={3} paintOrder="stroke">CSN screen, α near 1</text></g>
            <XAxis scale={x} y={side} tickCount={6} label="Share of the screen that is holes" labelDy={38} format={(v) => fmt(+v, 1)} />
            <YAxis scale={y} x={0} tickCount={5} label="Amplitude α" format={(v) => ([2, 5, 10, 20, 50].includes(+v) ? String(v) : '')} />
          </g>
        </svg>
        {tip.node}
      </div>
      <div className="legend" style={{ justifyContent: 'center' }}><ColorLegend scale={scale} scheme={colours.scheme} label="Intercept (µg/m³)" width={220} format={(v) => fmt(v, 0)} /></div>
      <Readouts items={[
        { label: 'Screens that leave |intercept| under 0.5', value: `${near.length} of ${cells.filter((c) => usable(c.fit)).length}`, sub: 'on this grid' },
        { label: 'MAC for slope 1 among them', value: macs.length ? `${fmt(macs[0], 0)} to ${fmt(macs[macs.length - 1], 0)}` : '—', sub: 'm²/g' },
        { label: 'IMPROVE’s screen', value: `holes ${IMPROVE_HOLES}`, sub: 'dots about 0.33 mm' },
        { label: 'SPARTAN’s screen', value: 'not known', sub: 'still to get from the SPARTAN laboratory' },
      ]} />
      <Note explain>A screen with few holes and a screen with many can both be made to zero the intercept by choosing α, so a zero intercept does not identify the screen, and it does not by itself support a MAC.
        Amplitude is not free in practice: White et al. see the pattern fade as the air moves slower through the filter, which is the argument to make from the geometry table below.</Note>
    </ChartFrame>
  )
}

/** the IMPROVE bend cut into a screen part and the rest, and the rest carried to Addis with SPARTAN's unknown screen */
function Split({ data, w }: Props) {
  const rows = useMemo(() => w.splits.map((s) => {
    const fit = (alpha: number, h = IMPROVE_HOLES) => fitModel(data.filters, data.lambda, compositeModel('k', '', '', alpha, s.c, h), data.mac)
    const any = d3.range(0.2, 0.91, 0.05).map((h) => fit(s.alpha, h)).filter((f): f is ModelFit => !!f && f.past === 0)
    return { ...s, same: fit(s.alpha), none: fit(1), lo: any.length ? d3.min(any, (f) => f.intercept)! : NaN, hi: any.length ? d3.max(any, (f) => f.intercept)! : NaN,
      macLo: any.length ? d3.min(any, (f) => f.mac)! : NaN, macHi: any.length ? d3.max(any, (f) => f.mac)! : NaN }
  }), [data, w.splits])
  const best = Math.min(...w.splits.map((s) => s.aic)), worst = Math.max(...w.splits.map((s) => s.aic))
  const a = rows[0], z = rows[rows.length - 1]
  return (
    <ChartFrame id="wp-split" title="IMPROVE cannot say how much of its bend is the screen — and that is what decides Addis if SPARTAN's filters have no pattern"
      subtitle={`The IMPROVE bend split between a screen pattern of amplitude α and a remaining saturating part, refitted for each α: every split fits the medians equally well (ΔAIC ${fmt(worst - best, 1)} at most). Filled dot: Addis corrected with both parts, as if SPARTAN had IMPROVE's screen. Ring: the screen part dropped, as if SPARTAN's deposits had no pattern. Bar: IMPROVE's amplitude on a screen with 20 to 90 % holes.`}
      provenance={<><code>{WARREN_FILE}</code>; Addis filters from <code>hips_linearise/linearise.json</code>.</>}>
      <IntervalStrip labelWidth={250} rowHeight={30} panels={[{ label: 'Addis intercept (µg/m³)', domain: [-5, 1], reference: 0 }, { label: 'MAC for slope 1 (m²/g)', domain: [5, 26], marker: { value: data.mac, label: 'MAC 10' } }]}
        blocks={[{ key: 's', rows: rows.map((r) => ({ key: String(r.alpha), label: r.alpha === 1 ? 'No screen part (all of it our curve)' : `Screen part α ${r.alpha}, rest τ∞ ${fmt(r.c, 1)}`,
          marks: [[{ mid: null, lo: r.lo, hi: r.hi, color: INK.muted, faint: true }, { mid: r.none?.intercept, color: COLORS.alpha10, shape: 'hollow' as const, r: 6 }, { mid: r.same?.intercept, color: COLORS.curve }],
            [{ mid: null, lo: Math.min(26, r.macLo), hi: Math.min(26, r.macHi), color: INK.muted, faint: true }, { mid: r.none?.mac, color: COLORS.alpha10, shape: 'hollow' as const, r: 6 }, { mid: r.same?.mac, color: COLORS.curve }]],
          tip: [`Screen part α ${r.alpha}, rest τ∞ ${fmt(r.c, 2)}, ΔAIC ${fmt(r.aic - best, 1)}`, `with IMPROVE's screen: ${line(r.same)}, MAC ${macText(r.same)}`, `with no pattern on SPARTAN: ${line(r.none)}, MAC ${macText(r.none)}`] })) }]} />
      <Key items={[{ label: "SPARTAN has IMPROVE's pattern", color: COLORS.curve, kind: 'dot' }, { label: 'SPARTAN has no pattern', color: COLORS.alpha10, kind: 'hollow' }, { label: 'same α, 20 to 90 % holes', color: INK.muted, kind: 'line' }]} />
      <Note>With IMPROVE's pattern the split matters little (intercept {signed(Math.min(...rows.map((r) => r.same!.intercept)), 1)} to {signed(Math.max(...rows.map((r) => r.same!.intercept)), 1)}, MAC {fmt(Math.min(...rows.map((r) => r.same!.mac)), 0)} to {fmt(Math.max(...rows.map((r) => r.same!.mac)), 0)}). With none, Addis runs from {line(a.none)} (MAC {macText(a.none)}) to {line(z.none)} (MAC {macText(z.none)}).
        Photographs of SPARTAN deposits and the hole fraction settle which end we are at.</Note>
    </ChartFrame>
  )
}

function Geometry({ data }: { data: Linearise }) {
  const v = median(data.filters.map((f) => f.volume_m3)), flow = (v / 24 / 60) * 1000
  const rows = [
    { net: 'IMPROVE', flow: 22.8, area: 3.53, h: '0.638', hole: 'dots about 0.33 mm', seen: 'Clear dots (White 2025, Fig. 7)' },
    { net: 'CSN, modified to 25 mm (one test)', flow: 6.7, area: 3.53, h: '0.283', hole: '0.1 mm, 0.2 mm apart', seen: 'A visible pattern: about 3 × the CSN face velocity (Fig. 13)' },
    { net: 'SPARTAN, Addis', flow, area: 3.53, h: 'not known', hole: 'not known', seen: 'To be photographed' },
    { net: 'CSN, standard', flow: 6.7, area: 11.86, h: '0.283', hole: '0.1 mm, 0.2 mm apart', seen: 'Nearly even (Fig. 13)' },
  ].map((r) => ({ ...r, face: r.flow / r.area })).sort((p, q) => q.face - p.face)
  type Row = (typeof rows)[number]
  const columns: Column<Row>[] = [
    { key: 'n', label: 'Sampler', value: (r) => r.net }, { key: 'f', label: 'Flow (L/min)', value: (r) => fmt(r.flow, 1), num: true }, { key: 'a', label: 'Deposit (cm²)', value: (r) => fmt(r.area, 2), num: true },
    { key: 'v', label: 'Flow per area (L/min/cm²)', value: (r) => fmt(r.face, 2), num: true }, { key: 'h', label: 'Hole share', value: (r) => r.h, num: true }, { key: 'd', label: 'Holes', value: (r) => r.hole }, { key: 's', label: 'Pattern seen', value: (r) => r.seen },
  ]
  return (
    <ChartFrame id="wp-geometry" title="Air moves through a SPARTAN filter far slower than through IMPROVE's — a reason to expect less pattern, not more"
      subtitle={`White et al. (2025, p. 225) tie the pattern to how hard the air presses the membrane onto the screen. Addis draws ${fmt(v, 1)} m³ on its median filter (taken over 24 hours).`}
      provenance={<>White et al. (2025), Table 2, Fig. 7 and 13; Addis volume from <code>hips_linearise/linearise.json</code>.</>}>
      <DataTable rows={rows} columns={columns} rowKey={(r) => r.net} highlight={(r) => r.net.startsWith('SPARTAN')} />
      <Note>If the pattern were the whole cause, a slower sampler should bend less. Addis does not: its exponent against FTIR EC is 0.46 where our curve expects 0.55. Either SPARTAN's screen patterns the deposit strongly
        despite the flow, or most of the bend is not the screen. An uneven deposit of another kind (the “mudcracks” of White et al., Fig. 12) would also do it.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 6. where to start
function Factors({ data, w, models, site, ceil }: Props & { site: SiteSet }) {
  const list = models.models.filter((m) => m.key !== 'none' && (m.key !== 'custom' || models.key === 'custom'))
  const taus = site.filters.map((f) => f.tau)
  const at = (m: Pick<HipsModel, 'invert'>) => factorOf(m, w.cut)
  const under = ({ x, y, iw, ih }: ScatterScales) => {
    const xs = range(0.02, 1.8, 0.01)
    const bins = d3.bin().domain([0, 1.8]).thresholds(36)(taus), peak = d3.max(bins, (b) => b.length) ?? 1
    return <g pointerEvents="none">
      <clipPath id="wp-factors-clip"><rect width={iw} height={ih} /></clipPath>
      {bins.map((b) => <rect key={b.x0} x={x(b.x0!)} width={Math.max(0, x(b.x1!) - x(b.x0!) - 1)} y={ih - (b.length / peak) * ih * 0.28} height={(b.length / peak) * ih * 0.28} fill={INK.muted} opacity={0.25} />)}
      <g clipPath="url(#wp-factors-clip)">
        {list.map((m) => <path key={m.key} d={pathOf(xs.map((t) => [x(t), y(factorOf(m, t))]))} fill="none" stroke={COLORS[m.key]} strokeWidth={m.key === models.key ? 3 : 1.75} strokeDasharray={m.key === 'full' ? '2 3' : undefined} />)}
        <Marker orient="v" at={x(w.cut)} from={0} to={ih} color={INK.text} label={`cut ${w.cut}`} />
        <Marker orient="v" at={x(w.limit_tau)} from={0} to={ih} color={COLORS.full} label={`limit ${fmt(w.limit_tau, 2)}`} />
      </g>
    </g>
  }
  const shareOf = (name: string, fs: { tau: number }[]) => ({ population: name, filters: fs.length, past_cut: fs.filter((f) => f.tau > w.cut).length / fs.length, past_limit: fs.filter((f) => f.tau > w.limit_tau).length / fs.length })
  const delhi = site.bySite.get('INDH')?.filters ?? []
  const all = [...site.bySite].filter(([c]) => c !== ADDIS).flatMap(([, s]) => s.filters)
  const shares = [...w.shares, ...(all.length ? [shareOf('SPARTAN, other sites', all)] : []), ...(delhi.length ? [shareOf('Delhi', delhi)] : []), shareOf('Addis Ababa', data.filters.filter((f) => f.ftir != null))]
  const full = list.find((m) => m.key === 'full')!
  return (
    <ChartFrame id="wp-factors" title={`Where to start correcting — at τ ${w.cut} Warren's upper bound is ${fmt(100 * (at(full) - 1), 0)} %, and past τ ${fmt(w.limit_tau, 2)} there is none`}
      subtitle={`The factor each correction multiplies Fabs by, against the filter's optical depth, over the distribution of ${site.name}'s filters (grey). Warren's fully pixelated curve is the most IMPROVE's screen can do at any α: a filter at τ ${w.cut} cannot be lower than that because of the screen; past ${fmt(w.limit_tau, 2)} (absorptance ${IMPROVE_HOLES}) there is no such limit. Our curve sits above his limit until it crosses it, which is the part a screen cannot explain. SPARTAN's screen has its own limit, unknown.`}
      provenance={<><code>{WARREN_FILE}</code>; {PAPER}.</>}
      controls={<SitePicker as="select" {...site.pick.picker} optionLabel={siteOption(site.sites)} title="The gallery's page site (psite)" />}>
      <ScatterPlot points={[]} xLabel="HIPS optical depth τ" yLabel="Factor on Fabs" xDomain={[0, 1.8]} yDomain={[1, 4]} nice={false} maxSide={SIDE} note={false} under={under} />
      <Key items={list.map((m) => ({ label: m.label, color: COLORS[m.key], kind: m.key === 'full' ? 'dash' as const : 'line' as const }))} />
      <Readouts items={list.map((m) => ({ label: `At τ ${w.cut}, ${m.short}`, value: `× ${fmt(at(m), 2)}`, color: COLORS[m.key], sub: `at τ 0.97: × ${fmt(factorOf(m, 0.97), 2)}` }))} />
      <DataTable rows={shares} rowKey={(r) => r.population} highlight={(r) => r.population === 'Addis Ababa'} columns={[
        { key: 'p', label: 'Filters', value: (r) => r.population }, { key: 'n', label: 'Number', value: (r) => r.filters.toLocaleString(), num: true },
        { key: 'c', label: `Past τ ${w.cut}: compromised Fabs`, value: (r) => pct(r.past_cut, r.past_cut < 0.1 ? 1 : 0), num: true }, { key: 'l', label: `Past τ ${fmt(w.limit_tau, 2)}: no bound`, value: (r) => pct(r.past_limit, r.past_limit < 0.1 ? 1 : 0), num: true },
      ]} />
      <Note explain>A cut to quote: below τ {w.cut} use Fabs as reported and state that it may be up to {fmt(100 * (at(full) - 1), 0)} % low on this account (our curve, which is not bounded by the screen, says {fmt(100 * (1 / (1 - w.cut / ceil.ceiling) - 1), 0)} %);
        from {w.cut} to {fmt(w.limit_tau, 2)} Fabs is compromised and a corrected value carries a range; past {fmt(w.limit_tau, 2)} it is shape only. Compromised, not lost: FTIR EC, OC and mass on those filters stand.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 7. the asks
interface Ask { ask: string; answer: ReactNode; chart?: [string, string]; state: 'Done' | 'Open' | 'Later' }
function Asks({ drain }: { drain: Drain | null }) {
  const rows: Ask[] = [
    { ask: 'The bend on all IMPROVE data, not only grouped medians', state: 'Done', chart: ['hl-bend', 'the bend'], answer: 'Every filter shaded, with the medians and a single-filter fit; the table under it refits on every filter and from 10 and 20 µg up. All of them bend.' },
    { ask: 'How the groups are made, and whether the grouping matters', state: 'Done', chart: ['wp-groups', 'the groups by period'], answer: 'The groups by period and season, every group in every year, the bend in five-year windows, and the same under other groupings.' },
    { ask: 'Only data in Warren’s trusted range', state: 'Done', chart: ['wp-range', 'trusted samples'], answer: 'Ceiling barely moves; the masked filters that were in are named. Clog flags still to apply.' },
    { ask: 'The same in Fabs, not τ, for IMPROVE and Addis', state: 'Done', chart: ['hc-fabs', 'in Fabs'], answer: 'Still bends. It is a loading effect, so each sampled volume has its own curve in Fabs.' },
    { ask: 'Write down the equation', state: 'Done', chart: ['wp-curves', 'the curves'], answer: 'Ours: τ = x / (1 + x/τ∞), x = aL. Warren’s: τ = −ln[h e^(−αx/ω) + (1 − h) e^(−x/ω)], ω = 1 + (α − 1)h. Ours is saturating, not quadratic.' },
    { ask: 'How did Warren get there, and how did we?', state: 'Done', chart: ['wp-routes', 'the two routes'], answer: 'Step by step, where they use opposite yardsticks, and Warren’s own CSN test rerun on the public pairs.' },
    { ask: 'Does Warren already say this?', state: 'Done', chart: ['wp-claims', 'claims'], answer: 'Yes (2025). We extend it: darker filters than his, and a bend larger than his screen can make in July to October.' },
    { ask: 'Where do we start correcting?', state: 'Done', chart: ['wp-factors', 'where to start'], answer: 'τ 0.45 is defensible from Warren’s upper bound; 1.02 is where no bound exists.' },
    { ask: 'A switch to correct only filters past a τ', state: 'Done', answer: 'Plot, HIPS, Linearised, Apply to: Filters past a τ.' },
    { ask: 'Addis and Delhi with only the light filters', state: 'Done', chart: ['hc-cut', 'below the cut'], answer: 'Delhi’s light filters need no correction (0.90x + 0.14, MAC 11). Addis has none below the cut.' },
    { ask: 'How uncertain is the correction?', state: 'Done', chart: ['wp-split', 'the screen split'], answer: 'The ceiling’s interval understates it: the screen question alone moves the Addis MAC from about 6 to 19.' },
    { ask: 'What is the exponent b, on the plot', state: 'Done', answer: 'b in Fabs ∝ (FTIR EC)^b; 1 is proportional, below 1 is compressed. Defined in the title of the 10 Nov exponent chart; the slide still needs it.' },
    { ask: 'Say compromised, not lost', state: 'Done', chart: ['wp-factors', 'where to start'], answer: 'Used here; check the deck.' },
    { ask: 'Name every axis on the slides', state: 'Open', answer: 'The deck is not in the repository.' },
    { ask: 'Which six urban IMPROVE sites, with a source', state: 'Open', answer: 'Atlanta, Birmingham, Fresno, Phoenix, Pittsburgh, Puget Sound. White (2025) names four collocated with CSN; Murphy et al. (2008) for the rest, to be read.' },
    { ask: 'Aethalometer at sites not adjusted to FTIR', state: 'Later', answer: 'Deferred a few weeks by Ann.' },
    { ask: 'Lower the loading without hurting FTIR', state: 'Open', answer: 'Not answered here: needs the FTIR detection limit against volume at Addis.' },
  ]
  const color = { Done: '#4f9d69', Open: '#b5446e', Later: INK.muted }
  return (
    <ChartFrame id="wp-asks" title="What Ann asked on 19 November, and where each answer is" provenance={<>19 Nov 2026 meeting transcript; <code>docs/hips-bend-checks-2026-11-20.md</code>.</>}>
      <DataTable rows={rows} rowKey={(r) => r.ask} columns={[
        { key: 'a', label: 'Asked', value: (r) => <strong>{r.ask}</strong> }, { key: 's', label: 'State', value: (r) => <span style={{ color: color[r.state], fontWeight: 700 }}>{r.state}</span> },
        { key: 'w', label: 'Answer', value: (r) => r.answer }, { key: 'c', label: 'Chart', value: (r) => (r.chart ? <ToChart id={r.chart[0]}>{r.chart[1]}</ToChart> : '') },
      ]} />
      {PRIVATE_BUILD && drain && <Note>Not on Ann's list, and new: the Phoenix drain-disc pairs, <ToChart id="wp-drain">part 4</ToChart>.</Note>}
    </ChartFrame>
  )
}

function Questions({ drain }: { drain: Drain | null }) {
  return (
    <section className="panel" id="wp-questions">
      <header className="frame-head"><div><h2>What would settle it, and who can</h2></div></header>
      <ul>
        <li><strong>SPARTAN laboratory:</strong> the support screen's hole share and hole size, flow through the filter, and photographs of a light and a dark Addis deposit. This decides which end of the screen split we are at.</li>
        <li><strong>Warren:</strong> the amplitude measured on individual IMPROVE deposits; what pattern he expects at SPARTAN's flow; whether the corrected IMPROVE record is out and with which α.</li>
        {PRIVATE_BUILD && drain && <li><strong>Warren, privately:</strong> are the Phoenix drain-disc deposits really even, and may we use those filters?</li>}
        <li><strong>CSN, now done for August 2020 on:</strong> the four collocated cities (<ToChart id="wp-csn">Warren’s test</ToChart>). Left to ask: why IMPROVE reads about 10 % above CSN on light filters, and Warren’s own 2019–2022 pairs, which start earlier than EPA’s file.</li>
        <li><strong>The Addis TOR filters:</strong> a curve fitted on Addis's own filters against thermal EC, which needs no transfer at all.</li>
      </ul>
    </section>
  )
}
