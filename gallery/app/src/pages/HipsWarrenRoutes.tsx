import { useMemo, type ReactNode } from 'react'
import { ChartFrame, Note } from '@/components/ChartFrame'
import { Segmented, Select } from '@/components/Controls'
import { LoadState } from '@/components/ExportState'
import { IntervalStrip, type IntervalMark, type IntervalRow } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { Marker, RefLine } from '@/components/RefLines'
import { ScatterPlot, type ScatterPoint, type ScatterScales } from '@/components/ScatterPlot'
import { useExport } from '@/lib/data'
import { fmt } from '@/lib/format'
import { curveModel, factorOf, fitModel, IMPROVE_HOLES, screenModel, WHITE_ALPHA, type HipsModel } from '@/lib/hipsPattern'
import { RESEARCH_SERIES } from '@/lib/palettes'
import { median, quantile } from '@/lib/stats'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { pathOf, range, SIDE, type Ceil, type Linearise } from '@/pages/HipsLinearisePage'
import { DataTable, Readouts, type Column } from '@/pages/hipsLineariseParts'
import { usePeriods } from '@/pages/HipsWarrenPeriods'
import { C, line, macText, pct, ToChart, type Warren } from '@/pages/HipsWarrenParts'

/**
 * How White et al. got to α = 10 and how we got to τ∞ 2.26: the two routes step by step, where each has filters to stand on,
 * White's own test (IMPROVE against CSN) run on the public pairs, and what every answer from either route does to Addis.
 * Data: gallery/data/export_hips_csn.py, warren.json, improve_periods.json.
 */
const FILE = 'hips_linearise/csn.json'
const COMMAND = 'uv run python gallery/data/export_hips_csn.py --aqs data/external/aqs_daily_spec'
type FitKey = 'curve' | 'alpha' | 'curve1' | 'alpha1' | 'alpha10' | 'line'
interface CsnFit { p: number[]; lo: number[]; hi: number[]; rms_log: number; aic: number }
interface CsnSet { key: string; label: string; pairs: number; groups: { n: number; x: number; tau: number; lo?: number; hi?: number }[]; fits: Record<FitKey, CsnFit> }
interface Csn {
  n_boot: number; first: string; last: string; min_n: number
  sites: { code: string; name: string; aqs: string; pairs: number }[]
  sets: CsnSet[]; by: { key: string; label: string; pairs: number; fits: Record<FitKey, CsnFit> }[]
  means: { key: string; dark: 'all' | 'below'; label: string; pairs: number; past: number; rows: { key: string; mean: number; ratio: number; lo: number; hi: number }[]; darkest_2pct_share: number }[]
  published: { csn: number; improve_alpha10: number; pairs: number }
  range: { improve_tau: number[]; csn_tau: number[] }
  /** site index, date, IMPROVE τ, CSN Fabs over IMPROVE Fabs */
  pairs: [number, string, number, number][]
}
export const useCsn = () => useExport<Csn>(FILE)
const PROV = <><code>{FILE}</code> (<code>gallery/data/export_hips_csn.py</code>): EPA AQS Fabs (parameter 63102, Met One SASS) and IMPROVE HIPS on standard 3.53 cm² deposits. Public data.</>
const PAPER = <>White et al. (2025, JAWMA 75, 209–232) and White et al. (2016, AST 50, 984–1002); ours from <code>docs/addis-reference-closure-2026-11-10.md</code>.</>
const REST_COLOR = RESEARCH_SERIES[3], CSN_COLOR = RESEARCH_SERIES[5]
/** the published ceiling's 90 % interval */
const PUBLISHED: [number, number] = [1.8212, 2.7014]
const ci = (f: CsnFit, i: number, d = 2) => `${fmt(f.p[i], d)} [${fmt(f.lo[i], d)}, ${f.hi[i] > 1000 ? 'no limit' : fmt(f.hi[i], d)}]`

export function RouteCharts({ data, w, ceil }: { data: Linearise; w: Warren; ceil: Ceil }) {
  const { data: csn, error } = useCsn()
  if (!csn) return <ChartFrame id="wp-routes" title="How Warren got there, and how we got there" provenance={PAPER}><LoadState what="the CSN pairs" error={error} command={COMMAND} /></ChartFrame>
  return <>
    <Routes w={w} ceil={ceil} csn={csn} data={data} />
    <Evidence csn={csn} w={w} data={data} />
    <WarrensTest csn={csn} ceil={ceil} data={data} />
    <Meet csn={csn} ceil={ceil} data={data} />
  </>
}

// ------------------------------------------------------------------ 1. the two routes, step by step
interface Step { step: string; them: ReactNode; us: ReactNode; kind: 'Same' | 'Different' | 'Opposite'; says: ReactNode }
const KIND: Record<Step['kind'], string> = { Same: '#4f9d69', Different: '#d08c2c', Opposite: '#b5446e' }

function Routes({ w, ceil, csn, data }: { w: Warren; ceil: Ceil; csn: Csn; data: Linearise }) {
  const raw = fitModel(data.filters, data.lambda, { invert: (t) => t }, data.mac)
  const ours = fitModel(data.filters, data.lambda, curveModel('k', '', '', ceil.ceiling), data.mac)
  const all = csn.sets.find((s) => s.key === 'all')!
  const rows: Step[] = [
    { step: 'The question', kind: 'Different', says: 'Their problem was the record; ours was one site.',
      them: <>Are 20 years of reported IMPROVE Fabs biased by the dots on the filters?</>,
      us: <>Why does FTIR EC at Addis read {line(raw)} against HIPS BC? Which of the two is not proportional?</> },
    { step: 'The first clue', kind: 'Different', says: 'They saw the deposit; we saw the response.',
      them: <>Micrographs: the soot sits in dots over the screen's holes. The instrument itself is right: even gel films of absorptance ½, ¾ and ⅞ read as they should.</>,
      us: <>On IMPROVE, HIPS absorption per thermal EC falls from 13.1 m²/g on the lightest filters to 8.4 on the darkest. <ToChart id="hl-bend">the bend</ToChart></> },
    { step: 'Which came first', kind: 'Opposite', says: 'Geometry first, then a number. Data first, then a reason.',
      them: <>The model. Two thicknesses, with the hole share known ({IMPROVE_HOLES}) and one unknown, α, the thick over the thin.</>,
      us: <>The data. A two-number curve: absorption per EC on a light filter, and the ceiling τ∞. No cause assumed.</> },
    { step: 'What is trusted as straight', kind: 'Opposite', says: 'Neither uses the other’s yardstick, so agreement means something.',
      them: <>HIPS on a light filter. CSN draws the same air at about a tenth of the darkness, and its deposits look even, so α is taken as 1 there.</>,
      us: <>Thermal (TOR) EC, at any loading. No CSN, no FTIR and no Addis filter enters.</> },
    { step: 'The filters', kind: 'Different', says: 'Few and dark against many and mostly light.',
      them: <>{csn.published.pairs.toLocaleString()} same-day pairs at four cities, 2019–2022. <ToChart id="wp-csn">rerun below</ToChart></>,
      us: <>347,936 filters at 210 sites, 2005–2025, put into 17 groups by the previous filter's EC. <ToChart id="wp-groups">the groups</ToChart></> },
    { step: 'The years', kind: 'Same', says: 'Ours sit inside the years Warren trusts; the newest are past both papers.',
      them: <>HIPS is stable from 2003 on. The 2016 paper uses 2005–2014 (masked filters were gone by the end of 2007, and TOR changed in 2005); the 2025 paper uses 2010–2020, and 2019–2022 against CSN.</>,
      us: <>2005–2025, where our TOR and HIPS tables overlap. On Warren's own years the ceiling is {fmt(w.sets.find((z) => z.key === 'white2016')!.tau_inf, 2)} (2005–2014) and {fmt(w.sets.find((z) => z.key === 'white2025')!.tau_inf, 2)} (2010–2020). <ToChart id="wp-range">the trusted samples</ToChart></> },
    { step: 'How the number was set', kind: 'Different', says: 'One number chosen so two means agree; one fitted to a shape, with an interval.',
      them: <>α = {WHITE_ALPHA} makes the two networks' means agree: {fmt(csn.published.improve_alpha10, 2)} and {fmt(csn.published.csn, 2)} Mm⁻¹. No interval is given.</>,
      us: <>Least squares through the 17 medians: τ∞ 2.26, 90 % interval {fmt(PUBLISHED[0], 2)} to {fmt(PUBLISHED[1], 2)} from resampling whole sites.</> },
    { step: 'The equation', kind: 'Same', says: 'Both are Beer–Lambert averaged over an uneven deposit.',
      them: <>τ = −ln[h e^(−αx/ω) + (1 − h) e^(−x/ω)], ω = 1 + (α − 1)h. Thickness variance 0.41.</>,
      us: <>τ = x / (1 + x/τ∞). Thickness variance {fmt(2 / ceil.ceiling, 2)}. <ToChart id="wp-curves">the curves</ToChart></> },
    { step: 'What was done with it', kind: 'Different', says: 'They bounded a network; we corrected a site.',
      them: <>Rural mean Fabs up 11 %. Past τ {fmt(w.limit_tau, 2)} they give no upper bound and expect α = {WHITE_ALPHA} to over-correct some samples.</>,
      us: <>Addis: {line(raw)} becomes {line(ours)}. Every Addis filter is past τ 0.45, and 40 % are past τ {fmt(w.limit_tau, 2)}.</> },
    { step: 'The weak point', kind: 'Opposite', says: 'Each is strong where the other is weak.',
      them: <>One α for every filter, and means carried by the darkest few: the darkest 2 % of days hold {pct(all ? csn.means.find((m) => m.key === 'all' && m.dark === 'all')!.darkest_2pct_share : NaN)} of CSN's total.</>,
      us: <>Anything that bends TOR EC bends the curve too, and the curve is fitted mostly below τ 1. <ToChart id="wp-season-addis">July to October</ToChart> shows it.</> },
  ]
  const head = { fontSize: 12.5, fontWeight: 700, padding: '6px 10px', borderBottom: '2px solid var(--ink-border)' } as const
  return (
    <ChartFrame id="wp-routes" title="Two routes to the same bend — Warren started from the screen and tested it on CSN; we started from Addis and fitted it on thermal EC"
      subtitle="Read across for one step, down for one story. The chip says how the two compare at that step."
      provenance={PAPER}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(250px, 100%), 1fr))', columnGap: 12, marginBottom: 4 }}>
        <div style={{ ...head, color: C.alpha10, borderColor: C.alpha10 }}>Warren White (2016, 2025)</div>
        <div style={{ ...head, color: C.curve, borderColor: C.curve }}>Ours (November 2026)</div>
      </div>
      {rows.map((r) => <div key={r.step} style={{ padding: '10px 0', borderBottom: '1px solid var(--ink-border)' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', gap: 8, marginBottom: 4 }}>
          <strong style={{ fontSize: 13 }}>{r.step}</strong>
          <span style={{ fontSize: 11, fontWeight: 700, color: KIND[r.kind], border: `1px solid ${KIND[r.kind]}`, borderRadius: 10, padding: '0 7px' }}>{r.kind}</span>
          <span style={{ fontSize: 12.5, color: 'var(--ink-muted)' }}>{r.says}</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(250px, 100%), 1fr))', gap: 12, fontSize: 13, lineHeight: 1.45 }}>
          <div style={{ borderLeft: `3px solid ${C.alpha10}`, paddingLeft: 10 }}>{r.them}</div>
          <div style={{ borderLeft: `3px solid ${C.curve}`, paddingLeft: 10 }}>{r.us}</div>
        </div>
      </div>)}
      <Note>They meet at the equation, and at <ToChart id="wp-csn">Warren's test</ToChart>: our curve fitted to their kind of pairs, with no TOR in the numbers, gives a ceiling of {ci(all.fits.curve, 1)}, or {ci(all.fits.curve1, 0)} if the networks may not differ by a constant, against 2.26 [1.82, 2.70] from TOR.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 2. where each has filters to stand on
function Evidence({ csn, w, data }: { csn: Csn; w: Warren; data: Linearise }) {
  const addis = data.filters.filter((f) => f.ftir != null).map((f) => f.tau)
  const med = w.medians.map((m) => m.tau)
  const strip = (q: number[], color: string): IntervalMark[] => [{ mid: q[2], lo: q[0], hi: q[4], color }]
  const row = (key: string, label: string, marks: IntervalMark[], tip: string[], strong = false): IntervalRow => ({ key, label, strong, marks: [marks], tip: [label, ...tip] })
  const five = (q: number[]) => `5th to 95th percentile τ ${fmt(q[0], 2)} to ${fmt(q[4], 2)}, median ${fmt(q[2], 2)}`
  const a = [0.05, 0.25, 0.5, 0.75, 0.95].map((v) => quantile(addis, v))
  const blocks = [
    { key: 'w', title: 'Warren', color: C.alpha10, rows: [
      row('gel', 'Gel films: the instrument on an even absorber', [0.5, 0.75, 0.875].map((A) => ({ mid: -Math.log(1 - A), color: C.alpha10, shape: 'hollow' as const })), ['absorptance 1/2, 3/4 and 7/8 (White et al. 2016): read correctly']),
      row('csn', 'CSN filters at the four cities (their yardstick)', strip(csn.range.csn_tau, C.alpha10), [five(csn.range.csn_tau)]),
      row('imp', 'IMPROVE filters of the same days', strip(csn.range.improve_tau, C.alpha10), [five(csn.range.improve_tau)]),
    ] },
    { key: 'u', title: 'Ours', color: C.curve, rows: [
      row('med', 'The 17 IMPROVE group medians the curve is fitted to', med.map((m) => ({ mid: m, color: C.curve, r: 3 })), [`group medians τ ${fmt(Math.min(...med), 2)} to ${fmt(Math.max(...med), 2)}`]),
      row('etad', 'Addis filters', strip(a, INK.text), [five(a)], true),
    ] },
  ]
  return (
    <ChartFrame id="wp-evidence" title={`Where each route has filters to stand on — half of Addis is past our darkest group median (τ ${fmt(Math.max(...med), 2)}), and inside the range of Warren's city pairs`}
      subtitle="Optical depth of the filters behind each step. Bars are the 5th to 95th percentile with the median; rings and small dots are single values."
      provenance={PROV}>
      <IntervalStrip labelWidth={300} rowHeight={28} blocks={blocks} panels={[{ label: 'HIPS optical depth τ', domain: [0, 2.2], marker: { value: w.limit_tau, label: `IMPROVE screen limit ${fmt(w.limit_tau, 2)}` } }]} />
      <Note>The median Addis filter is at τ {fmt(median(addis), 2)}. Our medians stop at τ {fmt(Math.max(...med), 2)}, so half of Addis rests on the curve's shape. The city pairs reach as dark as Addis, so they test that part directly.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 3. White's test, on the public pairs
function WarrensTest({ csn, ceil, data }: { csn: Csn; ceil: Ceil; data: Linearise }) {
  const [key, setKey] = useSearchParam<string>('wp.csn', 'all', csn.sets.map((s) => s.key))
  const [scale, setScale] = useSearchParam<'free' | 'one'>('wp.csns', 'free', ['free', 'one'])
  const { data: periods } = usePeriods()
  const rest = periods?.pooled.find((p) => p.season === 'rest')?.tau_inf
  const set = csn.sets.find((s) => s.key === key) ?? csn.sets[0]
  const s = scale === 'free' ? set.fits.curve.p[0] : 1
  const site = csn.sites.findIndex((x) => x.code === set.key)
  const inSet = useMemo(() => csn.pairs.filter(([i, d]) => {
    const m = +d.slice(5, 7), fire = m >= 7 && m <= 10
    return set.key === 'all' || (set.key === 'rest' && !fire) || (set.key === 'fire' && fire) || (set.key === 'white' && d < '2023') || (set.key === 'later' && d >= '2023') || i === site
  }), [csn, set.key, site])
  const points = useMemo<ScatterPoint[]>(() => [
    ...inSet.map(([i, d, t, k], n) => ({ key: `p${n}`, x: t, y: k * s, color: INK.muted, r: 1.8, opacity: 0.22, tip: [`${csn.sites[i].name}, ${d}`, `IMPROVE τ ${fmt(t, 2)}`, `CSN ÷ IMPROVE Fabs ${fmt(k, 2)}${s === 1 ? '' : `, × ${fmt(s, 2)} = ${fmt(k * s, 2)}`}`] })),
    ...set.groups.map((g, n) => ({ key: `g${n}`, x: g.tau, y: (g.x / g.tau) * s, color: CSN_COLOR, r: 5.5, opacity: 1, tip: [`Group of ${g.n} days`, `IMPROVE τ ${fmt(g.tau, 2)}; CSN implies ${fmt(g.x * s, 2)}`, `factor needed × ${fmt((g.x / g.tau) * s, 2)}${g.lo != null ? ` [${fmt(g.lo * s, 2)}, ${fmt(g.hi! * s, 2)}]` : ''}`] })),
  ], [inSet, set, s, csn.sites])
  const models: { label: string; color: string; m: HipsModel; dash?: string }[] = [
    { label: `Our curve, τ∞ ${fmt(ceil.ceiling, 2)} (TOR, all months)`, color: C.curve, m: curveModel('c', '', '', ceil.ceiling) },
    ...(rest ? [{ label: `IMPROVE TOR, November to June, τ∞ ${fmt(rest, 2)}`, color: REST_COLOR, m: curveModel('r', '', '', rest), dash: '7 4' }] : []),
    { label: `Warren, α = ${WHITE_ALPHA}`, color: C.alpha10, m: screenModel('a', '', '', WHITE_ALPHA) },
  ]
  const addis = data.filters.filter((f) => f.ftir != null).map((f) => f.tau)
  const under = ({ x, y, iw, ih }: ScatterScales) => <g pointerEvents="none">
    <clipPath id="wp-csn-clip"><rect width={iw} height={ih} /></clipPath>
    <rect x={x(quantile(addis, 0.05))} width={x(quantile(addis, 0.95)) - x(quantile(addis, 0.05))} y={0} height={ih} fill={INK.accent} opacity={0.07} />
    <text x={x(quantile(addis, 0.05)) + 6} y={ih - 8} fontSize={11} fill={INK.muted}>Addis filters, 5th to 95th percentile</text>
    <RefLine orient="h" at={y(1)} from={0} to={iw} />
    <g clipPath="url(#wp-csn-clip)">
      {models.map((m) => <path key={m.label} d={pathOf(range(0.02, 1.75, 0.01).map((t) => [x(t), y(factorOf(m.m, t))]))} fill="none" stroke={m.color} strokeWidth={2.25} strokeDasharray={m.dash} />)}
      {set.groups.map((g, i) => g.lo != null && <line key={i} x1={x(g.tau)} x2={x(g.tau)} y1={y(g.lo * s)} y2={y(g.hi! * s)} stroke={CSN_COLOR} strokeWidth={1.8} />)}
    </g>
    <Marker orient="v" at={x(1.016)} from={0} to={ih} color={C.full} label="screen limit 1.02" />
  </g>
  const best = Math.min(...(['curve', 'alpha', 'curve1', 'alpha1', 'alpha10', 'line'] as FitKey[]).map((k) => set.fits[k].aic))
  const fits: { k: FitKey; label: string; value: string }[] = [
    { k: 'curve', label: 'Our curve, networks may differ by a constant', value: `τ∞ ${ci(set.fits.curve, 1)}, constant ${fmt(set.fits.curve.p[0], 2)}` },
    { k: 'curve1', label: 'Our curve, no constant', value: `τ∞ ${ci(set.fits.curve1, 0)}` },
    { k: 'alpha', label: "Warren's model, α free, networks may differ by a constant", value: `α ${set.fits.alpha.p[1] > 1000 ? 'no limit' : ci(set.fits.alpha, 1, 1)}, constant ${fmt(set.fits.alpha.p[0], 2)}` },
    { k: 'alpha1', label: "Warren's model, α free, no constant", value: `α ${set.fits.alpha1.p[0] > 1000 ? 'no limit' : ci(set.fits.alpha1, 0, 1)}` },
    { k: 'alpha10', label: `Warren's α = ${WHITE_ALPHA}, networks may differ by a constant`, value: `constant ${fmt(set.fits.alpha10.p[0], 2)}` },
    { k: 'line', label: 'No bend at all', value: `constant ${fmt(set.fits.line.p[0], 2)}` },
  ]
  type M = Csn['means'][number]
  const of = (m: M, k: string) => { const r = m.rows.find((z) => z.key === k)!; return r.ratio > 1.6 ? 'blows up' : `${fmt(r.mean, 2)} (× ${fmt(r.ratio, 2)})` }
  const columns: Column<M>[] = [
    { key: 'l', label: 'Days', value: (m) => `${m.label}${m.dark === 'below' ? ', τ below 1.5' : ''}` }, { key: 'n', label: 'Pairs', value: (m) => m.pairs.toLocaleString(), num: true },
    { key: 'c', label: 'CSN', value: (m) => fmt(m.rows[0].mean, 2), num: true }, { key: 'r', label: 'IMPROVE as reported', value: (m) => of(m, 'none'), num: true },
    { key: 'a', label: `α = ${WHITE_ALPHA}`, value: (m) => of(m, 'alpha10'), num: true }, { key: 'o', label: 'our curve 2.26', value: (m) => of(m, 'curve'), num: true },
    { key: 'j', label: 'Nov–Jun curve 3.26', value: (m) => of(m, 'rest'), num: true },
  ]
  const all = csn.sets[0], later = csn.means.find((m) => m.key === 'later' && m.dark === 'all')!
  return (
    <ChartFrame id="wp-csn" title={`Warren's test, rerun on the public data — against CSN, with no TOR EC in the values and no FTIR, the bend is there at our size: τ∞ ${fmt(all.fits.curve.p[1], 1)} to ${fmt(all.fits.curve1.p[0], 1)}, beside 2.26 [1.82, 2.70] from TOR`}
      subtitle={`Each grey dot is one day at Birmingham, Fresno, Phoenix or Seattle (${csn.first.slice(0, 7)} to ${csn.last.slice(0, 7)}): how many times more Fabs CSN reports than IMPROVE, against how dark the IMPROVE filter is. Large dots: medians of days grouped by IMPROVE's TOR EC that day, which only sorts the days, with 90 % intervals from resampling site-months. Lines: the factor each correction would apply; none was fitted to these pairs. A correction is right where its line passes through the medians.`}
      provenance={PROV}
      controls={<><Select label="Days" value={set.key} options={csn.sets.map((z) => z.key)} onChange={setKey} optionLabel={(k) => csn.sets.find((z) => z.key === k)?.label ?? k} />
        <Segmented label="Networks may differ by a constant" value={scale} options={['free', 'one'] as const} onChange={setScale} optionLabel={(v) => (v === 'free' ? `Yes (× ${fmt(set.fits.curve.p[0], 2)})` : 'No')} title="A constant is a difference that is the same on a light and a dark filter (inlet, deposit area, blanks). It cannot make a bend." /></>}>
      <ScatterPlot points={points} xLabel="Optical depth of the IMPROVE filter, as reported" yLabel="Factor needed: CSN Fabs ÷ IMPROVE Fabs" xDomain={[0, 1.8]} yDomain={[0.5, 3.5]} nice={false} maxSide={SIDE} outside="pin" note={false} under={under} />
      <Key items={[{ label: 'Group medians', color: CSN_COLOR, kind: 'dot' }, { label: 'One day', color: INK.muted, kind: 'faded' }, ...models.map((m) => ({ label: m.label, color: m.color, kind: m.dash ? 'dash' as const : 'line' as const }))]} />
      <Readouts items={[
        { label: 'Our curve fitted to these pairs', value: `τ∞ ${fmt(set.fits.curve.p[1], 2)}`, color: CSN_COLOR, sub: `[${fmt(set.fits.curve.lo[1], 2)}, ${fmt(set.fits.curve.hi[1], 2)}], constant ${fmt(set.fits.curve.p[0], 2)}` },
        { label: 'With no constant', value: `τ∞ ${fmt(set.fits.curve1.p[0], 2)}`, color: CSN_COLOR, sub: `[${fmt(set.fits.curve1.lo[0], 2)}, ${fmt(set.fits.curve1.hi[0], 2)}]` },
        { label: 'α fitted to these pairs', value: set.fits.alpha.p[1] > 1000 ? 'no limit' : fmt(set.fits.alpha.p[1], 1), color: C.alpha10, sub: `[${fmt(set.fits.alpha.lo[1], 1)}, ${set.fits.alpha.hi[1] > 1000 ? 'no limit' : fmt(set.fits.alpha.hi[1], 1)}]; published ${WHITE_ALPHA}` },
        { label: `Warren's mean test after the paper`, value: `× ${fmt(later.rows.find((r) => r.key === 'alpha10')!.ratio, 2)}`, sub: `IMPROVE with α = ${WHITE_ALPHA} over CSN, 2023–2025` },
      ]} />
      <DataTable rows={fits} rowKey={(r) => r.k} highlight={(r) => r.k === 'curve'} columns={[
        { key: 'l', label: `Fitted to the ${set.groups.length} medians (${set.label.toLowerCase().startsWith('all') ? 'all pairs' : set.label})`, value: (r) => r.label }, { key: 'v', label: 'Value, 90 % interval', value: (r) => r.value },
        { key: 'm', label: 'Typical miss', value: (r) => pct(set.fits[r.k].rms_log, 1), num: true }, { key: 'd', label: 'ΔAIC', value: (r) => fmt(set.fits[r.k].aic - best, 1), num: true },
      ]} />
      <p style={{ margin: '14px 4px 4px', fontSize: 13, fontWeight: 600 }}>The same test as the paper runs it: mean Fabs (Mm⁻¹), and IMPROVE over CSN</p>
      <DataTable rows={csn.means} rowKey={(m) => `${m.key}/${m.dark}`} columns={columns} />
      <Note>From τ 0.5 to 1.1 the days follow our curve, a little above it, and the α = {WHITE_ALPHA} line is too low. At the darkest median (τ {fmt(set.groups[set.groups.length - 1].tau, 1)}) they need × {fmt((set.groups[set.groups.length - 1].x / set.groups[set.groups.length - 1].tau) * s, 1)}:
        α = {WHITE_ALPHA} gives × {fmt(factorOf(models[models.length - 1].m, set.groups[set.groups.length - 1].tau), 1)} and our curve × {fmt(factorOf(models[0].m, set.groups[set.groups.length - 1].tau), 1)}, so ours over-corrects the very darkest filters. A dark city can stop short of that group.</Note>
      <Note>Published: {fmt(csn.published.csn, 2)} for CSN and {fmt(csn.published.improve_alpha10, 2)} for IMPROVE with α = {WHITE_ALPHA}, on 2019–2022. EPA's public file starts in August 2020. Our curve has no finite mean when a filter sits at its ceiling, so it
        cannot run the mean test on all days; without the darkest few it overshoots CSN, because IMPROVE's light filters already read about {fmt(100 * (all.fits.curve.p[0] - 1), 0)} % above CSN's. That constant is why the test is fitted both ways.</Note>
      <DataTable rows={csn.by} rowKey={(r) => r.key} highlight={(r) => r.key === 'loading_ug'} columns={[
        { key: 'l', label: 'Days sorted by', value: (r) => r.label }, { key: 'c', label: 'Our curve, constant free', value: (r) => `τ∞ ${ci(r.fits.curve, 1)}` }, { key: 'o', label: 'No constant', value: (r) => `τ∞ ${ci(r.fits.curve1, 0)}` },
        { key: 'a', label: 'α, constant free', value: (r) => (r.fits.alpha.p[1] > 1000 ? 'no limit' : ci(r.fits.alpha, 1, 1)) },
      ]} />
      <Note explain>Sorting by either HIPS reading selects on that reading's own error: IMPROVE's hides the bend, CSN's exaggerates it at both ends. IMPROVE's previous filter hardly sorts days at only four sites. TOR EC that day shares
        no error with either HIPS reading, so it is the one used; the others are here so the choice can be seen: the ceiling runs from about 1.5 to 3.3 across them. ΔAIC under about 2 to 4 says the bent models cannot be told apart; only the straight line is excluded. CSN filters are treated as even, as White et al. do. If they bend a little too, every factor here is a lower bound.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 4. every answer, on one scale
function Meet({ csn, ceil, data }: { csn: Csn; ceil: Ceil; data: Linearise }) {
  const { data: periods } = usePeriods()
  const tau = median(data.filters.filter((f) => f.ftir != null).map((f) => f.tau))
  const blocks = useMemo(() => {
    const row = (key: string, label: string, color: string, mk: (v: number) => HipsModel, mid: number, lo?: number, hi?: number, name = 'τ∞', strong = false): IntervalRow => {
      const at = (v: number) => { const m = mk(v); return { m, k: factorOf(m, tau), f: fitModel(data.filters, data.lambda, m, data.mac) } }
      const c = at(mid), ends = lo != null && hi != null ? [at(lo), at(hi)] : []
      const span = (g: (e: typeof c) => number | undefined, cap = Infinity) => { const v = ends.map(g).filter((z): z is number => z != null && Number.isFinite(z)); return v.length === 2 ? { lo: Math.min(...v), hi: Math.min(cap, Math.max(...v)) } : {} }
      return { key, label, strong, marks: [
        [{ mid: c.k, ...span((e) => e.k, 2.2), color }], [{ mid: c.f?.intercept, ...span((e) => e.f?.intercept), color }], [{ mid: c.f?.mac, ...span((e) => e.f?.mac, 30), color }],
      ], tip: [label, `${name} ${fmt(mid, name === 'α' ? 1 : 2)}${lo != null ? ` [${fmt(lo, name === 'α' ? 1 : 2)}, ${fmt(hi!, name === 'α' ? 1 : 2)}]` : ', no interval published'}`, `factor at τ ${fmt(tau, 2)}: × ${fmt(c.k, 2)}`, `Addis ${line(c.f)}, MAC for slope 1 ${macText(c.f)}`] }
    }
    const curve = (c: number) => curveModel('k', '', '', c), screen = (a: number) => screenModel('k', '', '', a)
    const pooled = (k: string) => periods?.pooled.find((p) => p.season === k)
    const s = (k: string) => csn.sets.find((z) => z.key === k)!
    const r = pooled('rest')
    const alphaOf = (f: CsnFit, i: number) => [f.p[i], f.lo[i], Math.min(200, f.hi[i])] as const
    return [
      { key: 'u', title: 'Our route: IMPROVE against thermal EC', color: C.curve, rows: [
        row('page', 'All months, all sites (the page)', C.curve, curve, ceil.ceiling, PUBLISHED[0], PUBLISHED[1], 'τ∞', true),
        ...(r ? [row('rest', 'November to June', C.curve, curve, r.tau_inf, 1 / r.bend_hi, 1 / r.bend_lo)] : []),
        ...(r ? [row('restA', "November to June, Warren's model", C.curve, screen, r.models.find((m) => m.key === 'free')!.params[1], undefined, undefined, 'α')] : []),
      ] },
      { key: 'w', title: "Warren's route: IMPROVE against CSN", color: C.alpha10, rows: [
        row('pub', `Published, α = ${WHITE_ALPHA}`, C.alpha10, screen, WHITE_ALPHA, undefined, undefined, 'α', true),
        row('a', 'Pairs rerun, α, constant free', C.alpha10, screen, ...alphaOf(s('all').fits.alpha, 1), 'α'),
        row('a1', 'Pairs rerun, α, no constant', C.alpha10, screen, ...alphaOf(s('all').fits.alpha1, 0), 'α'),
      ] },
      { key: 'x', title: 'Both: our curve on Warren’s pairs', color: CSN_COLOR, rows: [
        row('c', 'All pairs, constant free', CSN_COLOR, curve, s('all').fits.curve.p[1], s('all').fits.curve.lo[1], s('all').fits.curve.hi[1], 'τ∞', true),
        row('c1', 'All pairs, no constant', CSN_COLOR, curve, s('all').fits.curve1.p[0], s('all').fits.curve1.lo[0], s('all').fits.curve1.hi[0]),
        row('cr', 'November to June, constant free', CSN_COLOR, curve, s('rest').fits.curve.p[1], s('rest').fits.curve.lo[1], s('rest').fits.curve.hi[1]),
        row('cf', 'July to October, constant free', CSN_COLOR, curve, s('fire').fits.curve.p[1], s('fire').fits.curve.lo[1], s('fire').fits.curve.hi[1]),
      ] },
    ]
  }, [csn, ceil.ceiling, data, periods, tau])
  const f = csn.sets.find((s) => s.key === 'fire')!.fits.curve, r = csn.sets.find((s) => s.key === 'rest')!.fits.curve
  const tor = periods?.pooled.find((p) => p.season === 'fire')
  return (
    <ChartFrame id="wp-meet" title="Where the two routes meet — every all-month or November to June answer multiplies the median Addis filter by about 1.4 to 1.75, and where in that range decides the intercept and the MAC"
      subtitle={`α and τ∞ are different units, so each answer is shown by what it does: the factor on an Addis filter of median darkness (τ ${fmt(tau, 2)}), then the Addis intercept and MAC after it. Bars are the 90 % interval of the fitted number alone, carried through. As measured: factor 1, intercept ${fmt(fitModel(data.filters, data.lambda, { invert: (t) => t }, data.mac)?.intercept, 1)}.`}
      provenance={<>{PROV} <code>hips_linearise/improve_periods.json</code>; {PAPER}</>}>
      <IntervalStrip labelWidth={250} rowHeight={27} minPanelWidth={180} blocks={blocks} panels={[
        { heading: 'Factor', label: 'Factor at the median Addis τ', domain: [1.2, 2.2] }, { heading: 'Addis intercept', label: 'Intercept (µg/m³)', domain: [-3, 3], reference: 0 }, { heading: 'MAC for slope 1', label: 'm²/g', domain: [7, 30], reference: data.mac }]} />
      <Note>{tor && <>Against TOR EC the July to October ceiling is {fmt(tor.tau_inf, 2)} [{fmt(1 / tor.bend_hi, 2)}, {fmt(1 / tor.bend_lo, 2)}]; against CSN it is {ci(f, 1)}, and {ci(r, 1)} in November to June, so at these four cities the months overlap. </>}
        That points to TOR, or to sites and smoke the cities do not have, for the extra July to October bend.</Note>
      <Note explain>The two routes do not share a yardstick, so their agreeing on the size at Addis's darkness is support for HIPS being what bends. It is not a result for the Addis MAC. All of it is IMPROVE's sampler and screen;
        SPARTAN's hole share and flow are still the untested step. The same factor at one darkness can come with different curvature, which is why the intercepts spread more than the factors.</Note>
    </ChartFrame>
  )
}
