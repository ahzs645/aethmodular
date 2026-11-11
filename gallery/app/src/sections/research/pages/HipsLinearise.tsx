import { useId, useMemo } from 'react'
import { ChartFrame, Note } from '@/components/ChartFrame'
import { Segmented, Select } from '@/components/Controls'
import { LoadState } from '@/components/ExportState'
import { IntervalStrip } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { Marker, RefLine } from '@/components/RefLines'
import { ScatterPlot, type ScatterPoint, type ScatterScales } from '@/components/ScatterPlot'
import { ScrubField } from '@/components/ScrubField'
import { useExport } from '@/lib/data'
import { fmt, signed } from '@/lib/format'
import { FixedSetNote } from '@/lib/markedRows'
import { RESEARCH_SERIES } from '@/lib/palettes'
import { median } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { SectionPageProps } from '../../types'
import { DataTable } from '../common/ui'
import { correction, fitAgainstHips, linearBc, linearise, noiseGain, predictedExponent, ratioMedians, saturate, type LinFilter } from '../common/linearise'

const COMMAND = 'uv run python gallery/data/export_research_hips_linearise.py'
const WORKFLOW = 'research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py'
const NOTE_DOC = 'docs/addis-reference-closure-2026-11-10.md'
const PROVENANCE = <>{WORKFLOW}; summary <code>{NOTE_DOC}</code> and its JSON; per-filter optical depths from the workflow's <code>load()</code> (<code>research/hips_linearise/linearise.json</code>).</>

const MEASURED = RESEARCH_SERIES[1]
const LINEAR = RESEARCH_SERIES[0]

type CeilingKey = 'central' | 'lo' | 'hi' | 'era'
const CEILING_KEYS: readonly CeilingKey[] = ['central', 'lo', 'hi', 'era']
const HIPS_VIEWS = ['measured', 'linearised'] as const
const SITE_VIEWS = ['clear', 'all'] as const
/** a site's exponent is read when its 95 % interval is narrower than this (12 of the 23 sites) */
const CLEAR_WIDTH = 0.5

interface Ceiling { population: string; samples: number; sites: number; years: string; tau_inf: number; tau_inf_lo: number; tau_inf_hi: number; unsaturated_fabs_over_ec: number; max_stratum_tau: number }
interface Linearise {
  generated: string
  deposit_area_cm2: number
  mac: number
  lambda: number
  a_per_ug: number
  ceilings: { key: CeilingKey; label: string; tau_inf: number }[]
  improve_ceilings: Ceiling[]
  strata: { n: number; loading_ug: number; tau: number; fabs_over_ec: number; fitted_tau: number }[]
  filters: LinFilter[]
  fit_measured: { slope: number; intercept: number; n: number }
  network: { site: string; code: string; n: number; median_optical_depth: number; exponent: number; lo: number; hi: number; predicted: number }[]
  season_lot: { season: string; lot: string; n: number; median_optical_depth: number; published_over_hips: number; published_over_linearised_hips: number; published_over_pm25: number }[]
}

/** the page's shared state: the ceiling every chart linearises with (URL `hl.ceil`) */
function useCeiling(data: Linearise | null) {
  const [key, setKey] = useSearchParam<CeilingKey>('hl.ceil', 'central', CEILING_KEYS)
  const list = data?.ceilings ?? []
  const pick = list.find((c) => c.key === key) ?? list[0]
  const control = <Segmented label="Ceiling τ∞" value={key} options={list.map((c) => c.key)} onChange={(v) => setKey(v as CeilingKey)}
    optionLabel={(o) => list.find((c) => c.key === o)?.label ?? o}
    title="The IMPROVE ceiling every chart on this page linearises with: the central fit, both ends of its 90 % interval, and the 2015–2025 fit (the period that contains the Addis record)" />
  return { ceiling: pick?.tau_inf ?? NaN, label: pick?.label ?? '', key, control, list }
}

/** a stroke path through (x, y) pairs in plot pixels, skipping non-finite points */
/** "y = 2.31x − 6.17" */
const eq = (f: { slope: number; intercept: number } | null | undefined) => (f ? `y = ${fmt(f.slope, 2)}x ${f.intercept < 0 ? '−' : '+'} ${fmt(Math.abs(f.intercept), 2)}` : '—')
const pathOf = (pts: [number, number][]) => pts.filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b)).map(([a, b], i) => `${i ? 'L' : 'M'}${a},${b}`).join('')
const range = (a: number, b: number, step: number) => Array.from({ length: Math.round((b - a) / step) + 1 }, (_, i) => a + i * step)

/**
 * What "HIPS linearised" means at Addis, step by step, as the 10 Nov closure note explains it: the IMPROVE bend that
 * sets the ceiling, the one-line inversion, how large the correction is on each Addis filter, what it does to the
 * FTIR EC fit and its intercept, the network check, and the season and lot ratios. One ceiling control drives every
 * chart; the per-filter numbers are recomputed in the browser (common/linearise.ts) and reproduce the workflow's
 * saved fits (linearise.test.mjs).
 */
export function HipsLinearise(_: SectionPageProps) {
  const { data, error } = useExport<Linearise>('research/hips_linearise/linearise.json')
  const ceil = useCeiling(data)
  if (error || !data) return (
    <ChartFrame id="hl-missing" title="Linearising HIPS at Addis" provenance={WORKFLOW}>
      <LoadState what="the linearisation export" error={error} command={COMMAND} />
    </ChartFrame>
  )
  return (
    <>
      <Bend data={data} ceil={ceil} />
      <Worked data={data} ceil={ceil} />
      <CorrectionCurve data={data} ceil={ceil} />
      <AgainstHips data={data} ceil={ceil} />
      <NetworkExponents data={data} />
      <SeasonLot data={data} ceil={ceil} />
      <Limits data={data} />
    </>
  )
}

type Ceil = ReturnType<typeof useCeiling>

/** a row of headline numbers under a chart */
function Readouts({ items }: { items: { label: string; value: string; color?: string; sub?: string }[] }) {
  return <div className="readouts">
    {items.map((it) => <div key={it.label} className="stat-tile">
      <div className="stat-label">{it.label}</div>
      <div className="stat-value"><strong style={it.color ? { color: it.color } : undefined}>{it.value}</strong></div>
      {it.sub && <div className="stat-sub">{it.sub}</div>}
    </div>)}
  </div>
}

// ------------------------------------------------------------------ 1. the bend
function Bend({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const [tauText, setTau] = useSearchParam<string>('hl.tau', '0.97')
  const taus = useMemo(() => data.filters.filter((f) => f.ftir != null).map((f) => f.tau), [data])
  const [lo, hi] = [Math.min(...taus), Math.max(...taus)]
  const t = Math.min(hi, Math.max(0.05, Number(tauText) || 0.97))
  const a = data.a_per_ug, c = ceil.ceiling
  const tl = linearise(t, c), L = tl / a
  const clip = useId()
  const points = useMemo<ScatterPoint[]>(() => data.strata.map((s) => ({
    x: s.loading_ug, y: s.tau, color: INK.muted, r: 3 + Math.sqrt(s.n) / 40, opacity: 0.85,
    tip: [`IMPROVE group, TOR EC ${fmt(s.loading_ug, 1)} µg`, `optical depth ${fmt(s.tau, 3)}`, `Fabs ÷ EC ${fmt(s.fabs_over_ec, 1)} m²/g`, `${s.n.toLocaleString()} samples`],
  })), [data])
  const under = ({ x, y, iw, ih }: ScatterScales) => {
    const xs = range(0, 90, 0.5)
    return <g>
      <defs><clipPath id={clip}><rect width={iw} height={ih} /></clipPath></defs>
      <g clipPath={`url(#${clip})`}>
        <rect x={0} y={y(hi)} width={iw} height={y(lo) - y(hi)} fill={MEASURED} opacity={0.1} />
        <text x={iw - 6} y={y(hi) - 6} textAnchor="end" fontSize={10.5} fontFamily={FONT.mono} fill={INK.muted}>Addis filters, τ {fmt(lo, 2)}–{fmt(hi, 2)}</text>
        <Marker orient="h" at={y(c)} from={0} to={iw} color={MEASURED} label={`ceiling τ∞ = ${fmt(c, 2)}`} />
        <path d={pathOf([[x(0), y(0)], [x(90), y(a * 90)]])} fill="none" stroke={LINEAR} strokeWidth={2} strokeDasharray="6 4" />
        <path d={pathOf(xs.map((v) => [x(v), y(saturate(a * v, c))]))} fill="none" stroke={MEASURED} strokeWidth={2} />
      </g>
    </g>
  }
  const children = ({ x, y, iw }: ScatterScales) => {
    if (!Number.isFinite(tl)) return null
    const onChart = x(L) < iw - 4
    // labels go left of the construction when it sits in the right part of the plot (a phone, a high ceiling)
    const flip = x(L) > iw - 110
    const dash = { stroke: INK.text, strokeWidth: 1.25, strokeDasharray: '3 3', fill: 'none' }
    return <g pointerEvents="none">
      <path d={`M${x(0)},${y(t)}H${x(L)}V${y(tl)}H${x(0)}`} {...dash} />
      <path d={`M${x(L)},${y(t)}V${y(0)}`} stroke={INK.muted} strokeDasharray="1 3" />
      <circle cx={x(L)} cy={y(t)} r={6} fill={MEASURED} stroke={INK.panel} strokeWidth={2} />
      <circle cx={x(L)} cy={y(tl)} r={6} fill={LINEAR} stroke={INK.panel} strokeWidth={2} />
      {onChart ? <>
        <text x={x(L) + (flip ? -9 : 9)} y={y(tl) + 4} textAnchor={flip ? 'end' : 'start'} fontSize={11} fontFamily={FONT.mono} fill={INK.text}>linearised {fmt(tl, 2)}</text>
        <text x={x(L) + (flip ? -9 : 9)} y={y(t) + 16} textAnchor={flip ? 'end' : 'start'} fontSize={11} fontFamily={FONT.mono} fill={INK.text}>measured {fmt(t, 2)}</text>
      </> : <text x={iw - 6} y={16} textAnchor="end" fontSize={11} fontFamily={FONT.mono} fill={INK.text}>linearised τ is off the chart →</text>}
    </g>
  }
  const primary = data.improve_ceilings[0]
  return (
    <ChartFrame id="hl-bend" title="The bend — HIPS optical depth against thermal EC on the filter, IMPROVE 2005–2025"
      subtitle={`Each grey dot is the median of one IMPROVE group (${primary.samples.toLocaleString()} samples at ${primary.sites} sites, grouped by the previous sample's TOR EC at the same site, which keeps each sample's own error out of the grouping); size ∝ samples. Blue dashed: what a proportional HIPS would read, τ = aL with a = ${fmt(a, 4)} µg⁻¹ (${fmt(primary.unsaturated_fabs_over_ec, 1)} m²/g on light filters). Orange: the fitted saturating response τ = aL / (1 + aL/τ∞). Move the measured τ: the construction runs across to the orange curve, down to the loading that curve implies, then up to the blue line. That height is the linearised τ.`}
      provenance={PROVENANCE}
      controls={<>
        <span className="control" title="The measured HIPS optical depth the construction follows: drag sideways across the Addis range or click to type">Measured τ <ScrubField label="Measured τ" value={t} min={0.05} max={+hi.toFixed(2)} step={0.01} width={96} format={(v) => fmt(v, 2)} onChange={(v) => setTau(String(+v.toFixed(2)))} /></span>
        {ceil.control}
      </>}>
      <ScatterPlot points={points} xLabel="TOR EC on the filter (µg)" yLabel="HIPS optical depth τ" xDomain={[0, 90]} yDomain={[0, 3.4]} nice={false}
        height={360} outside="clip" note={false} under={under}>{children}</ScatterPlot>
      <Key items={[{ label: 'IMPROVE group medians (size ∝ samples)', color: INK.muted, kind: 'dot' }, { label: 'proportional: τ = aL', color: LINEAR, kind: 'dash' },
        { label: 'saturating fit', color: MEASURED, kind: 'line' }, { label: 'Addis τ range', color: MEASURED, kind: 'faded' }]} />
      <Readouts items={[
        { label: 'Measured τ', value: fmt(t, 2), color: MEASURED },
        { label: 'Linearised τ', value: Number.isFinite(tl) ? fmt(tl, 2) : '—', color: LINEAR },
        { label: 'Correction ×', value: Number.isFinite(tl) ? fmt(tl / t, 2) : '—' },
        { label: 'Noise amplified ×', value: Number.isFinite(tl) ? fmt(noiseGain(t, c), 1) : '—', sub: '1 / (1 − τ/τ∞)²' },
      ]} />
      <Note>The ceiling buttons switch between the central fit, both ends of its 90 % interval, and the 2015–2025 fit (the period that contains the Addis record); the choice carries to every chart on this page. The noise factor is the slope of the inverse, 1 / (1 − τ/τ∞)². At 70 % of the ceiling it is about 11.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 2. one filter, step by step
function Worked({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const sorted = useMemo(() => [...data.filters].filter((f) => f.ftir != null).sort((p, q) => p.tau - q.tau), [data])
  const [id, setId] = useSearchParam<string>('hl.filter', 'ETAD-0019')
  const f = sorted.find((r) => r.id === id) ?? sorted[Math.floor(sorted.length / 2)]
  const A = data.deposit_area_cm2, MAC = data.mac, c = ceil.ceiling
  const fabs = f.hips_bc * MAC, V = f.volume_m3, r = f.tau / c, ok = r < 1
  const tl = linearise(f.tau, c), fl = (tl * 100 * A) / V, bcl = fl / MAC, L = tl / data.a_per_ug
  const rows: { step: string; what: string; value: string }[] = [
    { step: 'HIPS gives BC (Fabs ÷ 10)', what: 'as reported', value: `BC = ${fmt(f.hips_bc, 2)} µg/m³ · FTIR EC on this filter ${fmt(f.ftir, 2)} µg/m³` },
    { step: 'Back to absorption', what: 'Fabs = BC × MAC', value: `${fmt(f.hips_bc, 2)} × ${MAC} m²/g = ${fmt(fabs, 1)} Mm⁻¹` },
    { step: 'To optical depth', what: 'τ = Fabs · V / (100 · A)', value: `${fmt(fabs, 1)} × ${fmt(V, 2)} m³ / (100 × ${A} cm²) = ${fmt(f.tau, 3)}` },
    { step: 'How far up the curve', what: 'τ / τ∞', value: `${fmt(f.tau, 3)} / ${fmt(c, 2)} = ${fmt(r, 3)} → 1 − τ/τ∞ = ${fmt(1 - r, 3)}` },
    { step: 'Linearise', what: 'τ_lin = τ / (1 − τ/τ∞)', value: ok ? `${fmt(f.tau, 3)} / ${fmt(1 - r, 3)} = ${fmt(tl, 3)} (× ${fmt(tl / f.tau, 2)})` : 'τ ≥ τ∞: no finite value with this ceiling' },
    { step: 'Back to absorption', what: 'Fabs_lin = τ_lin · 100 · A / V', value: ok ? `${fmt(tl, 3)} × ${fmt(100 * A, 0)} / ${fmt(V, 2)} = ${fmt(fl, 1)} Mm⁻¹` : '—' },
    { step: 'Back to BC', what: 'BC_lin = Fabs_lin / MAC', value: ok ? `${fmt(fl, 1)} / ${MAC} = ${fmt(bcl, 2)} µg/m³ = ${fmt(f.hips_bc, 2)} × ${fmt(tl / f.tau, 2)}` : '—' },
    { step: 'What it implies on the filter', what: 'L = τ_lin / a', value: ok ? `${fmt(tl, 3)} / ${fmt(data.a_per_ug, 4)} µg⁻¹ = ${fmt(L, 1)} µg → ${fmt(L / V, 2)} µg/m³ at IMPROVE's light-filter MAC` : '—' },
    { step: 'Effect on the FTIR comparison', what: 'FTIR ÷ BC', value: ok ? `${fmt(f.ftir! / f.hips_bc, 2)} as measured → ${fmt(f.ftir! / bcl, 2)} linearised` : '—' },
    { step: 'Noise', what: 'dτ_lin/dτ', value: ok ? `±0.02 in τ becomes ±${fmt(0.02 * noiseGain(f.tau, c), 3)} in τ_lin (absolute × ${fmt(noiseGain(f.tau, c), 1)}, relative × ${fmt(1 / (1 - r), 2)})` : '—' },
  ]
  return (
    <ChartFrame id="hl-worked" title="One Addis filter through every step — from reported BC to linearised BC, at the chosen ceiling"
      subtitle="Pick a filter (sorted by optical depth). Every number is recomputed from its HIPS value and sampled volume with the ceiling chosen above. Because V, A and MAC cancel in the ratio, linearising BC is the same multiplication as linearising τ."
      provenance={PROVENANCE} exportable={false}
      controls={<Select label="Filter" value={f.id} options={sorted.map((r) => r.id)} onChange={setId}
        optionLabel={(o) => { const r = sorted.find((s) => s.id === o)!; return `${o} · ${r.season} · τ ${fmt(r.tau, 2)}` }} />}>
      <DataTable rows={rows} rowKey={(_r, i) => `${i}`} columns={[
        { key: 'n', label: '#', num: true, value: (_r) => String(rows.indexOf(_r) + 1) },
        { key: 's', label: 'Step', value: (r) => r.step }, { key: 'w', label: 'Formula', value: (r) => <code>{r.what}</code> }, { key: 'v', label: 'This filter', value: (r) => r.value }]} />
      <Note>Ceiling {ceil.label}. V is this filter's sampled air volume{f.volume_known ? '' : ' (not on its HIPS row: the Addis median stands in, volumes vary by under 2 %)'}. The loading L is "EC on the filter as IMPROVE's light filters would rate it", so it carries IMPROVE's light-filter MAC and is not a measured Addis EC.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 3. the correction factor
function CorrectionCurve({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const c = ceil.ceiling
  const { lo, hi } = { lo: data.ceilings.find((k) => k.key === 'lo')!.tau_inf, hi: data.ceilings.find((k) => k.key === 'hi')!.tau_inf }
  const shown = useMemo(() => data.filters.filter((f) => f.ftir != null), [data])
  const points = useMemo<ScatterPoint[]>(() => shown.flatMap((f) => {
    const k = correction(f.tau, c)
    return Number.isFinite(k) ? [{ id: f.id, x: f.tau, y: k, color: MEASURED, tip: [`${f.id} · ${f.season}${f.lot ? ` · lot ${f.lot}` : ''}`, `τ ${fmt(f.tau, 2)} → × ${fmt(k, 2)}`] }] : []
  }), [shown, c])
  const med = median(shown.map((f) => f.tau)), mc = correction(med, c)
  const facs = shown.map((f) => correction(f.tau, c)).filter(Number.isFinite)
  const at = (k: number) => median(shown.map((f) => correction(f.tau, k)).filter(Number.isFinite))
  const heaviest = shown.reduce((m, f) => (f.tau > m.tau ? f : m), shown[0])
  const clip = useId()
  const under = ({ x, y, iw, ih }: ScatterScales) => {
    const ts = range(0, 1.6, 0.01)
    const band = pathOf([...ts.map((t): [number, number] => [x(t), y(Math.min(5, correction(t, lo)))]), ...[...ts].reverse().map((t): [number, number] => [x(t), y(correction(t, hi))])]) + 'Z'
    return <g>
      <defs><clipPath id={clip}><rect width={iw} height={ih} /></clipPath></defs>
      <g clipPath={`url(#${clip})`}>
        <path d={band} fill={LINEAR} opacity={0.1} />
        <path d={pathOf(ts.map((t) => { const k = correction(t, c); return [x(t), k < 6 ? y(k) : NaN] }))} fill="none" stroke={LINEAR} strokeWidth={2} />
        <rect x={x(0)} y={y(1.3)} width={x(0.5) - x(0)} height={y(0.9) - y(1.3)} fill="none" stroke={INK.muted} strokeDasharray="2 3" />
        <text x={x(0.02)} y={y(1.3) - 6} fontSize={10.5} fontFamily={FONT.mono} fill={INK.muted}>most SPARTAN sites</text>
      </g>
    </g>
  }
  const children = ({ x, y }: ScatterScales) => <g pointerEvents="none">
    <line x1={x(med)} x2={x(med)} y1={y(0.9)} y2={y(mc)} stroke={INK.text} strokeDasharray="3 3" />
    <text x={x(med) + 6} y={y(mc) + 28} fontSize={11} fontFamily={FONT.mono} fill={INK.text}>Addis median τ {fmt(med, 2)} → × {fmt(mc, 2)}</text>
  </g>
  return (
    <ChartFrame id="hl-correction" title="The correction is small for light filters and large for Addis — correction factor against measured optical depth, every Addis filter"
      subtitle={`τ_lin / τ = 1 / (1 − τ/τ∞) against measured τ. The dots are the ${shown.length} Addis filters with a published FTIR EC. Most SPARTAN sites sit to the left of 0.5, where the factor is under 1.3; Addis sits where the curve steepens.`}
      provenance={PROVENANCE} controls={ceil.control}>
      <ScatterPlot points={points} xLabel="Measured HIPS optical depth τ" yLabel="Correction factor τ_lin / τ" xDomain={[0, 1.6]} yDomain={[0.9, 4.2]} nice={false}
        height={300} outside="pin" note={false} under={under}>{children}</ScatterPlot>
      <Key items={[{ label: `chosen ceiling ${fmt(c, 2)}`, color: LINEAR, kind: 'line' }, { label: `ceiling ${fmt(lo, 2)} to ${fmt(hi, 2)} (90 %)`, color: LINEAR, kind: 'faded' }, { label: 'Addis filters', color: MEASURED, kind: 'dot' }]} />
      <Note>Median Addis correction × {fmt(median(facs), 2)} at this ceiling (× {fmt(at(hi), 2)} to × {fmt(at(lo), 2)} across the 90 % interval). The heaviest filter ({heaviest.id}, τ {fmt(heaviest.tau, 2)}) is corrected × {fmt(correction(heaviest.tau, c), 2)}.</Note>
      <FixedSetNote what="the closure workflow's counted Addis filters with a published FTIR EC" />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 4. FTIR EC against HIPS
function AgainstHips({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const [view, setView] = useSearchParam<(typeof HIPS_VIEWS)[number]>('hl.hips', 'measured', HIPS_VIEWS)
  const c = ceil.ceiling, lin = view === 'linearised'
  const shown = useMemo(() => data.filters.filter((f) => f.ftir != null), [data])
  const points = useMemo<ScatterPoint[]>(() => shown.flatMap((f) => {
    const xl = linearBc(f, c), x = lin ? xl : f.hips_bc
    return Number.isFinite(x) ? [{ id: f.id, x, y: f.ftir!, color: lin ? LINEAR : MEASURED,
      tip: [`${f.id} · ${f.season}${f.lot ? ` · lot ${f.lot}` : ''}`, `FTIR EC ${fmt(f.ftir, 2)} µg/m³`, `HIPS BC ${fmt(f.hips_bc, 2)} → linearised ${fmt(xl, 2)} µg/m³`, `τ ${fmt(f.tau, 2)}`] }] : []
  }), [shown, c, lin])
  const fit = lin ? fitAgainstHips(data.filters, data.lambda, c) : fitAgainstHips(data.filters, data.lambda, null)
  const band = [data.ceilings.find((k) => k.key === 'lo')!, data.ceilings.find((k) => k.key === 'hi')!].map((k) => fitAgainstHips(data.filters, data.lambda, k.tau_inf))
  const children = ({ x, y }: ScatterScales) => {
    if (!fit) return null
    const [d0, d1] = x.domain()
    // the line starts where it meets y = 0, or at x = 0 when its intercept is positive; a ring marks that point
    const xz = -fit.intercept / fit.slope
    const x0 = Math.max(d0, xz), x1 = Math.min(d1, (y.domain()[1] - fit.intercept) / fit.slope)
    return <g pointerEvents="none">
      <line x1={x(x0)} y1={y(fit.slope * x0 + fit.intercept)} x2={x(x1)} y2={y(fit.slope * x1 + fit.intercept)} stroke={INK.fit} strokeWidth={2} />
      <circle cx={x(x0)} cy={y(fit.slope * x0 + fit.intercept)} r={5} fill="none" stroke={INK.fit} strokeWidth={2} />
    </g>
  }
  return (
    <ChartFrame id="hl-against-hips" title="Stretch the HIPS axis back out and the negative intercept disappears — SPARTAN published FTIR EC against HIPS BC, Addis Ababa"
      subtitle={`Each dot is one Addis filter (n = ${shown.length}). The FTIR EC values stay where they are; only the x value changes, and the darkest filters move furthest. The line is the Deming fit with the closure's error ratio λ = ${data.lambda}, refitted here at the chosen ceiling.`}
      provenance={PROVENANCE}
      controls={<>
        <Segmented label="HIPS" value={view} options={HIPS_VIEWS} onChange={setView} optionLabel={(o) => (o === 'measured' ? 'As measured (Fabs / 10)' : 'Linearised (/ 10)')} />
        {lin && ceil.control}
      </>}>
      <ScatterPlot points={points} xLabel={lin ? `HIPS BC linearised at τ∞ ${fmt(c, 2)} (µg/m³)` : 'HIPS BC, Fabs ÷ 10 (µg/m³)'} yLabel="FTIR EC (µg/m³)"
        xDomain={[0, 24]} yDomain={[0, 12.5]} nice={false} height={360} outside="pin" note={false}
        under={({ x, y }) => <><RefLine orient="h" at={y(0)} from={x(0)} to={x(24)} />
          <line x1={x(0)} y1={y(0)} x2={x(12.5)} y2={y(12.5)} stroke={INK.identity} strokeWidth={1.5} strokeDasharray="5 4" />
          <text x={x(12.5) + 4} y={y(12.5) + 12} fontSize={10} fontFamily={FONT.mono} fill={INK.muted}>1 : 1</text></>}>{children}</ScatterPlot>
      <Key items={[{ label: lin ? 'HIPS linearised' : 'HIPS as measured', color: lin ? LINEAR : MEASURED, kind: 'dot' }, { label: 'Deming fit', color: INK.fit, kind: 'line' }]} />
      <Readouts items={[
        { label: 'Deming fit', value: eq(fit), sub: fit ? `n = ${fit.n}` : undefined },
        { label: 'Intercept', value: fit ? `${signed(fit.intercept, 2)} µg/m³` : '—', color: lin ? LINEAR : MEASURED },
        { label: 'As measured', value: eq(data.fit_measured), color: MEASURED },
      ]} />
      <Note>Why this happens: when the high end of the x axis is squashed, a straight line through the cloud has to be steep, and a steep line through the middle of the data crosses zero far below the origin. Straighten the axis and the line flattens and passes through the origin. Across the ceiling's 90 % interval the linearised intercept runs from {signed(band[0]?.intercept, 2)} to {signed(band[1]?.intercept, 2)} µg/m³, which cannot be told apart from zero.</Note>
      <FixedSetNote what="the closure workflow's counted Addis filters with a published FTIR EC" />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 5. the network check
function NetworkExponents({ data }: { data: Linearise }) {
  const [view, setView] = useSearchParam<(typeof SITE_VIEWS)[number]>('hl.sites', 'clear', SITE_VIEWS)
  const sites = useMemo(() => data.network.filter((s) => view === 'all' || s.hi - s.lo < CLEAR_WIDTH), [data, view])
  const LABELS = new Set(['USPA', 'BDDU', 'ETAD', 'CLST', 'CLTA', 'INDH'])
  const points = useMemo<ScatterPoint[]>(() => sites.flatMap((s) => {
    const tip = [s.site, `median τ ${fmt(s.median_optical_depth, 2)} · n ${s.n}`, `measured ${fmt(s.exponent, 2)} [${fmt(s.lo, 2)}, ${fmt(s.hi, 2)}]`, `predicted ${fmt(s.predicted, 2)}`]
    return [
      { key: `${s.code}-p`, x: s.median_optical_depth, y: s.predicted, color: LINEAR, hollow: true, r: 5, tip },
      { key: s.code, x: s.median_optical_depth, y: s.exponent, color: INK.text, r: 5, opacity: 1, tip },
    ]
  }), [sites])
  const under = ({ x, y }: ScatterScales) => <g pointerEvents="none">
    {sites.map((s) => { const [y0, y1] = y.domain(); return <line key={s.code} x1={x(s.median_optical_depth)} x2={x(s.median_optical_depth)} y1={y(Math.max(y0, s.lo))} y2={y(Math.min(y1, s.hi))} stroke={INK.muted} strokeWidth={1.5} /> })}
  </g>
  const children = ({ x, y }: ScatterScales) => <g pointerEvents="none">
    {sites.filter((s) => LABELS.has(s.code)).map((s) => {
      const below = s.code === 'ETAD' || s.code.startsWith('CL'), left = s.code === 'ETAD'
      return <text key={s.code} x={x(s.median_optical_depth) + (left ? -8 : 8)} y={y(below ? Math.max(y.domain()[0], s.lo) : Math.min(y.domain()[1], s.hi)) + (below ? 14 : -6)}
        textAnchor={left ? 'end' : 'start'} fontSize={10.5} fontFamily={FONT.mono} fill={INK.text}>{s.site.split(',')[0]}</text>
    })}
  </g>
  const addis = data.network.find((s) => s.code === 'ETAD')
  const chile = data.network.filter((s) => s.code.startsWith('CL'))
  return (
    <ChartFrame id="hl-network" title="The same curve predicts the pattern across SPARTAN — exponent b in HIPS ∝ FTIR EC^b, by site median optical depth"
      subtitle="If FTIR EC and HIPS were both proportional to EC, HIPS would rise as FTIR EC to the power 1. The darker a site's filters, the more the IMPROVE curve says that exponent should drop below 1. The measured exponent uses PM2.5 mass as the instrumental variable, so random error in FTIR does not flatten it; bars are its 95 % interval."
      provenance={PROVENANCE}
      controls={<Segmented label="Sites" value={view} options={SITE_VIEWS} onChange={setView}
        optionLabel={(o) => (o === 'clear' ? `Interval under ${CLEAR_WIDTH} (${data.network.filter((s) => s.hi - s.lo < CLEAR_WIDTH).length})` : `All (${data.network.length})`)} />}>
      <ScatterPlot points={points} xLabel="Site median HIPS optical depth" yLabel="Exponent b" xDomain={[0, 1.05]} yDomain={[0.2, 1.4]} nice={false}
        height={320} outside="pin" note={false} refY={[1]} under={under}>{children}</ScatterPlot>
      <Key items={[{ label: 'measured, 95 % interval', color: INK.text, kind: 'dot' }, { label: 'predicted from the IMPROVE curve', color: LINEAR, kind: 'hollow' }]} />
      <Note>Darker sites fall further below 1, about as the curve predicts. Addis has the darkest filters and the lowest exponent ({fmt(addis?.exponent, 2)} against {fmt(addis?.predicted, 2)} predicted). The two Chilean sites ({chile.map((s) => fmt(s.exponent, 2)).join(', ')} against {fmt(Math.min(...chile.map((s) => s.predicted)), 2)}–{fmt(Math.max(...chile.map((s) => s.predicted)), 2)} predicted) are not explained. The predictions here use the central ceiling and each site's own filters as the 10 Nov note made them; the 11 Nov network page refines them per site (Meeting tab, HIPS across the network).</Note>
      <FixedSetNote what="the closure workflow's saved site exponents (all SPARTAN sites with at least 40 filters)" />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 6. seasons and lots
function SeasonLot({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const c = ceil.ceiling
  const groups = useMemo(() => data.season_lot.filter((s) => s.lot !== 'all').map((s) => {
    const own = data.filters.filter((f) => f.season === s.season && f.lot === s.lot)
    return { key: `${s.season}-${s.lot}`, label: `${s.season} · lot ${s.lot}`, ...ratioMedians(own, c) }
  }), [data, c])
  const by = (season: string, lot: string) => groups.find((g) => g.key === `${season}-${lot}`)
  const bega = by('Bega', '251'), kiremt = by('Kiremt', '251')
  const swing = (k: 'measured' | 'linearised') => (bega && kiremt ? kiremt[k] / bega[k] - 1 : NaN)
  return (
    <ChartFrame id="hl-season" title="Most of the seasonal swing was loading, not FTIR — median FTIR EC ÷ HIPS BC by season and filter lot"
      subtitle="Kiremt filters are the darkest, so they are the most compressed. Against HIPS as measured, FTIR looked like it changed a lot by season; linearised, most of that goes. Absolute levels drop because linearised HIPS is larger, so compare the spread, not the position. Recomputed at the chosen ceiling."
      provenance={PROVENANCE} controls={ceil.control}>
      <IntervalStrip blocks={[{ key: 's', rows: groups.map((g) => ({
        key: g.key, label: g.label, note: `τ ${fmt(g.tau, 2)} · n ${g.n}`,
        marks: [[{ mid: g.linearised, lo: Math.min(g.linearised, g.measured), hi: Math.max(g.linearised, g.measured), color: LINEAR, r: 6 }, { mid: g.measured, color: MEASURED, r: 6 }]],
        tip: [g.label, `against HIPS as measured ${fmt(g.measured, 2)}`, `against HIPS linearised ${fmt(g.linearised, 2)}`, `median τ ${fmt(g.tau, 2)} · n ${g.n}`],
      })) }]} panels={[{ label: 'Median FTIR EC ÷ HIPS BC', domain: [0, 1.4], format: (v) => fmt(v, 1) }]} labelWidth={130} noteWidth={110} rowHeight={30} />
      <Key items={[{ label: 'against HIPS as measured', color: MEASURED, kind: 'dot' }, { label: 'against HIPS linearised', color: LINEAR, kind: 'dot' }]} />
      <Note>Within lot 251, Bega to Kiremt goes from {fmt(bega?.measured, 2)} to {fmt(kiremt?.measured, 2)} ({fmt(100 * swing('measured'), 0)} %) against HIPS as measured, and from {fmt(bega?.linearised, 2)} to {fmt(kiremt?.linearised, 2)} ({fmt(100 * swing('linearised'), 0)} %) against linearised HIPS. Lot 248 still reads 15 to 20 % low, and it does so against gravimetric mass too, so that offset is not a HIPS effect.</Note>
      <FixedSetNote what="the closure workflow's counted Addis filters with a published FTIR EC" />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 7. what the ceiling decides
function Limits({ data }: { data: Linearise }) {
  const taus = data.filters.map((f) => f.tau)
  const rows = data.improve_ceilings.map((c) => ({ ...c, fit: fitAgainstHips(data.filters, data.lambda, c.tau_inf), predicted: predictedExponent(taus, c.tau_inf) }))
  return (
    <ChartFrame id="hl-limits" title="What linearising does not settle — the ceiling decides the intercept that is left, not whether the −6 goes"
      subtitle="Each IMPROVE population's ceiling, the Addis exponent it predicts, and the Addis fit of published FTIR EC on HIPS linearised with it (Deming, refitted here). The ceiling is borrowed from IMPROVE and changed over time; every choice removes the −6."
      provenance={PROVENANCE} exportable={false}>
      <DataTable rows={rows} rowKey={(r) => r.population} highlight={(r) => r.population === 'all sites'} columns={[
        { key: 'p', label: 'IMPROVE samples', value: (r) => `${r.population} (${r.samples.toLocaleString()}, ${r.years})` },
        { key: 'c', label: 'Ceiling τ∞ [90 %]', num: true, value: (r) => `${fmt(r.tau_inf, 2)} [${fmt(r.tau_inf_lo, 2)}, ${fmt(r.tau_inf_hi, 2)}]` },
        { key: 'm', label: 'Light-filter MAC', num: true, value: (r) => `${fmt(r.unsaturated_fabs_over_ec, 1)} m²/g` },
        { key: 'e', label: 'Addis exponent predicted', num: true, value: (r) => fmt(r.predicted, 2) },
        { key: 'f', label: 'Addis fit after linearising', num: true, value: (r) => eq(r.fit) },
      ]} />
      <Note>Noise grows near the ceiling: the correction multiplies measurement error on the heaviest filters, so per-filter linearised values are least reliable exactly where they matter most. Linearised HIPS is still optical, not the true Addis EC: its slope is FTIR level × 10 ÷ true MAC, and the MAC at Addis is unknown. The direct test is HIPS on the incoming Addis TOR filters, including some loaded to τ 0.5–1.5, which would measure the ceiling at Addis itself.</Note>
    </ChartFrame>
  )
}
