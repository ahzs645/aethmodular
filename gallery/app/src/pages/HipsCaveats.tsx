import { useId, useMemo, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Note } from '@/components/ChartFrame'
import { Select } from '@/components/Controls'
import { IntervalStrip } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { Marker } from '@/components/RefLines'
import { ScatterPlot, type ScatterScales } from '@/components/ScatterPlot'
import { writeAt } from '@/lib/claimDigits'
import { fmt, pct } from '@/lib/format'
import { PinRing } from '@/lib/highlight'
import { linearLevel } from '@/lib/hipsLinearise'
import { FixedSetNote } from '@/lib/markedRows'
import { FONT, INK } from '@/lib/theme'
import { HAS_IMPROVE_GROUPS, ImproveGroupDrawer, type Ceil, type Linearise } from '@/pages/HipsLinearisePage'
import { useSearchParam } from '@/lib/url'
import { LINEAR, MEASURED, Readouts, STAT_NAME, useStat } from '@/pages/hipsLineariseParts'

/**
 * Two things to keep in view when reading a linearised Addis number (the "Linearising HIPS at Addis" page):
 *
 *  - ImproveReach: IMPROVE's own darker filters do read lower per unit of thermal EC, and the IMPROVE record stops
 *    at an optical depth half the Addis filters are beyond, so for them the curve is an extrapolation
 *  - LevelChoices: the Addis level under every IMPROVE population's ceiling, at SPARTAN's MAC and at that
 *    population's own light-filter MAC, so the spread between defensible choices is visible beside the number
 *
 * Both are recomputed in the browser from the page's export (hips_linearise/linearise.json).
 */
const SOURCE = <>research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py; IMPROVE group medians, ceilings and per-filter optical depths in <code>hips_linearise/linearise.json</code> (docs/addis-reference-closure-2026-11-10.md).</>

const TAU_MAX = 1.6
const BIN = 0.05
/** the right edge of the level strip, µg/m³: a range end beyond it (or with no finite value) is drawn to the edge */
const LEVEL_MAX = 18

const pathOf = (pts: [number, number][]) => pts.map(([a, b], i) => `${i ? 'L' : 'M'}${a},${b}`).join('')

// ------------------------------------------------------------------ IMPROVE's darker filters, and where it stops
export function ImproveReach({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const c = ceil.ceiling
  const lo = data.ceilings.find((k) => k.key === 'lo')!.tau_inf, hi = data.ceilings.find((k) => k.key === 'hi')!.tau_inf
  const primary = data.improve_ceilings[0], light = primary.unsaturated_fabs_over_ec
  const lightest = data.strata.reduce((m, s) => (s.tau < m.tau ? s : m), data.strata[0])
  const darkest = data.strata.reduce((m, s) => (s.tau > m.tau ? s : m), data.strata[0])
  const reach = darkest.tau, reachAny = Math.max(...data.improve_ceilings.map((p) => p.max_stratum_tau))
  const taus = useMemo(() => data.filters.map((f) => f.tau), [data])
  const bins = useMemo(() => d3.bin().domain([0, TAU_MAX]).thresholds(d3.range(BIN, TAU_MAX, BIN))(taus), [taus])
  const beyond = taus.filter((t) => t > reach).length, beyondAny = taus.filter((t) => t > reachAny).length
  // the same IMPROVE groups as The bend, keyed the same way (1-based, in the export's order) and picked by the same
  // URL key (`hl.group`), so a group chosen on either chart is the one ringed on both; a click opens its samples
  const [group, setGroup] = useSearchParam<string>('hl.group', 'none')
  const [drawer, setDrawer] = useState(false)
  const points = useMemo(() => data.strata.map((s, i) => {
    const details = [`TOR EC ${fmt(s.loading_ug, 1)} µg`, `optical depth τ ${fmt(s.tau, 3)}`, `Fabs ÷ EC ${fmt(s.fabs_over_ec, 1)} m²/g`, `${s.n.toLocaleString()} samples`]
    return { key: String(i + 1), n: s.n, details, x: s.tau, y: s.fabs_over_ec, color: INK.muted, r: 3 + Math.sqrt(s.n) / 40, opacity: 0.85,
      tip: [`IMPROVE group, optical depth ${fmt(s.tau, 2)}`, `Fabs ÷ TOR EC ${fmt(s.fabs_over_ec, 1)} m²/g`, `TOR EC on the filter ${fmt(s.loading_ug, 1)} µg`, `${s.n.toLocaleString()} samples`,
        HAS_IMPROVE_GROUPS ? 'Click to view the samples in this group' : 'Click to select or clear this group'] }
  }), [data])
  const selected = points.find((p) => p.key === group)
  const pickGroup = (key: string) => {
    const next = !HAS_IMPROVE_GROUPS && key === group ? 'none' : key
    setGroup(next)
    setDrawer(HAS_IMPROVE_GROUPS && next !== 'none')
  }
  // the saturating response, as absorption per unit of EC: the light-filter value falling in a straight line to 0 at the ceiling
  const response = (t: number, k: number) => Math.max(0, light * (1 - t / k))
  const clip = useId()
  const under = ({ x, y, iw, ih }: ScatterScales) => {
    const ts = d3.range(0, TAU_MAX + 1e-9, 0.02)
    const line = (from: number, to: number) => pathOf(ts.filter((t) => t >= from - 1e-9 && t <= to + 1e-9).map((t) => [x(t), y(response(t, c))]))
    const band = `${pathOf([...ts.map((t): [number, number] => [x(t), y(response(t, hi))]), ...[...ts].reverse().map((t): [number, number] => [x(t), y(response(t, lo))])])}Z`
    const tallest = Math.max(1, ...bins.map((b) => b.length)), barMax = Math.min(54, ih * 0.2)
    const anySide = x(reachAny) + 130 < iw ? 'right' as const : 'left' as const
    return <g pointerEvents="none">
      <defs><clipPath id={clip}><rect width={iw} height={ih} /></clipPath></defs>
      <g clipPath={`url(#${clip})`}>
        <rect x={x(reach)} y={0} width={Math.max(0, iw - x(reach))} height={ih} fill={INK.shade} fillOpacity={0.75} />
        {bins.map((b) => (b.length ? <rect key={b.x0} x={x(b.x0!) + 1} width={Math.max(1, x(b.x1!) - x(b.x0!) - 2)}
          y={ih - (b.length / tallest) * barMax} height={(b.length / tallest) * barMax} fill={MEASURED} fillOpacity={0.6} /> : null))}
        <path d={band} fill={LINEAR} opacity={0.1} />
        <path d={line(0, reach)} fill="none" stroke={LINEAR} strokeWidth={2} />
        <path d={line(reach, TAU_MAX)} fill="none" stroke={LINEAR} strokeWidth={2} strokeDasharray="6 4" />
      </g>
      <Marker orient="v" at={x(reach)} from={0} to={ih} label={`IMPROVE stops, τ ${fmt(reach, 2)}`} labelSide="left" />
      <Marker orient="v" at={x(reachAny)} from={0} to={ih} label={`any subset, τ ${fmt(reachAny, 2)}`} labelSide={anySide} labelDy={anySide === 'left' ? 14 : 0} />
      <text x={x(Math.floor(Math.min(...taus) / BIN) * BIN) - 6} y={ih - 6} textAnchor="end" fontSize={10.5} fontFamily={FONT.mono} fill={INK.muted}>Addis filters</text>
    </g>
  }
  const loss = 1 - darkest.fabs_over_ec / lightest.fabs_over_ec
  return (
    <ChartFrame id="hl-reach" title={`IMPROVE's darker filters read ${pct(loss)} lower per unit of EC, and ${pct(beyond / taus.length)} of Addis filters are darker than any of them — HIPS absorption per unit of thermal EC against optical depth, IMPROVE 2005–2025`}
      subtitle={`Each grey dot is the median of one IMPROVE group (${primary.samples.toLocaleString()} samples at ${primary.sites} sites); size ∝ samples. Height is HIPS absorption per unit of thermal (TOR) EC on the filter: a HIPS that kept up would stay level. The blue line is the fitted response at the chosen ceiling, solid where IMPROVE has filters and dashed where it does not. The bars along the bottom count the Addis filters at each optical depth (their height is a count, not a value on the left axis). Shaded: darker than IMPROVE's darkest group.`}
      provenance={SOURCE}
      controls={<Select label="IMPROVE group" value={selected?.key ?? 'none'} options={['none', ...points.map((p) => p.key)]} onChange={pickGroup}
        optionLabel={(key) => points.find((p) => p.key === key)?.details[0] ?? 'None'} title="The IMPROVE group in focus, shared with The bend: click a dot to pick it" />}>
      <ScatterPlot points={points} xLabel="HIPS optical depth τ" yLabel="Fabs ÷ TOR EC (m²/g)" xDomain={[0, TAU_MAX]} yDomain={[0, 15]} nice={false}
        height={340} outside="clip" note={false} under={under} onPick={(p) => p.key && pickGroup(p.key)}>
        {({ x, y }) => selected && <PinRing cx={x(selected.x)} cy={y(selected.y)} r={selected.r} />}
      </ScatterPlot>
      {selected && <div role="status">
        <p className="chart-note"><strong>IMPROVE group median</strong> · {selected.details.join(' · ')}</p>
        {HAS_IMPROVE_GROUPS && <button type="button" className="btn" onClick={() => setDrawer(true)}>View group samples</button>}
        {HAS_IMPROVE_GROUPS && drawer && <ImproveGroupDrawer key={selected.key} group={selected} onClose={() => setDrawer(false)} />}
      </div>}
      <Key items={[{ label: 'IMPROVE group medians (size ∝ samples)', color: INK.muted, kind: 'dot' }, { label: `fitted response, ceiling ${fmt(c, 2)}`, color: LINEAR, kind: 'line' },
        { label: 'beyond IMPROVE: extrapolated', color: LINEAR, kind: 'dash' }, { label: `ceiling ${fmt(lo, 2)} to ${fmt(hi, 2)} (90 %)`, color: LINEAR, kind: 'faded' },
        { label: 'Addis filters (count)', color: MEASURED, kind: 'bar' }]} />
      <Readouts items={[
        { label: 'IMPROVE, lightest', value: fmt(lightest.fabs_over_ec, 1), sub: `m²/g at τ ${fmt(lightest.tau, 2)}` },
        { label: 'IMPROVE, darkest', value: fmt(darkest.fabs_over_ec, 1), sub: `m²/g at τ ${fmt(darkest.tau, 2)} · ${darkest.n.toLocaleString()} samples` },
        { label: 'Addis, darker', value: `${beyond} of ${taus.length}`, sub: `${pct(beyond / taus.length)} of its filters` },
        { label: 'Past any subset', value: `${beyondAny} of ${taus.length}`, sub: `beyond τ ${fmt(reachAny, 2)} · ${pct(beyondAny / taus.length)}` },
      ]} />
      <Note>So yes: IMPROVE sees the same loss on its own dark filters, against thermal EC. But IMPROVE is mostly rural, and only {(darkest.n / primary.samples * 100).toFixed(1)} % of its samples are in the darkest group. The lightest Addis filter (τ {fmt(Math.min(...taus), 2)}) is already where IMPROVE has lost {pct(1 - response(Math.min(...taus), c) / light)}; the Addis median (τ {fmt(d3.median(taus), 2)}) is at the edge of the all-sites record, and the low-OC/EC and urban subsets reach τ {fmt(reachAny, 2)}. Past the shaded edge the correction rests on the shape of the curve, not on filters.</Note>
      <FixedSetNote what="the closure workflow's IMPROVE group medians and counted Addis filters" />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ the level under every defensible choice
export function LevelChoices({ data }: { data: Linearise }) {
  const [stat] = useStat()
  const kind = stat === 'median_bc' ? 'median' as const : 'mean' as const
  const name = STAT_NAME[stat]
  const reported = useMemo(() => linearLevel(data.filters, null, kind), [data, kind])
  const rows = useMemo(() => data.improve_ceilings.map((p) => {
    const at = (ceiling: number) => linearLevel(data.filters, ceiling, kind)
    // a lower ceiling gives the higher level, so the interval's low end is the range's top
    return { p, mid: at(p.tau_inf), low: at(p.tau_inf_hi), high: at(p.tau_inf_lo), own: data.mac / p.unsaturated_fabs_over_ec,
      past: data.filters.filter((f) => f.tau >= p.tau_inf_lo).length }
  }), [data, kind])
  const cap = (v: number) => (Number.isFinite(v) ? Math.min(LEVEL_MAX, v) : LEVEL_MAX)
  const text = (v: number) => (Number.isFinite(v) ? fmt(v, 1) : 'no finite value')
  const title = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)
  const levels = rows.flatMap((r) => [{ r, v: r.mid, how: `MAC ${data.mac}` }, { r, v: r.mid * r.own, how: 'own MAC' }]).filter((l) => Number.isFinite(l.v))
  const least = levels.reduce((m, l) => (l.v < m.v ? l : m), levels[0]), most = levels.reduce((m, l) => (l.v > m.v ? l : m), levels[0])
  const central = rows[0]
  return (
    <ChartFrame id="hl-level" title={`The Addis ${name.toLowerCase()} runs from ${writeAt(least.v, 0)} to ${writeAt(most.v, 0)} µg/m³ with the ceiling and the MAC chosen — HIPS BC linearised with each IMPROVE population's ceiling`}
      subtitle={`One row per set of IMPROVE samples the ceiling can be fitted on. Filled: the Addis ${name.toLowerCase()} linearised with that ceiling and kept at MAC ${data.mac} m²/g, as SPARTAN reports BC. Hollow: the same at that population's own light-filter MAC (Fabs ÷ thermal EC on its lightest filters), the figure to hold against an EC. Bars: that ceiling's own 90 % interval. The dashed line is the value as reported.`}
      provenance={SOURCE}>
      <IntervalStrip blocks={[{ key: 'p', rows: rows.map((r) => ({
        key: r.p.population, label: `${title(r.p.population)} · τ∞ ${fmt(r.p.tau_inf, 2)}`, strong: r === central,
        marks: [[
          { mid: cap(r.mid), lo: cap(r.low), hi: cap(r.high), color: LINEAR, r: 5, dy: -5 },
          { mid: cap(r.mid * r.own), lo: cap(r.low * r.own), hi: cap(r.high * r.own), color: LINEAR, shape: 'hollow' as const, r: 5, dy: 6, faint: true },
        ]],
        tip: [`IMPROVE ${r.p.population} (${r.p.samples.toLocaleString()} samples, ${r.p.years})`, `ceiling τ∞ ${fmt(r.p.tau_inf, 2)} [${fmt(r.p.tau_inf_lo, 2)}, ${fmt(r.p.tau_inf_hi, 2)}] · light-filter MAC ${fmt(r.p.unsaturated_fabs_over_ec, 1)} m²/g`,
          `at MAC ${data.mac}: ${text(r.mid)} (${text(r.low)} to ${text(r.high)}) µg/m³`, `at its own MAC: ${text(r.mid * r.own)} (${text(r.low * r.own)} to ${text(r.high * r.own)}) µg/m³`,
          ...(r.past ? [`${r.past} Addis filters are at or past the low end of this ceiling (τ ${fmt(r.p.tau_inf_lo, 2)})`] : [])],
      })) }]} panels={[{ label: `${name} HIPS BC at Addis (µg/m³)`, domain: [0, LEVEL_MAX], marker: { value: reported, label: `as reported ${fmt(reported, 2)}` }, format: (v) => fmt(v, 0) }]}
        labelWidth={250} rowHeight={34} />
      <Key items={[{ label: `at MAC ${data.mac} m²/g, as SPARTAN reports BC`, color: LINEAR, kind: 'dot' }, { label: "at that population's own light-filter MAC", color: LINEAR, kind: 'hollow' },
        { label: "that ceiling's 90 % interval", color: LINEAR, kind: 'line' }]} />
      <Readouts items={[
        { label: 'As reported', value: fmt(reported, 2), sub: `µg/m³ · ${data.filters.length} filters` },
        { label: `Central, MAC ${data.mac}`, value: fmt(central.mid, 1), sub: `${text(central.low)} to ${text(central.high)}` },
        { label: 'Lowest row', value: fmt(least.v, 1), sub: `${least.r.p.population}, ${least.how}` },
        { label: 'Highest row', value: fmt(most.v, 1), sub: `${most.r.p.population}, ${most.how}` },
      ]} />
      <Note>Two choices move the number as much as the data do. The ceiling: 2015–2025 is the period that contains the Addis record, and its value sits above the central row's whole range. The MAC: SPARTAN divides by {data.mac} m²/g, while IMPROVE's light filters give {fmt(Math.min(...rows.map((r) => r.p.unsaturated_fabs_over_ec)), 1)} to {fmt(Math.max(...rows.map((r) => r.p.unsaturated_fabs_over_ec)), 1)} per unit of thermal EC, which pulls the level back down. A range end drawn to the right edge is beyond {LEVEL_MAX} µg/m³ or has no finite value (hover a row). No row is a measured Addis EC: every one borrows IMPROVE's ceiling, and the hollow marks borrow its MAC too. These are the {data.filters.length} Addis filters in this page's export, so the values differ slightly from the 11 Nov site statistics.</Note>
      <FixedSetNote what="the closure workflow's counted Addis filters and IMPROVE ceilings" />
    </ChartFrame>
  )
}
