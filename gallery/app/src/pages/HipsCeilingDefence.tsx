import { useMemo } from 'react'
import { ChartFrame, Note } from '@/components/ChartFrame'
import { LoadState } from '@/components/ExportState'
import { Key } from '@/components/Key'
import { Marker } from '@/components/RefLines'
import { ScatterPlot, type ScatterPoint, type ScatterScales } from '@/components/ScatterPlot'
import { useExport } from '@/lib/data'
import { fmt } from '@/lib/format'
import { fitAgainstHips, macForSlopeOne } from '@/lib/hipsLinearise'
import { RESEARCH_SERIES } from '@/lib/palettes'
import { INK } from '@/lib/theme'
import { pathOf, SIDE, type Ceil, type Linearise } from '@/pages/HipsLinearisePage'
import { DataTable, MEASURED, Readouts, type Column } from '@/pages/hipsLineariseParts'

/**
 * Is the IMPROVE ceiling defensible (Ann, 19 Nov 2026)? Every way of estimating it side by side, and what each does to
 * Addis. Data: gallery/data/export_hips_ceiling_checks.py (`hips_linearise/ceiling_checks.json`): the grouping's
 * instrument (two back, three back, next sample, HIPS lot), one ceiling within sites, the two periods, single-filter
 * fits with errors in both axes (errors from IMPROVE's collocated Phoenix samplers), and other curve shapes including
 * patterned deposits (White et al. 2025). The Addis consequences are refitted in the browser at every ceiling.
 */

export const CHECKS_COMMAND = 'uv run python gallery/data/export_hips_ceiling_checks.py'
const FILE = 'hips_linearise/ceiling_checks.json'

interface Row { key: string; label: string; tau_inf: number; tau_inf_lo?: number | null; tau_inf_hi?: number | null; samples?: number; sites?: number; groups?: number }
interface Shape { key: string; label: string; ceiling: number; rms_log: number; aic: number; params: number[] }
interface AddisRow { label: string; group: string; ceiling: number | null; filters: number; past_ceiling: number; slope: number; intercept: number; mac_for_slope_one: number; median_factor: number }
interface Checks {
  generated: string; n_boot: number
  published: { tau_inf: number; lo: number; hi: number }
  errors: {
    pairs: number; sites: string[]; first: string; last: string
    ECf: { n: number; floor: number; relative: number; median_reported_relative: number; z_robust_sd: number }
    fAbs: { n: number; floor: number; relative: number; median_reported_relative: number; z_robust_sd: number }
    light: { from_ug: number; to_ug: number; n: number; log_spread: number; measurement_part: number; real_spread: number }
  }
  eiv: Row[]; instruments: Row[]; within_site: Row; shapes: Shape[]
  support: { improve_filters: number; darkest_group_tau: number; addis_filters: number; addis_beyond_darkest_group: number; improve_darker_than_addis_median: number; improve_tau_quantiles: Record<string, number> }
  addis: AddisRow[]
}

const GROUP = {
  published: { label: 'Published (group medians)', color: MEASURED },
  instrument: { label: 'Another instrument for the grouping', color: RESEARCH_SERIES[0] },
  within: { label: 'Within sites', color: RESEARCH_SERIES[2] },
  era: { label: 'One period', color: RESEARCH_SERIES[5] },
  eiv: { label: 'Single filters, errors in both axes', color: RESEARCH_SERIES[4] },
  shape: { label: 'Another curve shape', color: RESEARCH_SERIES[3] },
} as const
type GroupKey = keyof typeof GROUP
interface Method { key: string; label: string; group: GroupKey; tau: number; lo?: number | null; hi?: number | null; note?: string }

const X_TOP = 5
const at = (v: number) => (v >= 90 ? 'no ceiling' : fmt(v, 2))

function methodsOf(c: Checks, data: Linearise): Method[] {
  const era = (p: string) => data.improve_ceilings.find((x) => x.population === p)
  const e1 = era('2005–2014'), e2 = era('2015–2025')
  return [
    { key: 'published', label: 'Previous sample, published edges', group: 'published', tau: c.published.tau_inf, lo: c.published.lo, hi: c.published.hi },
    ...c.instruments.filter((r) => r.key !== 'previous').map((r) => ({ key: r.key, label: r.label, group: 'instrument' as const, tau: r.tau_inf, lo: r.tau_inf_lo, hi: r.tau_inf_hi,
      note: `${(r.samples ?? 0).toLocaleString()} filters` })),
    { key: 'within', label: c.within_site.label, group: 'within', tau: c.within_site.tau_inf, lo: c.within_site.tau_inf_lo, hi: c.within_site.tau_inf_hi, note: `${c.within_site.sites} sites` },
    ...(e1 ? [{ key: 'e1', label: '2005–2014 filters only', group: 'era' as const, tau: e1.tau_inf, lo: e1.tau_inf_lo, hi: e1.tau_inf_hi }] : []),
    ...(e2 ? [{ key: 'e2', label: '2015–2025 filters only', group: 'era' as const, tau: e2.tau_inf, lo: e2.tau_inf_lo, hi: e2.tau_inf_hi }] : []),
    ...c.eiv.map((r) => ({ key: r.key, label: r.label, group: 'eiv' as const, tau: r.tau_inf, lo: r.tau_inf_lo, hi: r.tau_inf_hi })),
    ...c.shapes.filter((s) => s.key !== 'hyperbolic').map((s) => ({ key: s.key, label: s.label, group: 'shape' as const, tau: s.ceiling,
      note: `ΔAIC ${fmt(s.aic - (c.shapes.find((x) => x.key === 'hyperbolic')?.aic ?? s.aic), 1)} against the hyperbola` })),
  ]
}

// ------------------------------------------------------------------ 1. every estimate of the ceiling
export function CeilingMethods({ data }: { data: Linearise }) {
  const { data: c, error } = useExport<Checks>(FILE)
  const methods = useMemo(() => (c ? methodsOf(c, data) : []), [c, data])
  const addisMax = useMemo(() => Math.max(...data.filters.filter((f) => f.ftir != null).map((f) => f.tau)), [data.filters])
  const points = useMemo<ScatterPoint[]>(() => methods.map((m, i) => ({
    key: m.key, x: Math.min(m.tau, X_TOP), y: i + 0.5, color: GROUP[m.group].color, r: 5, opacity: 1,
    tip: [m.label, `τ∞ ${at(m.tau)}${m.lo != null && m.hi != null ? ` [${fmt(m.lo, 2)}, ${fmt(m.hi, 2)}]` : ''}`, GROUP[m.group].label, ...(m.note ? [m.note] : [])],
  })), [methods])
  const under = ({ x, y, ih }: ScatterScales) => c && <g pointerEvents="none">
    <rect x={x(c.published.lo)} y={0} width={x(c.published.hi) - x(c.published.lo)} height={ih} fill={INK.muted} opacity={0.1} />
    <rect x={x(0)} y={0} width={Math.max(0, x(addisMax) - x(0))} height={ih} fill={RESEARCH_SERIES[3]} opacity={0.06} />
    <Marker orient="v" at={x(addisMax)} from={0} to={ih} color={RESEARCH_SERIES[3]} label={`Addis filters reach τ ${fmt(addisMax, 2)}`} labelSide="left" />
    {methods.map((m, i) => m.lo != null && m.hi != null && <line key={m.key} x1={x(Math.min(m.lo, X_TOP))} x2={x(Math.min(m.hi, X_TOP))} y1={y(i + 0.5)} y2={y(i + 0.5)}
      stroke={GROUP[m.group].color} strokeWidth={2} opacity={0.6} />)}
  </g>
  return (
    <ChartFrame id="hc-methods" title="Every way of estimating the ceiling — the bend is in all of them; how far it bends depends on the method, from about 1.5 to 2.8"
      subtitle={`τ∞ of τ = aL / (1 + aL/τ∞) by method, with 90 % intervals from resampling whole IMPROVE sites (${c?.n_boot ?? 100} times) where there is one. Grey band: the published interval. Shaded from the left: ceilings below the darkest Addis filter, which cannot hold there, since Addis filters are measured above them. Past ${X_TOP} the point sits at the edge.`}
      provenance={<><code>{FILE}</code> (<code>gallery/data/export_hips_ceiling_checks.py</code>)</>}>
      {!c ? <LoadState what="the ceiling checks" error={error} command={CHECKS_COMMAND} /> : <>
        <ScatterPlot points={points} yBands={methods.map((m) => ({ key: m.key, label: m.label.length > 48 ? `${m.label.slice(0, 46)}…` : m.label, color: GROUP[m.group].color }))}
          bandLabelSize={11} xDomain={[0.8, X_TOP]} nice={false} outside="pin" height={methods.length * 22} margin={{ top: 22, right: 24, bottom: 46, left: 300 }}
          xLabel="Ceiling τ∞" density="sparse" note={false} under={under} />
        <Key items={(Object.keys(GROUP) as GroupKey[]).map((g) => ({ label: GROUP[g].label, color: GROUP[g].color, kind: 'dot' as const }))} />
        <Note explain>What holds: every method finds a bend, and it is stronger, not weaker, when each site keeps its own absorption per EC,
          so it is not heavy filters coming from different sites. The grouping's instrument does not matter (the sample two or three back,
          or the next one, give the same ceiling). What does not hold yet is one number: the period ({at(methods.find((m) => m.key === 'e1')?.tau ?? NaN)} for 2005–2014,
          {' '}{at(methods.find((m) => m.key === 'e2')?.tau ?? NaN)} for 2015–2025), fitting within sites and fitting single filters all move it further than the published
          interval. The HIPS-lot check has too few filters where the lot changes to say anything. A patterned deposit as White et al. (2025) describe it,
          or an exponential approach, fits the IMPROVE medians about as well, but puts the ceiling below filters Addis has measured.</Note>
        <ErrorNote c={c} />
      </>}
    </ChartFrame>
  )
}

function ErrorNote({ c }: { c: Checks }) {
  const e = c.errors
  return <p className="chart-note">Errors for the single-filter fits come from IMPROVE's collocated samplers at Phoenix ({e.sites.join(' and ')}, {e.pairs.toLocaleString()} days,
    {' '}{e.first.slice(0, 4)}–{e.last.slice(0, 4)}): TOR EC {fmt(100 * e.ECf.relative, 1)} % plus {fmt(e.ECf.floor, 3)} µg/m³ per filter, HIPS Fabs {fmt(100 * e.fAbs.relative, 1)} % plus {fmt(e.fAbs.floor, 3)} Mm⁻¹
    {' '}(IMPROVE reports {fmt(100 * e.ECf.median_reported_relative, 0)} % and {fmt(100 * e.fAbs.median_reported_relative, 0)} %; scored against the reports the pair differences
    have a spread of {fmt(e.ECf.z_robust_sd, 2)} and {fmt(e.fAbs.z_robust_sd, 2)}, so the reports run a little high). On light filters ({e.light.from_ug}–{e.light.to_ug} µg) absorption per EC
    spreads {fmt(100 * e.light.log_spread, 0)} % from filter to filter, of which measurement explains {fmt(100 * e.light.measurement_part, 0)} %: the rest, {fmt(100 * e.light.real_spread, 0)} %, is real,
    and it is as large as the errors. That is why the single-filter fits swing with what is assumed about errors, and why the grouping, which needs no
    error model, is the published method.</p>
}

// ------------------------------------------------------------------ 2. what each ceiling does to Addis
export function AddisAtCeiling({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const { data: c, error } = useExport<Checks>(FILE)
  const methods = useMemo(() => (c ? methodsOf(c, data) : []), [c, data])
  const hyperbolic = useMemo(() => methods.filter((m) => m.group !== 'shape'), [methods])
  // the Addis fit (FTIR EC on linearised HIPS BC, the closure's λ) at every ceiling, refitted here
  // from just past the darkest Addis filter: below it filters drop out of the fit and the curve kinks
  const start = useMemo(() => 1.01 * Math.max(...data.filters.filter((f) => f.ftir != null).map((f) => f.tau)), [data.filters])
  const grid = useMemo(() => Array.from({ length: 120 }, (_, i) => start + (i * (X_TOP - start)) / 119).map((tau) => {
    const f = fitAgainstHips(data.filters, data.lambda, tau)
    return { tau, intercept: f?.intercept ?? NaN, mac: macForSlopeOne(data.filters, data.lambda, tau, data.mac) }
  }), [data, start])
  const fitAt = (tau: number) => ({ intercept: fitAgainstHips(data.filters, data.lambda, tau)?.intercept ?? NaN, mac: macForSlopeOne(data.filters, data.lambda, tau, data.mac) })
  const marks = (field: 'intercept' | 'mac') => hyperbolic.filter((m) => m.tau >= 1.5 && m.tau <= X_TOP).map<ScatterPoint>((m) => {
    const v = fitAt(m.tau)
    return { key: `${field}/${m.key}`, x: m.tau, y: v[field], color: GROUP[m.group].color, r: 5, opacity: 1,
      tip: [m.label, `τ∞ ${fmt(m.tau, 2)}`, `Addis intercept ${fmt(v.intercept, 2)} µg/m³`, `MAC for slope 1: ${fmt(v.mac, 1)} m²/g`] }
  })
  const line = (field: 'intercept' | 'mac') => ({ x, y, iw, ih }: ScatterScales) => <g pointerEvents="none">
    <clipPath id={`hc-addis-${field}`}><rect width={iw} height={ih} /></clipPath>
    <g clipPath={`url(#hc-addis-${field})`}>
      {c && <rect x={x(c.published.lo)} y={0} width={x(c.published.hi) - x(c.published.lo)} height={ih} fill={INK.muted} opacity={0.1} />}
      {field === 'intercept' ? <Marker orient="h" at={y(0)} from={0} to={iw} color={INK.text} label="intercept 0" />
        : <Marker orient="h" at={y(10)} from={0} to={iw} color={INK.text} label="MAC 10 (SPARTAN's)" />}
      <Marker orient="v" at={x(ceil.ceiling)} from={0} to={ih} color={MEASURED} label={`page ceiling ${fmt(ceil.ceiling, 2)}`} />
      <path d={pathOf(grid.map((g) => [x(g.tau), y(g[field])]))} fill="none" stroke={INK.text} strokeWidth={1.75} />
    </g>
  </g>
  const inPublished = c ? [fitAt(c.published.lo), fitAt(c.published.hi)] : []
  const measured = c?.addis.find((r) => r.group === 'measured')
  return (
    <ChartFrame id="hc-addis-ceiling" title="What each ceiling does to Addis — every one removes the −6 intercept, but how much is left, and the MAC, depend on which"
      subtitle={`Addis published FTIR EC (y) on HIPS BC linearised at each ceiling (x of the fit), Deming with the closure's λ ${data.lambda}, refitted here at every τ∞ from 1.5 to ${X_TOP}: the intercept left (left), and the MAC at which the slope is 1 (right; MAC ÷ slope, the λ convention under which a MAC change is exact). Dots: the ceilings from the chart above. As measured the fit is ${measured ? `${fmt(measured.slope, 2)}x − ${fmt(-measured.intercept, 2)}` : '2.31x − 6.17'}.`}
      provenance={<>{FILE}; Addis filters from <code>hips_linearise/linearise.json</code>.</>}>
      {!c ? <LoadState what="the ceiling checks" error={error} command={CHECKS_COMMAND} /> : <>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 24, justifyContent: 'center' }}>
          <div><ScatterPlot title="Intercept left (µg/m³)" points={marks('intercept')} xLabel="Ceiling τ∞" yLabel="Addis intercept (µg/m³)" xDomain={[1.5, X_TOP]}
            yInclude={[-2, 5]} maxSide={SIDE * 0.8} outside="pin" note={false} under={line('intercept')} /></div>
          <div><ScatterPlot title="MAC for slope 1 (m²/g)" points={marks('mac')} xLabel="Ceiling τ∞" yLabel="MAC for slope 1 (m²/g)" yScale="log" xDomain={[1.5, X_TOP]}
            yDomain={[5, 200]} nice={false} maxSide={SIDE * 0.8} outside="pin" note={false} under={line('mac')} /></div>
        </div>
        <Key items={[{ label: 'The Addis fit at each ceiling', color: INK.text, kind: 'line' }, ...(Object.keys(GROUP) as GroupKey[]).filter((g) => g !== 'shape')
          .map((g) => ({ label: GROUP[g].label, color: GROUP[g].color, kind: 'dot' as const }))]} />
        <Readouts items={[
          { label: 'Intercept across the published interval', value: `${fmt(inPublished[0]?.intercept, 1)} to ${fmt(inPublished[1]?.intercept, 1)}`, sub: `τ∞ ${fmt(c.published.lo, 2)} to ${fmt(c.published.hi, 2)}` },
          { label: 'MAC for slope 1, same interval', value: `${fmt(inPublished[1]?.mac, 0)} to ${fmt(inPublished[0]?.mac, 0)}`, sub: 'm²/g' },
          { label: 'Addis beyond the darkest IMPROVE group', value: `${Math.round(100 * c.support.addis_beyond_darkest_group)} %`, sub: `τ over ${fmt(c.support.darkest_group_tau, 2)}; ${c.support.improve_darker_than_addis_median.toLocaleString()} single IMPROVE filters are darker than the median Addis filter` },
        ]} />
        <Note explain>Defensible from this: HIPS as measured cannot be used for a level at Addis, and any ceiling the methods support removes most
          of the −6. Not defensible yet: a number for the Addis MAC, or a claim that linearising "fixes" the intercept. Near the low end of the ceilings the
          correction overshoots (a positive intercept, a MAC far above any measured), and the published central value lands at zero partly by where it sits.
          The test that settles it is a ceiling fitted on Addis's own filters against thermal EC, from the Addis TOR filters.</Note>
        <AddisTable rows={c.addis} />
      </>}
    </ChartFrame>
  )
}

function AddisTable({ rows }: { rows: AddisRow[] }) {
  const columns: Column<AddisRow>[] = [
    { key: 'label', label: 'Ceiling from', value: (r) => r.label },
    { key: 'c', label: 'τ∞', value: (r) => (r.ceiling == null ? '—' : at(r.ceiling)), num: true },
    { key: 'fit', label: 'Addis fit', value: (r) => `${fmt(r.slope, 2)}x ${r.intercept < 0 ? '−' : '+'} ${fmt(Math.abs(r.intercept), 2)}`, num: true },
    { key: 'mac', label: 'MAC for slope 1', value: (r) => (r.mac_for_slope_one > 200 ? '> 200' : fmt(r.mac_for_slope_one, 1)), num: true },
    { key: 'k', label: 'Median factor', value: (r) => `×${fmt(r.median_factor, 2)}`, num: true },
    { key: 'past', label: 'Filters past the ceiling', value: (r) => (r.past_ceiling ? `${r.past_ceiling} of ${r.filters + r.past_ceiling}` : '0'), num: true },
  ]
  return <DataTable rows={rows} columns={columns} rowKey={(r) => r.label} highlight={(r) => r.group === 'published' && r.label.endsWith('central')} />
}
