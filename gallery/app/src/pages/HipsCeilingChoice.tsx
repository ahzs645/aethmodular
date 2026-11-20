import { useMemo } from 'react'
import { ChartFrame, Note } from '@/components/ChartFrame'
import { Segmented } from '@/components/Controls'
import { LoadState } from '@/components/ExportState'
import { Key } from '@/components/Key'
import { Marker } from '@/components/RefLines'
import { ScatterPlot, type ScatterPoint, type ScatterScales } from '@/components/ScatterPlot'
import { ScrubField } from '@/components/ScrubField'
import { useExport } from '@/lib/data'
import { fmt } from '@/lib/format'
import { PinRing } from '@/lib/highlight'
import { correction, saturate } from '@/lib/hipsLinearise'
import {
  dropOne, fitCeiling, groupsOf, ruleEdges, sweepRow, TAU_BOUNDS,
  type EdgeRule, type GroupBasis, type GroupingsFile, type Residual, type SweepRow,
} from '@/lib/hipsGroupings'
import { SidebarControls } from '@/lib/pageSidebar'
import { RESEARCH_SERIES } from '@/lib/palettes'
import { median } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { pathOf, range, SIDE, type Ceil, type Linearise } from '@/pages/HipsLinearisePage'
import { MEASURED, Readouts } from '@/pages/hipsLineariseParts'

/**
 * How the IMPROVE ceiling is decided, and how much the grouping behind it matters (Ann, 19 Nov 2026). The reader picks
 * how the filters are grouped (by the previous sample's loading, as published, or by the filter's own loading or τ), the
 * edges (published, equal count, log-spaced), how many groups, the smallest group kept and the residuals; the medians
 * are rebuilt and the curve refitted in the browser (lib/hipsGroupings), with the fit's error at every ceiling, the
 * ceiling with each group left out, and the precomputed sweep of groupings with site-resampled intervals.
 */

export const GROUPINGS_COMMAND = 'uv run python gallery/data/export_hips_improve_groupings.py'
const FILE = 'hips_linearise/improve_groupings.json'
const BASES: readonly GroupBasis[] = ['previous', 'own', 'tau']
const RULES: readonly EdgeRule[] = ['published', 'count', 'log']
const RESIDUALS: readonly Residual[] = ['log', 'abs']
const BASIS_LABEL: Record<GroupBasis, string> = { previous: 'Previous sample’s loading', own: 'Own loading', tau: 'Own τ' }
const BASIS_COLOR: Record<GroupBasis, string> = { previous: MEASURED, own: RESEARCH_SERIES[0], tau: RESEARCH_SERIES[3] }
const RULE_LABEL: Record<EdgeRule, string> = { published: 'Published edges', count: 'Equal count', log: 'Log-spaced' }
const LIVE = RESEARCH_SERIES[4]
const at = (v: number) => (v >= TAU_BOUNDS[1] * 0.98 ? 'no ceiling (runs to the bound)' : fmt(v, 2))

/** the reader's grouping, in the link (`hc.g.*`), shared by the three charts */
export function useGroupingChoice() {
  const [basis, setBasis] = useSearchParam<GroupBasis>('hc.g.basis', 'previous', BASES)
  const [ruleRaw, setRule] = useSearchParam<EdgeRule>('hc.g.rule', 'published', RULES)
  const [nText, setN] = useSearchParam<string>('hc.g.n', '17')
  const [minText, setMin] = useSearchParam<string>('hc.g.min', '150')
  const [residual, setResidual] = useSearchParam<Residual>('hc.g.res', 'log', RESIDUALS)
  const rule: EdgeRule = basis === 'tau' && ruleRaw === 'published' ? 'count' : ruleRaw
  const n = Math.min(40, Math.max(3, Math.round(Number(nText) || 17)))
  const minN = Math.min(5000, Math.max(1, Math.round(Number(minText) || 150)))
  const isPublished = basis === 'previous' && rule === 'published' && minN === 150 && residual === 'log'
  const reset = () => { setBasis('previous'); setRule('published'); setN('17'); setMin('150'); setResidual('log') }
  return { basis, setBasis, rule, setRule, n, setN: (v: number) => setN(String(v)), minN, setMin: (v: number) => setMin(String(v)), residual, setResidual, isPublished, reset }
}
type Choice = ReturnType<typeof useGroupingChoice>

function useGroupingFit(file: GroupingsFile | null, choice: Choice) {
  return useMemo(() => {
    if (!file) return null
    const basis = file.bases[choice.basis]
    const edges = ruleEdges(choice.rule, choice.n, basis)
    const { groups, dropped, droppedFilters } = groupsOf(file, basis, edges, choice.minN)
    const fit = fitCeiling(groups, choice.residual)
    const row = sweepRow(file, choice.basis, choice.rule, choice.n, choice.minN, choice.residual)
    return { groups, dropped, droppedFilters, fit, row, used: groups.reduce((a, g) => a + g.n, 0) }
  }, [file, choice.basis, choice.rule, choice.n, choice.minN, choice.residual])
}

/** the grouping controls, once in Page options (they drive the three charts) */
export function GroupingOptions({ choice }: { choice: Choice }) {
  return <SidebarControls title="IMPROVE grouping" tabNote={choice.isPublished ? 'as published' : 'changed'}>
    <div role="group" aria-label="IMPROVE grouping" style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
      <Segmented label="Group by" value={choice.basis} options={BASES} onChange={choice.setBasis} optionLabel={(b) => BASIS_LABEL[b]}
        title="Published: the previous sample's TOR EC loading at the same site. The filter's own loading or own τ are the groupings that select on TOR's or HIPS's own error." />
      <Segmented label="Edges" value={choice.rule} options={choice.basis === 'tau' ? RULES.filter((r) => r !== 'published') : RULES} onChange={choice.setRule}
        optionLabel={(r) => RULE_LABEL[r]} title="Published: the 17 hand-set edges (0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 11, 15, 20, 27, 36, 50, 70, 100 µg). Equal count: each group holds the same share of filters. Log-spaced: edges evenly spaced on a log scale." />
      {choice.rule !== 'published' && <span className="control" title="How many groups (edges that fall together merge)">Groups <ScrubField label="Groups" value={choice.n} min={3} max={40} step={1} width={70} format={(v) => String(Math.round(v))} onChange={(v) => choice.setN(Math.round(v))} /></span>}
      <span className="control" title="Groups with fewer filters are left out of the fit (published: 150)">Smallest group <ScrubField label="Smallest group" value={choice.minN} min={1} max={5000} step={10} width={84} format={(v) => String(Math.round(v))} onChange={(v) => choice.setMin(Math.round(v))} /></span>
      <Segmented label="Residuals" value={choice.residual} options={RESIDUALS} onChange={choice.setResidual} optionLabel={(r) => (r === 'log' ? 'log τ' : 'τ')}
        title="Published: least squares on log τ, so every group weighs alike. On τ, the dark groups weigh more." />
      {!choice.isPublished && <button type="button" className="btn" onClick={choice.reset}>As published</button>}
    </div>
  </SidebarControls>
}

// ------------------------------------------------------------------ 1. medians, the curve, and the fit's error at every ceiling
export function CeilingDecided({ data, ceil, choice }: { data: Linearise; ceil: Ceil; choice: Choice }) {
  const { data: file, error } = useExport<GroupingsFile>(FILE)
  const live = useGroupingFit(file, choice)
  const published = data.improve_ceilings[0]
  const ciLo = published.tau_inf_lo, ciHi = published.tau_inf_hi
  const addisTau = useMemo(() => median(data.filters.filter((f) => f.ftir != null).map((f) => f.tau)), [data.filters])
  const fit = live?.fit ?? null
  const points = useMemo<ScatterPoint[]>(() => (live?.groups ?? []).map((g, i) => ({
    key: `g${i}`, x: g.loading, y: g.tau, color: INK.text, r: 3 + Math.sqrt(g.n) / 40, opacity: 0.9,
    tip: [`Group ${i + 1}`, `${BASIS_LABEL[choice.basis]} ${fmt(g.lower, 3)}–${g.upper >= 1e3 ? '∞' : fmt(g.upper, 3)}`, `${g.n.toLocaleString()} filters`,
      `median TOR EC ${fmt(g.loading, 2)} µg`, `median τ ${fmt(g.tau, 3)}`],
  })), [live, choice.basis])
  const top = Math.max(90, ...points.map((p) => p.x))
  const under = ({ x, y, iw, ih }: ScatterScales) => {
    const xs = range(0, top, top / 300)
    return <g pointerEvents="none">
      <clipPath id="hc-ceiling-clip"><rect width={iw} height={ih} /></clipPath>
      <g clipPath="url(#hc-ceiling-clip)">
        <rect x={0} y={y(ciHi)} width={iw} height={Math.max(0, y(ciLo) - y(ciHi))} fill={INK.muted} opacity={0.08} />
        <path d={pathOf(xs.map((v) => [x(v), y(saturate(data.a_per_ug * v, published.tau_inf))]))} fill="none" stroke={MEASURED} strokeWidth={2} strokeDasharray="6 4" />
        <Marker orient="h" at={y(published.tau_inf)} from={0} to={iw} color={MEASURED} label={`published τ∞ ${fmt(published.tau_inf, 2)}`} />
        {fit && <>
          <path d={pathOf(xs.map((v) => [x(v), y(saturate(fit.a * v, fit.tauInf))]))} fill="none" stroke={LIVE} strokeWidth={2.25} />
          {/* the published label sits at the right end above its line, this one at the left end: below its line when it sits
              just under the published one, so the two never share a corner */}
          {fit.tauInf < 3.4 && <Marker orient="h" at={y(fit.tauInf)} from={0} to={iw} color={LIVE} label={`this grouping τ∞ ${fmt(fit.tauInf, 2)}`}
            labelSide="left" labelDy={fit.tauInf < published.tau_inf && y(fit.tauInf) - y(published.tau_inf) < 16 ? 15 : 0} />}
        </>}
      </g>
    </g>
  }
  // the profile: the fit's error at each ceiling with a refitted; the ceiling is its lowest point
  const profile = fit?.profile ?? []
  const rmsTop = fit ? Math.max(fit.rms * 4, fit.rms + 0.05) : 1
  const profilePoints = useMemo<ScatterPoint[]>(() => (fit ? [{ key: 'min', x: fit.tauInf, y: fit.rms, color: LIVE, r: 6,
    tip: ['Lowest error: the fitted ceiling', `τ∞ ${at(fit.tauInf)}`, `RMS residual ${fmt(fit.rms, 3)}`] }] : []), [fit])
  const profileUnder = ({ x, y, iw, ih }: ScatterScales) => <g pointerEvents="none">
    <clipPath id="hc-profile-clip"><rect width={iw} height={ih} /></clipPath>
    <g clipPath="url(#hc-profile-clip)">
      <rect x={x(ciLo)} y={0} width={Math.max(0, x(ciHi) - x(ciLo))} height={ih} fill={MEASURED} opacity={0.1} />
      <Marker orient="v" at={x(published.tau_inf)} from={0} to={ih} color={MEASURED} label={`published ${fmt(published.tau_inf, 2)}`} />
      <path d={pathOf(profile.map((p) => [x(p.tauInf), y(Math.min(p.rms, rmsTop * 1.2))]))} fill="none" stroke={LIVE} strokeWidth={2} />
    </g>
  </g>
  const row = live?.row
  return (
    <ChartFrame id="hc-ceiling" title="How the ceiling is decided — group the IMPROVE filters, take each group's medians, and fit the one curve through them; τ∞ is where the fit's error is lowest"
      subtitle={`Left: the group medians of TOR EC on the filter and HIPS optical depth under the grouping in Page options (IMPROVE grouping), the curve fitted through them (purple) and the published one (dashed orange, its 90 % interval shaded). Right: the fit's RMS residual at each ceiling, with a refitted at each; the ceiling is its lowest point, and a flat bottom means the medians pin it loosely. Medians are read from ${file ? `${Math.round((file.values.loading.ratio - 1) * 100)} %` : 'narrow'} bins, so a published grouping lands within about 0.01 of the workflow's fit.`}
      provenance={<><code>{FILE}</code> (<code>gallery/data/export_hips_improve_groupings.py</code>); refitted in the browser (<code>lib/hipsGroupings.ts</code>).</>}>
      {!file || !live ? <LoadState what="the IMPROVE groupings" error={error} command={GROUPINGS_COMMAND} /> : <>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 24, justifyContent: 'center' }}>
          <div><ScatterPlot title={`${live.groups.length} groups, ${live.used.toLocaleString()} filters`} points={points} xLabel="TOR EC on the filter (µg), group median" yLabel="HIPS optical depth τ, group median"
            xDomain={[0, top]} yDomain={[0, 3.4]} nice={false} maxSide={SIDE * 0.85} outside="pin" note={false} under={under} /></div>
          <div><ScatterPlot title="The fit's error at each ceiling" points={profilePoints} xScale="log" xLabel="Ceiling τ∞ (log scale)" yLabel={`RMS residual (${choice.residual === 'log' ? 'log τ' : 'τ'})`}
            xDomain={[0.5, 100]} yDomain={[0, rmsTop]} nice={false} maxSide={SIDE * 0.85} outside="pin" note={false} under={profileUnder} /></div>
        </div>
        <Key items={[
          { label: 'Group medians (size: filters)', color: INK.text, kind: 'dot' },
          { label: 'Curve through this grouping', color: LIVE, kind: 'line' },
          { label: 'Published curve, and its 90 % interval', color: MEASURED, kind: 'dash' },
        ]} />
        <Readouts items={[
          { label: 'τ∞, this grouping', value: fit ? at(fit.tauInf) : '—', color: LIVE,
            sub: row ? `90 % ${fmt(row.tau_inf_lo, 2)}–${fmt(row.tau_inf_hi, 2)} (${file.n_boot} site resamples, exact medians ${fmt(row.tau_inf, 2)})` : 'interval: only for the sweep’s groupings (log τ, smallest group 150)' },
          { label: 'Published', value: fmt(published.tau_inf, 2), color: MEASURED, sub: `90 % ${fmt(ciLo, 2)}–${fmt(ciHi, 2)}` },
          { label: 'Groups in the fit', value: String(live.groups.length), sub: live.dropped ? `${live.dropped} under ${choice.minN} filters left out (${live.droppedFilters.toLocaleString()} filters)` : 'none left out' },
          { label: `× on the median Addis filter (τ ${fmt(addisTau, 2)})`, value: fit ? fmt(correction(addisTau, fit.tauInf), 2) : '—', sub: `published ×${fmt(correction(addisTau, published.tau_inf), 2)}` },
        ]} />
        {fit && !fit.atBound && ceil.key !== 'custom' && <button type="button" className="btn" onClick={() => ceil.applyCustom(fit.tauInf)}>Linearise this page with τ∞ {fmt(fit.tauInf, 2)}</button>}
        {ceil.key === 'custom' && <p className="chart-note">This page is linearised with τ∞ {fmt(ceil.ceiling, 2)} from a grouping of yours; choose another ceiling under Page options, Ceiling τ∞.</p>}
        <Note explain>Why the published grouping is by the <em>previous</em> sample's loading: grouping by a filter's own TOR EC sorts
          filters partly by TOR's measurement error, which makes the curve look more bent than it is (τ∞ 1.1 to 1.9 under equal-count or
          log-spaced edges; the published edges happen to land at 2.27); grouping by its own
          τ sorts by HIPS's error and hides the bend altogether (no ceiling). The previous sample at the same site, within 7 days, tracks
          how loaded this filter is likely to be but knows nothing of this filter's errors. The 17 edges are roughly even on a log scale
          (each about 1.3 to 1.5 times the last) so that every group has hundreds of filters, and groups under 150 filters, the darkest
          tail, are left out. The sweep below shows that, grouped this way, the edges barely matter.</Note>
      </>}
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 2. which groups the ceiling rests on
export function CeilingLeverage({ choice }: { choice: Choice }) {
  const { data: file, error } = useExport<GroupingsFile>(FILE)
  const live = useGroupingFit(file, choice)
  const drops = useMemo(() => (live?.fit ? dropOne(live.groups, choice.residual) : []), [live, choice.residual])
  const points = useMemo<ScatterPoint[]>(() => drops.map(({ group: g, tauInf }, i) => ({
    key: `d${i}`, x: g.loading, y: Math.min(tauInf, 20), color: LIVE, r: 3 + Math.sqrt(g.n) / 40,
    tip: [`Without group ${i + 1}`, `median TOR EC ${fmt(g.loading, 2)} µg · ${g.n.toLocaleString()} filters`, `τ∞ ${at(tauInf)}`],
  })), [drops])
  const fit = live?.fit
  return (
    <ChartFrame id="hc-leverage" title="Which groups the ceiling rests on — τ∞ refitted with each group left out in turn"
      subtitle="Each dot is the ceiling refitted without one group, at that group's median loading. The light groups barely move it; the darkest few decide it, because they are the only ones well into the bend. The line is the fit with every group."
      provenance={<><code>{FILE}</code>; refitted in the browser.</>}>
      {!file || !live ? <LoadState what="the IMPROVE groupings" error={error} command={GROUPINGS_COMMAND} /> : !fit ? <p className="chart-note">Too few groups to fit.</p> : <>
        <ScatterPlot points={points} xLabel="TOR EC on the filter (µg), median of the group left out" yLabel="τ∞ without that group"
          xInclude={[0]} yInclude={[0, fit.tauInf * 1.4]} maxSide={SIDE * 0.85} outside="pin" note={false}
          under={({ y, iw }) => <Marker orient="h" at={y(fit.tauInf)} from={0} to={iw} color={INK.text} label={`every group: ${fmt(fit.tauInf, 2)}`} />} />
        <Note>Leaving out the darkest group moves τ∞ to {at(drops[drops.length - 1]?.tauInf ?? NaN)}; leaving out the lightest, to {at(drops[0]?.tauInf ?? NaN)}.</Note>
      </>}
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 3. the ceiling under every grouping
export function CeilingSweep({ data, choice }: { data: Linearise; choice: Choice }) {
  const { data: file, error } = useExport<GroupingsFile>(FILE)
  const live = useGroupingFit(file, choice)
  const published = data.improve_ceilings[0]
  const Y_TOP = 5
  const points = useMemo<ScatterPoint[]>(() => (file?.sweep ?? []).map((r: SweepRow) => ({
    key: `${r.basis}/${r.rule}/${r.groups}`, x: r.used, y: Math.min(r.tau_inf, Y_TOP), color: BASIS_COLOR[r.basis], r: r.rule === 'published' ? 6 : 4,
    shape: r.rule === 'published' ? 'square' as const : 'dot' as const, hollow: r.rule === 'log',
    tip: [`${BASIS_LABEL[r.basis]} · ${RULE_LABEL[r.rule]}${r.rule === 'published' ? '' : `, ${r.groups} asked`}`, `${r.used} groups in the fit`,
      `τ∞ ${at(r.tau_inf)}`, r.tau_inf < TAU_BOUNDS[1] * 0.98 ? `90 % ${fmt(r.tau_inf_lo, 2)}–${fmt(r.tau_inf_hi, 2)}` : 'the medians show no bend'],
  })), [file])
  const liveX = live?.groups.length ?? NaN, liveY = live?.fit ? Math.min(live.fit.tauInf, Y_TOP) : NaN
  const under = ({ x, y, iw }: ScatterScales) => <g pointerEvents="none">
    <rect x={0} y={y(published.tau_inf_hi)} width={iw} height={Math.max(0, y(published.tau_inf_lo) - y(published.tau_inf_hi))} fill={MEASURED} opacity={0.08} />
    <Marker orient="h" at={y(published.tau_inf)} from={0} to={iw} color={MEASURED} label={`published ${fmt(published.tau_inf, 2)}`} />
    {(file?.sweep ?? []).filter((r) => r.tau_inf < Y_TOP).map((r) => <line key={`${r.basis}/${r.rule}/${r.groups}`} x1={x(r.used)} x2={x(r.used)}
      y1={y(Math.min(Y_TOP, r.tau_inf_lo))} y2={y(Math.min(Y_TOP, r.tau_inf_hi))} stroke={BASIS_COLOR[r.basis]} strokeWidth={1.25} opacity={0.6} />)}
    <text x={iw - 4} y={y(Y_TOP) + 12} textAnchor="end" fontSize={10} fontFamily={FONT.mono} fill={INK.muted}>at the top: no ceiling (no bend in the medians)</text>
  </g>
  const prev = (file?.sweep ?? []).filter((r) => r.basis === 'previous' && r.used >= 6)
  return (
    <ChartFrame id="hc-sweep" title="The ceiling under every grouping — grouped by the previous sample's loading it stays near the published value whatever the edges; grouped by a filter's own TOR or HIPS it does not"
      subtitle={`τ∞ for each way of grouping (colour), each edge rule (square: published edges; filled: equal count; hollow: log-spaced) and 4 to 40 groups, fitted on exact medians (log τ, groups under ${file?.min_n ?? 150} filters left out), with 90 % intervals from ${file?.n_boot ?? 100} resamples of whole sites. The shaded band is the published interval. The dashed ring is the grouping in Page options.`}
      provenance={<><code>{FILE}</code> (<code>gallery/data/export_hips_improve_groupings.py</code>).</>}>
      {!file ? <LoadState what="the IMPROVE groupings" error={error} command={GROUPINGS_COMMAND} /> : <>
        <ScatterPlot points={points} xLabel="Groups in the fit" yLabel="Ceiling τ∞" xDomain={[0, 42]} yDomain={[0, Y_TOP]} nice={false}
          maxSide={SIDE} outside="pin" note={false} under={under}>
          {({ x, y }) => Number.isFinite(liveY) ? <PinRing cx={x(liveX)} cy={y(liveY)} r={6} /> : null}
        </ScatterPlot>
        <Key items={BASES.map((b) => ({ label: BASIS_LABEL[b], color: BASIS_COLOR[b], kind: 'dot' as const }))} />
        {prev.length > 0 && <Note>Grouped by the previous sample's loading with 6 or more groups, τ∞ runs from {fmt(Math.min(...prev.map((r) => r.tau_inf)), 2)} to {fmt(Math.max(...prev.map((r) => r.tau_inf)), 2)},
          inside or near the published interval. Grouped by a filter's own TOR EC it falls to 1.1 to 1.9 under any edges but the published ones (the bend overstated by TOR's error);
          grouped by its own τ there is no ceiling at all (the bend hidden by HIPS's error).</Note>}
      </>}
    </ChartFrame>
  )
}
