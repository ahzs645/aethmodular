import { useMemo, type ReactNode } from 'react'
import { AxisChoices, type AxisOption } from '@/components/AxisChoices'
import { ChartFrame, ExplainOnly, Note } from '@/components/ChartFrame'
import { Select } from '@/components/Controls'
import { ExportState, LoadState } from '@/components/ExportState'
import { type MeetingBundle } from '@/components/Figure'
import { Key } from '@/components/Key'
import { PageToc } from '@/components/PageToc'
import { ScatterPlot, type ScatterPoint, type ScatterScales } from '@/components/ScatterPlot'
import { ScrubField } from '@/components/ScrubField'
import { useExport } from '@/lib/data'
import type { FilterRecord } from '@/lib/highlight'
import { fmt } from '@/lib/format'
import { CENTRAL_CEILING, correction, EXPECTED_FROM, HIPS_SITE_READINGS, HIPS_STATUSES, IMPROVE_CEILINGS, SEEN_READING, SHAPE_ONLY_FROM, siteExposure, siteReading, STATUS_USE, statusMeaning, statusOf, underRead, USE_UP_TO, type HipsStatus } from '@/lib/hipsLinearise'
import { HIPS_RULES, MIN_TEST, networkFilters, ruleFactors, ruleLabel, RULE_KEYS, scoreRule, siteOutcomes, stepAt, type HipsRule, type NetFilter } from '@/lib/hipsRules'
import { useMac } from '@/lib/mac'
import { SidebarControls } from '@/lib/pageSidebar'
import { RESEARCH_SERIES } from '@/lib/palettes'
import type { SpartanReference } from '@/lib/referencePopulation'
import { median } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { Bend, CeilingOptions, COMMAND, CorrectionPlot, SIDE, useCeiling, Worked, type Ceil, type Linearise } from '@/pages/HipsLinearisePage'
import { DataTable, LINEAR, MEASURED, Readouts, type Column } from '@/pages/hipsLineariseParts'

const PROCEDURE = 'hips-procedure-20261111'
const SCREEN = 'research/ftir_hips_chem/workflows/run_hips_network_20261111.py'
const SOURCE = <>The public SPARTAN table (<code>spartan_reference.json</code>): every counted filter with HIPS and a sampled volume, the 11 Nov screen's own input ({SCREEN}; <code>docs/hips-loading-protocol.md</code>). Every number is recomputed here from the filters at the ceiling and rule chosen (<code>lib/hipsRules.ts</code>), and reproduces the screen's saved exponents (<code>hipsRules.test.mjs</code>).</>

/** the readings of the 11 Nov screen, in the order they are read, each with a colour of its own */
const READINGS = [
  'expected and seen', 'expected, not seen', 'expected, cannot be tested', 'not expected, but compressed',
  'not expected, not seen', 'not expected', 'HIPS rises faster than the reference', 'cannot tell',
] as const
const READING_COLOR: Record<string, string> = Object.fromEntries(READINGS.map((r, i) => [r, RESEARCH_SERIES[[3, 1, 4, 5, 2, 2, 0, 7][i]]]))

/** an exponent whose 95 % interval is wider than this says nothing (hips_loading.WIDE); its bar is drawn faint */
const WIDE = 0.6
const pct = (share: number, digits = 0) => `${fmt(100 * share, digits)} %`
const short = (code: string) => (siteReading(code)?.site ?? code).split(',')[0]
/** whether a rule moved a value at all: a site it leaves where it was is drawn with its measured dot alone */
const moves = (before: number, after: number) => Math.abs(after - before) > 1e-9 * Math.max(1, Math.abs(before))
const Part = ({ children }: { children: string }) => <h2 className="meeting-section-title">{children}</h2>

/** a filter's place across its site's row: the same every time, from its id */
function jitter(id: string): number {
  let h = 2166136261
  for (let i = 0; i < id.length; i++) h = Math.imul(h ^ id.charCodeAt(i), 16777619)
  return ((h >>> 0) / 4294967295 - 0.5) * 0.62
}

/**
 * A filter dot that clicks like every filter in the gallery: pinned on every chart of the page, its full record in
 * the drawer (a SPARTAN id resolves to its site's row there), with what this page computed for it on top.
 */
function filterRecord(f: NetFilter, chart: string, extra: FilterRecord['fields'] = []): FilterRecord {
  return { id: f.id, site: siteReading(f.code)?.site ?? f.code, date: f.date, source: `Linearising HIPS · ${chart}`, fields: [
    ...extra, ['HIPS Fabs (Mm⁻¹)', f.fabs], ['Sampled volume (m³)', f.volume], ['HIPS optical depth τ', f.tau],
    ['Expected under-read', pct(underRead(f.tau))], ['FTIR EC (µg/m³)', f.ec], ['PM2.5 (µg/m³)', f.pm],
  ] }
}

/** the site picked by a click on one of its site dots (URL `hn.site`): its filters stand out on every filter chart */
function useFocus(codes: readonly string[]) {
  const options = useMemo(() => ['', ...[...codes].sort((a, b) => short(a).localeCompare(short(b)))], [codes])
  const [site, setSite] = useSearchParam<string>('hn.site', '', codes.length ? options : [])
  const control = <Select label="Site in focus" value={site} options={options} onChange={setSite} optionLabel={(c) => (c ? short(c) : 'none')}
    title="A site whose filters are drawn in full on the filter charts, the rest faded. Clicking a site's dot on a site chart picks it too; click it again to clear." />
  return { site: site || null, toggle: (code: string) => setSite(code === site ? '' : code), clear: () => setSite(''), control }
}
type Focus = ReturnType<typeof useFocus>

/** the focus site's filters drawn on top, the rest faded */
function withFocus(points: ScatterPoint[], site: string | null): ScatterPoint[] {
  if (!site) return points
  return [...points.filter((p) => p.site !== site).map((p) => ({ ...p, dim: 0.15 })), ...points.filter((p) => p.site === site)]
}

/** what a filter chart says while a site is in focus, and the way out of it */
function FocusNote({ focus, filters }: { focus: Focus; filters: readonly NetFilter[] }) {
  if (!focus.site) return null
  const n = filters.filter((f) => f.code === focus.site).length
  return <Note>In focus: {short(focus.site)}'s {n.toLocaleString()} filters, drawn in full; the other sites' are faded. <button type="button" className="link-btn" onClick={focus.clear}>Show every site</button></Note>
}

/** a site dot's click: put that site in focus, or take it out; the dot's key is its site code, maybe with a -suffix */
const siteClick = (focus: Focus) => (p: ScatterPoint) => { const k = p.key ?? ''; focus.toggle(k.includes('-') ? k.slice(0, k.lastIndexOf('-')) : k) }
const SITE_TIP = 'click to put its filters in focus on the filter charts'
/** a site's dots drawn with a ring while it is in focus */
const ring = (site: string | null, code: string) => (site === code ? { stroke: INK.text, strokeWidth: 2 } : {})

/** the counted SPARTAN filters, and the statuses' colours as the 11 Nov procedure page draws them */
function useNetwork() {
  const { data: spartan, error } = useExport<SpartanReference>('spartan_reference.json')
  const { data: bundle } = useExport<MeetingBundle>(`meetings/${PROCEDURE}.json`)
  const filters = useMemo(() => networkFilters(spartan?.rows ?? []), [spartan])
  const exposure = useMemo(() => siteExposure(filters), [filters])
  const statusColor = useMemo(() => (s: HipsStatus) => bundle?.colors?.[s] ?? RESEARCH_SERIES[HIPS_STATUSES.indexOf(s) + 2], [bundle])
  return { filters, exposure, statusColor, error, loaded: !!spartan }
}
type Network = ReturnType<typeof useNetwork>

/** the page's rule: which filters the curve is applied to (URL `hn.rule`, and `hn.cut` for the cut-off in percent) */
function useRule() {
  const [rule, setRule] = useSearchParam<HipsRule>('hn.rule', 'filter', RULE_KEYS)
  const [cutText, setCut] = useSearchParam<string>('hn.cut', '20')
  const cut = Math.min(0.4, Math.max(0.05, (Number(cutText) || 20) / 100))
  const control = <>
    <Select label="Applied to" value={rule} options={[...RULE_KEYS]} onChange={(v) => setRule(v as HipsRule)} optionLabel={(k) => ruleLabel(k as HipsRule)}
      title="The rule by which the one curve is applied. Every filter: each by its own factor. Sites where expected, or seen: the same, only at those sites. Filters past a cut-off: only filters whose expected under-read is past it. One factor per site: the site's median filter's factor on all of its filters." />
    {rule === 'cut' && <span className="control" title="The expected under-read (optical depth ÷ 2.26) past which a filter is corrected: drag sideways or click to type">Cut-off <ScrubField
      label="Cut-off" value={Math.round(cut * 100)} min={5} max={40} step={1} width={78} unit="%" onChange={(v) => setCut(String(Math.round(v)))} /></span>}
  </>
  return { rule, cut, control }
}
type Rule = ReturnType<typeof useRule>

const ruleText = (r: Rule) => (r.rule === 'cut' ? `filters past a ${pct(r.cut)} under-read` : HIPS_RULES.find((x) => x.key === r.rule)!.short)

/**
 * Linearising HIPS across SPARTAN, as a story: the effect (IMPROVE, against thermal EC), the fix (one curve read
 * backwards on each filter), where the effect is expected and where a site's own filters show it, and then the same
 * filters before and after under each rule by which the fix could be applied: to every filter, only at the sites it
 * affects, only to filters past a cut-off, or as one factor per site. A ceiling and a rule, both in Page options,
 * drive every chart from part 3 on; nothing there is saved, it is recomputed from the public SPARTAN table.
 */
export function HipsStoryPage() {
  const { data, error } = useExport<Linearise>('hips_linearise/linearise.json')
  const ceil = useCeiling(data)
  const net = useNetwork()
  const rule = useRule()
  const codes = useMemo(() => [...net.exposure.keys()], [net.exposure])
  const focus = useFocus(codes)
  if (error || !data) return (
    <ChartFrame id="hn-missing" title="Linearising HIPS across SPARTAN" provenance={SCREEN}>
      <LoadState what="the linearisation export" error={error} command={COMMAND} />
    </ChartFrame>
  )
  if (!net.loaded) return <ExportState title="Linearising HIPS across SPARTAN" what="the public SPARTAN table" error={net.error} command="uv run aeth gallery rebuild" />
  return (
    <PageToc>
      <CeilingOptions ceil={ceil} />
      <SidebarControls title="How it is applied"><div role="group" aria-label="How the linearisation is applied" style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>{rule.control}{focus.control}</div></SidebarControls>

      <Part>1. The effect: on IMPROVE, HIPS falls behind thermal EC as the filter darkens</Part>
      <Bend data={data} ceil={ceil} />

      <Part>2. The fix: each filter is read back through the one curve</Part>
      <Worked data={data} ceil={ceil} />
      <NetworkOnCurve net={net} data={data} ceil={ceil} focus={focus} />

      <Part>3. Isolating it on SPARTAN: where it is expected, and where a site's own filters show it</Part>
      <SiteStrips net={net} ceil={ceil} focus={focus} />
      <SiteReadings focus={focus} />

      <Part>4. Before and after: the same filters under each rule</Part>
      <RuleResponse net={net} data={data} ceil={ceil} rule={rule} focus={focus} />
      <SiteLevels net={net} ceil={ceil} rule={rule} focus={focus} />
      <SiteCheck net={net} ceil={ceil} rule={rule} focus={focus} />

      <Part>5. Per filter or per site: what each rule changes, and what it leaves</Part>
      <RuleTable net={net} ceil={ceil} rule={rule} />
      <Learned />
    </PageToc>
  )
}

// ------------------------------------------------------------------ 2. every filter on the curve
function NetworkOnCurve({ net, data, ceil, focus }: { net: Network; data: Linearise; ceil: Ceil; focus: Focus }) {
  const c = ceil.ceiling
  const lo = data.ceilings.find((k) => k.key === 'lo')!.tau_inf, hi = data.ceilings.find((k) => k.key === 'hi')!.tau_inf
  const { filters, statusColor } = net
  const points = useMemo<ScatterPoint[]>(() => withFocus(filters.flatMap((f) => {
    const k = correction(f.tau, c), status = statusOf(underRead(f.tau))
    return Number.isFinite(k) ? [{ id: f.id, key: f.id, site: f.code, x: f.tau, y: k, color: statusColor(status), r: 2.6,
      tip: [`${f.id} · ${short(f.code)} · ${f.date}`, `Fabs ${fmt(f.fabs, 1)} Mm⁻¹ × ${fmt(f.volume, 2)} m³ → τ ${fmt(f.tau, 2)}`, `× ${fmt(k, 2)} → ${fmt(f.fabs * k, 1)} Mm⁻¹`, status],
      rec: filterRecord(f, 'every filter on the curve', [['Status', status], ['Its own factor', fmt(k, 3)], ['HIPS Fabs linearised (Mm⁻¹)', f.fabs * k]]) }] : []
  }), focus.site), [filters, c, statusColor, focus.site])
  const share = (test: (f: NetFilter) => boolean) => filters.filter(test).length / filters.length
  const med = median(filters.map((f) => f.tau))
  return (
    <ChartFrame id="hn-curve" title="Every SPARTAN filter on the one curve — the factor is the filter's own, and close to 1 on a light filter"
      subtitle={`Correction factor 1 / (1 − τ/τ∞) against each filter's measured optical depth (Fabs × its sampled volume ÷ 353): ${filters.length.toLocaleString()} filters at ${net.exposure.size} sites. The line and its band are the same for every filter; a filter differs only in where it falls.`}
      provenance={SOURCE}>
      <CorrectionPlot points={points} ceiling={c} lo={lo} hi={hi} median={med} medianLabel={`network median τ ${fmt(med, 2)} → × ${fmt(correction(med, c), 2)}`} lightBox />
      <FocusNote focus={focus} filters={filters} />
      <Key items={[{ label: `chosen ceiling ${fmt(c, 2)}`, color: LINEAR, kind: 'line' }, { label: `ceiling ${fmt(lo, 2)} to ${fmt(hi, 2)} (90 %)`, color: LINEAR, kind: 'faded' },
        ...HIPS_STATUSES.map((s) => ({ label: s, color: statusColor(s), kind: 'dot' as const }))]} />
      <Readouts items={[
        { label: 'Moved by under 5 %', value: pct(share((f) => correction(f.tau, c) <= 1.05)), sub: 'of filters, at this ceiling' },
        { label: 'Moved by over 25 %', value: pct(share((f) => !(correction(f.tau, c) <= 1.25))), sub: 'of filters' },
        ...HIPS_STATUSES.map((s) => ({ label: s, value: pct(share((f) => statusOf(underRead(f.tau)) === s)), color: statusColor(s) })),
        { label: 'At or past the ceiling', value: filters.filter((f) => !(f.tau < c)).length.toLocaleString(), sub: 'no finite value' },
      ]} />
      <Note explain>Nothing has to decide which filters to correct: a light filter's factor is already close to 1, and it grows smoothly as the filter darkens. Statuses are set at the central ceiling: used up to a {pct(USE_UP_TO)} under-read, with caution to {pct(SHAPE_ONLY_FROM)}, for shape only beyond.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 3. where it is expected, and where it is seen
/** the measures the strips can be read on: all rise with the filter's optical depth, so the rows and their order stay */
type StripX = 'tau' | 'under' | 'factor' | 'spread'
const STRIP_X: readonly StripX[] = ['tau', 'under', 'factor', 'spread']
const STRIP_AXES: AxisOption[] = [
  { key: 'tau', group: 'Measured', label: 'HIPS optical depth τ of the filter', roles: ['x'], hint: 'measured' },
  { key: 'under', group: 'From the IMPROVE curve', label: 'Expected under-read of the filter', roles: ['x'], hint: 'τ ÷ 2.26' },
  { key: 'factor', group: 'From the IMPROVE curve', label: 'Correction factor of the filter', roles: ['x'], hint: 'chosen ceiling' },
  { key: 'spread', group: 'From the IMPROVE curve', label: 'Uncertainty of the correction (±)', roles: ['x'], hint: '90 % ceilings' },
]
/** a value past the ceiling, where a factor has no finite answer: drawn pinned to the right edge */
const OFF_RIGHT = 1e6
const CEIL_LO = IMPROVE_CEILINGS.find((c) => c.key === 'lo')!.tau_inf, CEIL_HI = IMPROVE_CEILINGS.find((c) => c.key === 'hi')!.tau_inf
/** ± share of the central factor: half the factor's range between the two ends of the ceiling's 90 % interval */
const spreadOf = (tau: number) => {
  const a = correction(tau, CEIL_LO), b = correction(tau, CEIL_HI)
  return Number.isFinite(a) && Number.isFinite(b) ? (a - b) / 2 / correction(tau, CENTRAL_CEILING) : NaN
}
/** the reference marks, as expected under-reads, with what each one means on hover */
const STRIP_MARKS = [
  { share: EXPECTED_FROM, label: 'site: expected', hint: `A site whose median filter under-reads by more than ${EXPECTED_FROM * 100} % is one where the bend is expected (the 11 Nov screen's rule). It says nothing about which of its filters.` },
  { share: USE_UP_TO, label: 'caution', hint: statusMeaning('caution') },
  { share: SHAPE_ONLY_FROM, label: 'shape only', hint: statusMeaning('shape only') },
]

/** Each site's filters by how dark they are: the site's median decides "expected", and the spread around it is what a site rule cannot see. */
function SiteStrips({ net, ceil, focus }: { net: Network; ceil: Ceil; focus: Focus }) {
  const { filters, exposure, statusColor } = net
  const [xKey, setX] = useSearchParam<StripX>('hn.strips.x', 'tau', STRIP_X)
  const c = ceil.ceiling
  /** the chosen measure of a filter (or a site's median filter) from its optical depth, on the axis's own scale */
  const axis = useMemo(() => {
    const finite = (v: number) => (Number.isFinite(v) ? v : OFF_RIGHT)
    return {
      tau: { of: (t: number) => t, domain: [0, 0.7 * CENTRAL_CEILING] as [number, number], tick: (v: number) => fmt(v, 1), text: (v: number) => `τ ${fmt(v, 2)}` },
      under: { of: (t: number) => 100 * underRead(t), domain: [0, 70] as [number, number], tick: (v: number) => `${v} %`, text: (v: number) => `${fmt(v, 0)} %` },
      factor: { of: (t: number) => finite(correction(t, c)), domain: [1, 3.5] as [number, number], tick: (v: number) => `×${v}`, text: (v: number) => (v >= OFF_RIGHT ? 'past τ∞' : `×${fmt(v, 2)}`) },
      spread: { of: (t: number) => finite(100 * spreadOf(t)), domain: [0, 30] as [number, number], tick: (v: number) => `±${v} %`, text: (v: number) => (v >= OFF_RIGHT ? 'unbounded' : `±${fmt(v, 0)} %`) },
    }[xKey]
  }, [xKey, c])
  const sites = useMemo(() => [...exposure.values()].sort((a, b) => b.underRead - a.underRead), [exposure])
  const row = useMemo(() => new Map(sites.map((s, i) => [s.code, i])), [sites])
  const points = useMemo<ScatterPoint[]>(() => withFocus(filters.map((f) => {
    const u = underRead(f.tau), status = statusOf(u), k = correction(f.tau, c), sp = spreadOf(f.tau)
    const factorText = Number.isFinite(k) ? `× ${fmt(k, 2)} at τ∞ ${fmt(c, 2)}` : `past the ceiling ${fmt(c, 2)}`
    const spreadText = Number.isFinite(sp) ? `known to ±${fmt(100 * sp, 0)} %` : 'not bounded by the ceiling interval'
    return { id: f.id, key: f.id, site: f.code, x: axis.of(f.tau), y: row.get(f.code)! + 0.5 + jitter(f.id), color: statusColor(status), r: 2.2,
      tip: [`${f.id} · ${short(f.code)} · ${f.date}`, `τ ${fmt(f.tau, 2)} → expected under-read ${pct(u)}`, `factor ${factorText}, ${spreadText}`, `${status}: use for ${STATUS_USE[status]}`],
      rec: filterRecord(f, 'how dark each site\'s filters are', [['Status', status], ['Use for', STATUS_USE[status]], ['Correction factor', Number.isFinite(k) ? fmt(k, 3) : 'past the ceiling'],
        ['Correction known to', Number.isFinite(sp) ? `±${fmt(100 * sp, 1)} %` : 'not bounded'], ['Site median filter under-reads', pct(exposure.get(f.code)?.underRead ?? NaN)]]) }
  }), focus.site), [filters, row, statusColor, exposure, focus.site, axis, c])
  const bands = useMemo(() => sites.map((s) => ({ key: s.code, label: `${short(s.code)} · ${axis.text(axis.of(s.medianTau))}`,
    color: focus.site === s.code ? INK.accent : s.expected ? INK.text : INK.muted })), [sites, focus.site, axis])
  const marks = STRIP_MARKS.map((m) => ({ ...m, at: axis.of(m.share * CENTRAL_CEILING) }))
  /** a second scale along the top: the expected under-read, wherever the bottom axis is something else */
  const top = xKey === 'under' ? [] : [0, 10, 20, 30, 40, 50, 60, 70].map((u) => ({ u, at: axis.of((u / 100) * CENTRAL_CEILING) }))
  const children = ({ x, y }: ScatterScales) => {
    const [d0, d1] = x.domain()
    return <g>
      <g pointerEvents="none">
        {sites.map((s, i) => <line key={s.code} x1={x(Math.min(d1, axis.of(s.medianTau)))} x2={x(Math.min(d1, axis.of(s.medianTau)))} y1={y(i + 0.1)} y2={y(i + 0.9)} stroke={INK.text} strokeWidth={2} />)}
        {top.filter((t) => t.at >= d0 && t.at <= d1).map((t) => <g key={t.u} transform={`translate(${x(t.at)},0)`}>
          <line y1={0} y2={-4} stroke={INK.muted} />
          <text y={-7} textAnchor="middle" fontSize={10} fontFamily={FONT.mono} fill={INK.muted}>{t.u}%</text>
        </g>)}
        {top.length > 0 && <text x={-16} y={-7} textAnchor="end" fontSize={10} fontFamily={FONT.mono} fill={INK.muted}>expected under-read</text>}
      </g>
      {marks.filter((m) => m.at >= d0 && m.at <= d1).map((m, i, shown) => {
        // a label that would run into the one before it (the marks crowd on the uncertainty axis) goes up a row
        const up = i > 0 && x(m.at) - x(shown[i - 1].at) < 7 * shown[i - 1].label.length + 8 && i % 2 === 1
        return <text key={m.label} x={x(m.at) + 4} y={(top.length ? -21 : -3) - (up ? 12 : 0)} fontSize={10.5} fontFamily={FONT.mono} fill={INK.muted}
          style={{ cursor: 'help' }}><title>{m.hint}</title>{m.label}</text>
      })}
    </g>
  }
  const expected = new Set(sites.filter((s) => s.expected).map((s) => s.code))
  const darkAtLight = filters.filter((f) => !expected.has(f.code) && underRead(f.tau) > USE_UP_TO).length
  const lightAtDark = filters.filter((f) => expected.has(f.code) && underRead(f.tau) <= EXPECTED_FROM).length
  const atExpected = filters.filter((f) => expected.has(f.code)).length
  const lines = 'Lines at the 10 % mark (the bend is expected at a site whose median filter is past it), 20 % and 30 % (a filter\'s status); hover a status in the key, or a line\'s name, for what it allows.'
  const subtitle = {
    tau: `HIPS optical depth of every filter (Fabs × its sampled volume ÷ 353), the measurement itself, one row per site, darkest first; the bar is the site's median filter, and the row's label gives it. Along the top: the under-read the IMPROVE curve expects at the central ceiling (τ ÷ ${fmt(CENTRAL_CEILING, 2)}). ${lines}`,
    under: `Expected under-read (optical depth ÷ ${fmt(CENTRAL_CEILING, 2)}, the central ceiling) of every filter, one row per site, darkest first; the bar is the site's median filter, and the row's label gives it. ${lines}`,
    factor: `The factor 1 / (1 − τ/τ∞) that linearises each filter, at the chosen ceiling (τ∞ ${fmt(c, 2)}, Page options), one row per site, darkest first; the bar is the site's median filter. A filter past the ceiling has no finite factor and sits on the right edge. ${lines}`,
    spread: `How well each filter's correction is known: half the range of its factor between the two ends of the ceiling's 90 % interval (${fmt(CEIL_LO, 2)} and ${fmt(CEIL_HI, 2)}), as a share of its factor at the central ceiling. This is what the statuses measure. A filter darker than ${fmt(CEIL_LO, 2)} is not bounded and sits on the right edge. ${lines}`,
  }[xKey]
  return (
    <ChartFrame id="hn-strips" title="How dark each site's filters are — a site is read by its median filter, and its filters spread well to either side of it"
      subtitle={subtitle} provenance={SOURCE}>
      <AxisChoices options={STRIP_AXES} x={[xKey, (v) => setX(v as StripX)]} noun="measures">
        <ScatterPlot points={points} yBands={bands} bandLabelSize={11} xDomain={axis.domain} nice={false} outside="pin" height={sites.length * 19}
          margin={{ top: top.length ? 48 : 30, right: 24, bottom: 46, left: 150 }} xLabel={STRIP_AXES.find((o) => o.key === xKey)!.label} xFormat={axis.tick} pick="x"
          refX={marks.map((m) => m.at)} density="dense" note={false}>{children}</ScatterPlot>
      </AxisChoices>
      <FocusNote focus={focus} filters={filters} />
      <Key items={[...HIPS_STATUSES.map((s) => ({ label: s, color: statusColor(s), kind: 'dot' as const, hint: statusMeaning(s) })),
        { label: 'the site\'s median filter', color: INK.text, kind: 'bar' as const, hint: 'Half the site\'s filters are lighter than this and half darker. The site is "expected" when it is past the 10 % mark.' }]} />
      <Readouts items={[
        { label: 'Sites where it is expected', value: `${expected.size} of ${sites.length}`, sub: `${atExpected.toLocaleString()} of ${filters.length.toLocaleString()} filters` },
        { label: 'Light filters at those sites', value: lightAtDark.toLocaleString(), sub: `under ${pct(EXPECTED_FROM)}, at a site past it` },
        { label: 'Dark filters at the other sites', value: darkAtLight.toLocaleString(), sub: `past ${pct(USE_UP_TO)}, at a site under ${pct(EXPECTED_FROM)}` },
      ]} />
      <Note explain>A status says how well a filter's correction is known, not how large it is: the IMPROVE ceilings (1.71 to 2.70) agree on a light filter's factor and part fast on a dark one. Choose the axis by clicking its title. Optical depth is the measurement and needs no ceiling; the under-read and the factor read it through the curve; the uncertainty is what the statuses are set on.</Note>
      <Note explain>This is why the correction is made per filter: a site's median says how much of its record is affected, but not which of its filters. A rule that works by site treats the dark filters of a light site as clean, and one factor per site treats every filter of a row as if it sat on the bar.</Note>
    </ChartFrame>
  )
}

/** Expected against seen, site by site: the under-read of the site's median filter, and the exponent its own filters give. */
function SiteReadings({ focus }: { focus: Focus }) {
  const sites = HIPS_SITE_READINGS
  const tested = useMemo(() => sites.filter((s) => s.exponent != null), [sites])
  const points = useMemo<ScatterPoint[]>(() => tested.flatMap((s) => {
    const tip = [s.site, `median filter under-reads ${pct(s.expected_under_read)} · ${s.tested_filters} filters tested`,
      `measured b ${fmt(s.exponent, 2)} [${fmt(s.exponent_lo, 2)}, ${fmt(s.exponent_hi, 2)}]`, `the curve predicts ${fmt(s.exponent_predicted, 2)}`, s.reading, SITE_TIP]
    return [
      { key: `${s.code}-p`, x: 100 * s.expected_under_read, y: s.exponent_predicted!, color: INK.muted, hollow: true, r: 4, tip },
      { key: s.code, x: 100 * s.expected_under_read, y: s.exponent!, color: READING_COLOR[s.reading] ?? INK.text, r: 5.5, opacity: 1, tip, ...ring(focus.site, s.code) },
    ]
  }), [tested, focus.site])
  const under = ({ x, y }: ScatterScales) => <g pointerEvents="none">
    {tested.map((s) => { const [y0, y1] = y.domain(); return <line key={s.code} x1={x(100 * s.expected_under_read)} x2={x(100 * s.expected_under_read)}
      y1={y(Math.min(y1, Math.max(y0, s.exponent_lo!)))} y2={y(Math.max(y0, Math.min(y1, s.exponent_hi!)))} stroke={READING_COLOR[s.reading] ?? INK.muted} strokeWidth={1.5} opacity={s.exponent_hi! - s.exponent_lo! > WIDE ? 0.22 : 0.75} /> })}
  </g>
  const LABELS = new Set(['ETAD', 'BDDU', 'INJA', 'INDH', 'CLST', 'CLTA', 'USPA'])
  const children = ({ x, y }: ScatterScales) => <g pointerEvents="none">
    <text x={x(100 * EXPECTED_FROM) + 4} y={-3} fontSize={10.5} fontFamily={FONT.mono} fill={INK.muted}>expected past 10 %</text>
    {tested.filter((s) => LABELS.has(s.code)).map((s) => { const left = x(100 * s.expected_under_read) > x.range()[1] * 0.8
      return <text key={s.code} x={x(100 * s.expected_under_read) + (left ? -9 : 8)} y={y(Math.min(y.domain()[1], s.exponent!)) + 4} textAnchor={left ? 'end' : 'start'}
        fontSize={10.5} fontFamily={FONT.mono} fill={INK.text}>{s.site.split(',')[0]}</text> })}
  </g>
  const count = (reading: string) => sites.filter((s) => s.reading === reading).length
  const pinned = tested.filter((s) => s.exponent! > 1.7).map((s) => s.site.split(',')[0])
  const present = READINGS.filter((r) => count(r) > 0)
  const untested = sites.filter((s) => s.exponent == null && s.expected_under_read > EXPECTED_FROM)
  return (
    <ChartFrame id="hn-readings" title="Expected and seen — at most dark sites the site's own filters show the bend the curve expects; a few do not, and some cannot be tested"
      subtitle={`Each site's exponent b in HIPS ∝ (FTIR EC)^b on its own filters (PM2.5 as the instrument; bar: 95 % interval, faint where it is too wide to read; ${MIN_TEST} filters needed), against the under-read expected of its median filter. Hollow: what the curve predicts for that site. No curve is assumed in the measured exponent.`}
      provenance={<>{SOURCE} Readings and intervals: <code>src/lib/hipsSites.json</code> (hips_loading.identify, 500 bootstrap draws).</>}>
      <ScatterPlot points={points} xLabel="Expected under-read of the site's median filter" yLabel="Exponent b on the site's own filters" xDomain={[0, 45]} yDomain={[0.3, 1.7]} nice={false}
        xFormat={(v) => `${v} %`} maxSide={SIDE} outside="pin" note={false} refY={[1]} refX={[100 * EXPECTED_FROM]} under={under} onPick={siteClick(focus)}>{children}</ScatterPlot>
      <Key items={[...present.filter((r) => tested.some((s) => s.reading === r)).map((r) => ({ label: `${r} (${tested.filter((s) => s.reading === r).length})`, color: READING_COLOR[r], kind: 'dot' as const })),
        { label: 'predicted by the curve', color: INK.muted, kind: 'hollow' as const }]} />
      <Readouts items={[
        { label: 'Expected', value: `${sites.filter((s) => s.expected_under_read > EXPECTED_FROM).length} of ${sites.length}`, sub: 'sites, by their median filter' },
        { label: 'Expected and seen', value: String(count(SEEN_READING)), color: READING_COLOR[SEEN_READING], sub: 'the site\'s own filters show it' },
        { label: 'Expected, not seen', value: String(count('expected, not seen')), color: READING_COLOR['expected, not seen'], sub: 'interval includes or exceeds 1' },
        { label: 'Expected, cannot be tested', value: String(untested.length), color: READING_COLOR['expected, cannot be tested'], sub: untested.map((s) => s.site.split(',')[0]).join(', ') },
      ]} />
      <Note>The reference here is FTIR EC, which is not thermal EC: an exponent below 1 says HIPS is compressed against FTIR EC, and cannot say which of the two bends.{pinned.length ? ` Pinned to the top edge, with wide intervals: ${pinned.join(', ')}.` : ''}</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 4. before and after, under the rule
/** What the rule multiplies each filter by, against how dark the filter is: on the curve, left at 1, a step, or a stripe per site. */
function RuleResponse({ net, data, ceil, rule, focus }: { net: Network; data: Linearise; ceil: Ceil; rule: Rule; focus: Focus }) {
  const c = ceil.ceiling
  const lo = data.ceilings.find((k) => k.key === 'lo')!.tau_inf, hi = data.ceilings.find((k) => k.key === 'hi')!.tau_inf
  const { filters, exposure } = net
  const applied = useMemo(() => ruleFactors(filters, { rule: rule.rule, ceiling: c, cut: rule.cut }, exposure), [filters, exposure, rule.rule, rule.cut, c])
  const own = useMemo(() => ruleFactors(filters, { rule: 'filter', ceiling: c }, exposure), [filters, exposure, c])
  const ruleName = ruleText(rule)
  const points = useMemo<ScatterPoint[]>(() => withFocus(filters.flatMap((f, i) => {
    const k = applied[i]
    return Number.isFinite(k) ? [{ id: f.id, key: f.id, site: f.code, x: f.tau, y: k, color: k === 1 ? MEASURED : LINEAR, r: 2.6,
      tip: [`${f.id} · ${short(f.code)} · ${f.date}`, `τ ${fmt(f.tau, 2)}, its own factor × ${fmt(own[i], 2)}`, k === 1 ? 'left as measured' : `× ${fmt(k, 2)} applied`],
      rec: filterRecord(f, `what the rule does (${ruleName})`, [['Its own factor', fmt(own[i], 3)], ['Under this rule', k === 1 ? 'left as measured' : `× ${fmt(k, 3)}`]]) }] : []
  }), focus.site), [filters, applied, own, focus.site, ruleName])
  const n = filters.length
  const corrected = applied.filter((k) => k !== 1).length
  const off = applied.filter((k, i) => Math.abs(k / own[i] - 1) > 0.1).length
  const darkLeft = filters.filter((f, i) => applied[i] === 1 && underRead(f.tau) > USE_UP_TO).length
  return (
    <ChartFrame id="hn-response" title="What each rule does to a filter — only its own factor follows the curve; a site rule leaves dark filters at 1, a cut-off puts a step in the response"
      subtitle={`The factor applied to each of the ${n.toLocaleString()} filters against its measured optical depth, with the curve every filter would follow by its own factor. Now: ${ruleText(rule)}. Change the rule or the ceiling in Page options.`}
      provenance={SOURCE}>
      <CorrectionPlot points={points} ceiling={c} lo={lo} hi={hi} />
      <FocusNote focus={focus} filters={filters} />
      <Key items={[{ label: 'the curve: each filter\'s own factor', color: LINEAR, kind: 'line' }, { label: 'corrected under this rule', color: LINEAR, kind: 'dot' }, { label: 'left as measured', color: MEASURED, kind: 'dot' }]} />
      <Readouts items={[
        { label: 'Filters corrected', value: `${corrected.toLocaleString()} of ${n.toLocaleString()}`, sub: pct(corrected / n) },
        { label: 'Over a tenth from their own factor', value: off.toLocaleString(), sub: pct(off / n), color: off ? MEASURED : undefined },
        { label: `Left at 1, past a ${pct(USE_UP_TO)} under-read`, value: darkLeft.toLocaleString(), sub: 'dark filters the rule does not touch' },
        ...(rule.rule === 'cut' ? [{ label: 'Step at the cut-off', value: `× ${fmt(stepAt(rule.cut, c), 2)}`, color: MEASURED, sub: `a filter at ${pct(rule.cut)} is × 1, one just past it × ${fmt(stepAt(rule.cut, c), 2)}` }] : []),
      ]} />
      <Note explain>Every filter: all dots sit on the line. Sites where expected, or seen: the dots off the line along the bottom are dark filters at sites the rule leaves out. A cut-off: the dots jump from 1 to the line at the cut-off, so two filters of nearly the same darkness end up a quarter apart at a 20 % cut-off. One factor per site: each site is a flat stripe, too high for its light filters and too low for its dark ones.</Note>
    </ChartFrame>
  )
}

/** Each site's median HIPS BC as measured and under the rule, with the answer every filter's own factor gives. */
function SiteLevels({ net, ceil, rule, focus }: { net: Network; ceil: Ceil; rule: Rule; focus: Focus }) {
  const { mac } = useMac()
  const outcomes = useMemo(() => siteOutcomes(net.filters, { rule: rule.rule, ceiling: ceil.ceiling, cut: rule.cut }), [net.filters, rule.rule, rule.cut, ceil.ceiling])
  const sites = useMemo(() => [...outcomes].sort((a, b) => b.fabs - a.fabs), [outcomes])
  const showOwn = rule.rule !== 'filter'
  const points = useMemo<ScatterPoint[]>(() => sites.flatMap((s, i) => {
    const before = s.fabs / mac, after = s.fabsAfter / mac, ownBc = s.fabsOwn / mac
    const tip = [siteReading(s.code)?.site ?? s.code, `as measured ${fmt(before, 2)} µg/m³`, s.touched ? `under this rule ${fmt(after, 2)} (× ${fmt(after / before, 2)})` : 'left as measured under this rule',
      `every filter by its own factor ${fmt(ownBc, 2)}`, `${s.n} filters · median filter under-reads ${pct(s.underRead)}`, SITE_TIP]
    return [
      ...(showOwn && Number.isFinite(ownBc) ? [{ key: `${s.code}-o`, x: ownBc, y: i + 0.5, color: INK.muted, hollow: true, r: 4.5, tip }] : []),
      { key: `${s.code}-b`, x: before, y: i + 0.5, color: MEASURED, r: 4.5, opacity: 1, tip, ...ring(focus.site, s.code) },
      // a site the rule leaves where it was keeps its one, measured, dot
      ...(Number.isFinite(after) && moves(before, after) ? [{ key: `${s.code}-a`, x: after, y: i + 0.5, color: LINEAR, r: 4.5, opacity: 1, tip, ...ring(focus.site, s.code) }] : []),
    ]
  }), [sites, mac, showOwn, focus.site])
  const under = ({ x, y }: ScatterScales) => <g pointerEvents="none">
    {sites.map((s, i) => Number.isFinite(s.fabsAfter) && <line key={s.code} x1={x(s.fabs / mac)} x2={x(s.fabsAfter / mac)} y1={y(i + 0.5)} y2={y(i + 0.5)} stroke={LINEAR} strokeWidth={2} opacity={0.45} />)}
  </g>
  const top = (value: (s: (typeof sites)[number]) => number) => [...sites].filter((s) => Number.isFinite(value(s))).sort((a, b) => value(b) - value(a)).slice(0, 4).map((s) => `${short(s.code)} ${fmt(value(s) / mac, 1)}`).join(', ')
  const moved = sites.filter((s) => s.fabsAfter / s.fabs > 1.1).length
  const max = Math.max(...sites.flatMap((s) => [s.fabs, s.fabsAfter, s.fabsOwn]).filter(Number.isFinite)) / mac
  return (
    <ChartFrame id="hn-levels" title="A site's level before and after — the dark sites rise and change order; a site rule decides which rows move at all"
      subtitle={`Median HIPS BC (Fabs ÷ MAC ${fmt(mac, 1)}) of each site, highest as measured first: as measured, and with ${ruleText(rule)} corrected.${showOwn ? ' Hollow: every filter by its own factor, for comparison.' : ''}`}
      provenance={SOURCE}>
      <ScatterPlot points={points} yBands={sites.map((s) => ({ key: s.code, label: short(s.code), color: focus.site === s.code ? INK.accent : s.touched ? INK.text : INK.muted }))} bandLabelSize={11}
        xDomain={[0, max * 1.06]} nice={false} height={sites.length * 19} margin={{ top: 14, right: 24, bottom: 46, left: 120 }}
        xLabel="Median HIPS BC (µg/m³)" density="sparse" note={false} under={under} onPick={siteClick(focus)} />
      <Key items={[{ label: 'as measured', color: MEASURED, kind: 'dot' }, { label: `linearised: ${ruleText(rule)}`, color: LINEAR, kind: 'dot' },
        ...(showOwn ? [{ label: 'every filter by its own factor', color: INK.muted, kind: 'hollow' as const }] : [])]} />
      <Readouts items={[
        { label: 'Highest, as measured', value: top((s) => s.fabs), color: MEASURED },
        { label: 'Highest, under this rule', value: top((s) => s.fabsAfter), color: LINEAR },
        { label: 'Sites that rise by over a tenth', value: `${moved} of ${sites.length}` },
      ]} />
      <Note>A rule that corrects some sites and not others puts corrected and measured medians on one axis: a site left out can fall behind one that was corrected only because of the rule. Greyed site names are the sites this rule leaves as measured.</Note>
    </ChartFrame>
  )
}

/** The check, per site: the exponent of HIPS against FTIR EC as measured and under the rule. Proportional is 1. */
function SiteCheck({ net, ceil, rule, focus }: { net: Network; ceil: Ceil; rule: Rule; focus: Focus }) {
  const outcomes = useMemo(() => siteOutcomes(net.filters, { rule: rule.rule, ceiling: ceil.ceiling, cut: rule.cut }), [net.filters, rule.rule, rule.cut, ceil.ceiling])
  const sites = useMemo(() => outcomes.filter((s) => Number.isFinite(s.b)), [outcomes])
  const points = useMemo<ScatterPoint[]>(() => sites.flatMap((s, i) => {
    const tip = [siteReading(s.code)?.site ?? s.code, `as measured b ${fmt(s.b, 2)}`, s.touched ? `under this rule ${fmt(s.bAfter, 2)}` : 'left as measured under this rule',
      `every filter by its own factor ${fmt(s.bOwn, 2)}`, `${s.tested} filters tested · ${siteReading(s.code)?.reading ?? ''}`, SITE_TIP]
    return [
      { key: `${s.code}-b`, x: s.b, y: i + 0.5, color: MEASURED, r: 4.5, opacity: 1, tip, ...ring(focus.site, s.code) },
      ...(moves(s.b, s.bAfter) ? [{ key: `${s.code}-a`, x: s.bAfter, y: i + 0.5, color: LINEAR, r: 4.5, opacity: 1, tip, ...ring(focus.site, s.code) }] : []),
    ]
  }), [sites, focus.site])
  const DOMAIN: [number, number] = [0.3, 1.7]
  const clamp = (v: number) => Math.min(DOMAIN[1], Math.max(DOMAIN[0], v))
  const under = ({ x, y }: ScatterScales) => <g pointerEvents="none">
    {sites.map((s, i) => <line key={s.code} x1={x(clamp(s.b))} x2={x(clamp(s.bAfter))} y1={y(i + 0.5)} y2={y(i + 0.5)} stroke={LINEAR} strokeWidth={2} opacity={0.45} />)}
  </g>
  const near = (b: number) => Math.abs(b - 1) <= 0.1
  const closer = sites.filter((s) => Math.abs(s.bAfter - 1) < Math.abs(s.b - 1) - 0.02), further = sites.filter((s) => Math.abs(s.bAfter - 1) > Math.abs(s.b - 1) + 0.02)
  const names = (list: typeof sites) => list.map((s) => short(s.code)).join(', ') || 'none'
  const pinned = sites.filter((s) => Math.max(s.b, s.bAfter) > DOMAIN[1])
  return (
    <ChartFrame id="hn-check" title="Does the bend go — a site's exponent moves towards 1 only where its filters get their own factor; one factor per site leaves it where it was"
      subtitle={`Exponent b in HIPS ∝ (FTIR EC)^b on each site's own filters (PM2.5 as the instrument), darkest site first: as measured, and with ${ruleText(rule)} corrected. Proportional is 1. The ${sites.length} sites with at least ${MIN_TEST} filters to test.`}
      provenance={SOURCE}>
      <ScatterPlot points={points} yBands={sites.map((s) => ({ key: s.code, label: `${short(s.code)} · ${pct(s.underRead)}`, color: focus.site === s.code ? INK.accent : s.touched ? INK.text : INK.muted }))} bandLabelSize={11}
        xDomain={DOMAIN} nice={false} outside="pin" height={sites.length * 19} margin={{ top: 14, right: 24, bottom: 46, left: 150 }}
        xLabel="Exponent b on the site's own filters" xFormat={(v) => fmt(+v, 1)} density="sparse" note={false} refX={[1]} under={under} onPick={siteClick(focus)} />
      <Key items={[{ label: 'as measured (alone where the rule leaves the exponent unchanged)', color: MEASURED, kind: 'dot' }, { label: `linearised: ${ruleText(rule)}`, color: LINEAR, kind: 'dot' }]} />
      <Readouts items={[
        { label: 'Within 0.1 of proportional', value: `${sites.filter((s) => near(s.b)).length} → ${sites.filter((s) => near(s.bAfter)).length}`, sub: `of ${sites.length} sites, before → after` },
        { label: 'Moved towards 1', value: String(closer.length), color: LINEAR, sub: names(closer) },
        { label: 'Pushed away from 1', value: String(further.length), color: MEASURED, sub: names(further) },
      ]} />
      <Note>Two limits on reading this as proof. The reference is FTIR EC, not thermal EC, so the check cannot tell a bend in HIPS from one in FTIR EC. And "where seen" picks its sites on this same exponent, so that it pushes no site away from 1 is partly how it was chosen.{pinned.length ? ` Pinned to the right edge: ${names(pinned)}.` : ''}</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 5. the rules side by side
function RuleTable({ net, ceil, rule }: { net: Network; ceil: Ceil; rule: Rule }) {
  const rows = useMemo(() => RULE_KEYS.map((key) => ({ key, ...scoreRule(net.filters, { rule: key, ceiling: ceil.ceiling, cut: rule.cut }) })), [net.filters, ceil.ceiling, rule.cut])
  const n = rows[0]?.filters ?? 0
  const leaves: Record<HipsRule, ReactNode> = {
    filter: 'Nothing off its own factor. Light sites whose exponent is already at or above 1 are nudged further up',
    expected: 'Dark filters at light sites stay as measured; corrected and measured sites share one axis',
    seen: 'The same, and the sites that are dark but untestable or do not show it (Kanpur, Ilorin, Jaipur, Abu Dhabi …) stay as measured',
    cut: `A step of × ${fmt(stepAt(rule.cut, ceil.ceiling), 2)} at the cut-off: the contrast between light and dark filters is overstated`,
    site: 'The level moves, the bend does not: every filter of a site is scaled alike, so the exponent is unchanged',
  }
  const columns: Column<(typeof rows)[number]>[] = [
    { key: 'rule', label: 'Rule', value: (r) => (r.key === 'cut' ? `Filters past a ${pct(rule.cut)} cut-off` : ruleLabel(r.key)) },
    { key: 'by', label: 'Decided per', value: (r) => HIPS_RULES.find((x) => x.key === r.key)!.by },
    { key: 'corrected', label: 'Filters corrected', value: (r) => `${r.corrected.toLocaleString()} (${pct(r.corrected / n)})`, num: true },
    { key: 'off', label: 'Over a tenth from own factor', value: (r) => `${r.offTenth.toLocaleString()} (${pct(r.offTenth / n)})`, num: true },
    { key: 'closer', label: 'Sites moved towards b = 1', value: (r) => `${r.closer} of ${r.tested}`, num: true },
    { key: 'further', label: 'Pushed away', value: (r) => String(r.further), num: true },
    { key: 'near', label: 'Within 0.1 of 1, before → after', value: (r) => `${r.nearBefore} → ${r.nearAfter}`, num: true },
    { key: 'leaves', label: 'What it leaves', value: (r) => leaves[r.key] },
  ]
  return (
    <ChartFrame id="hn-rules" title="The five rules on the same filters — per filter is the only one that is right for every filter; a site rule is a choice about which sites to trust the curve at"
      subtitle={`${n.toLocaleString()} SPARTAN filters at the chosen ceiling (${fmt(ceil.ceiling, 2)}); the cut-off row follows the cut-off in Page options. The highlighted row is the rule the charts above are drawn with.`}
      provenance={SOURCE}>
      <DataTable rows={rows} columns={columns} rowKey={(r) => r.key} highlight={(r) => r.key === rule.rule} />
      <Note>"Moved towards" and "pushed away" count the sites with an exponent of their own whose distance from 1 changes by more than 0.02. The exponent is against FTIR EC; it tests the shape of the correction, not its level.</Note>
    </ChartFrame>
  )
}

function Learned() {
  return (
    <ExplainOnly>
      <section className="panel">
        <header className="frame-head"><div><h2>What the page shows, in order</h2></div></header>
        <p><strong>The effect is a property of the filter.</strong> How much HIPS under-reads depends on how dark that one filter is: its absorption times the air drawn through it. IMPROVE shows it against thermal EC; no SPARTAN site can, because none has thermal EC.</p>
        <p><strong>It is isolated twice.</strong> By the curve, which says where the bend is expected from Fabs and volume alone, and by each site's own filters, which show it or not without assuming the curve. The two agree at the sites that can be tested, with exceptions that are listed, not hidden.</p>
        <p><strong>Per filter needs no decision about who is affected.</strong> A light filter's factor is close to 1, so correcting every filter already leaves the clean sites nearly as they were.</p>
        <p><strong>A site rule is a statement of trust, not a different correction.</strong> "Where expected" and "where seen" apply the same per-filter factor at fewer sites. They avoid pushing sites that do not show the bend, at the price of mixing corrected and measured sites and leaving dark filters at light sites untouched.</p>
        <p><strong>Two rules that look reasonable fail.</strong> A cut-off on filters puts a step in the response and overstates the contrast. One factor per site moves the level and leaves the bend exactly where it was, because scaling every filter alike cannot change how HIPS rises with EC.</p>
      </section>
    </ExplainOnly>
  )
}
