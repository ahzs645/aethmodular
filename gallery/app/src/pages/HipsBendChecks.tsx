import { useMemo } from 'react'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { Segmented } from '@/components/Controls'
import { LoadState } from '@/components/ExportState'
import { Key } from '@/components/Key'
import { PageToc } from '@/components/PageToc'
import { Marker } from '@/components/RefLines'
import { ScatterPlot, type ScatterPoint, type ScatterScales } from '@/components/ScatterPlot'
import { useExport } from '@/lib/data'
import { fmt } from '@/lib/format'
import { useFirstSampleSite } from '@/lib/gallerySamples'
import { correction, DEPOSIT_AREA_CM2, fitAgainstHips, macForSlopeOne, saturate, type LinFilter } from '@/lib/hipsLinearise'
import { useExclusionSplit } from '@/lib/markedRows'
import { SitePicker, usePageSite } from '@/lib/siteFocus'
import { median } from '@/lib/stats'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import {
  AgainstHips, Bend, CeilingOptions, COMMAND, pathOf, PROVENANCE, range, SIDE, useCeiling, useHipsSiteFilters, WORKFLOW,
  type Ceil, type Linearise,
} from '@/pages/HipsLinearisePage'
import { DataTable, LINEAR, MEASURED, Readouts, type Column } from '@/pages/hipsLineariseParts'
import { CeilingDecided, CeilingLeverage, CeilingSweep, GroupingOptions, useGroupingChoice } from '@/pages/HipsCeilingChoice'
import { AddisAtCeiling, CeilingMethods } from '@/pages/HipsCeilingDefence'
import {
  DENSITY_COLOR, DensityCells, fitLabel, IMPROVE_DENSITY_COMMAND, useImproveDensity, type SingleFilterFit,
} from '@/pages/hipsDensity'

/**
 * The checks Ann asked for on 19 Nov 2026, before the 24 Nov meeting: the IMPROVE bend on every filter rather than
 * the group medians alone, the curve fitted to single filters, the same curve in the units HIPS is reported in
 * (Fabs against EC), and the Addis and Delhi crossplots with only the filters below an optical-depth cut, with the MAC
 * that brings each fit to slope 1. Account: docs/hips-bend-checks-2026-11-20.md.
 */

const Part = ({ children }: { children: string }) => <h2 className="meeting-section-title">{children}</h2>
const ADDIS = 'ETAD'
const DELHI = 'INDH'
/** a 20 % expected under-read at the central ceiling: the cut used in the meeting */
export const DEFAULT_CUT = 0.45
/** HIPS Fabs (Mm⁻¹) per unit optical depth at a sampled volume (m³): τ = Fabs × V ÷ (100 × A) */
const fabsPerTau = (volume: number) => (100 * DEPOSIT_AREA_CM2) / volume
const URBAN_NAMES: Record<string, string> = {
  ATLA1: 'Atlanta', BIRM1: 'Birmingham', FRES1: 'Fresno', PHOE1: 'Phoenix', PITT1: 'Pittsburgh', PUSO1: 'Puget Sound (Seattle)',
}

export function HipsChecksPage() {
  const { data, error } = useExport<Linearise>('hips_linearise/linearise.json')
  const ceil = useCeiling(data)
  const grouping = useGroupingChoice()
  if (error || !data) return (
    <ChartFrame id="hc-missing" title="HIPS bend checks" provenance={WORKFLOW}>
      <LoadState what="the linearisation export" error={error} command={COMMAND} />
    </ChartFrame>
  )
  return (
    <PageToc>
      <CeilingOptions ceil={ceil} />
      <GroupingOptions choice={grouping} />
      <Part>1. The bend on every IMPROVE filter, not only the group medians</Part>
      <Bend data={data} ceil={ceil} improveDefault="filters" />
      <BendFits data={data} />
      <Part>2. How the ceiling is decided, and how much the grouping matters</Part>
      <CeilingDecided data={data} ceil={ceil} choice={grouping} />
      <CeilingSweep data={data} choice={grouping} />
      <CeilingLeverage choice={grouping} />
      <Part>3. Is the ceiling defensible: every other way of estimating it, and what each does to Addis</Part>
      <CeilingMethods data={data} />
      <AddisAtCeiling data={data} ceil={ceil} />
      <Part>4. The same curve in the units HIPS reports: Fabs against EC</Part>
      <BendInFabs data={data} ceil={ceil} />
      <Part>5. Addis and Delhi with only the filters below an optical-depth cut</Part>
      <AgainstHips data={data} ceil={ceil} cutDefault={String(DEFAULT_CUT)} />
      <CutTable data={data} ceil={ceil} />
      <Part>6. From the 19 Nov meeting</Part>
      <MeetingNotes data={data} />
    </PageToc>
  )
}

// ------------------------------------------------------------------ the curve fitted to single filters
const POPULATIONS = [
  { key: 'all', name: 'all sites', label: 'All sites' },
  { key: 'ocec', name: 'sites with median OC/EC < 4', label: 'OC/EC < 4' },
  { key: 'urban', name: 'six urban IMPROVE sites', label: 'Six urban' },
] as const
type PopKey = (typeof POPULATIONS)[number]['key']

interface FitRow { label: string; samples: number; sites: number; tau_inf: number; lo: number; hi: number; primary?: boolean }

export function BendFits({ data }: { data: Linearise }) {
  const [pop, setPop] = useSearchParam<PopKey>('hc.pop', 'all', POPULATIONS.map((p) => p.key))
  const { data: density, error } = useImproveDensity()
  const name = POPULATIONS.find((p) => p.key === pop)!.name
  const addisTau = useMemo(() => median(data.filters.filter((f) => f.ftir != null).map((f) => f.tau)), [data.filters])
  const rows = useMemo<FitRow[]>(() => {
    const group = data.improve_ceilings.find((c) => c.population === name)
    const own = (density?.fits ?? []).filter((f: SingleFilterFit) => f.population === name)
    return [
      ...(group ? [{ label: 'Group medians (the curve the gallery uses)', samples: group.samples, sites: group.sites, tau_inf: group.tau_inf,
        lo: group.tau_inf_lo, hi: group.tau_inf_hi, primary: true }] : []),
      ...own.map((f) => ({ label: `Single filters, ${fitLabel(f)}`, samples: f.samples, sites: f.sites, tau_inf: f.tau_inf, lo: f.tau_inf_lo, hi: f.tau_inf_hi })),
    ]
  }, [data.improve_ceilings, density, name])
  const columns: Column<FitRow>[] = [
    { key: 'label', label: 'Curve fitted to', value: (r) => r.label },
    { key: 'n', label: 'Filters', value: (r) => r.samples.toLocaleString(), num: true },
    { key: 'sites', label: 'Sites', value: (r) => r.sites, num: true },
    { key: 'tau', label: 'τ∞', value: (r) => fmt(r.tau_inf, 2), num: true },
    { key: 'ci', label: '90 % (sites resampled)', value: (r) => `${fmt(r.lo, 2)} – ${fmt(r.hi, 2)}`, num: true },
    { key: 'cut', label: 'Under-read at τ 0.45', value: (r) => `${Math.round(100 * DEFAULT_CUT / r.tau_inf)} %`, num: true },
    { key: 'addis', label: `× on the median Addis filter (τ ${fmt(addisTau, 2)})`, value: (r) => fmt(correction(addisTau, r.tau_inf), 2), num: true },
  ]
  const all = (density?.fits ?? []).filter((f) => f.population === name)
  const span = all.length ? [Math.min(...all.map((f) => f.tau_inf)), Math.max(...all.map((f) => f.tau_inf))] : null
  return (
    <ChartFrame id="hc-fits" title="The bend does not come from the grouping — the same curve through single IMPROVE filters, every way it can be fitted"
      subtitle="τ∞ is the ceiling of τ = aL / (1 + aL/τ∞). The group-median fit is the one the gallery linearises with. The single-filter fits use every filter from a lowest EC loading, with residuals in τ (weights the dark filters) or in log τ (weights every filter alike, as the group fit does). Intervals resample whole sites, 200 times."
      provenance={<><code>hips_linearise/improve_density.json</code> (<code>gallery/data/export_hips_improve_density.py</code>); group fits from {WORKFLOW}.</>}
      controls={<Segmented label="IMPROVE sites" value={pop} options={POPULATIONS.map((p) => p.key)} onChange={setPop}
        optionLabel={(k) => POPULATIONS.find((p) => p.key === k)?.label ?? k} />}>
      {!density ? <LoadState what="the single-filter fits" error={error} command={IMPROVE_DENSITY_COMMAND} /> : <>
        <DataTable rows={rows} columns={columns} rowKey={(r) => r.label} highlight={(r) => !!r.primary} />
        <Note explain>Every fit bends: none is close to a straight line, so the bend is in the filters, not in how they are grouped.
          How far it bends is less settled. Over single filters, τ∞ runs from {fmt(span?.[0], 2)} to {fmt(span?.[1], 2)} depending on the
          residuals and the lowest loading, and most are above the group fit, which means a smaller correction (the last column).
          Two things pull single-filter fits about. A filter's own TOR EC is on the x axis, so TOR's error bends the fitted curve over
          a little more than the truth; and starting from a loading selects on that same error. The group fit avoids both by grouping
          on the previous sample's loading at the same site, which is why it stays the one used.</Note>
      </>}
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ the curve in Fabs against EC
export function BendInFabs({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const { data: density, error } = useImproveDensity()
  const { bySite, sites } = useHipsSiteFilters(data)
  const sitePick = usePageSite(sites, { prefer: useFirstSampleSite() ?? ADDIS })
  const code = sitePick.code || ADDIS
  const siteName = sites.find((s) => s.code === code)?.name ?? 'Addis Ababa'
  const siteFilters = useMemo(() => (code === ADDIS ? data.filters : bySite.get(code)?.filters ?? []), [code, data.filters, bySite])
  const { counted } = useExclusionSplit(code === ADDIS ? [] : siteFilters, (f) => f.id)
  const filters = (code === ADDIS ? siteFilters : counted).filter((f) => f.ftir != null)
  const c = ceil.ceiling, a = data.a_per_ug
  const u = a * 100 * DEPOSIT_AREA_CM2                // Fabs ÷ EC on light filters, m²/g
  const vImprove = density?.median_volume_m3 ?? NaN
  const vSite = median(filters.map((f) => f.volume_m3))
  /** Fabs (Mm⁻¹) the curve gives at an EC concentration (µg/m³) for a filter sampled at volume V */
  const curve = (conc: number, volume: number) => fabsPerTau(volume) * saturate(a * conc * volume, c)
  const improvePoints = useMemo<ScatterPoint[]>(() => (Number.isFinite(vImprove) ? data.strata.map((s, i) => ({
    key: `g${i}`, x: s.loading_ug / vImprove, y: s.tau * fabsPerTau(vImprove), color: INK.muted, r: 3 + Math.sqrt(s.n) / 40,
    tip: ['IMPROVE group median, at the median volume', `TOR EC ${fmt(s.loading_ug / vImprove, 2)} µg/m³`, `Fabs ${fmt(s.tau * fabsPerTau(vImprove), 1)} Mm⁻¹`, `${s.n.toLocaleString()} filters`],
  })) : []), [data.strata, vImprove])
  const sitePoints = useMemo<ScatterPoint[]>(() => filters.map((f: LinFilter) => ({
    id: f.id, key: f.id, x: f.ftir!, y: f.hips_bc * data.mac, color: MEASURED, r: 3, opacity: 0.6,
    tip: [f.id, `FTIR EC ${fmt(f.ftir, 2)} µg/m³`, `HIPS Fabs ${fmt(f.hips_bc * data.mac, 1)} Mm⁻¹`, `τ ${fmt(f.tau, 2)} · ${fmt(f.volume_m3, 2)} m³`],
  })), [filters, data.mac])
  const lines = (volume: number, top: number, other?: number) => ({ x, y, iw, ih }: ScatterScales) => {
    const xs = range(0, top, top / 200)
    return <g pointerEvents="none">
      <clipPath id={`hc-fabs-${volume}`}><rect width={iw} height={ih} /></clipPath>
      <g clipPath={`url(#hc-fabs-${volume})`}>
        {density && other == null && <DensityCells grid={density.concentration} x={x} y={y} />}
        <path d={pathOf([[x(0), y(0)], [x(top), y(u * top)]])} fill="none" stroke={LINEAR} strokeWidth={2} strokeDasharray="6 4" />
        <path d={pathOf(xs.map((v) => [x(v), y(curve(v, volume))]))} fill="none" stroke={MEASURED} strokeWidth={2} />
        {other != null && Number.isFinite(other) && <path d={pathOf(xs.map((v) => [x(v), y(curve(v, other))]))} fill="none" stroke={INK.muted} strokeWidth={1.5} strokeDasharray="2 3" />}
        <Marker orient="h" at={y(fabsPerTau(volume) * c)} from={0} to={iw} color={MEASURED} label={`ceiling ${fmt(fabsPerTau(volume) * c, 0)} Mm⁻¹ at ${fmt(volume, 1)} m³`} />
      </g>
    </g>
  }
  const siteTop = Math.max(1, ...filters.map((f) => f.ftir!)) * 1.05
  const siteFabsTop = Math.max(1, ...filters.map((f) => f.hips_bc * data.mac)) * 1.05
  const bend = (volume: number) => 0.2 * c * fabsPerTau(volume)
  return (
    <ChartFrame id="hc-fabs" title="In Fabs the bend sits at a filter loading, not a concentration — IMPROVE bends near 5 Mm⁻¹, a SPARTAN site at its own, far higher Fabs"
      subtitle={`HIPS Fabs (Mm⁻¹) against EC (µg/m³): IMPROVE against TOR EC, every filter shaded (log count) with its group medians at the median volume; ${siteName.split(',')[0]} against FTIR EC, one dot per filter. Fabs = τ × 100 × ${DEPOSIT_AREA_CM2} cm² ÷ V, so the same curve in τ becomes a different curve in Fabs at each sampled volume: drawn in orange at each panel's median volume, with the dashed blue proportional response (${fmt(u, 1)} m²/g, light filters). On the right, the dotted grey line is IMPROVE's curve at IMPROVE's volume.`}
      provenance={<><code>hips_linearise/improve_density.json</code>; {PROVENANCE}</>}
      controls={<SitePicker as="select" {...sitePick.picker} optionLabel={(v) => { const s = sites.find((x) => x.code === v); return s ? `${s.name} (${s.n})` : v }}
        title="The gallery's page site (psite): the other one-site charts on this page follow it" />}>
      {!density ? <LoadState what="every IMPROVE filter" error={error} command={IMPROVE_DENSITY_COMMAND} /> : <>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 24, justifyContent: 'center' }}>
          <div><ScatterPlot title={`IMPROVE, ${density.samples.toLocaleString()} filters`} points={improvePoints} xLabel="TOR EC (µg/m³)" yLabel="HIPS Fabs (Mm⁻¹)"
            xDomain={[0, 4]} yDomain={[0, 40]} nice={false} maxSide={SIDE * 0.8} outside="clip" note={false} under={lines(vImprove, 4)} /></div>
          <div>{sitePoints.length ? <ScatterPlot title={`${siteName.split(',')[0]}, ${sitePoints.length} filters`} points={sitePoints} xLabel="FTIR EC (µg/m³)" yLabel="HIPS Fabs (Mm⁻¹)"
            xDomain={[0, siteTop]} yDomain={[0, Math.max(siteFabsTop, fabsPerTau(vSite) * c * 1.05)]} maxSide={SIDE * 0.8} outside="pin" note={false}
            under={lines(vSite, siteTop, vImprove)} /> : <Empty>No filters with FTIR EC and HIPS at this site.</Empty>}</div>
        </div>
        <Key items={[
          { label: 'Every IMPROVE filter (count, log)', color: DENSITY_COLOR, kind: 'square' },
          { label: 'IMPROVE group medians, at the median volume', color: INK.muted, kind: 'dot' },
          { label: `${siteName.split(',')[0]} filters (FTIR EC)`, color: MEASURED, kind: 'dot' },
          { label: 'Proportional response', color: LINEAR, kind: 'dash' },
          { label: 'IMPROVE curve at the panel’s volume', color: MEASURED, kind: 'line' },
          { label: 'IMPROVE curve at IMPROVE’s volume', color: INK.muted, kind: 'dash' },
        ]} />
        <Readouts items={[
          { label: 'IMPROVE median volume', value: `${fmt(vImprove, 1)} m³`, sub: `20 % under-read at ${fmt(bend(vImprove), 1)} Mm⁻¹` },
          { label: `${siteName.split(',')[0]} median volume`, value: `${fmt(vSite, 2)} m³`, sub: `20 % under-read at ${fmt(bend(vSite), 0)} Mm⁻¹` },
          { label: 'Median filter here', value: `${fmt(median(filters.map((f) => f.hips_bc * data.mac)), 0)} Mm⁻¹`, sub: `τ ${fmt(median(filters.map((f) => f.tau)), 2)}` },
        ]} />
        <Note explain>Fabs carries HIPS's own corrections (scattering, the filter's reflectance), so it is the right quantity to report;
          those corrections do not undo this bend, which IMPROVE shows in Fabs as well as in τ. What changes in Fabs is where the bend
          starts: it depends on how much EC is on the filter, so a site that draws a tenth of IMPROVE's air reaches it at about ten times
          the Fabs. On a single concentration axis IMPROVE and a SPARTAN site cannot share one curve; that is why the bend is drawn
          against optical depth.</Note>
      </>}
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ every site, all filters against those below the cut
interface CutRow {
  code: string; name: string; n: number; below: number; past: number
  measured: ReturnType<typeof fitAgainstHips>; macMeasured: number
  belowFit: ReturnType<typeof fitAgainstHips>; macBelow: number
  linear: ReturnType<typeof fitAgainstHips>; macLinear: number
}
const MIN_BELOW = 10

export function CutTable({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const [cutText] = useSearchParam<string>('hl.hips.cut', String(DEFAULT_CUT))
  const cut = Number(cutText) > 0 ? Number(cutText) : DEFAULT_CUT
  const c = ceil.ceiling
  const { bySite, sites, loaded } = useHipsSiteFilters(data)
  const others = useMemo(() => [...bySite].filter(([code]) => code !== ADDIS).flatMap(([, s]) => s.filters), [bySite])
  const { counted } = useExclusionSplit(others, (f) => f.id)
  const rows = useMemo<CutRow[]>(() => {
    const keep = new Set(counted.map((f) => f.id))
    return sites.flatMap((s) => {
      const addis = s.code === ADDIS
      const own = (addis ? data.filters : (bySite.get(s.code)?.filters ?? []).filter((f) => keep.has(f.id))).filter((f) => f.ftir != null)
      if (own.length < 20) return []
      const lambda = addis ? data.lambda : 1
      const below = own.filter((f) => f.tau < cut)
      const fitBelow = below.length >= MIN_BELOW ? fitAgainstHips(below, lambda, null) : null
      return [{
        code: s.code, name: s.name, n: own.length, below: below.length, past: 1 - below.length / own.length,
        measured: fitAgainstHips(own, lambda, null), macMeasured: macForSlopeOne(own, lambda, null, data.mac),
        belowFit: fitBelow, macBelow: fitBelow ? macForSlopeOne(below, lambda, null, data.mac) : NaN,
        linear: fitAgainstHips(own, lambda, c), macLinear: macForSlopeOne(own, lambda, c, data.mac),
      }]
    }).sort((a, b) => b.past - a.past)
  }, [sites, bySite, counted, data, cut, c])
  const line = (f: CutRow['measured']) => (f ? `${fmt(f.slope, 2)}x ${f.intercept < 0 ? '−' : '+'} ${fmt(Math.abs(f.intercept), 2)}` : '—')
  const mac = (v: number) => (Number.isFinite(v) ? fmt(v, 1) : '—')
  const columns: Column<CutRow>[] = [
    { key: 'site', label: 'Site', value: (r) => r.name },
    { key: 'n', label: 'Filters', value: (r) => r.n, num: true },
    { key: 'past', label: `Past τ ${fmt(cut, 2)}`, value: (r) => `${Math.round(100 * r.past)} %`, num: true },
    { key: 'm', label: 'All, as measured', value: (r) => line(r.measured), num: true },
    { key: 'mm', label: 'MAC for 1:1', value: (r) => mac(r.macMeasured), num: true },
    { key: 'b', label: `Below τ ${fmt(cut, 2)}, as measured`, value: (r) => (r.belowFit ? `${line(r.belowFit)} (${r.below})` : `— (${r.below})`), num: true },
    { key: 'mb', label: 'MAC for 1:1', value: (r) => mac(r.macBelow), num: true },
    { key: 'l', label: 'All, linearised', value: (r) => line(r.linear), num: true },
    { key: 'ml', label: 'MAC for 1:1', value: (r) => mac(r.macLinear), num: true },
  ]
  const addis = rows.find((r) => r.code === ADDIS), delhi = rows.find((r) => r.code === DELHI)
  return (
    <ChartFrame id="hc-cut" title={`Every SPARTAN site, all filters against only those below τ ${fmt(cut, 2)} — the light filters alone give an intercept near zero without any correction`}
      subtitle={`Deming fits of FTIR EC (y) on HIPS BC = Fabs ÷ ${data.mac} (x), as on the crossplot above (λ ${data.lambda} at Addis, 1 elsewhere). "MAC for 1:1" is the MAC at which the same fit has slope 1; the intercept does not depend on it. A fit below the cut needs ${MIN_BELOW} filters. The cut follows the crossplot's "τ below" (0.45 when it shows all filters); linearised fits use the ceiling in Page options. Sites with 20 or more filters, darkest first.`}
      provenance={PROVENANCE}>
      {!loaded ? <LoadState what="the SPARTAN filters" /> : <>
        <DataTable rows={rows} columns={columns} rowKey={(r) => r.code} highlight={(r) => r.code === ADDIS || r.code === DELHI} />
        <Note>{delhi && <>Delhi: {Math.round(100 * delhi.past)} % of its {delhi.n} filters are past τ {fmt(cut, 2)} (not about a tenth, as said on 19 Nov); the {delhi.below} below it give {line(delhi.belowFit)}, slope 1 at MAC {mac(delhi.macBelow)}. </>}
          {addis && <>Addis: {addis.below === 0 ? `no filter is below τ ${fmt(cut, 2)}, so the cut cannot be tested there` : `${addis.below} filters below τ ${fmt(cut, 2)}`}; all {addis.n} linearised give {line(addis.linear)}, slope 1 at MAC {mac(addis.macLinear)}.</>}</Note>
      </>}
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ what the meeting asked, answered
export function MeetingNotes({ data }: { data: Linearise }) {
  const { data: density } = useImproveDensity()
  const primary = data.improve_ceilings[0]
  return (
    <section className="panel" id="hc-notes">
      <header className="frame-head"><div><h2>Answers and corrections to the 19 Nov meeting</h2></div></header>
      <ul>
        <li><strong>The equation.</strong> τ = aL / (1 + aL/τ∞), L the EC on the filter (µg), a = {fmt(data.a_per_ug, 4)} µg⁻¹ ({fmt(primary.unsaturated_fabs_over_ec, 1)} m²/g on light filters),
          τ∞ = {fmt(primary.tau_inf, 2)} [{fmt(primary.tau_inf_lo, 2)}, {fmt(primary.tau_inf_hi, 2)}]. Under-read τ/τ∞; factor 1/(1 − τ/τ∞). It is a saturating curve, not the quadratic said in the meeting.</li>
        <li><strong>Delhi.</strong> About half its filters are past τ 0.45, not about ten percent; the table above has every site.</li>
        <li><strong>The six urban IMPROVE sites</strong> (the sixth that was missing is Birmingham): {(density?.urban_sites ?? Object.keys(URBAN_NAMES)).map((s) => `${URBAN_NAMES[s] ?? s} (${s})`).join(', ')}. They are collocated with CSN; a citation for the list is still to find.</li>
        <li><strong>Warren's trusted range.</strong> The curve uses IMPROVE {density ? `${density.first} to ${density.last}` : '2005–2025'}; White et al. (2016) treat HIPS as comparable from 2003, so no earlier filter is in it.
          Whether White et al. (2016, 2025) already describe or exclude the heaviest filters is open: read both before writing that HIPS reads low.</li>
        <li><strong>Words.</strong> Filters past the cut have compromised Fabs; they are not lost, and their FTIR EC, OC and mass stand. HIPS is the check on FTIR EC (MAIA reports FTIR EC only).</li>
        <li><strong>Per filter, not per site.</strong> What decides the correction is the filter's τ (or Fabs at its volume), not its site; the 19 Nov story page compares the rules, and "Apply to" can now linearise only filters past a τ.</li>
      </ul>
    </section>
  )
}
