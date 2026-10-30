/**
 * decision_workbook_20261022 §6 [23]: the flag refitted on all of IMPROVE and applied to every SPARTAN PM2.5 filter.
 * Which sites and site-seasons it marks, how FTIR EC compares with HIPS BC there at the reader's MAC, whether the
 * spectrum-only flag points where the two disagree, and one site's filters on the FTIR–HIPS plane.
 *
 * With the per-filter table (section6_network_filters) every number is rebuilt here (decision/stats.ts) and follows
 * the MAC and Filters → excluded (lib/markedRows useExclusionSplit, by filter id: the export's list and hand exclusions;
 * under Mark the site scatter draws them as ×, the summaries say how many they leave out). Without it the notebook's site tables are drawn, re-expressed exactly at the reader's
 * MAC (ratios and Deming slopes × MAC/10 with λ ∝ MAC², intercepts unchanged), and the site scatter is not shown.
 */
import { useMemo } from 'react'
import { ChartFrame, Note } from '@/components/ChartFrame'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { Figure } from '@/components/Figure'
import { Key } from '@/components/Key'
import { useChartOptions } from '@/lib/chartOptions'
import { ExcludedNote, FixedSetNote, useCountedFilter, useExclusionSplit } from '@/lib/markedRows'
import { f2 } from '@/lib/format'
import { demingLambda } from '@/lib/macMath'
import { useMac } from '@/lib/mac'
import { num, type Row } from '@/lib/table'
import type { SpectralExport } from '../../../research/pages/spectralCommon'
import { provenance, rows } from '../../common'
import { crossCheck, flagTable, type FlagFilter, type FlagGroup } from './stats'

const NO_FILTERS: FlagFilter[] = []
const ADDIS = '#e8772e'
const OTHER = '#4e79a7'
const FLAGGED = '#d1495b'
const NOT_FLAGGED = '#4e79a7'
const FLAG_HIPS = 'spectrum + HIPS', FLAG_SPECTRUM = 'spectrum only'
const FLAG_COLORS = { [FLAG_HIPS]: '#d1495b', [FLAG_SPECTRUM]: '#4e79a7' }
const PANEL_COLORS = { 'published FTIR EC': '#4e79a7', 'this workbook (spline)': '#f28e2b' }

/** the export's per-filter table, or null when it was written without the Drive */
export function useFlagFilters(data: SpectralExport): FlagFilter[] | null {
  return useMemo(() => {
    const net = rows(data, 'network_filters')
    if (!net.length) return null
    const seasons = rows(data, 'seasons').map((r) => String(r.season ?? ''))
    return net.map((r) => ({ id: String(r.filter_id), code: String(r.code), season: seasons[num(r.season) ?? -1] ?? '',
      flagSpectrum: num(r.flag_spectrum) ?? NaN, flagHips: num(r.flag_hips) ?? NaN, outside: !!r.outside,
      ecModel: num(r.ec_model), ecPublished: num(r.ec_published), fabs: num(r.fabs) }))
  }, [data])
}

/** the notebook's saved site table as FlagGroups at the reader's MAC (exact: see the file comment) */
function savedGroups(table: Row[], m: ReturnType<typeof useMac>): FlagGroup[] {
  return table.map((r) => ({
    code: String(r.code ?? 'ETAD'), season: String(r.season ?? ''), n: num(r.n) ?? 0,
    meanFlagSpectrum: num(r['mean flag (spectrum)']) ?? NaN, meanFlagHips: num(r['mean flag (+ HIPS)']) ?? NaN,
    flaggedShare: num(r['flagged share (+ HIPS)']) ?? NaN, flaggedShareSpectrum: null, outsideShare: num(r['outside IMPROVE range']) ?? NaN,
    modelRatio: m.slope(num(r['this model ÷ HIPS (median)'])), publishedRatio: m.slope(num(r['published ÷ HIPS (median)'])), pairs: NaN,
    slope: m.slope(num(r['published vs HIPS slope'])), intercept: num(r.intercept), interceptRatio: num(r['intercept ÷ mean']),
  }))
}

export function FlagOnSpartan({ data }: { data: SpectralExport }) {
  const m = useMac()
  const all = useFlagFilters(data)
  const lambda10 = num(rows(data, 'settings')[0]?.lambda_addis) ?? 3.34
  const names = useMemo(() => new Map(rows(data, 'sites').map((r) => [String(r.code), String(r.site ?? r.code).split(',')[0]])), [data])
  const nameOf = (code: string) => names.get(code) ?? code
  // the counted filters by the one rule (lib/markedRows useCountedFilter: the export's exclusions unless Include, hand
  // exclusions, keep=); the split gives the × under Mark
  const split = useExclusionSplit(all ?? NO_FILTERS, (f) => f.id)
  const isCounted = useCountedFilter()
  const filters = useMemo(() => (all ? all.filter((f) => isCounted(f.id)) : null), [all, isCounted])
  const dropped = all && filters ? all.length - filters.length : 0

  const { sites, siteSeasons } = useMemo(() => filters
    ? { sites: flagTable(filters, 'site', m.mac, lambda10), siteSeasons: flagTable(filters, 'site-season', m.mac, lambda10) }
    : { sites: savedGroups(rows(data, 'sites'), m), siteSeasons: savedGroups(rows(data, 'site_seasons'), m) }, [filters, data, m, lambda10])
  const addisSeasons = useMemo(() => savedGroups(rows(data, 'addis_seasons'), m), [data, m])
  const cc = useMemo(() => crossCheck(siteSeasons), [siteSeasons])

  const live = filters != null
  const how = live
    ? `Rebuilt live from the workbook's per-filter table (${filters.length.toLocaleString()} SPARTAN filters at ${sites.length} sites${dropped ? `; ${dropped} excluded under Filters → excluded` : ''}): with every filter counted, at MAC 10 it reproduces the notebook's site tables (largest relative difference 4e-4, from the 4-decimal export).`
    : `The notebook's saved tables (${siteSeasons.length < 30 ? 'only the top 20 site-seasons were displayed; ' : ''}the per-filter table needs the Drive), re-expressed at ${m.label}.`
  const short = (g: FlagGroup) => `${nameOf(g.code)} (${g.n})`
  const label = (g: FlagGroup) => g.season ? `${nameOf(g.code)} · ${g.season} (n = ${g.n})` : `${nameOf(g.code)} (n = ${g.n})`

  // both flags by site and by site-season (the 30 most flagged), one table so the reader switches between them;
  // the share outside the IMPROVE range rides in the label (one value per group, not per flag)
  const SITE = 'Site', SEASON = 'Site-season (30 most flagged, n ≥ 10)'
  const flagLabel = (g: FlagGroup) => `${g.season ? `${nameOf(g.code)} · ${g.season}` : nameOf(g.code)} (${g.n}; ${Math.round(100 * g.outsideShare)} % outside)`
  const flagRows = (level: string, gs: FlagGroup[]): Row[] => gs.flatMap((g) => [
    { level, row: flagLabel(g), series: FLAG_HIPS, mean: g.meanFlagHips, share: g.flaggedShare },
    { level, row: flagLabel(g), series: FLAG_SPECTRUM, mean: g.meanFlagSpectrum, share: g.flaggedShareSpectrum },
  ])
  const groupRows: Row[] = [...flagRows(SITE, sites), ...flagRows(SEASON, siteSeasons.slice(0, 30))]
  // FTIR ÷ HIPS and the Deming fit by site: published EC and this workbook's model, at the reader's MAC
  const agreeRows: Row[] = sites.flatMap((g) => [
    { row: short(g), series: 'published FTIR EC', ratio: g.publishedRatio, slope: g.slope, interceptRatio: g.interceptRatio },
    { row: short(g), series: 'this workbook (spline)', ratio: g.modelRatio, slope: null, interceptRatio: null },
  ])

  const bySpectrum = [...sites].sort((a, b) => b.meanFlagSpectrum - a.meanFlagSpectrum)
  const paired = sites.filter((g) => g.modelRatio != null && g.publishedRatio != null)
  const lower = paired.filter((g) => g.modelRatio! < g.publishedRatio!).length

  return <>
    <Figure kind="intervals" id="ml-decision-flag-sites" data={groupRows} row="row" series="series" colors={FLAG_COLORS} choose={[{ column: 'level', label: 'Group by', default: SITE }]}
      panels={[{ value: 'mean', label: 'Mean flag (chance FTIR EC is impacted)', reference: 0.5 }, { value: 'share', label: 'Share of filters flagged (flag ≥ 0.5)' }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={300}
      title={`The + HIPS flag marks ${nameOf(sites[0]?.code ?? '')} first, then ${sites.slice(1, 3).map((g) => nameOf(g.code)).join(' and ')}; spectrum only puts ${bySpectrum.slice(0, 2).map((g) => nameOf(g.code)).join(' and ')} first (§6)`}
      subtitle={`The classifier from §5, refitted on all of IMPROVE and applied to every SPARTAN PM2.5 filter: its chance that the filter's FTIR EC is "impacted" (|FTIR − TOR| > 2 µg and > 50 % of TOR, as one IMPROVE filter in five is). The + HIPS flag reads FTIR ÷ HIPS, so at SPARTAN it partly restates the disagreement it is checked against; the spectrum-only flag does not. The label's share outside the IMPROVE range: loading, predicted EC, T², Q, dust or PTFE beyond IMPROVE's 0.5–99.5 percentiles. Sorted by the + HIPS flag; the notebook reports the flagged share for the + HIPS flag only. Addis in the workbook's own evaluation set (233 filters) by season: ${addisSeasons.map((g) => `${g.season} ${f2(g.meanFlagHips)}`).join(', ')}.`}
      provenance={<>{how} {provenance(data, live ? ['network_filters'] : ['sites', 'site_seasons'])}</>} />

    <Figure kind="intervals" id="ml-decision-flag-hips" data={agreeRows} row="row" series="series" colors={PANEL_COLORS}
      panels={[{ value: 'ratio', label: `FTIR EC ÷ HIPS BC, median (${m.label})`, reference: 1 }, { value: 'slope', label: `Published vs HIPS BC: Deming slope (${m.label})`, reference: 1 },
        { value: 'interceptRatio', label: 'Deming intercept ÷ mean EC', reference: 0 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={170}
      title={`At ${lower} of ${paired.length} sites the workbook's own model reads lower against HIPS than SPARTAN's published EC (§6, ${m.label})`}
      subtitle={`The notebook's cross-check, not a verdict (HIPS BC = Fabs ÷ MAC assumes the MAC). Deming of published FTIR EC on HIPS BC with the workbook's Addis error ratio (λ = ${f2(demingLambda(m.mac, lambda10))} at ${m.label}, ∝ MAC²), fitted where a site has at least 15 filters with both; the notebook does not fit its own model's EC. Same site order as above.`}
      provenance={<>{how} {provenance(data, live ? ['network_filters'] : ['sites'])}</>} />

    <CrossCheck groups={siteSeasons} cc={cc} label={label} live={live} data={data} how={how} nOut={split.marked.length} />
    {filters ? <SiteFilters filters={filters} marked={split.marked} names={names} lambda10={lambda10} data={data} dropped={dropped} />
      : <ChartFrame id="ml-decision-flag-filters" title="One site's filters on the FTIR–HIPS plane (§6)" exportable={false}>
        <Note>Needs the workbook's per-filter table (section6_network_filters.csv on the Drive); see the list at the top of the page.</Note>
      </ChartFrame>}
  </>
}

/** The notebook's site-season scatter: does the spectrum-only flag point where FTIR and HIPS disagree? */
function CrossCheck({ groups, cc, label, live, data, how, nOut }: {
  groups: FlagGroup[]; cc: ReturnType<typeof crossCheck>; label: (g: FlagGroup) => string; live: boolean; data: SpectralExport; how: string; nOut: number
}) {
  const m = useMac()
  const o = useChartOptions('ml-decision-crosscheck', [
    { key: 'y', label: 'Disagreement', default: 'intercept', choices: [{ value: 'intercept', label: 'Deming intercept ÷ mean' }, { value: 'ratio', label: '|ln (published ÷ HIPS)|' }] },
  ])
  const byRatio = o.values.y === 'ratio'
  const points: CrossPoint[] = groups.flatMap((g) => {
    const y = byRatio ? (g.publishedRatio != null && g.publishedRatio > 0 ? Math.abs(Math.log(g.publishedRatio)) : null) : g.interceptRatio
    if (y == null || g.interceptRatio == null || !Number.isFinite(g.meanFlagSpectrum)) return []
    return [{ x: g.meanFlagSpectrum, y, color: g.code === 'ETAD' ? ADDIS : OTHER, site: g.code,
      tip: [label(g), `mean flag (spectrum only): ${f2(g.meanFlagSpectrum)}`, `Deming intercept ÷ mean: ${f2(g.interceptRatio)}`, `published ÷ HIPS (median, ${m.label}): ${f2(g.publishedRatio)}`] }]
  })
  if (!points.length) return null
  // a site-season with a near-zero mean EC puts its intercept ÷ mean far off (Sherbrooke, about −69): the notebook's
  // axis stretched to it; here the range stops at −3 and the cut points are named
  const lo = byRatio ? null : -3
  const cut = lo == null ? [] : points.filter((p) => p.y < lo)
  const shown = points.filter((p) => !cut.includes(p))
  return (
    <ChartFrame id="ml-decision-crosscheck" controls={o.controls()}
      title={`The spectrum-only flag points weakly where FTIR and HIPS disagree: ρ = ${f2(cc.intercept)} with the intercept, ${f2(cc.ratio)} with the ratio (§6)`}
      subtitle={`Each point a site-season with at least 10 filters and a Deming fit (${cc.n}${live ? '' : ' of the top 20 the notebook displayed'}). Spearman's ρ of the spectrum-only flag against the Deming intercept ÷ mean (negative: the flag is high where the intercept is negative) and against |ln (published ÷ HIPS)| at ${m.label}; the notebook printed −0.33 and +0.25 at MAC 10 over 54 site-seasons. The intercept does not move with the MAC (λ ∝ MAC²); the ratio does. Addis (ETAD) in orange.${cut.length ? ` Below the axis (in ρ, not drawn): ${cut.map((p) => `${p.tip[0]} at ${f2(p.y)}`).join('; ')}.` : ''}`}
      provenance={<>{how} {provenance(data, live ? ['network_filters'] : ['site_seasons'])}</>}>
      <Crossplot title="" points={shown} fit={false} grouping="none" sameScale={false} maxSide={420}
        xLabel="Mean flag, spectrum only (chance FTIR EC is impacted)" yLabel={byRatio ? `|ln (published FTIR ÷ HIPS BC)|, ${m.label}` : 'Published FTIR vs HIPS: Deming intercept ÷ mean'} />
      <Key items={[{ label: 'Addis (ETAD) seasons', color: ADDIS }, { label: 'other site-seasons', color: OTHER }]} />
      {/* the site and site-season summaries above and here are on the counted filters; without the per-filter table, the notebook's */}
      {live ? <ExcludedNote count={nOut} /> : <FixedSetNote what="the notebook's saved site tables (the per-filter table is not in this export)" />}
    </ChartFrame>
  )
}

/** One site's filters: FTIR EC against HIPS BC at the reader's MAC, coloured by the + HIPS flag, refitted live */
function SiteFilters({ filters, marked, names, lambda10, data, dropped }: {
  filters: FlagFilter[]; marked: FlagFilter[]; names: Map<string, string>; lambda10: number; data: SpectralExport; dropped: number
}) {
  const m = useMac()
  const codes = useMemo(() => [...new Set(filters.map((f) => f.code))].sort((a, b) => (names.get(a) ?? a).localeCompare(names.get(b) ?? b)), [filters, names])
  const o = useChartOptions('ml-decision-flag-filters', [
    { key: 'site', label: 'Site', default: 'ETAD', style: 'select', choices: codes.map((c) => ({ value: c, label: `${names.get(c) ?? c} (${c})` })) },
    { key: 'ec', label: 'FTIR EC', default: 'published', choices: [{ value: 'published', label: 'SPARTAN published' }, { value: 'model', label: 'This workbook (spline, k = 4)' }] },
  ])
  const site = String(o.values.site), model = o.values.ec === 'model'
  const toPoints = (fs: FlagFilter[]): CrossPoint[] => fs.flatMap((f) => {
    const x = m.fromFabs(f.fabs), y = model ? f.ecModel : f.ecPublished
    if (f.code !== site || x == null || x <= 0 || y == null) return []
    const flagged = f.flagHips >= 0.5
    return [{ x, y, id: f.id, site: f.code, season: f.season || undefined, color: flagged ? FLAGGED : NOT_FLAGGED, hollow: f.outside,
      tip: [f.id, f.season, `HIPS BC (${m.label}): ${f2(x)} µg/m³`, `FTIR EC: ${f2(y)} µg/m³`, `flag + HIPS ${f2(f.flagHips)} · spectrum only ${f2(f.flagSpectrum)}`, f.outside ? 'outside the IMPROVE range' : 'inside the IMPROVE range'] }]
  })
  const pts = toPoints(filters)
  const markedPts = toPoints(marked)
  const flaggedN = pts.filter((p) => p.color === FLAGGED).length
  return (
    <ChartFrame id="ml-decision-flag-filters" controls={o.controls()}
      title={`${names.get(site) ?? site}: ${flaggedN} of ${pts.length} filters flagged, against HIPS BC at ${m.label} (§6)`}
      subtitle={`Every filter of the site with HIPS (the workbook's network set, Filters → excluded applied${dropped ? `: ${dropped} out across the network` : ''}). Red: the + HIPS flag at ≥ 0.5; hollow: outside the IMPROVE range. The fit is the gallery's chosen estimators refitted on these points (Deming with the workbook's Addis error ratio, λ = ${f2(demingLambda(m.mac, lambda10))} at ${m.label}). The flag itself is a model output and does not change with the MAC.`}
      provenance={provenance(data, ['network_filters'])}>
      <Crossplot title="" points={pts} marked={markedPts} errorRatio={demingLambda(m.mac, lambda10)} grouping="pooled" sameScale maxSide={440}
        xLabel={`HIPS BC = Fabs ÷ ${m.mac} (µg/m³)`} yLabel={`FTIR EC, ${model ? 'this workbook' : 'SPARTAN published'} (µg/m³)`} />
      <ExcludedNote count={markedPts.length} drawn />
      <Key items={[{ label: 'flagged (+ HIPS ≥ 0.5)', color: FLAGGED }, { label: 'not flagged', color: NOT_FLAGGED }, { label: 'hollow: outside the IMPROVE range', color: '#888' }]} />
    </ChartFrame>
  )
}
