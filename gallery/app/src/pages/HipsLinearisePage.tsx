import { useEffect, useId, useMemo, useState, type ReactNode } from 'react'
import { group as groupBy, rollup } from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { AxisChoices, type AxisOption } from '@/components/AxisChoices'
import { Segmented, Select } from '@/components/Controls'
import { DrawerScroll, DrawerShell } from '@/components/DrawerShell'
import { LoadState } from '@/components/ExportState'
import { XIcon } from '@/components/Icons'
import { IntervalStrip } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { IdentityLine, Marker, RefLine } from '@/components/RefLines'
import { ScatterPlot, type ScatterPoint, type ScatterScales } from '@/components/ScatterPlot'
import { ScrubField } from '@/components/ScrubField'
import { StepButton } from '@/components/StepButton'
import { useChartGrouping } from '@/components/GroupingDialog'
import { useExport, useLoaded } from '@/lib/data'
import { loadSpartanIndex } from '@/lib/spartan'
import { NETWORK_VIEWS, networkOfView, useFirstSampleSite, useGalleryNetwork, viewOfNetwork, type NetworkView } from '@/lib/gallerySamples'
import { NO_BAND_COLOR } from '@/lib/groupings'
import { ALL_SITES, SitePicker, usePageSite } from '@/lib/siteFocus'
import type { FilterRow } from '@/lib/types'
import { fmt, signed } from '@/lib/format'
import { PinRing, useHighlightActions } from '@/lib/highlight'
import { ExcludedNote, FixedSetNote, useExclusionSplit } from '@/lib/markedRows'
import { SidebarControls } from '@/lib/pageSidebar'
import { RESEARCH_SERIES } from '@/lib/palettes'
import type { SpartanReference } from '@/lib/referencePopulation'
import { isFiniteNumber, median } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { FilterDecimals, SiteDecimals } from '@/pages/HipsDecimals'
import { ImproveReach, LevelChoices } from '@/pages/HipsCaveats'
import { DataTable, LINEAR, MEASURED, Readouts, StatOptions } from '@/pages/hipsLineariseParts'
import { correction, fitAgainstHips, linearBc, linearise, noiseGain, opticalDepth, predictedExponent, previousLoadingGroups, ratioMedians, saturate, SPARTAN_LOADING_MAX_GAP_DAYS, type LinFilter, type LoadingGroup } from '@/lib/hipsLinearise'

export const COMMAND = 'uv run python gallery/data/export_hips_linearise.py'
export const WORKFLOW = 'research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py'
const NOTE_DOC = 'docs/addis-reference-closure-2026-11-10.md'
export const PROVENANCE = <>{WORKFLOW}; summary <code>{NOTE_DOC}</code> and its JSON; per-filter optical depths from the workflow's <code>load()</code> (<code>hips_linearise/linearise.json</code>).</>


type CeilingKey = 'central' | 'lo' | 'hi' | 'era'
const CEILING_KEYS: readonly CeilingKey[] = ['central', 'lo', 'hi', 'era']
const HIPS_SERIES = ['measured', 'linearised'] as const
const HIPS_VIEWS = ['both', ...HIPS_SERIES] as const
const SITE_VIEWS = ['clear', 'all'] as const
type BendSampleNetwork = 'IMPROVE' | 'SPARTAN'
/** the bend's axes: x is a loading on the filter, y an optical quantity; (EC, τ) is the view the curves are drawn in */
type BendX = 'ec' | 'pm' | 'fe'
type BendY = 'tau' | 'fabs' | 'mac'
const BEND_X: readonly BendX[] = ['ec', 'pm', 'fe']
const BEND_Y: readonly BendY[] = ['tau', 'fabs', 'mac']
const BEND_AXES: AxisOption[] = [
  { key: 'ec', group: 'On the filter (µg)', label: 'EC on the filter (µg)', roles: ['x'], hint: 'IMPROVE: TOR · SPARTAN: FTIR' },
  { key: 'pm', group: 'On the filter (µg)', label: 'PM2.5 on the filter (µg)', roles: ['x'], hint: 'SPARTAN only' },
  { key: 'fe', group: 'On the filter (µg)', label: 'Iron on the filter (µg)', roles: ['x'], hint: 'SPARTAN only' },
  { key: 'tau', group: 'HIPS', label: 'HIPS optical depth τ', roles: ['y'] },
  { key: 'fabs', group: 'HIPS', label: 'HIPS Fabs (Mm⁻¹)', roles: ['y'], hint: 'SPARTAN only' },
  { key: 'mac', group: 'HIPS', label: 'Fabs ÷ EC (m²/g)', roles: ['y'] },
]
const bendLabel = (k: string, network: NetworkView) => k === 'ec' && network !== 'Both' ? `${network === 'IMPROVE' ? 'TOR' : 'FTIR'} EC on the filter (µg)` : BEND_AXES.find((o) => o.key === k)?.label ?? k
const SPARTAN_BEND_VIEWS = ['groups', 'filters'] as const
const SPARTAN_BEND_COLOR = RESEARCH_SERIES[2]
const SPARTAN_GROUP_PREFIX = 'spartan/'
/** the IMPROVE groups' samples ship only with the research build (the public one has the medians alone) */
export const HAS_IMPROVE_GROUPS = import.meta.env.VITE_PUBLIC_BUILD !== '1'
/** every x-y chart on the page is a square plot area this wide at most (as the gallery's crossplots are) */
export const SIDE = 480
/** a site's exponent is read when its 95 % interval is narrower than this (12 of the 23 sites) */
const CLEAR_WIDTH = 0.5

interface Ceiling { population: string; samples: number; sites: number; years: string; tau_inf: number; tau_inf_lo: number; tau_inf_hi: number; unsaturated_fabs_over_ec: number; max_stratum_tau: number }
export interface Linearise {
  generated: string
  deposit_area_cm2: number
  mac: number
  lambda: number
  a_per_ug: number
  ceilings: { key: CeilingKey; label: string; tau_inf: number }[]
  improve_ceilings: Ceiling[]
  loading_edges_ug?: number[]
  strata: { n: number; loading_ug: number; tau: number; fabs_over_ec: number; fitted_tau: number }[]
  filters: LinFilter[]
  fit_measured: { slope: number; intercept: number; n: number }
  network: { site: string; code: string; n: number; median_optical_depth: number; exponent: number; lo: number; hi: number; predicted: number }[]
  season_lot: { season: string; lot: string; n: number; median_optical_depth: number; published_over_hips: number; published_over_linearised_hips: number; published_over_pm25: number }[]
}

/** the page's shared state: the ceiling every chart linearises with (URL `hl.ceil`) */
export function useCeiling(data: Linearise | null) {
  const [key, setKey] = useSearchParam<CeilingKey>('hl.ceil', 'central', CEILING_KEYS)
  const list = data?.ceilings ?? []
  const pick = list.find((c) => c.key === key) ?? list[0]
  const control = <Segmented value={key} options={list.map((c) => c.key)} onChange={setKey}
    optionLabel={(o) => list.find((c) => c.key === o)?.label ?? o}
    title="Choose the central fit, either end of its 90 % interval, or the 2015–2025 fit. The ceiling applies across this page." />
  return { ceiling: pick?.tau_inf ?? NaN, label: pick?.label ?? '', key, control, list }
}

/** a stroke path through (x, y) pairs in plot pixels, skipping non-finite points */
/** "y = 2.31x − 6.17" */
const eq = (f: { slope: number; intercept: number } | null | undefined) => (f ? `y = ${fmt(f.slope, 2)}x ${f.intercept < 0 ? '−' : '+'} ${fmt(Math.abs(f.intercept), 2)}` : '—')
export const pathOf = (pts: [number, number][]) => pts.filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b)).map(([a, b], i) => `${i ? 'L' : 'M'}${a},${b}`).join('')
export const range = (a: number, b: number, step: number) => Array.from({ length: Math.round((b - a) / step) + 1 }, (_, i) => a + i * step)

/**
 * What "HIPS linearised" means at Addis, step by step, as the 10 Nov closure note explains it: the IMPROVE bend that
 * sets the ceiling, the one-line inversion, how large the correction is on each Addis filter, what it does to the
 * FTIR EC fit and its intercept, the network check, and the season and lot ratios. One ceiling control drives every
 * chart; the per-filter numbers are recomputed in the browser (common/linearise.ts) and reproduce the workflow's
 * saved fits (linearise.test.mjs).
 */
export function HipsLinearisePage() {
  const { data, error } = useExport<Linearise>('hips_linearise/linearise.json')
  const ceil = useCeiling(data)
  if (error || !data) return (
    <ChartFrame id="hl-missing" title="Linearising HIPS at Addis" provenance={WORKFLOW}>
      <LoadState what="the linearisation export" error={error} command={COMMAND} />
    </ChartFrame>
  )
  return (
    <>
      <CeilingOptions ceil={ceil} />
      <StatOptions />
      <Bend data={data} ceil={ceil} />
      <Worked data={data} ceil={ceil} />
      <CorrectionCurve data={data} ceil={ceil} />
      <AgainstHips data={data} ceil={ceil} />
      <NetworkExponents data={data} />
      <SeasonLot data={data} ceil={ceil} />
      <ImproveReach data={data} ceil={ceil} />
      <LevelChoices data={data} />
      <SiteDecimals />
      <FilterDecimals data={data} ceil={ceil} />
      <Limits data={data} />
    </>
  )
}

export type Ceil = ReturnType<typeof useCeiling>

export function CeilingOptions({ ceil }: { ceil: Ceil }) {
  return <SidebarControls title="Ceiling τ∞" tabNote={`τ∞ ${fmt(ceil.ceiling, 2)}`}>
    <div role="group" aria-label="Ceiling τ∞">{ceil.control}</div>
  </SidebarControls>
}

// ------------------------------------------------------------------ 1. the bend
export function Bend({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const [tauText, setTau] = useSearchParam<string>('hl.tau', '0.97')
  const [group, setGroup] = useSearchParam<string>('hl.group', 'none')
  // the network is the gallery's (lib/gallerySamples; the HIPS pages open on IMPROVE); `hl.bend.network` is the old key
  const { network: galleryNetwork, setNetwork: setGalleryNetwork } = useGalleryNetwork()
  const network = viewOfNetwork(galleryNetwork)
  const setNetwork = (v: NetworkView) => setGalleryNetwork(networkOfView(v))
  const [legacyNetwork, setLegacyNetwork] = useSearchParam<string>('hl.bend.network', '')
  useEffect(() => {
    if (!(NETWORK_VIEWS as readonly string[]).includes(legacyNetwork)) return
    setGalleryNetwork(networkOfView(legacyNetwork as NetworkView))
    setLegacyNetwork('')
  }, [legacyNetwork]) // eslint-disable-line react-hooks/exhaustive-deps
  const [xKey, setX] = useSearchParam<BendX>('hl.bend.x', 'ec', BEND_X)
  const [yKey, setY] = useSearchParam<BendY>('hl.bend.y', 'tau', BEND_Y)
  /** the view the IMPROVE curves, the construction and the loading groups are drawn in */
  const canonical = xKey === 'ec' && yKey === 'tau'
  const [spartanView, setSpartanView] = useSearchParam<(typeof SPARTAN_BEND_VIEWS)[number]>('hl.bend.spartan', 'groups', SPARTAN_BEND_VIEWS)
  const [spartanGroup, setSpartanGroup] = useSearchParam<string>('hl.bend.group', 'none')
  const [sampleDrawer, setSampleDrawer] = useState<BendSampleNetwork | null>(null)
  const closeSamples = () => setSampleDrawer(null)
  const { openRecord } = useHighlightActions()
  const showImprove = network !== 'SPARTAN'
  const showSpartan = network !== 'IMPROVE'
  const bothNetworks = network === 'Both'
  const { data: spartan, error: spartanError } = useExport<SpartanReference>(showSpartan ? 'spartan_reference.json' : null)
  const spartanRows = useMemo(() => (spartan?.rows ?? []).flatMap((row) => {
    const ec = row['EC (FTIR)'], fabs = row['HIPS Fabs'], volume = row['Sampled volume']
    if (!isFiniteNumber(ec) || !isFiniteNumber(fabs) || fabs <= 0 || !isFiniteNumber(volume) || volume <= 0) return []
    const pm = row['PM2.5 mass'], fe = row['Iron']
    // every quantity the axes offer, per filter (loadings in µg; Fabs ÷ EC in m²/g)
    const values: Record<BendX | BendY, number> = { ec: ec * volume, pm: isFiniteNumber(pm) ? pm * volume : NaN, fe: isFiniteNumber(fe) ? fe * volume : NaN,
      tau: opticalDepth(fabs, volume), fabs, mac: ec > 0 ? fabs / ec : NaN }
    return [{ row, fabs, volume, values, x: values.ec, y: values.tau }]
  }), [spartan])
  // the site is the gallery's page site (lib/siteFocus `psite`), first the Samples' first site; `hl.bend.site` is the old key
  const names = useSpartanNames()
  const siteList = useMemo(() => [...new Map(spartanRows.map(({ row }) => [row.code, { code: row.code, name: names.get(row.code) ?? row.site }])).values()]
    .sort((a, b) => a.name.localeCompare(b.name)), [spartanRows, names])
  const firstSample = useFirstSampleSite()
  const sitePick = usePageSite(siteList, { legacy: ['hl.bend.site'], prefer: firstSample ?? ALL_SITES, all: `All ${siteList.length} sites` })
  const activeSite = sitePick.all ? 'all' : sitePick.code
  const siteRows = useMemo(() => spartanRows.filter(({ row }) => activeSite === 'all' || row.code === activeSite), [spartanRows, activeSite])
  // the reader's grouping (lib/groupings, `grp` in a link) colours the individual filters; URL `hl-bend.group`
  const groupRows = useMemo(() => siteRows.map(({ row }) => row as unknown as FilterRow), [siteRows])
  const { grp, picker: groupingPicker, missingNote } = useChartGrouping('hl-bend', groupRows, () => setSpartanView('filters'))
  const spartanPoints = useMemo(() => siteRows.flatMap<ScatterPoint>(({ row, fabs, volume, values }) => {
    const x = values[xKey], y = values[yKey]
    if (!Number.isFinite(x) || !Number.isFinite(y)) return []
    const band = grp?.labelOf(row as unknown as FilterRow) ?? null
    const site = names.get(row.code) ?? row.site
    return [{
      id: row.id, key: row.id, x, y, color: grp ? (band ? grp.colorOf(band) : NO_BAND_COLOR) : SPARTAN_BEND_COLOR, r: 3, opacity: 0.65,
      tip: [row.id, `${site} · ${row.date}`, `${bendLabel(xKey, 'SPARTAN')} ${fmt(x, 2)}`, `${bendLabel(yKey, 'SPARTAN')} ${fmt(y, 3)}`, ...(grp ? [`${grp.labels.length ? band ?? 'No data' : ''}`] : [])],
      rec: { id: row.id, site, date: row.date, source: 'The bend · SPARTAN', fields: [
        ['FTIR EC loading (µg)', values.ec], ['HIPS optical depth τ', values.tau], ['Sampled volume (m³)', volume], ['HIPS Fabs (Mm⁻¹)', fabs],
        ['PM2.5 loading (µg)', values.pm], ['Iron loading (µg)', values.fe], ['Fabs ÷ EC (m²/g)', values.mac],
      ] },
    }]
  }), [siteRows, xKey, yKey, grp, names])
  const { counted, nOut } = useExclusionSplit(siteRows, ({ row }) => row.id)
  // the loading groups bin by EC loading and take median τ: drawn only in the (EC, τ) view, and not under a grouping
  const showSpartanGroups = spartanView === 'groups' && !!data.loading_edges_ug?.length && canonical && !grp
  const spartanGroups = useMemo(() => {
    if (!showSpartanGroups || !data.loading_edges_ug) return []
    return previousLoadingGroups(counted.map(({ row, x, y }) => ({ id: row.id, site: row.code, date: row.date, loading: x, tau: y })), data.loading_edges_ug).map((bin) => {
      const x = median(bin.rows.map((row) => row[3])), y = median(bin.rows.map((row) => row[4]))
      const n = bin.rows.length, sites = new Set(bin.rows.map((row) => row[0])).size
      const binKey = String(bin.lower)
      const details = [`FTIR EC ${fmt(x, 1)} µg`, `optical depth τ ${fmt(y, 3)}`, `${n.toLocaleString()} samples across ${sites} ${sites === 1 ? 'site' : 'sites'}`]
      return { bin, binKey, key: `${SPARTAN_GROUP_PREFIX}${binKey}`, x, y, r: 3 + Math.sqrt(n) / 6, color: SPARTAN_BEND_COLOR, shape: 'square' as const, details,
        tip: ['SPARTAN group median', ...details, 'Click to view the samples in this group'] }
    })
  }, [counted, data.loading_edges_ug, showSpartanGroups])
  const selectedSpartan = spartanGroups.find(({ binKey }) => binKey === spartanGroup)
  const pickSpartanGroup = (key: string) => {
    setSpartanGroup(key)
    setSampleDrawer(key === 'none' ? null : 'SPARTAN')
  }
  const showSpartanFilter = (id: string) => {
    const record = spartanPoints.find((p) => p.id === id)?.rec
    if (record) {
      closeSamples()
      openRecord(record)
    }
  }
  const spartanDomain = useMemo<[number, number]>(() => [
    Math.min(0, ...spartanRows.map(({ x }) => x)), Math.max(90, ...spartanRows.map(({ x }) => x)),
  ], [spartanRows])
  const taus = useMemo(() => data.filters.filter((f) => f.ftir != null).map((f) => f.tau), [data])
  const [lo, hi] = [Math.min(...taus), Math.max(...taus)]
  const t = Math.min(hi, Math.max(0.05, Number(tauText) || 0.97))
  const a = data.a_per_ug, c = ceil.ceiling
  const tl = linearise(t, c), L = tl / a
  const clip = useId()
  const points = useMemo(() => data.strata.map((s, i) => {
    const details = [`TOR EC ${fmt(s.loading_ug, 1)} µg`, `optical depth τ ${fmt(s.tau, 3)}`, `Fabs ÷ EC ${fmt(s.fabs_over_ec, 1)} m²/g`, `${s.n.toLocaleString()} samples`]
    return {
      key: String(i + 1), x: s.loading_ug, y: yKey === 'mac' ? s.fabs_over_ec : s.tau, n: s.n, color: INK.muted, r: 3 + Math.sqrt(s.n) / 40, opacity: 0.85,
      details, tip: ['IMPROVE group median', ...details, HAS_IMPROVE_GROUPS ? 'Click to view the samples in this group' : 'Click to select or clear this group'],
    }
  }), [data.strata, yKey])
  /** IMPROVE's groups carry TOR EC loading, τ and Fabs ÷ EC only */
  const improveHasAxes = xKey === 'ec' && yKey !== 'fabs'
  const selected = improveHasAxes ? points.find((p) => p.key === group) : undefined
  const pickGroup = (key: string) => {
    setGroup(key)
    setSampleDrawer(HAS_IMPROVE_GROUPS && key !== 'none' ? 'IMPROVE' : null)
  }
  const spartanMarks = showSpartanGroups ? spartanGroups : spartanPoints
  const improveMarks = improveHasAxes ? points : []
  const plotPoints = useMemo(() => bothNetworks ? [...improveMarks, ...spartanMarks] : showImprove ? improveMarks : spartanMarks,
    [bothNetworks, showImprove, improveMarks, spartanMarks])
  const pickPoint = ({ key, rec }: ScatterPoint) => {
    if (rec) { closeSamples(); openRecord(rec); return }
    if (!key) return
    if (key.startsWith(SPARTAN_GROUP_PREFIX)) pickSpartanGroup(key.slice(SPARTAN_GROUP_PREFIX.length))
    else pickGroup(!HAS_IMPROVE_GROUPS && key === group ? 'none' : key)
  }
  const under = ({ x, y, iw, ih }: ScatterScales) => {
    if (!canonical) return null
    // SPARTAN alone gets the same reference as IMPROVE: its curves are IMPROVE's (thermal EC), drawn here to compare FTIR EC against
    const maxLoading = x.domain()[1]
    const xs = range(0, maxLoading, 0.5)
    return <g>
      <defs><clipPath id={clip}><rect width={iw} height={ih} /></clipPath></defs>
      <g clipPath={`url(#${clip})`}>
        <rect x={0} y={y(hi)} width={iw} height={y(lo) - y(hi)} fill={MEASURED} opacity={0.1} />
        <text x={iw - 6} y={y(hi) - 6} textAnchor="end" fontSize={10.5} fontFamily={FONT.mono} fill={INK.muted}>Addis filters, τ {fmt(lo, 2)}–{fmt(hi, 2)}</text>
        <Marker orient="h" at={y(c)} from={0} to={iw} color={MEASURED} label={`${showImprove ? '' : 'IMPROVE '}ceiling τ∞ = ${fmt(c, 2)}`} />
        <path d={pathOf([[x(0), y(0)], [x(maxLoading), y(a * maxLoading)]])} fill="none" stroke={LINEAR} strokeWidth={2} strokeDasharray="6 4" />
        <path d={pathOf(xs.map((v) => [x(v), y(saturate(a * v, c))]))} fill="none" stroke={MEASURED} strokeWidth={2} />
      </g>
    </g>
  }
  const construction = ({ x, y, iw }: ScatterScales) => {
    if (!Number.isFinite(tl) || !canonical) return null
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
  const spartanLabel = showSpartanGroups ? 'SPARTAN group medians (FTIR EC)' : 'SPARTAN individual filters (FTIR EC)'
  const spartanDescription = showSpartanGroups
    ? `SPARTAN groups use the IMPROVE loading bins and FTIR EC from the previous eligible date at that site, within ${SPARTAN_LOADING_MAX_GAP_DAYS} days. If several filters share that date, their median sets the bin. Click a square to see its filters by site.`
    : 'Each SPARTAN dot is an individual filter. FTIR EC loading is the reported concentration multiplied by sampled volume. Click a dot to open its filter record.'
  return (
    <ChartFrame id="hl-bend" title={!canonical ? `The bend, other axes — ${bendLabel(yKey, network)} against ${bendLabel(xKey, network)}, ${network === 'Both' ? 'IMPROVE and SPARTAN' : network}`
      : network === 'IMPROVE' ? 'The bend — HIPS optical depth against thermal EC on the filter, IMPROVE 2005–2025'
      : bothNetworks ? 'The bend — HIPS optical depth against EC on the filter, IMPROVE and SPARTAN'
      : 'The bend — HIPS optical depth against FTIR EC on the filter, SPARTAN'}
      subtitle={!canonical
        ? `The IMPROVE curves and the τ construction are drawn on EC against optical depth only. IMPROVE's groups carry TOR EC loading, τ and Fabs ÷ EC, so they show on EC against τ or Fabs ÷ EC; PM2.5, iron and Fabs are SPARTAN's (each filter's concentration × its sampled volume). Click an axis title to change it.${showSpartan ? ` ${spartanDescription}` : ''}`
        : bothNetworks
        ? `The grey dots and reference curves come from IMPROVE thermal EC (TOR). SPARTAN is overlaid at its FTIR-derived EC loading for visual comparison. ${spartanDescription}`
        : showSpartan ? `${spartanDescription} Optical depth uses measured HIPS Fabs and sampled volume. The reference is IMPROVE's, as on the IMPROVE view: the dashed blue line is the proportional response (a = ${fmt(a, 4)} µg⁻¹), the orange curve the fitted saturating response, the dashed orange line its ceiling, and the shaded band the Addis filters' τ range. Move the measured τ to trace the conversion to linearised τ.`
        : `Each grey dot is the median of one IMPROVE group, binned by the previous sample’s TOR EC loading at the same site (${primary.samples.toLocaleString()} samples at ${primary.sites} sites in the parent population). Dot size shows group sample count.${HAS_IMPROVE_GROUPS ? ' Click a dot to see its samples.' : ''} The dashed blue line shows a proportional response with a = ${fmt(a, 4)} µg⁻¹ (${fmt(primary.unsaturated_fabs_over_ec, 1)} m²/g on light filters). The orange curve shows the fitted saturating response. Move the measured τ to trace the conversion to linearised τ.`}
      provenance={<>{showImprove && PROVENANCE}{showSpartan && <> <code>spartan_reference.json</code> · paired FTIR EC, HIPS Fabs and sampled volume by filter ID.</>}</>}
      controls={<>
        <Segmented label="Network" value={network} options={NETWORK_VIEWS} onChange={(value) => { setNetwork(value); closeSamples() }}
          title="The gallery's network (Samples): on this page it stays with the page" />
        {showSpartan && siteList.length > 0 && <>
          <SitePicker label="SPARTAN site" as="select" {...sitePick.picker} onChange={(value) => { sitePick.set(value); closeSamples() }}
            title="The gallery's page site (psite): the other one-site charts on this page follow it" />
          {groupingPicker}
          {!!data.loading_edges_ug?.length && canonical && !grp && <Segmented label="SPARTAN view" value={spartanView} options={SPARTAN_BEND_VIEWS}
            optionLabel={(value) => value === 'groups' ? 'Group medians' : 'Individual filters'} onChange={(value) => { setSpartanView(value); closeSamples() }} />}
          {showSpartanGroups && <Select label="SPARTAN group" value={selectedSpartan?.binKey ?? 'none'}
            options={['none', ...spartanGroups.map((p) => p.binKey)]} onChange={pickSpartanGroup}
            optionLabel={(key) => { const g = spartanGroups.find((p) => p.binKey === key)?.bin; return g ? loadingRange(g) : 'None' }} />}
        </>}
        {canonical && <span className="control" title="The measured HIPS optical depth the construction follows: drag sideways across the Addis range or click to type">Measured τ <ScrubField label="Measured τ" value={t} min={0.05} max={+hi.toFixed(2)} step={0.01} width={96} format={(v) => fmt(v, 2)} onChange={(v) => setTau(String(+v.toFixed(2)))} /></span>}
        {showImprove && canonical && <>
          <Select label="IMPROVE group" value={selected?.key ?? 'none'} options={['none', ...points.map((p) => p.key)]} onChange={pickGroup}
            optionLabel={(key) => points.find((p) => p.key === key)?.details[0] ?? 'None'} />
        </>}
      </>}>
      {plotPoints.length > 0 && <>
        <AxisChoices options={BEND_AXES} x={[xKey, (v) => setX(v as BendX)]} y={[yKey, (v) => setY(v as BendY)]} noun="quantities">
        <ScatterPlot points={plotPoints} xLabel={bendLabel(xKey, network)} yLabel={bendLabel(yKey, network)} pick swap="hl-bend"
          {...(canonical ? { xDomain: showSpartan ? spartanDomain : [0, 90] as [number, number], yDomain: [0, 3.4] as [number, number], nice: false }
            : { xInclude: [0], yInclude: [0], xClip: [0, 0.995] as [number, number], yClip: [0, 0.995] as [number, number] })}
          maxSide={SIDE} outside={showSpartan ? 'pin' : 'clip'} note={showSpartan && !showSpartanGroups ? 'mark' : false} under={under} onPick={pickPoint}>
          {(scales) => <>
            {construction(scales)}
            {showImprove && selected && <PinRing cx={scales.x(selected.x)} cy={scales.y(selected.y)} r={selected.r} />}
            {showSpartan && selectedSpartan && <PinRing cx={scales.x(selectedSpartan.x)} cy={scales.y(selectedSpartan.y)} r={selectedSpartan.r} />}
          </>}
        </ScatterPlot>
        </AxisChoices>
        <Key items={[
          ...(showImprove && improveHasAxes ? [{ label: 'IMPROVE group medians (TOR EC)', color: INK.muted, kind: 'dot' as const }] : []),
          ...(canonical ? [{ label: 'IMPROVE proportional response', color: LINEAR, kind: 'dash' as const },
            { label: 'IMPROVE saturating fit', color: MEASURED, kind: 'line' as const },
            { label: 'Addis τ range', color: MEASURED, kind: 'faded' as const }] : []),
          ...(showSpartan && spartanMarks.length && !grp ? [{ label: spartanLabel, color: SPARTAN_BEND_COLOR, kind: showSpartanGroups ? 'square' as const : 'dot' as const }] : []),
          ...(showSpartan && grp ? [...grp.labels.map((label) => ({ label: `SPARTAN · ${label}`, color: grp.colorOf(label), kind: 'dot' as const })),
            { label: 'SPARTAN · no data', color: NO_BAND_COLOR, kind: 'dot' as const }] : []),
        ]} />
        {showSpartan && missingNote('drawn in grey')}
      </>}
      {showImprove && !improveHasAxes && !showSpartan && <Empty>IMPROVE's groups carry TOR EC loading, optical depth and Fabs ÷ EC only. Choose SPARTAN or Both for PM2.5, iron or Fabs, or click an axis title to go back.</Empty>}
      {showSpartan && (!spartan ? <LoadState what="SPARTAN filter measurements" error={spartanError} /> : spartanMarks.length === 0 && <Empty>{showSpartanGroups
        ? 'No eligible SPARTAN sample pairs at this site. Choose Individual filters to see its measurements.'
        : 'No paired SPARTAN FTIR EC, HIPS Fabs and sampled volume measurements.'}</Empty>)}
      {showSpartan && showSpartanGroups && <>
        <ExcludedNote count={nOut} />
        {selectedSpartan && <>
          <p className="chart-note"><strong>SPARTAN group median</strong> · {selectedSpartan.details.join(' · ')}</p>
          <button type="button" className="btn" onClick={() => setSampleDrawer('SPARTAN')}>{bothNetworks ? 'View SPARTAN group samples' : 'View group samples'}</button>
          {sampleDrawer === 'SPARTAN' && <LoadingGroupDrawer key={`${activeSite}/${selectedSpartan.key}`} network="SPARTAN" data={selectedSpartan.bin}
            details={selectedSpartan.details} onClose={closeSamples} onOpenFilter={showSpartanFilter} />}
        </>}
      </>}
      {showImprove && <>
        <div role="status">
          {selected && <p className="chart-note"><strong>IMPROVE group median</strong> · {selected.details.join(' · ')}</p>}
        </div>
        {HAS_IMPROVE_GROUPS && selected && <>
          <button type="button" className="btn" onClick={() => setSampleDrawer('IMPROVE')}>{bothNetworks ? 'View IMPROVE group samples' : 'View group samples'}</button>
          {sampleDrawer === 'IMPROVE' && <ImproveGroupDrawer key={selected.key} group={selected} onClose={closeSamples} />}
        </>}
      </>}
      {canonical && <><Readouts items={[
        { label: 'Measured τ', value: fmt(t, 2), color: MEASURED },
        { label: 'Linearised τ', value: Number.isFinite(tl) ? fmt(tl, 2) : '—', color: LINEAR },
        { label: 'Correction ×', value: Number.isFinite(tl) ? fmt(tl / t, 2) : '—' },
        { label: 'Noise amplified ×', value: Number.isFinite(tl) ? fmt(noiseGain(t, c), 1) : '—', sub: '1 / (1 − τ/τ∞)²' },
      ]} />
      <Note>Set the ceiling under Page options. The noise factor is the slope of the inverse, 1 / (1 − τ/τ∞)². At 70 % of the ceiling it is about 11.{!showImprove && ' The curves and the construction are IMPROVE\'s, fitted on thermal (TOR) EC; SPARTAN is drawn against FTIR EC, so a gap between its filters and the curves is partly the difference between the two ECs.'}</Note></>}
    </ChartFrame>
  )
}

const GROUP_PAGE_SIZE = 50
const loadingRange = (g: LoadingGroup) => g.upper == null ? `≥ ${fmt(g.lower, 1)} µg` : `${fmt(g.lower, 1)}–<${fmt(g.upper, 1)} µg`

export function ImproveGroupDrawer({ group, onClose }: { group: { key: string; n: number; details: string[] }; onClose: () => void }) {
  const { data, error } = useExport<LoadingGroup>(`research/hips_linearise_groups/${group.key}.json.gz`, (d) =>
    d.rows.length === group.n ? null : 'sample count differs from the plotted group')
  return <LoadingGroupDrawer network="IMPROVE" details={group.details} data={data} onClose={onClose}
    pending={<LoadState what="the IMPROVE group samples" error={error} command={`${COMMAND} --improve-groups`} />} />
}

function LoadingGroupDrawer({ network, details, data, onClose, pending, onOpenFilter }: {
  network: BendSampleNetwork; details: string[]; data: LoadingGroup | null; onClose: () => void
  pending?: ReactNode; onOpenFilter?: (id: string) => void
}) {
  const [site, setSite] = useState('all')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const bySite = useMemo(() => groupBy(data?.rows ?? [], ([code]) => code), [data])
  const matches = useMemo(() => {
    const rows = site === 'all' ? data?.rows ?? [] : bySite.get(site) ?? []
    const query = search.trim().toLowerCase()
    if (!query) return rows
    const terms = query.split(/\s+/)
    return rows.filter(([code, date]) => terms.every((term) => `${code} ${date}`.toLowerCase().includes(term)))
  }, [bySite, data, search, site])
  const counts = useMemo(() => rollup(matches, (rows) => rows.length, ([code]) => code), [matches])
  const start = page * GROUP_PAGE_SIZE
  const end = Math.min(start + GROUP_PAGE_SIZE, matches.length)
  const sections = groupBy(matches.slice(start, end), ([code]) => code)
  const ec = network === 'IMPROVE' ? 'TOR EC' : 'FTIR EC'
  const title = `${network} group samples`
  return <DrawerShell label={title} onClose={onClose} portal>
    <header className="drawer-head">
      <div><div className="drawer-id">{title}</div><div className="drawer-sub">{details.join(' · ')}</div></div>
      <button type="button" className="btn quiet x-btn" onClick={onClose} aria-label="Close group samples"><XIcon /></button>
    </header>
    {data ? <div className="drawer-section">
      <p>Previous sampling date’s {ec} loading at the same site was at least {fmt(data.lower, 1)} µg{data.upper != null && ` and below ${fmt(data.upper, 1)} µg`}.</p>
      <div className="drawer-actions">
        <Select label="Site" value={site} options={['all', ...bySite.keys()]}
          optionLabel={(code) => code === 'all' ? `All ${bySite.size} sites` : `${code} (${bySite.get(code)?.length.toLocaleString()})`}
          onChange={(value) => { setSite(value); setPage(0) }} />
        {/* the gallery's labelled field, as the Site box beside it (label in muted text, the box in .select) */}
        <label className="control">Search <input className="select" type="search" aria-label="Search by site code or date" placeholder="Site or YYYY-MM-DD" style={{ width: 150 }} value={search}
          onChange={(event) => { setSearch(event.target.value); setPage(0) }} /></label>
        <span className="control">
          <StepButton dir="prev" label="Previous sample page" disabled={page === 0} onClick={() => setPage(page - 1)} />
          <span role="status">{matches.length ? `${(start + 1).toLocaleString()}–${end.toLocaleString()} of ${matches.length.toLocaleString()}` : 'No matching samples'}</span>
          <StepButton dir="next" label="Next sample page" disabled={end >= matches.length} onClick={() => setPage(page + 1)} />
        </span>
      </div>
      <DrawerScroll label={`Samples in this ${network} group`}><table className="placement group-rows">
        <thead><tr>{onOpenFilter && <th scope="col">Filter</th>}<th scope="col">Sample date</th><th scope="col" className="num">Previous {ec} (µg)</th><th scope="col" className="num">{ec} (µg)</th><th scope="col" className="num">HIPS τ</th></tr></thead>
        {[...sections].map(([code, rows]) => <tbody key={code}>
          <tr><th scope="rowgroup" colSpan={onOpenFilter ? 5 : 4}>{code} <span className="u">· {counts.get(code)?.toLocaleString()}</span></th></tr>
          {rows.map(([, date, previous, loading, tau, id]) => <tr key={id ?? date}>
            {onOpenFilter && <td>{id && <button type="button" className="link-btn" onClick={() => onOpenFilter(id)}>{id}</button>}</td>}
            <td className="mono">{date}</td><td className="mono num">{fmt(previous, 2)}</td>
            <td className="mono num">{fmt(loading, 2)}</td><td className="mono num">{fmt(tau, 3)}</td>
          </tr>)}
        </tbody>)}
      </table></DrawerScroll>
    </div> : pending}
  </DrawerShell>
}

// ------------------------------------------------------------------ 2. one filter, step by step
export function Worked({ data, ceil }: { data: Linearise; ceil: Ceil }) {
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
      subtitle="Pick a filter (sorted by optical depth). Every number is recomputed from its HIPS value and sampled volume at the ceiling set in Page options. Because V, A and MAC cancel in the ratio, linearising BC is the same multiplication as linearising τ."
      provenance={PROVENANCE} exportable={false}
      controls={<Select label="Filter" value={f.id} options={sorted.map((r) => r.id)} onChange={setId}
        optionLabel={(o) => { const r = sorted.find((s) => s.id === o)!; return `${o} · ${r.season} · τ ${fmt(r.tau, 2)}` }} />}>
      <DataTable rows={rows} rowKey={(_r, i) => `${i}`} columns={[
        { key: 'n', label: '#', num: true, value: (_r) => String(rows.indexOf(_r) + 1) },
        { key: 's', label: 'Step', value: (r) => r.step }, { key: 'w', label: 'Formula', value: (r) => <code>{r.what}</code> }, { key: 'v', label: 'This filter', value: (r) => r.value }]} />
      <Note explain>Ceiling {ceil.label}. V is this filter's sampled air volume{f.volume_known ? '' : ' (not on its HIPS row: the Addis median stands in, volumes vary by under 2 %)'}. The loading L is "EC on the filter as IMPROVE's light filters would rate it", so it carries IMPROVE's light-filter MAC and is not a measured Addis EC.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 3. the correction factor
/**
 * The correction factor 1 / (1 − τ/τ∞) against measured optical depth: the curve at the chosen ceiling, the band of
 * the ceiling's 90 % interval, and a set of filters on it with their median marked. The curve is the same for every
 * set of filters; only where the filters fall on it differs (Addis here, any SPARTAN site on the part-by-part page).
 */
export function CorrectionPlot({ points, ceiling, lo, hi, median: med, medianLabel, lightBox = false }: {
  points: ScatterPoint[]; ceiling: number; lo: number; hi: number
  /** the median optical depth of the filters shown, marked with a dashed line up to the curve */
  median?: number; medianLabel?: string
  /** the dashed box where most SPARTAN sites sit (τ under 0.5, factor under 1.3) */
  lightBox?: boolean
}) {
  const clip = useId()
  const mc = med == null ? NaN : correction(med, ceiling)
  const under = ({ x, y, iw, ih }: ScatterScales) => {
    const ts = range(0, 1.6, 0.01)
    const band = pathOf([...ts.map((t): [number, number] => [x(t), y(Math.min(5, correction(t, lo)))]), ...[...ts].reverse().map((t): [number, number] => [x(t), y(correction(t, hi))])]) + 'Z'
    return <g>
      <defs><clipPath id={clip}><rect width={iw} height={ih} /></clipPath></defs>
      <g clipPath={`url(#${clip})`}>
        <path d={band} fill={LINEAR} opacity={0.1} />
        <path d={pathOf(ts.map((t) => { const k = correction(t, ceiling); return [x(t), k < 6 ? y(k) : NaN] }))} fill="none" stroke={LINEAR} strokeWidth={2} />
        {lightBox && <>
          <rect x={x(0)} y={y(1.3)} width={x(0.5) - x(0)} height={y(0.9) - y(1.3)} fill="none" stroke={INK.muted} strokeDasharray="2 3" />
          <text x={x(0.02)} y={y(1.3) - 6} fontSize={10.5} fontFamily={FONT.mono} fill={INK.muted}>most SPARTAN sites</text>
        </>}
      </g>
    </g>
  }
  const children = ({ x, y, iw }: ScatterScales) => {
    if (med == null || !Number.isFinite(mc)) return null
    // the label sits in the empty space above the curve, on a leader from the median point: right of it, or left when
    // the point is in the right part of the plot; a halo in the panel colour keeps it legible over the band and dots
    const flip = x(med) > iw * 0.55
    const px = x(med), py = y(mc), ly = Math.max(14, py - 72)
    return <g pointerEvents="none">
      <line x1={px} x2={px} y1={y(0.9)} y2={py} stroke={INK.text} strokeDasharray="3 3" />
      <line x1={x(0)} x2={px} y1={py} y2={py} stroke={INK.text} strokeDasharray="3 3" />
      <circle cx={px} cy={py} r={5} fill="none" stroke={INK.text} strokeWidth={1.75} />
      {medianLabel && <>
        <path d={`M${px},${py - 6}V${ly + 4}h${flip ? -6 : 6}`} fill="none" stroke={INK.text} strokeWidth={1} />
        <text x={px + (flip ? -9 : 9)} y={ly + 8} textAnchor={flip ? 'end' : 'start'} fontSize={11.5} fontWeight={600} fontFamily={FONT.mono}
          fill={INK.text} stroke={INK.panel} strokeWidth={4} strokeLinejoin="round" paintOrder="stroke">{medianLabel}</text>
      </>}
    </g>
  }
  return <ScatterPlot points={points} xLabel="Measured HIPS optical depth τ" yLabel="Correction factor τ_lin / τ" xDomain={[0, 1.6]} yDomain={[0.9, 4.2]} nice={false}
    maxSide={SIDE} outside="pin" note={false} under={under}>{children}</ScatterPlot>
}

export function CorrectionCurve({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const c = ceil.ceiling
  const { lo, hi } = { lo: data.ceilings.find((k) => k.key === 'lo')!.tau_inf, hi: data.ceilings.find((k) => k.key === 'hi')!.tau_inf }
  const shown = useMemo(() => data.filters.filter((f) => f.ftir != null), [data])
  const points = useMemo<ScatterPoint[]>(() => shown.flatMap((f) => {
    const k = correction(f.tau, c)
    return Number.isFinite(k) ? [{
      id: f.id, x: f.tau, y: k, color: MEASURED,
      tip: [`${f.id} · ${f.season}${f.lot ? ` · lot ${f.lot}` : ''}`, `τ ${fmt(f.tau, 2)} → × ${fmt(k, 2)}`],
      rec: {
        id: f.id, source: 'HIPS optical depth correction', site: 'Addis Ababa',
        fields: [
          ['Measured optical depth τ', fmt(f.tau, 3)],
          ['Ceiling τ∞', fmt(c, 2)],
          ['Correction factor', `× ${fmt(k, 2)}`],
          ['Linearised optical depth τ_lin', fmt(linearise(f.tau, c), 3)],
        ],
      },
    }] : []
  }), [shown, c])
  const med = median(shown.map((f) => f.tau)), mc = correction(med, c)
  const facs = shown.map((f) => correction(f.tau, c)).filter(Number.isFinite)
  const at = (k: number) => median(shown.map((f) => correction(f.tau, k)).filter(Number.isFinite))
  const heaviest = shown.reduce((m, f) => (f.tau > m.tau ? f : m), shown[0])
  return (
    <ChartFrame id="hl-correction" title="The correction is small for light filters and large for Addis — correction factor against measured optical depth, every Addis filter"
      subtitle={`τ_lin / τ = 1 / (1 − τ/τ∞) against measured τ. The dots are the ${shown.length} Addis filters with a published FTIR EC. Click a dot to open its filter record. Most SPARTAN sites sit to the left of 0.5, where the factor is under 1.3; Addis sits where the curve steepens.`}
      provenance={PROVENANCE}>
      <CorrectionPlot points={points} ceiling={c} lo={lo} hi={hi} median={med} medianLabel={`Addis median τ ${fmt(med, 2)} → × ${fmt(mc, 2)}`} lightBox />
      <Key items={[{ label: `chosen ceiling ${fmt(c, 2)}`, color: LINEAR, kind: 'line' }, { label: `ceiling ${fmt(lo, 2)} to ${fmt(hi, 2)} (90 %)`, color: LINEAR, kind: 'faded' }, { label: 'Addis filters', color: MEASURED, kind: 'dot' }]} />
      <Note explain>Median Addis correction × {fmt(median(facs), 2)} at this ceiling (× {fmt(at(hi), 2)} to × {fmt(at(lo), 2)} across the 90 % interval). The heaviest filter ({heaviest.id}, τ {fmt(heaviest.tau, 2)}) is corrected × {fmt(correction(heaviest.tau, c), 2)}.</Note>
      <FixedSetNote what="the closure workflow's counted Addis filters with a published FTIR EC" />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 4. FTIR EC against HIPS
type HipsY = 'ftir' | 'pm25'
const HIPS_Y: readonly HipsY[] = ['ftir', 'pm25']
const HIPS_Y_AXES: AxisOption[] = [
  { key: 'ftir', group: 'On the y axis', label: 'FTIR EC (µg/m³)', roles: ['y'], hint: 'SPARTAN published' },
  { key: 'pm25', group: 'On the y axis', label: 'PM2.5 mass (µg/m³)', roles: ['y'] },
]
const ADDIS_CODE = 'ETAD'
/** a site needs this many filters with FTIR EC, HIPS Fabs and a sampled volume to be offered */
const MIN_SITE_FILTERS = 20

/** SPARTAN site names by code (the network index): spartan_reference.json names only the four focus sites */
function useSpartanNames() {
  const { data: index } = useLoaded('spartan-index', loadSpartanIndex)
  return useMemo(() => new Map((index?.sites ?? []).map((s) => [s.code, s.name])), [index])
}

/** the filters "Stretch the HIPS axis" draws at a site: Addis is the closure workflow's set; any other site is rebuilt from the public SPARTAN rows */
function useHipsSiteFilters(data: Linearise) {
  const { data: spartan } = useExport<SpartanReference>('spartan_reference.json')
  const bySite = useMemo(() => {
    const out = new Map<string, { name: string; filters: LinFilter[] }>()
    for (const row of spartan?.rows ?? []) {
      const ec = row['EC (FTIR)'], fabs = row['HIPS Fabs'], volume = row['Sampled volume'], pm = row['PM2.5 mass']
      if (!isFiniteNumber(ec) || !isFiniteNumber(fabs) || fabs <= 0 || !isFiniteNumber(volume) || volume <= 0) continue
      const site = out.get(row.code) ?? { name: row.site, filters: [] }
      site.filters.push({ id: row.id, season: '', lot: null, tau: opticalDepth(fabs, volume), volume_m3: volume, volume_known: true,
        hips_bc: fabs / data.mac, ftir: ec, pm25: isFiniteNumber(pm) ? pm : null })
      out.set(row.code, site)
    }
    return out
  }, [spartan, data.mac])
  const refById = useMemo(() => new Map((spartan?.rows ?? []).map((r) => [r.id, r])), [spartan])
  const names = useSpartanNames()
  const sites = useMemo(() => {
    const list = [...bySite].filter(([code, s]) => code === ADDIS_CODE || s.filters.length >= MIN_SITE_FILTERS)
      .map(([code, s]) => ({ code, name: names.get(code) ?? s.name, n: code === ADDIS_CODE ? data.filters.filter((f) => f.ftir != null).length : s.filters.length }))
    if (!list.some((s) => s.code === ADDIS_CODE)) list.push({ code: ADDIS_CODE, name: 'Addis Ababa', n: data.filters.filter((f) => f.ftir != null).length })
    return list.sort((a, b) => a.name.localeCompare(b.name))
  }, [bySite, data.filters, names])
  return { bySite, refById, sites, loaded: !!spartan }
}

export function AgainstHips({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const [view, setView] = useSearchParam<(typeof HIPS_VIEWS)[number]>('hl.hips', 'both', HIPS_VIEWS)
  const [yKey, setY] = useSearchParam<HipsY>('hl.hips.y', 'ftir', HIPS_Y)
  const c = ceil.ceiling
  // the site is the gallery's page site (psite), first the Samples' first site; Addis keeps the closure workflow's filters and λ
  const { bySite, refById, sites } = useHipsSiteFilters(data)
  const sitePick = usePageSite(sites, { prefer: useFirstSampleSite() ?? ADDIS_CODE })
  const code = sitePick.code || ADDIS_CODE
  const addis = code === ADDIS_CODE
  const siteName = sites.find((s) => s.code === code)?.name ?? 'Addis Ababa'
  const siteFilters = useMemo(() => (addis ? data.filters : bySite.get(code)?.filters ?? []), [addis, data.filters, bySite, code])
  // another site's filters follow Filters → excluded; the Addis set is the workflow's own
  const { counted, nOut } = useExclusionSplit(addis ? [] : siteFilters, (f) => f.id)
  const filters = addis ? siteFilters : counted
  /** the Addis Deming λ is the closure workflow's (FTIR EC on HIPS); every other site and PM2.5 use λ = 1 */
  const lambda = addis && yKey === 'ftir' ? data.lambda : 1
  // the y the fit and the points read: FTIR EC or PM2.5 (fitAgainstHips reads `ftir`)
  const yFilters = useMemo(() => filters.map((f) => (yKey === 'ftir' ? f : { ...f, ftir: f.pm25 })), [filters, yKey])
  const yLabel = HIPS_Y_AXES.find((o) => o.key === yKey)!.label
  const yName = yKey === 'ftir' ? 'FTIR EC' : 'PM2.5'
  const groupRows = useMemo(() => yFilters.flatMap((f) => { const r = refById.get(f.id); return r ? [r as unknown as FilterRow] : [] }), [yFilters, refById])
  const { grp, picker: groupingPicker, missingNote } = useChartGrouping('hl-against-hips', groupRows)
  const series = useMemo(() => HIPS_SERIES.filter((s) => view === 'both' || view === s).map((key) => {
    const lin = key === 'linearised'
    return { key, lin, label: lin ? 'Linearised' : 'As measured', color: lin ? LINEAR : MEASURED,
      fit: fitAgainstHips(yFilters, lambda, lin ? c : null) }
  }), [yFilters, lambda, c, view])
  const shown = useMemo(() => yFilters.filter((f) => f.ftir != null), [yFilters])
  const points = useMemo<ScatterPoint[]>(() => series.flatMap(({ key, lin, label, color }) => shown.flatMap((f) => {
    const xl = linearBc(f, c), x = lin ? xl : f.hips_bc
    const ref = refById.get(f.id)
    const band = grp && ref ? grp.labelOf(ref as unknown as FilterRow) : null
    return Number.isFinite(x) ? [{ id: f.id, key: `${f.id}/${key}`, x, y: f.ftir!, color: grp ? (band ? grp.colorOf(band) : NO_BAND_COLOR) : color, hollow: lin,
      tip: [`${f.id} · ${label}`, addis ? `${f.season}${f.lot ? ` · lot ${f.lot}` : ''}` : `${siteName}${ref ? ` · ${ref.date}` : ''}`, `${yName} ${fmt(f.ftir, 2)} µg/m³`,
        `HIPS BC ${fmt(f.hips_bc, 2)} → linearised ${fmt(xl, 2)} µg/m³`, `τ ${fmt(f.tau, 2)}`, ...(grp ? [band ?? 'No data'] : [])] }] : []
  })), [shown, c, series, grp, refById, addis, siteName, yName])
  // FTIR EC against HIPS BC is one quantity twice: one square range with the 1:1 line (Addis keeps its 24 µg/m³).
  // PM2.5 is not, so each axis takes its own range.
  const span = (top: number, floor: number): [number, number] => { const step = top > 12 ? 6 : 2; return [0, Math.max(floor, Math.ceil(top / step) * step)] }
  const domain = useMemo<[number, number]>(() => span(Math.max(0, ...points.map(({ x, y }) => (yKey === 'ftir' ? Math.max(x, y) : x))), addis && yKey === 'ftir' ? 24 : 2), [points, addis, yKey])
  const yDomain = useMemo<[number, number]>(() => (yKey === 'ftir' ? domain : span(Math.max(0, ...points.map(({ y }) => y)), 2)), [points, yKey, domain])
  const top = useMemo(() => 1.08 * Math.max(0, ...shown.map((f) => f.ftir!)), [shown])
  const band = useMemo(() => data.ceilings.filter(({ key }) => key === 'lo' || key === 'hi')
    .map(({ tau_inf }) => fitAgainstHips(yFilters, lambda, tau_inf)), [data.ceilings, yFilters, lambda])
  const children = ({ x, y }: ScatterScales) => series.map(({ key, fit, lin, color }) => {
    if (!fit || fit.slope <= 0) return null
    const x0 = Math.max(domain[0], -fit.intercept / fit.slope)
    const x1 = Math.min(domain[1], (Math.min(yDomain[1], top) - fit.intercept) / fit.slope)
    if (x1 < x0) return null
    return <g key={key} pointerEvents="none">
      <line x1={x(x0)} y1={y(fit.slope * x0 + fit.intercept)} x2={x(x1)} y2={y(fit.slope * x1 + fit.intercept)} stroke={color} strokeWidth={2} strokeDasharray={lin ? '5 3' : undefined} />
      <circle cx={x(x0)} cy={y(fit.slope * x0 + fit.intercept)} r={5} fill="none" stroke={color} strokeWidth={2} />
    </g>
  })
  const story = addis && yKey === 'ftir'
  return (
    <ChartFrame id="hl-against-hips" title={story ? 'Stretch the HIPS axis back out and the negative intercept disappears — SPARTAN published FTIR EC against HIPS BC, Addis Ababa'
      : `Stretch the HIPS axis back out — ${yKey === 'ftir' ? 'SPARTAN published FTIR EC' : 'PM2.5 mass'} against HIPS BC, ${siteName}`}
      subtitle={`The ${shown.length} ${siteName.split(',')[0]} filters keep the same ${yName} on the y-axis. In Both, each filter appears at its measured and linearised HIPS BC where available. The Deming fits use λ = ${lambda}${lambda === 1 && addis ? ' (the closure λ is for FTIR EC)' : lambda === 1 ? ' (the closure λ was set for Addis, so it is not reused here)' : ''}, with linearisation at the page’s selected ceiling.${addis ? '' : ` HIPS BC is the public Fabs ÷ MAC ${data.mac}, optical depth from Fabs × sampled volume.`}`}
      provenance={<>{PROVENANCE}{!addis && <> Other sites: <code>spartan_reference.json</code> (public FTIR EC, HIPS Fabs, sampled volume, PM2.5).</>}</>}
      controls={<>
        <SitePicker as="select" {...sitePick.picker} optionLabel={(v) => { const s = sites.find((x) => x.code === v); return s ? `${s.name} (${s.n})` : v }}
          title="The gallery's page site (psite): the other one-site charts on this page follow it" />
        <Segmented label="HIPS" value={view} options={HIPS_VIEWS} onChange={setView} optionLabel={(o) => o === 'measured' ? 'As measured' : o === 'linearised' ? 'Linearised' : 'Both'} />
        {groupingPicker}
      </>}>
      <AxisChoices options={HIPS_Y_AXES} y={[yKey, (v) => setY(v as HipsY)]} noun="measurements">
        <ScatterPlot points={points} xLabel="HIPS BC (µg/m³)" yLabel={yLabel} pick="y" swap="hl-against-hips"
          xDomain={domain} yDomain={yDomain} nice={false} maxSide={SIDE} outside="pin" note={false}
          under={({ x, y }) => <><RefLine orient="h" at={y(0)} from={x(0)} to={x(domain[1])} />
            {yKey === 'ftir' && <IdentityLine x1={x(0)} y1={y(0)} x2={x(domain[1])} y2={y(domain[1])} />}</>}>{children}</ScatterPlot>
      </AxisChoices>
      <Key items={[...series.flatMap(({ label, lin, color }) => [
        { label: `HIPS ${label.toLowerCase()}`, color: grp ? INK.muted : color, kind: lin ? 'hollow' as const : 'dot' as const },
        { label: `${label} Deming fit`, color, kind: lin ? 'dash' as const : 'line' as const },
      ]), ...(grp ? [...grp.labels.map((label) => ({ label, color: grp.colorOf(label), kind: 'dot' as const })), { label: 'No data', color: NO_BAND_COLOR, kind: 'dot' as const }] : []),
      ...(yKey === 'ftir' ? [{ label: '1:1', color: INK.identity, kind: 'dash' as const }] : [])]} />
      {missingNote('drawn in grey')}
      <Readouts items={series.map(({ label, fit, color }) => ({ label, value: eq(fit), color,
        sub: fit ? `Intercept ${signed(fit.intercept, 2)} µg/m³ · n = ${fit.n}` : undefined }))} />
      {story ? <Note explain>Why this happens: when the high end of the x axis is squashed, a straight line through the cloud has to be steep, and a steep line through the middle of the data crosses zero far below the origin. Straighten the axis and the line flattens and passes through the origin. Across the ceiling's 90 % interval the linearised intercept runs from {signed(band[0]?.intercept, 2)} to {signed(band[1]?.intercept, 2)}, which cannot be told apart from zero.</Note>
        : <Note>Across the ceiling's 90 % interval the linearised intercept here runs from {signed(band[0]?.intercept, 2)} to {signed(band[1]?.intercept, 2)} µg/m³.{addis ? '' : ' The ceiling is IMPROVE\'s; the 11 Nov network page checks it against each site\'s own filters.'}</Note>}
      {addis ? <FixedSetNote what="the closure workflow's counted Addis filters with a published FTIR EC" /> : <ExcludedNote count={nOut} />}
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 5. the network check
export function NetworkExponents({ data }: { data: Linearise }) {
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
        maxSide={SIDE} outside="pin" note={false} refY={[1]} under={under}>{children}</ScatterPlot>
      <Key items={[{ label: 'measured, 95 % interval', color: INK.text, kind: 'dot' }, { label: 'predicted from the IMPROVE curve', color: LINEAR, kind: 'hollow' }]} />
      <Note explain>Darker sites fall further below 1, about as the curve predicts. Addis has the darkest filters and the lowest exponent ({fmt(addis?.exponent, 2)} against {fmt(addis?.predicted, 2)} predicted). The two Chilean sites ({chile.map((s) => fmt(s.exponent, 2)).join(', ')} against {fmt(Math.min(...chile.map((s) => s.predicted)), 2)}–{fmt(Math.max(...chile.map((s) => s.predicted)), 2)} predicted) are not explained. The predictions here use the central ceiling and each site's own filters as the 10 Nov note made them; the 11 Nov network page refines them per site (Meeting tab, HIPS across the network).</Note>
      <FixedSetNote what="the closure workflow's saved site exponents (all SPARTAN sites with at least 40 filters)" />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 6. seasons and lots
export function SeasonLot({ data, ceil }: { data: Linearise; ceil: Ceil }) {
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
      provenance={PROVENANCE}>
      <IntervalStrip blocks={[{ key: 's', rows: groups.map((g) => ({
        key: g.key, label: g.label, note: `τ ${fmt(g.tau, 2)} · n ${g.n}`,
        marks: [[{ mid: g.linearised, lo: Math.min(g.linearised, g.measured), hi: Math.max(g.linearised, g.measured), color: LINEAR, r: 6 }, { mid: g.measured, color: MEASURED, r: 6 }]],
        tip: [g.label, `against HIPS as measured ${fmt(g.measured, 2)}`, `against HIPS linearised ${fmt(g.linearised, 2)}`, `median τ ${fmt(g.tau, 2)} · n ${g.n}`],
      })) }]} panels={[{ label: 'Median FTIR EC ÷ HIPS BC', domain: [0, 1.4], format: (v) => fmt(v, 1) }]} labelWidth={130} noteWidth={110} rowHeight={30} />
      <Key items={[{ label: 'against HIPS as measured', color: MEASURED, kind: 'dot' }, { label: 'against HIPS linearised', color: LINEAR, kind: 'dot' }]} />
      <Note explain>Within lot 251, Bega to Kiremt goes from {fmt(bega?.measured, 2)} to {fmt(kiremt?.measured, 2)} ({fmt(100 * swing('measured'), 0)} %) against HIPS as measured, and from {fmt(bega?.linearised, 2)} to {fmt(kiremt?.linearised, 2)} ({fmt(100 * swing('linearised'), 0)} %) against linearised HIPS. Lot 248 still reads 15 to 20 % low, and it does so against gravimetric mass too, so that offset is not a HIPS effect.</Note>
      <FixedSetNote what="the closure workflow's counted Addis filters with a published FTIR EC" />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 7. what the ceiling decides
export function Limits({ data }: { data: Linearise }) {
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
      <Note explain>Noise grows near the ceiling: the correction multiplies measurement error on the heaviest filters, so per-filter linearised values are least reliable exactly where they matter most. Linearised HIPS is still optical, not the true Addis EC: its slope is FTIR level × 10 ÷ true MAC, and the MAC at Addis is unknown. The direct test is HIPS on the incoming Addis TOR filters, including some loaded to τ 0.5–1.5, which would measure the ceiling at Addis itself.</Note>
    </ChartFrame>
  )
}
