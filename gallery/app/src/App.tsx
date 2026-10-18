import { RegressionProvider, RegressionControls, parseEstimators, estimatorParam, type Estimator } from '@/lib/regressionPrefs'
import { startTransition, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useGalleryData } from '@/hooks/useGalleryData'
import { HighlightProvider, baseFilterId, useHighlight, type FilterRecord } from '@/lib/highlight'
import { NO_PMF, PMF_GROUP, PMF_SOURCE, isPmfField, pmfGroups, pmfKeyByDate, pmfLabel, pmfShareFields, withPmf, type PmfScheme } from '@/lib/pmfGroups'
import { EXPLAIN_AVAILABLE, PrefsProvider, usePrefs } from '@/lib/prefs'
import { listParam, listToParam, readParams, writeParams } from '@/lib/url'
import { useShared } from '@/lib/stable'
import { Scatterplot } from '@/charts/correlation/Scatterplot'
import { Correlogram } from '@/charts/correlation/Correlogram'
import { AvailabilityHeatmap } from '@/charts/correlation/AvailabilityHeatmap'
import { DensityHexbin } from '@/charts/correlation/DensityHexbin'
import { ConnectedScatter } from '@/charts/correlation/ConnectedScatter'
import { SpeciesGraph } from '@/charts/correlation/SpeciesGraph'
import { DistributionPanel } from '@/charts/distribution/DistributionPanel'
import { Histogram } from '@/charts/distribution/Histogram'
import { Ridgeline } from '@/charts/distribution/Ridgeline'
import { Timeseries } from '@/charts/evolution/Timeseries'
import { MonthlyBand } from '@/charts/evolution/MonthlyBand'
import { SeasonalClock } from '@/charts/evolution/SeasonalClock'
import { MonthlyHeatmap } from '@/charts/evolution/MonthlyHeatmap'
import { Barplot } from '@/charts/ranking/Barplot'
import { CorrelationRanking } from '@/charts/ranking/CorrelationRanking'
import { SiteRadar } from '@/charts/ranking/SiteRadar'
import { ParallelCoordinates } from '@/charts/ranking/ParallelCoordinates'
import { CompositionTreemap } from '@/charts/partOfWhole/CompositionTreemap'
import { SiteBubbleMap } from '@/charts/map/SiteBubbleMap'
import { MethodSankey } from '@/charts/flow/MethodSankey'
import { SourceStack } from '@/charts/pmf/SourceStack'
import { SourceSeasonality } from '@/charts/pmf/SourceSeasonality'
import { CensusPage } from '@/pages/CensusPage'
import { CalibrationPage } from '@/pages/CalibrationPage'
import { BaselineComparisonPage } from '@/pages/BaselineComparisonPage'
import { SimilarityPage } from '@/pages/SimilarityPage'
import { MeetingFollowupPage } from '@/pages/MeetingFollowupPage'
import { SatoshiFollowupPage } from '@/pages/SatoshiFollowupPage'
import { MeetingSpecPage } from '@/pages/MeetingSpecPage'
import { SPEC_MEETINGS } from '@/meetings'
import { BlanksPage } from '@/pages/BlanksPage'
import { DashboardPage } from '@/pages/DashboardPage'
import { dashParams, parseDash, type DashState, type Scope } from '@/lib/dashboard'
import type { DashFilters } from '@/components/dashboard/DashHeader'
import { Combobox } from '@/components/Combobox'
import { AxisFieldsProvider, type AxisRole } from '@/components/AxisFieldPopover'
import { ViewsMenu } from '@/components/ViewsMenu'
import { currentView, isEmbed, viewUrl } from '@/lib/views'
import { useExport } from '@/lib/data'
import { customSeasonMeta, customSeasonsParam, parseCustomSeasons } from '@/components/SeasonEditor'
import { SideDock, SubsetBar, type SubsetActions, type SubsetCaps, type SubsetState } from '@/components/SubsetBar'
import { SampleDrawer } from '@/components/SampleDrawer'
import { RecordDrawer } from '@/components/RecordDrawer'
import type { DateRange } from '@/components/DateBrush'
import type { FilterRow, SeasonMeta, SiteMeta } from '@/lib/types'
import { AXES_MODES, DEFAULT_AXES, type AxesMode, type AxesOpts } from '@/lib/axes'
import { calendarVariants, hasLocalCalendar, localSeasonsFor, qualifySeason } from '@/siteSeasons'
import { MAX_IMPROVE_SITES, MONTH_QUARTERS, NETWORK_DEFAULTS, hasImprove, hasSpartan, isNetwork, mergeGroups, networkColor, useNetworkData, type Network } from '@/lib/networks'
import type { SiteItem } from '@/components/SitePicker'
import { FilterEditsProvider, HandEditsBadge, type FilterGroup } from '@/lib/filterEdits'
import { GroupDrawer } from '@/components/GroupDrawer'
import { WeatherPage } from '@/pages/WeatherPage'
import { HIDDEN_DUPLICATES } from '@/fieldLineage'
import { PageSidebarProvider, PageSidebarSlot, SidebarControls } from '@/lib/pageSidebar'
import { RATIO_GROUP, RATIO_PRESETS, VOLUME, parseRatios, ratioKey, ratiosParam, viewRows, viewUnits, type Basis, type RatioDef } from '@/lib/viewFields'
import { RatiosProvider } from '@/components/RatioDialog'
import { GlossaryButton } from '@/components/Glossary'
import { PresentationProvider, PresentationSwitch, type Presentation } from '@/lib/presentation'
import { useSearchParam } from '@/lib/url'
import { CommandPaletteButton } from '@/components/CommandPalette'
import { HeaderNavProvider, useHeaderNav } from '@/lib/headerNav'
import { NavMenu, type NavGroup, type NavItem } from '@/components/NavMenu'
import { ExplainIcon, SidebarIcon, TextSizeIcon } from '@/components/Icons'
import { WEATHER_GROUP, type WeatherFile } from '@/lib/weather'
import { useHeaderHeight, useSidePanelOpen } from '@/hooks/useSidePanel'

/**
 * Tabs mirror the react-graph-gallery categories, plus the census. `hidden` tabs
 * stay out of the tab row but still open from a link (#tab=census); the row shows
 * one while it is the open tab.
 */
const TABS = [
  { key: 'census', label: 'Census', hidden: true },
  { key: 'dashboard', label: 'Dashboard', title: 'Your own page of gallery charts: add, arrange and size tiles' },
  { key: 'correlation', label: 'Correlation' },
  { key: 'distribution', label: 'Distribution' },
  { key: 'evolution', label: 'Evolution' },
  { key: 'ranking', label: 'Ranking' },
  { key: 'partOfWhole', label: 'Part of a whole' },
  { key: 'map', label: 'Map' },
  { key: 'weather', label: 'Weather', title: 'Weather over each filter’s sampling window: ERA5 at every SPARTAN site, the airport station at Addis' },
  { key: 'flow', label: 'Flow' },
  { key: 'pmf', label: 'PMF sources' },
  { key: 'calibration', label: 'Calibration' },
  { key: 'baseline', label: 'Baselines', title: 'AIRSpec spline vs VIBES baseline correction' },
  { key: 'similarity', label: 'Spectral similarity' },
  { key: 'meeting', label: 'Meeting', title: 'Meeting follow-ups and Satoshi’s 2 Oct questions' },
  { key: 'blanks', label: 'Blanks', title: 'PTFE field blanks behind the VIBES background' },
] as const

type TabKey = (typeof TABS)[number]['key']
/** A share build (VITE_SHARE_TABS=meeting,map; gallery/share/build_share.py) offers only these tabs, the first as landing tab. */
const SHARE_TABS: string[] = (import.meta.env.VITE_SHARE_TABS ?? '').split(',').map((t: string) => t.trim()).filter(Boolean)
const shared = (key: string) => SHARE_TABS.length === 0 || SHARE_TABS.includes(key)
const isTab = (v: string | null): v is TabKey => !!v && shared(v) && TABS.some((t) => t.key === v)
const LANDING_TAB = (SHARE_TABS.find((t) => TABS.some((x) => x.key === t)) ?? 'correlation') as TabKey

// hand-built meeting pages, then every declared one (src/meetings/*.meeting.ts), newest first
const MEETINGS: { key: string; label: string }[] = [
  { key: 'followup', label: 'Follow up meeting' },
  { key: 'satoshi', label: 'Satoshi 2 Oct' },
  ...SPEC_MEETINGS.map((m) => ({ key: m.id, label: m.label })),
]
type MeetingKey = string
/** the header's page menu (components/NavMenu): tabs in groups, the meeting pages under Meetings */
const tabItem = (key: string): NavItem[] => TABS.filter((t) => t.key === key && shared(t.key))
  .map((t) => ({ key: t.key, label: t.label, title: 'title' in t ? t.title : undefined, tab: t.key }))
const NAV_GROUPS: NavGroup[] = [
  { label: 'Charts', items: ['dashboard', 'correlation', 'distribution', 'evolution', 'ranking', 'partOfWhole', 'flow'].flatMap(tabItem) },
  { label: 'Sites', items: ['map', 'weather', 'pmf'].flatMap(tabItem) },
  { label: 'Calibration', items: ['calibration', 'baseline', 'similarity', 'blanks'].flatMap(tabItem) },
  { label: 'Meetings', items: shared('meeting') ? MEETINGS.map((m) => ({ key: m.key, label: m.label, tab: 'meeting', meeting: m.key })) : [] },
].filter((g) => g.items.length)

/** a row of a site without PMF, which the PMF filter passes through */
const OTHER_SITE = '__other_site'

/** what the ⌘K search offers: the tabs this build shows (lib/headerNav) */
const NAV_TABS = TABS.filter((t) => shared(t.key) && !('hidden' in t && t.hidden)).map((t) => ({ key: t.key as string, label: t.label as string, title: 'title' in t ? t.title as string : undefined }))
// Keep existing #tab=satoshi links working under the shared Meeting tab.
const tabFromParams = (p: URLSearchParams): TabKey =>
  p.get('tab') === 'satoshi' && shared('meeting') ? 'meeting' : isTab(p.get('tab')) ? p.get('tab') as TabKey : LANDING_TAB
const meetingFromParams = (p: URLSearchParams): MeetingKey =>
  p.get('tab') === 'satoshi' ? 'satoshi' : MEETINGS.find((m) => m.key === p.get('meeting'))?.key ?? 'followup'

/**
 * What each tab's underlying data can actually support. The PMF solution is
 * ETAD-only and carries no exclusion flags, so offering four site chips and a
 * 942-filter count there would just be wrong.
 */
/** the starting point for custom bins: northern-hemisphere quarters, renamed and repainted freely */
const DEFAULT_CUSTOM = [{ name: 'Winter', months: [12, 1, 2] }, { name: 'Spring', months: [3, 4, 5] }, { name: 'Summer', months: [6, 7, 8] }, { name: 'Autumn', months: [9, 10, 11] }]
/** cv=CODC:palmira,TWTA:outlook → { CODC: 'palmira', TWTA: 'outlook' } */
const parseVariants = (v: string | null): Record<string, string> =>
  Object.fromEntries((v ?? '').split(',').map((x) => x.split(':')).filter((x) => x.length === 2 && x[0] && x[1]))

/** FTIR EC under each spectral baseline: none is the SPARTAN product in filters.json */
type FtirBaseline = 'none' | 'spline' | 'vibes'
const FTIR_VARIANT = { spline: 'EC (FTIR, spline)', vibes: 'EC (FTIR, VIBES)' } as const

/** a comma-separated id list from the URL (filters dropped or kept by hand) */
const idSet = (v: string | null): Set<string> | null => (v ? new Set(v.split(',').filter(Boolean)) : null)

/** `ssites` in the URL: absent = the four focus sites, * = every SPARTAN site, - = none, else codes */
/** `mac` in the URL: a MAC for HIPS BC other than the export's (null: the export's) */
const parseMac = (v: string | null): number | null => { const n = Number(v); return v && Number.isFinite(n) && n > 0 ? n : null }

const parseSpartanSel = (v: string | null): string[] | 'all' | null =>
  v == null || v === '' ? null : v === '*' ? 'all' : v === '-' ? [] : v.split(',').filter(Boolean)

/** tabs whose charts can draw a whole network (Filters sidebar → network) */
const NETWORK_TABS = ['dashboard', 'correlation', 'distribution', 'evolution', 'ranking', 'partOfWhole', 'weather'] as const
const FILTERS: SubsetCaps = { seasons: true, sites: true, excluded: true, dates: true, fields: 'single', datasetLabel: 'SPARTAN filters' }
const CAPS: Record<Exclude<TabKey, 'census' | 'baseline' | 'similarity' | 'meeting' | 'blanks'>, SubsetCaps> = {
  correlation: { ...FILTERS, fields: 'pair', networks: true, fits: true },
  // each tile picks its own fields
  dashboard: { ...FILTERS, fields: 'none', networks: true, fits: true },
  distribution: { ...FILTERS, networks: true },
  evolution: { ...FILTERS, networks: true },
  ranking: { ...FILTERS, networks: true },
  partOfWhole: { ...FILTERS, fields: 'none', networks: true },
  weather: { ...FILTERS, networks: true, fits: true },
  map: FILTERS,
  flow: { ...FILTERS, fields: 'none' },
  pmf: {
    seasons: true,
    sites: false,
    excluded: false,
    dates: false,
    fields: 'none',
    datasetLabel: 'PMF filters',
    pmf: true,
    note: 'ETAD only · 2023 · no exclusion flags on the factor solution',
  },
  // the calibration tab's per-filter section scores Addis evaluation filters; only
  // the season filter means anything there, and only for those charts
  calibration: {
    fits: true,
    seasons: true,
    sites: false,
    excluded: false,
    dates: false,
    fields: 'none',
    datasetLabel: 'Addis evaluation filters',
    pmf: true,
    note: 'Seasons apply to the per-filter section (crossplot, residuals, dated series), refitted on the chosen filters. The grid charts above are fitted on every season.',
  },
}

// Default pair matches the table in AGENTS.md ("Fabs/MAC vs FTIR EC", OLS
// 1.898 / Deming 2.379 at ETAD), so the app opens agreeing with the repo's
// published numbers.
const DEFAULT_X = 'HIPS BC'
/** public SPARTAN BC where HIPS measured it (gallery/data/export_spartan_reference.py) */
const PUBLIC_BC = 'BC (public HIPS)'
const DEFAULT_Y = 'EC (FTIR)'
const DEFAULT_FIELD = 'EC (FTIR)'
/** the Dashboard's One site scope opens on Addis Ababa */
const DASH_SITE = 'ETAD'

/** rows of the picked sites in their site's colour (a network view's `paint`); other rows, and a
 * row already in its colour, are kept as they are (a row is wide, so a copy is not free) */
function paintRows(rows: FilterRow[], paint: ((site: string) => string | undefined) | undefined): FilterRow[] {
  if (!paint) return rows
  return rows.map((r) => { const c = paint(r.site); return c === undefined || r.color === c ? r : { ...r, color: c } })
}

export default function App() {
  const { data, error } = useGalleryData()
  // spline / VIBES FTIR EC for the Addis test set (gallery/data/export_baseline_ftir_ec.py); optional
  const baselines = useExport<{ rows: Record<string, { spline?: number; vibes?: number }> }>('baseline_ftir_ec.json').data
  // weather over each filter's sampling window (gallery/data/export_weather.py, lib/weather); optional
  const weather = useExport<WeatherFile>('weather/filters.json').data
  const focusRows = useMemo(() => {
    const rows = data?.filters.rows ?? []
    if (!baselines) return rows
    return rows.map((r) => {
      const b = baselines.rows[r.id]
      return b ? { ...r, [FTIR_VARIANT.spline]: b.spline ?? null, [FTIR_VARIANT.vibes]: b.vibes ?? null } : r
    })
  }, [data, baselines])

  // ---- state, seeded from the URL hash so a pasted link reopens the same view
  const init = useMemo(() => readParams(), [])
  const [tab, setTab] = useState<TabKey>(() => tabFromParams(init))
  const [meeting, setMeeting] = useState<MeetingKey>(() => meetingFromParams(init))
  // the ⌘K search's way to a tab or meeting page (lib/headerNav)
  const openTab = useCallback((t: string, m?: string) => { if (isTab(t)) setTab(t); if (m) setMeeting(m) }, [])
  // Mirrors get_clean_data: excluded samples are hidden by default but never
  // deleted, so you can always put them back and see what was removed.
  const [includeExcluded, setIncludeExcluded] = useState(init.get('excl') === '1')
  const [convention, setConvention] = useState<string | null>(init.get('cal'))
  // the reader's own month bins (calendar "custom"), shared by every site; in the URL as cs
  const [customSeasons, setCustomSeasons] = useState(() => parseCustomSeasons(init.get('cs')) ?? DEFAULT_CUSTOM)
  // a site's alternative calendar where the sources disagree (CODC Bogotá / Palmira, Taiwan review / outlook); in the URL as cv=CODC:palmira,…
  const [calVariants, setCalVariants] = useState<Record<string, string>>(() => parseVariants(init.get('cv')))
  const [activeSeasons, setActiveSeasons] = useState<Set<string> | null>(listParam(init.get('seasons')))
  const [activeSites, setActiveSites] = useState<Set<string> | null>(listParam(init.get('sites')))
  // which filters the chart tabs draw (lib/networks): SPARTAN by default, starting on the four focus
  // sites; IMPROVE; or the focus-sites-only view, kept for links (net=focus)
  const [network, setNetwork] = useState<Network>(isNetwork(init.get('net')) ? init.get('net') as Network : 'spartan')
  // SPARTAN sites by code: null = the four focus sites (default), 'all', or an explicit list ([] = none)
  const [spartanSel, setSpartanSel] = useState<string[] | 'all' | null>(() => parseSpartanSel(init.get('ssites')))
  // IMPROVE sites load one file each, so they are an explicit pick (codes), never "all"
  const [improveSites, setImproveSites] = useState<string[]>(() => (init.get('isites') ?? '').split(',').filter(Boolean).slice(0, MAX_IMPROVE_SITES))
  // PMF source filter (PMF and calibration tabs): which grouping, and which groups are kept
  const [pmfScheme, setPmfScheme] = useState<PmfScheme>(init.get('pmfg') === 'class' ? 'Marine vs combustion' : 'Dominant source')
  const [activePmf, setActivePmf] = useState<Set<string> | null>(listParam(init.get('pmf')))
  // each filter dimension can be switched off: seasonality is on by default; PMF source is off,
  // and switching it on keeps only filters that have a PMF day
  // hand edits from a brush, on top of the dataset's exclusions: drop these filters, or keep only these
  const [dropIds, setDropIds] = useState<Set<string>>(() => idSet(init.get('drop')) ?? new Set())
  const [keepIds, setKeepIds] = useState<Set<string> | null>(() => idSet(init.get('keep')))
  // which spectral baseline FTIR EC is shown with (Addis test set only for spline and VIBES)
  const [ftirBaseline, setFtirBaseline] = useState<FtirBaseline>(() => (['spline', 'vibes'] as string[]).includes(init.get('fb') ?? '') ? init.get('fb') as FtirBaseline : 'none')
  const [seasonsOn, setSeasonsOn] = useState(init.get('seas') !== '0')
  const [pmfOn, setPmfOn] = useState(init.get('pmfon') === '1')
  // with the PMF filter on: also hide the sites (and days) without PMF, so only PMF filters remain
  const [pmfOnly, setPmfOnly] = useState(init.get('pmfonly') === '1')
  const [dateRange, setDateRange] = useState<DateRange | null>(
    init.get('from') && init.get('to') ? [init.get('from')!, init.get('to')!] : null
  )
  const [xField, setXField] = useState(init.get('x') ?? DEFAULT_X)
  const [yField, setYField] = useState(init.get('y') ?? DEFAULT_Y)
  const [field, setField] = useState(init.get('f') ?? DEFAULT_FIELD)
  // the Dashboard tab: scope, layout and each scope's tiles (lib/dashboard)
  const [dash, setDash] = useState<DashState>(() => parseDash(init))
  // the reader's view of the measurements (lib/viewFields): own ratios, a MAC for HIPS BC, mass per filter
  const [ratios, setRatios] = useState<RatioDef[]>(() => parseRatios(init.get('ratios')))
  const [mac, setMac] = useState<number | null>(() => parseMac(init.get('mac')))
  const [basis, setBasis] = useState<Basis>(init.get('basis') === 'mass' ? 'mass' : 'conc')
  // meeting pages: as presented (default) or following the gallery (lib/presentation); ?mview=exploring
  const [presentation, setPresentation] = useSearchParam<Presentation>('mview', 'presented', ['presented', 'exploring'])
  const [estimators, setEstimators] = useState<Estimator[]>(() => parseEstimators(init.get('fits')))
  const [showCI, setShowCI] = useState(init.get('ci') === '1')
  const [pinnedId, setPinnedId] = useState<string | null>(init.get('pin'))
  // axes policy shared by every crossplot on the correlation tab
  const [axes, setAxes] = useState<AxesOpts>({
    identity: init.get('eq') !== '0',
    mode: (AXES_MODES as readonly string[]).includes(init.get('ax') ?? '') ? (init.get('ax') as AxesMode) : DEFAULT_AXES.mode,
    log: init.get('log') === '1',
    panels: init.get('pax') === 'own' ? 'Own' : 'Shared',
  })
  // the sample whose detail drawer is open; opening also pins
  const [openId, setOpenId] = useState<string | null>(null)
  // what the clicked chart knows about the open filter; a filter outside filters.json opens from this alone
  const [openRec, setOpenRec] = useState<FilterRecord | null>(null)
  // a bin of filters (a hex, a histogram bar) listed in the side panel
  const [group, setGroup] = useState<FilterGroup | null>(null)
  useEffect(() => setGroup(null), [tab])
  // hand edits from the charts' right-click menu and the side panel; the brush writes the same sets
  const dropFilters = useCallback((ids: Set<string>) => setDropIds((cur) => new Set([...cur, ...ids])), [])
  const keepFilters = useCallback((ids: Set<string>) => setKeepIds((cur) => (cur ? new Set([...ids].filter((id) => cur.has(id))) : new Set(ids))), [])
  // "review" panels list the current hand edits; ones put back stay listed (to undo) until the panel closes
  const shownGroup = useMemo<FilterGroup | null>(() => !group ? null
    : group.live === 'dropped' ? { ...group, ids: [...new Set([...group.ids, ...dropIds])] }
      : group.live === 'kept' ? { ...group, ids: [...new Set([...group.ids, ...(keepIds ?? [])])] } : group, [group, dropIds, keepIds])
  const restoreFilters = useCallback((ids: Set<string>) => setDropIds((cur) => new Set([...cur].filter((id) => !ids.has(id)))), [])
  const dismissSample = useCallback(() => {
    setPinnedId(null)
    setOpenId(null)
    setOpenRec(null)
  }, [])
  const openSample = useCallback((id: string) => {
    setPinnedId(id)
    setOpenId(id)
    setOpenRec(null)
  }, [])

  // x and y never show the same field: picking the other axis's field swaps the two
  const chooseX = useCallback((f: string) => { if (f === yField) setYField(xField); setXField(f) }, [xField, yField])
  const chooseY = useCallback((f: string) => { if (f === xField) setXField(yField); setYField(f) }, [xField, yField])
  // the Dashboard's scope picks the sites: one SPARTAN site, several (the four focus sites to
  // start), or every one; the header's site picker then edits that pick
  const setDashScope = useCallback((scope: Scope) => {
    setDash((d) => (d.scope === scope ? d : { ...d, scope, sheet: 0 }))
    if (scope !== 'sites' && !hasSpartan(network)) setNetwork('spartan')
    if (scope === 'network') setSpartanSel('all')
    if (scope === 'sites') setSpartanSel((sel) => (Array.isArray(sel) && sel.length > 1 ? sel : null))
    if (scope === 'site') setSpartanSel((sel) => (Array.isArray(sel) && sel.length >= 1 ? [sel[0]] : [DASH_SITE]))
  }, [network])
  const pickDashSite = useCallback((code: string) => setSpartanSel([code]), [])
  // a map bubble or a site card opens that site on its own
  const openDashSite = useCallback((code: string) => { setDashScope('site'); setSpartanSel([code]) }, [setDashScope])
  // a pick made in the Filters panel that no longer matches the scope moves it to Compare sites
  useEffect(() => {
    if (tab !== 'dashboard') return
    const one = Array.isArray(spartanSel) && spartanSel.length === 1 && hasSpartan(network)
    if ((dash.scope === 'site' && !one) || (dash.scope === 'network' && (spartanSel !== 'all' || !hasSpartan(network)))) setDash((d) => ({ ...d, scope: 'sites' }))
  }, [tab, dash.scope, spartanSel, network])
  // axis titles on the charts edit the same x / y / measurement as the sidebar (components/AxisFieldPopover)
  const setAxisField = useCallback((role: AxisRole, f: string) => (role === 'x' ? chooseX : role === 'y' ? chooseY : setField)(f), [chooseX, chooseY])
  const axisValues = useMemo(() => ({ x: xField, y: yField, field }), [xField, yField, field])

  useEffect(() => {
    writeParams({
      tab,
      meeting: tab === 'meeting' && meeting !== 'followup' ? meeting : null,
      fits: estimatorParam(estimators),
      ci: showCI ? '1' : null,
      cal: convention,
      cs: convention === 'custom' ? customSeasonsParam(customSeasons) : null,
      cv: Object.keys(calVariants).length ? Object.entries(calVariants).map(([c, k]) => `${c}:${k}`).join(',') : null,
      seasons: listToParam(activeSeasons),
      sites: listToParam(activeSites),
      net: network !== 'spartan' ? network : null,
      ssites: spartanSel === null ? null : spartanSel === 'all' ? '*' : spartanSel.length ? spartanSel.join(',') : '-',
      isites: improveSites.length ? improveSites.join(',') : null,
      pmfg: pmfScheme === 'Marine vs combustion' ? 'class' : null,
      pmf: listToParam(activePmf),
      seas: seasonsOn ? null : '0',
      pmfon: pmfOn ? '1' : null,
      pmfonly: pmfOnly ? '1' : null,
      from: dateRange?.[0],
      to: dateRange?.[1],
      excl: includeExcluded ? '1' : null,
      drop: dropIds.size ? [...dropIds].join(',') : null,
      fb: ftirBaseline !== 'none' ? ftirBaseline : null,
      keep: keepIds ? [...keepIds].join(',') : null,
      x: xField !== DEFAULT_X ? xField : null,
      y: yField !== DEFAULT_Y ? yField : null,
      f: field !== DEFAULT_FIELD ? field : null,
      ...dashParams(dash),
      ratios: ratiosParam(ratios),
      mac: mac !== null ? String(mac) : null,
      basis: basis === 'mass' ? 'mass' : null,
      pin: pinnedId,
      eq: axes.identity ? null : '0',
      ax: axes.mode !== DEFAULT_AXES.mode ? axes.mode : null,
      log: axes.log ? '1' : null,
      pax: axes.panels === 'Own' ? 'own' : null,
    })
  }, [tab, meeting, estimators, showCI, convention, calVariants, activeSeasons, activeSites, network, spartanSel, improveSites, dropIds, keepIds, ftirBaseline, customSeasons, pmfScheme, activePmf, seasonsOn, pmfOn, pmfOnly, dateRange, includeExcluded, xField, yField, field, dash, ratios, mac, basis, pinnedId, axes])

  // A pasted link, the back button, or an edited hash should all apply —
  // not just the hash that was there on first load.
  useEffect(() => {
    const onHash = () => {
      const p = readParams()
      setEstimators(parseEstimators(p.get('fits')))
      setShowCI(p.get('ci') === '1')
      if (isTab(p.get('tab')) || p.get('tab') === 'satoshi') {
        setTab(tabFromParams(p))
        if (tabFromParams(p) === 'meeting') setMeeting(meetingFromParams(p))
      }
      setConvention(p.get('cal'))
      setCustomSeasons(parseCustomSeasons(p.get('cs')) ?? DEFAULT_CUSTOM)
      setCalVariants(parseVariants(p.get('cv')))
      setActiveSeasons(listParam(p.get('seasons')))
      setActiveSites(listParam(p.get('sites')))
      setNetwork(isNetwork(p.get('net')) ? p.get('net') as Network : 'spartan')
      setSpartanSel(parseSpartanSel(p.get('ssites')))
      setImproveSites((p.get('isites') ?? '').split(',').filter(Boolean).slice(0, MAX_IMPROVE_SITES))
      setPmfScheme(p.get('pmfg') === 'class' ? 'Marine vs combustion' : 'Dominant source')
      setActivePmf(listParam(p.get('pmf')))
      setSeasonsOn(p.get('seas') !== '0')
      setPmfOn(p.get('pmfon') === '1')
      setPmfOnly(p.get('pmfonly') === '1')
      setDateRange(p.get('from') && p.get('to') ? [p.get('from')!, p.get('to')!] : null)
      setIncludeExcluded(p.get('excl') === '1')
      setDropIds(idSet(p.get('drop')) ?? new Set())
      setFtirBaseline((['spline', 'vibes'] as string[]).includes(p.get('fb') ?? '') ? p.get('fb') as FtirBaseline : 'none')
      setKeepIds(idSet(p.get('keep')))
      setXField(p.get('x') ?? DEFAULT_X)
      setYField(p.get('y') ?? DEFAULT_Y)
      setField(p.get('f') ?? DEFAULT_FIELD)
      setDash(parseDash(p))
      setRatios(parseRatios(p.get('ratios')))
      setMac(parseMac(p.get('mac')))
      setBasis(p.get('basis') === 'mass' ? 'mass' : 'conc')
      setPinnedId(p.get('pin'))
      setAxes({
        identity: p.get('eq') !== '0',
        mode: (AXES_MODES as readonly string[]).includes(p.get('ax') ?? '') ? (p.get('ax') as AxesMode) : DEFAULT_AXES.mode,
        log: p.get('log') === '1',
        panels: p.get('pax') === 'own' ? 'Own' : 'Shared',
      })
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  // The network only applies on the chart tabs; everything else always shows the focus sites.
  const net: Network = (NETWORK_TABS as readonly string[]).includes(tab) ? network : 'focus'
  const netData = useNetworkData(net, improveSites)
  // the chosen network's sites, rows and fields: independent of which sites are picked, so a site
  // pick does not rebuild (and re-join weather, PMF, units and seasons onto) every row
  const netBase = useMemo(() => {
    if (net === 'focus' || !data) return null
    const d = netData.data
    if (!d) return { sites: [] as SiteMeta[], focusCodes: [] as string[], spartanCodes: [] as string[], improveCodes: new Set<string>(), rows: [] as FilterRow[], fields: [] as string[], field_units: {} as Record<string, string>, field_groups: [] as { label: string; fields: string[] }[] }
    const sp = hasSpartan(net)
    // SPARTAN: the four focus sites keep their filters.json rows (HIPS, FTIR groups, exclusions,
    // local calendars) and their names and colours; the other sites bring the public ChemSpec rows
    const focus = sp ? new Map(data.meta.sites.map((x) => [x.code, x])) : new Map<string, SiteMeta>()
    const sites = d.sites.map((x) => focus.get(x.code) ?? x)
    for (const f of focus.values()) if (!sites.some((x) => x.code === f.code)) sites.push(f)
    const spartanCodes = sites.filter((x) => !d.improveCodes.has(x.code)).map((x) => x.code)
    // the focus sites also get the public fields filters.json does not carry (public HIPS BC, the trace
    // elements, the rest of the ions), so every axis spans every SPARTAN site; their own fields stay theirs
    const pub = new Map(d.rows.filter((r) => focus.has(r.code)).map((r) => [r.id, r]))
    const focusWithPub = focusRows.map((r): FilterRow => {
      const p = pub.get(r.id)
      if (!p) return r
      const add = Object.fromEntries(d.fields.filter((f) => !(f in r) && typeof p[f] === 'number').map((f) => [f, p[f]]))
      return Object.keys(add).length ? { ...r, ...add } : r
    })
    const rows: FilterRow[] = [...(sp ? focusWithPub : []), ...d.rows.filter((r) => !focus.has(r.code))]
    const fields = sp ? [...new Set([...data.meta.fields, ...d.fields])] : d.fields
    const groups = sp ? mergeGroups(data.meta.field_groups, d.field_groups) : d.field_groups
    return { sites, focusCodes: [...focus.keys()], spartanCodes, improveCodes: d.improveCodes, rows, fields,
      field_units: sp ? { ...d.field_units, ...data.meta.field_units } : d.field_units, field_groups: groups }
  }, [net, data, focusRows, netData.data])
  // the pick on top: the chosen sites, coloured in selection order (`paint` gives a row its site's colour)
  const netView = useMemo(() => {
    if (!netBase) return null
    const { sites, focusCodes, spartanCodes, rows, fields } = netBase
    const sp = hasSpartan(net), im = hasImprove(net)
    const chosenCodes = new Set([
      ...(sp ? (spartanSel === null ? focusCodes : spartanSel === 'all' ? spartanCodes : spartanSel) : []),
      ...(im ? improveSites : []),
    ])
    const focusColour = new Map((data?.meta.sites ?? []).map((x) => [x.code, x.color]))
    let i = 0
    const colour = new Map(sites.filter((x) => chosenCodes.has(x.code)).map((x) => [x.name, (sp && focusColour.get(x.code)) || networkColor(i++)]))
    const paint = (name: string) => colour.get(name)
    const all = sites.map((x) => ({ ...x, color: paint(x.name) ?? '#9aa5b1' }))
    const chosenNames = new Set(all.filter((x) => chosenCodes.has(x.code)).map((x) => x.name))
    // counts in the field pickers follow the exclusion toggle, like every chart (Addis: 189, not 190)
    const inChosen = rows.filter((r) => chosenNames.has(r.site) && (includeExcluded || !r.excluded))
    const coverage = Object.fromEntries(fields.map((f) => [f, inChosen.filter((r) => typeof r[f] === 'number').length]))
    return { ...netBase, all, chosen: all.filter((x) => chosenCodes.has(x.code)), chosenCodes, chosenNames, nSpartan: spartanCodes.length, paint, coverage }
  }, [netBase, net, data, improveSites, spartanSel, includeExcluded])
  // the ChemSpec restatements of the FTIR carbon are never offered; a link naming one falls back
  const activeFields = (netView ? netView.fields : data?.meta.fields ?? []).filter((f) => !HIDDEN_DUPLICATES.includes(f))
  // ratios whose two fields this view has: the presets, then the reader's own (lib/viewFields)
  const activeKey = activeFields.join('|')
  const ratioView = useMemo(() => {
    const has = new Set([...activeFields, ...(weather?.fields ?? []), ...pmfShareFields(data?.pmf ?? null)])
    const fits = (d: RatioDef) => has.has(d.num) && has.has(d.den)
    const presets = RATIO_PRESETS.filter(fits)
    const own = ratios.filter((d) => fits(d) && !presets.some((x) => ratioKey(x) === ratioKey(d)))
    return { presets, all: [...presets, ...own] }
  }, [activeKey, weather, ratios, data]) // eslint-disable-line react-hooks/exhaustive-deps
  const ratioFields = useMemo(() => ratioView.all.map(ratioKey), [ratioView])

  // Fields that vanished from the export, or that the chosen network does not measure,
  // fall back to that network's defaults instead of leaving a chart empty.
  useEffect(() => {
    if (!data || (netView && !netView.fields.length)) return
    // weather fields (lib/weather) load on their own, after this first check, so they are known by name
    const ok = (f: string) => activeFields.includes(f) || ratioFields.includes(f) || isPmfField(f) || /\((ERA5|station)\)$/.test(f)
    const d = NETWORK_DEFAULTS[net]
    if (!ok(xField)) setXField(ok(d.x) ? d.x : activeFields[0])
    if (!ok(yField)) setYField(ok(d.y) ? d.y : activeFields[1] ?? activeFields[0])
    if (!ok(field)) setField(ok(d.field) ? d.field : activeFields[0])
  }, [data, netView, activeFields, ratioFields, net, xField, yField, field])

  // Three SPARTAN sites (Kanpur, Ilorin, Singapore) have the public HIPS BC but nothing in HIPS
  // batches 1–51. When a newly chosen site would drop out of an axis for that reason, the axis moves
  // to the public field instead of leaving that site's panels empty. Only when the choice of sites
  // changes, and not over a field the link named, so HIPS BC can still be picked by hand.
  const seenSites = useRef<string | null>(null)
  const linkNamedField = useRef(!!(init.get('x') || init.get('y') || init.get('f')))
  // the axes this moved to public BC: when no chosen site needs it any more they go back to HIPS BC,
  // so a pick of every site and then the focus sites again does not leave public BC behind
  const fellBack = useRef(new Set<'x' | 'y' | 'f'>())
  useEffect(() => {
    if (!netView || !hasSpartan(net) || !netView.rows.length) return
    const key = [...netView.chosenCodes].sort().join(',')
    if (key === seenSites.current) return
    seenSites.current = key
    if (linkNamedField.current) { linkNamedField.current = false; return }
    const missing = netView.fields.includes(PUBLIC_BC) && netView.chosen.some((x) => {
      const own = netView.rows.filter((r) => r.site === x.name)
      return own.length && !own.some((r) => typeof r['HIPS BC'] === 'number') && own.some((r) => typeof r[PUBLIC_BC] === 'number')
    })
    if (!missing) {
      if (fellBack.current.has('x') && xField === PUBLIC_BC) setXField('HIPS BC')
      if (fellBack.current.has('y') && yField === PUBLIC_BC) setYField('HIPS BC')
      if (fellBack.current.has('f') && field === PUBLIC_BC) setField('HIPS BC')
      fellBack.current.clear()
      return
    }
    if (xField === 'HIPS BC') { setXField(PUBLIC_BC); fellBack.current.add('x') }
    if (yField === 'HIPS BC') { setYField(PUBLIC_BC); fellBack.current.add('y') }
    if (field === 'HIPS BC') { setField(PUBLIC_BC); fellBack.current.add('f') }
  }, [netView, net, xField, yField, field])

  const siteNames = useMemo(
    () => (netView ? (net === 'improve' ? netView.chosen : netView.all).map((s) => s.name) : data?.meta.sites.map((s) => s.name) ?? []),
    [data, netView, net])
  // old links named focus sites in `sites`; under SPARTAN that becomes a code pick
  useEffect(() => {
    if (!data || network !== 'spartan' || !activeSites || spartanSel !== null) return
    setSpartanSel(data.meta.sites.filter((x) => activeSites.has(x.name)).map((x) => x.code))
    setActiveSites(null)
  }, [data, network, activeSites, spartanSel])
  const pmfSiteName = data?.pmf?.site_name ?? 'Addis Ababa'
  // Season calendar. 'local' (the default) bins every filter by its own site's
  // calendar (docs/site-seasonality.md): Beijing's four seasons, Delhi's monsoon,
  // JPL's wet/dry windows, Addis's Bega/Belg/Kiremt. The shared Ethiopian month
  // bins stay available for like-month comparisons across sites.
  // every SPARTAN site and the two Santiago IMPROVE sites have their own calendar (siteSeasons.network.json)
  const conventions = useMemo<Record<string, SeasonMeta[]>>(() => ({ ...(data?.meta.season_conventions ?? {}), quarters: MONTH_QUARTERS, custom: customSeasonMeta(customSeasons) }), [data, customSeasons])
  // local calendars are the default wherever every site in view has one; otherwise shared quarters
  const siteCode = useMemo(() => new Map((netView?.all ?? []).map((x) => [x.name, x.code])), [netView])
  const allLocal = net === 'focus' || (!!netView && netView.chosen.length > 0 && netView.chosen.every((x) => hasLocalCalendar(x.name, x.code)))
  const defaultConvention = allLocal ? 'local' : 'quarters'
  const activeConvention = convention ?? defaultConvention
  const isLocal = activeConvention === 'local'
  // PMF is Addis-only, so its season names need no site prefix; the filter data
  // always carries four calendars, so labels are site-qualified and never collide
  // PMF and the calibration per-filter section are Addis-only: one calendar, bare season names
  const addisOnlyTab = tab === 'pmf' || tab === 'calibration'
  const qualify = !addisOnlyTab
  const calendarSites = useMemo(
    () => (addisOnlyTab ? [pmfSiteName] : netView ? netView.chosen.map((x) => x.name) : activeSites ? siteNames.filter((n) => activeSites.has(n)) : siteNames),
    [addisOnlyTab, pmfSiteName, activeSites, siteNames, netView]
  )
  const sharedSeasons = useMemo(
    () => conventions[isLocal ? data?.meta.season_convention ?? 'quarters' : activeConvention] ?? data?.meta.seasons ?? [],
    [data, conventions, isLocal, activeConvention]
  )
  const calendarOf = useCallback(
    // sites without a local calendar fall back to the shared quarters, not the Ethiopian bins
    (site: string) => {
      if (!isLocal) return sharedSeasons
      const code = siteCode.get(site)
      return localSeasonsFor(site, data?.meta, code, code ? calVariants[code] : undefined) ?? MONTH_QUARTERS
    },
    [isLocal, data, sharedSeasons, siteCode, calVariants]
  )
  const seasons = useMemo(
    () => isLocal
      ? calendarSites.flatMap((site) => calendarOf(site).map((se) => ({ ...se, site, name: qualifySeason(site, se.name, qualify) })))
      : sharedSeasons,
    [isLocal, calendarSites, calendarOf, qualify, sharedSeasons]
  )

  // month -> season for one site under the *currently selected* calendar, so
  // switching calendar re-labels every chart rather than needing a re-export
  const seasonOf = useCallback((site: string, month: number | null | undefined, fallback: string) => {
    if (!month) return fallback
    const se = calendarOf(site).find((x) => x.months.includes(month))
    return se ? (isLocal ? qualifySeason(site, se.name, qualify) : se.name) : fallback
  }, [calendarOf, isLocal, qualify])
  const seasonForFilter = useCallback((r: FilterRow) => seasonOf(r.site, r.month, r.season), [seasonOf])

  const inRange = useCallback(
    (r: { date: string }) => !dateRange || (r.date >= dateRange[0] && r.date <= dateRange[1]),
    [dateRange]
  )

  // Rows after the exclusion toggle and calendar relabel — the brush draws these.
  const rawSourceRows = useMemo(() => {
    const rows = netBase ? netBase.rows : focusRows
    // the weather fields ride on every row, so every chart and picker can use them
    const withWeather = !weather ? rows : rows.map((r) => {
      const w = weather.rows[r.id]
      return w ? { ...r, ...Object.fromEntries(weather.fields.map((f, i) => [f, w[i]])) } : r
    })
    // so do the PMF source shares and dominant source, at the site the PMF solution covers (lib/pmfGroups)
    return withPmf(withWeather, data?.pmf ?? null, pmfScheme)
  }, [netBase, focusRows, weather, data, pmfScheme])
  // the reader's view (lib/viewFields): MAC, mass per filter, ratios; the drawer keeps the exported values
  const exportMac = data?.meta.mac_value ?? 10
  const viewOpts = useMemo(() => ({
    mac: mac ?? exportMac, exportMac, basis, ratios: ratioView.all,
    units: { ...(netBase ? netBase.field_units : data?.meta.field_units ?? {}), ...(weather?.field_units ?? {}), [FTIR_VARIANT.spline]: 'ug/m3', [FTIR_VARIANT.vibes]: 'ug/m3' },
  }), [mac, exportMac, basis, ratioView, netBase, data, weather])
  const plainSourceRows = useMemo(() => viewRows(rawSourceRows, viewOpts), [rawSourceRows, viewOpts])
  // the site colours go on last, so a site pick repaints the rows without rebuilding them; the
  // lookups, counts and drawers (colours from meta.sites) take the rows unpainted
  const paint = netView?.paint
  const sourceRows = plainSourceRows
  const paintedSourceRows = useMemo(() => (tab === 'meeting' || tab === 'calibration' || tab === 'dashboard' ? paintRows(sourceRows, paint) : sourceRows), [tab, sourceRows, paint])
  // a meeting page as presented: the reader's ratios, but MAC as exported and µg/m³
  const presentOpts = useMemo(() => ({ ...viewOpts, mac: exportMac, basis: 'conc' as const }), [viewOpts, exportMac])
  const presentRows = useMemo(() => (tab === 'meeting' && presentation === 'presented' ? paintRows(viewRows(rawSourceRows, presentOpts), paint) : paintedSourceRows),
    [tab, presentation, rawSourceRows, presentOpts, paintedSourceRows, paint])
  const sourceById = useMemo(() => new Map(sourceRows.map((r) => [r.id, r])), [sourceRows])
  const plainBaseRows = useMemo(() => {
    if (!data) return []
    return plainSourceRows
      .filter((r) => includeExcluded || !r.excluded)
      .map((r) => {
        const s = seasonForFilter(r)
        return s !== r.season ? { ...r, season: s } : r
      })
  }, [data, plainSourceRows, includeExcluded, seasonForFilter])
  const baseRows = useMemo(() => paintRows(plainBaseRows, paint), [plainBaseRows, paint])
  // how many filters in view carry each ratio, for the pickers' counts and the dialog
  const inViewRows = useMemo(() => (netView ? baseRows.filter((r) => netView.chosenNames.has(r.site)) : baseRows), [baseRows, netView])
  const ratioCoverage = useMemo(() => Object.fromEntries(ratioFields.map((f) => [f, inViewRows.filter((r) => typeof r[f] === 'number').length])), [ratioFields, inViewRows])
  const countRatio = useCallback((d: RatioDef) =>
    inViewRows.filter((r) => typeof r[d.num] === 'number' && typeof r[d.den] === 'number' && r[d.den] !== 0).length, [inViewRows])
  const addRatio = useCallback((d: RatioDef) => setRatios((cur) => (cur.some((x) => ratioKey(x) === ratioKey(d)) || RATIO_PRESETS.some((x) => ratioKey(x) === ratioKey(d)) ? cur : [...cur, d])), [])
  const removeRatio = useCallback((f: string) => setRatios((cur) => cur.filter((x) => ratioKey(x) !== f)), [])

  // The season selection, kept only for seasons that exist under the current
  // calendar and sites; a selection that no longer names any season means "all".
  const seasonSel = useMemo(() => {
    if (!seasonsOn || !activeSeasons) return null
    const names = new Set(seasons.map((se) => se.name))
    const kept = new Set([...activeSeasons].filter((n) => names.has(n)))
    return kept.size === 0 || kept.size === names.size ? null : kept
  }, [seasonsOn, activeSeasons, seasons])

  // PMF groups under the chosen scheme; the calibration tab also offers "No PMF day",
  // since most Addis evaluation filters fall outside the 2023 PMF run
  const pmfGroupList = useMemo(() => pmfGroups(data?.pmf ?? null, pmfScheme, false), [data, pmfScheme])
  const pmfSel = useMemo(() => {
    if (!activePmf) return null
    const names = new Set(pmfGroupList.map((g) => g.name))
    const kept = new Set([...activePmf].filter((n) => names.has(n)))
    return kept.size === 0 || kept.size === names.size ? null : kept
  }, [activePmf, pmfGroupList])
  // the groups a row must fall in while the PMF filter is on (every source when none is narrowed);
  // "No PMF day" is never among them, so switching the filter on keeps only PMF-dated filters
  const pmfKeep = useMemo(
    () => (pmfOn ? pmfSel ?? new Set(pmfGroupList.map((g) => g.name)) : null),
    [pmfOn, pmfSel, pmfGroupList]
  )
  // on the chart tabs the PMF filter narrows only the PMF site's filters (by their dominant source);
  // other sites pass through untouched
  const pmfSite = data?.pmf?.site ?? null
  const chartPmf = (NETWORK_TABS as readonly string[]).includes(tab) && !!pmfKeep
  const inChartPmf = useCallback((r: FilterRow) => !chartPmf || (r.code !== pmfSite ? !pmfOnly : pmfKeep!.has(String(r[PMF_SOURCE] ?? NO_PMF))),
    [chartPmf, pmfSite, pmfKeep, pmfOnly])
  // a site with PMF is among the picked ones: the chart tabs then offer the PMF section
  const pmfSiteInView = useMemo(() => !!pmfSite && baseRows.some((r) => r.code === pmfSite && (netView ? netView.chosenNames.has(r.site) : !activeSites || activeSites.has(r.site))),
    [pmfSite, baseRows, netView, activeSites])

  // what the date brush draws: every filter except the date cut, so its range follows the picked sites
  const brushRows = useMemo(
    () => baseRows
      .filter((r) => !seasonSel || seasonSel.has(r.season))
      .filter((r) => (netView ? netView.chosenNames.has(r.site) : !activeSites || activeSites.has(r.site)))
      .filter(inChartPmf)
      .filter((r) => !dropIds.has(r.id) && (!keepIds || keepIds.has(r.id))),
    [baseRows, seasonSel, activeSites, netView, dropIds, keepIds, inChartPmf]
  )
  const rows = useMemo(
    () =>
      baseRows
        .filter(inRange)
        .filter((r) => !seasonSel || seasonSel.has(r.season))
        .filter((r) => (netView ? netView.chosenNames.has(r.site) : !activeSites || activeSites.has(r.site)))
        .filter(inChartPmf)
        .filter((r) => !dropIds.has(r.id) && (!keepIds || keepIds.has(r.id))),
    [baseRows, inRange, seasonSel, activeSites, netView, dropIds, keepIds, inChartPmf]
  )

  // the Weather tab's site table can narrow the Filters panel's pick to one site (and put it back)
  const setSpartanPick = useCallback((codes: string[]) => setSpartanSel(codes), [])

  // The map has its own site scope; keep date, season and exclusion choices, hand edits included.
  const mapRows = useMemo(
    () => baseRows.filter(inRange).filter((r) => !seasonSel || seasonSel.has(r.season))
      .filter((r) => !dropIds.has(r.id) && (!keepIds || keepIds.has(r.id))),
    [baseRows, inRange, seasonSel, dropIds, keepIds]
  )

  // PMF rows get the same calendar + season treatment as the filter rows, so
  // one season selection means the same thing on every tab.
  const pmfByDate = useMemo(() => pmfKeyByDate(data?.pmf ?? null), [data])
  const pmfLabelOfDate = useCallback(
    (d: string | null | undefined) => pmfLabel(data?.pmf ?? null, d ? pmfByDate.get(d.slice(0, 10)) : null, pmfScheme),
    [data, pmfByDate, pmfScheme]
  )

  const pmfRows = useMemo(() => {
    const src = data?.pmf?.rows ?? []
    return src
      .map((r) => ({ ...r, season: seasonOf(pmfSiteName, r.month, r.season) }))
      .filter((r) => !seasonSel || seasonSel.has(r.season))
      .filter((r) => !pmfKeep || pmfKeep.has(pmfLabel(data?.pmf ?? null, r.dominant_source, pmfScheme)))
  }, [data, seasonOf, pmfSiteName, seasonSel, pmfKeep, pmfScheme])

  const pinned: FilterRow | null = useMemo(
    () => {
      const raw = pinnedId && data ? sourceRows.find((r) => r.id === pinnedId) ?? data.filters.rows.find((r) => r.id === pinnedId) : null
      return raw ? { ...raw, season: seasonForFilter(raw) } : null
    },
    [pinnedId, data, sourceRows, seasonForFilter]
  )
  // filter ids the gallery knows, so calibration charts can open the drawer for Addis points
  const knownIds = useMemo(() => new Set([...(data?.filters.rows ?? []), ...sourceRows].map((r) => r.id)), [data, sourceRows])
  // sampling dates of the Addis evaluation filters the calibration runs score (the same set for every preset)
  const addisEvalDates = useMemo(
    () => data?.calibrationRuns?.runs.find((r) => r.target === 'addis')?.eval.date ?? [],
    [data]
  )
  const openRecord = useCallback((rec: FilterRecord) => {
    const id = baseFilterId(rec.id)
    if (knownIds.has(id)) {
      setPinnedId(id)
      setOpenId(id)
    } else {
      setPinnedId(id)
      setOpenId(null)
    }
    setOpenRec(rec)
  }, [knownIds])
  const openRow: FilterRow | null = useMemo(
    () => {
      const raw = openId && data ? rawSourceRows.find((r) => r.id === openId) ?? data.filters.rows.find((r) => r.id === openId) : null
      return raw ? { ...raw, season: seasonForFilter(raw) } : null
    },
    [openId, data, rawSourceRows, seasonForFilter]
  )

  // the charts' meta is rebuilt below on every render; shared with the last one, its unchanged
  // parts keep their identity and the charts' memos hold (lib/stable)
  const stable = useShared()
  if (error) {
    return (
      <Shell>
        <p style={{ color: 'var(--danger)', fontSize: 14 }}>
          Could not load the exported data: {error}
          <br />
          Run <code>python gallery/data/export_data.py</code> from the repo root, then reload.
        </p>
      </Shell>
    )
  }
  if (!data) return <Shell><p style={{ color: 'var(--ink-muted)' }}>Loading…</p></Shell>

  const { filters, meta: rawMeta, census, pmf } = data
  // Charts read meta.seasons for colours and ordering; hand them the active
  // calendar so the legends match what the subset bar says is in force.
  let meta = { ...rawMeta, season_conventions: conventions, seasons, season_convention: activeConvention,
    season_site: isLocal && calendarSites.length === 1 ? calendarSites[0] : undefined,
    // a network view hands the charts its own sites (the chosen ones), fields and units
    ...(netView ? { sites: netView.chosen, fields: netView.fields, field_units: netView.field_units, field_groups: netView.field_groups, coverage: netView.coverage } : {}) }
  // the ChemSpec carbon restates EC / OC (FTIR): keep the values in the rows, out of every picker
  meta.fields = meta.fields.filter((f) => !HIDDEN_DUPLICATES.includes(f))
  meta.field_groups = (meta.field_groups ?? []).map((g) => ({ ...g, fields: g.fields.filter((f) => !HIDDEN_DUPLICATES.includes(f)) })).filter((g) => g.fields.length)
  // the weather fields join the pickers as their own group, after the measured ones
  if (weather) {
    const present = weather.fields.filter((f) => sourceRows.some((r) => typeof r[f] === 'number'))
    meta.fields = [...meta.fields, ...present.filter((f) => !meta.fields.includes(f))]
    meta.field_units = { ...meta.field_units, ...weather.field_units }
    meta.field_groups = [...(meta.field_groups ?? []).filter((g) => g.label !== WEATHER_GROUP), ...(present.length ? [{ label: WEATHER_GROUP, fields: present }] : [])]
  }
  // the PMF source shares join the pickers as their own group where a site in view has PMF (lib/pmfGroups)
  const pmfPresent = pmfShareFields(pmf ?? null).filter((f) => sourceRows.some((r) => typeof r[f] === 'number'))
  if (pmfPresent.length) {
    meta.fields = [...meta.fields, ...pmfPresent.filter((f) => !meta.fields.includes(f))]
    meta.field_units = { ...meta.field_units, ...Object.fromEntries(pmfPresent.map((f) => [f, 'share'])) }
    meta.field_groups = [...(meta.field_groups ?? []).filter((g) => g.label !== PMF_GROUP), { label: PMF_GROUP, fields: pmfPresent }]
    meta.pmf_groups = pmfGroups(pmf ?? null, pmfScheme, true)
    meta.pmf_sources = (pmf?.sources ?? []).map((x) => ({ name: x.label, color: x.color }))
  }
  // the baseline variants are drawn but never offered in the pickers
  meta.field_units = { ...meta.field_units, [FTIR_VARIANT.spline]: 'ug/m3', [FTIR_VARIANT.vibes]: 'ug/m3' }
  // the drawer compares a filter with exported reference populations, so it keeps the exported units
  const drawerMeta = stable('drawerMeta', { ...meta })
  // the reader's view (lib/viewFields): ratios join the pickers; µg/m³ becomes µg/filter in mass mode
  meta.fields = [...meta.fields, ...ratioFields.filter((f) => !meta.fields.includes(f))]
  meta.field_groups = [...meta.field_groups.filter((g) => g.label !== RATIO_GROUP), ...(ratioFields.length ? [{ label: RATIO_GROUP, fields: ratioFields }] : [])]
  meta.field_units = viewUnits({ ...viewOpts, units: meta.field_units })
  meta.coverage = { ...meta.coverage, ...ratioCoverage }
  meta.mac_value = viewOpts.mac
  meta.basis = basis
  meta = stable('meta', meta)
  const presentMeta = presentation === 'presented'
    ? stable('presentMeta', { ...meta, field_units: viewUnits({ ...presentOpts, units: drawerMeta.field_units }), mac_value: exportMac, basis: 'conc' as const })
    : meta
  const eff = (f: string) => (f === 'EC (FTIR)' && ftirBaseline !== 'none' ? FTIR_VARIANT[ftirBaseline] : f)
  const xEff = eff(xField), yEff = eff(yField), fieldEff = eff(field)

  const nav = (
    <nav className="nav" aria-label="Pages">
      <NavMenu groups={NAV_GROUPS} tab={tab} meeting={meeting} onPick={(i) => openTab(i.tab, i.meeting)} extra={<DisplayToggles />} />
    </nav>
  )

  // the Dashboard's header draws the common filters from the same state and actions (components/dashboard/DashHeader)
  let dashFilters: DashFilters | null = null
  const subset = (() => {
    if (tab === 'census' || tab === 'baseline' || tab === 'similarity' || tab === 'meeting' || tab === 'blanks') return null
    const baseCaps = net === 'focus' ? CAPS[tab] : { ...CAPS[tab], datasetLabel: net === 'improve' ? 'IMPROVE samples' : net === 'both' ? 'SPARTAN filters + IMPROVE samples' : 'SPARTAN filters', excluded: sourceRows.some((r) => r.excluded) }
    // the chart tabs offer the PMF section while a site with PMF is picked
    const chartTab = (NETWORK_TABS as readonly string[]).includes(tab)
    const caps = chartTab && pmfSiteInView ? { ...baseCaps, pmf: true } : baseCaps
    const pmfMode = tab === 'pmf'
    const calMode = tab === 'calibration'
    // Counts are computed *before* the dimension's own filter is applied,
    // so a chip always shows what selecting it would give you.
    const pool: { season: string; site: string; pmf: string }[] = pmfMode
      ? (data.pmf?.rows ?? []).map((r) => ({ season: seasonOf(pmfSiteName, r.month, r.season), site: pmfSiteName, pmf: pmfLabel(data.pmf, r.dominant_source, pmfScheme) }))
      : calMode
        ? addisEvalDates.map((d) => ({ season: seasonOf(pmfSiteName, d ? Number(d.slice(5, 7)) : null, 'undated'), site: pmfSiteName, pmf: pmfLabelOfDate(d) }))
        // hand exclusions count like any other filter, so the chips add up to what the charts draw
        : baseRows.filter(inRange).filter((r) => !dropIds.has(r.id) && (!keepIds || keepIds.has(r.id)))
          .map((r) => ({ season: r.season, site: r.site, pmf: r.code === pmfSite ? String(r[PMF_SOURCE] ?? NO_PMF) : OTHER_SITE }))
    const usePmf = pmfMode || calMode || (chartTab && pmfSiteInView)
    const inSeason = (r: { season: string }) => !seasonSel || seasonSel.has(r.season)
    const inPmf = (r: { pmf: string }) => !usePmf || !pmfKeep || (r.pmf === OTHER_SITE ? !pmfOnly : pmfKeep.has(r.pmf))
    // each dimension's chips count what selecting them would give under the *other* filters
    const seasonPool = pool.filter(inPmf)
    const sitePool = seasonPool.filter(inSeason)
    const pmfCounts: Record<string, number> = Object.fromEntries(pmfGroupList.map((g) => [g.name, 0]))
    for (const r of pool.filter(inSeason)) if (r.pmf in pmfCounts) pmfCounts[r.pmf] += 1

    const seasonCounts: Record<string, number> = {}
    for (const s of seasons) seasonCounts[s.name] = 0
    // count only the picked sites (a network view keeps every site's rows for the site list)
    const pickedSite = (site: string) => (netView ? netView.chosenNames.has(site) : !activeSites || activeSites.has(site))
    for (const r of seasonPool.filter((r) => pmfMode || calMode || pickedSite(r.site))) {
      if (r.season in seasonCounts) seasonCounts[r.season] += 1
    }
    const siteCounts: Record<string, number> = {}
    for (const n of siteNames) siteCounts[n] = 0
    for (const r of sitePool) if (r.site in siteCounts) siteCounts[r.site] += 1
    // the map and list picker: focus and SPARTAN sites are keyed by name, IMPROVE (loaded per site) by code
    const siteItems: SiteItem[] = net === 'focus'
      ? rawMeta.sites.map((x) => ({ key: x.name, label: x.name, sub: x.location, lat: x.lat, lon: x.lon, color: x.color, n: siteCounts[x.name] ?? 0, selected: !activeSites || activeSites.has(x.name) }))
      : (netView?.all ?? []).map((x) => {
        const on = netView!.chosenCodes.has(x.code)
        return { key: x.code, label: x.name, group: netView!.improveCodes.has(x.code) ? 'IMPROVE' : 'SPARTAN', sub: net === 'improve' ? x.location : x.code, lat: x.lat, lon: x.lon,
          color: on ? x.color : '#9aa5b1', n: net === 'improve' && !on ? x.n_filters : siteCounts[x.name] ?? 0, selected: on }
      })

    const state: SubsetState = {
      basis, mac: viewOpts.mac, exportMac,
      pmfNote: chartTab && pmfSiteInView ? (pmfOnly ? `Only ${pmf?.site_name ?? 'the PMF site'}'s filters with a PMF day (${pmf?.n ?? 0} days) are drawn; sites without PMF are hidden.` : `Narrows ${pmf?.site_name ?? 'the PMF site'}'s filters only (${pmf?.n ?? 0} PMF days); other sites are unaffected.`) + ' Each source\'s share is also a measurement (PMF sources), and the scatterplot can colour by PMF source (Colour → Value).' : undefined,
      pmfOnly,
      withVolume: basis === 'mass' ? [inViewRows.filter((r) => typeof r[VOLUME] === 'number').length, inViewRows.length] : undefined,
      convention: activeConvention,
      defaultConvention,
      seasons,
      activeSeasons: seasonSel ?? new Set(seasons.map((s) => s.name)),
      seasonCounts,
      sites: siteNames,
      activeSites: activeSites ?? new Set(siteNames),
      siteCounts,
      includeExcluded,
      nExcluded: netView ? sourceRows.filter((r) => r.excluded).length : filters.n_excluded,
      dateRange,
      xField,
      yField,
      field,
      axes,
      pinned,
      pinnedInSubset: !!pinned && rows.some((r) => r.id === pinned.id),
      shown: pmfMode ? pmfRows.length : calMode ? pool.filter(inSeason).filter(inPmf).length : rows.length,
      total: pmfMode ? (data.pmf?.n ?? 0) : calMode ? pool.length : netView ? sourceRows.length : filters.n,
      pmfScheme,
      pmfGroups: pmfGroupList,
      activePmf: pmfSel ?? new Set(pmfGroupList.map((g) => g.name)),
      seasonsOn,
      pmfOn,
      pmfCounts,
      network: net,
      chosenNetwork: network,
      dropped: dropIds.size,
      droppedList: [...dropIds].map((id) => {
        const r = sourceById.get(id)
        return { id, label: r ? `${r.site} · ${r.date}` : 'not in the current network' }
      }),
      ftirBaseline,
      showFtirBaseline: [xField, yField, field].includes('EC (FTIR)'),
      kept: keepIds?.size ?? null,
      keptIds: keepIds ? [...keepIds] : [],
      networkLoading: netData.loading,
      networkError: netData.error,
      siteItems,
      siteLimitNote: net === 'improve' ? `${improveSites.length} of ${MAX_IMPROVE_SITES} IMPROVE sites`
        : net === 'both' && netView ? `${netView.chosen.length - improveSites.length} SPARTAN · ${improveSites.length}/${MAX_IMPROVE_SITES} IMPROVE` : undefined,
      calendarOptions: ['local', ...Object.keys(conventions)],
      customSeasons,
      // sites in view whose sources disagree, with the calendar each is binned by
      calendarVariants: isLocal ? calendarSites.flatMap((site) => {
        const code = siteCode.get(site)
        const options = calendarVariants(site, code)
        return options && code ? [{ site, code, options, value: calVariants[code] ?? options[0].key }] : []
      }) : [],
      sitesAtDefault: net === 'spartan' ? spartanSel === null : net === 'improve' ? improveSites.length > 0 : net === 'both' ? spartanSel === null && improveSites.length === 0 : !activeSites,
      allSitesSelected: hasSpartan(net) ? !!netView && netView.chosen.length - (net === 'both' ? improveSites.length : 0) === netView.nSpartan : !activeSites,
    }
    const actions: SubsetActions = {
      onBasis: setBasis,
      onPmfOnly: setPmfOnly,
      onMac: setMac,
      onConvention: (v) => {
        setConvention(v)
        setActiveSeasons(null)
      },
      onToggleSeason: (name) => {
        const cur = new Set(seasonSel ?? seasons.map((s) => s.name))
        if (cur.has(name)) cur.delete(name)
        else cur.add(name)
        setActiveSeasons(cur.size === 0 || cur.size === seasons.length ? null : cur)
      },
      onAllSeasons: () => setActiveSeasons(null),
      onPmfScheme: (v) => { setPmfScheme(v); setActivePmf(null) },
      onTogglePmf: (name) => {
        const cur = new Set(pmfSel ?? pmfGroupList.map((g) => g.name))
        if (cur.has(name)) cur.delete(name)
        else cur.add(name)
        setActivePmf(cur.size === 0 || cur.size === pmfGroupList.length ? null : cur)
      },
      onAllPmf: () => setActivePmf(null),
      onSeasonsOn: (v) => { setSeasonsOn(v); if (!v) setActiveSeasons(null) },
      onPmfOn: (v) => { setPmfOn(v); if (!v) setActivePmf(null) },
      // the sidebar ticks the box itself (SitePicker); the charts follow as a transition, which a
      // newer click interrupts, so clicking through sites stays responsive. Updates are functional
      // because a click can land while the previous pick is still rendering
      onToggleSite: (key) => startTransition(() => {
        // a site belongs to one network: IMPROVE codes load per site, SPARTAN codes go in spartanSel
        if (hasImprove(net) && netView?.improveCodes.has(key)) {
          setImproveSites((cur) => cur.includes(key) ? cur.filter((c) => c !== key) : cur.length >= MAX_IMPROVE_SITES ? cur : [...cur, key])
          return
        }
        if (hasSpartan(net) && netView) {
          const spAll = netView.all.filter((x) => !netView.improveCodes.has(x.code)).map((x) => x.code)
          const focusCodes = data?.meta.sites.map((x) => x.code) ?? []
          setSpartanSel((sel) => {
            const cur = new Set(sel === null ? focusCodes : sel === 'all' ? spAll : sel)
            if (cur.has(key)) cur.delete(key)
            else cur.add(key)
            return cur.size === spAll.length ? 'all' : [...cur]
          })
          return
        }
        if (hasSpartan(net)) return
        setActiveSites((sel) => {
          const cur = new Set<string>(sel ?? siteNames)
          if (cur.has(key)) cur.delete(key)
          else cur.add(key)
          return cur.size === 0 || cur.size === siteNames.length ? null : cur
        })
      }),
      onAllSites: () => startTransition(() => (hasSpartan(net) ? setSpartanSel('all') : setActiveSites(null))),
      // "Deselect all" empties the pick: the charts show nothing until sites are chosen again
      onClearSites: () => startTransition(() => {
        if (hasImprove(net)) setImproveSites([])
        if (hasSpartan(net)) setSpartanSel([])
        if (net === 'focus') setActiveSites(null)
      }),
      onFocusSites: () => startTransition(() => setSpartanSel(null)),
      onFtirBaseline: setFtirBaseline,
      onCustomSeasons: (v) => { setCustomSeasons(v); setActiveSeasons(null) },
      // the default calendar is left out of the link
      onCalendarVariant: (code, key, isDefault) => {
        setCalVariants((cur) => { const next = { ...cur }; if (isDefault) delete next[code]; else next[code] = key; return next })
        setActiveSeasons(null)
      },
      onDropIds: dropFilters,
      // keeping again narrows the kept set to filters in both brushes
      onKeepIds: keepFilters,
      onRestoreIds: restoreFilters,
      onClearDrop: () => setDropIds(new Set()),
      onClearKeep: () => setKeepIds(null),
      onNetwork: (n) => {
        // turning a network on or off keeps the other network's pick
        if (!hasSpartan(n)) setSpartanSel(null)
        if (!hasImprove(n)) setImproveSites([])
        setNetwork(n)
        // each network opens on its own natural pair (focus: HIPS BC vs FTIR EC; IMPROVE: HIPS BC vs TOR EC)
        setXField(NETWORK_DEFAULTS[n].x)
        setYField(NETWORK_DEFAULTS[n].y)
        setField(NETWORK_DEFAULTS[n].field)
        setActiveSites(null)
        setActiveSeasons(null)
        setConvention(null)
        setPinnedId(null)
      },
      onIncludeExcluded: setIncludeExcluded,
      onDateRange: setDateRange,
      onXField: chooseX,
      onYField: chooseY,
      onField: setField,
      onAxes: setAxes,
      onUnpin: dismissSample,
      onOpenPinned: () => pinnedId && setOpenId(pinnedId),
      onReset: () => {
        setActiveSeasons(null)
        setActiveSites(null)
        setSpartanSel(null)
        setDropIds(new Set())
        setKeepIds(null)
        if (net === 'improve') setImproveSites([])
        setActivePmf(null)
        setSeasonsOn(true)
        setPmfOn(false)
        setDateRange(null)
        setIncludeExcluded(false)
        setConvention(null)
      },
    }
    if (tab === 'dashboard') dashFilters = { state, actions, brushRows }
    return <SubsetBar meta={meta} caps={caps} brushRows={brushRows} state={state} actions={actions} />
  })()

  return (
    <HeaderNavProvider tabs={NAV_TABS} meetings={MEETINGS} open={openTab}>
    <RegressionProvider methods={estimators} setMethods={setEstimators} showCI={showCI} setShowCI={setShowCI}>
    <PrefsProvider>
    <HighlightProvider pinnedId={pinnedId} setPinned={setPinnedId} openSample={openSample} openRecord={openRecord} onDismiss={dismissSample}>
    {/* hand edits hold on every tab: each chart that draws a filter leaves the excluded ones out */}
    <PageSidebarProvider>
    {/* "Define a ratio…" in every field picker (components/RatioDialog) */}
    <RatiosProvider meta={meta} presets={ratioView.presets} custom={ratios} onAdd={addRatio} onRemove={removeRatio} count={countRatio}>
    <FilterEditsProvider enabled dropped={dropIds} kept={keepIds}
      onDrop={dropFilters} onKeep={keepFilters} onRestore={restoreFilters} group={group} onGroup={setGroup}>
      {openRow && (
        <SampleDrawer
          row={openRow}
          allRows={rawSourceRows.some((r) => r.id === openRow.id) ? rawSourceRows : filters.rows}
          meta={drawerMeta}
          xField={xField}
          yField={yField}
          focusField={field}
          pinned={pinnedId === openRow.id}
          onPin={(id) => id === null ? dismissSample() : setPinnedId(id)}
          onOpen={openSample}
          onClose={dismissSample}
          context={openRec}
        />
      )}
      {!openRow && openRec && <RecordDrawer rec={openRec} onClose={dismissSample} />}
      {!openRow && !openRec && shownGroup && <GroupDrawer group={shownGroup} byId={sourceById} meta={meta}
        fields={tab === 'correlation' ? [xField, yField] : [field]} onClose={() => setGroup(null)} />}
      {isEmbed() && <a className="embed-open" href="#" onClick={async (e) => { e.preventDefault(); window.open((await viewUrl(currentView())), '_blank', 'noopener') }}>Open in the gallery ↗</a>}
      {(NETWORK_TABS as readonly string[]).includes(tab) && <BrushBar onKeep={keepFilters} onDrop={dropFilters} />}
      <Shell nav={nav} dash={tab === 'dashboard'} wide={tab === 'baseline' || tab === 'calibration' || tab === 'similarity' || tab === 'meeting' || tab === 'blanks'} editsBadge={!subset}>
        {/* tabs without the Filters panel: a sidebar for the page's own controls, with the fits where the page fits */}
        {subset ?? ((tab === 'baseline' || tab === 'blanks' || tab === 'meeting')
          && <PageSidebar fits={tab === 'baseline' || meeting === 'followup' && tab === 'meeting'} />)}
        {tab === 'census' && <CensusPage census={census} />}
        {tab === 'similarity' && <SimilarityPage />}
        {tab === 'meeting' && <PresentationProvider mode={presentation} set={setPresentation}>
          <div className="panel" aria-label="Meeting selection">
            <Combobox label="Meeting" value={meeting} options={MEETINGS} onChange={setMeeting} />
          </div>
          <SidebarControls title="Meeting figures">
            <PresentationSwitch ignored={dropIds.size + (keepIds ? 1 : 0)} viewChanged={viewOpts.mac !== exportMac || basis !== 'conc'} />
          </SidebarControls>
          {meeting === 'followup' ? <MeetingFollowupPage galleryRows={filters.rows} galleryMeta={data.meta} viewRows={presentRows} viewMeta={presentMeta} />
            : meeting === 'satoshi' ? <SatoshiFollowupPage />
            : <MeetingSpecPage key={meeting} spec={SPEC_MEETINGS.find((m) => m.id === meeting) ?? SPEC_MEETINGS[0]} />}
        </PresentationProvider>}
        {tab === 'blanks' && <BlanksPage />}
        {tab === 'baseline' && <BaselineComparisonPage mac={exportMac} demingLambda={data.calibration?.deming_lambda_mac10 ?? 1} />}
        {tab === 'calibration' && <CalibrationPage calib={data.calibration} runs={data.calibrationRuns} meta={drawerMeta} galleryRows={paintedSourceRows} galleryMeta={meta} knownIds={knownIds} seasonSel={seasonSel} pmfSel={pmfKeep} pmfLabelOfDate={pmfLabelOfDate} />}
        {tab === 'dashboard' && <DashboardPage dash={dash} setDash={setDash} filters={dashFilters} onScope={setDashScope} onPickSite={pickDashSite} display={<DisplayToggles />}
          data={{ rows, meta, axes, pmf: pmf ?? null, pmfRows, eff, defaults: { x: xField, y: yField, field }, openSite: openDashSite, pmfLabelOfDate,
            calibration: data.calibrationRuns ? { runs: data.calibrationRuns, lambda10: data.calibration?.deming_lambda_mac10 ?? 2.96, knownIds, galleryRows: paintedSourceRows, galleryMeta: meta, meta: drawerMeta } : null }} />}
        <AxisFieldsProvider meta={meta} value={axisValues} set={setAxisField}>
        {tab === 'correlation' && (
          <>
            <Scatterplot rows={rows} meta={meta} xField={xEff} yField={yEff} axes={axes} />
            <DensityHexbin rows={rows} meta={meta} xField={xEff} yField={yEff} axes={axes} />
            <ConnectedScatter rows={rows} meta={meta} xField={xEff} yField={yEff} axes={axes} />
            <Correlogram rows={rows} meta={meta} />
            <SpeciesGraph rows={rows} meta={meta} />
            <AvailabilityHeatmap rows={rows} meta={meta} xField={xEff} yField={yEff} />
          </>
        )}
        {tab === 'distribution' && (
          <>
            <DistributionPanel rows={rows} meta={meta} field={fieldEff} />
            <Histogram rows={rows} meta={meta} field={fieldEff} />
            <Ridgeline rows={rows} meta={meta} field={fieldEff} />
          </>
        )}
        {tab === 'evolution' && (
          <>
            <Timeseries rows={rows} meta={meta} field={fieldEff} />
            <MonthlyBand rows={rows} meta={meta} field={fieldEff} />
            <SeasonalClock rows={rows} meta={meta} field={fieldEff} />
            <MonthlyHeatmap rows={rows} meta={meta} field={fieldEff} />
          </>
        )}
        {tab === 'ranking' && (
          <>
            <CorrelationRanking rows={rows} meta={meta} target={fieldEff} />
            <Barplot rows={rows} meta={meta} field={fieldEff} />
            <SiteRadar rows={rows} meta={meta} />
            <ParallelCoordinates rows={rows} meta={meta} />
          </>
        )}
        {tab === 'partOfWhole' && <CompositionTreemap rows={rows} meta={meta} />}
        {tab === 'weather' && <WeatherPage rows={rows} picked={netView?.chosenCodes ?? new Set()} onPick={hasSpartan(net) ? setSpartanPick : undefined}
          meta={meta} field={fieldEff} xField={xEff} yField={yEff} axes={axes} setField={setField} pmfLabelOfDate={pmfLabelOfDate} />}
        </AxisFieldsProvider>
        {tab === 'map' && <SiteBubbleMap rows={mapRows} meta={meta} field={fieldEff} focusSite={activeSites?.size === 1 ? [...activeSites][0] : undefined} />}
        {tab === 'flow' && <MethodSankey rows={rows} meta={meta} />}
        {tab === 'pmf' &&
          (pmf ? (
            <>
              <SourceStack pmf={pmf} rows={pmfRows} meta={meta} />
              <SourceSeasonality pmf={pmf} rows={pmfRows} meta={meta} />
            </>
          ) : (
            <p style={{ fontSize: 13, color: 'var(--ink-muted)' }}>
              No PMF solution exported. Run <code>python gallery/data/export_data.py</code> with the ETAD factor CSVs present.
            </p>
          ))}
      </Shell>
    </FilterEditsProvider>
    </RatiosProvider>
    </PageSidebarProvider>
    </HighlightProvider>
    </PrefsProvider>
    </RegressionProvider>
    </HeaderNavProvider>
  )
}

/** `wide` gives the long research tabs room for the table-of-contents column. */
/** `editsBadge`: show the hand edits in the header, for tabs without the Filters panel that lists them */
/** `dash`: the Dashboard, full width, its options in the header (pages/DashboardPage portals them into #dash-bar) */
function Shell({ children, nav, wide = false, editsBadge = false, dash = false }: { children: React.ReactNode; nav?: React.ReactNode; wide?: boolean; editsBadge?: boolean; dash?: boolean }) {
  const container = dash ? 'container dash' : wide ? 'container wide' : 'container'
  return (
    <>
      <header className="app-header">
        <div className={container}>
          <div className="app-title-row">
            <h1 className="app-title">Air Gallery</h1>
            {editsBadge && <HandEditsBadge />}
            {nav ?? <span className="nav" />}
            {dash && <div id="dash-bar" className="dash-bar-slot" />}
            <HeaderTools />
          </div>
        </div>
      </header>
      <div className={container}>{children}</div>
    </>
  )
}

/** Explain toggle (absent in public builds, see lib/prefs) and the Views menu (presets, saved views, share and embed). */
function HeaderTools() {
  return (
    <span className="header-tools">
      <span className="header-display"><DisplayToggles /></span>
      <CommandPaletteButton />
      <GlossaryButton />
      <ViewsMenu />
    </span>
  )
}

/** Readable text, Explain, the sections list: in the header, or in the page menu on a phone */
function DisplayToggles() {
  const { explain, setExplain, density, setDensity } = usePrefs()
  return (
    <>
      <button type="button" className={`btn quiet icon-btn${density === 'readable' ? ' on' : ''}`} aria-pressed={density === 'readable'}
        aria-label="Readable text" onClick={() => setDensity(density === 'readable' ? 'compact' : 'readable')}
        title={density === 'readable' ? 'Readable: larger text everywhere (click for compact)' : 'Compact text (click for readable: larger text in every chart, control and table); remembered in this browser'}>
        <TextSizeIcon />
      </button>
      {EXPLAIN_AVAILABLE && <button type="button" className={`btn quiet icon-btn${explain ? ' on' : ''}`} aria-pressed={explain} aria-label="Explain"
        onClick={() => setExplain(!explain)}
        title={explain ? 'Explain is on: every chart shows its explanation and provenance (click to hide)' : "Explain: show every chart's explanation and provenance, and the sidebar's captions"}>
        <ExplainIcon />
      </button>}
      <TocToggle />
    </>
  )
}

/** Show or hide the "On this tab" contents sidebar (components/PageToc), on tabs that have one. */
function TocToggle() {
  const nav = useHeaderNav()
  if (!nav?.tocPresent) return null
  return <button type="button" className={`btn quiet icon-btn${nav.tocShown ? ' on' : ''}`} aria-pressed={nav.tocShown} aria-label="Sections on this tab"
    onClick={() => nav.setTocShown(!nav.tocShown)}
    title={nav.tocShown ? 'Hide the list of sections and give charts the full width' : 'Show the list of sections on this tab'}>
    <SidebarIcon />
  </button>
}

/** The sidebar of a tab without the Filters panel: the page's own controls (lib/pageSidebar), and the fits where it draws them. */
function PageSidebar({ fits }: { fits: boolean }) {
  const [open, setOpen] = useSidePanelOpen()
  const top = useHeaderHeight()
  return <>
    <SideDock open={open} top={top} tab={<button className="subset-tab" type="button" aria-expanded={open} onClick={() => setOpen(!open)} title={open ? 'Hide the filters' : 'Show the filters'}><span className="subset-tab-label">Filters</span></button>}>
      <aside className="panel subset subset-panel" aria-label="Filters">
        <header className="subset-head"><strong>Filters</strong><span style={{ marginLeft: 'auto' }}><button type="button" className="btn quiet" onClick={() => setOpen(false)} aria-label="Close filters" title="Close">✕</button></span></header>
        <PageSidebarSlot />
        {fits && <RegressionControls />}
      </aside>
    </SideDock>
  </>
}

/** While a brush is active, a bar at the bottom offers to keep only the brushed filters or exclude them (both go in the link). */
function BrushBar({ onKeep, onDrop }: { onKeep: (ids: Set<string>) => void; onDrop: (ids: Set<string>) => void }) {
  const hl = useHighlight()
  if (!hl.selected) return null
  const ids = hl.selected
  return (
    <div className="brush-bar" role="region" aria-label="Brushed filters">
      <strong>{ids.size} brushed</strong>
      <button type="button" className="btn" onClick={() => { onKeep(ids); hl.setSelected(null) }} title="Filter every chart to these filters">Keep only these</button>
      <button type="button" className="btn" onClick={() => { onDrop(ids); hl.setSelected(null) }} title="Remove these filters from every chart; the Filters panel can put them back">Exclude these</button>
      <button type="button" className="btn quiet" onClick={() => hl.setSelected(null)} aria-label="Clear the brush">×</button>
    </div>
  )
}
