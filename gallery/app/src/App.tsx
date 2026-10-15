import { RegressionProvider, RegressionControls, parseEstimators, estimatorParam, type Estimator } from '@/lib/regressionPrefs'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useGalleryData } from '@/hooks/useGalleryData'
import { HighlightProvider, baseFilterId, useHighlight, type FilterRecord } from '@/lib/highlight'
import { NO_PMF, pmfGroups, pmfKeyByDate, pmfLabel, type PmfScheme } from '@/lib/pmfGroups'
import { EXPLAIN_AVAILABLE, PrefsProvider, usePrefs } from '@/lib/prefs'
import { listParam, listToParam, readParams, writeParams } from '@/lib/url'
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
import { BlanksPage } from '@/pages/BlanksPage'
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
import { FilterEditsProvider, type FilterGroup } from '@/lib/filterEdits'
import { GroupDrawer } from '@/components/GroupDrawer'
import { WeatherPage } from '@/pages/WeatherPage'
import { HIDDEN_DUPLICATES } from '@/fieldLineage'
import { WEATHER_GROUP, type WeatherFile } from '@/lib/weather'
import { useHeaderHeight, useSidePanelOpen } from '@/hooks/useSidePanel'

/**
 * Tabs mirror the react-graph-gallery categories, plus the census. `hidden` tabs
 * stay out of the tab row but still open from a link (#tab=census); the row shows
 * one while it is the open tab.
 */
const TABS = [
  { key: 'census', label: 'Census', hidden: true },
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

const MEETINGS = [
  { key: 'followup', label: 'Follow up meeting' },
  { key: 'satoshi', label: 'Satoshi 2 Oct' },
] as const
type MeetingKey = (typeof MEETINGS)[number]['key']
// Keep existing #tab=satoshi links working under the shared Meeting tab.
const tabFromParams = (p: URLSearchParams): TabKey =>
  p.get('tab') === 'satoshi' && shared('meeting') ? 'meeting' : isTab(p.get('tab')) ? p.get('tab') as TabKey : LANDING_TAB
const meetingFromParams = (p: URLSearchParams): MeetingKey =>
  p.get('tab') === 'satoshi' || p.get('meeting') === 'satoshi' ? 'satoshi' : 'followup'

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
const parseSpartanSel = (v: string | null): string[] | 'all' | null =>
  v == null || v === '' ? null : v === '*' ? 'all' : v === '-' ? [] : v.split(',').filter(Boolean)

/** tabs whose charts can draw a whole network (Filters sidebar → network) */
const NETWORK_TABS = ['correlation', 'distribution', 'evolution', 'ranking', 'partOfWhole', 'weather'] as const
const FILTERS: SubsetCaps = { seasons: true, sites: true, excluded: true, dates: true, fields: 'single', datasetLabel: 'SPARTAN filters' }
const CAPS: Record<Exclude<TabKey, 'census' | 'baseline' | 'similarity' | 'meeting' | 'blanks'>, SubsetCaps> = {
  correlation: { ...FILTERS, fields: 'pair', networks: true, fits: true },
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
  const [dateRange, setDateRange] = useState<DateRange | null>(
    init.get('from') && init.get('to') ? [init.get('from')!, init.get('to')!] : null
  )
  const [xField, setXField] = useState(init.get('x') ?? DEFAULT_X)
  const [yField, setYField] = useState(init.get('y') ?? DEFAULT_Y)
  const [field, setField] = useState(init.get('f') ?? DEFAULT_FIELD)
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
      from: dateRange?.[0],
      to: dateRange?.[1],
      excl: includeExcluded ? '1' : null,
      drop: dropIds.size ? [...dropIds].join(',') : null,
      fb: ftirBaseline !== 'none' ? ftirBaseline : null,
      keep: keepIds ? [...keepIds].join(',') : null,
      x: xField !== DEFAULT_X ? xField : null,
      y: yField !== DEFAULT_Y ? yField : null,
      f: field !== DEFAULT_FIELD ? field : null,
      pin: pinnedId,
      eq: axes.identity ? null : '0',
      ax: axes.mode !== DEFAULT_AXES.mode ? axes.mode : null,
      log: axes.log ? '1' : null,
      pax: axes.panels === 'Own' ? 'own' : null,
    })
  }, [tab, meeting, estimators, showCI, convention, calVariants, activeSeasons, activeSites, network, spartanSel, improveSites, dropIds, keepIds, ftirBaseline, customSeasons, pmfScheme, activePmf, seasonsOn, pmfOn, dateRange, includeExcluded, xField, yField, field, pinnedId, axes])

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
      setDateRange(p.get('from') && p.get('to') ? [p.get('from')!, p.get('to')!] : null)
      setIncludeExcluded(p.get('excl') === '1')
      setDropIds(idSet(p.get('drop')) ?? new Set())
      setFtirBaseline((['spline', 'vibes'] as string[]).includes(p.get('fb') ?? '') ? p.get('fb') as FtirBaseline : 'none')
      setKeepIds(idSet(p.get('keep')))
      setXField(p.get('x') ?? DEFAULT_X)
      setYField(p.get('y') ?? DEFAULT_Y)
      setField(p.get('f') ?? DEFAULT_FIELD)
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
  // the chosen network's sites, coloured in selection order, and its rows recoloured to match
  const netView = useMemo(() => {
    if (net === 'focus' || !data) return null
    const d = netData.data
    if (!d) return { all: [] as SiteMeta[], chosen: [] as SiteMeta[], chosenCodes: new Set<string>(), chosenNames: new Set<string>(), improveCodes: new Set<string>(), nSpartan: 0, rows: [] as FilterRow[], fields: [] as string[], field_units: {} as Record<string, string>, field_groups: [] as { label: string; fields: string[] }[], coverage: {} as Record<string, number> }
    const sp = hasSpartan(net), im = hasImprove(net)
    // SPARTAN: the four focus sites keep their filters.json rows (HIPS, FTIR groups, exclusions,
    // local calendars) and their names and colours; the other sites bring the public ChemSpec rows
    const focus = sp ? new Map(data.meta.sites.map((x) => [x.code, x])) : new Map<string, SiteMeta>()
    const sites = d.sites.map((x) => focus.get(x.code) ?? x)
    for (const f of focus.values()) if (!sites.some((x) => x.code === f.code)) sites.push(f)
    const spartanCodes = sites.filter((x) => !d.improveCodes.has(x.code)).map((x) => x.code)
    const chosenCodes = new Set([
      ...(sp ? (spartanSel === null ? [...focus.keys()] : spartanSel === 'all' ? spartanCodes : spartanSel) : []),
      ...(im ? improveSites : []),
    ])
    let i = 0
    const colour = new Map(sites.filter((x) => chosenCodes.has(x.code)).map((x) => [x.name, focus.get(x.code)?.color ?? networkColor(i++)]))
    const paint = (name: string) => colour.get(name) ?? '#9aa5b1'
    const all = sites.map((x) => ({ ...x, color: paint(x.name) }))
    // the focus sites also get the public HIPS BC, so one absorption axis spans every SPARTAN site
    const pubBC = new Map<string, number>(d.rows.flatMap((r) => (focus.has(r.code) && typeof r[PUBLIC_BC] === 'number' ? [[r.id, r[PUBLIC_BC] as number]] : [])))
    const focusWithPub = focusRows.map((r): FilterRow => { const v = pubBC.get(r.id); return v === undefined ? r : { ...r, [PUBLIC_BC]: v } })
    const rows: FilterRow[] = [...(sp ? focusWithPub : []), ...d.rows.filter((r) => !focus.has(r.code))]
      .map((r) => ({ ...r, color: paint(r.site) }))
    const fields = sp ? [...new Set([...data.meta.fields, ...d.fields])] : d.fields
    const groups = sp ? mergeGroups(data.meta.field_groups, d.field_groups) : d.field_groups
    const chosenNames = new Set(all.filter((x) => chosenCodes.has(x.code)).map((x) => x.name))
    // counts in the field pickers follow the exclusion toggle, like every chart (Addis: 189, not 190)
    const inChosen = rows.filter((r) => chosenNames.has(r.site) && (includeExcluded || !r.excluded))
    const coverage = Object.fromEntries(fields.map((f) => [f, inChosen.filter((r) => typeof r[f] === 'number').length]))
    return { all, chosen: all.filter((x) => chosenCodes.has(x.code)), chosenCodes, chosenNames, improveCodes: d.improveCodes, nSpartan: spartanCodes.length,
      rows, fields, field_units: sp ? { ...d.field_units, ...data.meta.field_units } : d.field_units, field_groups: groups, coverage }
  }, [net, data, focusRows, netData.data, improveSites, spartanSel, includeExcluded])
  // the ChemSpec restatements of the FTIR carbon are never offered; a link naming one falls back
  const activeFields = (netView ? netView.fields : data?.meta.fields ?? []).filter((f) => !HIDDEN_DUPLICATES.includes(f))

  // Fields that vanished from the export, or that the chosen network does not measure,
  // fall back to that network's defaults instead of leaving a chart empty.
  useEffect(() => {
    if (!data || (netView && !netView.fields.length)) return
    // weather fields (lib/weather) load on their own, after this first check, so they are known by name
    const ok = (f: string) => activeFields.includes(f) || /\((ERA5|station)\)$/.test(f)
    const d = NETWORK_DEFAULTS[net]
    if (!ok(xField)) setXField(ok(d.x) ? d.x : activeFields[0])
    if (!ok(yField)) setYField(ok(d.y) ? d.y : activeFields[1] ?? activeFields[0])
    if (!ok(field)) setField(ok(d.field) ? d.field : activeFields[0])
  }, [data, netView, activeFields, net, xField, yField, field])

  // Sites outside the four focus sites have no in-house HIPS BC, only the public HIPS BC. When a
  // chosen site would drop out of an axis for that reason, the axis moves to the public field
  // (which every chosen site with HIPS data carries) instead of leaving that site's panels empty.
  useEffect(() => {
    if (!netView || !hasSpartan(net) || !netView.fields.includes(PUBLIC_BC)) return
    const missing = netView.chosen.some((x) => {
      const own = netView.rows.filter((r) => r.site === x.name)
      return own.length && !own.some((r) => typeof r['HIPS BC'] === 'number') && own.some((r) => typeof r[PUBLIC_BC] === 'number')
    })
    if (!missing) return
    if (xField === 'HIPS BC') setXField(PUBLIC_BC)
    if (yField === 'HIPS BC') setYField(PUBLIC_BC)
    if (field === 'HIPS BC') setField(PUBLIC_BC)
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
  const sourceRows = useMemo(() => {
    const rows = netView ? netView.rows : focusRows
    if (!weather) return rows
    // the weather fields ride on every row, so every chart and picker can use them
    return rows.map((r) => {
      const w = weather.rows[r.id]
      return w ? { ...r, ...Object.fromEntries(weather.fields.map((f, i) => [f, w[i]])) } : r
    })
  }, [netView, focusRows, weather])
  const sourceById = useMemo(() => new Map(sourceRows.map((r) => [r.id, r])), [sourceRows])
  const baseRows = useMemo(() => {
    if (!data) return []
    return sourceRows
      .filter((r) => includeExcluded || !r.excluded)
      .map((r) => {
        const s = seasonForFilter(r)
        return s !== r.season ? { ...r, season: s } : r
      })
  }, [data, sourceRows, includeExcluded, seasonForFilter])

  // The season selection, kept only for seasons that exist under the current
  // calendar and sites; a selection that no longer names any season means "all".
  const seasonSel = useMemo(() => {
    if (!seasonsOn || !activeSeasons) return null
    const names = new Set(seasons.map((se) => se.name))
    const kept = new Set([...activeSeasons].filter((n) => names.has(n)))
    return kept.size === 0 || kept.size === names.size ? null : kept
  }, [seasonsOn, activeSeasons, seasons])

  // what the date brush draws: every filter except the date cut, so its range follows the picked sites
  const brushRows = useMemo(
    () => baseRows
      .filter((r) => !seasonSel || seasonSel.has(r.season))
      .filter((r) => (netView ? netView.chosenNames.has(r.site) : !activeSites || activeSites.has(r.site)))
      .filter((r) => !dropIds.has(r.id) && (!keepIds || keepIds.has(r.id))),
    [baseRows, seasonSel, activeSites, netView, dropIds, keepIds]
  )
  const rows = useMemo(
    () =>
      baseRows
        .filter(inRange)
        .filter((r) => !seasonSel || seasonSel.has(r.season))
        .filter((r) => (netView ? netView.chosenNames.has(r.site) : !activeSites || activeSites.has(r.site)))
        .filter((r) => !dropIds.has(r.id) && (!keepIds || keepIds.has(r.id))),
    [baseRows, inRange, seasonSel, activeSites, netView, dropIds, keepIds]
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
      const raw = openId && data ? sourceRows.find((r) => r.id === openId) ?? data.filters.rows.find((r) => r.id === openId) : null
      return raw ? { ...raw, season: seasonForFilter(raw) } : null
    },
    [openId, data, sourceRows, seasonForFilter]
  )

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
  const meta = { ...rawMeta, season_conventions: conventions, seasons, season_convention: activeConvention,
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
  // the baseline variants are drawn but never offered in the pickers
  meta.field_units = { ...meta.field_units, [FTIR_VARIANT.spline]: 'ug/m3', [FTIR_VARIANT.vibes]: 'ug/m3' }
  const eff = (f: string) => (f === 'EC (FTIR)' && ftirBaseline !== 'none' ? FTIR_VARIANT[ftirBaseline] : f)
  const xEff = eff(xField), yEff = eff(yField), fieldEff = eff(field)

  const nav = (
    <nav className="nav">
      {TABS.filter((t) => shared(t.key) && (!('hidden' in t && t.hidden) || t.key === tab)).map((t) => (
        <button key={t.key} className={`nav-btn ${tab === t.key ? 'active' : ''}`} title={'title' in t ? t.title : undefined} onClick={() => setTab(t.key)}>
          {t.label}
        </button>
      ))}
    </nav>
  )

  const subset = (() => {
    if (tab === 'census' || tab === 'baseline' || tab === 'similarity' || tab === 'meeting' || tab === 'blanks') return null
    const caps = net === 'focus' ? CAPS[tab] : { ...CAPS[tab], datasetLabel: net === 'improve' ? 'IMPROVE samples' : net === 'both' ? 'SPARTAN filters + IMPROVE samples' : 'SPARTAN filters', excluded: sourceRows.some((r) => r.excluded) }
    const pmfMode = tab === 'pmf'
    const calMode = tab === 'calibration'
    // Counts are computed *before* the dimension's own filter is applied,
    // so a chip always shows what selecting it would give you.
    const pool: { season: string; site: string; pmf: string }[] = pmfMode
      ? (data.pmf?.rows ?? []).map((r) => ({ season: seasonOf(pmfSiteName, r.month, r.season), site: pmfSiteName, pmf: pmfLabel(data.pmf, r.dominant_source, pmfScheme) }))
      : calMode
        ? addisEvalDates.map((d) => ({ season: seasonOf(pmfSiteName, d ? Number(d.slice(5, 7)) : null, 'undated'), site: pmfSiteName, pmf: pmfLabelOfDate(d) }))
        // hand exclusions count like any other filter, so the chips add up to what the charts draw
        : baseRows.filter(inRange).filter((r) => !dropIds.has(r.id) && (!keepIds || keepIds.has(r.id))).map((r) => ({ season: r.season, site: r.site, pmf: NO_PMF }))
    const usePmf = pmfMode || calMode
    const inSeason = (r: { season: string }) => !seasonSel || seasonSel.has(r.season)
    const inPmf = (r: { pmf: string }) => !usePmf || !pmfKeep || pmfKeep.has(r.pmf)
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
      onToggleSite: (key) => {
        // a site belongs to one network: IMPROVE codes load per site, SPARTAN codes go in spartanSel
        if (hasImprove(net) && netView?.improveCodes.has(key)) {
          setImproveSites((cur) => cur.includes(key) ? cur.filter((c) => c !== key) : cur.length >= MAX_IMPROVE_SITES ? cur : [...cur, key])
          return
        }
        if (hasSpartan(net)) {
          const cur = new Set([...(netView?.chosenCodes ?? [])].filter((c) => !netView?.improveCodes.has(c)))
          if (cur.has(key)) cur.delete(key)
          else cur.add(key)
          setSpartanSel(cur.size === (netView?.nSpartan ?? -1) ? 'all' : [...cur])
          return
        }
        const cur = new Set<string>(activeSites ?? siteNames)
        if (cur.has(key)) cur.delete(key)
        else cur.add(key)
        setActiveSites(cur.size === 0 || cur.size === siteNames.length ? null : cur)
      },
      onAllSites: () => (hasSpartan(net) ? setSpartanSel('all') : setActiveSites(null)),
      // "Deselect all" empties the pick: the charts show nothing until sites are chosen again
      onClearSites: () => {
        if (hasImprove(net)) setImproveSites([])
        if (hasSpartan(net)) setSpartanSel([])
        if (net === 'focus') setActiveSites(null)
      },
      onFocusSites: () => setSpartanSel(null),
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
    return <SubsetBar meta={meta} caps={caps} brushRows={brushRows} state={state} actions={actions} />
  })()

  return (
    <RegressionProvider methods={estimators} setMethods={setEstimators} showCI={showCI} setShowCI={setShowCI}>
    <PrefsProvider>
    <HighlightProvider pinnedId={pinnedId} setPinned={setPinnedId} openSample={openSample} openRecord={openRecord} onDismiss={dismissSample}>
    <FilterEditsProvider enabled={(NETWORK_TABS as readonly string[]).includes(tab)} dropped={dropIds} kept={keepIds}
      onDrop={dropFilters} onKeep={keepFilters} onRestore={restoreFilters} group={group} onGroup={setGroup}>
      {openRow && (
        <SampleDrawer
          row={openRow}
          allRows={sourceRows.some((r) => r.id === openRow.id) ? sourceRows : filters.rows}
          meta={meta}
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
      <Shell nav={nav} wide={tab === 'baseline' || tab === 'calibration' || tab === 'similarity' || tab === 'meeting' || tab === 'blanks'}>
        {/* tabs without the filter panel still get the fit controls where they draw fits */}
        {subset ?? ((tab === 'baseline' || (tab === 'meeting' && meeting === 'followup')) && <RegressionSidebar />)}
        {tab === 'census' && <CensusPage census={census} />}
        {tab === 'similarity' && <SimilarityPage />}
        {tab === 'meeting' && <>
          <div className="panel" aria-label="Meeting selection">
            <Combobox label="Meeting" value={meeting} options={MEETINGS} onChange={setMeeting} />
          </div>
          {meeting === 'followup' ? <MeetingFollowupPage galleryRows={filters.rows} galleryMeta={data.meta} /> : <SatoshiFollowupPage />}
        </>}
        {tab === 'blanks' && <BlanksPage />}
        {tab === 'baseline' && <BaselineComparisonPage mac={meta.mac_value} demingLambda={data.calibration?.deming_lambda_mac10 ?? 1} />}
        {tab === 'calibration' && <CalibrationPage calib={data.calibration} runs={data.calibrationRuns} meta={meta} knownIds={knownIds} seasonSel={seasonSel} pmfSel={pmfKeep} pmfLabelOfDate={pmfLabelOfDate} />}
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
    </HighlightProvider>
    </PrefsProvider>
    </RegressionProvider>
  )
}

/** `wide` gives the long research tabs room for the table-of-contents column. */
function Shell({ children, nav, wide = false }: { children: React.ReactNode; nav?: React.ReactNode; wide?: boolean }) {
  const container = wide ? 'container wide' : 'container'
  return (
    <>
      <header className="app-header">
        <div className={container}>
          <div className="app-title-row">
            <h1 className="app-title">Aethmodular graph gallery</h1>
            <HeaderTools />
          </div>
          {nav}
        </div>
      </header>
      <div className={container}>{children}</div>
    </>
  )
}

/** Explain toggle (absent in public builds, see lib/prefs) and the Views menu (presets, saved views, share and embed). */
function HeaderTools() {
  const { explain, setExplain } = usePrefs()
  return (
    <span className="header-tools">
      {EXPLAIN_AVAILABLE && <label className="toggle" title="Show every chart's explanation and provenance, and the subset bar's captions">
        <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} />
        Explain
      </label>}
      <ViewsMenu />
    </span>
  )
}

/** The filter panel's counterpart on tabs that have fits but no filters: same edge tab, same place, same open state. */
function RegressionSidebar() {
  const [open, setOpen] = useSidePanelOpen()
  const top = useHeaderHeight()
  return <>
    <SideDock open={open} top={top} tab={<button className="subset-tab" type="button" aria-expanded={open} onClick={() => setOpen(!open)} title={open ? 'Hide the fits' : 'Show the fits'}><span className="subset-tab-label">Fits</span></button>}>
      <aside className="panel subset subset-panel" aria-label="Fits">
        <header className="subset-head"><strong>Fits</strong><span style={{ marginLeft: 'auto' }}><button type="button" className="btn quiet" onClick={() => setOpen(false)} aria-label="Close fits" title="Close">✕</button></span></header>
        <RegressionControls />
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
