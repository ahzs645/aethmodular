import { useCallback, useEffect, useMemo, useState } from 'react'
import { useGalleryData } from '@/hooks/useGalleryData'
import { HighlightProvider, baseFilterId, type FilterRecord } from '@/lib/highlight'
import { NO_PMF, pmfGroups, pmfKeyByDate, pmfLabel, type PmfScheme } from '@/lib/pmfGroups'
import { PrefsProvider, usePrefs } from '@/lib/prefs'
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
import { SubsetBar, type SubsetActions, type SubsetCaps, type SubsetState } from '@/components/SubsetBar'
import { SampleDrawer } from '@/components/SampleDrawer'
import { RecordDrawer } from '@/components/RecordDrawer'
import type { DateRange } from '@/components/DateBrush'
import type { FilterRow } from '@/lib/types'
import { AXES_MODES, DEFAULT_AXES, type AxesMode, type AxesOpts } from '@/lib/axes'
import { localSeasonsFor, qualifySeason } from '@/siteSeasons'

/** Tabs mirror the react-graph-gallery categories, plus the census. */
const TABS = [
  { key: 'census', label: 'Census' },
  { key: 'correlation', label: 'Correlation' },
  { key: 'distribution', label: 'Distribution' },
  { key: 'evolution', label: 'Evolution' },
  { key: 'ranking', label: 'Ranking' },
  { key: 'partOfWhole', label: 'Part of a whole' },
  { key: 'map', label: 'Map' },
  { key: 'flow', label: 'Flow' },
  { key: 'pmf', label: 'PMF sources' },
  { key: 'calibration', label: 'Calibration' },
  { key: 'baseline', label: 'AIRSpec / VIBES' },
  { key: 'similarity', label: 'Spectral similarity' },
  { key: 'meeting', label: 'Meeting 17 Sep follow-up' },
] as const

type TabKey = (typeof TABS)[number]['key']
const isTab = (v: string | null): v is TabKey => !!v && TABS.some((t) => t.key === v)

/**
 * What each tab's underlying data can actually support. The PMF solution is
 * ETAD-only and carries no exclusion flags, so offering four site chips and a
 * 942-filter count there would just be wrong.
 */
const FILTERS: SubsetCaps = { seasons: true, sites: true, excluded: true, dates: true, fields: 'single', datasetLabel: 'SPARTAN filters' }
const CAPS: Record<Exclude<TabKey, 'census' | 'baseline' | 'similarity' | 'meeting'>, SubsetCaps> = {
  correlation: { ...FILTERS, fields: 'pair' },
  distribution: FILTERS,
  evolution: FILTERS,
  ranking: FILTERS,
  partOfWhole: { ...FILTERS, fields: 'none' },
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
const DEFAULT_Y = 'EC (FTIR)'
const DEFAULT_FIELD = 'EC (FTIR)'

export default function App() {
  const { data, error } = useGalleryData()

  // ---- state, seeded from the URL hash so a pasted link reopens the same view
  const init = useMemo(() => readParams(), [])
  const [tab, setTab] = useState<TabKey>(isTab(init.get('tab')) ? (init.get('tab') as TabKey) : 'census')
  // Mirrors get_clean_data: excluded samples are hidden by default but never
  // deleted, so you can always put them back and see what was removed.
  const [includeExcluded, setIncludeExcluded] = useState(init.get('excl') === '1')
  const [convention, setConvention] = useState<string | null>(init.get('cal'))
  const [activeSeasons, setActiveSeasons] = useState<Set<string> | null>(listParam(init.get('seasons')))
  const [activeSites, setActiveSites] = useState<Set<string> | null>(listParam(init.get('sites')))
  // PMF source filter (PMF and calibration tabs): which grouping, and which groups are kept
  const [pmfScheme, setPmfScheme] = useState<PmfScheme>(init.get('pmfg') === 'class' ? 'Marine vs combustion' : 'Dominant source')
  const [activePmf, setActivePmf] = useState<Set<string> | null>(listParam(init.get('pmf')))
  // each filter dimension can be switched off: seasonality is on by default; PMF source is off,
  // and switching it on keeps only filters that have a PMF day
  const [seasonsOn, setSeasonsOn] = useState(init.get('seas') !== '0')
  const [pmfOn, setPmfOn] = useState(init.get('pmfon') === '1')
  const [dateRange, setDateRange] = useState<DateRange | null>(
    init.get('from') && init.get('to') ? [init.get('from')!, init.get('to')!] : null
  )
  const [xField, setXField] = useState(init.get('x') ?? DEFAULT_X)
  const [yField, setYField] = useState(init.get('y') ?? DEFAULT_Y)
  const [field, setField] = useState(init.get('f') ?? DEFAULT_FIELD)
  const [pinnedId, setPinnedId] = useState<string | null>(init.get('pin'))
  // axes policy shared by every crossplot on the correlation tab
  const [axes, setAxes] = useState<AxesOpts>({
    identity: init.get('eq') !== '0',
    mode: (AXES_MODES as readonly string[]).includes(init.get('ax') ?? '') ? (init.get('ax') as AxesMode) : DEFAULT_AXES.mode,
    log: init.get('log') === '1',
  })
  // the sample whose detail drawer is open; opening also pins
  const [openId, setOpenId] = useState<string | null>(null)
  // what the clicked chart knows about the open filter; a filter outside filters.json opens from this alone
  const [openRec, setOpenRec] = useState<FilterRecord | null>(null)
  const openSample = useCallback((id: string) => {
    setPinnedId(id)
    setOpenId(id)
    setOpenRec(null)
  }, [])

  useEffect(() => {
    writeParams({
      tab,
      cal: convention,
      seasons: listToParam(activeSeasons),
      sites: listToParam(activeSites),
      pmfg: pmfScheme === 'Marine vs combustion' ? 'class' : null,
      pmf: listToParam(activePmf),
      seas: seasonsOn ? null : '0',
      pmfon: pmfOn ? '1' : null,
      from: dateRange?.[0],
      to: dateRange?.[1],
      excl: includeExcluded ? '1' : null,
      x: xField !== DEFAULT_X ? xField : null,
      y: yField !== DEFAULT_Y ? yField : null,
      f: field !== DEFAULT_FIELD ? field : null,
      pin: pinnedId,
      eq: axes.identity ? null : '0',
      ax: axes.mode !== DEFAULT_AXES.mode ? axes.mode : null,
      log: axes.log ? '1' : null,
    })
  }, [tab, convention, activeSeasons, activeSites, pmfScheme, activePmf, seasonsOn, pmfOn, dateRange, includeExcluded, xField, yField, field, pinnedId, axes])

  // A pasted link, the back button, or an edited hash should all apply —
  // not just the hash that was there on first load.
  useEffect(() => {
    const onHash = () => {
      const p = readParams()
      if (isTab(p.get('tab'))) setTab(p.get('tab') as TabKey)
      setConvention(p.get('cal'))
      setActiveSeasons(listParam(p.get('seasons')))
      setActiveSites(listParam(p.get('sites')))
      setPmfScheme(p.get('pmfg') === 'class' ? 'Marine vs combustion' : 'Dominant source')
      setActivePmf(listParam(p.get('pmf')))
      setSeasonsOn(p.get('seas') !== '0')
      setPmfOn(p.get('pmfon') === '1')
      setDateRange(p.get('from') && p.get('to') ? [p.get('from')!, p.get('to')!] : null)
      setIncludeExcluded(p.get('excl') === '1')
      setXField(p.get('x') ?? DEFAULT_X)
      setYField(p.get('y') ?? DEFAULT_Y)
      setField(p.get('f') ?? DEFAULT_FIELD)
      setPinnedId(p.get('pin'))
      setAxes({
        identity: p.get('eq') !== '0',
        mode: (AXES_MODES as readonly string[]).includes(p.get('ax') ?? '') ? (p.get('ax') as AxesMode) : DEFAULT_AXES.mode,
        log: p.get('log') === '1',
      })
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  // Fields that vanished from the export shouldn't leave a chart empty forever.
  useEffect(() => {
    if (!data) return
    const ok = (f: string) => data.meta.fields.includes(f)
    if (!ok(xField)) setXField(ok(DEFAULT_X) ? DEFAULT_X : data.meta.fields[0])
    if (!ok(yField)) setYField(ok(DEFAULT_Y) ? DEFAULT_Y : data.meta.fields[1] ?? data.meta.fields[0])
    if (!ok(field)) setField(ok(DEFAULT_FIELD) ? DEFAULT_FIELD : data.meta.fields[0])
  }, [data, xField, yField, field])

  const siteNames = useMemo(() => data?.meta.sites.map((s) => s.name) ?? [], [data])
  const pmfSiteName = data?.pmf?.site_name ?? 'Addis Ababa'
  // Season calendar. 'local' (the default) bins every filter by its own site's
  // calendar (docs/site-seasonality.md): Beijing's four seasons, Delhi's monsoon,
  // JPL's wet/dry windows, Addis's Bega/Belg/Kiremt. The shared Ethiopian month
  // bins stay available for like-month comparisons across sites.
  const activeConvention = convention ?? 'local'
  const isLocal = activeConvention === 'local'
  // PMF is Addis-only, so its season names need no site prefix; the filter data
  // always carries four calendars, so labels are site-qualified and never collide
  // PMF and the calibration per-filter section are Addis-only: one calendar, bare season names
  const addisOnlyTab = tab === 'pmf' || tab === 'calibration'
  const qualify = !addisOnlyTab
  const calendarSites = useMemo(
    () => (addisOnlyTab ? [pmfSiteName] : activeSites ? siteNames.filter((n) => activeSites.has(n)) : siteNames),
    [addisOnlyTab, pmfSiteName, activeSites, siteNames]
  )
  const sharedSeasons = useMemo(
    () => data?.meta.season_conventions[isLocal ? data.meta.season_convention : activeConvention] ?? data?.meta.seasons ?? [],
    [data, isLocal, activeConvention]
  )
  const calendarOf = useCallback(
    (site: string) => (isLocal ? localSeasonsFor(site, data?.meta) ?? sharedSeasons : sharedSeasons),
    [isLocal, data, sharedSeasons]
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
  const baseRows = useMemo(() => {
    if (!data) return []
    return data.filters.rows
      .filter((r) => includeExcluded || !r.excluded)
      .map((r) => {
        const s = seasonForFilter(r)
        return s !== r.season ? { ...r, season: s } : r
      })
  }, [data, includeExcluded, seasonForFilter])

  // The season selection, kept only for seasons that exist under the current
  // calendar and sites; a selection that no longer names any season means "all".
  const seasonSel = useMemo(() => {
    if (!seasonsOn || !activeSeasons) return null
    const names = new Set(seasons.map((se) => se.name))
    const kept = new Set([...activeSeasons].filter((n) => names.has(n)))
    return kept.size === 0 || kept.size === names.size ? null : kept
  }, [seasonsOn, activeSeasons, seasons])

  const rows = useMemo(
    () =>
      baseRows
        .filter(inRange)
        .filter((r) => !seasonSel || seasonSel.has(r.season))
        .filter((r) => !activeSites || activeSites.has(r.site)),
    [baseRows, inRange, seasonSel, activeSites]
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
      const raw = pinnedId && data ? data.filters.rows.find((r) => r.id === pinnedId) : null
      return raw ? { ...raw, season: seasonForFilter(raw) } : null
    },
    [pinnedId, data, seasonForFilter]
  )
  // filter ids the gallery knows, so calibration charts can open the drawer for Addis points
  const knownIds = useMemo(() => new Set((data?.filters.rows ?? []).map((r) => r.id)), [data])
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
      setOpenId(null)
    }
    setOpenRec(rec)
  }, [knownIds])
  const openRow: FilterRow | null = useMemo(
    () => {
      const raw = openId && data ? data.filters.rows.find((r) => r.id === openId) : null
      return raw ? { ...raw, season: seasonForFilter(raw) } : null
    },
    [openId, data, seasonForFilter]
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
  const meta = { ...rawMeta, seasons, season_convention: activeConvention,
    season_site: isLocal && calendarSites.length === 1 ? calendarSites[0] : undefined }

  const nav = (
    <nav className="nav">
      {TABS.map((t) => (
        <button key={t.key} className={`nav-btn ${tab === t.key ? 'active' : ''}`} onClick={() => setTab(t.key)}>
          {t.label}
        </button>
      ))}
    </nav>
  )

  const subset = (() => {
    if (tab === 'census' || tab === 'baseline' || tab === 'similarity' || tab === 'meeting') return null
    const caps = CAPS[tab]
    const pmfMode = tab === 'pmf'
    const calMode = tab === 'calibration'
    // Counts are computed *before* the dimension's own filter is applied,
    // so a chip always shows what selecting it would give you.
    const pool: { season: string; site: string; pmf: string }[] = pmfMode
      ? (data.pmf?.rows ?? []).map((r) => ({ season: seasonOf(pmfSiteName, r.month, r.season), site: pmfSiteName, pmf: pmfLabel(data.pmf, r.dominant_source, pmfScheme) }))
      : calMode
        ? addisEvalDates.map((d) => ({ season: seasonOf(pmfSiteName, d ? Number(d.slice(5, 7)) : null, 'undated'), site: pmfSiteName, pmf: pmfLabelOfDate(d) }))
        : baseRows.filter(inRange).map((r) => ({ season: r.season, site: r.site, pmf: NO_PMF }))
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
    for (const r of seasonPool.filter((r) => pmfMode || calMode || !activeSites || activeSites.has(r.site))) {
      if (r.season in seasonCounts) seasonCounts[r.season] += 1
    }
    const siteCounts: Record<string, number> = {}
    for (const n of siteNames) siteCounts[n] = 0
    for (const r of sitePool) if (r.site in siteCounts) siteCounts[r.site] += 1

    const state: SubsetState = {
      convention: activeConvention,
      defaultConvention: 'local',
      seasons,
      activeSeasons: seasonSel ?? new Set(seasons.map((s) => s.name)),
      seasonCounts,
      sites: siteNames,
      activeSites: activeSites ?? new Set(siteNames),
      siteCounts,
      includeExcluded,
      nExcluded: filters.n_excluded,
      dateRange,
      xField,
      yField,
      field,
      axes,
      pinned,
      pinnedInSubset: !!pinned && rows.some((r) => r.id === pinned.id),
      shown: pmfMode ? pmfRows.length : calMode ? pool.filter(inSeason).filter(inPmf).length : rows.length,
      total: pmfMode ? (data.pmf?.n ?? 0) : calMode ? pool.length : filters.n,
      pmfScheme,
      pmfGroups: pmfGroupList,
      activePmf: pmfSel ?? new Set(pmfGroupList.map((g) => g.name)),
      seasonsOn,
      pmfOn,
      pmfCounts,
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
      onToggleSite: (name) => {
        const cur = new Set(activeSites ?? siteNames)
        if (cur.has(name)) cur.delete(name)
        else cur.add(name)
        setActiveSites(cur.size === 0 || cur.size === siteNames.length ? null : cur)
      },
      onAllSites: () => setActiveSites(null),
      onIncludeExcluded: setIncludeExcluded,
      onDateRange: setDateRange,
      onXField: setXField,
      onYField: setYField,
      onField: setField,
      onAxes: setAxes,
      onUnpin: () => setPinnedId(null),
      onOpenPinned: () => pinnedId && setOpenId(pinnedId),
      onReset: () => {
        setActiveSeasons(null)
        setActiveSites(null)
        setActivePmf(null)
        setSeasonsOn(true)
        setPmfOn(false)
        setDateRange(null)
        setIncludeExcluded(false)
        setConvention(null)
      },
    }
    return <SubsetBar meta={meta} caps={caps} brushRows={baseRows} state={state} actions={actions} />
  })()

  return (
    <PrefsProvider>
    <HighlightProvider pinnedId={pinnedId} setPinned={setPinnedId} openSample={openSample} openRecord={openRecord}>
      {openRow && (
        <SampleDrawer
          row={openRow}
          allRows={filters.rows}
          meta={meta}
          xField={xField}
          yField={yField}
          focusField={field}
          pinned={pinnedId === openRow.id}
          onPin={setPinnedId}
          onOpen={openSample}
          onClose={() => { setOpenId(null); setOpenRec(null) }}
          context={openRec}
        />
      )}
      {!openRow && openRec && <RecordDrawer rec={openRec} onClose={() => setOpenRec(null)} />}
      <Shell nav={nav} wide={tab === 'baseline' || tab === 'calibration' || tab === 'similarity' || tab === 'meeting'}>
        {subset}
        {tab === 'census' && <CensusPage census={census} />}
        {tab === 'similarity' && <SimilarityPage />}
        {tab === 'meeting' && <MeetingFollowupPage />}
        {tab === 'baseline' && <BaselineComparisonPage mac={meta.mac_value} demingLambda={data.calibration?.deming_lambda_mac10 ?? 1} />}
        {tab === 'calibration' && <CalibrationPage calib={data.calibration} runs={data.calibrationRuns} meta={meta} knownIds={knownIds} seasonSel={seasonSel} pmfSel={pmfKeep} pmfLabelOfDate={pmfLabelOfDate} />}
        {tab === 'correlation' && (
          <>
            <Scatterplot rows={rows} meta={meta} xField={xField} yField={yField} axes={axes} />
            <DensityHexbin rows={rows} meta={meta} xField={xField} yField={yField} axes={axes} />
            <ConnectedScatter rows={rows} meta={meta} xField={xField} yField={yField} axes={axes} />
            <Correlogram rows={rows} meta={meta} />
            <SpeciesGraph rows={rows} meta={meta} />
            <AvailabilityHeatmap rows={rows} meta={meta} xField={xField} yField={yField} />
          </>
        )}
        {tab === 'distribution' && (
          <>
            <DistributionPanel rows={rows} meta={meta} field={field} />
            <Histogram rows={rows} meta={meta} field={field} />
            <Ridgeline rows={rows} meta={meta} field={field} />
          </>
        )}
        {tab === 'evolution' && (
          <>
            <Timeseries rows={rows} meta={meta} field={field} />
            <MonthlyBand rows={rows} meta={meta} field={field} />
            <SeasonalClock rows={rows} meta={meta} field={field} />
            <MonthlyHeatmap rows={rows} meta={meta} field={field} />
          </>
        )}
        {tab === 'ranking' && (
          <>
            <CorrelationRanking rows={rows} meta={meta} target={field} />
            <Barplot rows={rows} meta={meta} field={field} />
            <SiteRadar rows={rows} meta={meta} />
            <ParallelCoordinates rows={rows} meta={meta} />
          </>
        )}
        {tab === 'partOfWhole' && <CompositionTreemap rows={rows} meta={meta} />}
        {tab === 'map' && <SiteBubbleMap rows={rows} meta={meta} field={field} />}
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
    </HighlightProvider>
    </PrefsProvider>
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

/** Explain toggle and copy-link, at the right of the title row. */
function HeaderTools() {
  const { explain, setExplain } = usePrefs()
  const [copied, setCopied] = useState(false)
  return (
    <span className="header-tools">
      <label className="toggle" title="Show every chart's explanation and provenance, and the subset bar's captions">
        <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} />
        explain
      </label>
      <button
        type="button"
        className="btn quiet"
        title="The whole view (tab, subset, axes, pinned filter) is in the URL"
        onClick={() => navigator.clipboard?.writeText(window.location.href).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) })}
      >
        {copied ? 'copied' : 'copy link'}
      </button>
    </span>
  )
}
