import { useCallback, useEffect, useMemo, useState } from 'react'
import { useGalleryData } from '@/hooks/useGalleryData'
import { HighlightProvider } from '@/lib/highlight'
import { PrefsProvider, usePrefs } from '@/lib/prefs'
import { listParam, listToParam, readParams, writeParams } from '@/lib/url'
import { Scatterplot } from '@/charts/correlation/Scatterplot'
import { Correlogram } from '@/charts/correlation/Correlogram'
import { AvailabilityHeatmap } from '@/charts/correlation/AvailabilityHeatmap'
import { DensityHexbin } from '@/charts/correlation/DensityHexbin'
import { ConnectedScatter } from '@/charts/correlation/ConnectedScatter'
import { DistributionPanel } from '@/charts/distribution/DistributionPanel'
import { Histogram } from '@/charts/distribution/Histogram'
import { Ridgeline } from '@/charts/distribution/Ridgeline'
import { Timeseries } from '@/charts/evolution/Timeseries'
import { MonthlyBand } from '@/charts/evolution/MonthlyBand'
import { Barplot } from '@/charts/ranking/Barplot'
import { CorrelationRanking } from '@/charts/ranking/CorrelationRanking'
import { SiteRadar } from '@/charts/ranking/SiteRadar'
import { CompositionTreemap } from '@/charts/partOfWhole/CompositionTreemap'
import { SiteBubbleMap } from '@/charts/map/SiteBubbleMap'
import { MethodSankey } from '@/charts/flow/MethodSankey'
import { SourceStack } from '@/charts/pmf/SourceStack'
import { SourceSeasonality } from '@/charts/pmf/SourceSeasonality'
import { CensusPage } from '@/pages/CensusPage'
import { CalibrationPage } from '@/pages/CalibrationPage'
import { SubsetBar, type SubsetActions, type SubsetCaps, type SubsetState } from '@/components/SubsetBar'
import { SampleDrawer } from '@/components/SampleDrawer'
import type { DateRange } from '@/components/DateBrush'
import type { FilterRow } from '@/lib/types'
import { AXES_MODES, DEFAULT_AXES, type AxesMode, type AxesOpts } from '@/lib/axes'

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
] as const

type TabKey = (typeof TABS)[number]['key']
const isTab = (v: string | null): v is TabKey => !!v && TABS.some((t) => t.key === v)

/**
 * What each tab's underlying data can actually support. The PMF solution is
 * ETAD-only and carries no exclusion flags, so offering four site chips and a
 * 942-filter count there would just be wrong.
 */
const FILTERS: SubsetCaps = { seasons: true, sites: true, excluded: true, dates: true, fields: 'single', datasetLabel: 'filters' }
const CAPS: Record<Exclude<TabKey, 'census' | 'calibration'>, SubsetCaps> = {
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
    note: 'ETAD only · 2023 · no exclusion flags on the factor solution',
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
  const openSample = useCallback((id: string) => {
    setPinnedId(id)
    setOpenId(id)
  }, [])

  useEffect(() => {
    writeParams({
      tab,
      cal: convention,
      seasons: listToParam(activeSeasons),
      sites: listToParam(activeSites),
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
  }, [tab, convention, activeSeasons, activeSites, dateRange, includeExcluded, xField, yField, field, pinnedId, axes])

  // A pasted link, the back button, or an edited hash should all apply —
  // not just the hash that was there on first load.
  useEffect(() => {
    const onHash = () => {
      const p = readParams()
      if (isTab(p.get('tab'))) setTab(p.get('tab') as TabKey)
      setConvention(p.get('cal'))
      setActiveSeasons(listParam(p.get('seasons')))
      setActiveSites(listParam(p.get('sites')))
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

  const activeConvention = convention ?? data?.meta.season_convention ?? 'dry_feb'
  const seasons = useMemo(
    () => data?.meta.season_conventions[activeConvention] ?? data?.meta.seasons ?? [],
    [data, activeConvention]
  )
  const siteNames = useMemo(() => data?.meta.sites.map((s) => s.name) ?? [], [data])
  const pmfSiteName = data?.pmf?.site_name ?? 'Addis Ababa'

  // month -> season under the *currently selected* calendar, so switching
  // convention re-labels every chart rather than needing a re-export
  const seasonOfMonth = useMemo(() => {
    const m = new Map<number, string>()
    for (const s of seasons) for (const mo of s.months) m.set(mo, s.name)
    return m
  }, [seasons])

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
        const s = r.month ? seasonOfMonth.get(r.month) : undefined
        return s && s !== r.season ? { ...r, season: s } : r
      })
  }, [data, includeExcluded, seasonOfMonth])

  const rows = useMemo(
    () =>
      baseRows
        .filter(inRange)
        .filter((r) => !activeSeasons || activeSeasons.has(r.season))
        .filter((r) => !activeSites || activeSites.has(r.site)),
    [baseRows, inRange, activeSeasons, activeSites]
  )

  // PMF rows get the same calendar + season treatment as the filter rows, so
  // one season selection means the same thing on every tab.
  const pmfRows = useMemo(() => {
    const src = data?.pmf?.rows ?? []
    return src
      .map((r) => ({ ...r, season: seasonOfMonth.get(r.month) ?? r.season }))
      .filter((r) => !activeSeasons || activeSeasons.has(r.season))
  }, [data, seasonOfMonth, activeSeasons])

  const pinned: FilterRow | null = useMemo(
    () => (pinnedId && data ? data.filters.rows.find((r) => r.id === pinnedId) ?? null : null),
    [pinnedId, data]
  )
  // filter ids the gallery knows, so calibration charts can open the drawer for Addis points
  const knownIds = useMemo(() => new Set((data?.filters.rows ?? []).map((r) => r.id)), [data])
  const openRow: FilterRow | null = useMemo(
    () => (openId && data ? data.filters.rows.find((r) => r.id === openId) ?? null : null),
    [openId, data]
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
  const meta = { ...rawMeta, seasons, season_convention: activeConvention }

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
    if (tab === 'census' || tab === 'calibration') return null
    const caps = CAPS[tab]
    const pmfMode = tab === 'pmf'
    // Counts are computed *before* the dimension's own filter is applied,
    // so a chip always shows what selecting it would give you.
    const seasonPool = pmfMode
      ? (data.pmf?.rows ?? []).map((r) => ({ season: seasonOfMonth.get(r.month) ?? r.season, site: pmfSiteName }))
      : baseRows.filter(inRange).map((r) => ({ season: r.season, site: r.site }))
    const sitePool = seasonPool.filter((r) => !activeSeasons || activeSeasons.has(r.season))

    const seasonCounts: Record<string, number> = {}
    for (const s of seasons) seasonCounts[s.name] = 0
    for (const r of seasonPool.filter((r) => !activeSites || activeSites.has(r.site))) {
      if (r.season in seasonCounts) seasonCounts[r.season] += 1
    }
    const siteCounts: Record<string, number> = {}
    for (const n of siteNames) siteCounts[n] = 0
    for (const r of sitePool) if (r.site in siteCounts) siteCounts[r.site] += 1

    const state: SubsetState = {
      convention: activeConvention,
      seasons,
      activeSeasons: activeSeasons ?? new Set(seasons.map((s) => s.name)),
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
      shown: pmfMode ? pmfRows.length : rows.length,
      total: pmfMode ? (data.pmf?.n ?? 0) : filters.n,
    }
    const actions: SubsetActions = {
      onConvention: (v) => {
        setConvention(v)
        setActiveSeasons(null)
      },
      onToggleSeason: (name) => {
        const cur = new Set(activeSeasons ?? seasons.map((s) => s.name))
        if (cur.has(name)) cur.delete(name)
        else cur.add(name)
        setActiveSeasons(cur.size === 0 || cur.size === seasons.length ? null : cur)
      },
      onAllSeasons: () => setActiveSeasons(null),
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
        setDateRange(null)
        setIncludeExcluded(false)
        setConvention(null)
      },
    }
    return <SubsetBar meta={meta} caps={caps} brushRows={baseRows} state={state} actions={actions} />
  })()

  return (
    <PrefsProvider>
    <HighlightProvider pinnedId={pinnedId} setPinned={setPinnedId} openSample={openSample}>
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
          onClose={() => setOpenId(null)}
        />
      )}
      <Shell nav={nav}>
        {subset}
        {tab === 'census' && <CensusPage census={census} />}
        {tab === 'calibration' && <CalibrationPage calib={data.calibration} runs={data.calibrationRuns} meta={meta} knownIds={knownIds} />}
        {tab === 'correlation' && (
          <>
            <Scatterplot rows={rows} meta={meta} xField={xField} yField={yField} axes={axes} />
            <DensityHexbin rows={rows} meta={meta} xField={xField} yField={yField} axes={axes} />
            <ConnectedScatter rows={rows} meta={meta} xField={xField} yField={yField} axes={axes} />
            <Correlogram rows={rows} meta={meta} />
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
          </>
        )}
        {tab === 'ranking' && (
          <>
            <CorrelationRanking rows={rows} meta={meta} target={field} />
            <Barplot rows={rows} meta={meta} field={field} />
            <SiteRadar rows={rows} meta={meta} />
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

function Shell({ children, nav }: { children: React.ReactNode; nav?: React.ReactNode }) {
  return (
    <>
      <header className="app-header">
        <div className="container">
          <div className="app-title-row">
            <h1 className="app-title">Aethmodular graph gallery</h1>
            <HeaderTools />
          </div>
          {nav}
        </div>
      </header>
      <div className="container">{children}</div>
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
