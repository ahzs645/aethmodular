import { lazyView } from '@/components/LazyView'
import { RegressionProvider, RegressionControls, parseEstimators, estimatorParam, type Estimator } from '@/lib/regressionPrefs'
import { startTransition, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useGalleryData } from '@/hooks/useGalleryData'
import { replacePublicFilter, supplementFilter } from '@/lib/filterMerge'
import { useSpartanSource } from '@/lib/spartanSource'
import { SpartanDataProvider } from '@/components/SpartanDataSettings'
import { HighlightProvider, baseFilterId, type FilterRecord, useHighlightActions } from '@/lib/highlight'
import { NO_PMF, PMF_GROUP, PMF_SOURCE, isPmfField, pmfGroups, pmfKeyByDate, pmfLabel, pmfShareFields, withPmf, type PmfScheme } from '@/lib/pmfGroups'
import { PrefsProvider } from '@/lib/prefs'
import { PanelAxesProvider } from '@/lib/panelAxes'
import { listParam, listToParam, readParams, writeParams } from '@/lib/url'
import { useShared } from '@/lib/stable'
import { PAPER_CALENDAR } from '@/lib/pmfPaper'
import { NO_WEATHER, isDatedCalendar, seasonOfDate } from '@/lib/seasons'
import { DEFAULT_RAIN_RULE, RAIN_SEASONS, parseRainRule, rainRuleParam, type RainRule } from '@/lib/rainSeasons'
import { useRainCalendars } from '@/hooks/useRainCalendars'
import { AVAILABILITY_VIEWS, AVAILABILITY_VIEW_PARAM, DataAvailabilityPage } from '@/pages/DataAvailabilityPage'
import { InvestigationsPage, type InvestigationArea } from '@/pages/InvestigationsPage'
import { AreaHeader, HUB_TAB, type AreaPage } from '@/components/AreaHeader'
import { SiteBubbleMap, SpartanCompositionMap } from '@/charts/map/lazy'
import { SPEC_MEETINGS, meetingLabel } from '@/meetings'
import { DEFAULT_EXPERIMENT, EXPERIMENTS, LISTED_EXPERIMENTS } from '@/experiments'
import { DashboardPage } from '@/pages/DashboardPage'
import { dashParams, parseDash, type DashState, type Scope } from '@/lib/dashboard'
import type { DashFilters } from '@/components/dashboard/DashHeader'
import { AxisFieldsProvider, type AxisRole } from '@/components/AxisFieldPopover'
import { ViewsMenu } from '@/components/ViewsMenu'
import { HeaderMore } from '@/components/HeaderMore'
import { currentView, isEmbed, viewUrl } from '@/lib/views'
import { useExport } from '@/lib/data'
import { customSeasonMeta, customSeasonsParam, parseCustomSeasons } from '@/components/SeasonEditor'
import { SideDock, SubsetBar, type SubsetActions, type SubsetCaps, type SubsetState } from '@/components/SubsetBar'
import { ExportSitesProvider } from '@/lib/exportSites'
import { SampleDrawer } from '@/components/SampleDrawer'
import { RecordDrawer } from '@/components/RecordDrawer'
import type { DateRange } from '@/components/DateBrush'
import type { FilterRow, SeasonMeta, SiteMeta } from '@/lib/types'
import { AXES_MODES, DEFAULT_AXES, PAX_PARAM, paxFromParam, type AxesMode, type AxesOpts } from '@/lib/axes'
import { calendarNote, calendarVariants, hasLocalCalendar, localSeasonsFor, qualifySeason, seasonSources } from '@/siteSeasons'
import { dynamicPlaceholder, parseSiteRules, siteRulesParam, type SiteRules } from '@/lib/dynamicSeasons'
import { useDynamicCalendars } from '@/hooks/useDynamicCalendars'
import { MAX_IMPROVE_SITES, MONTH_QUARTERS, NETWORK_DEFAULTS, hasImprove, hasSpartan, isNetwork, mergeGroups, networkColor, useNetworkData, type Network } from '@/lib/networks'
import { siteAka, siteCountry, type SiteItem } from '@/components/SitePicker'
import { keepUnderSsr, parseSsrHide, ssrCounts, type SsrHide } from '@/lib/ssrFilters'
import { FilterEditsProvider, HandEditsBadge, type FilterGroup } from '@/lib/filterEdits'
import { exclModeParam, exclusionReason, exportIn, parseExclMode, type ExclMode } from '@/lib/exclusionMode'
import { MarkedRowsProvider } from '@/lib/markedRows'
import { GroupDrawer } from '@/components/GroupDrawer'
import { HIDDEN_DUPLICATES } from '@/fieldLineage'
import { PageSidebarProvider, PageSidebarSlot, SidebarControls, usePageSidebarNote } from '@/lib/pageSidebar'
import { RATIO_GROUP, RATIO_PRESETS, VOLUME, parseRatios, ratioKey, ratiosParam, viewRows, viewUnits, type Basis, type RatioDef } from '@/lib/viewFields'
import { RatiosProvider } from '@/components/RatioDialog'
import { GroupingsProvider } from '@/components/GroupingDialog'
import { groupingsParam, mergeGroupings, parseGroupings, readSavedGroupings, writeSavedGroupings, type GroupingDef } from '@/lib/groupings'
import { GlossaryButton } from '@/components/Glossary'
import { PresentationProvider, PresentationSwitch, type Presentation } from '@/lib/presentation'
import { LAMBDA_10, MacProvider, demingLambda } from '@/lib/mac'
import { useSearchParam } from '@/lib/url'
import { CommandPaletteButton } from '@/components/CommandPalette'
import { UndoToast } from '@/components/UndoToast'
import { HeaderNavProvider, useHeaderNav } from '@/lib/headerNav'
import { NavMenu, type NavGroup, type NavItem } from '@/components/NavMenu'
import { SidebarIcon, SingleChartIcon, XIcon } from '@/components/Icons'
import { DisplayControls } from '@/components/DisplayControls'
import { SingleFit, SingleViewProvider, SingleViewToggle, useSingleViewPref } from '@/lib/singleView'
import { WEATHER_GROUP, useWeather } from '@/lib/weather'
import { AERONET_GROUP, useAeronet } from '@/lib/aeronet'
// flagged sections (src/sections/sections.json): a public build resolves a private one to sections/omitted.ts
import SECTIONS_MANIFEST from '@/sections/sections.json'
import instruments from '@/sections/instruments'
import research from '@/sections/research'
import ml from '@/sections/ml'
import { defaultPage } from '@/sections/types'
import { useHeaderHeight, useSidePanelOpen } from '@/hooks/useSidePanel'
import { EQUIVALENTS } from '@/lib/fieldEquivalents'
import { ChartBuilderPage } from '@/pages/ChartBuilderPage'
import { useChartBuilderEnabled } from '@/lib/chartBuilderPrefs'

const CensusPage = lazyView(() => import('@/pages/CensusPage').then(m => m.CensusPage))
const CalibrationPage = lazyView(() => import('@/pages/CalibrationPage').then(m => m.CalibrationPage))
const BaselineComparisonPage = lazyView(() => import('@/pages/BaselineComparisonPage').then(m => m.BaselineComparisonPage))
const SimilarityPage = lazyView(() => import('@/pages/SimilarityPage').then(m => m.SimilarityPage))
const MeetingFollowupPage = lazyView<NonNullable<Parameters<typeof import('@/pages/MeetingFollowupPage')['MeetingFollowupPage']>[0]>>(() => import('@/pages/MeetingFollowupPage').then(m => m.MeetingFollowupPage))
const SatoshiFollowupPage = lazyView(() => import('@/pages/SatoshiFollowupPage').then(m => m.SatoshiFollowupPage))
const MeetingSpecPage = lazyView(() => import('@/pages/MeetingSpecPage').then(m => m.MeetingSpecPage))
const BlanksPage = lazyView(() => import('@/pages/BlanksPage').then(m => m.BlanksPage))
const ExperimentalPage = lazyView(() => import('@/pages/ExperimentalPage').then(m => m.ExperimentalPage))
const WeatherPage = lazyView(() => import('@/pages/WeatherPage').then(m => m.WeatherPage))
const AeronetPage = lazyView(() => import('@/pages/AeronetPage').then(m => m.AeronetPage))

const Scatterplot = lazyView(() => import('@/charts/correlation/Scatterplot').then(m => m.Scatterplot))
const Correlogram = lazyView(() => import('@/charts/correlation/Correlogram').then(m => m.Correlogram))
const AvailabilityHeatmap = lazyView(() => import('@/charts/correlation/AvailabilityHeatmap').then(m => m.AvailabilityHeatmap))
const DensityHexbin = lazyView(() => import('@/charts/correlation/DensityHexbin').then(m => m.DensityHexbin))
const ConnectedScatter = lazyView(() => import('@/charts/correlation/ConnectedScatter').then(m => m.ConnectedScatter))
const SpeciesGraph = lazyView(() => import('@/charts/correlation/SpeciesGraph').then(m => m.SpeciesGraph))
const DistributionPanel = lazyView(() => import('@/charts/distribution/DistributionPanel').then(m => m.DistributionPanel))
const Histogram = lazyView(() => import('@/charts/distribution/Histogram').then(m => m.Histogram))
const Ridgeline = lazyView(() => import('@/charts/distribution/Ridgeline').then(m => m.Ridgeline))
const Timeseries = lazyView(() => import('@/charts/evolution/Timeseries').then(m => m.Timeseries))
const MonthlyBand = lazyView(() => import('@/charts/evolution/MonthlyBand').then(m => m.MonthlyBand))
const SeasonalClock = lazyView(() => import('@/charts/evolution/SeasonalClock').then(m => m.SeasonalClock))
const MonthlyHeatmap = lazyView(() => import('@/charts/evolution/MonthlyHeatmap').then(m => m.MonthlyHeatmap))
const Barplot = lazyView(() => import('@/charts/ranking/Barplot').then(m => m.Barplot))
const CorrelationRanking = lazyView(() => import('@/charts/ranking/CorrelationRanking').then(m => m.CorrelationRanking))
const SiteRadar = lazyView(() => import('@/charts/ranking/SiteRadar').then(m => m.SiteRadar))
const ParallelCoordinates = lazyView(() => import('@/charts/ranking/ParallelCoordinates').then(m => m.ParallelCoordinates))
const CompositionTreemap = lazyView(() => import('@/charts/partOfWhole/CompositionTreemap').then(m => m.CompositionTreemap))
const FunctionalGroups = lazyView(() => import('@/charts/partOfWhole/FunctionalGroups').then(m => m.FunctionalGroups))
const MethodSankey = lazyView(() => import('@/charts/flow/MethodSankey').then(m => m.MethodSankey))
const SourceStack = lazyView(() => import('@/charts/pmf/SourceStack').then(m => m.SourceStack))
const SplitFit = lazyView(() => import('@/charts/correlation/SplitFit').then(m => m.SplitFit))
const NetworkRankPlacement = lazyView(() => import('@/charts/ranking/NetworkRankPlacement').then(m => m.NetworkRankPlacement))
const PmfDiagnostics = lazyView(() => import('@/charts/pmf/PmfDiagnostics').then(m => m.PmfDiagnostics))

/**
 * Tabs mirror the react-graph-gallery categories, plus the census. `hidden` tabs
 * stay out of the tab row but still open from a link (#tab=census); the row shows
 * one while it is the open tab.
 */
const TABS = [
  { key: 'census', label: 'Census', title: 'What the notebooks already draw, mapped onto the gallery’s chart types' },
  { key: 'dashboard', label: 'Dashboard', title: 'Your own page of gallery charts: add, arrange and size tiles' },
  { key: 'builder', label: 'Chart builder', title: 'Free-form measures and facets with Graphic Walker' },
  { key: 'correlation', label: 'Correlation' },
  { key: 'distribution', label: 'Distribution' },
  { key: 'evolution', label: 'Evolution' },
  { key: 'ranking', label: 'Ranking' },
  { key: 'partOfWhole', label: 'Composition', title: 'What the PM2.5 is made of, and the FTIR organic functional groups, per site and season' },
  { key: 'map', label: 'Map' },
  { key: 'investigations', label: 'Investigations', title: 'Cards for the work beyond the chart tabs: Meetings, Experimental, Research diagnostics, Aethalometer and Data availability' },
  { key: 'availability', label: 'Data availability', title: 'Notebook coverage analyses, product inventory, sample-day overlap and filter history' },
  { key: 'weather', label: 'Weather', title: 'Weather over each filter’s sampling window: ERA5 at every SPARTAN site, the airport station at Addis' },
  { key: 'aethalometer', label: 'Aethalometer', title: 'MA350 aethalometers at the focus sites (Instruments section; unlisted in public builds)' },
  { key: 'research', label: 'Research diagnostics', title: 'Notebook analyses brought into the gallery (Research section; private in public builds)' },
  { key: 'ml', label: 'Machine learning', title: 'The notebooks’ model-fitting results, as they ran: static (Machine learning section; private in public builds)' },
  { key: 'aeronet', label: 'AERONET', title: 'The sun photometer column over each filter’s sampling window: AOD, absorption AOD and SSA at every SPARTAN site with a station' },
  { key: 'flow', label: 'Flow' },
  { key: 'pmf', label: 'PMF sources' },
  { key: 'calibration', label: 'Calibration' },
  { key: 'baseline', label: 'Baselines', title: 'AIRSpec spline vs VIBES baseline correction' },
  { key: 'similarity', label: 'Spectral similarity' },
  { key: 'meeting', label: 'Meeting', title: 'Meeting follow-ups and Satoshi’s 2 Oct questions' },
  { key: 'blanks', label: 'Blanks', title: 'PTFE field blanks behind the VIBES background' },
  { key: 'experimental', label: 'Experimental', title: 'Analyses still being worked out: what the FTIR EC flag sees, and where it is blind' },
] as const

type TabKey = (typeof TABS)[number]['key']
/** A share build (VITE_SHARE_TABS=meeting,map; gallery/share/build_share.py) offers only these tabs, the first as landing tab. */
const SHARE_TABS: string[] = (import.meta.env.VITE_SHARE_TABS ?? '').split(',').map((t: string) => t.trim()).filter(Boolean)
/** the flagged sections compiled into this build, their tabs, and every tab a section owns (compiled or not) */
const SECTIONS = [instruments, research, ml].filter((s) => s.tabs.length)
const SECTION_TABS = new Map(SECTIONS.flatMap((s) => s.tabs.map((t) => [t.key, t] as const)))
const SECTION_OWNED = new Set<string>(Object.values(SECTIONS_MANIFEST.sections).flatMap((s) => s.tabs))
const shared = (key: string) => (!SECTION_OWNED.has(key) || SECTION_TABS.has(key)) && (SHARE_TABS.length === 0 || SHARE_TABS.includes(key))
/** a public build (VITE_PUBLIC_BUILD=1) leaves the Experimental pages out of the menu and the Investigations cards:
 *  they open only from a link (#tab=experimental), and the menu lists them only while one is open.
 *  VITE_SHOW_EXPERIMENTAL=1 lists them there too; any other build always lists them. */
const SHOW_EXPERIMENTAL = import.meta.env.VITE_PUBLIC_BUILD !== '1' || import.meta.env.VITE_SHOW_EXPERIMENTAL === '1'
/** a public build's tabs of sections marked "unlisted": compiled, but like Experimental they open from a link
 *  (#tab=aethalometer), and the menu shows their group only while one is open. Any other build lists them. */
const UNLISTED = new Set<string>(import.meta.env.VITE_PUBLIC_BUILD === '1'
  ? Object.values(SECTIONS_MANIFEST.sections).filter((s) => s.public === 'unlisted').flatMap((s) => s.tabs) : [])
/** a public build also leaves the Census out of the menu, the search and the Investigations cards (#tab=census still opens it) */
const PUBLIC_UNLISTED = import.meta.env.VITE_PUBLIC_BUILD === '1' ? ['census'] : []
/** tabs the menu, the ⌘K search and the Investigations cards leave out unless one is open */
const UNLISTED_TABS = new Set<string>([...UNLISTED, ...PUBLIC_UNLISTED, ...(SHOW_EXPERIMENTAL ? [] : ['experimental'])])
const isTab = (v: string | null): v is TabKey => !!v && shared(v) && TABS.some((t) => t.key === v)
const LANDING_TAB = (SHARE_TABS.find((t) => TABS.some((x) => x.key === t)) ?? 'correlation') as TabKey

// the hand-built meeting pages and every declared one (src/meetings/*.meeting.ts), newest first, each named by its date
const MEETINGS: { key: string; label: string; title?: string }[] = [
  { key: 'followup', date: '2026-09-17', title: 'The 17 Sep meeting’s follow-ups: the analog bias check and the figures behind it' },
  { key: 'satoshi', date: '2026-10-02', title: 'Satoshi’s 2 Oct questions: analog overlap, the swap test, VIP, the fit ladder and the envelope' },
  ...SPEC_MEETINGS.map((m) => ({ key: m.id, date: m.date, title: m.title })),
].sort((a, b) => b.date.localeCompare(a.date)).map(({ date, ...m }) => ({ ...m, label: meetingLabel(date) }))
type MeetingKey = string
/** the header's page menu (components/NavMenu): tabs in groups, the meeting pages under Meetings */
const tabItem = (key: string): NavItem[] => TABS.filter((t) => t.key === key && shared(t.key))
  .map((t) => ({ key: t.key, label: t.label, title: 'title' in t ? t.title : undefined, tab: t.key }))
/**
 * The Investigations tab's areas, in card and menu order: tabs with pages of their own (each drawn under
 * components/AreaHeader). Every compiled section's tab is one, so a new section joins without a change here.
 */
const AREA_TABS = [...new Set(['meeting', 'experimental', 'research', 'ml', 'aethalometer', ...SECTION_TABS.keys(), 'availability', 'census'])].filter(shared)
const areaPages = (t: string): readonly AreaPage[] =>
  t === 'meeting' ? MEETINGS : t === 'experimental' ? LISTED_EXPERIMENTS : t === 'availability' ? AVAILABILITY_VIEWS : t === 'census' ? CENSUS_PAGES
    : (SECTION_TABS.get(t)?.pages ?? []).filter((p) => !p.hidden)
/** the Census is one page */
const CENSUS_PAGES: AreaPage[] = [{ key: 'census', label: 'Notebook chart census', title: 'Every figure-producing notebook cell, classified by chart type, against what the gallery already draws' }]
const tabTitle = (t: string) => { const x = TABS.find((y) => y.key === t); return x && 'title' in x ? x.title : '' }
const AREA_NAMES: Record<string, { label: string; about: string }> = {
  census: { label: 'Census', about: 'The notebook estate’s figures classified by chart type and mapped onto the react-graph-gallery taxonomy: what the gallery already has, and what it adds' },
  meeting: { label: 'Meetings', about: 'A page for each meeting: the figures shown, the questions asked and their follow-ups' },
}
const AREAS: InvestigationArea[] = AREA_TABS.flatMap((t) => areaPages(t).length ? [{
  tab: t, label: AREA_NAMES[t]?.label ?? SECTION_TABS.get(t)?.label ?? TABS.find((x) => x.key === t)?.label ?? t,
  about: AREA_NAMES[t]?.about ?? SECTION_TABS.get(t)?.title ?? tabTitle(t), pages: areaPages(t),
}] : [])
/** a page of an area in the menu: Meetings picks its page as `meeting`, every other area as `sub` */
const areaItem = (a: InvestigationArea, p: AreaPage, heading?: string): NavItem => ({
  key: `${a.tab}-${p.key}`, label: heading ? p.label : a.label, title: heading ? p.title : a.about, tab: a.tab, heading,
  // a one-page area (Census) has no sub-page to match
  ...(a.tab === 'meeting' ? { meeting: p.key } : a.pages.length > 1 ? { sub: p.key } : {}),
})
const NAV_GROUPS: NavGroup[] = [
  { label: 'Charts', items: ['dashboard', 'correlation', 'distribution', 'evolution', 'ranking', 'partOfWhole', 'pmf', 'flow', 'builder'].flatMap(tabItem) },
  { label: 'Sites', items: ['map', 'weather', 'aeronet'].flatMap(tabItem) },
  { label: 'Calibration', items: ['calibration', 'baseline', 'similarity', 'blanks'].flatMap(tabItem) },
].filter((g) => g.items.length)
/** the Investigations cards: the areas this build lists */
const HUB_AREAS = AREAS.filter((a) => !UNLISTED_TABS.has(a.tab))
const HUB_ITEM: NavItem[] = tabItem(HUB_TAB).map((i) => ({ ...i, label: 'All investigations', home: true }))
/**
 * The Investigations group, which depends on where the reader is: inside an area, the cards and that
 * area's own pages (under its name); anywhere else, the cards and the areas, each opening on its first page.
 */
const investigationsGroup = (tab: string): NavGroup => {
  const open = AREAS.find((a) => a.tab === tab)
  return { label: 'Investigations', items: open
    ? [...HUB_ITEM, ...open.pages.map((p) => areaItem(open, p, open.label))]
    : [...HUB_ITEM, ...HUB_AREAS.map((a) => areaItem(a, a.pages[0]))] }
}
/** the menu for the open tab: unlisted tabs (UNLISTED_TABS) only while one of them is open */
const navGroups = (tab: string, builderEnabled: boolean) => [...NAV_GROUPS, investigationsGroup(tab)]
  .map((g) => ({ ...g, items: g.items.filter((i) => i.tab === tab || !UNLISTED_TABS.has(i.tab) && (i.tab !== 'builder' || builderEnabled)) }))
  .filter((g) => g.items.length)
const AVAILABILITY_KEYS = AVAILABILITY_VIEWS.map((v) => v.key)

/** a row of a site without PMF, which the PMF filter passes through */
const OTHER_SITE = '__other_site'

/** what the ⌘K search offers: the tabs this build shows (lib/headerNav) */
const NAV_TABS = TABS.filter((t) => shared(t.key) && !('hidden' in t && t.hidden) && !UNLISTED_TABS.has(t.key)).map((t) => ({ key: t.key as string, label: t.label as string, title: 'title' in t ? t.title as string : undefined,
  group: NAV_GROUPS.find((g) => g.items.some((i) => i.tab === t.key))?.label ?? 'Investigations' }))
// Keep existing #tab=satoshi links working under the shared Meeting tab.
const tabFromParams = (p: URLSearchParams): TabKey =>
  p.get('tab') === 'satoshi' && shared('meeting') ? 'meeting' : isTab(p.get('tab')) ? p.get('tab') as TabKey : LANDING_TAB
const experimentFromParams = (p: URLSearchParams): string => EXPERIMENTS.find((e) => e.key === p.get('xp'))?.key ?? DEFAULT_EXPERIMENT
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

/**
 * A site's own seasons, edited by the reader (the ✎ in a site's season section): code -> bins. In the
 * URL as sse={"ETAD":[["Bega",[10,11,12,1]],…]} and saved in this browser, so they come back
 * without the link; a link that carries its own wins.
 */
type SiteSeasons = Record<string, { name: string; months: number[] }[]>
const SITE_SEASONS_KEY = 'gallery.siteSeasons'
const parseSiteSeasons = (v: string | null): SiteSeasons | null => {
  try {
    const o = v ? JSON.parse(v) : null
    if (!o || typeof o !== 'object' || Array.isArray(o)) return null
    return Object.fromEntries(Object.entries(o).flatMap(([c, bins]) => { const b = parseCustomSeasons(JSON.stringify(bins)); return b ? [[c, b]] : [] }))
  } catch { return null }
}
const siteSeasonsParam = (v: SiteSeasons) => (Object.keys(v).length ? JSON.stringify(Object.fromEntries(Object.entries(v).map(([c, b]) => [c, JSON.parse(customSeasonsParam(b)!)]))) : null)
const readSavedSiteSeasons = (): SiteSeasons => { try { return parseSiteSeasons(localStorage.getItem(SITE_SEASONS_KEY)) ?? {} } catch { return {} } }
/** a site's dynamic seasons (lib/dynamicSeasons), in the link as dyn= and saved in this browser like sse */
const SITE_RULES_KEY = 'gallery.siteRules'
const readSavedSiteRules = (): SiteRules => { try { return parseSiteRules(localStorage.getItem(SITE_RULES_KEY)) ?? {} } catch { return {} } }
/** "Bega (Oct-Jan, dry)" -> "Bega": the editor adds the month range back from the months */
const bareSeasonName = (name: string) => name.replace(/\s*\([^)]*\)\s*$/, '') || name

/** FTIR EC under each spectral baseline: none is the SPARTAN product in filters.json */
type FtirBaseline = 'none' | 'spline' | 'vibes'
const FTIR_VARIANT = { spline: 'EC (FTIR, spline)', vibes: 'EC (FTIR, VIBES)' } as const

/** a comma-separated id list from the URL (filters dropped or kept by hand) */
const idSet = (v: string | null): Set<string> | null => (v ? new Set(v.split(',').filter(Boolean)) : null)
/** `dropwhy` in the URL: a JSON object, filter id -> the reader's reason for excluding it by hand */
const parseWhy = (v: string | null): Record<string, string> => {
  try { const o = v ? JSON.parse(v) : null; return o && typeof o === 'object' && !Array.isArray(o) ? Object.fromEntries(Object.entries(o).filter(([, r]) => typeof r === 'string')) as Record<string, string> : {} } catch { return {} }
}

/** `ssites` in the URL: absent = the four focus sites, * = every SPARTAN site, - = none, else codes */
/** `mac` in the URL: a MAC for HIPS BC other than the export's (null: the export's) */
const parseMac = (v: string | null): number | null => { const n = Number(v); return v && Number.isFinite(n) && n > 0 ? n : null }

const parseSpartanSel = (v: string | null): string[] | 'all' | null =>
  v == null || v === '' ? null : v === '*' ? 'all' : v === '-' ? [] : v.split(',').filter(Boolean)

/** tabs whose charts can draw a whole network (Filters sidebar → network) */
const NETWORK_TABS = ['dashboard', 'builder', 'correlation', 'distribution', 'evolution', 'ranking', 'partOfWhole', 'map', 'weather', 'aeronet', 'aethalometer', 'research'] as const
const FILTERS: SubsetCaps = { seasons: true, sites: true, excluded: true, dates: true, fields: 'single', datasetLabel: 'SPARTAN filters' }
const CAPS: Record<Exclude<TabKey, 'investigations' | 'availability' | 'census' | 'baseline' | 'similarity' | 'meeting' | 'blanks' | 'experimental' | 'ml'>, SubsetCaps> = {
  correlation: { ...FILTERS, fields: 'pair', networks: true, fits: true },
  // each tile picks its own fields
  dashboard: { ...FILTERS, fields: 'none', networks: true, fits: true },
  builder: { ...FILTERS, fields: 'none', networks: true },
  distribution: { ...FILTERS, networks: true },
  evolution: { ...FILTERS, networks: true },
  ranking: { ...FILTERS, networks: true },
  partOfWhole: { ...FILTERS, fields: 'none', networks: true },
  weather: { ...FILTERS, networks: true, fits: true },
  aeronet: { ...FILTERS, networks: true, fits: true },
  aethalometer: { ...FILTERS, networks: true, fits: true },
  research: { ...FILTERS, networks: true, fits: true },
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

/** the rain rule's seasons before a site's rain has loaded (or where it has none): its usual Addis months */
const RAIN_PLACEHOLDER: SeasonMeta[] = RAIN_SEASONS.map((s) => ({ name: s.name, color: s.color, months: [...s.months] }))
// a weather calendar before (or without) its weather: the names, no dated spans, so nothing is classified yet
const RAIN_PENDING: SeasonMeta[] = RAIN_SEASONS.map((s) => ({ name: s.name, color: s.color, months: [...s.months], spans: [] }))
// where a weather calendar cannot place a filter (no weather loaded, none in the record, or a long gap)
/** no excluded filters to mark (one array, so the charts' memos hold) */
const NO_ROWS: FilterRow[] = []
const NO_WEATHER_SEASON: SeasonMeta = { name: NO_WEATHER, color: '#9aa5b1', months: [], spans: [] }
const withNoWeather = (cal: SeasonMeta[]) => [...cal, NO_WEATHER_SEASON]


/**
 * How many rows measure `field`, counted once per rows array and field: App rebuilds the charts' meta on every
 * render (a pin, a hover that opens a record), and recounting every field over every row each time cost ~65 ms.
 */
const measuredCounts = new WeakMap<readonly FilterRow[], Map<string, number>>()
function measuredCount(rows: readonly FilterRow[], field: string): number {
  let counts = measuredCounts.get(rows)
  if (!counts) measuredCounts.set(rows, counts = new Map())
  let n = counts.get(field)
  if (n === undefined) {
    n = 0
    for (const r of rows) { const v = r[field]; if (typeof v === 'number' && Number.isFinite(v)) n++ }
    counts.set(field, n)
  }
  return n
}
export default function App() {
  const builderEnabled = useChartBuilderEnabled()
  const navTabs = useMemo(() => NAV_TABS.filter(t => t.key !== 'builder' || builderEnabled), [builderEnabled])
  const init = useMemo(() => readParams(), [])
  const [tab, setTab] = useState<TabKey>(() => tabFromParams(init))
  const [dashboardCalibration, setDashboardCalibration] = useState(false)
  const { data, error, calibrationLoading, calibrationError, retryCalibration } = useGalleryData(
    ['calibration', 'baseline'].includes(tab) ? true : tab === 'dashboard' && dashboardCalibration ? 'runs' : false)
  // spline / VIBES FTIR EC (gallery/data/export_baseline_ftir_ec.py): the Addis test set from the
  // frozen runs, every other SPARTAN PM2.5 filter with a spectrum from the network baselines; optional
  const baselines = useExport<{ rows: Record<string, { spline?: number; vibes?: number }> }>('baseline_ftir_ec.json').data
  const focusRows = useMemo(() => {
    const rows = data?.filters.rows ?? []
    if (!baselines) return rows
    return rows.map((r) => {
      const b = baselines.rows[r.id]
      return b ? { ...r, [FTIR_VARIANT.spline]: b.spline ?? null, [FTIR_VARIANT.vibes]: b.vibes ?? null } : r
    })
  }, [data, baselines])

  // ---- state, seeded from the URL hash so a pasted link reopens the same view
  const [meeting, setMeeting] = useState<MeetingKey>(() => meetingFromParams(init))
  // the ⌘K search's way to a tab or meeting page (lib/headerNav)
  const [experiment, setExperiment] = useState<string>(() => experimentFromParams(init))
  // a flagged section's sub-page (URL `aep`; empty = its first page)
  const [sectionPage, setSectionPage] = useState<string>(() => init.get('aep') ?? '')
  // the Data availability view (pages/DataAvailabilityPage keeps it in the query; this is the same key, for the menu)
  const [availView, setAvailView] = useSearchParam<string>(AVAILABILITY_VIEW_PARAM, AVAILABILITY_KEYS[0], AVAILABILITY_KEYS)
  const setAvailRef = useRef(setAvailView)
  setAvailRef.current = setAvailView
  const openTab = useCallback((t: string, m?: string, x?: string) => {
    if (isTab(t)) setTab(t)
    if (m) setMeeting(m)
    if (!x) return
    if (t === 'meeting') setMeeting(x)
    else if (SECTION_TABS.has(t)) setSectionPage(x)
    else if (t === 'availability') setAvailRef.current(x)
    else if (t === 'experimental') setExperiment(x)
  }, [])
  // Mirrors get_clean_data: excluded samples are hidden by default but never
  // deleted, so you can always put them back and see what was removed. Or the
  // reader draws them as × left out of every statistic (lib/exclusionMode):
  // excl=1 includes them, excl=mark marks them (hand exclusions too)
  const [exclMode, setExclModeState] = useState<ExclMode>(() => parseExclMode(init.get('excl')))
  // the mode to go back to when one of the export's exclusions is left out again under include
  const exclOff = useRef<ExclMode>(exclMode === 'include' ? 'hide' : exclMode)
  const setExclMode = useCallback((m: ExclMode) => { if (m !== 'include') exclOff.current = m; setExclModeState(m) }, [])
  const includeExcluded = exclMode === 'include'
  // or only some of them, put back one at a time from the side panel (incl=ID,ID)
  const [inclIds, setInclIds] = useState<Set<string>>(() => idSet(init.get('incl')) ?? new Set())
  /** is a row in view as far as the export's exclusions go */
  const exclIn = useCallback((r: FilterRow) => exportIn(r, { mode: exclMode, inclIds }), [exclMode, inclIds])
  const [convention, setConvention] = useState<string | null>(init.get('cal'))
  // the reader's own month bins (calendar "custom"), shared by every site; in the URL as cs
  const [customSeasons, setCustomSeasons] = useState(() => parseCustomSeasons(init.get('cs')) ?? DEFAULT_CUSTOM)
  // the rain rule's settings (lib/rainSeasons), in the URL as rr=window,onset,heavy
  const [rainRule, setRainRule] = useState<RainRule>(() => parseRainRule(init.get('rr')) ?? DEFAULT_RAIN_RULE)
  // a site's alternative calendar where the sources disagree (CODC Bogotá / Palmira, Taiwan review / outlook); in the URL as cv=CODC:palmira,…
  const [calVariants, setCalVariants] = useState<Record<string, string>>(() => parseVariants(init.get('cv')))
  const [siteSeasons, setSiteSeasons] = useState<SiteSeasons>(() => parseSiteSeasons(init.get('sse')) ?? readSavedSiteSeasons())
  // saved in the link's own form, so one parser reads both
  useEffect(() => { try { localStorage.setItem(SITE_SEASONS_KEY, siteSeasonsParam(siteSeasons) ?? '{}') } catch { /* private window */ } }, [siteSeasons])
  const [siteRules, setSiteRules] = useState<SiteRules>(() => parseSiteRules(init.get('dyn')) ?? readSavedSiteRules())
  useEffect(() => { try { localStorage.setItem(SITE_RULES_KEY, siteRulesParam(siteRules) ?? '{}') } catch { /* private window */ } }, [siteRules])
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
  // the reasons written beside hand exclusions; they travel in the link (and a shared view) with the ids
  const [dropWhy, setDropWhy] = useState<Record<string, string>>(() => parseWhy(init.get('dropwhy')))
  const dropWhyParam = useMemo(() => { const kept = Object.entries(dropWhy).filter(([id, r]) => dropIds.has(id) && r.trim()); return kept.length ? JSON.stringify(Object.fromEntries(kept)) : null }, [dropWhy, dropIds])
  const [keepIds, setKeepIds] = useState<Set<string> | null>(() => idSet(init.get('keep')))
  // which spectral baseline FTIR EC is shown with (spline and VIBES where a filter has a native spectrum)
  const [ftirBaseline, setFtirBaseline] = useState<FtirBaseline>(() => (['spline', 'vibes'] as string[]).includes(init.get('fb') ?? '') ? init.get('fb') as FtirBaseline : 'none')
  const [seasonsOn, setSeasonsOn] = useState(init.get('seas') !== '0')
  const [pmfOn, setPmfOn] = useState(init.get('pmfon') === '1')
  // with the PMF filter on: also hide the sites (and days) without PMF, so only PMF filters remain
  const [pmfOnly, setPmfOnly] = useState(init.get('pmfonly') === '1')
  // hide filters whose black carbon is the reflectometer's (lib/ssrFilters): '' keeps them
  const [ssrHide, setSsrHide] = useState<SsrHide>(() => parseSsrHide(init.get('ssr')))
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
  // the reader's groupings (lib/groupings): a measurement split into labelled bands, for panels or colour
  // kept in the URL (`grp`, shareable) and in this browser (localStorage), merged on load
  const [groupings, setGroupings] = useState<GroupingDef[]>(() => mergeGroupings(parseGroupings(init.get('grp')), readSavedGroupings()))
  useEffect(() => writeSavedGroupings(groupings), [groupings])
  const [mac, setMac] = useState<number | null>(() => parseMac(init.get('mac')))
  const [basis, setBasis] = useState<Basis>(init.get('basis') === 'mass' ? 'mass' : 'conc')
  // meeting pages: as presented (default) or following the gallery (lib/presentation); ?mview=exploring
  const [presentation, setPresentation] = useSearchParam<Presentation>('mview', 'presented', ['presented', 'exploring'])
  // one chart at a time (lib/singleView); the Dashboard presents its own way
  const [singleOn, setSingleOn] = useSingleViewPref()
  const [estimators, setEstimators] = useState<Estimator[]>(() => parseEstimators(init.get('fits')))
  const [showCI, setShowCI] = useState(init.get('ci') === '1')
  const [pinnedId, setPinnedId] = useState<string | null>(init.get('pin'))
  // axes policy shared by every crossplot on the correlation tab
  const [axes, setAxes] = useState<AxesOpts>({
    identity: init.get('eq') !== '0',
    mode: (AXES_MODES as readonly string[]).includes(init.get('ax') ?? '') ? (init.get('ax') as AxesMode) : DEFAULT_AXES.mode,
    log: init.get('log') === '1',
    panels: paxFromParam(init.get('pax')),
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
      xp: tab === 'experimental' && experiment !== DEFAULT_EXPERIMENT ? experiment : null,
      aep: SECTION_TABS.has(tab) && sectionPage && sectionPage !== defaultPage(SECTION_TABS.get(tab)!.pages) ? sectionPage : null,
      fits: estimatorParam(estimators),
      ci: showCI ? '1' : null,
      cal: convention,
      cs: convention === 'custom' ? customSeasonsParam(customSeasons) : null,
      rr: convention === 'rain_rule' && rainRuleParam(rainRule) !== rainRuleParam(DEFAULT_RAIN_RULE) ? rainRuleParam(rainRule) : null,
      cv: Object.keys(calVariants).length ? Object.entries(calVariants).map(([c, k]) => `${c}:${k}`).join(',') : null,
      sse: siteSeasonsParam(siteSeasons),
      dyn: siteRulesParam(siteRules),
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
      ssr: ssrHide || null,
      from: dateRange?.[0],
      to: dateRange?.[1],
      excl: exclModeParam(exclMode),
      incl: !includeExcluded && inclIds.size ? [...inclIds].join(',') : null,
      drop: dropIds.size ? [...dropIds].join(',') : null,
      dropwhy: dropWhyParam,
      fb: ftirBaseline !== 'none' ? ftirBaseline : null,
      keep: keepIds ? [...keepIds].join(',') : null,
      x: xField !== DEFAULT_X ? xField : null,
      y: yField !== DEFAULT_Y ? yField : null,
      f: field !== DEFAULT_FIELD ? field : null,
      ...dashParams(dash),
      ratios: ratiosParam(ratios),
      grp: groupingsParam(groupings),
      mac: mac !== null ? String(mac) : null,
      basis: basis === 'mass' ? 'mass' : null,
      pin: pinnedId,
      eq: axes.identity ? null : '0',
      ax: axes.mode !== DEFAULT_AXES.mode ? axes.mode : null,
      log: axes.log ? '1' : null,
      pax: PAX_PARAM[axes.panels ?? 'Shared'],
    })
  }, [tab, meeting, experiment, sectionPage, estimators, showCI, convention, calVariants, siteSeasons, siteRules, activeSeasons, activeSites, network, spartanSel, improveSites, dropIds, dropWhyParam, keepIds, ftirBaseline, customSeasons, rainRule, pmfScheme, activePmf, seasonsOn, pmfOn, pmfOnly, ssrHide, dateRange, exclMode, inclIds, xField, yField, field, dash, ratios, groupings, mac, basis, pinnedId, axes])

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
        if (tabFromParams(p) === 'experimental') setExperiment(experimentFromParams(p))
        if (SECTION_TABS.has(tabFromParams(p))) setSectionPage(p.get('aep') ?? '')
      }
      setConvention(p.get('cal'))
      setCustomSeasons(parseCustomSeasons(p.get('cs')) ?? DEFAULT_CUSTOM)
      setRainRule(parseRainRule(p.get('rr')) ?? DEFAULT_RAIN_RULE)
      setCalVariants(parseVariants(p.get('cv')))
      setSiteSeasons(parseSiteSeasons(p.get('sse')) ?? readSavedSiteSeasons())
      setSiteRules(parseSiteRules(p.get('dyn')) ?? readSavedSiteRules())
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
      setSsrHide(parseSsrHide(p.get('ssr')))
      setDateRange(p.get('from') && p.get('to') ? [p.get('from')!, p.get('to')!] : null)
      setExclMode(parseExclMode(p.get('excl')))
      setInclIds(idSet(p.get('incl')) ?? new Set())
      setDropIds(idSet(p.get('drop')) ?? new Set())
      setDropWhy(parseWhy(p.get('dropwhy')))
      setFtirBaseline((['spline', 'vibes'] as string[]).includes(p.get('fb') ?? '') ? p.get('fb') as FtirBaseline : 'none')
      setKeepIds(idSet(p.get('keep')))
      setXField(p.get('x') ?? DEFAULT_X)
      setYField(p.get('y') ?? DEFAULT_Y)
      setField(p.get('f') ?? DEFAULT_FIELD)
      setDash(parseDash(p))
      setRatios(parseRatios(p.get('ratios')))
      setGroupings((cur) => mergeGroupings(parseGroupings(p.get('grp')), cur))
      setMac(parseMac(p.get('mac')))
      setBasis(p.get('basis') === 'mass' ? 'mass' : 'conc')
      setPinnedId(p.get('pin'))
      setAxes({
        identity: p.get('eq') !== '0',
        mode: (AXES_MODES as readonly string[]).includes(p.get('ax') ?? '') ? (p.get('ax') as AxesMode) : DEFAULT_AXES.mode,
        log: p.get('log') === '1',
        panels: paxFromParam(p.get('pax')),
      })
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  // The network only applies on the chart tabs; everything else always shows the focus sites.
  const net: Network = (NETWORK_TABS as readonly string[]).includes(tab) ? network : 'focus'
  const spartanSource = useSpartanSource()
  const netData = useNetworkData(net, improveSites, spartanSel ?? data?.meta.sites.map(s => s.code) ?? [], spartanSource.source, spartanSource.revision)
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
    const sites = d.sites.map((x) => { const f = focus.get(x.code); return f ? { ...x, name: f.name, color: f.color, timezone: f.timezone } : x })
    for (const f of focus.values()) if (!sites.some((x) => x.code === f.code)) sites.push(f)
    const spartanCodes = sites.filter((x) => !d.improveCodes.has(x.code)).map((x) => x.code)
    // the focus sites also get the public fields filters.json does not carry (public HIPS BC, the trace
    // elements, the rest of the ions), and missing measurements filled on the same physical ID.
    const pub = new Map(d.rows.filter((r) => focus.has(r.code)).map((r) => [r.id, r]))
    const loadedFocus = new Set(d.rows.filter(r => focus.has(r.code)).map(r => r.code))
    const focusIds = new Set(focusRows.map(r => r.id))
    const focusWithPub = focusRows.filter(r => loadedFocus.has(r.code))
      .map(r => {
        const extra = pub.get(r.id)
        return extra?.public_live_status && d.publicFields
          ? replacePublicFilter(supplementFilter(r, extra, d.fields), extra, d.publicFields)
          : supplementFilter(r, extra, d.fields)
      })
    const rows: FilterRow[] = [...(sp ? focusWithPub : []), ...d.rows.filter((r) => !focus.has(r.code) || !focusIds.has(r.id)).map(r => focus.has(r.code) ? { ...r, site: focus.get(r.code)!.name } : r)]
    const fields = sp ? [...new Set([...data.meta.fields, ...d.fields])] : d.fields
    const groups = sp ? mergeGroups(data.meta.field_groups, d.field_groups) : d.field_groups
    return { sites, focusCodes: [...focus.keys()], spartanCodes, improveCodes: d.improveCodes, rows, fields,
      field_units: sp ? { ...d.field_units, ...data.meta.field_units } : d.field_units, field_groups: groups }
  }, [net, data, focusRows, netData.data])
  // weather over each filter's sampling window (export_weather.py, lib/weather), for the sites
  // whose rows are in view: one small file per site, loaded as sites are picked
  const weatherCodes = useMemo(() => [...new Set((netBase ? netBase.rows : focusRows).map((r) => r.code))], [netBase, focusRows])
  const weather = useWeather(weatherCodes)
  // the sun photometer over the same windows (export_aeronet.py, lib/aeronet), loaded the same way
  const aeronet = useAeronet(weatherCodes)
  // and the compiled sections' own per-filter fields (the aethalometer); none in a build that omits them,
  // and an unlisted section's only on its own tab, so the other tabs' pickers don't show its group
  const instrumentFields = instruments.useFields(instruments.tabs.some((t) => UNLISTED.has(t.key)) && !instruments.tabs.some((t) => t.key === tab) ? [] : weatherCodes)
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
    const inChosen = rows.filter((r) => chosenNames.has(r.site) && exclIn(r))
    const coverage = Object.fromEntries(fields.map((f) => [f, inChosen.filter((r) => typeof r[f] === 'number').length]))
    return { ...netBase, all, chosen: all.filter((x) => chosenCodes.has(x.code)), chosenCodes, chosenNames, nSpartan: spartanCodes.length, paint, coverage }
  }, [netBase, net, data, improveSites, spartanSel, exclIn])
  // the ChemSpec restatements of the FTIR carbon are never offered; a link naming one falls back
  const activeFields = (netView ? netView.fields : data?.meta.fields ?? []).filter((f) => !HIDDEN_DUPLICATES.includes(f))
  // ratios whose two fields this view has: the presets, then the reader's own (lib/viewFields)
  const activeKey = activeFields.join('|')
  const ratioView = useMemo(() => {
    const has = new Set([...activeFields, ...(weather?.fields ?? []), ...(aeronet?.fields ?? []), ...(instrumentFields?.fields ?? []), ...pmfShareFields(data?.pmf ?? null)])
    const fits = (d: RatioDef) => has.has(d.num) && has.has(d.den)
    const presets = RATIO_PRESETS.filter(fits)
    const own = ratios.filter((d) => fits(d) && !presets.some((x) => ratioKey(x) === ratioKey(d)))
    return { presets, all: [...presets, ...own] }
  }, [activeKey, weather, aeronet, instrumentFields, ratios, data]) // eslint-disable-line react-hooks/exhaustive-deps
  const ratioFields = useMemo(() => ratioView.all.map(ratioKey), [ratioView])

  // Fields that vanished from the export, or that the chosen network does not measure,
  // fall back to that network's defaults instead of leaving a chart empty.
  useEffect(() => {
    if (!data || (netView && !netView.fields.length)) return
    // weather and AERONET fields (lib/weather, lib/aeronet) load on their own, after this first check, so they are known by name
    const ok = (f: string) => activeFields.includes(f) || ratioFields.includes(f) || isPmfField(f) || /\((ERA5|station|AERONET|aethalometer)\)$/.test(f)
    const d = NETWORK_DEFAULTS[net]
    if (!ok(xField)) setXField(ok(d.x) ? d.x : activeFields[0])
    if (!ok(yField)) setYField(ok(d.y) ? d.y : activeFields[1] ?? activeFields[0])
    if (!ok(field)) setField(ok(d.field) ? d.field : activeFields[0])
  }, [data, netView, activeFields, ratioFields, net, xField, yField, field])

  // Some SPARTAN sites never had HIPS: Kanpur, Ilorin and Singapore have SPARTAN's public HIPS BC but nothing in
  // HIPS batches 1–51, and the older mesh-Teflon sites (Buenos Aires, Dalhousie, Kelowna, Lethbridge, Hanoi …)
  // have only the smoke-stain reflectometer's BC (lib/fieldEquivalents cites SPARTAN's methods). When a newly
  // chosen site would drop out of an axis for that reason, the axis moves to the first documented stand-in a
  // chosen site has, instead of leaving that site's panels empty. Only when the choice of sites changes, and
  // not over a field the link named, so HIPS BC can still be picked by hand.
  const seenSites = useRef<string | null>(null)
  const linkNamedField = useRef(!!(init.get('x') || init.get('y') || init.get('f')))
  // the axes this moved to a stand-in, and to which: when no chosen site needs it any more they go back to
  // HIPS BC, so a pick of every site and then the focus sites again does not leave a stand-in behind
  const fellBack = useRef(new Map<'x' | 'y' | 'f', string>())
  useEffect(() => {
    if (!netView || !hasSpartan(net) || !netView.rows.length) return
    const key = [...netView.chosenCodes].sort().join(',')
    if (key === seenSites.current) return
    seenSites.current = key
    if (linkNamedField.current) { linkNamedField.current = false; return }
    const num = (r: FilterRow, f: string) => typeof r[f] === 'number'
    const lacking = netView.chosen.map((x) => netView.rows.filter((r) => r.site === x.name)).filter((own) => own.length && !own.some((r) => num(r, 'HIPS BC')))
    const target = lacking.length ? EQUIVALENTS['HIPS BC'].map((e) => e.field)
      .find((f) => netView.fields.includes(f) && lacking.some((own) => own.some((r) => num(r, f)))) : undefined
    const back = (axis: 'x' | 'y' | 'f', cur: string, set: (f: string) => void) => { if (fellBack.current.get(axis) === cur) set('HIPS BC') }
    if (!target) {
      back('x', xField, setXField); back('y', yField, setYField); back('f', field, setField)
      fellBack.current.clear()
      return
    }
    const move = (axis: 'x' | 'y' | 'f', cur: string, set: (f: string) => void) => {
      if (cur === 'HIPS BC' || fellBack.current.get(axis) === cur) { set(target); fellBack.current.set(axis, target) }
    }
    move('x', xField, setXField); move('y', yField, setYField); move('f', field, setField)
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
  // pmf_paper: the PMF paper's four Addis seasons, split by date (lib/pmfPaper, lib/seasons seasonOfDate)
  const conventions = useMemo<Record<string, SeasonMeta[]>>(() => ({ ...(data?.meta.season_conventions ?? {}), pmf_paper: PAPER_CALENDAR, rain_rule: RAIN_PLACEHOLDER, quarters: MONTH_QUARTERS, custom: customSeasonMeta(customSeasons) }), [data, customSeasons])
  // local calendars are the default wherever every site in view has one; otherwise shared quarters
  const siteCode = useMemo(() => new Map((netView?.all ?? []).map((x) => [x.name, x.code])), [netView])
  const allLocal = net === 'focus' || (!!netView && netView.chosen.length > 0 && netView.chosen.every((x) => hasLocalCalendar(x.name, x.code)))
  const defaultConvention = allLocal ? 'local' : 'quarters'
  const activeConvention = convention ?? defaultConvention
  // the rain rule is per site too: every site's seasons come from its own rain, year by year
  const rainOn = activeConvention === 'rain_rule'
  const isLocal = activeConvention === 'local' || rainOn
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
  const codeOfSite = useCallback((site: string) => siteCode.get(site) ?? data?.meta.sites.find((x) => x.name === site)?.code, [siteCode, data])
  const rainCals = useRainCalendars(rainOn ? calendarSites.map(codeOfSite).filter((c): c is string => !!c) : [], rainRule)
  const dynCals = useDynamicCalendars(siteRules)
  const calendarOf = useCallback(
    // sites without a local calendar fall back to the shared quarters, not the Ethiopian bins
    (site: string) => {
      if (rainOn) return withNoWeather(rainCals.cals.get(codeOfSite(site) ?? '') ?? RAIN_PENDING)
      if (!isLocal) return sharedSeasons
      const code = codeOfSite(site)
      // a site's weather rule, then its own month bins, win over its published calendar
      const rule = code ? siteRules[code] : undefined
      if (rule) return withNoWeather(dynCals.cals.get(code!) ?? dynamicPlaceholder(rule))
      const own = code ? siteSeasons[code] : undefined
      if (own) return customSeasonMeta(own)
      return localSeasonsFor(site, data?.meta, code, code ? calVariants[code] : undefined) ?? MONTH_QUARTERS
    },
    [isLocal, data, sharedSeasons, calVariants, rainOn, rainCals, codeOfSite, siteSeasons, siteRules, dynCals]
  )
  const seasons = useMemo(
    () => isLocal
      ? calendarSites.flatMap((site) => calendarOf(site).map((se) => ({ ...se, site, name: qualifySeason(site, se.name, qualify) })))
      : sharedSeasons,
    [isLocal, calendarSites, calendarOf, qualify, sharedSeasons]
  )

  // month -> season for one site under the *currently selected* calendar, so
  // switching calendar re-labels every chart rather than needing a re-export
  // a calendar with day boundaries goes by the sampling date (lib/seasons seasonOfDate)
  const seasonOf = useCallback((site: string, month: number | null | undefined, fallback: string, date?: string | null) => {
    if (!month && !date) return fallback
    const cal = calendarOf(site)
    const se = seasonOfDate(date, month, cal)
    // a weather calendar that cannot place the filter says so, rather than keeping the exported season
    const name = se?.name ?? (isDatedCalendar(cal) ? NO_WEATHER : null)
    return name ? (isLocal ? qualifySeason(site, name, qualify) : name) : fallback
  }, [calendarOf, isLocal, qualify])
  const seasonForFilter = useCallback((r: FilterRow) => seasonOf(r.site, r.month, r.season, r.date), [seasonOf])

  const inRange = useCallback(
    (r: { date: string }) => !dateRange || (r.date >= dateRange[0] && r.date <= dateRange[1]),
    [dateRange]
  )

  // Rows after the exclusion toggle and calendar relabel — the brush draws these.
  const rawSourceRows = useMemo(() => {
    const base = netBase ? netBase.rows : focusRows
    // the network's rows get their spline / VIBES FTIR EC too (focus rows already carry theirs)
    const rows = !baselines ? base : base.map((r) => {
      const b = baselines.rows[r.id]
      return b && r[FTIR_VARIANT.spline] === undefined ? { ...r, [FTIR_VARIANT.spline]: b.spline ?? null, [FTIR_VARIANT.vibes]: b.vibes ?? null } : r
    })
    // the weather fields ride on every row, so every chart and picker can use them
    const withWeather = !weather ? rows : rows.map((r) => {
      const w = weather.rows[r.id]
      return w ? { ...r, ...Object.fromEntries(weather.fields.map((f, i) => [f, w[i]])) } : r
    })
    // and the AERONET column over the same window
    const withAeronet = !aeronet ? withWeather : withWeather.map((r) => {
      const a = aeronet.rows[r.id]
      return a ? { ...r, ...Object.fromEntries(aeronet.fields.map((f, i) => [f, a[i]])) } : r
    })
    // so do the PMF source shares and dominant source, at the site the PMF solution covers (lib/pmfGroups)
    const withSections = !instrumentFields ? withAeronet : withAeronet.map((r) => {
      const a = instrumentFields.rows[r.id]
      return a ? { ...r, ...Object.fromEntries(instrumentFields.fields.map((f, i) => [f, a[i]])) } : r
    })
    return withPmf(withSections, data?.pmf ?? null, pmfScheme)
  }, [netBase, focusRows, weather, aeronet, instrumentFields, data, pmfScheme, baselines])
  // the reader's view (lib/viewFields): MAC, mass per filter, ratios; the drawer keeps the exported values
  const exportMac = data?.meta.mac_value ?? 10
  const viewOpts = useMemo(() => ({
    mac: mac ?? exportMac, exportMac, basis, ratios: ratioView.all,
    units: { ...(netBase ? netBase.field_units : data?.meta.field_units ?? {}), ...(weather?.field_units ?? {}), ...(aeronet?.field_units ?? {}), ...(instrumentFields?.field_units ?? {}), [FTIR_VARIANT.spline]: 'ug/m3', [FTIR_VARIANT.vibes]: 'ug/m3' },
  }), [mac, exportMac, basis, ratioView, netBase, data, weather, aeronet, instrumentFields])
  const plainSourceRows = useMemo(() => viewRows(rawSourceRows, viewOpts), [rawSourceRows, viewOpts])
  // the site colours go on last, so a site pick repaints the rows without rebuilding them; the
  // lookups, counts and drawers (colours from meta.sites) take the rows unpainted
  const paint = netView?.paint
  const sourceRows = plainSourceRows
  // one or more of the export's exclusions in or out of view; all of them in is the same as excl=1
  const includeIds = useCallback((ids: string[], on: boolean) => {
    const all = (netView ? sourceRows : rawSourceRows).filter((r) => r.excluded).map((r) => r.id)
    const cur = new Set(includeExcluded ? all : inclIds)
    for (const id of ids) { if (on) cur.add(id); else cur.delete(id) }
    const allIn = all.length > 0 && all.every((id) => cur.has(id))
    setExclMode(allIn ? 'include' : includeExcluded ? exclOff.current : exclMode)
    setInclIds(allIn ? new Set() : cur)
  }, [netView, sourceRows, rawSourceRows, includeExcluded, exclMode, setExclMode, inclIds])
  const dashboardReferenceRows = useMemo(() => plainSourceRows.map(r => ({ ...r, season: seasonForFilter(r) })), [plainSourceRows, seasonForFilter])
  const paintedSourceRows = useMemo(() => (tab === 'meeting' || tab === 'calibration' || tab === 'dashboard' ? paintRows(sourceRows, paint) : sourceRows), [tab, sourceRows, paint])
  // a meeting page as presented: the reader's ratios, but MAC as exported and µg/m³
  const presentOpts = useMemo(() => ({ ...viewOpts, mac: exportMac, basis: 'conc' as const }), [viewOpts, exportMac])
  const presentRows = useMemo(() => (tab === 'meeting' && presentation === 'presented' ? paintRows(viewRows(rawSourceRows, presentOpts), paint) : paintedSourceRows),
    [tab, presentation, rawSourceRows, presentOpts, paintedSourceRows, paint])
  const sourceById = useMemo(() => new Map(sourceRows.map((r) => [r.id, r])), [sourceRows])
  const plainBaseRows = useMemo(() => {
    if (!data) return []
    return plainSourceRows
      .filter(exclIn)
      .map((r) => {
        const s = seasonForFilter(r)
        return s !== r.season ? { ...r, season: s } : r
      })
  }, [data, plainSourceRows, exclIn, seasonForFilter])
  const baseRows = useMemo(() => paintRows(plainBaseRows, paint), [plainBaseRows, paint])
  // how many filters in view carry each ratio, for the pickers' counts and the dialog
  const inViewRows = useMemo(() => (netView ? baseRows.filter((r) => netView.chosenNames.has(r.site)) : baseRows), [baseRows, netView])
  const ratioCoverage = useMemo(() => Object.fromEntries(ratioFields.map((f) => [f, inViewRows.filter((r) => typeof r[f] === 'number').length])), [ratioFields, inViewRows])
  const countRatio = useCallback((d: RatioDef) =>
    inViewRows.filter((r) => typeof r[d.num] === 'number' && typeof r[d.den] === 'number' && r[d.den] !== 0).length, [inViewRows])
  const addRatio = useCallback((d: RatioDef) => setRatios((cur) => (cur.some((x) => ratioKey(x) === ratioKey(d)) || RATIO_PRESETS.some((x) => ratioKey(x) === ratioKey(d)) ? cur : [...cur, d])), [])
  const removeRatio = useCallback((f: string) => setRatios((cur) => cur.filter((x) => ratioKey(x) !== f)), [])
  const saveGrouping = useCallback((d: GroupingDef) => setGroupings((cur) => cur.some((x) => x.id === d.id) ? cur.map((x) => (x.id === d.id ? d : x)) : [...cur, d]), [])
  const removeGrouping = useCallback((id: string) => setGroupings((cur) => cur.filter((x) => x.id !== id)), [])

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
  const inSsr = useMemo(() => keepUnderSsr(ssrHide), [ssrHide])
  // a site with PMF is among the picked ones: the chart tabs then offer the PMF section
  const pmfSiteInView = useMemo(() => !!pmfSite && baseRows.some((r) => r.code === pmfSite && (netView ? netView.chosenNames.has(r.site) : !activeSites || activeSites.has(r.site))),
    [pmfSite, baseRows, netView, activeSites])

  // the picked sites' SSR filters: the Samples panel offers to hide them only while there are any
  const ssrInView = useMemo(() => ssrCounts(baseRows.filter((r) => (netView ? netView.chosenNames.has(r.site) : !activeSites || activeSites.has(r.site)))),
    [baseRows, netView, activeSites])

  // what the date brush draws: every filter except the date cut, so its range follows the picked sites
  const brushRows = useMemo(
    () => baseRows
      .filter((r) => !seasonSel || seasonSel.has(r.season))
      .filter((r) => (netView ? netView.chosenNames.has(r.site) : !activeSites || activeSites.has(r.site)))
      .filter(inChartPmf)
      .filter(inSsr)
      .filter((r) => !dropIds.has(r.id) && (!keepIds || keepIds.has(r.id))),
    [baseRows, seasonSel, activeSites, netView, dropIds, keepIds, inChartPmf, inSsr]
  )
  const rows = useMemo(
    () =>
      baseRows
        .filter(inRange)
        .filter((r) => !seasonSel || seasonSel.has(r.season))
        .filter((r) => (netView ? netView.chosenNames.has(r.site) : !activeSites || activeSites.has(r.site)))
        .filter(inChartPmf)
        .filter(inSsr)
        .filter((r) => !dropIds.has(r.id) && (!keepIds || keepIds.has(r.id))),
    [baseRows, inRange, seasonSel, activeSites, netView, dropIds, keepIds, inChartPmf, inSsr]
  )

  // Filters → excluded: Mark (lib/exclusionMode, lib/markedRows): the excluded filters the view would otherwise show
  // (same dates, seasons, sites, PMF and keep), for the point charts to draw as × beside `rows`, never in them
  const exclState = useMemo(() => ({ mode: exclMode, inclIds, dropIds, dropWhy }), [exclMode, inclIds, dropIds, dropWhy])
  const markedRows = useMemo(() => {
    if (exclMode !== 'mark' || !data) return NO_ROWS
    const out = plainSourceRows.filter((r) => !exclIn(r) || dropIds.has(r.id)).map((r) => {
      const s = seasonForFilter(r)
      return s !== r.season ? { ...r, season: s } : r
    })
    return paintRows(out, paint)
      .filter(inRange)
      .filter((r) => !seasonSel || seasonSel.has(r.season))
      .filter((r) => (netView ? netView.chosenNames.has(r.site) : !activeSites || activeSites.has(r.site)))
      .filter(inChartPmf)
      .filter(inSsr)
      .filter((r) => !keepIds || keepIds.has(r.id))
  }, [exclMode, data, plainSourceRows, exclIn, dropIds, seasonForFilter, paint, inRange, seasonSel, netView, activeSites, inChartPmf, inSsr, keepIds])
  const knownFilter = useCallback((id: string) => sourceById.has(id) || sourceById.has(baseFilterId(id)), [sourceById])
  const markReason = useCallback((id: string) => {
    const r = sourceById.get(id) ?? sourceById.get(baseFilterId(id))
    return exclusionReason(r ?? { id }, exclState) ?? (r ? null : exclusionReason({ id: baseFilterId(id) }, exclState))
  }, [sourceById, exclState])

  // every filter of the sites in view, before the date, season and exclusion filters: what a site measured at
  // all, so the dashboard can tell a field the site never measured from one this view's filters leave out
  const measuredRows = useMemo(() => (tab !== 'dashboard' ? [] : paintedSourceRows.filter((r) => (netView ? netView.chosenNames.has(r.site) : !activeSites || activeSites.has(r.site)))),
    [tab, paintedSourceRows, netView, activeSites])

  // the Weather tab's site table can narrow the Filters panel's pick to one site (and put it back)
  const setSpartanPick = useCallback((codes: string[]) => setSpartanSel(codes), [])

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
      .map((r) => ({ ...r, season: seasonOf(pmfSiteName, r.month, r.season, r.date) }))
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
  // so do the AERONET fields, as a group of their own
  if (aeronet) {
    const present = aeronet.fields.filter((f) => sourceRows.some((r) => typeof r[f] === 'number'))
    meta.fields = [...meta.fields, ...present.filter((f) => !meta.fields.includes(f))]
    meta.field_units = { ...meta.field_units, ...aeronet.field_units }
    meta.field_groups = [...(meta.field_groups ?? []).filter((g) => g.label !== AERONET_GROUP), ...(present.length ? [{ label: AERONET_GROUP, fields: present }] : [])]
  }
  // and each compiled section's fields
  for (const extra of [instrumentFields]) {
    if (!extra) continue
    const present = extra.fields.filter((f) => sourceRows.some((r) => typeof r[f] === 'number'))
    meta.fields = [...meta.fields, ...present.filter((f) => !meta.fields.includes(f))]
    meta.field_units = { ...meta.field_units, ...extra.field_units }
    meta.field_groups = [...(meta.field_groups ?? []).filter((g) => g.label !== extra.group), ...(present.length ? [{ label: extra.group, fields: present }] : [])]
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
  meta.coverage = { ...meta.coverage, ...ratioCoverage, ...Object.fromEntries(meta.fields.map(f => [f, measuredCount(rows, f)])) }
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
      <NavMenu groups={navGroups(tab, builderEnabled)} tab={tab} meeting={meeting} onPick={(i) => openTab(i.tab, i.meeting, i.sub)}
        sub={SECTION_TABS.has(tab) ? sectionPage || defaultPage(SECTION_TABS.get(tab)!.pages) : tab === 'availability' ? availView : experiment} />
    </nav>
  )

  // the Dashboard's header draws the common filters from the same state and actions (components/dashboard/DashHeader)
  let dashFilters: DashFilters | null = null
  const subset = (() => {
    if (tab === HUB_TAB || tab === 'availability' || tab === 'census' || tab === 'baseline' || tab === 'similarity' || tab === 'meeting' || tab === 'blanks' || tab === 'experimental' || tab === 'ml') return null
    const baseCaps = net === 'focus' ? CAPS[tab] : { ...CAPS[tab], datasetLabel: net === 'improve' ? 'IMPROVE samples' : net === 'both' ? 'SPARTAN filters + IMPROVE samples' : 'SPARTAN filters', excluded: sourceRows.some((r) => r.excluded) }
    // the chart tabs offer the PMF section while a site with PMF is picked
    const chartTab = (NETWORK_TABS as readonly string[]).includes(tab)
    const caps = chartTab && pmfSiteInView ? { ...baseCaps, pmf: true } : baseCaps
    const pmfMode = tab === 'pmf'
    const calMode = tab === 'calibration'
    // Counts are computed *before* the dimension's own filter is applied,
    // so a chip always shows what selecting it would give you.
    const pool: { season: string; site: string; pmf: string }[] = pmfMode
      ? (data.pmf?.rows ?? []).map((r) => ({ season: seasonOf(pmfSiteName, r.month, r.season, r.date), site: pmfSiteName, pmf: pmfLabel(data.pmf, r.dominant_source, pmfScheme) }))
      : calMode
        ? addisEvalDates.map((d) => ({ season: seasonOf(pmfSiteName, d ? Number(d.slice(5, 7)) : null, 'undated', d), site: pmfSiteName, pmf: pmfLabelOfDate(d) }))
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
      ? rawMeta.sites.map((x) => ({ key: x.name, label: x.name, aka: siteAka(x.name), sub: x.location, lat: x.lat, lon: x.lon, color: x.color, n: siteCounts[x.name] ?? 0, selected: !activeSites || activeSites.has(x.name) }))
      : (netView?.all ?? []).map((x) => {
        const on = netView!.chosenCodes.has(x.code)
        // a site in SPARTAN's index with no filters in the public release (Dhaka's CODC, Baltimore's USBA): listed,
        // greyed and last, rather than offered as if picking it would draw something
        const none = x.n_filters === 0 && !netView!.improveCodes.has(x.code)
        return { key: x.code, label: x.name, aka: siteAka(x.code, x.name), country: netView!.improveCodes.has(x.code) ? undefined : siteCountry(x.name, x.location), group: netView!.improveCodes.has(x.code) ? 'IMPROVE' : 'SPARTAN', sub: none ? `${x.code} · no public filters` : net === 'improve' ? x.location : x.code, lat: x.lat, lon: x.lon,
          color: on ? x.color : '#9aa5b1', n: net === 'improve' && !on ? x.n_filters : siteCounts[x.name] ?? 0, selected: on, disabled: none && !on }
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
      exclMode,
      // the export's exclusions currently put back in view, all of them or picked one by one
      includedIds: (netView ? sourceRows : rawSourceRows).filter((r) => r.excluded && exclIn(r)).map((r) => r.id),
      // counted from the same rows the list under it shows (the export's count misses rows the app marks itself)
      nExcluded: (netView ? sourceRows : rawSourceRows).filter((r) => r.excluded).length,
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
      ssr: ssrInView.any > 0 ? { mode: ssrHide, ...ssrInView } : undefined,
      network: net,
      chosenNetwork: network,
      dropped: dropIds.size,
      droppedList: [...dropIds].map((id) => {
        const r = sourceById.get(id)
        return { id, label: r ? `${r.site} · ${r.date}` : 'not in the current network', reason: dropWhy[id] ?? '' }
      }),
      // the repo's own exclusions (the export's `excluded` rows), with the reason it recorded
      excludedList: (netView ? sourceRows : rawSourceRows).filter((r) => r.excluded)
        .map((r) => ({ id: r.id, label: `${r.site} · ${r.date}`, reason: String(r.exclusion_reason ?? '') })),
      ftirBaseline,
      showFtirBaseline: [xField, yField, field].includes('EC (FTIR)'),
      kept: keepIds?.size ?? null,
      keptIds: keepIds ? [...keepIds] : [],
      networkLoading: netData.loading,
      networkError: netData.error,
      siteItems,
      siteLimitNote: net === 'improve' ? `${improveSites.length} of ${MAX_IMPROVE_SITES} IMPROVE sites`
        : net === 'both' && netView ? `${netView.chosen.length - improveSites.length} SPARTAN · ${improveSites.length}/${MAX_IMPROVE_SITES} IMPROVE` : undefined,
      // Addis's two Ethiopian calendars are its own switch and custom bins are each site's ✎, so the
      // picker leaves them out unless a link chose one
      calendarOptions: ['local', ...Object.keys(conventions)].filter((k) => !['dry_feb', 'belg_feb', 'custom', 'rain_rule'].includes(k) || k === activeConvention),
      customSeasons,
      rainRule,
      // sites in view whose sources disagree, with the calendar each is binned by
      calendarVariants: isLocal && !rainOn ? calendarSites.flatMap((site) => {
        const code = codeOfSite(site)
        const options = calendarVariants(site, code)
        return options && code && !siteSeasons[code] && !siteRules[code] ? [{ site, code, options, value: calVariants[code] ?? options[0].key }] : []
      }) : [],
      // every site's seasons, editable under the per-site calendar: the reader's, or its published ones to start from
      siteCalendars: isLocal && !rainOn ? calendarSites.flatMap((site) => {
        const code = codeOfSite(site)
        if (!code) return []
        const own = siteSeasons[code]
        const base = localSeasonsFor(site, data?.meta, code, calVariants[code]) ?? MONTH_QUARTERS
        // the references hold for the published calendar only, not the reader's own bins or rules
        const published = !own && !siteRules[code]
        const sources = published ? seasonSources(code, calVariants[code]) : []
        const weather = !siteRules[code] || dynCals.cals.has(code) ? undefined : dynCals.failed.has(code) ? 'unavailable' as const : 'loading' as const
        return [{ site, code, edited: !!own, rule: siteRules[code], weather, sources, note: published ? calendarNote(code) : undefined, seasons: own ?? base.map((se) => ({ name: bareSeasonName(se.name), months: se.months })) }]
      }) : [],
      sitesAtDefault: net === 'spartan' ? spartanSel === null : net === 'improve' ? improveSites.length > 0 : net === 'both' ? spartanSel === null && improveSites.length === 0 : !activeSites,
      allSitesSelected: hasSpartan(net) ? !!netView && netView.chosen.length - (net === 'both' ? improveSites.length : 0) === netView.nSpartan : !activeSites,
    }
    const actions: SubsetActions = {
      onBasis: setBasis,
      onPmfOnly: setPmfOnly,
      onSsr: setSsrHide,
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
      onToggleSeasons: (names) => {
        const cur = new Set(seasonSel ?? seasons.map((s) => s.name))
        const allOn = names.every((n) => cur.has(n))
        for (const n of names) if (allOn) cur.delete(n); else cur.add(n)
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
      onRainRule: (v) => { setRainRule(v); setActiveSeasons(null) },
      // the default calendar is left out of the link
      onSiteRule: (code, rule) => {
        setSiteRules((cur) => { const next = { ...cur }; if (rule) next[code] = rule; else delete next[code]; return next })
        setActiveSeasons(null)
      },
      onSiteSeasons: (code, bins) => {
        setSiteSeasons((cur) => { const next = { ...cur }; if (bins) next[code] = bins; else delete next[code]; return next })
        setActiveSeasons(null)
      },
      onCalendarVariant: (code, key, isDefault) => {
        setCalVariants((cur) => { const next = { ...cur }; if (isDefault) delete next[code]; else next[code] = key; return next })
        setActiveSeasons(null)
      },
      onDropIds: dropFilters,
      // keeping again narrows the kept set to filters in both brushes
      onKeepIds: keepFilters,
      onRestoreIds: restoreFilters,
      onClearDrop: () => setDropIds(new Set()),
      onDropReason: (id: string, reason: string) => setDropWhy((cur) => ({ ...cur, [id]: reason })),
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
      onIncludeExcluded: (v) => { setExclMode(v ? 'include' : exclOff.current); setInclIds(new Set()) },
      onExclMode: (m) => { setExclMode(m); if (m === 'include') setInclIds(new Set()) },
      onIncludeIds: includeIds,
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
        setSsrHide('')
        setDateRange(null)
        setExclMode('hide')
        setInclIds(new Set())
        setConvention(null)
      },
    }
    if (tab === 'dashboard') dashFilters = { state, actions, brushRows }
    return <SubsetBar meta={meta} caps={caps} brushRows={brushRows} state={state} actions={actions} />
  })()

  return (
    <SpartanDataProvider value={{ active: hasSpartan(net), codes: netView?.chosen.filter(s => !netView.improveCodes.has(s.code)).map(s => s.code) ?? [],
      loading: netData.loading, error: netData.error, sites: netData.data?.publicStatus ?? [] }}>
    <HeaderNavProvider tabs={navTabs} meetings={MEETINGS} open={openTab}>
    <RegressionProvider methods={estimators} setMethods={setEstimators} showCI={showCI} setShowCI={setShowCI}>
    <PrefsProvider>
    <SingleViewProvider on={singleOn} setOn={setSingleOn} available={tab !== 'dashboard' && tab !== 'builder'} page={`${tab}/${meeting}/${experiment}`}>
    {/* Panel axes for every multi-panel crossplot (lib/panelAxes): the sidebar's setting, offered on each chart too */}
    <PanelAxesProvider value={axes.panels ?? 'Shared'} onChange={(panels) => setAxes((a) => ({ ...a, panels }))}>
    <ExportSitesProvider seasons={meta.seasons} sites={[...new Map([...rawMeta.sites, ...meta.sites].map(s => [s.code, s])).values()]}>
    <HighlightProvider pinnedId={pinnedId} setPinned={setPinnedId} openSample={openSample} openRecord={openRecord} onDismiss={dismissSample}>
    {/* hand edits hold on every tab: each chart that draws a filter leaves the excluded ones out */}
    <PageSidebarProvider>
    {/* "Define a ratio…" in every field picker (components/RatioDialog) */}
    <RatiosProvider meta={meta} presets={ratioView.presets} custom={ratios} onAdd={addRatio} onRemove={removeRatio} count={countRatio}>
    {/* "New grouping…" in the charts that split or colour by a grouping (components/GroupingDialog) */}
    <GroupingsProvider meta={meta} rows={rows} list={groupings} onSave={saveGrouping} onRemove={removeGrouping}>
    {/* the MAC every chart reads (lib/mac useMac): the reader's, or the export's on a meeting page as presented */}
    <MacProvider mac={tab === 'meeting' && presentation === 'presented' ? exportMac : viewOpts.mac} exportMac={exportMac} lambda10={data?.calibration?.deming_lambda_mac10 ?? LAMBDA_10}>
    <FilterEditsProvider enabled dropped={dropIds} kept={keepIds}
      onDrop={dropFilters} onKeep={keepFilters} onRestore={restoreFilters} group={group} onGroup={setGroup}>
    {/* Filters → excluded: Mark — the excluded filters in view, drawn as × by the point charts, counted by none */}
    <MarkedRowsProvider mode={exclMode} rows={markedRows} reasonOf={markReason} known={knownFilter}>
      {openRow && (
        <SampleDrawer
          row={openRow}
          seasonOf={seasonOf}
          allRows={rawSourceRows.some((r) => r.id === openRow.id) ? rawSourceRows : filters.rows}
          meta={drawerMeta}
          studyRows={filters.rows}
          xField={xField}
          yField={yField}
          focusField={field}
          pinned={pinnedId === openRow.id}
          onPin={(id) => id === null ? dismissSample() : setPinnedId(id)}
          onOpen={openSample}
          onClose={dismissSample}
          context={openRec}
          // whether the charts draw this filter, and the one action that flips it
          exclusion={(() => {
            const id = openRow.id
            if (openRow.excluded) return exclIn(openRow)
              ? { state: 'export-in' as const, toggle: () => includeIds([id], false) }
              : { state: 'export-out' as const, toggle: () => includeIds([id], true) }
            return dropIds.has(id)
              ? { state: 'hand' as const, toggle: () => restoreFilters(new Set([id])) }
              : { state: 'in' as const, toggle: () => dropFilters(new Set([id])) }
          })()}
        />
      )}
      {!openRow && openRec && <RecordDrawer rec={openRec} onClose={dismissSample} />}
      {!openRow && !openRec && shownGroup && <GroupDrawer group={shownGroup} byId={sourceById} meta={meta}
        fields={tab === 'correlation' ? [xField, yField] : [field]} onClose={() => setGroup(null)} />}
      {isEmbed() && <a className="embed-open" href="#" onClick={async (e) => { e.preventDefault(); window.open((await viewUrl(currentView())), '_blank', 'noopener') }}>Open in the gallery ↗</a>}
      {(NETWORK_TABS as readonly string[]).includes(tab) && <BrushBar onKeep={keepFilters} onDrop={dropFilters} />}
      <Shell nav={nav} dash={tab === 'dashboard'} wide={tab === 'builder' || tab === 'baseline' || tab === 'calibration' || tab === 'similarity' || tab === 'meeting' || tab === 'blanks' || tab === 'experimental'} editsBadge={!subset}>
        {/* tabs without the Filters panel: a sidebar for the page's own controls, with the fits where the page fits */}
        {subset ?? ((tab === 'baseline' || tab === 'blanks' || tab === 'meeting' || tab === 'experimental' || tab === 'similarity')
          && <PageSidebar fits={tab === 'baseline' || meeting === 'followup' && tab === 'meeting'} label={tab === 'similarity' ? 'Target' : 'Samples'} />)}
        {netData.error && (NETWORK_TABS as readonly string[]).includes(tab) && <div className="network-data-error" role="alert"><strong>The selected network data could not be loaded.</strong> {netData.error}
          <button className="btn" type="button" onClick={() => setNetwork('focus')}>Open bundled focus sites</button></div>}
        {tab === HUB_TAB && <InvestigationsPage areas={HUB_AREAS} onOpen={(t, p) => openTab(t, undefined, p ?? AREAS.find((a) => a.tab === t)?.pages[0]?.key)} />}
        {tab === 'availability' && <DataAvailabilityPage />}
        {tab === 'builder' && <ChartBuilderPage rows={rows} meta={meta} enabled={builderEnabled} />}
        {tab === 'census' && <><AreaHeader area="Census" pages={CENSUS_PAGES} value="census" onPage={() => {}} /><CensusPage census={census} /></>}
        {tab === 'similarity' && <SimilarityPage />}
        {tab === 'meeting' && <PresentationProvider mode={presentation} set={setPresentation}>
          <AreaHeader area="Meetings" pages={MEETINGS} value={meeting} onPage={setMeeting} />
          <SidebarControls title="Meeting figures">
            <PresentationSwitch ignored={dropIds.size + (keepIds ? 1 : 0)} viewChanged={viewOpts.mac !== exportMac || basis !== 'conc'} />
          </SidebarControls>
          {meeting === 'followup' ? <MeetingFollowupPage galleryRows={filters.rows} galleryMeta={data.meta} viewRows={presentRows} viewMeta={presentMeta} />
            : meeting === 'satoshi' ? <SatoshiFollowupPage />
            : <MeetingSpecPage key={meeting} spec={SPEC_MEETINGS.find((m) => m.id === meeting) ?? SPEC_MEETINGS[0]} />}
        </PresentationProvider>}
        {tab === 'blanks' && <BlanksPage />}
        {tab === 'experimental' && <ExperimentalPage experiment={experiment} onPick={setExperiment} />}
        {tab === 'baseline' && (calibrationLoading ? <p>Loading calibration data…</p> : <BaselineComparisonPage mac={viewOpts.mac} demingLambda={demingLambda(viewOpts.mac, data.calibration?.deming_lambda_mac10 ?? LAMBDA_10)} />)}
        {tab === 'calibration' && (calibrationLoading ? <p>Loading calibration data…</p> : <CalibrationPage calib={data.calibration} runs={data.calibrationRuns} meta={drawerMeta} galleryRows={paintedSourceRows} sourceRows={rawSourceRows} galleryMeta={meta} knownIds={knownIds} seasonSel={seasonSel} pmfSel={pmfKeep} pmfLabelOfDate={pmfLabelOfDate} />)}
        {tab === 'dashboard' && <DashboardPage dash={dash} setDash={setDash} filters={dashFilters} onScope={setDashScope} onPickSite={pickDashSite} display={<DisplayToggles />}
          data={{ rows, measured: measuredRows, referenceRows: dashboardReferenceRows, meta, axes,
            provenance: { generated: data.filters.generated, source: data.filters.source, mac: viewOpts.mac, basis, baseline: ftirBaseline, calendar: convention, estimators, sourceMode: spartanSource.source, network: net, publicSources: netData.data?.publicStatus ?? [] },
            calibrationLoading, calibrationError, retryCalibration, requireCalibration: () => setDashboardCalibration(true), pmf: pmf ?? null, pmfRows, eff, defaults: { x: xField, y: yField, field }, openSite: openDashSite, pmfLabelOfDate,
            calibration: data.calibrationRuns ? { runs: data.calibrationRuns, lambda10: data.calibration?.deming_lambda_mac10 ?? LAMBDA_10, knownIds, galleryRows: paintedSourceRows, sourceRows: rawSourceRows, galleryMeta: meta, meta: drawerMeta } : null }} />}
        <AxisFieldsProvider meta={meta} value={axisValues} set={setAxisField}>
        {tab === 'correlation' && (
          <>
            <Scatterplot rows={rows} meta={meta} xField={xEff} yField={yEff} axes={axes} />
            <DensityHexbin rows={rows} meta={meta} xField={xEff} yField={yEff} axes={axes} />
            <ConnectedScatter rows={rows} meta={meta} xField={xEff} yField={yEff} axes={axes} />
            <SplitFit rows={rows} meta={meta} xField={xEff} yField={yEff} />
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
            <NetworkRankPlacement rows={rows} meta={meta} field={fieldEff} />
            <Barplot rows={rows} meta={meta} field={fieldEff} />
            <SiteRadar rows={rows} meta={meta} />
            <ParallelCoordinates rows={rows} meta={meta} />
          </>
        )}
        {tab === 'partOfWhole' && <><CompositionTreemap rows={rows} meta={meta} /><FunctionalGroups rows={rows} meta={meta} /></>}
        {tab === 'weather' && <WeatherPage rows={rows}
          meta={meta} field={fieldEff} xField={xEff} yField={yEff} axes={axes} setField={setField} pmfLabelOfDate={pmfLabelOfDate} />}
        {SECTION_TABS.has(tab) && (() => {
          const T = SECTION_TABS.get(tab)!
          return <T.Page rows={rows} meta={meta} field={fieldEff} xField={xEff} yField={yEff} axes={axes} setField={setField}
            picked={netView?.chosenCodes ?? new Set()} onPick={hasSpartan(net) ? setSpartanPick : undefined}
            page={T.pages.some((p) => p.key === sectionPage) ? sectionPage : defaultPage(T.pages)} onPage={setSectionPage} />
        })()}
        {tab === 'aeronet' && <AeronetPage rows={rows} picked={netView?.chosenCodes ?? new Set()} onPick={hasSpartan(net) ? setSpartanPick : undefined}
          meta={meta} field={fieldEff} axes={axes} setField={setField} />}
        </AxisFieldsProvider>
        {tab === 'map' && <SiteBubbleMap rows={rows} meta={meta} field={fieldEff} />}
        {tab === 'map' && <SpartanCompositionMap />}
        {tab === 'flow' && <MethodSankey rows={rows} meta={meta} />}
        {tab === 'pmf' &&
          (pmf ? (
            // the stack (its View 'by season' is the former Source apportionment by season), then coverage and dominance
            <><SourceStack pmf={pmf} rows={pmfRows} meta={meta} /><PmfDiagnostics pmf={pmf} rows={pmfRows} meta={meta} /></>
          ) : (
            <p style={{ fontSize: 13, color: 'var(--ink-muted)' }}>
              No PMF solution exported. Run <code>python gallery/data/export_data.py</code> with the ETAD factor CSVs present.
            </p>
          ))}
      </Shell>
    </MarkedRowsProvider>
    </FilterEditsProvider>
    </MacProvider>
    </GroupingsProvider>
    </RatiosProvider>
    </PageSidebarProvider>
    </HighlightProvider>
    </ExportSitesProvider>
    </PanelAxesProvider>
    </SingleViewProvider>
    </PrefsProvider>
    </RegressionProvider>
    </HeaderNavProvider>
    </SpartanDataProvider>
  )
}

/** `wide` gives the long research tabs room for the table-of-contents column. */
/** `editsBadge`: show the hand edits in the header, for tabs without the Filters panel that lists them */
/** `dash`: the Dashboard, full width, its options in the header (pages/DashboardPage portals them into #dash-bar) */
function Shell({ children, nav, wide = false, editsBadge = false, dash = false }: { children: React.ReactNode; nav?: React.ReactNode; wide?: boolean; editsBadge?: boolean; dash?: boolean }) {
  const container = dash ? 'container dash' : wide ? 'container wide' : 'container'
  // the sticky header's height, for what sticks under it (a chart's options column, styles.css --header-h)
  const headerH = useHeaderHeight()
  useEffect(() => { document.documentElement.style.setProperty('--header-h', `${headerH}px`) }, [headerH])
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
      <div className={container} data-single-root><SingleFit>{children}</SingleFit></div>
    </>
  )
}

/** Explain toggle (absent in public builds, see lib/prefs) and the Views menu (presets, saved views, share and embed). */
function HeaderTools() {
  return (
    <span className="header-tools">
      <span className="header-display"><DisplayToggles /></span>
      <CommandPaletteButton />
      <UndoToast />
      {/* on a phone these two fold into the ⋯ menu, with the display toggles */}
      <span className="header-wide"><GlossaryButton /><ViewsMenu /></span>
      <HeaderMore display={<DisplayToggles />} />
    </span>
  )
}

/** Readable text, Explain, the sections list: in the header, or in its ⋯ menu on a phone (components/HeaderMore) */
function DisplayToggles() {
  return (
    <>
      <DisplayControls />
      <SingleViewToggle icon={<SingleChartIcon />} />
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

/**
 * The sidebar of a tab without the Filters panel: the page's own controls (lib/pageSidebar), and the fits where it draws them.
 * `label` names it for what it holds (the Similarity tab's is its Target); the page's tab note (SidebarControls `tabNote`)
 * sits under the name on the edge tab.
 */
function PageSidebar({ fits, label = 'Samples' }: { fits: boolean; label?: string }) {
  const [open, setOpen] = useSidePanelOpen()
  const top = useHeaderHeight()
  const note = usePageSidebarNote()
  const noun = label.toLowerCase()
  return <>
    <SideDock open={open} top={top} onClose={() => setOpen(false)} tab={<button className="subset-tab" type="button" aria-expanded={open} onClick={() => setOpen(!open)}
      title={`${open ? 'Hide' : 'Show'} the ${noun}${note ? ` (${note})` : ''}`}>
      <span className="subset-tab-label">{label}</span>
      {note && <span className="subset-tab-count">{note}</span>}
    </button>}>
      <aside className="panel subset subset-panel" aria-label={label}>
        <header className="subset-head"><strong>{label}</strong><span style={{ marginLeft: 'auto' }}><button type="button" className="btn quiet x-btn" onClick={() => setOpen(false)} aria-label={`Close ${noun}`} title="Close"><XIcon /></button></span></header>
        <PageSidebarSlot />
        {fits && <RegressionControls />}
      </aside>
    </SideDock>
  </>
}

/** While a brush is active, a bar at the bottom offers to keep only the brushed filters or exclude them (both go in the link). */
function BrushBar({ onKeep, onDrop }: { onKeep: (ids: Set<string>) => void; onDrop: (ids: Set<string>) => void }) {
  const hl = useHighlightActions()
  if (!hl.selected) return null
  const ids = hl.selected
  return (
    <div className="brush-bar" role="region" aria-label="Brushed filters">
      <strong>{ids.size} brushed</strong>
      <button type="button" className="btn" onClick={() => { onKeep(ids); hl.setSelected(null) }} title="Filter every chart to these filters">Keep only these</button>
      <button type="button" className="btn" onClick={() => { onDrop(ids); hl.setSelected(null) }} title="Remove these filters from every chart; the Filters panel can put them back">Exclude these</button>
      <button type="button" className="btn quiet x-btn" onClick={() => hl.setSelected(null)} aria-label="Clear the brush"><XIcon /></button>
    </div>
  )
}
