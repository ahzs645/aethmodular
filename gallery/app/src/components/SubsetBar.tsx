import { PageSidebarSlot } from '@/lib/pageSidebar'
import { RegressionControls } from '@/lib/regressionPrefs'
import { chemText } from '@/lib/format'
import { CONVENTION_LABEL, seasonOnly } from '@/lib/seasons'
import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useDimensions } from '@/hooks/useGalleryData'
import { useHeaderHeight, useSidePanelOpen, useSidePanelView, type SidePanelView } from '@/hooks/useSidePanel'
import { DateBrush, type DateRange } from '@/components/DateBrush'
import { FieldSelect, prettyUnit } from '@/components/FieldSelect'
import { Segmented, SwapButton, Toggle } from '@/components/ChartFrame'
import { AXES_MODES, PANEL_AXES, type AxesOpts } from '@/lib/axes'
import { fmt } from '@/lib/stats'
import { usePrefs } from '@/lib/prefs'
import { useHighlightActions } from '@/lib/highlight'
import { useFilterEdits } from '@/lib/filterEdits'
import type { FilterRow, MetaFile, SeasonMeta } from '@/lib/types'
import type { SeasonSource } from '@/siteSeasons'
import { PMF_SCHEMES, type PmfGroup, type PmfScheme } from '@/lib/pmfGroups'
import { NETWORKS, hasImprove, hasSpartan, networkOf, type Network } from '@/lib/networks'
import { SitePicker, type SiteItem } from '@/components/SitePicker'
import { Combobox } from '@/components/Combobox'
import { useRatios } from '@/components/RatioDialog'
import { MAC_CHOICES, type Basis } from '@/lib/viewFields'
import { SeasonEditor } from '@/components/SeasonEditor'
import { DynamicSeasonEditor } from '@/components/DynamicSeasonEditor'
import { defaultRuleFor, type DynRule } from '@/lib/dynamicSeasons'
import { DEFAULT_RAIN_RULE, type RainRule } from '@/lib/rainSeasons'
import { XIcon } from '@/components/Icons'
import { CiteRef } from '@/components/CiteRef'

export interface SubsetCaps {
  /** the tab's charts respond to the season filter */
  seasons: boolean
  /** the tab has more than one site to choose between */
  sites: boolean
  /** the tab's data carries exclusion flags */
  excluded: boolean
  /** the tab's data has dates the brush can cut */
  dates: boolean
  /** which axis fields the tab's charts share: an x/y pair, one measurement, or none */
  fields: 'pair' | 'single' | 'none'
  /** the tab draws regression fits, so the estimator choices belong in the panel */
  fits?: boolean
  /** e.g. "filters" or "PMF filters" — what the count is counting */
  datasetLabel: string
  /** shown when a control is absent because the data can't support it */
  note?: string
  /** the tab's filters can be grouped by their ETAD PMF source */
  pmf?: boolean
  /** the tab's charts can show a whole network (SPARTAN, IMPROVE) instead of the focus sites */
  networks?: boolean
}

export interface SubsetState {
  convention: string
  defaultConvention: string
  seasons: SeasonMeta[]
  activeSeasons: Set<string>
  seasonCounts: Record<string, number>
  sites: string[]
  activeSites: Set<string>
  siteCounts: Record<string, number>
  includeExcluded: boolean
  nExcluded: number
  /** the export's exclusions currently put back in view (all of them under includeExcluded, else the ones picked) */
  includedIds?: string[]
  dateRange: DateRange | null
  xField: string
  yField: string
  field: string
  axes: AxesOpts
  pinned: FilterRow | null
  pinnedInSubset: boolean
  shown: number
  total: number
  pmfScheme?: PmfScheme
  pmfGroups?: PmfGroup[]
  activePmf?: Set<string>
  pmfCounts?: Record<string, number>
  seasonsOn?: boolean
  pmfOn?: boolean
  /** the network the charts draw (always 'focus' on tabs without network support) */
  network: Network
  /** the network the reader picked, kept while a focus-only tab is open */
  chosenNetwork: Network
  networkLoading?: boolean
  networkError?: string
  /** sites for the map and list picker */
  siteItems: SiteItem[]
  /** e.g. "3 of 10" for IMPROVE, where sites load one by one */
  siteLimitNote?: string
  /** calendars the picker offers */
  calendarOptions: string[]
  /** the reader's custom month bins, edited when the calendar is 'custom' */
  customSeasons?: { name: string; months: number[] }[]
  /** the rain rule's settings, edited when the calendar is 'rain_rule' (lib/rainSeasons) */
  rainRule?: RainRule
  /** sites in view with more than one defensible calendar, and the one each is binned by */
  calendarVariants?: { site: string; code: string; options: { key: string; label: string; note?: string }[]; value: string }[]
  /** under the per-site calendar, each site's seasons: the reader's own (edited) or its published ones */
  siteCalendars?: { site: string; code: string; edited: boolean; rule?: DynRule; sources?: SeasonSource[]; seasons: { name: string; months: number[] }[] }[]
  /** filters dropped by hand from a brush, and the size of a "keep only" set (null: none) */
  dropped?: number
  /** the hand-excluded filters, each with a "site · date" label */
  droppedList?: { id: string; label: string; reason?: string }[]
  /** the repo's own exclusions (rows the export marked `excluded`), with their recorded reason */
  excludedList?: { id: string; label: string; reason: string }[]
  /** spectral baseline for FTIR EC, offered when an axis or the measurement is EC (FTIR) */
  ftirBaseline?: 'none' | 'spline' | 'vibes'
  showFtirBaseline?: boolean
  kept?: number | null
  /** the "keep only" set's ids, for its review panel */
  keptIds?: string[]
  /** the site pick is the network's default (SPARTAN: the four focus sites) */
  sitesAtDefault?: boolean
  allSitesSelected?: boolean
  /** the units view (lib/viewFields): concentration or mass per filter, and the MAC for HIPS BC */
  basis?: Basis
  mac?: number
  exportMac?: number
  /** the PMF section's scope on a tab that mixes sites (only the PMF site's filters are narrowed) */
  pmfNote?: string
  /** hide the sites without PMF too, so only PMF filters remain */
  pmfOnly?: boolean
  /** filters in view with a sampled volume, of all in view (mass per filter needs one) */
  withVolume?: [number, number]
}

export interface SubsetActions {
  onConvention: (v: string) => void
  onToggleSeason: (name: string) => void
  /** one site's seasons at once: drop them all if every one is on, else switch them all on */
  onToggleSeasons?: (names: string[]) => void
  onAllSeasons: () => void
  onPmfScheme?: (v: PmfScheme) => void
  onTogglePmf?: (name: string) => void
  onAllPmf?: () => void
  onSeasonsOn?: (v: boolean) => void
  onPmfOn?: (v: boolean) => void
  onToggleSite: (key: string) => void
  onAllSites: () => void
  onClearSites?: () => void
  onFocusSites?: () => void
  onDropIds?: (ids: Set<string>) => void
  onFtirBaseline?: (v: 'none' | 'spline' | 'vibes') => void
  onCustomSeasons?: (v: { name: string; months: number[] }[]) => void
  /** a site's own seasons (by site code); null puts back its published calendar */
  onSiteSeasons?: (code: string, seasons: { name: string; months: number[] }[] | null) => void
  /** a site's dynamic seasons from its own weather (lib/dynamicSeasons); null drops the rule */
  onSiteRule?: (code: string, rule: DynRule | null) => void
  onRainRule?: (v: RainRule) => void
  onCalendarVariant?: (code: string, key: string, isDefault: boolean) => void
  onKeepIds?: (ids: Set<string>) => void
  onClearDrop?: () => void
  /** the reader's reason for a hand exclusion (kept in the link as `dropwhy`) */
  onDropReason?: (id: string, reason: string) => void
  /** put some hand-excluded filters back */
  onRestoreIds?: (ids: Set<string>) => void
  onClearKeep?: () => void
  onNetwork: (n: Network) => void
  onIncludeExcluded: (v: boolean) => void
  /** put some of the export's exclusions back (on) or leave them out again */
  onIncludeIds?: (ids: string[], on: boolean) => void
  onDateRange: (r: DateRange | null) => void
  onXField: (v: string) => void
  onYField: (v: string) => void
  onField: (v: string) => void
  onAxes: (v: AxesOpts) => void
  onUnpin: () => void
  onOpenPinned: () => void
  onReset: () => void
  onBasis?: (b: Basis) => void
  onPmfOnly?: (v: boolean) => void
  onMac?: (mac: number | null) => void
}

/** The calendar picker's wording lives with the calendars (lib/seasons). */
const CALENDAR_LABEL = CONVENTION_LABEL

/**
 * Global subset controls, as a floating side panel. Two tabs on the left edge
 * open it over the page without pushing the charts, so they update live beside
 * it: Filters (which filters are in view: network, sites, seasons, PMF, dates,
 * exclusions; it shows how many are in view and whether anything is filtered)
 * and Plot (how they are drawn: fits, axes, units, ratios, FTIR baseline).
 * Only the controls the active tab can use are shown: on the PMF tab the site
 * chips, the excluded toggle and the axis pickers are absent and a note says
 * why, and where nothing is plotted against a field there is no Plot tab.
 *
 * Everything here is mirrored into the URL hash (lib/url.ts), so the panel is
 * also the shareable state of the app.
 */
export function SubsetBar({
  meta,
  caps,
  brushRows,
  state: s,
  actions: a,
}: {
  meta: MetaFile
  caps: SubsetCaps
  /** rows the date brush draws — before the date filter, after the exclusion toggle */
  brushRows: FilterRow[]
  state: SubsetState
  actions: SubsetActions
}) {
  const [open, setOpen] = useSidePanelOpen()
  const [chosenView, setView] = useSidePanelView()
  // sit just below the sticky app header, whatever height its tab row wraps to
  const top = useHeaderHeight()
  const { explain } = usePrefs()
  // the Plot tab only where the open tab draws fits or picks fields
  const hasPlot = !!caps.fits || caps.fields !== 'none'
  const view: SidePanelView = hasPlot ? chosenView : 'filters'
  // a tab opens the panel on its half; the showing half's tab closes it
  const toggle = (v: SidePanelView) => { if (open && view === v) setOpen(false); else { setView(v); setOpen(true) } }
  const hl = useHighlightActions()
  const edits = useFilterEdits()
  // the review panel shows the fields the open tab is drawing
  const reviewFields = caps.fields === 'pair' ? [s.xField, s.yField] : caps.fields === 'single' ? [s.field] : []
  // both kinds of exclusion in one place: the sidebar's "excluded" row, and the same lists in a dialog
  const [exclusionsOpen, setExclusionsOpen] = useState(false)
  // the site whose seasons are open in the editor (its ✎)
  const [editingSite, setEditingSite] = useState<string | null>(null)
  const reviewHand = () => edits.openGroup({ title: 'Excluded by hand', ids: (s.droppedList ?? []).map((d) => d.id), fields: reviewFields, live: 'dropped' })

  const allSeasons = s.activeSeasons.size === s.seasons.length
  const allSites = s.sitesAtDefault ?? s.activeSites.size === s.sites.length
  const seasonsOn = s.seasonsOn ?? true
  const pmfOn = !!caps.pmf && !!s.pmfOn
  const allPmf = !pmfOn || !s.pmfGroups || (s.activePmf?.size ?? 0) === s.pmfGroups.length
  const handEdits = (s.dropped ?? 0) > 0 || s.kept != null
  const isDefault =
    allSeasons && allSites && !pmfOn && seasonsOn && !s.includedIds?.length && !s.dateRange && s.convention === s.defaultConvention && !handEdits
  const nActive = (caps.seasons && seasonsOn && !allSeasons ? 1 : 0) + (caps.sites && !allSites ? 1 : 0) + (pmfOn ? 1 : 0) + (s.dateRange ? 1 : 0) + (s.includedIds?.length && caps.excluded ? 1 : 0) + (handEdits ? 1 : 0)

  // seasons grouped by the site whose calendar they belong to (one group under the shared bins)
  const seasonGroups: { site: string | null; seasons: SeasonMeta[] }[] = []
  for (const se of s.seasons) {
    const g = seasonGroups.find((x) => x.site === (se.site ?? null))
    if (g) g.seasons.push(se)
    else seasonGroups.push({ site: se.site ?? null, seasons: [se] })
  }
  // one reference list for the sites in view: a paper shared by several sites keeps one number
  const citeNo = new Map<string, number>()
  for (const g of seasonGroups) for (const src of s.siteCalendars?.find((x) => x.site === g.site)?.sources ?? []) {
    if (!citeNo.has(src.url)) citeNo.set(src.url, citeNo.size + 1)
  }
  // the chip sits under its site's heading, so drop the site prefix from the label
  const shortSeason = (se: SeasonMeta) => (se.site && se.name.startsWith(`${se.site} · `) ? se.name.slice(se.site.length + 3) : se.name)

  const summary = (
    <div className="subset-summary">
      {caps.seasons && seasonsOn && !allSeasons && <span>seasons: {[...s.activeSeasons].join(', ')}</span>}
      {pmfOn && <span>PMF: {allPmf ? 'every source (PMF days only)' : [...(s.activePmf ?? [])].join(', ')}</span>}
      {s.dateRange && <span>{s.dateRange[0]} → {s.dateRange[1]}</span>}
      {!!s.includedIds?.length && caps.excluded && <span>+{s.includedIds.length} excluded put back</span>}
      {hl.selected && (
        <span className="brush-actions">
          <span className="pill" title="Filters picked by a brush; every chart dims the rest. A highlight until you keep or exclude them.">
            {hl.selected.size} brushed
            <button type="button" className="btn quiet x-btn" style={{ marginLeft: 4 }} onClick={() => hl.setSelected(null)} aria-label="Clear the brush"><XIcon /></button>
          </span>
          {a.onKeepIds && <button type="button" className="btn quiet" title="Filter every chart to the brushed filters"
            onClick={() => { a.onKeepIds!(hl.selected!); hl.setSelected(null) }}>Keep only these</button>}
          {a.onDropIds && <button type="button" className="btn quiet" title="Remove the brushed filters from every chart (kept in the link; Clear puts them back)"
            onClick={() => { a.onDropIds!(hl.selected!); hl.setSelected(null) }}>Exclude these</button>}
        </span>
      )}
      {(s.dropped ?? 0) > 0 && <span className="pill">
        <button type="button" className="pill-link" onClick={() => setExclusionsOpen(true)}
          title="Excluded from the charts' right-click menu, the side panel or a brush: open the list (reasons, put back)">{s.dropped} excluded by hand</button>
        <button type="button" className="btn quiet x-btn" style={{ marginLeft: 4 }} onClick={a.onClearDrop} aria-label="Put every hand-excluded filter back" title="Put them all back"><XIcon /></button>
      </span>}
      {s.kept != null && <span className="pill">keeping {s.kept}
        <button type="button" className="btn quiet" style={{ padding: '0 4px', marginLeft: 4 }} title="List the kept filters in the side panel"
          onClick={() => edits.openGroup({ title: 'Kept by hand (every other filter is hidden)', ids: s.keptIds ?? [], fields: reviewFields, live: 'kept' })}>review</button>
        <button type="button" className="btn quiet x-btn" style={{ marginLeft: 2 }} onClick={a.onClearKeep} aria-label="Stop keeping only the brushed filters"><XIcon /></button></span>}
    </div>
  )

  // the Plot tab: how the filters in view are drawn
  const plotOptions = (
    <>
      {/* only where fits are drawn; each chart offers the fit grouping that fits its layout */}
      {caps.fits && <RegressionControls grouping={false} />}

      {caps.fields === 'pair' && (
        <>
          <Row label="axes">
            <FieldSelect label="x" value={s.xField} meta={meta} onChange={a.onXField} />
            <SwapButton onClick={() => a.onXField(s.yField)} />
            <FieldSelect label="y" value={s.yField} meta={meta} onChange={a.onYField} />
            {explain && <span className="row-note">shared by every chart on this tab</span>}
          </Row>
          <Row label="crossplot axes">
            <Toggle
              label="1:1 line"
              checked={s.axes.identity}
              onChange={(v) => a.onAxes({ ...s.axes, identity: v })}
              title="Squares the panels and keeps equal units per pixel on both axes"
            />
            <Segmented
              value={s.axes.mode}
              options={AXES_MODES}
              onChange={(v) => a.onAxes({ ...s.axes, mode: v })}
              title="Data: 0 always visible, negatives kept · From 0,0: both axes start at zero (default) · Show intercept: keep all points and mark fitted intercepts with compact margins"
            />
            <Toggle label="Log axes" checked={s.axes.log} onChange={(v) => a.onAxes({ ...s.axes, log: v })} />
            <Segmented label="Panel axes" value={(s.axes.panels ?? 'Shared') as any} options={PANEL_AXES as any}
              onChange={(v: string) => a.onAxes({ ...s.axes, panels: v as AxesOpts['panels'] })}
              title="Several panels: Shared gives every panel the same range, so slopes and levels compare by eye · Per site (per-season layout): the seasons of one site share a range · Own zooms each panel to its own data. With the 1:1 line on, each panel stays square with equal x and y spans." />
            {explain && <span className="row-note">applies to the scatterplot, hexbin and connected scatter together</span>}
          </Row>
        </>
      )}
      {caps.fields === 'single' && (
        <Row label="measurement">
          <FieldSelect label="" value={s.field} meta={meta} onChange={a.onField} />
          {explain && (
            <span className="row-note">
              {meta.coverage?.[s.field] ?? '—'} of {meta.sites.reduce((n, x) => n + x.n_filters, 0)} filters carry {s.field}
              {prettyUnit(meta.field_units?.[s.field]) ? ` · ${prettyUnit(meta.field_units?.[s.field])}` : ''}
            </span>
          )}
        </Row>
      )}

      {caps.fields !== 'none' && a.onBasis && a.onMac && (
        <Row label="units">
          <Segmented value={s.basis === 'mass' ? 'Mass per filter' : 'Concentration'} options={['Concentration', 'Mass per filter'] as any}
            onChange={(v: string) => a.onBasis!(v === 'Mass per filter' ? 'mass' : 'conc')}
            title="Concentration: µg/m³ in the air. Mass per filter: µg on the filter (concentration × sampled volume), what FTIR sees. IMPROVE samples about 33 m³ a day, SPARTAN about 7." />
          <Combobox label="MAC" value={String(s.mac ?? s.exportMac ?? 10)} minWidth={70}
            title="Mass absorption cross-section (m²/g) turning HIPS absorption into BC: HIPS BC = Fabs / MAC. It multiplies the absorption axis, so it scales a fit's slope; the intercept stays put for OLS and moves only slightly for Deming (fixed error ratio)."
            options={MAC_CHOICES.map((m) => ({ key: String(m), label: m === s.exportMac ? `${m} (export)` : String(m) }))}
            onChange={(v) => a.onMac!(Number(v) === s.exportMac ? null : Number(v))} />
          {s.basis === 'mass' && s.withVolume && <span className="method-note" style={{ display: 'block' }}>
            {s.withVolume[0].toLocaleString()} of {s.withVolume[1].toLocaleString()} filters in view have a sampled volume; the rest drop out
          </span>}
          {explain && <span className="row-note">MAC scales the slope of a fit against HIPS BC; the intercept barely moves (exactly unchanged for OLS)</span>}
        </Row>
      )}
      {caps.fields !== 'none' && <RatioRow explain={explain} />}

      {caps.fields !== 'none' && s.showFtirBaseline && a.onFtirBaseline && (
        <Row label="FTIR baseline">
          <Segmented value={({ none: 'None', spline: 'Spline', vibes: 'VIBES' } as const)[s.ftirBaseline ?? 'none'] as any} options={['None', 'Spline', 'VIBES'] as any}
            onChange={(v: string) => a.onFtirBaseline!(v === 'Spline' ? 'spline' : v === 'VIBES' ? 'vibes' : 'none')}
            title="FTIR EC with no baseline correction (the SPARTAN product), the spline (AIRSpec) baseline, or the VIBES baseline. Spline and VIBES come from the same all-IMPROVE calibration pool: the Addis test set from the frozen calibration runs, every other SPARTAN PM2.5 filter with a native spectrum and a sampled volume from those models applied to the network baselines. Filters without one (and IMPROVE) drop out of EC (FTIR) charts while they are chosen." />
          {s.ftirBaseline && s.ftirBaseline !== 'none' && <span className="method-note" style={{ display: 'block' }}>SPARTAN filters with a native spectrum (28 sites, about 3,250 filters); Addis test set from the frozen runs</span>}
        </Row>
      )}
    </>
  )

  const net = NETWORKS.find((x) => x.key === s.network)!
  const sitesNote = s.network === 'improve' ? 'IMPROVE EC/OC: TOR (thermal-optical reflectance) · absorption from HIPS'
    : 'SPARTAN EC/OC: FTIR-derived · no TOR reference on these filters'
  return (
    <SideDock open={open} top={top} onClose={() => setOpen(false)} tab={
      <div className="side-dock-tabs" role="tablist" aria-label="Side panel" aria-orientation="vertical">
        <button
          type="button"
          role="tab"
          className={`subset-tab${open && view === 'filters' ? ' active' : ''}`}
          onClick={() => toggle('filters')}
          aria-selected={view === 'filters'}
          aria-expanded={open && view === 'filters'}
          title={open && view === 'filters' ? 'Hide the filters' : 'Show the filters: network, sites, seasons, dates, exclusions'}
        >
          <span className="subset-tab-label">Filters</span>
          <span className="subset-tab-count">{s.shown}/{s.total}</span>
          {nActive > 0 && <span className="subset-tab-badge" title={`${nActive} filter${nActive === 1 ? '' : 's'} active`}>{nActive}</span>}
        </button>
        {hasPlot && (
          <button
            type="button"
            role="tab"
            className={`subset-tab${open && view === 'plot' ? ' active' : ''}`}
            onClick={() => toggle('plot')}
            aria-selected={view === 'plot'}
            aria-expanded={open && view === 'plot'}
            title={open && view === 'plot' ? 'Hide the plot options' : 'Show the plot options: fits, axes, units, ratios, FTIR baseline'}
          >
            <span className="subset-tab-label">Plot</span>
          </button>
        )}
      </div>
    }>
        <aside className="panel subset subset-panel" aria-label={view === 'plot' ? 'Plot options' : 'Filters'}>
          <header className="subset-head">
            <strong>{view === 'plot' ? 'Plot options' : 'Filters'}</strong>
            {view === 'filters' && <span className="pill" title={`${s.shown} of ${s.total} ${caps.datasetLabel} in view`}>{s.shown} of {s.total}</span>}
            <span style={{ marginLeft: 'auto', display: 'inline-flex', gap: 4 }}>
              {view === 'filters' && !isDefault && <button type="button" className="btn quiet" onClick={a.onReset} title="Clear every filter">Reset</button>}
              <button type="button" className="btn quiet x-btn" onClick={() => setOpen(false)} aria-label={view === 'plot' ? 'Close plot options' : 'Close filters'} title="Close"><XIcon /></button>
            </span>
          </header>
          {view === 'plot' && plotOptions}
          {view === 'filters' && <>
            {summary}
            <Row label="network">
              {caps.networks ? (
                <span className="net-toggles" role="group" aria-label="Networks (pick one or both)"
                  title={s.network === 'both' ? 'SPARTAN and IMPROVE together: HIPS absorption and PM₂.₅ mass are measured by both; EC/OC by different methods (FTIR in SPARTAN, TOR in IMPROVE)' : net.note}>
                  {s.network === 'focus' && <button type="button" className="net-toggle" aria-pressed="true" disabled>Focus sites</button>}
                  {(['spartan', 'improve'] as const).map((k) => {
                    const on = k === 'spartan' ? hasSpartan(s.network) : hasImprove(s.network)
                    const other = k === 'spartan' ? hasImprove(s.network) : hasSpartan(s.network)
                    return <button key={k} type="button" className="net-toggle" aria-pressed={on}
                      title={`${NETWORKS.find((x) => x.key === k)!.note}${on && other ? ' · click to show only the other network' : !on ? ' · click to add' : ''}`}
                      onClick={() => a.onNetwork(s.network === 'focus' ? k : on ? (other ? networkOf(k !== 'spartan', k !== 'improve') : s.network) : 'both')}>
                      {k === 'spartan' ? 'SPARTAN' : 'IMPROVE'}
                    </button>
                  })}
                </span>
              ) : (
                <span className="method-note">{s.chosenNetwork === 'focus' ? net.label : `This tab shows the focus sites; ${NETWORKS.find((x) => x.key === s.chosenNetwork)?.label} applies on Correlation, Distribution, Evolution, Ranking, Part of a whole and Weather.`}</span>
              )}
              {caps.networks && (s.networkLoading || s.networkError) && <span className="method-note" style={{ display: 'block' }}>{s.networkLoading ? 'loading…' : `could not load: ${s.networkError}`}</span>}
            </Row>
            {/* the sites belong to the network picked above, so they sit right under it */}
            {caps.sites && (
              <Row label="sites">
                <SitePicker orderSession={open} items={s.siteItems} onToggle={a.onToggleSite}
                  onAll={s.network === 'improve' ? undefined : a.onAllSites} allSelected={!!s.allSitesSelected}
                  onClear={s.network === 'focus' ? undefined : a.onClearSites}
                  chipsOnly={s.network === 'focus'} limitNote={s.siteLimitNote}
                  noun={s.network === 'improve' ? 'IMPROVE sites' : s.network === 'spartan' ? 'SPARTAN sites' : 'sites'}
                  allLabel={s.network === 'both' ? 'All SPARTAN' : 'Select all'} />
              </Row>
            )}
            {caps.seasons && (
              <section className={`filter-section ${seasonsOn ? '' : 'off'}`}>
                <SectionSwitch label="Seasonality" on={seasonsOn} onChange={(v) => a.onSeasonsOn?.(v)}
                  offNote="off · every season" title="Switch off to ignore seasons entirely" />
                {seasonsOn && (
                  <>
                    <Row label="calendar">
                      <Combobox label="" title="Season calendar" value={s.convention} onChange={a.onConvention}
                        options={s.calendarOptions.map((k) => ({ key: k, label: `${CALENDAR_LABEL[k] ?? k}${k === s.defaultConvention ? ' (default)' : ''}` }))} />
                      {!['local', 'rain_rule', 'pmf_paper', 'quarters'].includes(s.convention) && <span className="row-note" style={{ display: 'block' }}>
                        {s.convention === 'custom'
                          ? 'Your own seasons, the same at every site. Pick a season, click months to move them into it, rename freely; they travel in the link.'
                          : <>Ethiopia’s seasons applied to every site by month, so sites compare month for month; outside Ethiopia the names only mark the months (“Belg” at Beijing is just those months). February sits in <strong style={{ color: 'var(--ink-text)' }}>{s.seasons.find((x) => x.months.includes(2))?.name ?? '—'}</strong>.</>}
                      </span>}
                    </Row>
                    {s.convention === 'custom' && s.customSeasons && a.onCustomSeasons && <SeasonEditor seasons={s.customSeasons} onChange={a.onCustomSeasons} />}
                    {s.convention === 'rain_rule' && s.rainRule && a.onRainRule && <RainRuleEditor rule={s.rainRule} onChange={a.onRainRule} />}
                    <div className="season-chips">
                      <Chip active={allSeasons} color="var(--ink-muted)" onClick={a.onAllSeasons} label="All" />
                    </div>
                    {seasonGroups.map((g) => (
                      <div key={g.site ?? 'shared'} className="season-group">
                        {(() => {
                          const cal = s.siteCalendars?.find((x) => x.site === g.site)
                          const editable = !!cal && !!a.onSiteSeasons
                          // the site heading helps when several calendars are listed, and carries the ✎
                          if (!g.site || (seasonGroups.length < 2 && !editable)) return null
                          const editing = editable && editingSite === `${cal!.code}:months`
                          const ruling = editable && !!a.onSiteRule && editingSite === `${cal!.code}:rule`
                          const rule = cal?.rule
                          return <>
                            <span className="season-site">
                              <span className="swatch" style={{ background: meta.sites.find((x) => x.name === g.site)?.color ?? 'var(--ink-muted)' }} />
                              {a.onToggleSeasons ? (() => {
                                const names = g.seasons.filter((se) => (s.seasonCounts[se.name] ?? 0) > 0).map((se) => se.name)
                                const on = names.some((n) => s.activeSeasons.has(n))
                                const flip = () => names.length && a.onToggleSeasons!(on ? names.filter((n) => s.activeSeasons.has(n)) : names)
                                return <span className={`season-site-name${on ? '' : ' off'}`} role="button" tabIndex={0} aria-pressed={on}
                                  title={on ? `Hide every ${g.site} season` : `Show every ${g.site} season`}
                                  onClick={flip} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); flip() } }}>{g.site}</span>
                              })() : g.site}
                              {cal?.sources?.length ? <span className="cite-refs">
                                {cal.sources.map((src) => <CiteRef key={src.url} n={citeNo.get(src.url)!} source={src} />)}
                              </span> : null}
                              {rule ? <span className="season-edited" title="Seasons from this site's own daily weather: saved in this browser and in the link">dynamic</span>
                                : cal?.edited && <span className="season-edited" title="Your own seasons for this site: saved in this browser and in the link">edited</span>}
                              {editable && <span className="season-site-tools">
                                {(cal!.edited || rule) && <button type="button" className="btn quiet"
                                  onClick={() => { a.onSiteSeasons!(cal!.code, null); a.onSiteRule?.(cal!.code, null); setEditingSite(null) }}
                                  title={`Put back ${g.site}'s published seasons`}>Reset</button>}
                                {a.onSiteRule && (rule || !editing) && <button type="button" className={`btn quiet season-edit-btn${ruling ? ' on' : ''}`} aria-pressed={ruling}
                                  title={ruling ? 'Done' : rule ? `Edit ${g.site}'s weather rules` : `Set ${g.site}'s seasons from its own daily weather: rain, temperature, humidity, wind`}
                                  onClick={() => {
                                    if (ruling) return setEditingSite(null)
                                    if (!rule) a.onSiteRule!(cal!.code, defaultRuleFor(cal!.code))
                                    setEditingSite(`${cal!.code}:rule`)
                                  }}>{ruling ? 'Done' : rule ? 'Rules' : 'Dynamic'}</button>}
                                {!rule && !ruling && <button type="button" className={`btn quiet season-edit-btn${editing ? ' on' : ''}`} aria-pressed={editing}
                                  aria-label={editing ? `Done editing ${g.site}'s seasons` : `Edit ${g.site}'s seasons`}
                                  title={editing ? 'Done' : `Edit ${g.site}'s seasons: rename them and move months between them`}
                                  onClick={() => setEditingSite(editing ? null : `${cal!.code}:months`)}>{editing ? 'Done' : '✎'}</button>}
                              </span>}
                            </span>
                            {editing && !rule && <SeasonEditor seasons={cal!.seasons} onChange={(v) => a.onSiteSeasons!(cal!.code, v)} />}
                            {ruling && rule && <DynamicSeasonEditor code={cal!.code} rule={rule} onChange={(r) => a.onSiteRule!(cal!.code, r)} />}
                          </>
                        })()}
                        {(() => {
                          // where the sources disagree, the reader picks the calendar (docs/site-seasonality-network.md)
                          const v = s.calendarVariants?.find((x) => x.site === g.site)
                          if (!v || !a.onCalendarVariant) return null
                          const note = v.options.find((o) => o.key === v.value)?.note
                          return <div className="season-variant">
                            <Segmented value={v.value} options={v.options.map((o) => o.key)} optionLabel={(k) => v.options.find((o) => o.key === k)?.label ?? k}
                              title={note} onChange={(k) => a.onCalendarVariant!(v.code, k, k === v.options[0].key)} />
                          </div>
                        })()}
                        <div className="season-chips">
                          {g.seasons.map((se) => {
                            const n = s.seasonCounts[se.name] ?? 0
                            return (
                              <Chip
                                key={se.name}
                                active={s.activeSeasons.has(se.name)}
                                color={se.color}
                                disabled={n === 0}
                                onClick={() => n > 0 && a.onToggleSeason(se.name)}
                                label={`${shortSeason(se)} · ${n}`}
                              />
                            )
                          })}
                        </div>
                      </div>
                    ))}
                  </>
                )}
              </section>
            )}

            {caps.pmf && s.pmfGroups && s.pmfGroups.length > 0 && (
              <section className={`filter-section ${pmfOn ? '' : 'off'}`}>
                <SectionSwitch label="PMF source" on={pmfOn} onChange={(v) => a.onPmfOn?.(v)}
                  offNote="off · PMF not used"
                  title="Switch on to keep only filters with an ETAD PMF day (102 days of 2023), then narrow by source" />
                {pmfOn && (
                  <>
                    <Segmented value={s.pmfScheme ?? PMF_SCHEMES[0]} options={PMF_SCHEMES} onChange={(v) => a.onPmfScheme?.(v)} title="Dominant source: the largest of the five ETAD PMF factors that day · Marine vs combustion: the Sea Salt + Polluted Marine factors against wood, charcoal and fossil fuel" />
                    <div className="season-chips">
                      <Chip active={allPmf} color="var(--ink-muted)" onClick={() => a.onAllPmf?.()} label="All" />
                      {s.pmfGroups.map((g) => {
                        const n = s.pmfCounts?.[g.name] ?? 0
                        return (
                          <Chip
                            key={g.name}
                            active={s.activePmf?.has(g.name) ?? true}
                            color={g.color}
                            disabled={n === 0}
                            onClick={() => n > 0 && a.onTogglePmf?.(g.name)}
                            label={`${g.name} · ${n}`}
                          />
                        )
                      })}
                    </div>
                    {s.pmfNote && a.onPmfOnly && <Toggle label="Only filters with a PMF day (hide sites without PMF)" checked={!!s.pmfOnly} onChange={a.onPmfOnly} />}
                    <span className={s.pmfNote ? 'method-note' : 'row-note'} style={{ display: 'block' }}>
                      {s.pmfNote ?? 'Only filters with an ETAD PMF day (102 days of 2023, matched by exact date) are kept while this is on. Combines with seasonality.'}
                    </span>
                  </>
                )}
              </section>
            )}

            {caps.dates && (
              <Row label="date range">
                <MeasuredBrush rows={brushRows} meta={meta} range={s.dateRange} onChange={a.onDateRange} />
                <span className="brush-hint">
                  {s.dateRange ? (
                    <>
                      {s.dateRange[0]} → {s.dateRange[1]}{' '}
                      <button type="button" className="btn quiet" onClick={() => a.onDateRange(null)}>Clear</button>
                    </>
                  ) : explain ? (
                    'drag across the timeline to keep only a date range; snaps to whole months'
                  ) : (
                    'all dates · drag to cut a range'
                  )}
                </span>
              </Row>
            )}

            {(caps.excluded || (s.dropped ?? 0) > 0) && (
              <Row label="excluded">
                <div className="exclusions-wrap">
                  <button type="button" className="btn quiet exclusions-open" onClick={() => setExclusionsOpen(true)} title="Show both lists in a larger window">Open ⤢</button>
                  <ExclusionLists s={s} a={a} caps={caps} onReview={reviewHand} />
                </div>
              </Row>
            )}
            {exclusionsOpen && <ExclusionsDialog onClose={() => setExclusionsOpen(false)}>
              <ExclusionLists s={s} a={a} caps={caps} onReview={reviewHand} wide />
            </ExclusionsDialog>}

            <Row label="">
              {caps.note && explain && <span className="row-note">{caps.note}</span>}
              {caps.sites && <span className="method-note">{sitesNote}</span>}
            </Row>

            {/* a page's own filters (lib/pageSidebar SidebarControls), as on the tabs without this panel */}
            <PageSidebarSlot />

            {s.pinned && (
              <PinnedCard row={s.pinned} meta={meta} fields={[s.xField, s.yField, s.field]} inSubset={s.pinnedInSubset} onUnpin={a.onUnpin} onOpen={a.onOpenPinned} />
            )}
          </>}
        </aside>
    </SideDock>
  )
}

/**
 * The left side panel and its edge tab, as one sliding unit: the tab is attached to the
 * panel's right edge, so it travels with the panel instead of jumping on its own. The
 * panel stays mounted while closed (slid out of view and inert), so opening animates.
 * Shared by the Filters panel and the Fits panel.
 */
export function SideDock({ open, top, tab, onClose, children }: { open: boolean; top: number; tab: React.ReactNode; onClose?: () => void; children: React.ReactNode }) {
  const body = useRef<HTMLDivElement>(null)
  // the single-chart view moves the page over to sit beside an open dock (styles.css)
  useEffect(() => {
    document.documentElement.dataset.dock = open ? 'open' : 'closed'
    return () => { delete document.documentElement.dataset.dock }
  }, [open])
  // closed: out of the tab order and hidden from assistive tech, though still painted for the slide
  useEffect(() => { body.current?.toggleAttribute('inert', !open) }, [open])
  // open, the panel runs the full window height over the app header (like the sample drawer);
  // the tab keeps its place just below the header so it does not move up and down
  // on a phone the dock is a bottom sheet (styles.css) and the scrim behind it closes it
  return <>
    {open && onClose && <div className="sheet-scrim dock-scrim" onClick={onClose} aria-hidden />}
    <div className={`side-dock ${open ? 'open' : ''}`}>
      <div ref={body} className="side-dock-body" aria-hidden={!open}>{children}</div>
      <div className="side-dock-tab" style={{ top: top + 24 }}>{tab}</div>
    </div>
  </>
}

/**
 * The record behind a clicked point. Charts only ever show two or three
 * numbers per filter; this is where "what *is* that outlier?" gets answered
 * without opening the pickle.
 */
function PinnedCard({
  row,
  meta,
  fields,
  inSubset,
  onUnpin,
  onOpen,
}: {
  row: FilterRow
  meta: MetaFile
  fields: string[]
  inSubset: boolean
  onUnpin: () => void
  onOpen: () => void
}) {
  const shown = [...new Set([...fields, 'EC (FTIR)', 'HIPS BC', 'PM2.5 mass'])].filter((f) =>
    meta.fields.includes(f)
  )
  return (
    <div className="pinned">
      <span className="id">📌 {row.id}</span>
      <span>
        <span style={{ display: 'inline-block', width: 9, height: 9, borderRadius: 5, background: row.color as string, marginRight: 5 }} />
        {row.site} · {row.date} · {seasonOnly(String(row.season ?? ''))}
      </span>
      {shown.map((f) => (
        <span key={f} className="kv">
          <span>{chemText(f)}</span> {fmt(row[f] as number | null, 3)}
        </span>
      ))}
      {row.excluded && <span className="reason">excluded: {row.exclusion_reason}</span>}
      {!inSubset && <span className="row-note">(outside the current subset)</span>}
      <span style={{ marginLeft: 'auto', display: 'inline-flex', gap: 4 }}>
        <button type="button" className="btn" onClick={onOpen}>Details</button>
        <button type="button" className="btn quiet" onClick={onUnpin}>Unpin</button>
      </span>
    </div>
  )
}

/** The date brush measures its own box, because the panel mounts it only when opened. */
function MeasuredBrush(props: Omit<React.ComponentProps<typeof DateBrush>, 'width'>) {
  const wrap = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrap)
  return (
    <div ref={wrap} style={{ width: '100%' }}>
      <DateBrush {...props} width={width} />
    </div>
  )
}

/** A filter dimension's header with an on/off switch; off means the dimension is ignored entirely. */
function SectionSwitch({ label, on, onChange, offNote, title }: { label: string; on: boolean; onChange: (v: boolean) => void; offNote: string; title?: string }) {
  return (
    <div className="section-switch">
      <button type="button" role="switch" aria-checked={on} className={`switch ${on ? 'on' : ''}`} onClick={() => onChange(!on)} title={title}>
        <span className="knob" />
      </button>
      <button type="button" className="section-switch-label" onClick={() => onChange(!on)} title={title}>{label}</button>
      {!on && <span className="switch-note">{offNote}</span>}
    </div>
  )
}

/** the reader's own ratios (lib/viewFields), each removable; presets live in the pickers */
function RatioRow({ explain }: { explain: boolean }) {
  const r = useRatios()
  if (!r) return null
  return (
    <Row label="ratios">
      <span className="ratio-list">
        {r.custom.map((d) => {
          const f = `${d.num} / ${d.den}`
          return <span key={f} className="chip active">{f}<button type="button" className="btn quiet x-btn" aria-label={`Remove ${f}`} title="Remove this ratio" onClick={() => r.remove(f)}><XIcon /></button></span>
        })}
        <button type="button" className="btn quiet" onClick={() => r.open(() => {})}>+ Define a ratio</button>
      </span>
      {explain && <span className="row-note">ratios sit in every field picker under Ratios, with presets (EC / PM2.5 mass, OC / EC …)</span>}
    </Row>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="row">
      <span className="row-label">{label}</span>
      {children}
    </div>
  )
}

function Chip({
  active,
  color,
  onClick,
  label,
  disabled = false,
}: {
  active: boolean
  color: string
  onClick: () => void
  label: string
  disabled?: boolean
}) {
  const cls = ['chip', active ? 'active' : '', disabled ? 'disabled' : ''].filter(Boolean).join(' ')
  return (
    <button
      type="button"
      className={cls}
      onClick={onClick}
      disabled={disabled}
      title={disabled ? 'no data under the current selection' : undefined}
      style={{ ['--chip' as string]: color }}
    >
      {label}
    </button>
  )
}

/**
 * The rain rule's three settings: smoothing window and the two thresholds on the rolling daily
 * rain (lib/rainSeasons). Each site's seasons, year by year, follow from its own ERA5 rain.
 */
function RainRuleEditor({ rule, onChange }: { rule: RainRule; onChange: (r: RainRule) => void }) {
  const num = (k: keyof RainRule, label: string, step: number, title: string) => (
    <label className="rain-rule-field" title={title}>{label}
      <input type="number" className="select" step={step} min={k === 'window' ? 3 : 0} value={rule[k]}
        onChange={(e) => { const v = Number(e.target.value); if (Number.isFinite(v)) { const next = { ...rule, [k]: v }; if (next.heavy > next.onset && next.window >= 3) onChange(next) } }} />
    </label>
  )
  const isDefault = rule.window === DEFAULT_RAIN_RULE.window && rule.onset === DEFAULT_RAIN_RULE.onset && rule.heavy === DEFAULT_RAIN_RULE.heavy
  return (
    <div className="rain-rule">
      <div className="rain-rule-fields">
        {num('window', 'Window (days)', 1, 'Days in the centred rolling mean of daily rain')}
        {num('onset', 'Rains begin (mm/day)', 0.5, 'Mean rain above which the dry season ends')}
        {num('heavy', 'Heavy rains (mm/day)', 0.5, 'Mean rain above which it is the More Rainy season')}
        {!isDefault && <button type="button" className="btn quiet" onClick={() => onChange(DEFAULT_RAIN_RULE)}>Reset</button>}
      </div>
    </div>
  )
}

/**
 * Every exclusion in one place: the data export's own (with the recorded reason; the toggle puts them back in) and
 * the reader's by hand (from the right-click menu, a brush or the side panel; reasons and ids travel in the link).
 * The lists are always open: in the sidebar they scroll, `wide` is the dialog's roomier version.
 */
function ExclusionLists({ s, a, caps, onReview, wide = false }: {
  s: SubsetState; a: SubsetActions; caps: SubsetCaps; onReview: () => void; wide?: boolean
}) {
  const hl = useHighlightActions()
  // one list: the data export's exclusions (where this tab can show them), then the reader's own
  const repo = caps.excluded ? s.excludedList ?? [] : [], hand = s.droppedList ?? []
  const back = new Set(s.includedIds ?? [])
  const repoOut = repo.filter((d) => !back.has(d.id)).length
  const nOut = repoOut + hand.length
  return (
    <div className={`exclusions${wide ? ' wide' : ''}`}>
      <section>
        <div className="exclusions-head">
          <strong>Excluded{repo.length + hand.length ? ` (${nOut} of ${repo.length + hand.length})` : ''}</strong>
          {repo.length > 0 && (repoOut > 0
            ? <button type="button" className="btn quiet dropped-review" onClick={() => a.onIncludeExcluded(true)}
              title="Put every sample the data export excludes back in every chart">Include the export's</button>
            : <button type="button" className="btn quiet dropped-review" onClick={() => a.onIncludeExcluded(false)}
              title="Leave out every sample the data export excludes again (the default)">Exclude the export's</button>)}
          {hand.length > 0 && <>
            {a.onClearDrop && <button type="button" className="btn quiet dropped-review" onClick={a.onClearDrop} title="Put every hand-excluded filter back">Put back all by hand</button>}
            <button type="button" className="btn quiet dropped-review" onClick={onReview}
              title="List the hand-excluded filters in the side panel: site, date, values; put any or all back">Review ›</button>
          </>}
          {repo.length + hand.length > 0 && <CopyExclusions rows={[
            ...repo.filter((d) => !back.has(d.id)).map((d) => ({ ...d, source: 'repo' })),
            ...hand.map((d) => ({ id: d.id, label: d.label, reason: d.reason ?? '', source: 'by hand' })),
          ]} />}
        </div>
        <div className="exclusions-sub">
          {repo.length + hand.length === 0
            ? 'None. Right-click a point (Exclude this filter), or brush points and choose Exclude these.'
            : 'Untick a sample to put it back in every chart. Add your own by right-clicking a point (Exclude this filter) or brushing points (Exclude these).'}
        </div>
        {repo.length + hand.length > 0 && <div className="exclusions-list">
          {repo.map((d) => {
            const out = !back.has(d.id)
            return <div key={d.id} className={`excluded-row${out ? '' : ' back'}`}>
              <input type="checkbox" checked={out} disabled={!a.onIncludeIds} aria-label={`${out ? 'Put back' : 'Exclude'} ${d.id}`}
                title={out ? 'Excluded by the data export: untick to put it back' : 'Put back: tick to leave it out again'}
                onChange={(e) => a.onIncludeIds?.([d.id], !e.target.checked)} />
              <button type="button" className="pill-link" onClick={() => hl.openSample(d.id)} title={`${d.label} · open this filter's details`}>{d.id}</button>
              <span className="excluded-where">{d.label}</span>
              <span className="excluded-source" title="Excluded in the data export, with the reason recorded there">export</span>
              <span className="excluded-reason">{d.reason || 'no reason recorded'}</span>
            </div>
          })}
          {/* each hand exclusion with a reason the reader writes; the reasons travel in the link with the ids */}
          {hand.map((d) => <div key={d.id} className="excluded-row">
            <input type="checkbox" checked disabled={!a.onRestoreIds} aria-label={`Put ${d.id} back`} title="Excluded by hand: untick to put it back"
              onChange={() => a.onRestoreIds?.(new Set([d.id]))} />
            <button type="button" className="pill-link" onClick={() => hl.openSample(d.id)} title={`${d.label} · open this filter's details`}>{d.id}</button>
            <span className="excluded-where">{d.label}</span>
            <span className="excluded-source hand" title="Excluded by you, from a chart or this panel; kept in the link">by hand</span>
            {a.onDropReason
              ? <input className="select excluded-why" value={d.reason ?? ''} placeholder="Why excluded? (kept in the link)" aria-label={`Reason for excluding ${d.id}`}
                onChange={(e) => a.onDropReason!(d.id, e.target.value)} />
              : d.reason ? <span className="excluded-reason">{d.reason}</span> : null}
          </div>)}
        </div>}
      </section>
    </div>
  )
}

/** The exclusions in a dialog over the page (Esc or a click outside closes it). */
function ExclusionsDialog({ onClose, children }: { onClose: () => void; children: React.ReactNode }) {
  useEffect(() => {
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [onClose])
  return createPortal(
    <div className="export-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="export-dialog exclusions-dialog" role="dialog" aria-modal="true" aria-label="Excluded samples">
        <header className="exclusions-dialog-head">
          <strong>Excluded samples</strong>
          <button type="button" className="btn quiet x-btn" onClick={onClose} aria-label="Close"><XIcon /></button>
        </header>
        {children}
      </div>
    </div>, document.body)
}

/** Copy a list of exclusions (id, site · date, reason) as JSON, e.g. to add to the repo's exclusion list. */
function CopyExclusions({ rows }: { rows: { id: string; label: string; reason: string; source: string }[] }) {
  const [done, setDone] = useState(false)
  const copy = () => {
    const json = JSON.stringify(rows.map((r) => ({ id: r.id, where: r.label, reason: r.reason || null, excluded: r.source })), null, 2)
    navigator.clipboard?.writeText(json).then(() => { setDone(true); setTimeout(() => setDone(false), 1500) }).catch(() => {})
  }
  return <button type="button" className="btn quiet excluded-copy" onClick={copy} title="Copy these exclusions and their reasons as JSON">{done ? 'Copied ✓' : 'Copy as JSON'}</button>
}
