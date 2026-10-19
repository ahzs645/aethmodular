import { RegressionControls } from '@/lib/regressionPrefs'
import { chemText } from '@/lib/format'
import { CONVENTION_LABEL, seasonOnly } from '@/lib/seasons'
import { useEffect, useRef } from 'react'
import { useDimensions } from '@/hooks/useGalleryData'
import { useHeaderHeight, useSidePanelOpen } from '@/hooks/useSidePanel'
import { DateBrush, type DateRange } from '@/components/DateBrush'
import { FieldSelect, prettyUnit } from '@/components/FieldSelect'
import { Segmented, SwapButton, Toggle } from '@/components/ChartFrame'
import { AXES_MODES, PANEL_AXES, type AxesOpts } from '@/lib/axes'
import { fmt } from '@/lib/stats'
import { usePrefs } from '@/lib/prefs'
import { useHighlight } from '@/lib/highlight'
import { useFilterEdits } from '@/lib/filterEdits'
import type { FilterRow, MetaFile, SeasonMeta } from '@/lib/types'
import { PMF_SCHEMES, type PmfGroup, type PmfScheme } from '@/lib/pmfGroups'
import { NETWORKS, hasImprove, hasSpartan, networkOf, type Network } from '@/lib/networks'
import { SitePicker, type SiteItem } from '@/components/SitePicker'
import { Combobox } from '@/components/Combobox'
import { useRatios } from '@/components/RatioDialog'
import { MAC_CHOICES, type Basis } from '@/lib/viewFields'
import { SeasonEditor } from '@/components/SeasonEditor'
import { DEFAULT_RAIN_RULE, type RainRule } from '@/lib/rainSeasons'

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
  /** filters dropped by hand from a brush, and the size of a "keep only" set (null: none) */
  dropped?: number
  /** the hand-excluded filters, each with a "site · date" label */
  droppedList?: { id: string; label: string }[]
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
  onRainRule?: (v: RainRule) => void
  onCalendarVariant?: (code: string, key: string, isDefault: boolean) => void
  onKeepIds?: (ids: Set<string>) => void
  onClearDrop?: () => void
  /** put some hand-excluded filters back */
  onRestoreIds?: (ids: Set<string>) => void
  onClearKeep?: () => void
  onNetwork: (n: Network) => void
  onIncludeExcluded: (v: boolean) => void
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
 * Global subset controls, as a floating side panel. A tab on the left edge
 * shows how many filters are in view and whether anything is filtered; it
 * opens the panel over the page without pushing the charts, so they update
 * live beside it. Only the controls the active tab can use are shown: on the
 * PMF tab the site chips, the excluded toggle and the axis pickers are absent
 * and a note says why.
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
  // sit just below the sticky app header, whatever height its tab row wraps to
  const top = useHeaderHeight()
  const { explain } = usePrefs()
  const hl = useHighlight()
  const edits = useFilterEdits()
  // the review panel shows the fields the open tab is drawing
  const reviewFields = caps.fields === 'pair' ? [s.xField, s.yField] : caps.fields === 'single' ? [s.field] : []

  const allSeasons = s.activeSeasons.size === s.seasons.length
  const allSites = s.sitesAtDefault ?? s.activeSites.size === s.sites.length
  const seasonsOn = s.seasonsOn ?? true
  const pmfOn = !!caps.pmf && !!s.pmfOn
  const allPmf = !pmfOn || !s.pmfGroups || (s.activePmf?.size ?? 0) === s.pmfGroups.length
  const handEdits = (s.dropped ?? 0) > 0 || s.kept != null
  const isDefault =
    allSeasons && allSites && !pmfOn && seasonsOn && !s.includeExcluded && !s.dateRange && s.convention === s.defaultConvention && !handEdits
  const nActive = (caps.seasons && seasonsOn && !allSeasons ? 1 : 0) + (caps.sites && !allSites ? 1 : 0) + (pmfOn ? 1 : 0) + (s.dateRange ? 1 : 0) + (s.includeExcluded && caps.excluded ? 1 : 0) + (handEdits ? 1 : 0)

  // seasons grouped by the site whose calendar they belong to (one group under the shared bins)
  const seasonGroups: { site: string | null; seasons: SeasonMeta[] }[] = []
  for (const se of s.seasons) {
    const g = seasonGroups.find((x) => x.site === (se.site ?? null))
    if (g) g.seasons.push(se)
    else seasonGroups.push({ site: se.site ?? null, seasons: [se] })
  }
  // the chip sits under its site's heading, so drop the site prefix from the label
  const shortSeason = (se: SeasonMeta) => (se.site && se.name.startsWith(`${se.site} · `) ? se.name.slice(se.site.length + 3) : se.name)

  const summary = (
    <div className="subset-summary">
      {caps.seasons && seasonsOn && !allSeasons && <span>seasons: {[...s.activeSeasons].join(', ')}</span>}
      {pmfOn && <span>PMF: {allPmf ? 'every source (PMF days only)' : [...(s.activePmf ?? [])].join(', ')}</span>}
      {s.dateRange && <span>{s.dateRange[0]} → {s.dateRange[1]}</span>}
      {s.includeExcluded && caps.excluded && <span>+{s.nExcluded} excluded</span>}
      {hl.selected && (
        <span className="brush-actions">
          <span className="pill" title="Filters picked by a brush; every chart dims the rest. A highlight until you keep or exclude them.">
            {hl.selected.size} brushed
            <button type="button" className="btn quiet" style={{ padding: '0 4px', marginLeft: 4 }} onClick={() => hl.setSelected(null)} aria-label="Clear the brush">×</button>
          </span>
          {a.onKeepIds && <button type="button" className="btn quiet" title="Filter every chart to the brushed filters"
            onClick={() => { a.onKeepIds!(hl.selected!); hl.setSelected(null) }}>Keep only these</button>}
          {a.onDropIds && <button type="button" className="btn quiet" title="Remove the brushed filters from every chart (kept in the link; Clear puts them back)"
            onClick={() => { a.onDropIds!(hl.selected!); hl.setSelected(null) }}>Exclude these</button>}
        </span>
      )}
      {(s.dropped ?? 0) > 0 && <details className="dropped-details">
        <summary title="Excluded from the charts' right-click menu, the side panel or a brush; open to put single filters back">
          <span className="pill">{s.dropped} excluded by hand ▾
            <button type="button" className="btn quiet" style={{ padding: '0 4px', marginLeft: 4 }} onClick={(e) => { e.preventDefault(); a.onClearDrop?.() }} aria-label="Put every hand-excluded filter back" title="Put them all back">×</button></span>
        </summary>
        <div className="dropped-list">
          <button type="button" className="btn quiet dropped-review" title="List every excluded filter in the side panel: site, date, values; put any or all back"
            onClick={() => edits.openGroup({ title: 'Excluded by hand', ids: (s.droppedList ?? []).map((d) => d.id), fields: reviewFields, live: 'dropped' })}>Review all {s.dropped} ›</button>
          {(s.droppedList ?? []).map((d) => <span key={d.id} className="pill" title={d.label}>
            <button type="button" className="pill-link" onClick={() => hl.openSample(d.id)} title={`${d.label} · open this filter's details`}>{d.id}</button>
            {a.onRestoreIds && <button type="button" className="btn quiet" style={{ padding: '0 4px', marginLeft: 4 }} onClick={() => a.onRestoreIds!(new Set([d.id]))} aria-label={`Put ${d.id} back`} title="Put this filter back">↺</button>}</span>)}
        </div>
      </details>}
      {s.kept != null && <span className="pill">keeping {s.kept}
        <button type="button" className="btn quiet" style={{ padding: '0 4px', marginLeft: 4 }} title="List the kept filters in the side panel"
          onClick={() => edits.openGroup({ title: 'Kept by hand (every other filter is hidden)', ids: s.keptIds ?? [], fields: reviewFields, live: 'kept' })}>review</button>
        <button type="button" className="btn quiet" style={{ padding: '0 4px', marginLeft: 2 }} onClick={a.onClearKeep} aria-label="Stop keeping only the brushed filters">×</button></span>}
    </div>
  )

  const net = NETWORKS.find((x) => x.key === s.network)!
  const sitesNote = s.network === 'improve' ? 'IMPROVE EC/OC: TOR (thermal-optical reflectance) · absorption from HIPS'
    : 'SPARTAN EC/OC: FTIR-derived · no TOR reference on these filters'
  return (
    <SideDock open={open} top={top} onClose={() => setOpen(false)} tab={
      <button
        type="button"
        className="subset-tab"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        title={open ? 'Hide the filters' : 'Show the filters'}
      >
        <span className="subset-tab-label">Filters</span>
        <span className="subset-tab-count">{s.shown}/{s.total}</span>
        {nActive > 0 && <span className="subset-tab-badge" title={`${nActive} filter${nActive === 1 ? '' : 's'} active`}>{nActive}</span>}
      </button>
    }>
        <aside className="panel subset subset-panel" aria-label="Filters">
          <header className="subset-head">
            <strong>Filters</strong>
            <span className="pill" title={`${s.shown} of ${s.total} ${caps.datasetLabel} in view`}>{s.shown} of {s.total}</span>
            <span style={{ marginLeft: 'auto', display: 'inline-flex', gap: 4 }}>
              {!isDefault && <button type="button" className="btn quiet" onClick={a.onReset} title="Clear every filter">Reset</button>}
              <button type="button" className="btn quiet" onClick={() => setOpen(false)} aria-label="Close filters" title="Close">✕</button>
            </span>
          </header>
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
              <SitePicker items={s.siteItems} onToggle={a.onToggleSite}
                onAll={s.network === 'improve' ? undefined : a.onAllSites} allSelected={!!s.allSitesSelected}
                onClear={s.network === 'focus' ? undefined : a.onClearSites}
                chipsOnly={s.network === 'focus'} limitNote={s.siteLimitNote}
                noun={s.network === 'improve' ? 'IMPROVE sites' : s.network === 'spartan' ? 'SPARTAN sites' : 'sites'}
                allLabel={s.network === 'both' ? 'All SPARTAN' : 'Select all'} />
            </Row>
          )}
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
                  title="Per-site panels: Shared gives every site the same range, so slopes and levels compare by eye · Own fits each panel to its site's data. With the 1:1 line on, each panel stays square with equal x and y spans." />
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

          {caps.seasons && (
            <section className={`filter-section ${seasonsOn ? '' : 'off'}`}>
              <SectionSwitch label="Seasonality" on={seasonsOn} onChange={(v) => a.onSeasonsOn?.(v)}
                offNote="off · every season" title="Switch off to ignore seasons entirely" />
              {seasonsOn && (
                <>
                  <Row label="calendar">
                    <Combobox label="" title="Season calendar" value={s.convention} onChange={a.onConvention}
                      options={s.calendarOptions.map((k) => ({ key: k, label: `${CALENDAR_LABEL[k] ?? k}${k === s.defaultConvention ? ' (default)' : ''}` }))} />
                    <span className="row-note" style={{ display: 'block' }}>
                      {s.convention === 'local'
                        ? 'Each filter is binned by its own site’s calendar: the met service’s or a site paper’s month bins (docs/site-seasonality.md, docs/site-seasonality-network.md); sites without one use quarters. Fixed month bins, not observed weather.'
                        : s.convention === 'custom'
                        ? 'Your own bins, applied to every site. Pick a season, click months to move them into it, rename freely; they travel in the link.'
                        : s.convention === 'quarters'
                        ? 'The same three-month bins at every site, for like-month comparison; Dec–Feb is winter in the north and summer in the south.'
                        : <>Shared month bins across every site, for like-month comparison: “Belg” at Beijing only means those months. February sits in <strong style={{ color: 'var(--ink-text)' }}>{s.seasons.find((x) => x.months.includes(2))?.name ?? '—'}</strong>.</>}
                    </span>
                  </Row>
                  {s.convention === 'custom' && s.customSeasons && a.onCustomSeasons && <SeasonEditor seasons={s.customSeasons} onChange={a.onCustomSeasons} />}
                  {s.convention === 'rain_rule' && s.rainRule && a.onRainRule && <RainRuleEditor rule={s.rainRule} onChange={a.onRainRule} />}
                  <div className="season-chips">
                    <Chip active={allSeasons} color="var(--ink-muted)" onClick={a.onAllSeasons} label="All" />
                  </div>
                  {seasonGroups.map((g) => (
                    <div key={g.site ?? 'shared'} className="season-group">
                      {/* the site heading only helps when several calendars are listed */}
                      {g.site && seasonGroups.length > 1 && (
                        <span className="season-site">
                          <span className="swatch" style={{ background: meta.sites.find((x) => x.name === g.site)?.color ?? 'var(--ink-muted)' }} />
                          {g.site}
                        </span>
                      )}
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

          <Row label="">
            {caps.excluded ? (
              <label className="toggle">
                <input type="checkbox" checked={s.includeExcluded} onChange={(e) => a.onIncludeExcluded(e.target.checked)} />
                Include {s.nExcluded} excluded samples
              </label>
            ) : null}
            {caps.note && explain && <span className="row-note">{caps.note}</span>}
            {caps.sites && <span className="method-note">{sitesNote}</span>}
          </Row>

          {s.pinned && (
            <PinnedCard row={s.pinned} meta={meta} fields={[s.xField, s.yField, s.field]} inSubset={s.pinnedInSubset} onUnpin={a.onUnpin} onOpen={a.onOpenPinned} />
          )}
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
          return <span key={f} className="chip active">{f}<button type="button" className="btn quiet" aria-label={`Remove ${f}`} title="Remove this ratio" onClick={() => r.remove(f)}>✕</button></span>
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
      <p className="method-note" style={{ display: 'block', margin: '4px 0' }}>Each site and year from its own daily rain (ERA5), {rule.window}-day mean: rains begin above {rule.onset} mm/day (First Dry → Less Rainy), heavy rains above {rule.heavy} (More Rainy) until the last such day (Second Dry). Fitted to the PMF paper's 2023 Addis dates.</p>
      <div className="rain-rule-fields">
        {num('window', 'Window (days)', 1, 'Days in the centred rolling mean of daily rain')}
        {num('onset', 'Rains begin (mm/day)', 0.5, 'Mean rain above which the dry season ends')}
        {num('heavy', 'Heavy rains (mm/day)', 0.5, 'Mean rain above which it is the More Rainy season')}
        {!isDefault && <button type="button" className="btn quiet" onClick={() => onChange(DEFAULT_RAIN_RULE)}>Reset</button>}
      </div>
    </div>
  )
}
