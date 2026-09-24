import { useEffect, useRef, useState } from 'react'
import { useDimensions } from '@/hooks/useGalleryData'
import { DateBrush, type DateRange } from '@/components/DateBrush'
import { FieldSelect, prettyUnit } from '@/components/FieldSelect'
import { Segmented, SwapButton, Toggle } from '@/components/ChartFrame'
import { AXES_MODES, type AxesOpts } from '@/lib/axes'
import { fmt } from '@/lib/stats'
import { usePrefs } from '@/lib/prefs'
import { useHighlight } from '@/lib/highlight'
import type { FilterRow, MetaFile, SeasonMeta } from '@/lib/types'
import { PMF_SCHEMES, type PmfGroup, type PmfScheme } from '@/lib/pmfGroups'

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
  /** e.g. "filters" or "PMF filters" — what the count is counting */
  datasetLabel: string
  /** shown when a control is absent because the data can't support it */
  note?: string
  /** the tab's filters can be grouped by their ETAD PMF source */
  pmf?: boolean
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
  onToggleSite: (name: string) => void
  onAllSites: () => void
  onIncludeExcluded: (v: boolean) => void
  onDateRange: (r: DateRange | null) => void
  onXField: (v: string) => void
  onYField: (v: string) => void
  onField: (v: string) => void
  onAxes: (v: AxesOpts) => void
  onUnpin: () => void
  onOpenPinned: () => void
  onReset: () => void
}

/** The calendar picker's wording: per-site is the default; the Ethiopian bins are shared month bins. */
const CALENDAR_LABEL: Record<string, string> = {
  local: 'Per site · each site’s local calendar',
  dry_feb: 'Shared Ethiopian month bins · Feb in dry',
  belg_feb: 'Shared Ethiopian month bins · Feb in Belg',
}

const OPEN_KEY = 'gallery.filters.open'
const readOpen = () => { try { return localStorage.getItem(OPEN_KEY) === '1' } catch { return false } }
const writeOpen = (v: boolean) => { try { localStorage.setItem(OPEN_KEY, v ? '1' : '0') } catch { /* private window */ } }

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
  const [open, setOpenState] = useState(readOpen)
  const setOpen = (v: boolean) => { setOpenState(v); writeOpen(v) }
  const [top, setTop] = useState(96)
  const { explain } = usePrefs()
  const hl = useHighlight()

  // sit just below the sticky app header, whatever height its tab row wraps to
  useEffect(() => {
    const measure = () => setTop((document.querySelector('.app-header') as HTMLElement | null)?.offsetHeight ?? 96)
    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [])

  const allSeasons = s.activeSeasons.size === s.seasons.length
  const allSites = s.activeSites.size === s.sites.length
  const seasonsOn = s.seasonsOn ?? true
  const pmfOn = !!caps.pmf && !!s.pmfOn
  const allPmf = !pmfOn || !s.pmfGroups || (s.activePmf?.size ?? 0) === s.pmfGroups.length
  const isDefault =
    allSeasons && allSites && !pmfOn && seasonsOn && !s.includeExcluded && !s.dateRange && s.convention === s.defaultConvention
  const nActive = (caps.seasons && seasonsOn && !allSeasons ? 1 : 0) + (caps.sites && !allSites ? 1 : 0) + (pmfOn ? 1 : 0) + (s.dateRange ? 1 : 0) + (s.includeExcluded && caps.excluded ? 1 : 0)

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
      {caps.sites && !allSites && <span>sites: {[...s.activeSites].join(', ')}</span>}
      {pmfOn && <span>PMF: {allPmf ? 'every source (PMF days only)' : [...(s.activePmf ?? [])].join(', ')}</span>}
      {s.dateRange && <span>{s.dateRange[0]} → {s.dateRange[1]}</span>}
      {s.includeExcluded && caps.excluded && <span>+{s.nExcluded} excluded</span>}
      {hl.selected && (
        <span className="pill" title="Filters picked by a brush; every chart dims the rest. A highlight, not a filter: statistics still use the whole subset.">
          {hl.selected.size} brushed
          <button type="button" className="btn quiet" style={{ padding: '0 4px', marginLeft: 4 }} onClick={() => hl.setSelected(null)}>×</button>
        </span>
      )}
    </div>
  )

  return (
    <>
      <button
        type="button"
        className={`subset-tab ${open ? 'open' : ''}`}
        style={{ top: top + 24 }}
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        title={open ? 'Hide the filters' : 'Show the filters'}
      >
        <span className="subset-tab-label">Filters</span>
        <span className="subset-tab-count">{s.shown}/{s.total}</span>
        {nActive > 0 && <span className="subset-tab-badge" title={`${nActive} filter${nActive === 1 ? '' : 's'} active`}>{nActive}</span>}
      </button>

      {open && (
        <aside className="panel subset subset-panel" style={{ top, height: `calc(100vh - ${top}px)` }} aria-label="Filters">
          <header className="subset-head">
            <strong>Filters</strong>
            <span className="pill">{s.shown} of {s.total} {caps.datasetLabel}</span>
            <span style={{ marginLeft: 'auto', display: 'inline-flex', gap: 4 }}>
              {!isDefault && <button type="button" className="btn quiet" onClick={a.onReset} title="Clear every filter">reset</button>}
              <button type="button" className="btn quiet" onClick={() => setOpen(false)} aria-label="Close filters" title="Close">✕</button>
            </span>
          </header>
          {summary}

          {caps.fields === 'pair' && (
            <>
              <Row label="axes">
                <FieldSelect label="x" value={s.xField} meta={meta} onChange={a.onXField} />
                <SwapButton onClick={() => { a.onXField(s.yField); a.onYField(s.xField) }} />
                <FieldSelect label="y" value={s.yField} meta={meta} onChange={a.onYField} />
                {explain && <span className="row-note">shared by every chart on this tab</span>}
              </Row>
              <Row label="crossplot axes">
                <Toggle
                  label="1:1 line"
                  checked={s.axes.identity}
                  onChange={(v) => a.onAxes({ ...s.axes, identity: v })}
                  title="Squares the panels, shares the x and y domain, and turns on the Deming slope"
                />
                <Segmented
                  value={s.axes.mode}
                  options={AXES_MODES}
                  onChange={(v) => a.onAxes({ ...s.axes, mode: v })}
                  title="Data: 0 always visible, negatives kept · From 0,0: both axes start at zero · Show intercept: extend to the fitted lines' y-intercepts and mark them"
                />
                <Toggle label="log axes" checked={s.axes.log} onChange={(v) => a.onAxes({ ...s.axes, log: v })} />
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

          {caps.sites && (
            <Row label="sites">
              <Chip active={allSites} color="var(--ink-muted)" onClick={a.onAllSites} label="all" />
              {s.sites.map((name) => {
                const n = s.siteCounts[name] ?? 0
                return (
                  <Chip
                    key={name}
                    active={s.activeSites.has(name)}
                    color={meta.sites.find((x) => x.name === name)?.color ?? 'var(--ink-muted)'}
                    disabled={n === 0}
                    onClick={() => n > 0 && a.onToggleSite(name)}
                    label={`${name} · ${n}`}
                  />
                )
              })}
            </Row>
          )}

          {caps.seasons && (
            <section className={`filter-section ${seasonsOn ? '' : 'off'}`}>
              <SectionSwitch label="Seasonality" on={seasonsOn} onChange={(v) => a.onSeasonsOn?.(v)}
                offNote="off · every season" title="Switch off to ignore seasons entirely" />
              {seasonsOn && (
                <>
                  <Row label="calendar">
                    <select className="select" value={s.convention} onChange={(e) => a.onConvention(e.target.value)}>
                      {['local', ...Object.keys(meta.season_conventions)].map((k) => (
                        <option key={k} value={k}>
                          {CALENDAR_LABEL[k] ?? k}
                          {k === s.defaultConvention ? ' (default)' : ''}
                        </option>
                      ))}
                    </select>
                    <span className="row-note" style={{ display: 'block' }}>
                      {s.convention === 'local'
                        ? 'Each filter is binned by its own site’s calendar (docs/site-seasonality.md). Fixed month bins, not observed weather.'
                        : <>Shared month bins across every site, for like-month comparison: “Belg” at Beijing only means those months. February sits in <strong style={{ color: 'var(--ink-text)' }}>{s.seasons.find((x) => x.months.includes(2))?.name ?? '—'}</strong>.</>}
                    </span>
                  </Row>
                  <div className="season-chips">
                    <Chip active={allSeasons} color="var(--ink-muted)" onClick={a.onAllSeasons} label="all" />
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
                    <Chip active={allPmf} color="var(--ink-muted)" onClick={() => a.onAllPmf?.()} label="all" />
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
                  <span className="row-note" style={{ display: 'block' }}>
                    Only filters with an ETAD PMF day (102 days of 2023, matched by exact date) are kept while this is on. Combines with seasonality.
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
                    <button type="button" className="btn quiet" onClick={() => a.onDateRange(null)}>clear</button>
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
                include {s.nExcluded} excluded samples
              </label>
            ) : null}
            {caps.note && explain && <span className="row-note">{caps.note}</span>}
            {caps.sites && <span className="method-note">SPARTAN EC/OC: FTIR-derived · no TOR reference on these filters</span>}
            <span className="row-tail">MAC = {meta.mac_value}</span>
          </Row>

          {s.pinned && (
            <PinnedCard row={s.pinned} meta={meta} fields={[s.xField, s.yField, s.field]} inSubset={s.pinnedInSubset} onUnpin={a.onUnpin} onOpen={a.onOpenPinned} />
          )}
        </aside>
      )}
    </>
  )
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
  const shown = [...new Set([...fields, 'EC (FTIR)', 'EC (ChemSpec FTIR)', 'HIPS BC', 'PM2.5 mass'])].filter((f) =>
    meta.fields.includes(f)
  )
  return (
    <div className="pinned">
      <span className="id">📌 {row.id}</span>
      <span>
        <span style={{ display: 'inline-block', width: 9, height: 9, borderRadius: 5, background: row.color as string, marginRight: 5 }} />
        {row.site} · {row.date} · {row.season}
      </span>
      {shown.map((f) => (
        <span key={f} className="kv">
          <span>{f}</span> {fmt(row[f] as number | null, 3)}
        </span>
      ))}
      {row.excluded && <span className="reason">excluded: {row.exclusion_reason}</span>}
      {!inSubset && <span className="row-note">(outside the current subset)</span>}
      <span style={{ marginLeft: 'auto', display: 'inline-flex', gap: 4 }}>
        <button type="button" className="btn" onClick={onOpen}>details</button>
        <button type="button" className="btn quiet" onClick={onUnpin}>unpin</button>
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
