import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Combobox } from '@/components/Combobox'
import { DateBrush } from '@/components/DateBrush'
import { SitePicker, siteFullName } from '@/components/SitePicker'
import { StepButton } from '@/components/StepButton'
import type { SubsetActions, SubsetState } from '@/components/SubsetBar'
import { useSidePanelOpen, useSidePanelView } from '@/hooks/useSidePanel'
import { useMediaQuery } from '@/hooks/useMediaQuery'
import { ROWS, SCOPES, SCOPE_LABEL, type Layout, type Scope } from '@/lib/dashboard'
import type { FilterRow, MetaFile } from '@/lib/types'

/** what the Dashboard's header needs from App's Filters panel: the same state and actions */
export interface DashFilters { state: SubsetState; actions: SubsetActions; brushRows: FilterRow[] }

export interface SheetActions {
  names: string[]
  active: number
  onPick: (i: number) => void
  onAdd: () => void
  onDuplicate: () => void
  onRename: (name: string) => void
  onDelete: () => void
  onReset: () => void
}

/** below this width the header keeps the scope and sheet and folds the rest into ⋯ */
const NARROW = 820

/**
 * The Dashboard's options, one line in the app header beside the page menu (Kibana and
 * Grafana keep scope, time and filters in one bar): the scope and its site picker, seasons,
 * dates, the full Filters panel behind one button (with the count), the sheet, then Grid /
 * Focus and Edit. On a narrow screen everything but the scope and sheet folds into ⋯.
 */
export function DashHeader({ meta, filters, scope, onScope, onPickSite, layout, onLayout, edit, onEdit, onAdd, rowsUsed, sheets, display, onPresent }: {
  meta: MetaFile
  filters: DashFilters | null
  scope: Scope
  onScope: (s: Scope) => void
  onPickSite: (code: string) => void
  layout: Layout
  onLayout: (l: Layout) => void
  edit: boolean
  onEdit: (v: boolean) => void
  onAdd: () => void
  /** rows the current tiles need; more than the grid's means it scrolls */
  rowsUsed: number
  sheets: SheetActions
  /** the header's display toggles (readable text, explain), moved into ⋯ to keep one line */
  display?: ReactNode
  /** present the sheets full screen (absent on a phone) */
  onPresent?: () => void
}) {
  const [, setFiltersOpen] = useSidePanelOpen()
  const [, setPanelView] = useSidePanelView()
  const narrow = useMediaQuery(`(max-width: ${NARROW}px)`)
  if (!filters) return null
  const { state: s, actions: a } = filters
  const spartan = s.siteItems.filter((x) => x.group !== 'IMPROVE')
  const picked = s.siteItems.filter((x) => x.selected)
  const allSeasons = s.activeSeasons.size === s.seasons.length
  const others = (s.includedIds?.length ? 1 : 0) + (s.pmfOn ? 1 : 0) + ((s.dropped ?? 0) > 0 || s.kept != null ? 1 : 0) + (s.convention !== s.defaultConvention ? 1 : 0)

  const scopeSeg = <span className="seg dash-scope" role="radiogroup" aria-label="Scope">
    {SCOPES.map((k) => <button key={k} type="button" role="radio" aria-checked={scope === k} className={scope === k ? 'active' : ''}
      onClick={() => onScope(k)} title={SCOPE_TITLE[k]}>{SCOPE_LABEL[k]}</button>)}
  </span>
  // ‹ › step through the sites in the list's order (wrapping); the picker keeps one width, so the arrows stay put
  const siteOrder = [...spartan].sort((x, y) => x.label.localeCompare(y.label))
  const siteAt = picked.length === 1 ? siteOrder.findIndex((x) => x.key === picked[0].key) : -1
  const siteStep = (d: number) => siteOrder[siteAt < 0 ? (d > 0 ? 0 : siteOrder.length - 1) : (siteAt + d + siteOrder.length) % siteOrder.length]
  const stepSite = (d: number) => { const to = siteStep(d); if (to) onPickSite(to.key) }
  const sitePicker = scope === 'site'
    ? <span className="dash-site-step">
      <StepButton dir="prev" label="Previous site" title={`Previous site: ${siteStep(-1)?.label ?? ''}`} disabled={siteOrder.length < 2} onClick={() => stepSite(-1)} />
      <Combobox label="" value={picked.length === 1 ? picked[0].key : ''} title="The site every tile shows"
        options={siteOrder.map((x) => ({ key: x.key, label: siteFullName(x), hint: String(x.n) }))}
        onChange={onPickSite} />
      <StepButton dir="next" label="Next site" title={`Next site: ${siteStep(1)?.label ?? ''}`} disabled={siteOrder.length < 2} onClick={() => stepSite(1)} />
    </span>
    : scope === 'sites'
      ? <HeaderPopover label={picked.length <= 2 ? picked.map((x) => x.label).join(', ') || 'No sites' : `${picked.length} sites`} title="Which sites the tiles compare" wide>
        <SitePicker items={s.siteItems} onToggle={a.onToggleSite} onAll={s.network === 'improve' ? undefined : a.onAllSites}
          allSelected={!!s.allSitesSelected} onClear={s.network === 'focus' ? undefined : a.onClearSites} onFocus={a.onFocusSites}
          chipsOnly={s.network === 'focus'} limitNote={s.siteLimitNote} />
      </HeaderPopover>
      : <span className="dash-head-note" title="Every site in the network; All filters switches to IMPROVE">{picked.length} sites</span>
  const seasons = s.seasons.length > 0 && <HeaderPopover label={allSeasons || !s.seasonsOn ? 'Seasons' : `${s.activeSeasons.size}/${s.seasons.length} seasons`}
    active={!allSeasons && !!s.seasonsOn} title="Seasons under the calendar in force (All filters changes the calendar)">
    {/* grouped by the site whose calendar each season is (one group under shared bins), as in the Filters panel */}
    <div className="dash-season-groups">
      {[...new Set(s.seasons.map((se) => se.site ?? ''))].map((site) => (
        <div key={site || 'all'} className="dash-season-group">
          {site && <div className="dash-season-site">{site}</div>}
          <div className="dash-chips">
            {s.seasons.filter((se) => (se.site ?? '') === site).map((se) => <button key={se.name} type="button" className={`chip${s.activeSeasons.has(se.name) ? ' active' : ''}`}
              style={{ ['--chip' as string]: se.color }} onClick={() => a.onToggleSeason(se.name)} disabled={!s.seasonCounts[se.name]} title={se.name}>
              {site && se.name.startsWith(`${site} · `) ? se.name.slice(site.length + 3) : se.name} <span className="dash-chip-n">{s.seasonCounts[se.name] ?? 0}</span></button>)}
          </div>
        </div>
      ))}
    </div>
    {!allSeasons && <button type="button" className="btn quiet" onClick={a.onAllSeasons}>Every season</button>}
  </HeaderPopover>
  const dates = <HeaderPopover label={s.dateRange ? `${s.dateRange[0].slice(0, 7)} → ${s.dateRange[1].slice(0, 7)}` : 'Dates'} active={!!s.dateRange} title="Sampling dates: drag across the strip" wide>
    <DateBrush rows={filters.brushRows} meta={meta} range={s.dateRange} onChange={a.onDateRange} width={narrow ? 300 : 560} />
    {s.dateRange && <button type="button" className="btn quiet" onClick={() => a.onDateRange(null)}>Every date</button>}
  </HeaderPopover>
  const allFilters = <button type="button" className={`btn${others ? ' on' : ''}`} onClick={() => { setPanelView('filters'); setFiltersOpen(true) }}
    title={`${s.shown} of ${s.total} filters in view. Every filter: network, calendar, exclusions, PMF source, hand edits, units, fits`}>
    Filters <span className="dash-count">{s.shown.toLocaleString()}</span>{others ? <span className="dash-badge">{others}</span> : null}
  </button>
  const sheetMenu = <SheetMenu sheets={sheets} />
  const layoutSeg = <span className="seg" role="radiogroup" aria-label="Layout">
    <button type="button" role="radio" aria-checked={layout === 'grid'} className={layout === 'grid' ? 'active' : ''} onClick={() => onLayout('grid')}
      title="Grid: every tile at its own size, fitted to the window">▦</button>
    <button type="button" role="radio" aria-checked={layout === 'focus'} className={layout === 'focus' ? 'active' : ''} onClick={() => onLayout('focus')}
      title="Focus: headline numbers on top, one chart large, the others in a rail">◧</button>
  </span>

  // a phone has no room for three scope buttons: the scope is a small menu of its own
  const scopeMenu = <HeaderPopover label={SCOPE_LABEL[scope]} title="Scope: one site, compare sites, or the network">
    <div className="dash-sheets" role="listbox" aria-label="Scope">
      {SCOPES.map((k) => <button key={k} type="button" role="option" data-close aria-selected={scope === k} className={`dash-sheet${scope === k ? ' active' : ''}`}
        onClick={() => onScope(k)} title={SCOPE_TITLE[k]}>{SCOPE_LABEL[k]}</button>)}
    </div>
  </HeaderPopover>
  if (narrow) return (
    <div className="dash-head" role="toolbar" aria-label="Dashboard options">
      {scopeMenu}{sheetMenu}
      <span className="dash-head-right">
        <HeaderPopover label="⋯" title="Sites, seasons, dates, filters and layout" align="right">
          <div className="dash-more">{sitePicker}{seasons}{dates}{allFilters}{layoutSeg}<button type="button" className="btn" onClick={onAdd}>+ Add chart</button>{display}</div>
        </HeaderPopover>
      </span>
    </div>
  )
  return (
    <div className="dash-head" role="toolbar" aria-label="Dashboard options">
      {scopeSeg}{sitePicker}{seasons}{dates}{allFilters}
      <span className="dash-sep" />
      {sheetMenu}
      <span className="dash-head-right">
        {edit && rowsUsed > ROWS && <span className="dash-warn" title="The tiles reach below the window, so the grid scrolls: shrink or remove some to fit">{rowsUsed}/{ROWS} rows</span>}
        {edit && <button type="button" className="btn" onClick={onAdd}>+ Add</button>}
        {layoutSeg}
        {onPresent && !edit && <button type="button" className="btn" onClick={onPresent}
          title="Present: the sheets full screen with no controls; → and ← step through them, Esc ends">▶ Present</button>}
        <button type="button" className={`btn${edit ? ' on' : ''}`} aria-pressed={edit} onClick={() => onEdit(!edit)}
          title={edit ? 'Done arranging' : 'Arrange: drag a tile by its bar, resize from its corner, add and remove tiles'}>{edit ? 'Done' : 'Edit'}</button>
        {display && <HeaderPopover label="Aa" title="Display: readable text, explanations" align="right"><div className="dash-more">{display}</div></HeaderPopover>}
      </span>
    </div>
  )
}

const SCOPE_TITLE: Record<Scope, string> = {
  site: 'One site in depth: its seasons, time series, composition',
  sites: 'Sites side by side (starts on the four focus sites)',
  network: 'Every site in the network: map first, a click opens a site',
}

/** the sheet picker: a scope's named dashboards (Grafana's dashboard links, Power BI's pages) */
function SheetMenu({ sheets: sh }: { sheets: SheetActions }) {
  const [renaming, setRenaming] = useState(false)
  const name = sh.names[sh.active] ?? 'Sheet'
  return (
    <HeaderPopover label={name} title="Sheets: this scope's named dashboards">
      <div className="dash-sheets" role="listbox" aria-label="Sheets">
        {sh.names.map((n, i) => <button key={i} type="button" role="option" data-close aria-selected={i === sh.active} className={`dash-sheet${i === sh.active ? ' active' : ''}`}
          onClick={() => sh.onPick(i)}>{n}</button>)}
      </div>
      {renaming
        ? <form className="dash-rename" onSubmit={(e) => { e.preventDefault(); const v = new FormData(e.currentTarget).get('name'); if (typeof v === 'string' && v.trim()) sh.onRename(v.trim()); setRenaming(false) }}>
          <input name="name" className="select" defaultValue={name} autoFocus aria-label="Sheet name" /><button type="submit" className="btn">Save</button>
        </form>
        : <div className="dash-sheet-actions">
          <button type="button" className="btn quiet" onClick={sh.onAdd}>+ New</button>
          <button type="button" className="btn quiet" onClick={sh.onDuplicate}>Duplicate</button>
          <button type="button" className="btn quiet" onClick={() => setRenaming(true)}>Rename</button>
          {sh.names.length > 1 && <button type="button" className="btn quiet" onClick={sh.onDelete}>Delete</button>}
          <button type="button" className="btn quiet" onClick={sh.onReset} title="This scope's starter sheets">Reset</button>
        </div>}
    </HeaderPopover>
  )
}


/** a header button that opens a small panel below it; a click outside (not in a picker's popup) or Esc closes it */
function HeaderPopover({ label, title, active = false, wide = false, align = 'left', children }: {
  label: string; title?: string; active?: boolean; wide?: boolean; align?: 'left' | 'right'; children: ReactNode
}) {
  const [open, setOpen] = useState(false)
  const root = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    if (!open) return
    const off = (e: PointerEvent) => {
      const t = e.target as Element
      // a combobox's options and the map dialog are drawn outside the popover
      if (!root.current?.contains(t) && !t.closest?.('[role="listbox"], .combobox-menu, .map-dialog-backdrop')) setOpen(false)
    }
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('pointerdown', off)
    window.addEventListener('keydown', esc)
    return () => { window.removeEventListener('pointerdown', off); window.removeEventListener('keydown', esc) }
  }, [open])
  return (
    <span className="dash-pop" ref={root}>
      <button type="button" className={`btn${active ? ' on' : ''}${open ? ' open' : ''}`} aria-expanded={open} title={title} onClick={() => setOpen(!open)}>
        {label} <span aria-hidden className="dash-caret">▾</span>
      </button>
      {/* a choice marked data-close (a scope, a sheet) closes the popover once picked */}
      {open && <div className={`dash-pop-panel${wide ? ' wide' : ''}${align === 'right' ? ' right' : ''}`}
        onClick={(e) => { if ((e.target as Element).closest('[data-close]')) setOpen(false) }}>{children}</div>}
    </span>
  )
}
