import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Empty } from '@/components/ChartFrame'

// the dialog's map loads MapLibre, so it is fetched when the dialog first opens
const SiteMap = lazy(() => import('@/components/SiteMapPicker').then((m) => ({ default: m.SiteMapPicker })))

/** One pickable site: a row in the search box's list and a dot on the map. */
export interface SiteItem {
  key: string
  label: string
  /** state, country or site code under the name */
  sub?: string
  /** another name the site goes by (JPL is Pasadena): shown beside the name and found by the search */
  aka?: string
  /** the country, for a site whose name lacks it (the focus sites keep their short names: "Addis Ababa", then ", Ethiopia") */
  country?: string
  /** the network, when two are shown together */
  group?: string
  lat: number | null
  lon: number | null
  color: string
  /** filters in the current subset (or in the site file before it is loaded) */
  n: number
  selected: boolean
  /** nothing to draw (no filters in the network's public release): greyed, listed last, not pickable */
  disabled?: boolean
}

/**
 * Other names the readers use for a site, by its code or name: the sheets and the notebooks call SPARTAN's JPL
 * site (USPA) Pasadena. Shown beside the name in the pickers and matched by their search.
 */
export const SITE_AKA: Readonly<Record<string, string>> = { USPA: 'Pasadena', JPL: 'Pasadena' }
export const siteAka = (codeOrName: string, name?: string) => SITE_AKA[codeOrName] ?? (name ? SITE_AKA[name] : undefined)
/** the country a SPARTAN site's name leaves out, from its location ("Beijing" at "Beijing, China"); none when the name has one */
export const siteCountry = (name: string, location?: string | null) => (name.includes(',') || !location?.includes(',') ? undefined : location.split(',').pop()!.trim() || undefined)
/** a site's whole name on one line: "JPL (Pasadena), United States" */
export const siteFullName = (it: Pick<SiteItem, 'label' | 'aka' | 'country'>) => `${it.label}${it.aka ? ` (${it.aka})` : ''}${it.country ? `, ${it.country}` : ''}`

/**
 * Run `fn` once the browser has painted the current frame (the ticked box), so the charts' work
 * starts after it rather than before. rAF then a task is the usual "yield to paint"; the timeout
 * covers a hidden page, where rAF does not fire.
 */
function afterPaint(fn: () => void) {
  let done = false
  const run = () => { if (!done) { done = true; fn() } }
  requestAnimationFrame(() => setTimeout(run, 0))
  setTimeout(run, 100)
}

/**
 * Sites for the Filters sidebar: a search box that filters a scrolling checkbox list (picked
 * sites first), quick buttons (focus sites, all, none) and **Map…**, a large map dialog for
 * picking by clicking. All edit one selection. With only the four focus sites, plain chips.
 */
export function SitePicker({ items: itemsIn, onToggle: toggleIn, onAll, onClear, onFocus, allSelected = false, chipsOnly = false, limitNote, noun = 'sites', allLabel = 'Select all', mapPicker = true, orderSession }: {
  items: SiteItem[]
  /** Sidebar open state: snapshot selected-first order when reopened; undefined preserves input order. */
  orderSession?: boolean
  onToggle: (key: string) => void
  /** select every site (absent where sites load one by one, e.g. IMPROVE) */
  onAll?: () => void
  /** deselect every site: the charts then show nothing until sites are picked */
  onClear?: () => void
  /** back to the four focus sites (SPARTAN's default) */
  onFocus?: () => void
  allSelected?: boolean
  /** few sites (the focus-sites view): toggle chips, no list */
  chipsOnly?: boolean
  /** e.g. "3 of 10 IMPROVE sites" */
  limitNote?: string
  noun?: string
  allLabel?: string
  /** the Map… dialog; off where the picker already sits beside a map */
  mapPicker?: boolean
}) {
  const [mapOpen, setMapOpen] = useState(false)
  const [query, setQuery] = useState('')
  // a click ticks the box at once (local state, a cheap render) while App applies the pick to the
  // charts as a transition; an entry goes once the items agree, or after a while if App refused it
  // (the IMPROVE site cap)
  const [pending, setPending] = useState<Map<string, { on: boolean; t: number }>>(() => new Map())
  useEffect(() => {
    setPending((cur) => {
      if (!cur.size) return cur
      const now = performance.now(), byKey = new Map(itemsIn.map((it) => [it.key, it.selected]))
      const next = new Map([...cur].filter(([k, p]) => byKey.get(k) !== p.on && now - p.t < 2000))
      return next.size === cur.size ? cur : next
    })
  }, [itemsIn])
  const items = useMemo(() => (pending.size ? itemsIn.map((it) => {
    const p = pending.get(it.key)
    return p && p.on !== it.selected ? { ...it, selected: p.on } : it
  }) : itemsIn), [itemsIn, pending])
  const onToggle = (key: string) => {
    const it = items.find((x) => x.key === key)
    if (it) setPending((cur) => new Map(cur).set(key, { on: !it.selected, t: performance.now() }))
    afterPaint(() => toggleIn(key))
  }
  const selected = items.filter((it) => it.selected)
  const mixed = new Set(items.map((it) => it.group)).size > 1
  const mapButton = mapPicker && <button type="button" className="btn quiet" onClick={() => setMapOpen(true)} title="Pick sites on a large map">Map…</button>
  const q = query.trim().toLowerCase()
  const order = useMemo(() => new Map((orderSession === undefined ? itemsIn : [...itemsIn.filter(it => it.selected), ...itemsIn.filter(it => !it.selected)]).map((it, i) => [it.key, i])), [orderSession, itemsIn.map(it => it.key).join('|')])
  const shown = useMemo(() => {
    const hit = q ? items.filter((it) => `${it.label} ${it.aka ?? ''} ${it.country ?? ''} ${it.sub ?? ''} ${it.key}`.toLowerCase().includes(q)) : items
    // Keep a checkbox at its original position while the reader edits the current selection.
    return [...hit].sort((a, b) => Number(!!a.disabled) - Number(!!b.disabled) || (order.get(a.key) ?? Infinity) - (order.get(b.key) ?? Infinity))
  }, [items, q, order])
  return (
    <div className="site-picker">
      {chipsOnly ? (
        <div className="season-chips">
          {onAll && <button type="button" className={`chip ${allSelected ? 'active' : ''}`} style={{ ['--chip' as string]: 'var(--ink-muted)' }} onClick={() => { setPending(new Map()); onAll() }}>All</button>}
          {items.map((it) => (
            <button key={it.key} type="button" className={`chip ${it.selected ? 'active' : ''} ${it.n === 0 ? 'disabled' : ''}`}
              style={{ ['--chip' as string]: it.color }} disabled={it.n === 0} onClick={() => onToggle(it.key)}>
              {it.label} · {it.n}
            </button>
          ))}
          {mapButton}
        </div>
      ) : (
        <>
          <div className="site-picker-bar">
            {onFocus && <button type="button" className="btn quiet" onClick={() => { setPending(new Map()); onFocus() }} title="Beijing, Delhi, JPL and Addis Ababa">Focus sites</button>}
            {onAll && <button type="button" className="btn quiet" onClick={() => { setPending(new Map()); onAll() }} disabled={allSelected}>{allLabel}</button>}
            {onClear && <button type="button" className="btn quiet" onClick={() => { setPending(new Map()); onClear() }} disabled={selected.length === 0}>Deselect all</button>}
            <span className="site-picker-count">{limitNote ?? `${selected.length} of ${items.length}`}</span>
          </div>
          <div className="site-picker-bar">
            <input className="select site-filter" type="search" placeholder={`Search ${items.length} ${noun}…`} value={query}
              onChange={(e) => setQuery(e.target.value)} aria-label={`Search ${noun}`} />
            {mapButton}
          </div>
          <ul className="site-list" aria-label={noun}>
            {shown.map((it) => (
              <li key={it.key}>
                <label className={`site-row ${it.selected ? 'on' : ''}${it.disabled ? ' disabled' : ''}`} title={it.disabled ? `${it.label}: no filters in the public release yet` : undefined}>
                  <input type="checkbox" checked={it.selected} disabled={it.disabled} onChange={() => onToggle(it.key)} />
                  <span className="site-check" style={{ ['--chip' as string]: it.color }} aria-hidden="true" />
                  <span className="site-name">{it.label}{it.aka && <span className="site-aka"> ({it.aka})</span>}{it.country && `, ${it.country}`}{it.sub && <span className="site-sub"> · {it.sub}</span>}</span>
                  {it.group && mixed && <span className={`site-net ${it.group.toLowerCase()}`}>{it.group}</span>}
                  <span className="site-n">{it.n.toLocaleString()}</span>
                </label>
              </li>
            ))}
            {shown.length === 0 && <li className="site-empty">No site matches “{query}”.</li>}
          </ul>
          {selected.length === 0 && <span className="site-empty">No {noun} selected: the charts are empty until you pick some.</span>}
        </>
      )}
      {mapOpen && <MapDialog items={items} onToggle={onToggle} onClose={() => setMapOpen(false)} limitNote={limitNote} noun={noun} />}
    </div>
  )
}

/** A large map in a dialog: click dots to add or remove sites; scroll, drag or +/− to move. */
function MapDialog({ items, onToggle, onClose, limitNote, noun }: { items: SiteItem[]; onToggle: (key: string) => void; onClose: () => void; limitNote?: string; noun: string }) {
  const dialog = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    dialog.current?.focus()
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  const n = items.filter((it) => it.selected).length
  return createPortal(
    <div className="map-dialog-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="map-dialog" role="dialog" aria-modal="true" aria-label={`Pick ${noun} on the map`} tabIndex={-1} ref={dialog}>
        <header>
          <strong>Pick {noun} on the map</strong>
          <span className="site-picker-count">{limitNote ?? `${n} of ${items.length} selected`}</span>
          <button type="button" className="btn" onClick={onClose}>Done</button>
        </header>
        <Suspense fallback={<Empty>Loading the map…</Empty>}><SiteMap items={items} onToggle={onToggle} /></Suspense>
      </div>
    </div>,
    document.body,
  )
}
