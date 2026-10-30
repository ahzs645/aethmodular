import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { baseFilterId, useHighlightActions } from '@/lib/highlight'
import { usePresentation } from '@/lib/presentation'
import { useExportSites } from '@/lib/exportSites'
import type { ExportSite } from '@/lib/exportColors'
import { UndoIcon } from '@/components/Icons'

/**
 * Hand edits from inside the charts: right-click a filter (or a bin of filters) to
 * exclude it, keep only it, or open it. The edits are the same drop / keep sets the
 * brush writes, so they show up in the Filters panel and travel in the link.
 *
 * A mark opts in with a data attribute, read by one document-level listener:
 *   data-fid="ETAD-0144"                          one filter
 *   data-fids="id1,id2,…" data-ftitle="…"         a bin (a hex, a histogram bar)
 * A chart that finds its point itself (the scatterplot's nearest-point hit layer)
 * calls `openMenu` from its own onContextMenu instead.
 */
export interface FilterGroup {
  /** e.g. "Hex at HIPS BC ≈ 1.2, EC (FTIR) ≈ 0.8" */
  title: string
  ids: string[]
  /** fields worth a column in the side panel, in order (defaults to the chart's x and y) */
  fields?: string[]
  /** a review of the hand edits: the panel keeps adding filters as they are excluded (or kept) */
  live?: 'dropped' | 'kept'
}

interface MenuTarget { ids: string[]; title: string; group?: FilterGroup }

/**
 * A chart's own item in this menu: an element marked data-menu-action="Label" (and data-menu-detail, any
 * string) adds "Label" to the menu when right-clicked; choosing it dispatches MENU_ACTION on the element, with
 * the detail, for the chart to handle (charts/correlation/Correlogram opens a matrix cell as a crossplot).
 */
export const MENU_ACTION = 'gallery:menu-action'
interface MenuAction { el: Element; label: string; detail: string }
const menuActionOf = (t: Element): MenuAction | null => {
  const el = t.closest('[data-menu-action]')
  return el ? { el, label: el.getAttribute('data-menu-action')!, detail: el.getAttribute('data-menu-detail') ?? '' } : null
}

/** a filter id's site: its code, the part before the last dash (ETAD-0052 → ETAD) */
const siteOf = (id: string) => id.replace(/-[^-]*$/, '')

/**
 * The sites a chart draws, from its panel: the filter ids on its marks (data-fid, data-fids) and the sites its
 * marks are tagged with (data-export-site, by code or name). The right-click menu offers to put back only the
 * filters excluded by hand from these sites: one excluded at Addis has nothing to do with a chart of Delhi.
 */
function panelSites(el: Element | null, sites: readonly ExportSite[], ids: string[] = []): Set<string> {
  const out = new Set(ids.map(siteOf))
  const panel = el?.closest('.panel')
  if (!panel) return out
  for (const m of panel.querySelectorAll('[data-fid]')) out.add(siteOf(m.getAttribute('data-fid')!))
  for (const m of panel.querySelectorAll('[data-fids]')) for (const id of (m.getAttribute('data-fids') ?? '').split(',')) if (id) out.add(siteOf(id))
  const byName = new Map(sites.flatMap((x) => [[x.code, x.code], [x.name, x.code]] as [string, string][]))
  for (const m of panel.querySelectorAll('[data-export-site]')) { const code = byName.get(m.getAttribute('data-export-site')!); if (code) out.add(code) }
  return out
}

interface FilterEdits {
  /** whether the open tab honours drop / keep (the network tabs) */
  enabled: boolean
  dropped: Set<string>
  kept: Set<string> | null
  drop: (ids: Iterable<string>) => void
  keep: (ids: Iterable<string>) => void
  restore: (ids: Iterable<string>) => void
  group: FilterGroup | null
  openGroup: (g: FilterGroup | null) => void
  /** `e.target`: the clicked mark, whose chart says which sites' hand exclusions the menu offers to put back */
  openMenu: (e: { clientX: number; clientY: number; target?: EventTarget | null; preventDefault: () => void; stopPropagation?: () => void }, target: MenuTarget) => void
}

const Ctx = createContext<FilterEdits>({
  enabled: false, dropped: new Set(), kept: null, drop: () => {}, keep: () => {}, restore: () => {},
  group: null, openGroup: () => {}, openMenu: () => {},
})
export const useFilterEdits = () => useContext(Ctx)

/** a SPARTAN filter id after the replicate suffix is dropped (ETAD-0144-8 -> ETAD-0144) */
const SPARTAN_ID = /^[A-Z]{4}-\d{4}$/

/**
 * The hand edits as a test any chart can apply to the filters it draws, so an exclusion made
 * on one page holds on every page: a dropped filter is out everywhere; "keep only" narrows
 * SPARTAN filters (ids the gallery knows), never the IMPROVE pools or blanks a page also shows.
 * `note` words what a chart says about the filters it hid.
 */
export function useHandEdits() {
  const { dropped, kept } = useContext(Ctx)
  // a meeting page shown as presented (lib/presentation) is untouched by exclusions made elsewhere
  const presented = usePresentation() === 'presented'
  return useMemo(() => {
    if (presented) return { isOut: () => false, note: () => null, active: false }
    const isOut = (id: string | null | undefined) => {
      if (!id) return false
      const b = baseFilterId(id)
      return dropped.has(b) || dropped.has(id) || (!!kept && SPARTAN_ID.test(b) && !kept.has(b) && !kept.has(id))
    }
    const note = (n: number, { refit = false, saved = false }: { refit?: boolean; saved?: boolean } = {}) =>
      n <= 0 ? null : `${n} filter${n === 1 ? '' : 's'} excluded by hand ${n === 1 ? 'is' : 'are'} hidden here${refit ? ' and left out of the refit' : ''}${saved ? '; the saved statistics were computed with every filter' : ''}.`
    return { isOut, note, active: dropped.size > 0 || kept !== null }
  }, [dropped, kept, presented])
}

/** the attributes a mark carries; spread onto a <circle>, <path>, <rect> … */
export const fidProps = (id: string) => ({ 'data-fid': id })
export const fidsProps = (ids: string[], title: string) => ({ 'data-fids': ids.join(','), 'data-ftitle': title })

export function FilterEditsProvider({ children, enabled, dropped, kept, onDrop, onKeep, onRestore, group, onGroup }: {
  children: ReactNode
  enabled: boolean
  dropped: Set<string>
  kept: Set<string> | null
  onDrop: (ids: Set<string>) => void
  onKeep: (ids: Set<string>) => void
  onRestore: (ids: Set<string>) => void
  group: FilterGroup | null
  onGroup: (g: FilterGroup | null) => void
}) {
  const hl = useHighlightActions()
  const [menu, setMenu] = useState<{ x: number; y: number; target: MenuTarget | null; scope: Set<string>; action: MenuAction | null } | null>(null)
  const sites = useExportSites()

  const drop = useCallback((ids: Iterable<string>) => onDrop(new Set(ids)), [onDrop])
  const keep = useCallback((ids: Iterable<string>) => onKeep(new Set(ids)), [onKeep])
  const restore = useCallback((ids: Iterable<string>) => onRestore(new Set(ids)), [onRestore])
  const openMenu = useCallback<FilterEdits['openMenu']>((e, target) => {
    if (!enabled) return
    e.preventDefault()
    e.stopPropagation?.()
    const el = e.target instanceof Element ? e.target : null
    setMenu({ x: e.clientX, y: e.clientY, target, scope: panelSites(el, sites, target?.ids), action: el ? menuActionOf(el) : null })
  }, [enabled, sites])

  // marks that carry data-fid / data-fids, and the empty plot area (brush and undo only)
  useEffect(() => {
    if (!enabled) return
    const onContext = (e: MouseEvent) => {
      if (e.defaultPrevented || !(e.target instanceof Element)) return
      const t = e.target
      if (!t.closest('.panel') || t.closest('.frame-head, .frame-controls, input, textarea, select')) return
      const one = t.closest<Element>('[data-fid]')
      const many = t.closest<Element>('[data-fids]')
      let target: MenuTarget | null = null
      if (one) target = { ids: [one.getAttribute('data-fid')!], title: one.getAttribute('data-fid')! }
      else if (many) {
        const ids = (many.getAttribute('data-fids') ?? '').split(',').filter(Boolean)
        const title = many.getAttribute('data-ftitle') ?? `${ids.length} filters`
        if (ids.length) target = { ids, title, group: { title, ids } }
      }
      // on the empty plot only the brush and the undo (for this chart's sites) are on offer; elsewhere keep the browser's menu
      const scope = panelSites(t, sites, target?.ids)
      const undo = [...dropped].some((id) => scope.has(siteOf(id)))
      const action = menuActionOf(t)
      if (!target && !action && !(t.closest('svg') && (hl.selected || undo))) return
      e.preventDefault()
      setMenu({ x: e.clientX, y: e.clientY, target, scope, action })
    }
    document.addEventListener('contextmenu', onContext)
    return () => document.removeEventListener('contextmenu', onContext)
  }, [enabled, hl.selected, dropped, sites])

  const value = useMemo<FilterEdits>(() => ({ enabled, dropped, kept, drop, keep, restore, group, openGroup: onGroup, openMenu }),
    [enabled, dropped, kept, drop, keep, restore, group, onGroup, openMenu])
  return (
    <Ctx.Provider value={value}>
      {children}
      {menu && <FilterMenu {...menu} onClose={() => setMenu(null)} />}
    </Ctx.Provider>
  )
}

function FilterMenu({ x, y, target, scope, action, onClose }: { x: number; y: number; target: MenuTarget | null; scope: Set<string>; action: MenuAction | null; onClose: () => void }) {
  const ed = useFilterEdits()
  const hl = useHighlightActions()
  const ref = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState({ left: x, top: y })

  // stay on screen: flip left / up when the menu would run off the edge
  useEffect(() => {
    const r = ref.current?.getBoundingClientRect()
    if (!r) return
    setPos({ left: x + r.width > window.innerWidth - 8 ? Math.max(8, x - r.width) : x, top: y + r.height > window.innerHeight - 8 ? Math.max(8, y - r.height) : y })
    ref.current?.querySelector<HTMLButtonElement>('button')?.focus()
  }, [x, y])
  useEffect(() => {
    const close = (e: Event) => { if (!(e.target instanceof Node && ref.current?.contains(e.target))) onClose() }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault()
        const items = [...(ref.current?.querySelectorAll<HTMLButtonElement>('button') ?? [])]
        const i = items.indexOf(document.activeElement as HTMLButtonElement)
        items[(i + (e.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length]?.focus()
      }
    }
    document.addEventListener('mousedown', close, true)
    document.addEventListener('scroll', close, true)
    window.addEventListener('blur', onClose)
    window.addEventListener('keydown', onKey)
    // the tooltips hide while the menu is open (styles.css html.ctx-open)
    document.documentElement.classList.add('ctx-open')
    return () => {
      document.documentElement.classList.remove('ctx-open')
      document.removeEventListener('mousedown', close, true)
      document.removeEventListener('scroll', close, true)
      window.removeEventListener('blur', onClose)
      window.removeEventListener('keydown', onKey)
    }
  }, [onClose])

  const act = (f: () => void) => () => { f(); onClose() }
  const n = target?.ids.length ?? 0
  const single = n === 1 ? target!.ids[0] : null
  const brushed = hl.selected && !(target && n === hl.selected.size && target.ids.every((id) => hl.selected!.has(id))) ? hl.selected : null
  const alreadyOut = target ? target.ids.filter((id) => ed.dropped.has(id)) : []
  const plural = (k: number) => `${k} filter${k === 1 ? '' : 's'}`
  // the hand exclusions this chart could show again: from the sites it draws
  const here = [...ed.dropped].filter((id) => scope.has(siteOf(id)))

  return (
    <div ref={ref} className="ctx-menu" role="menu" style={pos} onContextMenu={(e) => e.preventDefault()}>
      {action && <button type="button" role="menuitem" onClick={act(() => action.el.dispatchEvent(new CustomEvent(MENU_ACTION, { bubbles: true, detail: action.detail })))}>{action.label}</button>}
      {action && target && <div className="ctx-sep" />}
      {target && <div className="ctx-head">{single ?? target.title}{!single && <span> · {plural(n)}</span>}</div>}
      {single && <button type="button" role="menuitem" onClick={act(() => hl.openSample(single))}>Open details</button>}
      {target?.group && <button type="button" role="menuitem" onClick={act(() => ed.openGroup(target.group!))}>List these {plural(n)} in the side panel</button>}
      {target && alreadyOut.length < n && <button type="button" role="menuitem" onClick={act(() => ed.drop(target.ids))}>Exclude {single ? 'this filter' : `these ${n}`}</button>}
      {target && alreadyOut.length > 0 && <button type="button" role="menuitem" onClick={act(() => ed.restore(alreadyOut))}>Put {single ? 'it' : `${alreadyOut.length}`} back</button>}
      {target && <button type="button" role="menuitem" onClick={act(() => ed.keep(target.ids))}>Keep only {single ? 'this filter' : `these ${n}`}</button>}
      {single && <button type="button" role="menuitem" onClick={act(() => { navigator.clipboard?.writeText(single) })}>Copy ID</button>}
      {brushed && <>
        {(target || action) && <div className="ctx-sep" />}
        <div className="ctx-head">Brushed · {plural(brushed.size)}</div>
        <button type="button" role="menuitem" onClick={act(() => { ed.drop(brushed); hl.setSelected(null) })}>Exclude the brushed filters</button>
        <button type="button" role="menuitem" onClick={act(() => { ed.keep(brushed); hl.setSelected(null) })}>Keep only the brushed filters</button>
      </>}
      {here.length > 0 && <>
        {(target || brushed || action) && <div className="ctx-sep" />}
        <button type="button" role="menuitem" onClick={act(() => ed.restore(here))}>
          {here.length === ed.dropped.size ? `Put back all ${plural(here.length)} excluded by hand` : `Put back the ${plural(here.length)} excluded by hand at these sites`}
        </button>
      </>}
    </div>
  )
}

/**
 * The hand edits for a page without the Filters panel (Baselines, Meeting, Similarity, Blanks):
 * the count, a review of them in the side panel, and a way to put them all back, so a filter
 * hidden by an exclusion made on another page is never hidden silently.
 */
export function HandEditsBadge() {
  const ed = useFilterEdits()
  if (!ed.dropped.size && !ed.kept) return null
  return (
    <span className="hand-edits-badge" title="Filters excluded (or kept) by hand on any page; every chart that draws them follows">
      {ed.dropped.size > 0 && <>
        <button type="button" className="pill-link" onClick={() => ed.openGroup({ title: 'Excluded by hand', ids: [...ed.dropped], live: 'dropped' })}>{ed.dropped.size} excluded by hand</button>
        <button type="button" className="btn quiet x-btn" onClick={() => ed.restore(ed.dropped)} title="Put every hand-excluded filter back" aria-label="Put every hand-excluded filter back"><UndoIcon /></button>
      </>}
      {ed.kept && <button type="button" className="pill-link" onClick={() => ed.openGroup({ title: 'Kept by hand (every other filter is hidden)', ids: [...ed.kept!], live: 'kept' })}>keeping {ed.kept.size}</button>}
    </span>
  )
}
