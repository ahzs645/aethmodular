import { useMemo } from 'react'
import { useSearchParam } from './url.ts'

/**
 * Which panels a chart shows, and in what order, kept in the link: one definition for every chart that lets
 * the reader choose its panels (the weather timeline's lanes, the HIPS transfer crossplots, the aethalometer
 * agreement panels), instead of an ad hoc list per chart.
 *
 *   const pick = usePanelPick('wt', ['temp', 'prcp', 'rh', 'blh', 'field'], ['temp', 'prcp', 'rh', 'blh', 'field'])
 *   pick.ids                       → the panels to draw, top to bottom (or left to right)
 *   pick.hide('rh') · pick.add('ws') · pick.move('field', -1) · pick.replace('rh', 'ws') · pick.reset()
 *   <PanelPicker pick={pick} labelOf={…} />   (components/PanelPicker: chips with ×, drag or arrows to reorder)
 *
 * Stored as `<chartId>.panels=a|b|c` (url.ts useSearchParam, so under a Dashboard tile's ParamScope prefix in a
 * tile, like lib/chartOptions' `<chartId>.<key>`); the default list is never written, and an empty choice is
 * `-` so "no panels" survives a reload. Ids the chart cannot draw (a field missing from this export) are
 * dropped at render, not from the link, so they come back when the data does. While `all` is empty (options
 * still loading) a link's ids are kept as they are. `legacy` reads an older key or value format, so links made
 * before a chart moved onto this still open as they were; the next change writes the new key and clears the old.
 *
 * Panels/FacetGrid take the result as `show={pick.ids}` / `order={pick.ids}`.
 */

/** the stored value of an empty choice (an absent key means the defaults) */
export const NO_PANELS = '-'

/** a link's ids, in order, unique, those this chart can draw (all of them while `all` is empty) */
export function parsePanels(raw: string | null | undefined, all: readonly string[], defaults: readonly string[]): string[] {
  const known = (k: string) => all.length === 0 || all.includes(k)
  if (raw == null || raw === '') return unique(defaults).filter(known)
  if (raw === NO_PANELS) return []
  return unique(raw.split('|').filter(Boolean)).filter(known)
}

/** the stored value for a choice: '' (not written) for the defaults, `-` for none */
export function formatPanels(ids: readonly string[], defaults: readonly string[]): string {
  const u = unique(ids)
  if (sameList(u, unique(defaults))) return ''
  return u.length ? u.join('|') : NO_PANELS
}

/** `id` added at the end (or at `at`), once */
export function addPanel(ids: readonly string[], id: string, at = ids.length): string[] {
  if (ids.includes(id)) return [...ids]
  const out = [...ids]
  out.splice(Math.max(0, Math.min(at, out.length)), 0, id)
  return out
}

/** `id` shown in its place in `all` (a toggle among a fixed set keeps the set's order) */
export function showPanel(ids: readonly string[], id: string, all: readonly string[]): string[] {
  if (ids.includes(id)) return [...ids]
  const rank = all.indexOf(id)
  if (rank < 0) return [...ids, id]
  const at = ids.findIndex((k) => { const r = all.indexOf(k); return r < 0 || r > rank })
  return addPanel(ids, id, at < 0 ? ids.length : at)
}

export const hidePanel = (ids: readonly string[], id: string): string[] => ids.filter((k) => k !== id)

/** `id` moved `by` places (−1 up or left), clamped to the ends */
export function movePanel(ids: readonly string[], id: string, by: number): string[] {
  const i = ids.indexOf(id)
  if (i < 0 || !by) return [...ids]
  const j = Math.max(0, Math.min(ids.length - 1, i + by))
  const out = [...ids]
  out.splice(i, 1)
  out.splice(j, 0, id)
  return out
}

/** `id` moved to stand where `target` stands (a drag onto another chip) */
export function movePanelTo(ids: readonly string[], id: string, target: string): string[] {
  const j = ids.indexOf(target)
  return j < 0 ? [...ids] : movePanel(ids, id, j - ids.indexOf(id))
}

/** `from` swapped for `to`; if `to` is already shown the two trade places (an axis title's picker) */
export function replacePanel(ids: readonly string[], from: string, to: string): string[] {
  if (from === to) return [...ids]
  return ids.map((k) => (k === from ? to : k === to ? from : k))
}

export interface PanelPickLegacy {
  /** the older key under the chart's scope (default: the current key, read in an older value format) */
  key?: string
  /** the ids an older value stood for; null: not one of its values */
  map: (raw: string) => readonly string[] | null
}

/**
 * Resolves what to read: the current key's value when it names any id, else the first legacy reading that
 * applies, else the defaults. Pure, for the tests; `raws` holds each legacy key's value (by index).
 */
export function resolvePanels(raw: string, legacy: readonly PanelPickLegacy[], legacyRaws: readonly string[], all: readonly string[], defaults: readonly string[]): string[] {
  if (raw === NO_PANELS) return []
  if (raw) {
    const ids = parsePanels(raw, all, defaults)
    if (ids.length) return ids
    for (const l of legacy) if (!l.key) { const m = l.map(raw); if (m) return parsePanels(m.join('|') || NO_PANELS, all, defaults) }
    return ids
  }
  for (let i = 0; i < legacy.length; i++) {
    const v = legacyRaws[i]
    if (!legacy[i].key || !v) continue
    const m = legacy[i].map(v)
    if (m) return parsePanels(m.join('|') || NO_PANELS, all, defaults)
  }
  return parsePanels(null, all, defaults)
}

export interface PanelPick {
  /** the panels to draw, in order */
  ids: string[]
  /** every panel the chart can draw, in its own order */
  all: readonly string[]
  defaults: readonly string[]
  isDefault: boolean
  has: (id: string) => boolean
  set: (ids: readonly string[]) => void
  /** shown in its place among `all` */
  show: (id: string) => void
  hide: (id: string) => void
  toggle: (id: string) => void
  /** appended (or inserted at `at`) */
  add: (id: string, at?: number) => void
  move: (id: string, by: number) => void
  moveTo: (id: string, target: string) => void
  replace: (from: string, to: string) => void
  reset: () => void
}

/**
 * The reader's choice and order of a chart's panels, in the link (see the top of this file). `all` and
 * `defaults` need not be memoised; the returned `ids` keeps its identity until the choice changes.
 */
export function usePanelPick(chartId: string, all: readonly string[], defaults: readonly string[], { key = 'panels', legacy = [] }: {
  /** the key under the chart's id (default 'panels': `<chartId>.panels`) */
  key?: string
  legacy?: readonly PanelPickLegacy[]
} = {}): PanelPick {
  const [raw, setRaw] = useSearchParam<string>(`${chartId}.${key}`, '')
  // one hook per legacy key, a fixed list per chart
  const old = legacy.map((l) => useSearchParam<string>(l.key ?? `${chartId}.${key}`, '')) // eslint-disable-line react-hooks/rules-of-hooks
  const oldRaws = old.map((o) => o[0])
  const allSig = all.join('|'), defSig = defaults.join('|'), oldSig = oldRaws.join('\u0000')
  const ids = useMemo(() => resolvePanels(raw, legacy, oldRaws, all, defaults), [raw, oldSig, allSig, defSig]) // eslint-disable-line react-hooks/exhaustive-deps
  const set = (next: readonly string[]) => {
    // the older keys go when the reader first changes the choice: the link then says it one way
    legacy.forEach((l, i) => { if (l.key && oldRaws[i]) old[i][1]('') })
    setRaw(formatPanels(next, defaults))
  }
  return {
    ids, all, defaults,
    isDefault: sameList(ids, parsePanels(null, all, defaults)),
    has: (id) => ids.includes(id),
    set,
    show: (id) => set(showPanel(ids, id, all)),
    hide: (id) => set(hidePanel(ids, id)),
    toggle: (id) => set(ids.includes(id) ? hidePanel(ids, id) : showPanel(ids, id, all)),
    add: (id, at) => set(addPanel(ids, id, at)),
    move: (id, by) => set(movePanel(ids, id, by)),
    moveTo: (id, target) => set(movePanelTo(ids, id, target)),
    replace: (from, to) => set(replacePanel(ids, from, to)),
    reset: () => set(defaults),
  }
}

function unique(ids: readonly string[]): string[] { return [...new Set(ids)] }
function sameList(a: readonly string[], b: readonly string[]) { return a.length === b.length && a.every((k, i) => k === b[i]) }
