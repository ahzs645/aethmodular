import { useSyncExternalStore } from 'react'
import createWebShareEngine from '@firstform/json-url/web-share'
import { currentView, viewHref, type ViewState } from './views.ts'

/**
 * Custom sections: charts picked from any page as you go, kept on one page and shared as one link.
 *
 * An item is not a copy of a chart: it is the chart's key on its page and the whole view it was picked in
 * (lib/views ViewState: tab, sites, seasons, dates, fields, exclusions and the chart's options). The section page
 * draws each one in its own embed of that view (`view=single&chart=<key>`, components/SectionFrame), so charts from
 * different pages, under different filters, sit together exactly as they were picked, and every chart of the
 * gallery can be picked with no list of what can be drawn.
 *
 * Sections live in this browser (localStorage), and the address holds a section's id. A share link holds the
 * section itself as one json-url token (`#tab=custom&sec=<token>`, the codec of lib/views), so it needs no server;
 * a reader who opens one sees it read-only until they save their own copy.
 */
export interface SectionItem {
  /** this section's own id for the item: never shared, a link's items get new ones */
  id: string
  title: string
  /** the chart's frame key on its page (ChartFrame `data-frame-key`), `tile:<id>` for a Dashboard tile, or empty for a block of text (`title` its heading, `note` its body) */
  chart: string
  view: ViewState
  /** across both columns */
  wide?: boolean
  /** a set height (absent: Auto, which grows to fit the chart) */
  height?: ItemHeight
  note?: string
}
export type ItemHeight = 'S' | 'M' | 'L'
export const HEIGHTS: Record<ItemHeight, number> = { S: 400, M: 560, L: 760 }
export interface Section { id: string; name: string; about?: string; items: SectionItem[] }

/** how many charts a section holds. Only those near the screen are live (components/SectionFrame); this keeps a link and an export a sensible size */
export const MAX_ITEMS = 40
/** blocks of text between them do not count, within reason */
const MAX_BLOCKS = MAX_ITEMS * 2
const MAX_NAME = 120
const MAX_NOTE = 2000

const ID = 's-'
const newId = () => Math.random().toString(36).slice(2, 10)
export const newSection = (name = 'Untitled section'): Section => ({ id: `${ID}${newId()}`, name, items: [] })

/** a value of `sec` in the address: a section of this browser, else a shared section's token */
export const isSectionId = (sec: string) => sec.startsWith(ID)

// ---- the model: pure, each returns a new section

/** query keys that place a view on the screen, not what it shows: an item sets its own */
const VIEW_ONLY = new Set(['view', 'chart', 'side', 'focus', 'toc', 'embed', 'embedchart', 'v', 'dmax', 'dfocus', 'dedit'])
/** a view as an item keeps it */
export function itemView(view: ViewState): ViewState {
  return { h: { ...view.h }, q: Object.fromEntries(Object.entries(view.q).filter(([k]) => !VIEW_ONLY.has(k))) }
}

const canon = (o: Record<string, string>) => JSON.stringify(Object.entries(o).sort(([a], [b]) => a.localeCompare(b)))
const sameViewOf = (a: ViewState, b: ViewState) => canon(a.h) === canon(b.h) && canon(a.q) === canon(b.q)
export const sameItem = (a: Pick<SectionItem, 'chart' | 'view'>, b: Pick<SectionItem, 'chart' | 'view'>) => a.chart === b.chart && sameViewOf(a.view, b.view)

export const itemTab = (item: Pick<SectionItem, 'view'>) => item.view.h.tab ?? ''
export const isText = (item: Pick<SectionItem, 'chart'>) => !item.chart
export const tileKey = (id: string) => `tile:${id}`
const tileOf = (chart: string) => (chart.startsWith('tile:') ? chart.slice(5) : null)
export const chartCount = (s: Section) => s.items.filter((i) => !isText(i)).length

/** A new chart at the end. The same chart at other settings is a new item (two seasons, side by side); the same
 * chart at the same settings, or a full section, leaves it unchanged. */
export function addItem(s: Section, item: Omit<SectionItem, 'id'>): Section {
  if (chartCount(s) >= MAX_ITEMS || s.items.some((i) => sameItem(i, item))) return s
  return { ...s, items: [...s.items, { ...item, id: newId() }] }
}
/** A block of text, before the item `before` or at the end. */
export function addText(s: Section, before?: string): Section {
  const block: SectionItem = { id: newId(), title: '', chart: '', view: { h: {}, q: {} } }
  const at = s.items.findIndex((i) => i.id === before)
  return { ...s, items: at < 0 ? [...s.items, block] : [...s.items.slice(0, at), block, ...s.items.slice(at)] }
}
export const removeItem = (s: Section, id: string): Section => ({ ...s, items: s.items.filter((i) => i.id !== id) })
export const patchItem = (s: Section, id: string, patch: Partial<Omit<SectionItem, 'id'>>): Section =>
  ({ ...s, items: s.items.map((i) => (i.id === id ? { ...i, ...patch } : i)) })
/** one place earlier or later; at either end it stays */
export function moveItem(s: Section, id: string, by: -1 | 1): Section {
  const at = s.items.findIndex((i) => i.id === id)
  const to = at + by
  if (at < 0 || to < 0 || to >= s.items.length) return s
  const items = [...s.items]
  ;[items[at], items[to]] = [items[to], items[at]]
  return { ...s, items }
}

// ---- the link

interface WireItem { t: string; c?: string; h?: Record<string, string>; q?: Record<string, string>; w?: 1; z?: ItemHeight; m?: string }
interface Wire { id: string; n: string; a?: string; i: WireItem[] }

export function toWire(s: Section): Wire {
  return {
    id: s.id, n: s.name, ...(s.about ? { a: s.about } : {}),
    i: s.items.map((x) => (isText(x) ? { t: x.title, ...(x.note ? { m: x.note } : {}) } : {
      t: x.title, c: x.chart, h: x.view.h, ...(Object.keys(x.view.q).length ? { q: x.view.q } : {}),
      ...(x.wide ? { w: 1 as const } : {}), ...(x.height ? { z: x.height } : {}), ...(x.note ? { m: x.note } : {}),
    })),
  }
}

const strings = (o: unknown): Record<string, string> => o && typeof o === 'object' && !Array.isArray(o)
  ? Object.fromEntries(Object.entries(o).filter((e): e is [string, string] => typeof e[1] === 'string')) : {}
const text = (v: unknown, max: number) => (typeof v === 'string' ? v.slice(0, max) : '')

/** A token is whatever a link held, so nothing in it is trusted: every field is checked and cut to size. Throws on anything that is not a section. */
export function fromWire(wire: unknown): Section {
  const w = wire as Partial<Wire> | null
  if (!w || typeof w !== 'object' || !Array.isArray(w.i) || typeof w.n !== 'string') throw new Error('not a section')
  let charts = 0
  const items = w.i.slice(0, MAX_BLOCKS).flatMap((x): SectionItem[] => {
    if (!x) return []
    const note = text(x.m, MAX_NOTE)
    if (x.c === undefined) {
      const heading = text(x.t, MAX_NAME)
      return heading || note ? [{ id: newId(), title: heading, chart: '', view: { h: {}, q: {} }, ...(note ? { note } : {}) }] : []
    }
    if (typeof x.c !== 'string' || !x.c || ++charts > MAX_ITEMS) return []
    const view = itemView({ h: strings(x.h), q: strings(x.q) })
    const height = typeof x.z === 'string' && x.z in HEIGHTS ? x.z : undefined
    return [{ id: newId(), title: text(x.t, MAX_NAME) || x.c, chart: x.c, view, ...(x.w ? { wide: true } : {}), ...(height ? { height } : {}), ...(note ? { note } : {}) }]
  })
  const about = text(w.a, MAX_NOTE)
  return { id: typeof w.id === 'string' && isSectionId(w.id) ? w.id : newSection().id, name: text(w.n, MAX_NAME) || 'Shared section', ...(about ? { about } : {}), items }
}

/** whether two sections read the same, whatever their items' own ids */
export const sameSection = (a: Section, b: Section) => JSON.stringify(toWire(a)) === JSON.stringify(toWire(b))

const engine = createWebShareEngine<Wire>({ version: '1', codecs: ['lz', 'raw'], maxLength: 32000 })
export const encodeSection = async (s: Section) => (await engine.compressBest(toWire(s))).token as string
export const decodeSection = async (token: string) => fromWire(await engine.decompress(token))
/** Why a section has no link: the only way encoding fails is the token's length. */
export const tooLongText = 'This section is too long for a link. Hand-excluded filter lists take the most room: remove the charts that carry them, or clear them first.'

export async function sectionUrl(s: Section) {
  return `${window.location.origin}${window.location.pathname}#tab=custom&sec=${await encodeSection(s)}`
}

/** A chart's own view, for the section page's embed (`frame`: only the chart, with the section's card as its title) or to open it on its page. */
export function itemHref(item: Pick<SectionItem, 'chart' | 'view'>, frame: boolean) {
  const tile = tileOf(item.chart)
  // a Dashboard tile opens maximised, which is its page's own way of showing one chart
  const place: Record<string, string> = tile ? { dmax: tile } : { view: 'single', chart: item.chart, ...(frame ? { side: 'top' } : {}) }
  return viewHref({ h: item.view.h, q: { ...item.view.q, ...place, ...(frame ? { embed: '1', embedchart: '1' } : {}) } })
}

// ---- this browser's sections

const lastOf = <T,>(list: T[]): T | undefined => list[list.length - 1]
/** the section charts are being added to: the one chosen, else the newest */
export const activeOf = (sections: Section[], active: string | null) => sections.find((s) => s.id === active) ?? lastOf(sections)

const KEY = 'gallery.sections'
interface Store { sections: Section[]; active: string | null }
const EMPTY: Store = { sections: [], active: null }

function read(): Store {
  try {
    const s = JSON.parse(localStorage.getItem(KEY) ?? 'null') as Store | null
    return s && Array.isArray(s.sections) ? { sections: s.sections, active: s.active ?? null } : EMPTY
  } catch { return EMPTY }
}

let store: Store = typeof localStorage === 'undefined' ? EMPTY : read()
const listeners = new Set<() => void>()
const emit = () => listeners.forEach((l) => l())
function commit(next: Store) {
  store = next
  try { localStorage.setItem(KEY, JSON.stringify(next)) } catch { /* a private window: kept for this session */ }
  emit()
}
// a chart added in another window of the gallery shows here too
if (typeof window !== 'undefined') window.addEventListener('storage', (e) => { if (e.key === KEY) { store = read(); emit() } })

const subscribe = (l: () => void) => { listeners.add(l); return () => { listeners.delete(l) } }
export const useSections = () => useSyncExternalStore(subscribe, () => store)
export const sectionsNow = () => store

export const setActiveSection = (id: string) => { if (store.active !== id) commit({ ...store, active: id }) }
export function createSection(name?: string) {
  const s = newSection(name)
  commit({ sections: [...store.sections, s], active: s.id })
  return s
}
/** a shared section as this browser's own, under a new id, so it never overwrites the reader's copy or the sender's */
export function saveSection(s: Section) {
  const copy = { ...s, id: newSection().id }
  commit({ sections: [...store.sections, copy], active: copy.id })
  return copy
}
export function updateSection(id: string, change: (s: Section) => Section) {
  commit({ ...store, sections: store.sections.map((s) => (s.id === id ? change(s) : s)) })
}
export function deleteSection(id: string) {
  const sections = store.sections.filter((s) => s.id !== id)
  commit({ sections, active: store.active === id ? lastOf(sections)?.id ?? null : store.active })
}
/** put a section back as it was (the page's Undo) */
export const restoreSections = (s: Store) => commit(s)

/** The section charts are added to: the active one, or a new one the first time. */
function target() {
  return activeOf(store.sections, store.active) ?? createSection()
}

/** the chart as it is on the page now */
export const pickedItem = (chart: string, title: string) => ({ chart, title, view: itemView(currentView()) })

/** A chart's + button: on while the active section holds it at the settings on screen, and one press adds or takes it out. */
export function toggleItem(chart: string, title: string) {
  const s = target()
  const item = pickedItem(chart, title)
  const hit = s.items.find((i) => sameItem(i, item))
  setActiveSection(s.id)
  updateSection(s.id, (x) => (hit ? removeItem(x, hit.id) : addItem(x, item)))
}

/** What the Views menu shares from a section's page: a link to a section of this browser would open nothing elsewhere. */
export async function shareableView(view: ViewState): Promise<ViewState> {
  const sec = view.h.sec
  const s = sec && isSectionId(sec) ? store.sections.find((x) => x.id === sec) : undefined
  return s ? { ...view, h: { ...view.h, sec: await encodeSection(s) } } : view
}
