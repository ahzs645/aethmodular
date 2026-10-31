import { useSyncExternalStore } from 'react'
import { interpolateLab, piecewise, quantize, color } from 'd3'
import { isBuiltInPalette, isCustomPalette, setCustomPalettes, type Palette, type PaletteKind } from './palettes.ts'

/**
 * The reader's own palettes, made in the colour picker (components/ColourSchemePicker "+ New palette")
 * and kept in this browser's localStorage, one list per kind. Each is saved as the stops the reader
 * picked; lib/palettes gets them resampled to a built-in palette's length (8 sequential, 7 diverging,
 * so a diverging palette's middle stop stays its neutral), or as given for a categorical set. Every
 * colour picker of that kind lists them after the built-in palettes. A link that names one opens on
 * the chart's default in a browser that does not have it.
 */
const KEY = 'aeth-gallery:palettes'
const KINDS: readonly PaletteKind[] = ['sequential', 'diverging', 'categorical']
/** the most stops a reader can pick (a categorical set beyond this is no longer tellable apart) */
export const MAX_STOPS = 12
export const MIN_STOPS = 2

interface Saved { name: string; stops: string[] }
type Store = Record<PaletteKind, Saved[]>

const HEX = /^#[0-9a-f]{6}$/i
/** a colour as #rrggbb, or null for anything a colour input cannot show */
export function toHex(c: string): string | null {
  const t = c.trim()
  if (HEX.test(t)) return t.toLowerCase()
  if (/^#?[0-9a-f]{3}$/i.test(t) || /^[0-9a-f]{6}$/i.test(t)) { const h = t.replace('#', ''); return `#${(h.length === 3 ? [...h].map((x) => x + x).join('') : h).toLowerCase()}` }
  const parsed = color(t)
  return parsed ? parsed.formatHex() : null
}

/** the stops as the chart's colours: a ramp resampled to the built-in length, a categorical set as given */
export function paletteFromStops(kind: PaletteKind, stops: readonly string[]): string[] {
  if (kind === 'categorical' || stops.length < 2) return [...stops]
  return quantize(piecewise(interpolateLab, [...stops]), kind === 'sequential' ? 8 : 7).map((c) => color(c)!.formatHex())
}

const empty = (): Store => ({ sequential: [], diverging: [], categorical: [] })
function read(): Store {
  const out = empty()
  try {
    const raw = typeof localStorage === 'undefined' ? null : localStorage.getItem(KEY)
    const parsed = raw ? JSON.parse(raw) as Partial<Store> : {}
    for (const kind of KINDS) for (const p of parsed[kind] ?? []) {
      const stops = Array.isArray(p?.stops) ? p.stops.map((c) => toHex(String(c))).filter((c): c is string => !!c) : []
      if (typeof p?.name === 'string' && p.name.trim() && stops.length >= MIN_STOPS && !isBuiltInPalette(kind, p.name)) out[kind].push({ name: p.name.trim(), stops })
    }
  } catch { /* a corrupt entry: start empty rather than break every chart */ }
  return out
}

let store: Store = empty()
let version = 0
const listeners = new Set<() => void>()
function apply(next: Store, persist: boolean) {
  store = next
  for (const kind of KINDS) setCustomPalettes(kind, store[kind].map((p): Palette => ({
    name: p.name, colours: paletteFromStops(kind, p.stops), source: 'Your palette, saved in this browser', note: 'yours, saved in this browser',
  })))
  if (persist) try { localStorage.setItem(KEY, JSON.stringify(store)) } catch { /* private mode or full: kept for this visit */ }
  version += 1
  for (const l of listeners) l()
}
apply(read(), false)
// another tab saved or deleted one
if (typeof window !== 'undefined') window.addEventListener('storage', (e) => { if (e.key === KEY) apply(read(), false) })

/** re-renders on any change to the reader's palettes (returns a counter that changes with them) */
export const useCustomPalettes = () => useSyncExternalStore((l) => { listeners.add(l); return () => { listeners.delete(l) } }, () => version)

/** the stops the reader picked for one of their palettes, for editing it */
export const customStops = (kind: PaletteKind, name: string) => store[kind].find((p) => p.name === name)?.stops

/** why a name cannot be used for a palette of this kind, or null when it can (`editing`: the palette being renamed) */
export function nameProblem(kind: PaletteKind, name: string, editing?: string): string | null {
  const n = name.trim()
  if (!n) return 'Give the palette a name'
  if (isBuiltInPalette(kind, n)) return 'A built-in palette has that name'
  if (n !== editing && isCustomPalette(kind, n)) return 'You already have a palette with that name'
  return null
}

/** a free name: "My palette", then "My palette 2" … */
export function freeName(kind: PaletteKind, base = 'My palette') {
  for (let i = 1; ; i++) { const n = i === 1 ? base : `${base} ${i}`; if (!nameProblem(kind, n)) return n }
}

/** add a palette, or replace `editing` (renamed if the name changed) in its place */
export function saveCustomPalette(kind: PaletteKind, name: string, stops: readonly string[], editing?: string) {
  const entry = { name: name.trim(), stops: [...stops] }
  const list = [...store[kind]]
  const at = editing ? list.findIndex((p) => p.name === editing) : -1
  if (at >= 0) list[at] = entry
  else list.push(entry)
  apply({ ...store, [kind]: list }, true)
}

export function deleteCustomPalette(kind: PaletteKind, name: string) {
  apply({ ...store, [kind]: store[kind].filter((p) => p.name !== name) }, true)
}
