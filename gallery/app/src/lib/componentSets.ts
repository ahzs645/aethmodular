/**
 * Component sets: what the Stacked bars (charts/partOfWhole/CompositionTreemap) stack, and in which colours.
 * A set draws on one basis: SPARTAN's mass reconstruction (lib/spartanReconstruction, twelve components that
 * sum to PM2.5) or the measured species (FTIR carbon, ions, elements). Never both: SPARTAN's Dust already holds
 * the crustal elements and its Sodium is the sea-salt split, not the ion, so a mixed stack counts mass twice.
 *
 * The gallery ships ready-made sets (BUILT_IN_SETS); the reader makes their own in the editor
 * (components/ComponentSetPicker): which components, their stacking order, their colours. The reader's sets live
 * in the URL (`csets`, so a shared link carries them) and in this browser, as the groupings do (lib/groupings).
 */
import { RECON_FIELDS } from './spartanReconstruction'

export type Basis = 'recon' | 'measured'
export const BASIS_LABEL: Record<Basis, string> = { recon: 'SPARTAN reconstruction', measured: 'Measured species' }

export interface ComponentSetDef {
  /** stable key, also the chart's `<id>.parts` value */
  id: string
  name: string
  basis: Basis
  /** the components, foot of the stack first */
  fields: string[]
  /** a colour per component; a missing one takes the chart's palette */
  colors?: Record<string, string>
  /** the reader's name for a component (legend, labels, tooltips); a missing one keeps the component's own */
  labels?: Record<string, string>
  /** the legend's order: as stacked, foot first (the default), or top of the stack first, as the bars read */
  legend?: LegendOrder
}
export type LegendOrder = 'foot' | 'top'
export const LEGEND_ORDER_LABEL: Record<LegendOrder, string> = { foot: 'Foot of the stack first', top: 'Top of the stack first' }
/** a component's name in a set: the reader's, else its own */
export const labelIn = (set: ComponentSetDef | BuiltInSet, field: string) => ('labels' in set && set.labels?.[field]?.trim()) || field

/** a ready-made set: its components follow what the view has */
export interface BuiltInSet { id: string; name: string; basis: Basis; note: string; fields: (available: string[]) => string[] }

const listed = (names: string[]) => (av: string[]) => av.filter((f) => names.includes(f))
const CARBON_IONS = ['OC (FTIR)', 'EC (FTIR)', 'Sulfate', 'Nitrate', 'Ammonium', 'Potassium', 'Sodium', 'Calcium', 'Magnesium', 'Nitrite', 'Phosphate', 'Bromide']
export const BUILT_IN_SETS: BuiltInSet[] = [
  { id: 'figure', name: 'SPARTAN figure', basis: 'recon', note: "all twelve components of SPARTAN's network figure: they sum to PM2.5", fields: listed(RECON_FIELDS) },
  { id: 'nowater', name: 'Without water & residual', basis: 'recon', note: 'water and the residual are what is left of the mass once the rest is taken out',
    fields: listed(RECON_FIELDS.filter((f) => f !== 'Water' && f !== 'Residual')) },
  { id: 'inorganic', name: 'Secondary inorganic', basis: 'recon', note: 'sulfate, nitrate and ammonium', fields: listed(['Sulfate', 'Nitrate', 'Ammonium']) },
  { id: 'carbon', name: 'Carbon & residual', basis: 'recon', note: 'black and organic carbon, and the residual (mostly the non-carbon mass of the organics)', fields: listed(['Black Carbon', 'Org. Carbon', 'Residual']) },
  { id: 'crustal', name: 'Dust & sea salt', basis: 'recon', note: 'dust, trace element oxides, sea-salt sodium and chloride', fields: listed(['Dust', 'TEO', 'Sodium', 'Chloride']) },
  { id: 'all', name: 'Measured species', basis: 'measured', note: "the chart's earlier default: FTIR OC and EC, the ions, crustal elements and trace metals as measured (they do not close the mass)", fields: (av) => av },
  { id: 'carbonions', name: 'Measured carbon & ions', basis: 'measured', note: 'FTIR OC and EC with the measured ions', fields: listed(CARBON_IONS) },
]

/** a set as the chart draws it: its components that the view has, in its order */
export function resolveSet(set: ComponentSetDef | BuiltInSet, available: string[]): string[] {
  if (typeof set.fields === 'function') return set.fields(available)
  return set.fields.filter((f) => available.includes(f))
}

export function describeSet(set: ComponentSetDef | BuiltInSet, available?: string[]): string {
  const n = available ? resolveSet(set, available).length : Array.isArray(set.fields) ? set.fields.length : null
  const own = 'colors' in set ? [set.colors && Object.keys(set.colors).length ? 'own colours' : null, set.labels && Object.keys(set.labels).length ? 'renamed' : null] : []
  return [BASIS_LABEL[set.basis], n === null ? null : `${n} component${n === 1 ? '' : 's'}`, ...own].filter(Boolean).join(' · ')
}

export const newSetId = () => `c${Date.now().toString(36)}${Math.floor(Math.random() * 1296).toString(36)}`

// ---------------------------------------------------------------- the URL (`csets`) and this browser
const toB64 = (s: string) => btoa(String.fromCharCode(...new TextEncoder().encode(s))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
const fromB64 = (t: string) => new TextDecoder().decode(Uint8Array.from(atob(t.replace(/-/g, '+').replace(/_/g, '/')), (c) => c.charCodeAt(0)))
const HEX = /^#[\da-f]{6}$/i

export function setsParam(defs: ComponentSetDef[]): string {
  if (!defs.length) return ''
  // [id, name, basis, fields, colours, names, legend order]: the trailing ones only when set, so short sets stay short
  return toB64(JSON.stringify(defs.map((d) => {
    const row: unknown[] = [d.id, d.name, d.basis, d.fields, d.colors ?? {}, d.labels ?? {}, d.legend ?? 'foot']
    while (row.length > 4 && (row[row.length - 1] === 'foot' || (typeof row[row.length - 1] === 'object' && !Object.keys(row[row.length - 1] as object).length))) row.pop()
    return row
  })))
}

export function parseSets(param: string | null): ComponentSetDef[] {
  if (!param) return []
  try {
    const raw = JSON.parse(fromB64(param)) as unknown[]
    return raw.flatMap((x) => {
      if (!Array.isArray(x) || x.length < 4) return []
      const [id, name, basis, fields, colors, labels, legend] = x as unknown[]
      if (typeof id !== 'string' || (basis !== 'recon' && basis !== 'measured') || !Array.isArray(fields)) return []
      const own = colors && typeof colors === 'object'
        ? Object.fromEntries(Object.entries(colors as Record<string, unknown>).filter(([, c]) => typeof c === 'string' && HEX.test(c))) as Record<string, string>
        : undefined
      const names = labels && typeof labels === 'object'
        ? Object.fromEntries(Object.entries(labels as Record<string, unknown>).filter(([, l]) => typeof l === 'string' && l.trim()).map(([f, l]) => [f, (l as string).trim()]))
        : {}
      return [{ id, name: String(name ?? ''), basis, fields: fields.map(String), ...(own && Object.keys(own).length ? { colors: own } : {}),
        ...(Object.keys(names).length ? { labels: names } : {}), ...(legend === 'top' ? { legend: 'top' as const } : {}) }]
    })
  } catch {
    return []
  }
}

const SAVED_KEY = 'aeth-gallery:component-sets'
export function readSavedSets(): ComponentSetDef[] {
  try { return parseSets(localStorage.getItem(SAVED_KEY)) } catch { return [] }
}
export function writeSavedSets(defs: ComponentSetDef[]) {
  try {
    const v = setsParam(defs)
    if (v) localStorage.setItem(SAVED_KEY, v)
    else localStorage.removeItem(SAVED_KEY)
  } catch { /* private window or blocked storage: the link still carries them */ }
}
/** `first` wins on a shared id */
export function mergeSets(first: ComponentSetDef[], second: ComponentSetDef[]): ComponentSetDef[] {
  const ids = new Set(first.map((d) => d.id))
  return [...first, ...second.filter((d) => !ids.has(d.id))]
}
