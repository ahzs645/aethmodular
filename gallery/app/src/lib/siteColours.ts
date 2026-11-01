/**
 * Site and season colours, looked up once. The colours themselves are not invented here: they come from
 * research/ftir_hips_chem/scripts/config.py through meta.json (meta.sites[].color, meta.seasons[].color),
 * so the gallery and the matplotlib figures agree. A name meta does not list falls back to INK.muted
 * unless the caller says otherwise.
 *
 * `colouring` is the gallery's usual rule for a chart that draws filters from one or several sites: by
 * season when one site is in view, by site when several. `colourGroups` splits rows the same two ways
 * and pairs each group with its colour (the wind roses and polar plot).
 */
import { INK } from './theme.ts'
import { seasonOnly } from './seasons.ts'
import type { FilterRow, MetaFile } from './types'

/** anything with a name and a colour: a meta site or season, a PMF group */
export interface Named { name: string; color: string }

/** the colour of the entry called `name` in `list`, or `fallback` */
export const namedColor = (list: readonly Named[] | null | undefined, name: unknown, fallback: string = INK.muted): string =>
  list?.find((s) => s.name === name)?.color ?? fallback

/** a site's colour by its name (meta.sites) */
export const siteColor = (meta: { sites: readonly Named[] }, name: unknown, fallback: string = INK.muted): string =>
  namedColor(meta.sites, name, fallback)

/** a site's colour by its code ('ETAD') */
export const siteCodeColor = (meta: { sites: readonly (Named & { code: string })[] }, code: unknown, fallback: string = INK.muted): string =>
  meta.sites.find((s) => s.code === code)?.color ?? fallback

/** a season's colour by its full name under the calendar in view (meta.seasons) */
export const seasonColor = (meta: { seasons: readonly Named[] }, name: unknown, fallback: string = INK.muted): string =>
  namedColor(meta.seasons, name, fallback)

export interface Colouring {
  /** 'season' with one site in view, else 'site' */
  by: 'season' | 'site'
  key: (r: FilterRow) => string
  label: (key: string) => string
  color: (key: string) => string
  /** the keys present in `rows`, in the calendar's or the site list's order */
  order: (rows: FilterRow[]) => string[]
}

/** colour filters by season when the rows come from one site, else by site */
export function colouring(rows: FilterRow[], meta: MetaFile): Colouring {
  const one = new Set(rows.map((r) => r.site)).size <= 1
  if (one) {
    const names = meta.seasons.map((s) => s.name)
    return {
      by: 'season', key: (r) => String(r.season ?? ''), label: (k) => seasonOnly(k) || 'No season',
      color: (k) => seasonColor(meta, k),
      order: (rs) => [...new Set(rs.map((r) => String(r.season ?? '')))].sort((a, b) => names.indexOf(a) - names.indexOf(b)),
    }
  }
  const names = meta.sites.map((s) => s.name)
  return {
    by: 'site', key: (r) => r.site, label: (k) => k,
    color: (k) => siteColor(meta, k),
    order: (rs) => [...new Set(rs.map((r) => r.site))].sort((a, b) => names.indexOf(a) - names.indexOf(b)),
  }
}

/**
 * Rows split by site or by season, each group as [name, colour, rows] in order of first appearance
 * (as d3.groups orders them). Season groups take the season as the rows carry it.
 */
export function colourGroups<R extends { site: string; season?: unknown }>(
  rows: readonly R[], by: 'site' | 'season', meta: { sites: readonly Named[]; seasons: readonly Named[] },
): [string, string, R[]][] {
  const groups = new Map<string, R[]>()
  for (const r of rows) {
    const k = (by === 'site' ? r.site : r.season) as string
    const g = groups.get(k)
    if (g) g.push(r)
    else groups.set(k, [r])
  }
  return [...groups].map(([k, rs]) => [k, by === 'site' ? siteColor(meta, k) : seasonColor(meta, k), rs])
}
