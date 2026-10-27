/**
 * What the AERONET tab's charts share: how a filter is coloured (by season when one site is in view, by site
 * when several), and the record a crossplot point opens.
 */
import { INK } from '@/lib/theme'
import { seasonOnly } from '@/lib/seasons'
import type { FilterRecord } from '@/lib/highlight'
import type { FilterRow, MetaFile } from '@/lib/types'

export interface Colouring {
  /** 'season' with one site in view, else 'site' */
  by: 'season' | 'site'
  key: (r: FilterRow) => string
  label: (key: string) => string
  color: (key: string) => string
  /** the keys present in `rows`, in the calendar's or the site list's order */
  order: (rows: FilterRow[]) => string[]
}

export function colouring(rows: FilterRow[], meta: MetaFile): Colouring {
  const one = new Set(rows.map((r) => r.site)).size <= 1
  if (one) {
    const names = meta.seasons.map((s) => s.name)
    return {
      by: 'season', key: (r) => String(r.season ?? ''), label: (k) => seasonOnly(k) || 'No season',
      color: (k) => meta.seasons.find((s) => s.name === k)?.color ?? INK.muted,
      order: (rs) => [...new Set(rs.map((r) => String(r.season ?? '')))].sort((a, b) => names.indexOf(a) - names.indexOf(b)),
    }
  }
  const names = meta.sites.map((s) => s.name)
  return {
    by: 'site', key: (r) => r.site, label: (k) => k,
    color: (k) => meta.sites.find((s) => s.name === k)?.color ?? INK.muted,
    order: (rs) => [...new Set(rs.map((r) => r.site))].sort((a, b) => names.indexOf(a) - names.indexOf(b)),
  }
}

/** the record a point opens: the filter and the values the chart drew */
export const recordOf = (r: FilterRow, source: string, fields: [string, number | string | null | undefined][]): FilterRecord =>
  ({ id: r.id, source, site: r.site, date: r.date, fields })

export const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
