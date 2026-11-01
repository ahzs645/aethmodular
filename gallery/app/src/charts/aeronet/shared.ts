/**
 * What the AERONET tab's charts share: how a filter is coloured (by season when one site is in view, by site
 * when several), and the record a crossplot point opens.
 */
import type { FilterRecord } from '@/lib/highlight'
import type { FilterRow } from '@/lib/types'

// the colouring rule is shared by every tab now (lib/siteColours); re-exported for the AERONET charts
export { colouring, type Colouring } from '@/lib/siteColours'

/** the record a point opens: the filter and the values the chart drew */
export const recordOf = (r: FilterRow, source: string, fields: [string, number | string | null | undefined][]): FilterRecord =>
  ({ id: r.id, source, site: r.site, date: r.date, fields })

export const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
