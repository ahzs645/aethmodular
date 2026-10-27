/**
 * The hourly records behind the Overview's sub-daily charts (overview/hourly/<CODE>.json), as hours,
 * days and episodes under the site's season calendar and the page's span.
 */
import { useMemo } from 'react'
import type { FilterRow, MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { loadHourly, spanDates, useOverviewIndex, useSiteFiles, type OverviewIndex, type OverviewSite, type Span } from './data'
import { dayShapes, toEpisodes, toHours, type DayShape, type Episode, type Hour } from './hourly'

export interface HourlySeries {
  /** the site code (the hourly record belongs to the filter site; at Addis, MA350-0238) */
  code: string
  site: string
  info: OverviewSite
  hours: Hour[]
  days: DayShape[]
  episodes: Episode[]
  /** the episode threshold, µg/m³ (the record's 95th-percentile hour) */
  threshold: number
}

export function useHourlySeries(codes: readonly string[], meta: MetaFile, rows: readonly FilterRow[], span: Span) {
  const index = useOverviewIndex()
  const have = codes.filter((c) => index.data?.sites[c])
  const files = useSiteFiles(have, loadHourly)
  const series = useMemo<HourlySeries[]>(() => have.flatMap((code) => {
    const f = files[code], info = index.data?.sites[code]
    const site = meta.sites.find((s) => s.code === code)?.name
    if (!f || !info || !site) return []
    const seasons = seasonsForSite(meta, site)
    const [lo, hi] = spanDates(rows.filter((r) => r.site === site), span)
    const hours = toHours(f, seasons, lo, hi)
    return [{ code, site, info, hours, days: dayShapes(hours), episodes: toEpisodes(info.episodes, seasons, lo, hi), threshold: info.threshold / 1000 }]
  }), [have.join(','), files, index.data, meta, rows, span]) // eslint-disable-line react-hooks/exhaustive-deps
  return { index: index.data as OverviewIndex | null, error: index.error, series, loading: have.some((c) => !(c in files)) }
}
