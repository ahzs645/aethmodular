export interface ProductCoverage {
  n_samples: number | null
  date_min: string | null
  date_max: string | null
  completeness_pct: number | null
  median_interval_h: number | null
  p95_interval_h: number | null
  max_gap_days: number | null
  newer_date_max: string | null
  inventory_n_samples?: number
  inventory_date_max?: string
}
export interface AvailabilitySite {
  code: string
  name: string
  blanks: number
  products: Record<string, ProductCoverage>
}
export interface StudyStream { days: string[]; reported_days: number; filters: number; analysis_last: string | null }
export interface StudySite {
  code: string
  name: string
  color: string
  ftir: StudyStream
  hips: StudyStream
  aeth: { days: string[]; days50: string[] | null; days80: string[] | null; record_days: string[]; mean_completeness: number | null; over100_days: number | null; file: string } | null
  paired_filters: { id: string; days: string[] }[]
  date_conflicts: number
}
export interface NotebookAvailability {
  schema_version: number
  generated: string
  network: {
    products: { key: string; label: string; unit: string }[]
    sites: AvailabilitySite[]
    monthly: { code: string; product: string; month: string; count: number }[]
    public_release: string
    hips_inventory_total: number
    hips_nonpositive: number
    future_inventory_endpoints: { code: string; product: string; advertised_date: string }[]
  }
  study: StudySite[]
  /** AERONET at each SPARTAN site with a station (export_availability.build_aeronet); absent from older exports */
  aeronet?: AeronetAvailability
  orientation: { network: string; quantity: string; units: string; n: number; n_sites: number; date_min: string; date_max: string }[]
  sources: { path: string; sha256: string; role: string }[]
}
export interface AeronetAvailabilitySite {
  code: string
  name: string
  station: string
  km: number
  /** stations within 50 km that were ranked */
  candidates: number
  filters: number
  /** filters with at least one direct-sun AOD retrieval inside their sampling window */
  filters_aod: number
  /** filters with at least one almucantar inversion inside their window */
  filters_inv: number
  filter_first: string | null
  filter_last: string | null
  /** days with a daily AOD, and with a daily inversion, at the station over the pulled span */
  aod_days: number
  inv_days: number
  daily_first: string | null
  daily_last: string | null
}
export interface AeronetAvailabilityMonth { code: string; month: string; filters: number; filters_aod: number; filters_inv: number; aod_days: number; inv_days: number }
export interface AeronetAvailability { level: string; source: string; sites: AeronetAvailabilitySite[]; monthly: AeronetAvailabilityMonth[] }
export type CompletenessRule = 'Any finite IR' | 'At least 50%' | 'At least 80%'

export function intersectDays(a: string[], b: string[]): string[] {
  const other = new Set(b)
  return [...new Set(a)].filter(day => other.has(day)).sort()
}
export function studyOverlap(site: StudySite, rule: CompletenessRule) {
  const aeth = !site.aeth ? null : rule === 'At least 50%' ? site.aeth.days50 : rule === 'At least 80%' ? site.aeth.days80 : site.aeth.days
  const both = intersectDays(site.ftir.days, site.hips.days)
  const triple = aeth === null ? null : intersectDays(both, aeth)
  const aethSet = new Set(aeth ?? [])
  return { aeth, both, triple, matched: site.paired_filters.length,
    matchedWithAeth: aeth === null ? null : site.paired_filters.filter(f => f.days.some(day => aethSet.has(day))).length }
}
/** Include every calendar month between endpoints, without a local-time shift. */
export function calendarMonths(first: string, last: string): string[] {
  if (!/^\d{4}-(0[1-9]|1[0-2])$/.test(first) || !/^\d{4}-(0[1-9]|1[0-2])$/.test(last) || first > last) return []
  const months: string[] = []
  const date = new Date(`${first}-01T00:00:00Z`)
  while (date.toISOString().slice(0, 7) <= last) {
    months.push(date.toISOString().slice(0, 7))
    date.setUTCMonth(date.getUTCMonth() + 1)
  }
  return months
}
/** Calendar days in a 'YYYY-MM' month (0 for anything else), for a monthly share of photometer days. */
export function daysInMonth(month: string): number {
  if (!/^\d{4}-(0[1-9]|1[0-2])$/.test(month)) return 0
  const [y, m] = month.split('-').map(Number)
  return new Date(Date.UTC(y, m, 0)).getUTCDate()
}
