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
  orientation: { network: string; quantity: string; units: string; n: number; n_sites: number; date_min: string; date_max: string }[]
  sources: { path: string; sha256: string; role: string }[]
}
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
