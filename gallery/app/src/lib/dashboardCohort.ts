import type { FilterRow } from './types.ts'

export interface CohortFilter { site?: string; season?: string; from?: string; to?: string; ids?: string[]; label?: string }
export const EMPTY_COHORT: CohortFilter = {}
const date = (v: unknown): v is string => typeof v === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(v)
/** URL input is untrusted. Unsupported or malformed options never become executable predicates. */
export function parseCohort(text: string): CohortFilter {
  try {
    const v = JSON.parse(text)
    if (!v || typeof v !== 'object' || Array.isArray(v)) return {}
    return { ...(typeof v.site === 'string' && v.site ? { site: v.site } : {}),
      ...(typeof v.season === 'string' && v.season ? { season: v.season } : {}),
      ...(date(v.from) ? { from: v.from } : {}), ...(date(v.to) ? { to: v.to } : {}),
      ...(Array.isArray(v.ids) && v.ids.every((id: unknown) => typeof id === 'string') ? { ids: [...new Set<string>(v.ids)] } : {}),
      ...(typeof v.label === 'string' ? { label: v.label.slice(0, 300) } : {}) }
  } catch { return {} }
}

/** Reference IDs retain membership while current data values, units and methods remain explicit. */
export function selectCohort(pageRows: FilterRow[], sourceRows: FilterRow[], filter: CohortFilter) {
  const ids = filter.ids ? new Set(filter.ids) : null
  return (ids ? sourceRows : pageRows).filter(r => (!ids || ids.has(r.id)) && (!filter.site || r.site === filter.site)
    && (!filter.season || r.season === filter.season) && (!filter.from || !!r.date && r.date.slice(0, 10) >= filter.from)
    && (!filter.to || !!r.date && r.date.slice(0, 10) <= filter.to))
}

/** Spreadsheet-safe CSV: preserve missing values and quote delimiters, newlines and formulas. */
export function rowsCsv(rows: FilterRow[], fields: string[]): string {
  const columns = [...new Set(['id', 'site', 'date', 'season', ...fields, 'excluded', 'exclusion_reason'])]
  const cell = (v: unknown) => {
    let s = v == null ? '' : String(v)
    if (/^[=+\-@\t\r]/.test(s) && typeof v !== 'number') s = "'" + s
    return '"' + s.replace(/"/g, '""') + '"'
  }
  return [columns.map(cell).join(','), ...rows.map(r => columns.map(c => cell(r[c])).join(','))].join('\r\n')
}
