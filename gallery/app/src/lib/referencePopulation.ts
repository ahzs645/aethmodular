/** Physical-filter reference populations. Each field uses its own finite coverage. */
export interface ReferenceRow {
  id: string; site: string; code: string; date: string; excluded: boolean
  [field: string]: unknown
}
export interface SpartanReference {
  schema_version: number; source: string; note: string; n: number; n_sites: number
  fields: string[]; field_units: Record<string, string>; field_sources: Record<string, string>
  rows: ReferenceRow[]
}
export const isFiniteNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
export const cleanReference = (rows: ReferenceRow[], site?: string) => rows.filter(r => !r.excluded && (!site || r.site === site))
export interface Placement { v: number; pct: number; min: number; q1: number; med: number; q3: number; max: number; n: number }
const quantile = (sorted: number[], p: number) => {
  const i = (sorted.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i)
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo)
}
export function placements(row: ReferenceRow, fields: string[], reference: ReferenceRow[]): Record<string, Placement> {
  const out: Record<string, Placement> = {}
  for (const field of fields) {
    const v = row[field]
    if (!isFiniteNumber(v)) continue
    const values = reference.map(r => r[field]).filter(isFiniteNumber).sort((a, b) => a - b)
    if (values.length < 4) continue
    out[field] = { v, pct: values.filter(x => x < v).length / values.length * 100,
      min: values[0], q1: quantile(values, .25), med: quantile(values, .5), q3: quantile(values, .75), max: values[values.length - 1], n: values.length }
  }
  return out
}
/** A filter outside another site's range stays visible at the edge. */
export function stripPosition(value: number, p: Pick<Placement, 'min' | 'max'>) {
  return p.min === p.max ? value < p.min ? 0 : value > p.max ? 100 : 50 : Math.max(0, Math.min(100, 100 * (value - p.min) / (p.max - p.min)))
}
export function ratioReference(reference: ReferenceRow[], numerator: string, denominator: string) {
  const values = reference.flatMap(r => isFiniteNumber(r[numerator]) && isFiniteNumber(r[denominator]) && r[denominator] !== 0
    ? [(r[numerator] as number) / (r[denominator] as number)] : []).filter(isFiniteNumber).sort((a, b) => a - b)
  return { n: values.length, median: values.length ? quantile(values, .5) : null }
}
