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
/** the reference filters at `site` that count; `counts` is the gallery's Filters → excluded rule (default: the export's exclusions out) */
export const cleanReference = (rows: ReferenceRow[], site?: string, counts: (r: ReferenceRow) => boolean = r => !r.excluded) =>
  rows.filter(r => counts(r) && (!site || r.site === site))
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
/**
 * A network too large to ship as rows (IMPROVE, ~388,000 filters; gallery/data/export_improve_reference.py):
 * per measurement, `steps + 1` quantiles of all its finite values, and an even sample of whole filters,
 * columnar, for the crossplot.
 */
export interface QuantileReference {
  schema_version: number; note: string; n: number; n_sites: number; date_min: string; date_max: string
  fields: string[]; field_units: Record<string, string>; steps: number
  quantiles: Record<string, { n: number; q: number[] }>
  site_names: Record<string, string>; sample_per_site: number
  sample: { code: string[]; date: string[]; poc: number[] } & Record<string, (number | null)[] | string[] | number[]>
}

/** The sample as rows (ids as the gallery's IMPROVE rows: CODE-date, then -POC for a collocated sampler). */
export function sampleRows(ref: QuantileReference): ReferenceRow[] {
  const { code, date, poc } = ref.sample
  return code.map((c, i) => ({ id: `${c}-${date[i]}${poc[i] !== 1 ? `-${poc[i]}` : ''}`, code: c, site: ref.site_names[c] ?? c, date: date[i], excluded: false,
    ...Object.fromEntries(ref.fields.map(f => [f, (ref.sample[f] as (number | null)[])[i]]).filter(([, v]) => isFiniteNumber(v))) }))
}

/**
 * `placements` from a quantile table: min, quartiles, median and max are its entries; the percentile (the
 * share of values strictly below) is read between the two quantiles around the value, so on a network's
 * thousands of values it is good to one step (0.1 % for 1,000 steps). A value tied with the lowest
 * quantile is at 0, like the row count.
 */
export function quantilePlacements(row: ReferenceRow, fields: string[], ref: Pick<QuantileReference, 'quantiles'>): Record<string, Placement> {
  const out: Record<string, Placement> = {}
  for (const field of fields) {
    const v = row[field], t = ref.quantiles[field]
    if (!isFiniteNumber(v) || !t || t.n < 4) continue
    const q = t.q, steps = q.length - 1
    let i = 0
    while (i <= steps && q[i] < v) i++
    const pct = i === 0 ? 0 : i > steps ? 100 : ((i - 1) + (q[i] > q[i - 1] ? (v - q[i - 1]) / (q[i] - q[i - 1]) : 0)) / steps * 100
    const at = (p: number) => q[Math.round(p * steps)]
    out[field] = { v, pct, min: q[0], q1: at(0.25), med: at(0.5), q3: at(0.75), max: q[steps], n: t.n }
  }
  return out
}

/** A filter outside another site's range stays visible at the edge. */
export function stripPosition(value: number, p: Pick<Placement, 'min' | 'max'>) {
  return p.min === p.max ? value < p.min ? 0 : value > p.max ? 100 : 50 : Math.max(0, Math.min(100, 100 * (value - p.min) / (p.max - p.min)))
}
/** A ratio's place in the reference, as `placements` gives a measurement's (null with < 4 reference values). */
export function ratioPlacement(value: number, reference: ReferenceRow[], numerator: string, denominator: string): Placement | null {
  const values = reference.flatMap(r => isFiniteNumber(r[numerator]) && isFiniteNumber(r[denominator]) && r[denominator] !== 0
    ? [(r[numerator] as number) / (r[denominator] as number)] : []).filter(isFiniteNumber).sort((a, b) => a - b)
  if (!isFiniteNumber(value) || values.length < 4) return null
  return { v: value, pct: values.filter(x => x < value).length / values.length * 100,
    min: values[0], q1: quantile(values, .25), med: quantile(values, .5), q3: quantile(values, .75), max: values[values.length - 1], n: values.length }
}
export function ratioReference(reference: ReferenceRow[], numerator: string, denominator: string) {
  const values = reference.flatMap(r => isFiniteNumber(r[numerator]) && isFiniteNumber(r[denominator]) && r[denominator] !== 0
    ? [(r[numerator] as number) / (r[denominator] as number)] : []).filter(isFiniteNumber).sort((a, b) => a - b)
  return { n: values.length, median: values.length ? quantile(values, .5) : null }
}

/** Match by physical filter ID, keeping known precision, joined fields and exclusions. */
export function mergeReferenceRows(exported: ReferenceRow[], known: ReferenceRow[]): ReferenceRow[] {
  const byId = new Map(known.map(r => [r.id, r]))
  return exported.map(extra => {
    const primary = byId.get(extra.id)
    if (!primary) return extra
    if (primary.code !== extra.code) throw new Error('Reference filter ID belongs to a different site')
    const additions = Object.fromEntries(Object.entries(extra).filter(([f, v]) =>
      (primary[f] == null || (typeof primary[f] === 'number' && !Number.isFinite(primary[f]))) && isFiniteNumber(v)))
    return { ...extra, ...primary, ...additions, excluded: primary.excluded || extra.excluded,
      exclusion_reason: [...new Set([primary.exclusion_reason, extra.exclusion_reason].filter(v => typeof v === 'string' && v))].join('; ') || null }
  })
}
