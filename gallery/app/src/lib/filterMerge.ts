import type { FilterRow } from './types'

/** Backfill missing measurements by physical filter ID while retaining the
 * existing values' precision, frozen predictions, and every exclusion flag. */
export function supplementFilter(primary: FilterRow, extra: FilterRow | undefined, fields: readonly string[]): FilterRow {
  if (!extra) return primary
  if (extra.id !== primary.id || extra.code !== primary.code) throw new Error('Cannot merge measurements from different physical filters')
  const additions: Record<string, number> = {}
  for (const field of fields) {
    const old = primary[field], value = extra[field]
    const missing = old == null || (typeof old === 'number' && !Number.isFinite(old))
    if (missing && typeof value === 'number' && Number.isFinite(value)) additions[field] = value
  }
  const excluded = primary.excluded || extra.excluded
  const reason = extra.excluded
    ? [...new Set([primary.exclusion_reason, extra.exclusion_reason].filter(Boolean))].join('; ') || null
    : primary.exclusion_reason
  const added = Object.keys(additions)
  if (!added.length && excluded === primary.excluded && reason === primary.exclusion_reason) return primary
  const filled = [...new Set([...String(primary.network_filled_fields ?? '').split('; ').filter(Boolean), ...added])]
  return { ...primary, ...additions, excluded, exclusion_reason: reason,
    ...(filled.length ? { network_filled_fields: filled.join('; ') } : {}) }
}

/** An explicitly chosen live public release replaces its public fields, including
 * missing values. Lab measurements, predictions and exclusion flags stay intact. */
export function replacePublicFilter(primary: FilterRow, extra: FilterRow, fields: readonly string[]): FilterRow {
  const merged = supplementFilter(primary, extra, fields)
  const values = Object.fromEntries(fields.map(f => [f, typeof extra[f] === 'number' && Number.isFinite(extra[f]) ? extra[f] : null]))
  const provenance = Object.fromEntries(Object.entries(extra).filter(([key]) => key.startsWith('public_')))
  return { ...merged, ...values, ...provenance, date: extra.date, year: extra.year, month: extra.month }
}
