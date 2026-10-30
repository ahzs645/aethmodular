import type { FilterRow, MetaFile } from './types.ts'

/** Shared measurement identity for tile information, contributing rows and snapshot manifests. */
export function metricDefinition(field: string, meta: MetaFile) {
  const method = /SSR estimate/.test(field) ? 'HIPS BC estimated from the SSR transfer curve (method 220)' : /ChemSpec.*FTIR|FTIR/.test(field) ? 'FTIR-derived' : /TOR/.test(field) ? 'Thermal-optical reflectance (TOR)'
    : /SSR/.test(field) ? 'Smoke-stain reflectometer (SSR)' : /HIPS/.test(field) ? 'HIPS optical absorption'
      : /ERA5/.test(field) ? 'ERA5 reanalysis' : /AERONET/.test(field) ? 'AERONET' : 'See source measurement metadata'
  return { field, unit: meta.field_units[field] ?? '', method, counting: 'Unique physical filter IDs',
    missing: 'Only finite numeric values contribute; paired charts require all selected fields.',
    caveat: /FTIR/.test(field) ? 'Public ChemSpec FTIR and in-house FTIR are the same product family, not independent validation.'
      : /HIPS|SSR/.test(field) ? 'Optical BC depends on the stated conversion; it is not thermal-optical EC.' : '' }
}

/** The population carrying every requested measurement, without inventing zeros for missing values. */
export function measuredRows(rows: FilterRow[], fields: string[], dated = false): FilterRow[] {
  const seen = new Set<string>()
  return rows.filter(r => {
    if (seen.has(r.id) || (dated && !r.date) || !fields.every(f => typeof r[f] === 'number' && Number.isFinite(r[f]))) return false
    seen.add(r.id)
    return true
  })
}

/** Normalize only changed branches, avoiding a stringify/replace/parse copy of large exports. */
export function normalizeSiteIdentity<T>(value: T): T {
  if (typeof value === 'string') return value.replace(/\bPasadena\b/g, 'JPL') as T
  if (!value || typeof value !== 'object') return value
  let changed = false
  const pairs = Object.entries(value).map(([key, v]) => {
    const name = key.replace(/\bPasadena\b/g, 'JPL'), next = normalizeSiteIdentity(v)
    changed ||= name !== key || next !== v
    return [name, next] as const
  })
  return changed ? (Array.isArray(value) ? pairs.map(([, v]) => v) : Object.fromEntries(pairs)) as T : value
}
