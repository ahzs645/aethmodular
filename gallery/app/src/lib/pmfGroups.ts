import type { PmfFile } from '@/lib/types'

/**
 * The ETAD PMF solution as a filter axis, mirroring the calibration explorer's
 * research/ftir_ec_phase3/scripts/pmf_source_groups.py: the dominant factor of
 * each PMF day (the export has already normalised the GF fractions, which is
 * what makes the dominant source meaningful), or the marine vs combustion
 * contrast the group asked about. A filter is joined to a PMF day by exact
 * sampling date; one with no PMF day is "No PMF day", a selectable group,
 * never silently pooled with the rest.
 */
export const PMF_SCHEMES = ['Dominant source', 'Marine vs combustion'] as const
export type PmfScheme = (typeof PMF_SCHEMES)[number]
export const NO_PMF = 'No PMF day'

const CLASS_OF: Record<string, 'Marine' | 'Combustion'> = {
  sea_salt: 'Marine',
  polluted_marine: 'Marine',
  wood: 'Combustion',
  charcoal: 'Combustion',
  fossil_fuel: 'Combustion',
}
const CLASS_COLOR = { Marine: '#2c7fb8', Combustion: '#c2502e' }
const NO_PMF_COLOR = '#aab2bb'

export interface PmfGroup { name: string; color: string }

/** The groups a scheme offers, in the solution's own source order. */
export function pmfGroups(pmf: PmfFile | null, scheme: PmfScheme, withNoPmf: boolean): PmfGroup[] {
  if (!pmf) return []
  const groups: PmfGroup[] = scheme === 'Dominant source'
    ? pmf.sources.map((s) => ({ name: s.label, color: s.color }))
    : (['Marine', 'Combustion'] as const).map((c) => ({ name: c, color: CLASS_COLOR[c] }))
  return withNoPmf ? [...groups, { name: NO_PMF, color: NO_PMF_COLOR }] : groups
}

/** A PMF day's dominant-source key as the chosen scheme's label. */
export function pmfLabel(pmf: PmfFile | null, key: string | null | undefined, scheme: PmfScheme): string {
  if (!pmf || !key) return NO_PMF
  if (scheme === 'Marine vs combustion') return CLASS_OF[key] ?? NO_PMF
  return pmf.sources.find((s) => s.key === key)?.label ?? NO_PMF
}

/** sampling date → dominant-source key, for joining other filters onto PMF days. */
export function pmfKeyByDate(pmf: PmfFile | null): Map<string, string> {
  const out = new Map<string, string>()
  for (const r of pmf?.rows ?? []) if (r.dominant_source) out.set(r.date, r.dominant_source)
  return out
}

/**
 * PMF on the chart tabs: the rows of the site the PMF file covers (pmf.site; only ETAD so far,
 * but nothing here assumes it) get each source's share of that day as a measurement
 * ("Charcoal share (PMF)", 0–1) and the dominant source under the chosen scheme as a label
 * (PMF_SOURCE), so the shares work as axes, in ratios and for colour-by-value, and the label
 * for colour-by-source and the Filters panel's PMF section. A filter matches its PMF day by
 * filter id, else by sampling date; rows of other sites are untouched.
 */
export const PMF_GROUP = 'PMF sources'
export const PMF_SOURCE = 'pmf_source'
export const pmfShareField = (label: string) => `${label} share (PMF)`
export const isPmfField = (f: string) => / share \(PMF\)$/.test(f)
/** In a field list under the PMF sources heading, a share field is just its source: "Fossil Fuel Combustion". */
export const pickerName = (f: string) => f.replace(/ share \(PMF\)$/, '')

export function pmfShareFields(pmf: PmfFile | null): string[] {
  return (pmf?.sources ?? []).map((s) => pmfShareField(s.label))
}

export function withPmf<R extends { id: string; code?: string; date?: string }>(rows: R[], pmf: PmfFile | null, scheme: PmfScheme): R[] {
  if (!pmf?.rows.length) return rows
  const byId = new Map(pmf.rows.map((r) => [r.id, r]))
  const byDate = new Map(pmf.rows.map((r) => [r.date, r]))
  return rows.map((r) => {
    if (r.code !== pmf.site) return r
    const p = byId.get(r.id) ?? (r.date ? byDate.get(r.date.slice(0, 10)) : undefined)
    if (!p) return { ...r, [PMF_SOURCE]: NO_PMF }
    const shares = Object.fromEntries(pmf.sources.map((s) => [pmfShareField(s.label), p.fraction[s.label] ?? null]))
    return { ...r, ...shares, [PMF_SOURCE]: pmfLabel(pmf, p.dominant_source, scheme) }
  })
}
