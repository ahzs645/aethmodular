/**
 * The PMF solution's coverage and dominance, from filters.json and pmf.json (pure, for charts/pmf and the node test
 * pmfCoverage.test.mjs):
 *
 *  - which filters of the PMF site the PMF analysed (ETAD_Factor_Analysis c11–13: a filter "has PMF" when a PMF day
 *    lies within one day of it; here first by filter id, which every exported PMF row carries, then by that date rule);
 *  - how dominant the dominant factor is (dominant_source_threshold_analysis c12): the largest source's share of the
 *    day's total, on the normalised GF fractions or on the K_F contributions in µg/m³;
 *  - the biomass share (ETAD_Factor_Analysis c36–38): Wood Burning + Charcoal as a % of the explained sources.
 */

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

/** the largest source and its share of the sum of the sources (null without a positive total) */
export function dominantShare(values: Record<string, number | null | undefined>, labels: readonly string[]): { source: string; share: number } | null {
  let best: string | null = null, top = -Infinity, total = 0
  for (const l of labels) {
    const v = values[l]
    if (!finite(v)) continue
    total += v
    if (v > top) { top = v; best = l }
  }
  return best === null || !(total > 0) ? null : { source: best, share: top / total }
}

/** the PMF sources ETAD_Factor_Analysis counts as biomass burning */
export const BIOMASS_SOURCES = ['Wood Burning', 'Charcoal'] as const

/** biomass as a % of the explained sources (`(GF wood + GF charcoal) / sum(GF) × 100`); null without both or a total */
export function biomassPct(fraction: Record<string, number | null | undefined>, labels: readonly string[], biomass: readonly string[] = BIOMASS_SOURCES): number | null {
  let total = 0, part = 0, seen = 0
  for (const l of labels) {
    const v = fraction[l]
    if (!finite(v)) continue
    total += v
    if (biomass.includes(l)) { part += v; seen++ }
  }
  return seen === biomass.length && total > 0 ? (100 * part) / total : null
}

const DAY = 86_400_000
const dayOf = (d: string) => Date.parse(`${d.slice(0, 10)}T00:00:00Z`)

/**
 * Each filter's PMF day, if any: the PMF row with its id, else one dated within `toleranceDays` of it (the
 * notebook's `abs(filter_date - pmf_date) <= 1 day`). A PMF row is used once; `unmatched` are the PMF rows no filter took.
 */
export function matchPmf<F extends { id: string; date: string }, P extends { id: string; date: string }>(filters: readonly F[], pmf: readonly P[], toleranceDays = 1) {
  const byId = new Map(pmf.map((p) => [p.id, p]))
  const used = new Set<P>()
  const of = new Map<string, P>()
  for (const f of filters) {
    const p = byId.get(f.id)
    if (p && !used.has(p)) { of.set(f.id, p); used.add(p) }
  }
  // the rest by date, nearest first
  const left = pmf.filter((p) => !used.has(p)).map((p) => ({ p, t: dayOf(p.date) }))
  for (const f of filters) {
    if (of.has(f.id) || !left.length) continue
    const t = dayOf(f.date)
    let best: { p: P; t: number } | null = null
    for (const c of left) if (!used.has(c.p) && Math.abs(c.t - t) <= toleranceDays * DAY && (!best || Math.abs(c.t - t) < Math.abs(best.t - t))) best = c
    if (best) { of.set(f.id, best.p); used.add(best.p) }
  }
  return { of, unmatched: pmf.filter((p) => !used.has(p)) }
}

/** how many values sit at or below and above each threshold (the notebook's "Below/Above" box) */
export function belowAbove(values: readonly number[], thresholds: readonly number[]): { threshold: number; below: number; above: number }[] {
  const v = values.filter(Number.isFinite)
  return thresholds.map((t) => ({ threshold: t, below: v.filter((x) => x <= t).length, above: v.filter((x) => x > t).length }))
}

/** per field: the filters that measured it, and how many of those the PMF analysed (ETAD_Factor_Analysis c13) */
export function fieldCoverage<R extends Record<string, unknown>>(rows: readonly R[], fields: readonly string[], hasPmf: (r: R) => boolean) {
  return fields.map((field) => {
    const measured = rows.filter((r) => finite(r[field]))
    return { field, all: measured.length, pmf: measured.filter(hasPmf).length }
  })
}
