/**
 * The filters behind Research › Moisture and MAC, as pure functions (node-tested in moistureMac.test.mjs): each
 * filter's effective MAC (HIPS Fabs / FTIR EC, EC > 0) with its moisture, the notebook's percentile trim, and which
 * moisture field to open on for the sites in view.
 */
import { quantile } from '../../../lib/stats.ts'
import { num } from '../../../lib/table.ts'

export const EC = 'EC (FTIR)', FABS = 'HIPS Fabs'
export const MIN_FILTERS = 10

export interface MacRow { site: string }
export interface MacPoint<R> { r: R; ec: number; fabs: number; m: number; mac: number }

export const macPoints = <R extends MacRow>(rows: readonly R[], moisture: string): MacPoint<R>[] => rows.flatMap((r) => {
  const ec = num((r as Record<string, unknown>)[EC]), fabs = num((r as Record<string, unknown>)[FABS]), m = num((r as Record<string, unknown>)[moisture])
  return ec !== null && fabs !== null && m !== null && ec > 0 ? [{ r, ec, fabs, m, mac: fabs / ec }] : []
})

/** the notebook trims effective MAC to its 1st–99th percentile; per site, so a site reads the same alone or among others */
export function trimBySite<R extends MacRow>(points: readonly MacPoint<R>[]): MacPoint<R>[] {
  const bySite = new Map<string, MacPoint<R>[]>()
  for (const p of points) bySite.set(p.r.site, [...(bySite.get(p.r.site) ?? []), p])
  const bounds = new Map([...bySite].map(([s, ps]) => [s, [quantile(ps.map((p) => p.mac), 0.01), quantile(ps.map((p) => p.mac), 0.99)]]))
  return points.filter((p) => { const [lo, hi] = bounds.get(p.r.site)!; return p.mac >= lo && p.mac <= hi })
}

const countBySite = <R extends MacRow>(rows: readonly R[], moisture: string) => {
  const n = new Map<string, number>()
  for (const p of macPoints(rows, moisture)) n.set(p.r.site, (n.get(p.r.site) ?? 0) + 1)
  return n
}

/**
 * The first candidate that leaves every site its filters: at least MIN_FILTERS, or all of them where a site has
 * fewer filters with Fabs and EC. So AERONET stays the default where the photometer covers the filters and ERA5 takes
 * over where it does not; failing that, the first candidate with any filters.
 */
export function defaultMoisture<R extends MacRow>(rows: readonly R[], candidates: readonly string[]): string | null {
  const pairs = new Map<string, number>()
  for (const r of rows) {
    const rec = r as Record<string, unknown>, ec = num(rec[EC])
    if (ec !== null && ec > 0 && num(rec[FABS]) !== null) pairs.set(r.site, (pairs.get(r.site) ?? 0) + 1)
  }
  const leavesEverySite = (f: string) => { const n = countBySite(rows, f); return [...pairs].every(([s, p]) => (n.get(s) ?? 0) >= Math.min(MIN_FILTERS, p)) }
  return candidates.find(leavesEverySite) ?? candidates.find((f) => macPoints(rows, f).length > 0) ?? null
}
