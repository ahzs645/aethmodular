import { regression, type RegressionOptions, type RegressionStats } from './stats.ts'

/** one pooled fit, one per site, or one per season (the season of each filter under the active calendar) */
export type FitGrouping = 'pooled' | 'site' | 'season'
export interface RegressionGroup { name: string; color?: string; stats: RegressionStats; domain: [number, number] }

/** Pair first, then fit each population with its own aligned uncertainty and bootstrap vectors. */
export function groupedRegressions<T>(points: readonly T[], grouping: FitGrouping, access: {
  x: (p: T) => number | null | undefined
  y: (p: T) => number | null | undefined
  site: (p: T) => string
  color: (p: T) => string
  options: (members: T[]) => RegressionOptions
  /** for grouping 'season'; without it every point is 'Unknown season' */
  season?: (p: T) => string
  seasonColor?: (p: T) => string
}): RegressionGroup[] {
  const valid = points.filter(p => typeof access.x(p) === 'number' && Number.isFinite(access.x(p)) && typeof access.y(p) === 'number' && Number.isFinite(access.y(p)))
  const populations = new Map<string, T[]>()
  for (const p of valid) {
    const name = grouping === 'site' ? access.site(p) || 'Unknown site' : grouping === 'season' ? access.season?.(p) || 'Unknown season' : 'All selected sites'
    const members = populations.get(name) ?? []
    members.push(p)
    populations.set(name, members)
  }
  return [...populations].flatMap(([name, members]) => {
    const xs = members.map(p => access.x(p)!), ys = members.map(p => access.y(p)!)
    const stats = regression(xs, ys, access.options(members))
    return stats ? [{ name, stats, color: grouping === 'site' ? access.color(members[0]) : grouping === 'season' ? access.seasonColor?.(members[0]) : undefined,
      domain: [Math.min(...xs), Math.max(...xs)] as [number, number] }] : []
  })
}

/**
 * One fit per band of a grouping (lib/groupings applyGrouping): points in no band are left out, the
 * fits come in the bands' order and colours. Shared by the charts that offer "Fit per group" (hexbin,
 * the filter drawer's crossplot).
 */
export function bandRegressions<T>(points: readonly T[], band: { labels: string[]; labelOf: (p: T) => string | null; colorOf: (label: string) => string }, access: {
  x: (p: T) => number | null | undefined
  y: (p: T) => number | null | undefined
  options: (members: T[]) => RegressionOptions
}): RegressionGroup[] {
  return groupedRegressions(points.filter((p) => band.labelOf(p) !== null), 'site', {
    ...access, site: (p) => band.labelOf(p)!, color: (p) => band.colorOf(band.labelOf(p)!),
  }).sort((a, b) => band.labels.indexOf(a.name) - band.labels.indexOf(b.name))
}
