import { regression, type RegressionOptions, type RegressionStats } from './stats.ts'

export type FitGrouping = 'pooled' | 'site'
export interface RegressionGroup { name: string; color?: string; stats: RegressionStats; domain: [number, number] }

/** Pair first, then fit each population with its own aligned uncertainty and bootstrap vectors. */
export function groupedRegressions<T>(points: readonly T[], grouping: FitGrouping, access: {
  x: (p: T) => number | null | undefined
  y: (p: T) => number | null | undefined
  site: (p: T) => string
  color: (p: T) => string
  options: (members: T[]) => RegressionOptions
}): RegressionGroup[] {
  const valid = points.filter(p => typeof access.x(p) === 'number' && Number.isFinite(access.x(p)) && typeof access.y(p) === 'number' && Number.isFinite(access.y(p)))
  const populations = new Map<string, T[]>()
  for (const p of valid) {
    const name = grouping === 'site' ? access.site(p) || 'Unknown site' : 'All selected sites'
    const members = populations.get(name) ?? []
    members.push(p)
    populations.set(name, members)
  }
  return [...populations].flatMap(([name, members]) => {
    const xs = members.map(p => access.x(p)!), ys = members.map(p => access.y(p)!)
    const stats = regression(xs, ys, access.options(members))
    return stats ? [{ name, stats, color: grouping === 'site' ? access.color(members[0]) : undefined,
      domain: [Math.min(...xs), Math.max(...xs)] as [number, number] }] : []
  })
}
