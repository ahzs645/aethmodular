import { normalizeSeasonName } from './seasons.ts'

export interface ExportSeason { name: string; months: number[]; color: string; site?: string }
export interface PaletteSeason { key: string; label: string; color: string; group: string; groupLabel: string }

/** Consolidation requires the same name AND month membership; never infer equivalence
 * from a shared color or a short label such as “Winter”. */
export function paletteSeason(season: ExportSeason): PaletteSeason {
  const label = season.site && season.name.startsWith(`${season.site} · `) ? season.name.slice(season.site.length + 3) : season.name
  const months = [...new Set(season.months)].sort((a, b) => a - b)
  return { key: season.name, label: season.name, color: season.color,
    group: JSON.stringify([normalizeSeasonName(label), months]), groupLabel: label }
}
export function seasonColorRows(seasons: readonly PaletteSeason[], consolidate: boolean): PaletteSeason[] {
  if (!consolidate) return [...seasons]
  return [...new Map(seasons.map(s => [s.group, { ...s, key: s.group, label: s.groupLabel }])).values()]
}
export function seasonExportColor(season: PaletteSeason, prefs: { consolidateSeasons: boolean; seasons: Record<string, string> }, all: readonly PaletteSeason[]) {
  if (!prefs.consolidateSeasons) return prefs.seasons[season.key] ?? season.color
  const first = all.find(s => s.group === season.group) ?? season
  return prefs.seasons[season.group] ?? prefs.seasons[first.key] ?? first.color
}
