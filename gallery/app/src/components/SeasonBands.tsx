import type { ScaleTime } from 'd3'
import type { SeasonMeta } from '@/lib/types'

/**
 * Season shading behind month and date axes, one look on every chart
 * (docs/gallery-chart-standards.md §1): full-height bands, or a thin strip
 * above the plot. A hovered season (from the legend) deepens its own bands and
 * fades the others.
 */
export const SEASON_SHADE = { base: 0.1, hovered: 0.24, other: 0.03, strip: 0.55 } as const

/** Background opacity for one season's shading, given the season the legend is hovering. */
export const seasonShade = (name: string, hover: string | null | undefined) =>
  hover == null ? SEASON_SHADE.base : hover === name ? SEASON_SHADE.hovered : SEASON_SHADE.other

export interface SeasonBand { x: number; w: number; color: string; name: string }

/** Month-by-month bands along a time axis, under one calendar, clipped to [0, width]. */
export function timeSeasonBands(x: ScaleTime<number, number>, seasons: readonly Pick<SeasonMeta, 'name' | 'months' | 'color' | 'from' | 'spans'>[], width: number): SeasonBand[] {
  return mergeBands(seasonBandsRaw(x, seasons, width))
}

/** adjacent bands of one season (a month-by-month calendar draws one per month) as one band */
export function mergeBands(bands: SeasonBand[]): SeasonBand[] {
  const out: SeasonBand[] = []
  for (const b of [...bands].sort((p, q) => p.x - q.x)) {
    const last = out[out.length - 1]
    if (last && last.name === b.name && Math.abs(last.x + last.w - b.x) < 0.5) last.w = b.x + b.w - last.x
    else out.push({ ...b })
  }
  return out
}

function seasonBandsRaw(x: ScaleTime<number, number>, seasons: readonly Pick<SeasonMeta, 'name' | 'months' | 'color' | 'from' | 'spans'>[], width: number): SeasonBand[] {
  const [d0, d1] = x.domain() as [Date, Date]
  // a data-computed calendar (lib/rainSeasons, lib/dynamicSeasons): its exact date ranges, and nothing where
  // it has none (weather still loading, or outside the record), not the months' usual seasons
  if (seasons.some((s) => Array.isArray(s.spans))) {
    const out: SeasonBand[] = []
    for (const s of seasons) for (const [a0, b0] of s.spans ?? []) {
      const a = Math.max(0, x(new Date(`${a0}T00:00:00`))), b = Math.min(width, x(new Date(`${b0}T00:00:00`)))
      if (b > a) out.push({ x: a, w: b - a, color: s.color, name: s.name })
    }
    return out.sort((p, q) => p.x - q.x)
  }
  // a calendar with day boundaries (lib/seasons seasonOfDate): each season from its first day to the next's
  const dated = seasons.filter((s) => s.from).sort((a, b) => a.from!.localeCompare(b.from!))
  if (dated.length) {
    const out: SeasonBand[] = []
    for (let year = d0.getFullYear() - 1; year <= d1.getFullYear(); year++) {
      dated.forEach((s, i) => {
        const start = new Date(`${year}-${s.from}T00:00:00`)
        const end = i + 1 < dated.length ? new Date(`${year}-${dated[i + 1].from}T00:00:00`) : new Date(`${year + 1}-${dated[0].from}T00:00:00`)
        const a = Math.max(0, x(start)), b = Math.min(width, x(end))
        if (b > a) out.push({ x: a, w: b - a, color: s.color, name: s.name })
      })
    }
    return out
  }
  const byMonth = new Map<number, { name: string; color: string }>()
  for (const s of seasons) for (const m of s.months) byMonth.set(m, s)
  const bands: SeasonBand[] = []
  let cur = new Date(d0.getFullYear(), d0.getMonth(), 1)
  while (cur < d1) {
    const next = new Date(cur.getFullYear(), cur.getMonth() + 1, 1)
    const s = byMonth.get(cur.getMonth() + 1)
    if (s) {
      const a = Math.max(0, x(cur)), b = Math.min(width, x(next))
      if (b > a) bands.push({ x: a, w: b - a, color: s.color, name: s.name })
    }
    cur = next
  }
  return bands
}

/** Bands for a 1–12 month axis whose ticks sit at month centres `x(m)`, `step` px apart. */
export function monthSeasonBands(x: (m: number) => number, seasonOf: (m: number) => { name: string; color: string } | undefined, step: number, width: number): SeasonBand[] {
  const bands: SeasonBand[] = []
  for (let m = 1; m <= 12; m++) {
    const s = seasonOf(m)
    if (!s) continue
    const a = Math.max(0, x(m) - step / 2), b = Math.min(width, x(m) + step / 2)
    if (b > a) bands.push({ x: a, w: b - a, color: s.color, name: s.name })
  }
  return bands
}

/** Draws the bands: full height behind the data, or (`strip`) a 6 px strip just above the plot. */
export function SeasonBands({ bands, height, hover, strip = false }: { bands: SeasonBand[]; height: number; hover?: string | null; strip?: boolean }) {
  return <g pointerEvents="none">
    {bands.map((b, i) => strip
      ? <rect key={i} data-export-season={b.name} data-export-color={b.color} x={b.x} y={-6} width={b.w} height={6} fill={b.color} fillOpacity={SEASON_SHADE.strip} />
      : <rect key={i} data-export-season={b.name} data-export-color={b.color} x={b.x} y={0} width={b.w} height={height} fill={b.color} fillOpacity={seasonShade(b.name, hover)} />)}
  </g>
}
