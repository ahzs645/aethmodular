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
export function timeSeasonBands(x: ScaleTime<number, number>, seasons: readonly Pick<SeasonMeta, 'name' | 'months' | 'color'>[], width: number): SeasonBand[] {
  const [d0, d1] = x.domain() as [Date, Date]
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
