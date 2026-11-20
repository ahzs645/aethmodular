import type { ScatterScales } from '@/components/ScatterPlot'
import { useExport } from '@/lib/data'
import { fmt } from '@/lib/format'
import { RESEARCH_SERIES } from '@/lib/palettes'

/**
 * Every IMPROVE filter on the HIPS bend, without grouping (gallery/data/export_hips_improve_density.py,
 * `hips_linearise/improve_density.json`): two count grids and the saturating curve fitted to single filters.
 * Ann asked for it on 19 Nov 2026 so that the curve does not rest on the 17 group medians alone.
 */

export const IMPROVE_DENSITY_COMMAND = 'uv run python gallery/data/export_hips_improve_density.py'
export const IMPROVE_DENSITY_FILE = 'hips_linearise/improve_density.json'

/** a sparse count grid: [column, row, count] per non-empty cell */
export interface DensityGrid {
  x: { lower: number; upper: number; step: number }
  y: { lower: number; upper: number; step: number }
  cells: [number, number, number][]
  inside: number
  outside: number
  max: number
}
export interface SingleFilterFit {
  population: string; from_ug: number; residual: 'absolute' | 'log'; samples: number; sites: number
  a_per_ug: number; tau_inf: number; tau_inf_lo: number; tau_inf_hi: number
}
export interface ImproveDensity {
  generated: string; source: string; samples: number; sites: number; first: string; last: string
  median_volume_m3: number; volume_p10_p90_m3: [number, number]; deposit_area_cm2: number; urban_sites: string[]
  /** TOR EC on the filter (µg) against HIPS optical depth τ */
  loading: DensityGrid
  /** TOR EC (µg/m³) against HIPS Fabs (Mm⁻¹) */
  concentration: DensityGrid
  fits: SingleFilterFit[]
}

export const useImproveDensity = (on = true) => useExport<ImproveDensity>(on ? IMPROVE_DENSITY_FILE : null)

/** the cells' colour, and the colour of a curve fitted to single filters (the group-median curve keeps MEASURED) */
export const DENSITY_COLOR = RESEARCH_SERIES[4]
export const SINGLE_FIT_COLOR = RESEARCH_SERIES[3]
export const PRIMARY_POPULATION = 'all sites'

/** a single-filter fit's key in a link: `<from µg>/<residual>` */
export const fitKey = (f: Pick<SingleFilterFit, 'from_ug' | 'residual'>) => `${f.from_ug}/${f.residual}`
export const fitLabel = (f: Pick<SingleFilterFit, 'from_ug' | 'residual'>) =>
  `${f.from_ug > 0 ? `from ${fmt(f.from_ug, 0)} µg` : 'every filter'}, ${f.residual === 'log' ? 'log τ' : 'τ'} residuals`

/** the grid's cells as rectangles in plot pixels, shaded by log count; drawn under the points and curves */
export function DensityCells({ grid, x, y, color = DENSITY_COLOR }: {
  grid: DensityGrid; x: ScatterScales['x']; y: ScatterScales['y']; color?: string
}) {
  const top = Math.log1p(grid.max)
  const w = Math.abs(x(grid.x.lower + grid.x.step) - x(grid.x.lower)), h = Math.abs(y(grid.y.lower + grid.y.step) - y(grid.y.lower))
  return <g pointerEvents="none">
    {grid.cells.map(([i, j, n]) => {
      const x0 = grid.x.lower + i * grid.x.step, y1 = grid.y.lower + (j + 1) * grid.y.step
      return <rect key={`${i}/${j}`} x={x(x0)} y={y(y1)} width={w + 0.25} height={h + 0.25} fill={color}
        fillOpacity={0.1 + 0.8 * Math.log1p(n) / top} />
    })}
  </g>
}
