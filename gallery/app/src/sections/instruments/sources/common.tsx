/**
 * Pieces the Sources (AAE) charts share: which sites are in view, the notebooks each chart stands in
 * for, the fossil / mixed / biomass shading behind an AAE axis, and the counts matrix.
 */
import type { ReactNode } from 'react'
import { useColourScheme } from '@/lib/colourSchemes'
import * as d3 from 'd3'
import { HeatMatrix } from '@/components/HeatMatrix'
import { FONT, INK } from '@/lib/theme'
import { fitsWidth } from '@/lib/fitLabel'
import { pct } from '@/lib/format'
import { CLASS_COLOR, type AaeRegions } from './aae'

const NB = 'research/ftir_hips_chem/'
export const NOTEBOOK = {
  addis01: `${NB}addis_01_source_apportionment.ipynb`,
  addis01daily: `${NB}addis_01_source_apportionment_daily.ipynb`,
  dominantComparison: `${NB}dominant_source_comparison.ipynb`,
  dominantThreshold: `${NB}dominant_source_threshold_analysis.ipynb`,
  thresholdBySource: `${NB}threshold_by_source_analysis.ipynb`,
  regression: `${NB}source_apportionment_regression.ipynb`,
  addisSource: `${NB}addis_ababa_source_analysis.ipynb`,
  optics: `${NB}scripts/optics.py`,
}
/** a provenance line: the notebooks, then what the chart reads */
export const provenance = (notebooks: string[], reads: string) => <>{notebooks.map((n, i) => <span key={n}>{i ? ', ' : ''}<code>{n}</code></span>)}. {reads}</>

/**
 * Fossil / mixed / biomass bands behind an AAE axis, each named. `scale` maps an AAE to pixels; with
 * `orient` 'x' the bands are columns across a plot `length` wide and `depth` tall, with 'y' rows.
 */
export function RegionBands({ scale, length, depth, regions, orient = 'x' }: {
  scale: (v: number) => number; length: number; depth: number; regions: AaeRegions; orient?: 'x' | 'y'
}) {
  const clamp = (p: number) => Math.max(0, Math.min(length, p))
  const bands = [
    { cls: 'Fossil fuel' as const, a: null, b: regions.fossil_max, name: `fossil ≤ ${regions.fossil_max}` },
    { cls: 'Mixed' as const, a: regions.fossil_max, b: regions.biomass_min, name: 'mixed' },
    { cls: 'Biomass' as const, a: regions.biomass_min, b: null, name: `biomass ≥ ${regions.biomass_min}` },
  ]
  const lo = orient === 'x' ? 0 : length, hi = orient === 'x' ? length : 0   // where an open end runs to
  return <g pointerEvents="none">
    {bands.map((s) => {
      const p0 = clamp(s.a == null ? lo : scale(s.a)), p1 = clamp(s.b == null ? hi : scale(s.b))
      const a = Math.min(p0, p1), w = Math.abs(p1 - p0)
      if (w < 1) return null
      return orient === 'x'
        ? <g key={s.cls}><rect x={a} y={0} width={w} height={depth} fill={CLASS_COLOR[s.cls]} fillOpacity={0.1} />
          {fitsWidth(s.name, w, 10) && <text x={a + w / 2} y={11} textAnchor="middle" fontSize={10} fill={INK.muted} fontFamily={FONT.family}>{s.name}</text>}</g>
        : <g key={s.cls}><rect x={0} y={a} width={depth} height={w} fill={CLASS_COLOR[s.cls]} fillOpacity={0.1} />
          {w > 14 && <text x={depth - 4} y={a + 11} textAnchor="end" fontSize={10} fill={INK.muted} fontFamily={FONT.family}>{s.name}</text>}</g>
    })}
  </g>
}

/**
 * Counts of one labelling against another (an agreement matrix): each cell the count and its share of
 * the row, shaded by that share; the diagonal-like cells a reader should look at can be outlined.
 */
export function CountMatrix({ id, rows, cols, count, rowLabel, colLabel, rowColor, colColor, outline, cellWidth = 92 }: {
  /** the chart's id: the shading's palette is kept in the link under it (lib/colourSchemes) */
  id: string
  rows: string[]; cols: string[]; count: (r: string, c: string) => number
  rowLabel: string; colLabel: string
  rowColor?: (r: string) => string; colColor?: (c: string) => string
  outline?: (r: string, c: string) => boolean
  cellWidth?: number
}) {
  const totals = new Map(rows.map((r) => [r, cols.reduce((a, c) => a + count(r, c), 0)]))
  const colours = useColourScheme(id, 'sequential')
  const ramp = d3.interpolateRgbBasis(colours.stops)
  // cells use 10–85 % of the palette, so an empty-looking share still shows and a full row's label stays readable
  const key = d3.scaleSequential((v: number) => ramp(0.1 + 0.75 * v)).domain([0, 1])
  return <HeatMatrix rows={rows.map((r) => ({ key: r, color: rowColor?.(r) }))} columns={cols.map((c) => ({ key: c, color: colColor?.(c) }))}
    cellWidth={[52, cellWidth]} cellHeight={28} gap={2} colLabels="wrap" colLabelWeight={600} rowHeader={rowLabel} colHeader={colLabel} cellLabels="always"
    extraColumns={[{ key: 'n', header: 'n', side: 'right', text: (i) => String(totals.get(rows[i]) ?? 0) }]}
    cell={(i, j) => {
      const r = rows[i], c = cols[j], tot = totals.get(r) ?? 0, n = count(r, c), share = tot ? n / tot : 0
      return { fill: n ? ramp(0.1 + 0.75 * share) : null, label: n ? `${n} · ${pct(share)}` : '·', outline: outline?.(r, c),
        tip: [`${r} → ${c}`, `${n} of ${tot} (${pct(share)} of the row)`] }
    }}
    legend={{ scale: key, scheme: colours.scheme, width: 140, label: 'share of the row', format: (v) => pct(v) }} />
}

/** a one-line reading under a chart */
export const Reading = ({ children }: { children: ReactNode }) => <p className="chart-note">{children}</p>
