/**
 * Pieces the Sources (AAE) charts share: which sites are in view, the notebooks each chart stands in
 * for, the fossil / mixed / biomass shading behind an AAE axis, and the counts matrix.
 */
import { useRef, type ReactNode } from 'react'
import * as d3 from 'd3'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { HeatCell } from '@/components/HeatCell'
import { FONT, INK } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
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

/** the site codes in view that have this field on some row, in the gallery's site order */
export function codesWith(rows: readonly FilterRow[], meta: MetaFile, field: string): string[] {
  const have = new Set(rows.filter((r) => typeof r[field] === 'number').map((r) => r.code))
  return meta.sites.map((s) => s.code).filter((c) => have.has(c))
}

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
          {w > 6.2 * s.name.length && <text x={a + w / 2} y={11} textAnchor="middle" fontSize={10} fill={INK.muted} fontFamily={FONT.family}>{s.name}</text>}</g>
        : <g key={s.cls}><rect x={0} y={a} width={depth} height={w} fill={CLASS_COLOR[s.cls]} fillOpacity={0.1} />
          {w > 14 && <text x={depth - 4} y={a + 11} textAnchor="end" fontSize={10} fill={INK.muted} fontFamily={FONT.family}>{s.name}</text>}</g>
    })}
  </g>
}

/**
 * Counts of one labelling against another (an agreement matrix): each cell the count and its share of
 * the row, shaded by that share; the diagonal-like cells a reader should look at can be outlined.
 */
export function CountMatrix({ rows, cols, count, rowLabel, colLabel, rowColor, colColor, outline, cellWidth = 92 }: {
  rows: string[]; cols: string[]; count: (r: string, c: string) => number
  rowLabel: string; colLabel: string
  rowColor?: (r: string) => string; colColor?: (c: string) => string
  outline?: (r: string, c: string) => boolean
  cellWidth?: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const left = Math.min(220, 14 + 6.6 * Math.max(...rows.map((r) => r.length), rowLabel.length * 0.8))
  const cw = Math.max(52, Math.min(cellWidth, (width - left - 64) / Math.max(1, cols.length)))
  const ch = 28, top = cols.some((c) => c.length * 6.4 > cw - 6) ? 52 : 40
  const totals = new Map(rows.map((r) => [r, cols.reduce((a, c) => a + count(r, c), 0)]))
  const w = left + cols.length * cw + 60
  const shade = d3.interpolateRgb(INK.panel, INK.accent)
  return <div ref={ref} style={{ position: 'relative', overflowX: 'auto' }}>
    <svg width={w} height={top + rows.length * ch + 10} fontFamily={FONT.family} style={{ display: 'block' }}>
      <text x={left + (cols.length * cw) / 2} y={12} textAnchor="middle" fontSize={11} fill={INK.muted}>{colLabel}</text>
      <text x={left - 8} y={top - 8} textAnchor="end" fontSize={11} fill={INK.muted}>{rowLabel}</text>
      {cols.map((c, j) => <text key={c} x={left + j * cw + cw / 2} y={top - 8} textAnchor="middle" fontSize={11} fontWeight={600} fill={colColor?.(c) ?? INK.text}>
        {/* a heading wider than its cell takes two lines */}
        {(c.length * 6.4 > cw - 6 ? twoLines(c) : [c]).map((t, k, all) => <tspan key={k} x={left + j * cw + cw / 2} dy={k ? 13 : -(all.length - 1) * 13}>{t}</tspan>)}
      </text>)}
      <text x={left + cols.length * cw + 8} y={top - 8} fontSize={11} fill={INK.muted}>n</text>
      {rows.map((r, i) => {
        const tot = totals.get(r) ?? 0
        return <g key={r}>
          <text x={left - 8} y={top + i * ch + ch / 2} dy="0.35em" textAnchor="end" fontSize={11} fill={rowColor?.(r) ?? INK.text}>{r}</text>
          {cols.map((c, j) => {
            const n = count(r, c), share = tot ? n / tot : 0
            return <HeatCell key={c} x={left + j * cw + 1} y={top + i * ch + 1} width={cw - 2} height={ch - 2}
              fill={n ? shade(0.1 + 0.75 * share) : null} label={n ? `${n} · ${Math.round(100 * share)}%` : '·'} outline={outline?.(r, c)}
              onEnter={(e) => tip.show(e, [`${r} → ${c}`, `${n} of ${tot} (${Math.round(100 * share)} % of the row)`])} onLeave={tip.hide} />
          })}
          <text x={left + cols.length * cw + 8} y={top + i * ch + ch / 2} dy="0.35em" fontSize={11} fontFamily={FONT.mono} fill={INK.muted}>{tot}</text>
        </g>
      })}
    </svg>
    {tip.node}
  </div>
}

/** split a label at the space nearest its middle */
function twoLines(text: string): string[] {
  const spaces = [...text.matchAll(/ /g)].map((m) => m.index ?? 0)
  if (!spaces.length) return [text]
  const at = spaces.reduce((a, b) => (Math.abs(b - text.length / 2) < Math.abs(a - text.length / 2) ? b : a))
  return [text.slice(0, at), text.slice(at + 1)]
}

/** a one-line reading under a chart */
export const Reading = ({ children }: { children: ReactNode }) => <p className="chart-note">{children}</p>
