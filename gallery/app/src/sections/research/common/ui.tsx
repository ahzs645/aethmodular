/**
 * Small pieces the HIPS / meteorology research pages share: reading a row's number, the page's own site and
 * season pickers (kept in the URL under the page's prefix), season colours, a plain results table, ranked bars
 * and the crossplot points every page builds from the rows.
 */
import { useMemo, useRef, type ReactNode } from 'react'
import * as d3 from 'd3'
import { Segmented, Select } from '@/components/ChartFrame'
import type { CrossLine, CrossPoint } from '@/components/Crossplot'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/format'
import { FONT, INK } from '@/lib/theme'
import { seasonOnly } from '@/lib/seasons'
import { useSearchParam } from '@/lib/url'
import type { FilterRow, MetaFile } from '@/lib/types'
import { VOLUME } from '@/lib/viewFields'
import type { Fit } from './stats'

export const num = (r: Record<string, unknown> | undefined, f: string): number | null => {
  const v = r?.[f]
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}
export const has = (r: Record<string, unknown>, ...fs: string[]) => fs.every((f) => num(r, f) !== null)

/**
 * The rows in µg/m³ whatever the reader's basis. Under mass per filter (Filters → units, lib/viewFields) every
 * µg/m³ field arrives × the filter's sampled volume but HIPS Fabs (Mm⁻¹) does not, so a page that sets Fabs or
 * Fabs/MAC against EC, or works out loadings itself, would mix the two. Those pages are notebook reproductions
 * in µg/m³; they take their rows through this and say so (BASIS_NOTE).
 */
export function useConcRows(rows: FilterRow[], meta: MetaFile): FilterRow[] {
  return useMemo(() => {
    if (meta.basis !== 'mass') return rows
    const perFilter = Object.entries(meta.field_units).filter(([, u]) => u === 'µg/filter').map(([f]) => f)
    return rows.map((r) => {
      const v = num(r, VOLUME)
      const out: Record<string, unknown> = { ...r }
      for (const f of perFilter) if (f in r) out[f] = v && v > 0 && num(r, f) !== null ? (r[f] as number) / v : null
      return out as FilterRow
    })
  }, [rows, meta])
}
export const BASIS_NOTE = 'This page works in µg/m³ (the notebooks\' units), so it ignores mass per filter under Filters → units.'

export const ADDIS = 'Addis Ababa'
export const LOW = '#d08c2c', HIGH = '#2c7fb8', MID = '#4f9d69'
/** a categorical ramp for series that are neither sites nor seasons */
export const SERIES = ['#2c7fb8', '#d08c2c', '#4f9d69', '#b5446e', '#6a51a3', '#8c564b', '#e377c2', '#7f7f7f']

/** the sites in view, Addis first when present (most of these notebooks are about Addis) */
export function sitesOf(rows: FilterRow[]): string[] {
  const s = [...new Set(rows.map((r) => r.site))]
  return s.includes(ADDIS) ? [ADDIS, ...s.filter((x) => x !== ADDIS)] : s
}

/** the page's site, in the URL as `<prefix>.site`; Addis by default */
export function useSite(rows: FilterRow[], prefix: string, allowAll = false): [string, (s: string) => void, string[]] {
  const sites = useMemo(() => sitesOf(rows), [rows])
  const options = allowAll && sites.length > 1 ? ['All sites', ...sites] : sites
  const [site, setSite] = useSearchParam<string>(`${prefix}.site`, '')
  const chosen = options.includes(site) ? site : options[0] ?? ''
  return [chosen, setSite, options]
}

export function SitePick({ value, options, onChange }: { value: string; options: string[]; onChange: (s: string) => void }) {
  if (options.length <= 1) return null
  return options.length <= 5
    ? <Segmented label="Site" value={value} options={options} onChange={onChange} optionLabel={(o) => o} />
    : <Select label="Site" value={value} options={options} onChange={onChange} />
}

export const ofSite = (rows: FilterRow[], site: string) => (site === 'All sites' ? rows : rows.filter((r) => r.site === site))

/** the seasons of these rows in calendar order (by the first month each covers), with colours from the meta */
export function seasonsOf(rows: FilterRow[], meta: MetaFile): { name: string; color: string }[] {
  const firstMonth = new Map<string, number>()
  for (const r of rows) if (r.season && r.month) {
    // start the year in October, as the Addis calendars do, so Dry / Belg / Kiremt read in order
    const m = ((r.month + 2) % 12)
    firstMonth.set(r.season, Math.min(firstMonth.get(r.season) ?? 99, m))
  }
  const all = [...meta.seasons, ...Object.values(meta.season_conventions ?? {}).flat()]
  return [...firstMonth.keys()].sort((a, b) => firstMonth.get(a)! - firstMonth.get(b)!)
    .map((name, i) => ({ name, color: all.find((s) => s.name === name)?.color ?? SERIES[i % SERIES.length] }))
}

export const shortSeason = (s: string) => seasonOnly(s).replace(/\s*\(.*\)\s*$/, '')

/** a crossplot point for a filter row, with the usual tip and record */
export function point(r: FilterRow, x: number, y: number, color: string, xLabel: string, yLabel: string, source: string, extra: string[] = [], hollow = false): CrossPoint {
  return { x, y, color, hollow, id: r.id, tip: [r.id, `${r.site} · ${r.date}${r.season ? ` · ${shortSeason(r.season)}` : ''}`, `${xLabel} ${fmt(x, 2)} · ${yLabel} ${fmt(y, 2)}`, ...extra],
    rec: { id: r.id, source, site: r.site, date: r.date, fields: [[xLabel, x], [yLabel, y]] } }
}

/** a saved OLS line for Crossplot's extraLines */
export const line = (f: Fit | { slope: number; intercept: number } | null, color: string, label: string, dash?: string): CrossLine[] =>
  f ? [{ slope: f.slope, intercept: f.intercept, color, label, dash, method: 'ols', stats: null }] : []

export interface Column<T> { key: string; label: ReactNode; value: (row: T) => ReactNode; num?: boolean; title?: string }

/** a results table in the gallery's table style; scrolls sideways on a phone instead of the page */
export function DataTable<T>({ rows, columns, rowKey, highlight }: { rows: T[]; columns: Column<T>[]; rowKey: (r: T, i: number) => string; highlight?: (r: T) => boolean }) {
  return (
    <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
      <table className="census-table" style={{ minWidth: Math.min(880, columns.length * 88) }}>
        <thead><tr>{columns.map((c) => <th key={c.key} title={c.title} style={{ textTransform: 'none', letterSpacing: 0, textAlign: c.num ? 'right' : 'left' }}>{c.label}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={rowKey(r, i)} style={highlight?.(r) ? { background: 'var(--accent-soft)' } : undefined}>
          {columns.map((c) => <td key={c.key} style={c.num ? { fontFamily: FONT.mono, textAlign: 'right', whiteSpace: 'nowrap' } : undefined}>{c.value(r)}</td>)}
        </tr>)}</tbody>
      </table>
    </div>
  )
}

export interface Bar { key: string; label: string; value: number | null; signed?: number | null; note?: string; tip?: string[] }

/**
 * Ranked horizontal bars (a screen's score or R²): length = value, colour = the sign of `signed` when given
 * (blue positive, red negative, as the gallery draws r), else the accent.
 */
export function Bars({ bars, label, max, format = (v) => fmt(v, 2) }: { bars: Bar[]; label: string; max?: number; format?: (v: number) => string }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const labelW = Math.min(240, Math.max(120, width * 0.34)), noteW = 70
  // labels end at the bars: cut what would run off the left edge (about 6 px a character at 11.5 px)
  const maxChars = Math.max(10, Math.floor((labelW - 8) / 6.2))
  const w = Math.max(120, width - labelW - noteW - 8)
  const shown = bars.filter((b) => b.value !== null)
  const x = d3.scaleLinear().domain([0, max ?? (d3.max(shown, (b) => b.value!) || 1)]).range([0, w]).nice()
  const rowH = 20
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg width={labelW + w + noteW} height={shown.length * rowH + 30} fontFamily={FONT.family}>
        {shown.map((b, i) => {
          const c = b.signed == null ? INK.accent : b.signed >= 0 ? INK.positive : INK.negative
          return <g key={b.key} transform={`translate(0,${i * rowH})`} onMouseMove={(e) => b.tip && tip.show(e, b.tip)} onMouseLeave={tip.hide}>
            <text x={labelW - 6} y={rowH / 2} dy="0.35em" textAnchor="end" fontSize={11.5} fill={INK.text}>{b.label.length > maxChars ? `${b.label.slice(0, maxChars - 1)}…` : b.label}</text>
            <rect x={labelW} y={3} width={Math.max(1, x(b.value!))} height={rowH - 6} fill={c} fillOpacity={0.75} />
            <text x={labelW + x(b.value!) + 4} y={rowH / 2} dy="0.35em" fontSize={10.5} fontFamily={FONT.mono} fill={INK.muted}>{format(b.value!)}{b.note ? `  ${b.note}` : ''}</text>
          </g>
        })}
        <g transform={`translate(${labelW},${shown.length * rowH + 4})`}>
          <line x1={0} x2={w} stroke={INK.axis} />
          {x.ticks(4).map((t) => <text key={t} x={x(t)} y={14} textAnchor="middle" fontSize={10} fill={INK.muted}>{format(t)}</text>)}
          <text x={w / 2} y={26} textAnchor="middle" fontSize={11} fill={INK.muted}>{label}</text>
        </g>
      </svg>
      {tip.node}
    </div>
  )
}

/** "slope 0.403 · intercept 2.830 · R² 0.785 · n 378" */
export const fitText = (f: Fit | null, digits = 3) => f ? `slope ${fmt(f.slope, digits)} · intercept ${fmt(f.intercept, digits)} · R² ${fmt(f.r2, 3)} · n ${f.n}` : '—'

/** a p value: three significant figures, or "< 0.0001" */
export const pText = (p: number) => (!Number.isFinite(p) ? '—' : p < 1e-4 ? '< 0.0001' : p.toPrecision(2))

/** "Precipitable water (AERONET)" -> "PW", "Relative humidity (station)" -> "RH": initials of a multi-word name */
export const initials = (field: string) => { const w = field.split(' (')[0].split(/\s+/); return w.length > 1 ? w.map((x) => x[0].toUpperCase()).join('') : w[0] }

/** a two-column grid of panels on a wide screen, one column on a phone */
export function Grid({ children, min = 340 }: { children: ReactNode; min?: number }) {
  return <div style={{ display: 'grid', gridTemplateColumns: `repeat(auto-fit, minmax(min(${min}px, 100%), 1fr))`, gap: 18, alignItems: 'start' }}>{children}</div>
}
