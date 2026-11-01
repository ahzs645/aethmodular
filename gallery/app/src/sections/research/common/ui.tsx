/**
 * Small pieces the HIPS / meteorology research pages share: reading a row's number, the page's own site and
 * season pickers (kept in the URL under the page's prefix), season colours, a plain results table, ranked bars
 * and the crossplot points every page builds from the rows.
 */
import { useMemo, type ReactNode } from 'react'
import { BarRows } from '@/components/BarRows'
import type { CrossLine, CrossPoint } from '@/components/Crossplot'
import { fmt } from '@/lib/format'
import { FONT } from '@/lib/theme'
import { shortSeason } from '@/lib/seasons'
import { ALL_SITES, SitePicker, usePageSite } from '@/lib/siteFocus'
import type { FilterRow, MetaFile } from '@/lib/types'
import { VOLUME } from '@/lib/viewFields'
import type { OlsFit as Fit } from '@/lib/stats'
import { num as cell } from '@/lib/table'
import { RESEARCH_SERIES } from '@/lib/palettes'
import { namedColor } from '@/lib/siteColours'

/** a row's field as a finite number, or null (lib/table num on r[f]) */
export const num = (r: Record<string, unknown> | undefined, f: string): number | null => cell(r?.[f])
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
/** a categorical ramp for series that are neither sites nor seasons (lib/palettes RESEARCH_SERIES) */
export const SERIES = RESEARCH_SERIES
/** a low / high / middle split: amber, blue and green from the same ramp */
export const LOW = SERIES[1], HIGH = SERIES[0], MID = SERIES[2]

/** the sites in view, Addis first when present (most of these notebooks are about Addis) */
export function sitesOf(rows: FilterRow[]): string[] {
  const s = [...new Set(rows.map((r) => r.site))]
  return s.includes(ADDIS) ? [ADDIS, ...s.filter((x) => x !== ADDIS)] : s
}

/**
 * The page's site (lib/siteFocus usePageSite: the gallery's `psite`, so the next single-site page opens
 * on the same site; the page's old `<prefix>.site` key still opens). Addis by default, or every site
 * with `allowAll`. Returns site names, as the pages filter rows by name.
 */
export function useSite(rows: FilterRow[], prefix: string, allowAll = false): [string, (s: string) => void, string[]] {
  const sites = useMemo(() => {
    const code = new Map<string, string>()
    for (const r of rows) if (!code.has(r.site)) code.set(r.site, r.code || r.site)
    return sitesOf(rows).map((name) => ({ name, code: code.get(name)! }))
  }, [rows])
  const pick = usePageSite(sites, { legacy: [`${prefix}.site`], all: allowAll && sites.length > 1 ? 'All sites' : undefined, prefer: allowAll ? ALL_SITES : undefined })
  const names = pick.options.map(pick.labelOf)
  const set = (name: string) => pick.set(name === 'All sites' ? ALL_SITES : sites.find((s) => s.name === name)?.code ?? name)
  return [pick.labelOf(pick.code), set, names]
}

/** the research pages' site control (lib/siteFocus SitePicker) over site names */
export function SitePick({ value, options, onChange }: { value: string; options: string[]; onChange: (s: string) => void }) {
  return <SitePicker value={value} options={options} onChange={onChange} />
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
    .map((name, i) => ({ name, color: namedColor(all, name, SERIES[i % SERIES.length]) }))
}

/** the season's name without its site or month range (lib/seasons), as the research pages label seasons */
export { shortSeason }

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
  // components/BarRows draws them; this keeps the research pages' short form
  return <BarRows rows={bars.map((b) => ({ key: b.key, label: b.label, value: b.value, signed: b.signed, note: b.note, tip: b.tip }))}
    max={max} format={format} axis={{ label, ticks: 4 }} />
}

/** a p value: three significant figures, or "< 0.0001" */
export const pText = (p: number) => (!Number.isFinite(p) ? '—' : p < 1e-4 ? '< 0.0001' : p.toPrecision(2))

/** "Precipitable water (AERONET)" -> "PW", "Relative humidity (station)" -> "RH": initials of a multi-word name */
export const initials = (field: string) => { const w = field.split(' (')[0].split(/\s+/); return w.length > 1 ? w.map((x) => x[0].toUpperCase()).join('') : w[0] }

/** a two-column grid of panels on a wide screen, one column on a phone */
export function Grid({ children, min = 340 }: { children: ReactNode; min?: number }) {
  return <div style={{ display: 'grid', gridTemplateColumns: `repeat(auto-fit, minmax(min(${min}px, 100%), 1fr))`, gap: 18, alignItems: 'start' }}>{children}</div>
}
