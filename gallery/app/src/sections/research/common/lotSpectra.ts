/**
 * char_12 §4's PTFE lot comparison for any site (Research → PTFE lot comparison): the per-filter spectra and
 * envelope-escape scores of gallery/data/export_research_lot_spectra.py (research/lot_spectra/), which filters a
 * chart uses, and the closed-form summaries char_12 draws from them — the per-wavenumber centre and band of each
 * lot, box-plot statistics of the escape score, the lot contrast with its Mann–Whitney test, and char_12's band
 * means. Node-tested in lotSpectra.test.mjs against char_12's saved tables. Pure: no React, no gallery imports.
 */
import { isDryName, mannWhitney } from './charcoal.ts'
import { quantile } from './stats.ts'

export type ViewKey = 'raw' | 'deriv' | 'airspec'
export const VIEW_KEYS: readonly ViewKey[] = ['raw', 'deriv', 'airspec']

export interface LotView {
  label: string; unit: string; binning: string
  /** the view's wavenumbers, one per stored point (descending) */
  wn: number[]
  file: string; points: number; bytes: number; max_abs_error: number
}
export interface LotCheckRow { subset: string; n248: number; n251: number; median248: number; median251: number; p: number | null }
export interface LotExport {
  schema_version: number
  generated: string
  generated_from: string
  notebook: string
  inputs: string[]
  /** one entry per filter, columnar; `id` is the gallery's row id (ETAD-0017), `filter_id` the SPARTAN one (ETAD-0017-1) */
  filters: {
    id: string[]; filter_id: string[]; code: string[]; type: string[]; date: string[]
    lot: (number | null)[]
    /** in char_12's 239-filter Addis cohort */
    char12: number[]
    /** char_12's envelope-escape score, recomputed by the exporter for every filter */
    escape: number[]
    instrument: (string | null)[]; qc: (string | null)[]; analyses: number[]
  }
  /** the SPARTAN pulls' own site names (<SITE>_site.csv), for sites the gallery has no metadata for */
  site_names: Record<string, string>
  views: Record<ViewKey, LotView>
  regions: { ch: [number, number]; co: [number, number] }
  bands: Record<string, [number, number]>
  /** each filter's mean area-normalised AIRSpec value in each band, ×10⁻⁴ (charcoal_spectra.band_area, unbinned) */
  band_means: Record<string, number[]>
  improve: { n: number; deriv_p05: number[]; deriv_p95: number[]; deriv_mean: number[]; escape_median: number; escape_p90: number }
  check: {
    cohort: number
    lot_effect: LotCheckRow[]
    lot_season: { group: string; n: number; median: number; q1: number; q3: number }[]
    escape: { group: string; n: number; median: number }[]
    saved: {
      lot_effect?: LotCheckRow[]
      escape_by_group?: { group: string; n: number; median: number }[]
      lot_bands?: { band: string; median248: number; median251: number; p: number }[]
    }
    saved_from: string
    airspec_vs_r: { filters: number; median_abs_diff: number; p99_abs_diff: number; median_rel_to_peak: number } | null
  }
}

/**
 * A view's binary (float32 [F] per-filter scale, then int16 [F × P]) decoded into one Float32Array of F × P values,
 * filter-major: filter i's spectrum is values[i·P … (i+1)·P).
 */
export function decodeView(buf: ArrayBuffer, n: number, p: number): Float32Array {
  if (buf.byteLength !== n * 4 + n * p * 2) throw new Error(`view size ${buf.byteLength} does not match ${n} filters × ${p} points`)
  const scale = new Float32Array(buf, 0, n)
  const q = new Int16Array(buf, n * 4, n * p)
  const out = new Float32Array(n * p)
  for (let i = 0; i < n; i++) { const s = scale[i]; for (let j = 0; j < p; j++) out[i * p + j] = q[i * p + j] * s }
  return out
}

/** char_12's calendar (config.season_for_month, dry_feb): Dry Oct–Feb, Belg Mar–May, Kiremt Jun–Sep */
export const NOTEBOOK_SEASONS = [
  { name: 'Dry (Oct-Feb)', months: [10, 11, 12, 1, 2] },
  { name: 'Belg (Mar-May)', months: [3, 4, 5] },
  { name: 'Kiremt (Jun-Sep)', months: [6, 7, 8, 9] },
] as const
export const notebookSeason = (month: number) => NOTEBOOK_SEASONS.find((s) => (s.months as readonly number[]).includes(month))?.name ?? ''

/** one exported filter a chart uses */
export interface Member {
  /** index into the export's filters (and each view's rows) */
  i: number
  id: string; code: string; date: string; month: number
  season: string
  lot: number | null
  escape: number
  instrument: string | null
  /** joined to a gallery row in view (false: an exported spectrum the gallery has no row for, or the notebook set) */
  inView: boolean
}

/**
 * Which filters the charts use. `view`: exported spectra whose gallery row is in view (the reader's sites, seasons,
 * dates and exclusions; the row's season). `plus`: those and, for the sites in view, the PM2.5 spectra the gallery
 * has no row for at all (`known` = every gallery id), seasoned by the reader's calendar. `notebook`: char_12's 239
 * Addis filters with its own calendar, which reproduces its tables.
 */
export type MemberSet = 'view' | 'plus' | 'notebook'
export const MEMBER_SETS: readonly MemberSet[] = ['view', 'plus', 'notebook']

export function membersOf(exp: LotExport, set: MemberSet, rows: readonly { id: string; code: string; season: string }[],
  known: ReadonlySet<string>, seasonOf: (code: string, date: string, month: number) => string | null): { members: Member[]; unmatched: number } {
  const f = exp.filters
  const byId = new Map(rows.map((r) => [r.id, r]))
  const codes = new Set(rows.map((r) => r.code))
  const members: Member[] = []
  let unmatched = 0
  for (let i = 0; i < f.id.length; i++) {
    const month = Number(f.date[i].slice(5, 7))
    const base = { i, id: f.id[i], code: f.code[i], date: f.date[i], month, lot: f.lot[i], escape: f.escape[i], instrument: f.instrument[i] }
    if (set === 'notebook') {
      if (f.char12[i]) members.push({ ...base, season: notebookSeason(month), inView: false })
      continue
    }
    const r = byId.get(f.id[i])
    if (r) { members.push({ ...base, season: r.season, inView: true }); continue }
    if (!codes.has(f.code[i]) || known.has(f.id[i])) continue
    unmatched++
    if (set === 'plus' && f.type[i] === 'PM2.5') {
      const season = seasonOf(f.code[i], f.date[i], month)
      if (season) members.push({ ...base, season, inView: false })
    }
  }
  return { members, unmatched }
}

/** the season a chart holds fixed by default: the dry one where the calendar names one, else the first */
export const defaultSeason = (seasons: readonly string[]) => seasons.find(isDryName) ?? seasons[0] ?? ''

/**
 * The season a site's panel holds fixed: the reader's pick where the site has filters in it, else (a per-site
 * calendar names its seasons differently) the site's dry-named season, else its season with the most filters.
 */
export function seasonAt(members: readonly Pick<Member, 'season'>[], chosen: string): string {
  const n = new Map<string, number>()
  for (const m of members) n.set(m.season, (n.get(m.season) ?? 0) + 1)
  if (n.has(chosen) || !n.size) return chosen
  return [...n.keys()].find(isDryName) ?? [...n.entries()].sort((a, b) => b[1] - a[1])[0][0]
}

/**
 * Per-wavenumber centre and band of a set of spectra (columns [from, to) of a decoded view): char_12 draws the
 * mean and the 5–95 % (04) or 10–90 % (04b) band; quantiles are numpy's linear default.
 */
export function profile(values: Float32Array, points: number, idx: readonly number[], from: number, to: number,
  centre: 'mean' | 'median', band: [number, number] | null): { mid: Float64Array; lo: Float64Array; hi: Float64Array } {
  const k = to - from, mid = new Float64Array(k), lo = new Float64Array(k), hi = new Float64Array(k)
  const col = new Float64Array(idx.length)
  for (let j = 0; j < k; j++) {
    let sum = 0
    for (let a = 0; a < idx.length; a++) { const v = values[idx[a] * points + from + j]; col[a] = v; sum += v }
    if (!idx.length) { mid[j] = lo[j] = hi[j] = NaN; continue }
    const sorted = col.slice().sort()
    const q = (p: number) => { const h = (sorted.length - 1) * p, l = Math.floor(h); return sorted[l] + (sorted[Math.min(l + 1, sorted.length - 1)] - sorted[l]) * (h - l) }
    mid[j] = centre === 'mean' ? sum / idx.length : q(0.5)
    lo[j] = band ? q(band[0]) : NaN
    hi[j] = band ? q(band[1]) : NaN
  }
  return { mid, lo, hi }
}

/** the columns of a descending wavenumber grid inside [lo, hi] (one contiguous run) */
export function columnsIn(wn: readonly number[], lo: number, hi: number): [number, number] {
  let from = -1, to = -1
  for (let j = 0; j < wn.length; j++) if (wn[j] >= lo && wn[j] <= hi) { if (from < 0) from = j; to = j + 1 }
  return from < 0 ? [0, 0] : [from, to]
}

/** matplotlib's box plot of a list: median, quartiles, whiskers at the furthest data within 1.5 × IQR, the fliers */
export function boxStats(v: readonly number[]) {
  const q1 = quantile(v, 0.25), q3 = quantile(v, 0.75), iqr = q3 - q1
  const inside = v.filter((x) => x >= q1 - 1.5 * iqr && x <= q3 + 1.5 * iqr)
  return { n: v.length, median: quantile(v, 0.5), q1, q3, wlo: Math.min(...inside), whi: Math.max(...inside), fliers: v.filter((x) => x < q1 - 1.5 * iqr || x > q3 + 1.5 * iqr) }
}

/** char_12 §4's lot contrast over a set of filters: n and median escape per lot, Mann–Whitney between two lots */
export function lotContrast(members: readonly Pick<Member, 'lot' | 'escape'>[], a: number, b: number) {
  const ea = members.filter((m) => m.lot === a).map((m) => m.escape), eb = members.filter((m) => m.lot === b).map((m) => m.escape)
  return { na: ea.length, nb: eb.length, medianA: quantile(ea, 0.5), medianB: quantile(eb, 0.5), test: mannWhitney(ea, eb) }
}

/** the lots in a set of members, most filters first (unknown lots left out) */
export function lotsOf(members: readonly Pick<Member, 'lot'>[]): number[] {
  const n = new Map<number, number>()
  for (const m of members) if (m.lot !== null) n.set(m.lot, (n.get(m.lot) ?? 0) + 1)
  return [...n.keys()].sort((x, y) => n.get(y)! - n.get(x)! || x - y)
}

/** charcoal_spectra.band_area: the mean of a spectrum's values inside a window (×1e4 in char_12's table) */
export function bandMean(values: Float32Array, points: number, i: number, wn: readonly number[], window: [number, number]): number {
  let s = 0, n = 0
  for (let j = 0; j < points; j++) if (wn[j] >= window[0] && wn[j] <= window[1]) { s += values[i * points + j]; n++ }
  return n ? s / n : NaN
}

/** char_12 §3: the share of the pertinent wavenumbers where a 2nd-derivative spectrum leaves the IMPROVE 5–95 % envelope */
export function escapeOf(values: Float32Array, points: number, i: number, p05: readonly number[], p95: readonly number[]): number {
  let out = 0
  for (let j = 0; j < points; j++) { const v = values[i * points + j]; if (v < p05[j] || v > p95[j]) out++ }
  return out / points
}
