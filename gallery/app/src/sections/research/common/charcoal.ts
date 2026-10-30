/**
 * The char notebooks' per-filter charcoal scores (research/ftir_etad_char char_02/03/05/06, exported by
 * gallery/data/export_research_charcoal.py) joined to the gallery's Addis rows, and the closed-form summaries the
 * notebooks draw from them: medians by group, the share of filters raw feedstock matches better, the monthly
 * anomaly rate, the lot controls and their Mann–Whitney tests (node-tested in charcoal.test.mjs against the
 * notebooks' own tables). Pure: no React, no gallery imports.
 */
import { median, quantile } from './stats.ts'

/** one row of the export's per_filter table (char_06 per_filter_classification.csv + char_03 + char_02) */
export interface CharcoalScore {
  FilterId: string; date: string; season: string; lot: number | null
  r_char: number; r_raw: number; delta: number; component: string
  Fabs: number | null; EC: number | null
  library_r?: number | null; implied_T?: number | null; p_burned?: number | null
}

/** the gallery row fields the join reads (a FilterRow has them all) */
export interface JoinRow { id: string; date: string; month: number | null; season: string; excluded?: boolean; [field: string]: unknown }

/**
 * Which filters a chart scores. `view`: the gallery's Addis rows in view that have a score (the reader's seasons,
 * dates and exclusions). `plus`: those and the scored filters the gallery has no row for (sampled after its Addis
 * series ends), seasoned by the reader's calendar. `notebook`: all scored filters with the notebook's own seasons,
 * Fabs and EC, which is what the notebooks' tables are computed on.
 */
export type ScoreSet = 'view' | 'plus' | 'notebook'
export const SCORE_SETS: readonly ScoreSet[] = ['view', 'plus', 'notebook']

export interface Scored<R extends JoinRow = JoinRow> {
  /** the gallery id (ETAD-0017), the notebooks' FilterId (ETAD-0017-1) */
  id: string; filterId: string
  date: string; month: number
  season: string
  lot: number | null
  rChar: number; rRaw: number; delta: number
  /** in char_06's low-similarity mixture component */
  low: boolean
  /** HIPS Fabs (Mm⁻¹) and FTIR EC (µg/m³): the gallery row's where there is one, else the notebook's */
  fabs: number | null; ec: number | null
  libraryR: number | null; impliedT: number | null; pBurned: number | null
  /** the gallery row it joined to (null for a filter the gallery does not have, or in `notebook` mode) */
  row: R | null
}

/** the gallery's row id for a SPARTAN ExternalFilterId: ETAD-0017-1 → ETAD-0017 (the scan suffix dropped) */
export const galleryId = (filterId: string) => filterId.trim().replace(/^([A-Za-z]+-\d+)-\d+$/, '$1')

const finite = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

/**
 * The scores for a set of filters. `seasonOf` names a filter's season under the reader's calendar from its date
 * (used for `plus` filters, which have no gallery row); `fabs`/`ec` are the gallery fields to read.
 */
export function joinScores<R extends JoinRow>(scores: readonly CharcoalScore[], rows: readonly R[], set: ScoreSet,
  seasonOf: (date: string, month: number) => string | null, fields = { fabs: 'HIPS Fabs', ec: 'EC (FTIR)' },
  known: (id: string) => boolean = () => false): { scored: Scored<R>[]; outside: number } {
  const byId = new Map(rows.map((r) => [r.id, r]))
  // every score's gallery id: ids the gallery has at all are "in the gallery", whether or not they are in view
  const scored: Scored<R>[] = []
  let outside = 0
  for (const s of scores) {
    const id = galleryId(s.FilterId), month = Number(s.date.slice(5, 7))
    const base = { id, filterId: s.FilterId, date: s.date, month, lot: finite(s.lot), rChar: s.r_char, rRaw: s.r_raw, delta: s.delta, low: s.component === 'low-similarity',
      libraryR: finite(s.library_r), impliedT: finite(s.implied_T), pBurned: finite(s.p_burned) }
    if (set === 'notebook') { scored.push({ ...base, season: s.season, fabs: finite(s.Fabs), ec: finite(s.EC), row: null }); continue }
    const r = byId.get(id)
    if (r) { scored.push({ ...base, date: r.date || s.date, season: r.season, fabs: finite(r[fields.fabs]), ec: finite(r[fields.ec]), row: r }); continue }
    outside++
    // `plus` adds the filters the gallery has no row for at all (sampled after its Addis series ends); one it has but
    // leaves out of view (excluded, or cut by date, season or site) stays out, as on every other chart
    if (set === 'plus' && !known(id)) {
      const season = seasonOf(s.date, month)
      if (season) scored.push({ ...base, season, fabs: finite(s.Fabs), ec: finite(s.EC), row: null })
    }
  }
  return { scored, outside }
}

/**
 * Filters → excluded: Mark: the scores of the gallery's excluded rows in view (`rows`), joined as `view` joins the
 * rows it counts, less any filter the set already counts (`counted`: `plus` takes a filter it finds no row in view
 * for as a newer one). These are drawn as × and counted in notes only, never in a median, share or test.
 */
export function markedScores<R extends JoinRow>(scores: readonly CharcoalScore[], rows: readonly R[], counted: readonly Pick<Scored, 'id'>[],
  fields?: { fabs: string; ec: string }): Scored<R>[] {
  if (!rows.length) return []
  const have = new Set(counted.map((s) => s.id))
  return joinScores(scores, rows, 'view', () => null, fields).scored.filter((s) => !have.has(s.id))
}

/**
 * Is a season the dry one? The notebooks' calendar rule is Oct–Feb ("Dry Season"); the gallery's calendars call it
 * "Dry (Oct-Feb)" or "Bega (Oct-Jan, dry)". A calendar with no dry season (quarters) falls back to the months.
 */
export const isDryName = (season: string) => /\bdry\b|bega/i.test(season)
export function dryOf(scored: readonly Pick<Scored, 'season' | 'month'>[]): (s: Pick<Scored, 'season' | 'month'>) => boolean {
  return scored.some((s) => isDryName(s.season)) ? (s) => isDryName(s.season) : (s) => [10, 11, 12, 1, 2].includes(s.month)
}

/** the char_05 / char_07 headline over a set of filters */
export interface Headline { n: number; rawMedian: number; charMedian: number; rawWins: number; deltaMedian: number; libraryMedian: number }
export function headline(scored: readonly Pick<Scored, 'rRaw' | 'rChar' | 'delta' | 'libraryR'>[]): Headline {
  const lib = scored.map((s) => s.libraryR).filter((v): v is number => v !== null)
  return {
    n: scored.length,
    rawMedian: median(scored.map((s) => s.rRaw)),
    charMedian: median(scored.map((s) => s.rChar)),
    rawWins: scored.length ? scored.filter((s) => s.delta > 0).length / scored.length : NaN,
    deltaMedian: median(scored.map((s) => s.delta)),
    libraryMedian: lib.length ? median(lib) : NaN,
  }
}

/** a headline per group, in the order the groups are given */
export function byGroup<T extends Pick<Scored, 'rRaw' | 'rChar' | 'delta' | 'libraryR'>>(scored: readonly T[], groupOf: (s: T) => string, order: readonly string[]): (Headline & { group: string })[] {
  return order.map((group) => ({ group, ...headline(scored.filter((s) => groupOf(s) === group)) })).filter((g) => g.n > 0)
}

/** char_06 §2: the share of filters with delta > 0.05 in each calendar month, all years pooled */
export function monthlyRate(scored: readonly Pick<Scored, 'month' | 'delta'>[], cut = 0.05): { month: number; n: number; rate: number }[] {
  return Array.from({ length: 12 }, (_, i) => {
    const m = scored.filter((s) => s.month === i + 1)
    return { month: i + 1, n: m.length, rate: m.length ? m.filter((s) => s.delta > cut).length / m.length : NaN }
  })
}

/**
 * Mann–Whitney U of a against b, two-sided, as scipy.stats.mannwhitneyu computes it for samples this size: U of the
 * first sample, the normal approximation with the tie correction and a continuity correction of 0.5.
 */
export function mannWhitney(a: readonly number[], b: readonly number[]): { u: number; p: number } | null {
  const n1 = a.length, n2 = b.length
  if (n1 < 1 || n2 < 1) return null
  const all = [...a.map((v) => ({ v, g: 0 })), ...b.map((v) => ({ v, g: 1 }))].sort((x, y) => x.v - y.v)
  const rank = new Array<number>(all.length)
  let tie = 0
  for (let i = 0; i < all.length;) {
    let j = i
    while (j + 1 < all.length && all[j + 1].v === all[i].v) j++
    const t = j - i + 1
    for (let k = i; k <= j; k++) rank[k] = (i + j) / 2 + 1
    tie += t ** 3 - t
    i = j + 1
  }
  const r1 = all.reduce((s, x, i) => s + (x.g === 0 ? rank[i] : 0), 0)
  const u = r1 - (n1 * (n1 + 1)) / 2
  const n = n1 + n2, mu = (n1 * n2) / 2
  const sd = Math.sqrt((n1 * n2 / 12) * ((n + 1) - tie / (n * (n - 1))))
  if (!(sd > 0)) return { u, p: 1 }
  const z = (Math.abs(u - mu) - 0.5) / sd
  return { u, p: Math.min(1, 2 * normalSf(Math.max(0, z))) }
}

/** upper tail of the standard normal, accurate in the far tail (erfc by its continued fraction) */
function normalSf(z: number): number {
  const x = z / Math.SQRT2
  if (x < 3) {
    // erfc via the series for erf
    let sum = x, term = x
    for (let k = 1; k < 200; k++) { term *= (-x * x) / k; const add = term / (2 * k + 1); sum += add; if (Math.abs(add) < 1e-17) break }
    return 0.5 * (1 - (2 / Math.sqrt(Math.PI)) * sum)
  }
  // Lentz continued fraction for erfc(x)
  let f = x, c = x, d = 0
  for (let k = 1; k < 300; k++) {
    const a = k / 2
    d = x + a * d; d = 1 / d
    c = x + a / c
    const delta = c * d
    f *= delta
    if (Math.abs(delta - 1) < 1e-15) break
  }
  return 0.5 * (Math.exp(-x * x) / Math.sqrt(Math.PI)) / f
}

/** median and quartiles of a list, for a strip row */
export const spread = (v: readonly number[]) => ({ n: v.length, mid: v.length ? median(v) : NaN, lo: v.length ? quantile(v, 0.25) : NaN, hi: v.length ? quantile(v, 0.75) : NaN })
