/**
 * What the charts do with an excluded filter (Filters → excluded): the repo's own exclusions (the export's `excluded`
 * rows) and the reader's hand exclusions (`drop=`).
 *
 *   - hide (default): not drawn, not counted, as get_clean_data does.
 *   - include: the export's exclusions are put back and count in every fit and summary (`excl=1`, as before). Hand
 *     exclusions stay out: the reader made them, and puts one back from the list.
 *   - mark: drawn, as a muted ×, where a chart draws filters as points, but left out of every fit, R², summary,
 *     count, ratio and median (`excl=mark`). Charts of distributions or summaries draw nothing extra and say how
 *     many excluded filters they leave out.
 *
 * A filter put back one at a time (`incl=ID,…`) counts in every mode.
 */
export type ExclMode = 'hide' | 'include' | 'mark'
export const EXCL_MODES: readonly ExclMode[] = ['hide', 'include', 'mark']

/** `excl` in the URL: 1 (or include) = include, mark = mark, anything else = hide */
export function parseExclMode(v: string | null | undefined): ExclMode {
  return v === '1' || v === 'include' ? 'include' : v === 'mark' ? 'mark' : 'hide'
}
/** the mode's `excl` value; hide is the default and stays out of the link */
export function exclModeParam(m: ExclMode): string | null {
  return m === 'include' ? '1' : m === 'mark' ? 'mark' : null
}

export interface ExclState {
  mode: ExclMode
  /** the export's exclusions put back one at a time */
  inclIds: ReadonlySet<string>
  /** the reader's hand exclusions */
  dropIds: ReadonlySet<string>
  /** the reader's reason beside each hand exclusion */
  dropWhy?: Readonly<Record<string, string>>
}
type Row = { id: string; excluded?: boolean | null; exclusion_reason?: unknown }

/** is a filter in view as far as the export's exclusions go (counted in every statistic) */
export function exportIn(r: Row, s: Pick<ExclState, 'mode' | 'inclIds'>): boolean {
  return !r.excluded || s.mode === 'include' || s.inclIds.has(r.id)
}

/** why a filter is out of the statistics, in words; null when it counts */
export function exclusionReason(r: Row, s: ExclState): string | null {
  if (!exportIn(r, s)) return String(r.exclusion_reason ?? '').trim() || 'excluded in the data export'
  if (s.dropIds.has(r.id)) {
    const why = s.dropWhy?.[r.id]?.trim()
    return why ? `by hand (${why})` : 'by hand'
  }
  return null
}

/**
 * The rows every statistic is computed from, and, under mark only, the excluded ones the point charts draw as ×.
 * Every row lands in exactly one of `counted`, `marked` or neither (hidden).
 */
export function splitExcluded<T extends Row>(rows: readonly T[], s: ExclState): { counted: T[]; marked: T[] } {
  const counted: T[] = [], marked: T[] = []
  for (const r of rows) {
    if (exclusionReason(r, s) === null) counted.push(r)
    else if (s.mode === 'mark') marked.push(r)
  }
  return { counted, marked }
}

/** the hover lines of a marked filter: what it is, and why it is out */
export function markTip(r: { id: string; site?: string; date?: string }, reason: string, extra: string[] = []): string[] {
  return [r.id, [r.site, r.date].filter(Boolean).join(' · '), ...extra, `excluded: ${reason}`, 'drawn only: not in any fit, R² or count'].filter(Boolean)
}

/** the × of a marked filter, as SVG path data, centred on (x, y) with half-width `s` */
export function crossPath(x: number, y: number, s = 3.2): string {
  const f = (v: number) => +v.toFixed(2)
  return `M${f(x - s)},${f(y - s)}L${f(x + s)},${f(y + s)}M${f(x - s)},${f(y + s)}L${f(x + s)},${f(y - s)}`
}

/** where a filter stands under the current mode: counted in every statistic, drawn as × only, or not shown */
export type ExclStatus = 'counted' | 'marked' | 'hidden'
export function exclStatus(reason: string | null, mode: ExclMode): ExclStatus {
  return reason === null ? 'counted' : mode === 'mark' ? 'marked' : 'hidden'
}

/**
 * Any page's own per-filter items (explorer predictions, holdout filters, spectra, scores …) split by the gallery's
 * exclusions, joined by filter id: `counted` feed every statistic, `marked` are drawn as × under Mark, the rest are
 * hidden. An item without an id, or one the gallery does not know, counts. `reasonOf` is lib/markedRows useMarkReason.
 */
export function splitById<T>(items: readonly T[], idOf: (t: T) => string | null | undefined,
  reasonOf: (id: string) => string | null, mode: ExclMode): { counted: T[]; marked: T[]; hidden: T[] } {
  const counted: T[] = [], marked: T[] = [], hidden: T[] = []
  for (const t of items) {
    const id = idOf(t)
    const status = id ? exclStatus(reasonOf(id), mode) : 'counted'
    ;(status === 'counted' ? counted : status === 'marked' ? marked : hidden).push(t)
  }
  return { counted, marked, hidden }
}
