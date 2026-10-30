import { createContext, useContext, useMemo, type ReactNode } from 'react'
import { ChartAside } from '@/lib/singleView'
import { usePresentation } from '@/lib/presentation'
import { exclStatus, markTip, splitById, type ExclMode, type ExclStatus } from '@/lib/exclusionMode'
import type { FilterRow } from '@/lib/types'
import type { SweepMark } from '@/components/SweepPlot'

/**
 * Filters → excluded: Mark (lib/exclusionMode). The charts keep receiving the counted rows for every statistic; the
 * excluded filters in view (the same sites, seasons, dates and PMF pick) come from here, for the point charts to
 * draw as × (components/ScatterLayer `marks`) and for the summary charts to count in a note (ExcludedNote).
 */
interface Marked {
  mode: ExclMode
  /** the excluded filters in view; empty unless the mode is mark */
  rows: FilterRow[]
  /** why a filter is out, for its hover; null when it counts */
  reasonOf: (id: string) => string | null
  /** does the gallery have a row for this filter at all (in view or not); pages joining their own items ask it */
  known?: (id: string) => boolean
}
const Ctx = createContext<Marked>({ mode: 'hide', rows: [], reasonOf: () => null })
/**
 * The context as a chart sees it: a meeting page shown As presented (lib/presentation) is untouched by the
 * gallery's exclusion choices, so there every chart reads Hide: no marks, no notes, whatever Filters → excluded says.
 */
function useMarked(): Marked {
  const ctx = useContext(Ctx)
  const presented = usePresentation() === 'presented'
  return useMemo(() => (presented && ctx.mode !== 'hide' ? { ...ctx, mode: 'hide' as const, rows: [] } : ctx), [ctx, presented])
}

export function MarkedRowsProvider({ mode, rows, reasonOf, known, children }: Marked & { children: ReactNode }) {
  const value = useMemo(() => ({ mode, rows: mode === 'mark' ? rows : [], reasonOf, known }), [mode, rows, reasonOf, known])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export const useExclMode = () => useMarked().mode
/** the excluded filters a point chart draws as × (none unless Filters → excluded is Mark); never feed these to a fit */
export const useMarkedRows = () => useMarked().rows
/** why a filter is excluded (null: it counts); a crossplot fed outside the rows uses it for hand exclusions */
export const useMarkReason = () => useMarked().reasonOf

const UNKNOWN = () => false
/** does the gallery have a row for a filter id at all (in view or not): a page tells its newer filters from cut ones */
export const useKnownFilter = (): ((id: string) => boolean) => useMarked().known ?? UNKNOWN

/** where any filter id stands under the current mode (lib/exclusionMode exclStatus), for pages with their own rows */
export function useExclStatus(): (id: string | null | undefined) => ExclStatus {
  const { mode, reasonOf } = useMarked()
  return useMemo(() => (id) => (id ? exclStatus(reasonOf(id), mode) : 'counted'), [mode, reasonOf])
}

/**
 * The one way a page with its own per-filter items (not the gallery rows: explorer predictions, holdout filters,
 * spectra, scores) follows Filters → excluded: `counted` feed every fit, median and count; `marked` are drawn as ×
 * (components/Crossplot `marked`, ScatterLayer `marks`) under Mark; the hidden ones are left out. Memoised on `items`.
 */
export function useExclusionSplit<T>(items: readonly T[], idOf: (t: T) => string | null | undefined): { counted: T[]; marked: T[]; nOut: number } {
  const { mode, reasonOf } = useMarked()
  return useMemo(() => {
    const s = splitById(items, idOf, reasonOf, mode)
    return { counted: s.counted, marked: s.marked, nOut: s.marked.length + s.hidden.length }
  }, [items, mode, reasonOf]) // eslint-disable-line react-hooks/exhaustive-deps
}

/** one excluded filter drawn as a × at plot pixels `at` (components/ScatterLayer); `tip` is its hover */
export interface MarkPoint { id: string; at: readonly [number, number]; tip: string[] }

/**
 * The marks of the excluded filters a chart can place: `at` gives a row's plot position, or null where the chart
 * would not draw it (a missing value, another panel). `extra` adds the chart's own values to the hover.
 */
export function toMarks(rows: readonly FilterRow[], at: (r: FilterRow) => readonly [number, number] | null,
  reasonOf: (id: string) => string | null, extra?: (r: FilterRow) => string[]): MarkPoint[] {
  const out: MarkPoint[] = []
  for (const r of rows) {
    const p = at(r)
    if (!p || !Number.isFinite(p[0]) || !Number.isFinite(p[1])) continue
    out.push({ id: r.id, at: p, tip: markTip(r, reasonOf(r.id) ?? 'excluded', extra?.(r)) })
  }
  return out
}

/** the same for a components/SweepPlot (`marks`): `at` gives a row's x (a date, number or category) and y */
export function toSweepMarks(rows: readonly FilterRow[], at: (r: FilterRow) => readonly [SweepMark['x'], number] | null,
  reasonOf: (id: string) => string | null, extra?: (r: FilterRow) => string[]): SweepMark[] {
  const out: SweepMark[] = []
  for (const r of rows) {
    const p = at(r)
    if (!p || !Number.isFinite(p[1])) continue
    out.push({ x: p[0], y: p[1], tip: markTip(r, reasonOf(r.id) ?? 'excluded', extra?.(r)) })
  }
  return out
}

/**
 * A page's marks built the way it builds its points: `build` takes the excluded filters in view (useMarkedRows) and
 * returns what the page would draw for them (e.g. CrossPoint[] for components/Crossplot `marked`). Memoised on `deps`.
 */
export function useMarkedPoints<T>(build: (rows: FilterRow[]) => T[], deps: unknown[]): T[] {
  const rows = useMarkedRows()
  return useMemo(() => (rows.length ? build(rows) : []), [rows, ...deps]) // eslint-disable-line react-hooks/exhaustive-deps
}

/**
 * Under Mark, the note on a chart that counts filters: how many excluded filters with its measurement(s) are in
 * view and left out. `drawn`: the chart draws them as × (the point charts); else it only says they are not counted.
 */
export function ExcludedNote({ fields = [], drawn = false, of, count }: {
  fields?: string[]; drawn?: boolean; of?: (r: FilterRow) => boolean
  /** a page with its own items (useExclusionSplit): how many it marks, instead of counting the gallery rows */
  count?: number
}) {
  const rows = useMarkedRows()
  const mode = useExclMode()
  const n = useMemo(() => (count !== undefined ? (mode === 'mark' ? count : 0)
    : rows.filter((r) => fields.every((f) => typeof r[f] === 'number' && Number.isFinite(r[f] as number)) && (!of || of(r))).length),
    [count, mode, rows, fields.join('|'), of]) // eslint-disable-line react-hooks/exhaustive-deps
  if (!n) return null
  const what = `${n} excluded filter${n === 1 ? '' : 's'}`
  return <ChartAside><p className="chart-note excluded-note">{drawn
    ? `${what} drawn as × where they fall inside the axes (Filters → excluded: Mark); not counted in any fit, R² or summary.`
    : `${what} not counted (Filters → excluded: Mark draws them only on the point charts).`}</p></ChartAside>
}

/**
 * The one note for a chart that draws a fixed filter set (a notebook's locked holdout, an explorer evaluation set, a
 * saved table) on which Filters → excluded does not act: shown under Include or Mark, so the reader knows the
 * switch is not ignored by accident. `what` names the set.
 */
export function FixedSetNote({ what, hand = false, subject = 'this chart' }: {
  what: string
  /** a filter excluded by hand still leaves this chart's points (and any live fit): only the export's exclusions do not act */
  hand?: boolean
  /** what the note is about: 'this chart' (default) or e.g. 'this page' in a page intro */
  subject?: string
}) {
  const mode = useExclMode()
  if (mode === 'hide') return null
  const name = mode === 'mark' ? 'Mark' : 'Include'
  return <ChartAside><p className="chart-note excluded-note">{hand
    ? <>Filters → excluded ({name}) acts here only through filters you exclude by hand: {subject} draws {what}, as the notebook fixed it.</>
    : <>Filters → excluded ({name}) does not act here: {subject} draws {what}, as the notebook fixed it.</>}</p></ChartAside>
}
